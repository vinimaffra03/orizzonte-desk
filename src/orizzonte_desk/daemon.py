from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket
from pydantic import BaseModel, Field

from orizzonte_desk import __version__
from orizzonte_desk import runtime_primitives as primitives
from orizzonte_desk.config import Settings
from orizzonte_desk.controller import AgentController, ControlError
from orizzonte_desk.engine import TradingEngine
from orizzonte_desk.exchange import AccountSnapshot
from orizzonte_desk.models import AgentStatus, Environment
from orizzonte_desk.paths import AppPaths
from orizzonte_desk.release import ReleaseError, ReleaseManager
from orizzonte_desk.storage import StateStore
from orizzonte_desk.stream import HyperliquidStream


class ArmRequest(BaseModel):
    environment: Environment
    budget_usdc: float = Field(gt=0)
    confirmation: str


class ReleaseApprovalRequest(BaseModel):
    release_id: str
    confirmation: str


class ReleaseVerifyRequest(BaseModel):
    release_id: str


class TestnetSmokeRequest(BaseModel):
    budget_usdc: float = Field(ge=20)
    confirmation: str


class SecretMutationRequest(BaseModel):
    account_address: str
    secret_key: str | None = None


class MainnetAuthorizationRequest(BaseModel):
    budget_usdc: float = Field(gt=0, le=500)
    confirmation: str


class MainnetRevokeRequest(BaseModel):
    authorization_id: str


class Runtime:
    def __init__(
        self, controller: AgentController, store: StateStore, engine: TradingEngine
    ) -> None:
        self.controller = controller
        self.store = store
        self.engine = engine
        self.heartbeat_task: asyncio.Task[None] | None = None
        self.stream_task: asyncio.Task[None] | None = None
        self.last_engine_tick: datetime | None = None
        self.consecutive_failures = 0

    async def heartbeat(self) -> None:
        while True:
            await asyncio.sleep(10)
            state = self.store.agent_state()
            if state.status in {AgentStatus.RUNNING, AgentStatus.PAUSED, AgentStatus.LOCKED}:
                try:
                    gateway = self.controller.gateway(state.environment)
                    now = datetime.now(UTC)
                    snapshot = gateway.reconcile()
                    account = state.account_address or "paper"
                    pending_entry = any(
                        item["status"] in {"intent", "submitting", "partially_filled"}
                        for item in self.store.orders(
                            open_only=True,
                            environment=state.environment,
                            account_address=account,
                        )
                    )
                    dead_man = primitives.dead_man_action(
                        status=state.status.value,
                        has_positions=bool(snapshot.positions),
                        pending_entry=pending_entry,
                        positions_protected=_positions_are_protected(snapshot),
                    )
                    if dead_man == "schedule":
                        gateway.schedule_dead_man(30)
                    elif dead_man == "clear":
                        gateway.clear_dead_man()
                    if (
                        self.last_engine_tick is None
                        or (now - self.last_engine_tick).total_seconds() >= 60
                        or (snapshot.positions and not _positions_are_protected(snapshot))
                    ):
                        await asyncio.to_thread(self.engine.tick)
                        self.last_engine_tick = now
                        after_tick = gateway.reconcile()
                        if after_tick.positions and _positions_are_protected(after_tick):
                            gateway.clear_dead_man()
                    current = self.store.agent_state()
                    if current.status is AgentStatus.RUNNING:
                        self.store.save_agent_state(
                            current.model_copy(update={"last_heartbeat": datetime.now(UTC)})
                        )
                    self.consecutive_failures = 0
                except Exception as exc:
                    self.consecutive_failures += 1
                    self.store.event(
                        "connectivity",
                        "Falha no heartbeat/dead man's switch",
                        level="ERROR",
                        payload={"error": str(exc)},
                    )
                    if self.consecutive_failures >= 3:
                        current = self.store.agent_state()
                        if current.status is AgentStatus.RUNNING:
                            self.store.latch_lock(
                                "connectivity",
                                reason="Três falhas consecutivas de heartbeat",
                                payload={"error": str(exc)},
                            )
                            self.store.save_agent_state(
                                current.model_copy(
                                    update={
                                        "status": AgentStatus.PAUSED,
                                        "metadata": current.metadata
                                        | {"restart_reconcile_required": True},
                                    }
                                )
                            )


def _protection_management_status(
    status: AgentStatus,
    positions: list[dict[str, Any]],
    orders: list[dict[str, Any]],
) -> dict[str, Any]:
    position_symbols = {
        str(position.get("coin") or position.get("symbol") or "")
        for position in positions
        if position.get("coin") or position.get("symbol")
    }
    protection_kinds: dict[str, set[str]] = {symbol: set() for symbol in position_symbols}
    protection_order_count = 0
    for order in orders:
        payload = order.get("payload")
        if not isinstance(payload, dict):
            payload = {}
        symbol = str(order.get("symbol") or payload.get("coin") or payload.get("symbol") or "")
        reduce_only = bool(payload.get("reduceOnly", payload.get("reduce_only", False)))
        kind = _protection_kind(payload)
        locally_owned = str(payload.get("kind", "")).lower() in {"sl", "tp"}
        if symbol in protection_kinds and kind and (reduce_only or locally_owned):
            protection_kinds[symbol].add(kind)
            protection_order_count += 1
    protected = sorted(
        symbol for symbol, kinds in protection_kinds.items() if {"sl", "tp"} <= kinds
    )
    unprotected = sorted(position_symbols - set(protected))
    manage_only = status in {AgentStatus.PAUSED, AgentStatus.LOCKED}
    return {
        "active": bool(positions)
        and status
        in {
            AgentStatus.RUNNING,
            AgentStatus.PAUSED,
            AgentStatus.LOCKED,
        },
        "mode": "manage_only"
        if manage_only
        else "running"
        if status is AgentStatus.RUNNING
        else "inactive",
        "position_count": len(positions),
        "protection_order_count": protection_order_count,
        "protected_symbols": protected,
        "unprotected_symbols": unprotected,
    }


def create_app(paths: AppPaths | None = None, settings: Settings | None = None) -> FastAPI:
    app_paths = paths or AppPaths.discover()
    app_settings = settings or Settings.load(app_paths.config)
    store = StateStore(app_paths.database)
    store.initialize()
    controller = AgentController(app_paths, app_settings, store)
    recovered = store.agent_state()
    requires_recovery = (
        recovered.environment is not Environment.PAPER and recovered.status is AgentStatus.RUNNING
    ) or (
        recovered.environment is Environment.MAINNET
        and recovered.status is not AgentStatus.DISARMED
    )
    if requires_recovery:
        recovered_metadata = recovered.metadata | {"restart_reconcile_required": True}
        if recovered.environment is Environment.MAINNET:
            recovered_metadata["requires_mainnet_reauthorization"] = True
        store.save_agent_state(
            recovered.model_copy(
                update={
                    "status": AgentStatus.PAUSED,
                    "metadata": recovered_metadata,
                }
            )
        )
        store.event(
            "recovery",
            "Restart detectado; sessão pausada até preflight e reconciliação explícitos",
            level="WARNING",
        )
    engine = TradingEngine(app_paths, app_settings, store, controller)
    runtime = Runtime(controller, store, engine)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        runtime.heartbeat_task = asyncio.create_task(runtime.heartbeat())
        runtime.stream_task = asyncio.create_task(HyperliquidStream(store).run())
        yield
        tasks: list[asyncio.Task[None]] = []
        if runtime.heartbeat_task:
            runtime.heartbeat_task.cancel()
            tasks.append(runtime.heartbeat_task)
        if runtime.stream_task:
            runtime.stream_task.cancel()
            tasks.append(runtime.stream_task)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    application = FastAPI(
        title="Orizzonte Desk",
        version=__version__,
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )

    def operational_state() -> dict[str, Any]:
        agent = store.agent_state()
        account = agent.account_address or "paper"
        metadata = dict(agent.metadata)
        position_payloads = [
            item["payload"]
            for item in store.positions(
                environment=agent.environment,
                account_address=account,
            )
        ]
        open_orders = store.orders(
            open_only=True,
            environment=agent.environment,
            account_address=account,
        )
        metadata["positions"] = position_payloads
        metadata["orders"] = open_orders
        metadata["testnet_certificate"] = controller.testnet_certificate_status()
        metadata["mainnet_authorization"] = controller.mainnet_authorization_status()
        metadata["protection_management"] = _protection_management_status(
            agent.status,
            position_payloads,
            open_orders,
        )
        stream_state = store.get("market_stream")
        reconciliation = store.get("exchange_reconciliation")
        metadata["stream_status"] = (
            "connected"
            if isinstance(stream_state, dict) and stream_state.get("connected")
            else "disconnected"
        )
        metadata["reconciliation"] = reconciliation or {"status": "pending"}
        try:
            pointer = json.loads(
                ReleaseManager(app_paths).approved_pointer.read_text(encoding="utf-8")
            )
            metadata["approved_release_id"] = pointer.get("release_id")
        except (OSError, ValueError, AttributeError):
            metadata["approved_release_id"] = None
        return agent.model_dump(mode="json") | {"metadata": metadata}

    @application.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "timestamp": datetime.now(UTC),
            "agent": operational_state(),
            "stream": store.get("market_stream"),
            "reconciliation": store.get("exchange_reconciliation"),
            "locks": {
                name: store.lock(name)
                for name in ("daily_loss", "drawdown", "unprotected_position", "connectivity")
                if store.lock(name)
            },
        }

    @application.get("/state")
    def state() -> Any:
        return operational_state()

    @application.get("/events")
    def events(limit: int = 100) -> list[dict[str, Any]]:
        return store.recent_events(min(max(limit, 1), 500))

    def control(action: str, payload: ArmRequest | None = None) -> Any:
        try:
            if action == "arm" and payload:
                return controller.arm(
                    environment=payload.environment,
                    budget_usdc=payload.budget_usdc,
                    confirmation=payload.confirmation,
                )
            return getattr(controller, action)()
        except ControlError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @application.post("/control/arm")
    def arm(request: ArmRequest) -> Any:
        return control("arm", request)

    for name in ("start", "pause", "flatten", "disarm"):
        application.add_api_route(
            f"/control/{name}",
            lambda action=name: control(action),
            methods=["POST"],
            name=name,
        )

    @application.post("/release/build")
    def release_build() -> Any:
        try:
            return ReleaseManager(app_paths).build()
        except ReleaseError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @application.post("/release/verify")
    def release_verify(request: ReleaseVerifyRequest) -> Any:
        try:
            return ReleaseManager(app_paths).verify(request.release_id)
        except ReleaseError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @application.post("/release/approve")
    def release_approve(request: ReleaseApprovalRequest) -> Any:
        try:
            return ReleaseManager(app_paths).approve(request.release_id, request.confirmation)
        except ReleaseError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @application.post("/testnet/preflight")
    def testnet_preflight() -> Any:
        try:
            return controller.testnet_preflight()
        except ControlError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @application.post("/testnet/reconcile")
    def testnet_reconcile() -> Any:
        try:
            return controller.testnet_reconcile()
        except ControlError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @application.post("/testnet/smoke")
    def testnet_smoke(request: TestnetSmokeRequest) -> Any:
        try:
            return controller.testnet_smoke(
                budget_usdc=request.budget_usdc,
                confirmation=request.confirmation,
            )
        except ControlError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @application.post("/internal/secrets/{environment}/generate")
    def secret_generate(environment: Environment, request: SecretMutationRequest) -> Any:
        try:
            return controller.secret_generate(
                environment,
                secret_key=request.secret_key,
                account_address=request.account_address,
            )
        except ControlError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @application.post("/internal/secrets/{environment}/verify")
    def secret_verify(environment: Environment) -> Any:
        try:
            return controller.secret_verify(environment)
        except ControlError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @application.post("/internal/secrets/{environment}/rotate")
    def secret_rotate(environment: Environment, request: SecretMutationRequest) -> Any:
        try:
            return controller.secret_rotate(
                environment,
                secret_key=request.secret_key,
                account_address=request.account_address,
            )
        except ControlError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @application.get("/internal/secrets/{environment}/status")
    def secret_status(environment: Environment) -> Any:
        try:
            return controller.secret_status(environment)
        except ControlError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @application.get("/internal/testnet/certificate/status")
    def testnet_certificate_status() -> Any:
        return controller.testnet_certificate_status()

    @application.post("/internal/mainnet/authorize")
    def mainnet_authorize(request: MainnetAuthorizationRequest) -> Any:
        try:
            return controller.issue_mainnet_authorization(
                budget_usdc=request.budget_usdc,
                confirmation=request.confirmation,
            )
        except ControlError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @application.get("/internal/mainnet/authorization/status")
    def mainnet_authorization_status() -> Any:
        return controller.mainnet_authorization_status()

    @application.post("/internal/mainnet/authorization/revoke")
    def mainnet_authorization_revoke(request: MainnetRevokeRequest) -> Any:
        return controller.revoke_mainnet_authorization(request.authorization_id)

    @application.websocket("/stream")
    async def stream(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            while True:
                await websocket.send_json(
                    {
                        "state": store.agent_state().model_dump(mode="json"),
                        "events": store.recent_events(20),
                    }
                )
                await asyncio.sleep(1)
        finally:
            await websocket.close()

    return application


def _protection_kind(order: dict[str, Any]) -> str | None:
    return primitives.protection_kind(order)


def _positions_are_protected(snapshot: AccountSnapshot) -> bool:
    return bool(snapshot.positions) and all(
        primitives.has_native_protection_pair(position, snapshot.open_orders)
        for position in snapshot.positions
    )


app = create_app()
