from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket
from pydantic import BaseModel, Field

from orizzonte_desk.config import Settings
from orizzonte_desk.controller import AgentController, ControlError
from orizzonte_desk.engine import TradingEngine
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
            if state.status is AgentStatus.RUNNING:
                try:
                    gateway = self.controller.gateway(state.environment)
                    gateway.schedule_dead_man(30)
                    now = datetime.now(UTC)
                    if (
                        self.last_engine_tick is None
                        or (now - self.last_engine_tick).total_seconds() >= 60
                    ):
                        await asyncio.to_thread(self.engine.tick)
                        self.last_engine_tick = now
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


def create_app(paths: AppPaths | None = None, settings: Settings | None = None) -> FastAPI:
    app_paths = paths or AppPaths.discover()
    app_settings = settings or Settings.load(app_paths.config)
    store = StateStore(app_paths.database)
    store.initialize()
    controller = AgentController(app_paths, app_settings, store)
    recovered = store.agent_state()
    if recovered.environment is not Environment.PAPER and recovered.status is AgentStatus.RUNNING:
        store.save_agent_state(
            recovered.model_copy(
                update={
                    "status": AgentStatus.PAUSED,
                    "metadata": recovered.metadata | {"restart_reconcile_required": True},
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
        version="0.1.0",
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )

    def operational_state() -> dict[str, Any]:
        agent = store.agent_state()
        metadata = dict(agent.metadata)
        metadata["positions"] = [item["payload"] for item in store.positions()]
        metadata["orders"] = store.orders(open_only=True)
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


app = create_app()
