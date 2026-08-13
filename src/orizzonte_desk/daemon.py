from __future__ import annotations

import asyncio
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
from orizzonte_desk.storage import StateStore
from orizzonte_desk.stream import HyperliquidStream


class ArmRequest(BaseModel):
    environment: Environment
    budget_usdc: float = Field(gt=0)
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

    async def heartbeat(self) -> None:
        while True:
            await asyncio.sleep(10)
            state = self.store.agent_state()
            if state.status is AgentStatus.RUNNING:
                updated = state.model_copy(update={"last_heartbeat": datetime.now(UTC)})
                self.store.save_agent_state(updated)
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
                except Exception as exc:
                    self.store.event(
                        "connectivity",
                        "Falha no heartbeat/dead man's switch",
                        level="ERROR",
                        payload={"error": str(exc)},
                    )


def create_app(paths: AppPaths | None = None, settings: Settings | None = None) -> FastAPI:
    app_paths = paths or AppPaths.discover()
    app_settings = settings or Settings.load(app_paths.config)
    store = StateStore(app_paths.database)
    store.initialize()
    controller = AgentController(app_paths, app_settings, store)
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

    @application.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "timestamp": datetime.now(UTC), "agent": store.agent_state()}

    @application.get("/state")
    def state() -> Any:
        return store.agent_state()

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
