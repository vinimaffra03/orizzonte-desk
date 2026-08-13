from __future__ import annotations

from fastapi.testclient import TestClient

from orizzonte_desk.controller import AgentController
from orizzonte_desk.daemon import create_app
from orizzonte_desk.engine import TradingEngine
from orizzonte_desk.models import AgentStatus
from orizzonte_desk.storage import StateStore
from orizzonte_desk.stream import HyperliquidStream


def test_daemon_health_is_local_state(app_paths, settings, monkeypatch) -> None:
    async def idle_stream(_: HyperliquidStream) -> None:
        import asyncio

        await asyncio.Event().wait()

    monkeypatch.setattr(HyperliquidStream, "run", idle_stream)
    with TestClient(create_app(app_paths, settings)) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["agent"]["status"] == "disarmed"


def test_engine_does_nothing_while_disarmed(app_paths, settings) -> None:
    store = StateStore(app_paths.database)
    store.initialize()
    controller = AgentController(app_paths, settings, store)
    engine = TradingEngine(app_paths, settings, store, controller)
    assert engine.tick() == {"acted": False, "reason": f"state:{AgentStatus.DISARMED}"}


def test_stream_records_user_events(app_paths) -> None:
    store = StateStore(app_paths.database)
    store.initialize()
    stream = HyperliquidStream(store)
    stream._handle({"channel": "userFills", "data": {"fills": []}})
    assert store.get("market_stream")["channel"] == "userFills"
    assert store.recent_events(1)[0]["category"] == "exchange_event"
