from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from orizzonte_desk.controller import AgentController, ControlError
from orizzonte_desk.daemon import create_app
from orizzonte_desk.engine import TradingEngine
from orizzonte_desk.exchange import AccountSnapshot, AssetMetadata, PaperGateway
from orizzonte_desk.models import AgentState, AgentStatus, Environment, Side, Signal
from orizzonte_desk.release import ReleaseManager
from orizzonte_desk.storage import StateStore
from orizzonte_desk.stream import HyperliquidStream


def _signal(symbol: str = "ETH") -> Signal:
    return Signal(
        timestamp=datetime.now(UTC),
        symbol=symbol,
        side=Side.LONG,
        score=0.9,
        probability=0.9,
        entry_reference=2_000 if symbol == "ETH" else 100,
        stop_distance=50 if symbol == "ETH" else 2,
        atr=25 if symbol == "ETH" else 1,
        regime="bull",
    )


def test_paper_ledger_survives_restart_and_deduplicates_intent(app_paths) -> None:
    store = StateStore(app_paths.database)
    store.initialize()
    first = PaperGateway(10_000, store=store)
    response = first.place_entry_with_protection(
        _signal(),
        size=1,
        stop_price=1_950,
        take_profit_price=2_100,
        slippage=0,
        idempotency_key="2026-01-01:ETH:long",
    )
    replay = first.place_entry_with_protection(
        _signal(),
        size=1,
        stop_price=1_950,
        take_profit_price=2_100,
        slippage=0,
        idempotency_key="2026-01-01:ETH:long",
    )
    assert replay["idempotent_replay"] is True
    assert replay["entry_cloid"] == response["entry_cloid"]
    assert len(first.snapshot().positions) == 1

    restarted = PaperGateway(1, store=store)
    assert len(restarted.reconcile().positions) == 1
    restarted.update_market("ETH", 2_020, event_id="ETH:bar-1")
    cash_after_mark = restarted.equity
    restarted.update_market("ETH", 2_020, event_id="ETH:bar-1")
    assert restarted.equity == cash_after_mark
    restarted.flatten_all(slippage=0)
    assert PaperGateway(1, store=store).snapshot().positions == ()
    assert len(store.fills()) == 2
    assert restarted.equity > 10_000


def test_stream_fill_and_order_replays_are_idempotent(app_paths) -> None:
    store = StateStore(app_paths.database)
    store.initialize()
    stream = HyperliquidStream(store)
    fill = {"coin": "BTC", "px": "60000", "sz": "0.01", "tid": 99, "fee": "0.1"}
    message = {"channel": "userFills", "data": {"isSnapshot": True, "fills": [fill]}}
    stream._handle(message)
    stream._handle(message)
    assert len(store.fills()) == 1
    assert store.get("exchange_stream_snapshot")["fills"] == 1

    update = {
        "channel": "orderUpdates",
        "data": [{"order": {"coin": "BTC", "cloid": "0xabc", "oid": 7}, "status": "open"}],
    }
    stream._handle(update)
    stream._handle(update)
    assert len(store.orders()) == 1


class _TimeoutAfterAcceptanceGateway:
    def __init__(self) -> None:
        self.positions: tuple[dict[str, Any], ...] = ()
        self.flattened = False

    def asset_metadata(self, symbol: str) -> AssetMetadata:
        return AssetMetadata(symbol, 2, 0.01, 4)

    def reconcile(self) -> AccountSnapshot:
        return self.snapshot()

    def snapshot(self) -> AccountSnapshot:
        return AccountSnapshot(10_000, 10_000, (), self.positions, {"SOL": 100})

    def place_entry_with_protection(self, *_: Any, **__: Any) -> dict[str, Any]:
        self.positions = (
            {
                "coin": "SOL",
                "szi": "1",
                "entryPx": "100",
                "positionValue": "100",
                "leverage": {"type": "isolated", "value": 10},
            },
        )
        raise TimeoutError("accepted but response lost")

    def flatten_all(self, *, slippage: float = 0.01) -> list[dict[str, Any]]:
        self.flattened = True
        self.positions = ()
        return []

    def replace_protection(self, *_: Any, **__: Any) -> list[dict[str, Any]]:
        return []

    def reduce_position(self, *_: Any, **__: Any) -> dict[str, Any]:
        return {}


class _FilledGateway(_TimeoutAfterAcceptanceGateway):
    def __init__(self) -> None:
        super().__init__()
        self.orders: list[dict[str, Any]] = []
        self.notionals: list[float] = []

    def snapshot(self) -> AccountSnapshot:
        return AccountSnapshot(10_000, 10_000, tuple(self.orders), self.positions, {})

    def place_entry_with_protection(self, signal: Signal, **kwargs: Any) -> dict[str, Any]:
        size = float(kwargs["size"])
        self.notionals.append(size * signal.entry_reference)
        position = {
            "coin": signal.symbol,
            "szi": str(size * signal.side.sign),
            "entryPx": str(signal.entry_reference),
            "positionValue": str(size * signal.entry_reference),
            "leverage": {"type": "isolated", "value": 10},
        }
        self.positions += (position,)
        self.orders.extend(
            {
                "coin": signal.symbol,
                "reduceOnly": True,
                "size": size,
                "kind": kind,
                "side": "A" if signal.side is Side.LONG else "B",
            }
            for kind in ("sl", "tp")
        )
        return {
            "status": "ok",
            "entry_cloid": str(kwargs.get("idempotency_key")),
            "filled_size": size,
            "average_price": signal.entry_reference,
        }


def _market() -> pd.DataFrame:
    end = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    rows: list[dict[str, Any]] = []
    for symbol, price in (("BTC", 60_000), ("ETH", 2_000), ("SOL", 100), ("XRP", 1)):
        for offset in range(60):
            timestamp = end - timedelta(hours=59 - offset)
            rows.append(
                {
                    "timestamp": timestamp,
                    "symbol": symbol,
                    "open": price,
                    "high": price * 1.01,
                    "low": price * 0.99,
                    "close": price,
                    "volume": 100,
                    "funding_rate": 0,
                }
            )
    return pd.DataFrame(rows)


class _InfoResponse:
    def __init__(self, payload: object) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> object:
        return self.payload


class _InfoClient:
    def __init__(self, timestamp_ms: int) -> None:
        self.timestamp_ms = timestamp_ms
        self.calls: list[str] = []

    def post(self, _url: str, *, json: dict[str, Any]) -> _InfoResponse:
        request_type = str(json["type"])
        self.calls.append(request_type)
        if request_type == "candleSnapshot":
            return _InfoResponse(
                [
                    {
                        "t": self.timestamp_ms,
                        "o": "100",
                        "h": "101",
                        "l": "99",
                        "c": "100",
                        "v": "10",
                    }
                ]
            )
        return _InfoResponse(
            [
                {
                    "coin": str(json["coin"]),
                    "fundingRate": "0.000125",
                    "premium": "0",
                    "time": self.timestamp_ms,
                }
            ]
        )


def test_market_window_uses_real_funding_and_hourly_cache(app_paths, settings) -> None:
    store = StateStore(app_paths.database)
    store.initialize()
    controller = AgentController(app_paths, settings, store)
    engine = TradingEngine(app_paths, settings, store, controller)
    end = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    client = _InfoClient(int(end.timestamp() * 1000))
    engine.client = client  # type: ignore[assignment]

    first = engine._market_window(Environment.PAPER)
    second = engine._market_window(Environment.PAPER)

    assert set(first["funding_rate"]) == {0.000125}
    assert first.equals(second)
    assert client.calls.count("fundingHistory") == 4
    assert client.calls.count("candleSnapshot") == 4


def test_timeout_after_acceptance_flattens_and_locks(app_paths, settings, monkeypatch) -> None:
    store = StateStore(app_paths.database)
    store.initialize()
    store.save_agent_state(
        AgentState(status=AgentStatus.RUNNING, environment=Environment.PAPER, budget_usdc=1_000)
    )
    controller = AgentController(app_paths, settings, store)
    gateway = _TimeoutAfterAcceptanceGateway()
    monkeypatch.setattr(controller, "gateway", lambda _: gateway)
    engine = TradingEngine(app_paths, settings, store, controller)
    monkeypatch.setattr(engine, "_market_window", lambda _: _market())
    monkeypatch.setattr(engine.generator, "latest", lambda *_args, **_kwargs: [_signal("SOL")])

    outcome = engine.tick()
    assert outcome["action"] == "execution_failure_lock"
    assert gateway.flattened is True
    assert gateway.positions == ()
    assert store.agent_state().status is AgentStatus.LOCKED
    assert store.lock("unprotected_position")["locked"] is True
    assert any(item["status"] == "intent" for item in store.orders())


def test_engine_updates_notional_cap_between_two_entries(app_paths, settings, monkeypatch) -> None:
    store = StateStore(app_paths.database)
    store.initialize()
    store.save_agent_state(
        AgentState(status=AgentStatus.RUNNING, environment=Environment.PAPER, budget_usdc=1_000)
    )
    controller = AgentController(app_paths, settings, store)
    gateway = _FilledGateway()
    monkeypatch.setattr(controller, "gateway", lambda _: gateway)
    engine = TradingEngine(app_paths, settings, store, controller)
    monkeypatch.setattr(engine, "_market_window", lambda _: _market())
    signals = [
        _signal("SOL").model_copy(update={"stop_distance": 0.143}),
        _signal("ETH").model_copy(update={"stop_distance": 2.857}),
    ]
    monkeypatch.setattr(engine.generator, "latest", lambda *_args, **_kwargs: signals)
    engine.tick()
    assert len(gateway.notionals) == 2
    assert sum(gateway.notionals) <= settings.risk.max_notional_multiple * 1_000 + 0.01


def test_restart_pauses_nonpaper_running_session(app_paths, settings, monkeypatch) -> None:
    store = StateStore(app_paths.database)
    store.initialize()
    store.save_agent_state(
        AgentState(
            status=AgentStatus.RUNNING,
            environment=Environment.TESTNET,
            budget_usdc=100,
        )
    )

    async def idle_stream(_: HyperliquidStream) -> None:
        await asyncio.Event().wait()

    monkeypatch.setattr(HyperliquidStream, "run", idle_stream)
    with TestClient(create_app(app_paths, settings)) as client:
        state = client.get("/state").json()
        assert state["status"] == "paused"
        assert state["metadata"]["restart_reconcile_required"] is True
        paths = {route.path for route in client.app.routes}
        assert {
            "/release/build",
            "/release/verify",
            "/release/approve",
            "/testnet/preflight",
            "/testnet/smoke",
            "/testnet/reconcile",
        } <= paths


def test_mainnet_is_fail_closed_even_with_exact_confirmation(app_paths, settings) -> None:
    store = StateStore(app_paths.database)
    store.initialize()
    controller = AgentController(app_paths, settings, store)
    with pytest.raises(ControlError, match="Mainnet"):
        controller.arm(
            environment=Environment.MAINNET,
            budget_usdc=100,
            confirmation=controller.expected_confirmation(Environment.MAINNET, 100),
        )


def test_testnet_smoke_confirmation_matches_cli(app_paths, settings) -> None:
    store = StateStore(app_paths.database)
    store.initialize()
    controller = AgentController(app_paths, settings, store)
    with pytest.raises(ControlError, match="Gate combinado ausente"):
        controller.testnet_smoke(budget_usdc=25, confirmation="TESTNET SMOKE 25.00")
    with pytest.raises(ControlError, match="Confirmação inválida"):
        controller.testnet_smoke(budget_usdc=25, confirmation="RUN TESTNET SMOKE 25.00")


def test_daemon_release_and_testnet_payload_contracts(app_paths, settings, monkeypatch) -> None:
    async def idle_stream(_: HyperliquidStream) -> None:
        await asyncio.Event().wait()

    monkeypatch.setattr(HyperliquidStream, "run", idle_stream)
    monkeypatch.setattr(
        ReleaseManager, "verify", lambda _self, release_id: {"release_id": release_id}
    )
    monkeypatch.setattr(
        ReleaseManager,
        "approve",
        lambda _self, release_id, confirmation: {
            "release_id": release_id,
            "confirmation": confirmation,
        },
    )
    monkeypatch.setattr(
        AgentController,
        "testnet_preflight",
        lambda _self: {"environment": "testnet", "ok": True},
    )
    monkeypatch.setattr(
        AgentController,
        "testnet_smoke",
        lambda _self, **kwargs: kwargs | {"environment": "testnet"},
    )
    persisted = StateStore(app_paths.database)
    persisted.initialize()
    persisted.upsert_position("BTC", {"coin": "BTC", "szi": "0.01"})
    persisted.upsert_order(
        "0xcontract",
        symbol="BTC",
        side="long",
        status="open",
        payload={"coin": "BTC", "reduceOnly": True},
    )
    with TestClient(create_app(app_paths, settings)) as client:
        operational = client.get("/state").json()["metadata"]
        assert operational["positions"][0]["coin"] == "BTC"
        assert operational["orders"][0]["client_order_id"] == "0xcontract"
        assert (
            client.post("/release/verify", json={"release_id": "release-abc"}).json()["release_id"]
            == "release-abc"
        )
        assert (
            client.post(
                "/release/approve",
                json={
                    "release_id": "release-abc",
                    "confirmation": "APPROVE RELEASE release-abc",
                },
            ).status_code
            == 200
        )
        assert client.post("/testnet/preflight").json()["ok"] is True
        smoke = client.post(
            "/testnet/smoke",
            json={"budget_usdc": 25, "confirmation": "TESTNET SMOKE 25.00"},
        )
        assert smoke.status_code == 200
        assert smoke.json()["confirmation"] == "TESTNET SMOKE 25.00"
