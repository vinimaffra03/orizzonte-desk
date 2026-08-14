from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from orizzonte_desk.exchange import (
    AccountSnapshot,
    AssetMetadata,
    ExchangeError,
    HyperliquidGateway,
    cloid_for_key,
)
from orizzonte_desk.models import Environment, Side, Signal
from orizzonte_desk.storage import StateStore


def _signal() -> Signal:
    return Signal(
        timestamp=datetime.now(UTC),
        symbol="BTC",
        side=Side.LONG,
        score=0.9,
        probability=0.9,
        entry_reference=60_000,
        stop_distance=600,
        atr=300,
        regime="bull",
    )


class _PartialFillExchange:
    def __init__(self) -> None:
        self.protection_sizes: list[float] = []

    def update_leverage(self, *_: Any, **__: Any) -> dict[str, Any]:
        return {"status": "ok"}

    def market_open(self, *_: Any, **__: Any) -> dict[str, Any]:
        return {
            "status": "ok",
            "response": {
                "data": {"statuses": [{"filled": {"totalSz": "0.4", "avgPx": "60010", "oid": 1}}]}
            },
        }

    def order(self, _symbol: str, _buy: bool, size: float, *_: Any, **__: Any) -> dict[str, Any]:
        self.protection_sizes.append(size)
        return {"status": "ok"}


def _gateway(app_paths) -> HyperliquidGateway:
    store = StateStore(app_paths.database)
    store.initialize()
    gateway = object.__new__(HyperliquidGateway)
    gateway.environment = Environment.TESTNET
    gateway.account_address = "0x" + "1" * 40
    gateway.wallet_address = "0x" + "2" * 40
    gateway.store = store
    gateway._metadata = {"BTC": AssetMetadata("BTC", 5, 1e-5, 1)}
    return gateway


def test_partial_fill_protection_uses_actual_filled_quantity(app_paths) -> None:
    gateway = _gateway(app_paths)
    exchange = _PartialFillExchange()
    gateway.exchange = exchange
    gateway.info = SimpleNamespace()
    result = gateway.place_entry_with_protection(
        _signal(),
        size=1,
        stop_price=59_400,
        take_profit_price=60_600,
        slippage=0.01,
        idempotency_key="bar:BTC:long",
    )
    assert result["filled_size"] == 0.4
    assert exchange.protection_sizes == [0.4, 0.4]
    assert len(gateway.store.fills()) == 1


def test_deterministic_cloid_is_stable_and_16_bytes() -> None:
    first = str(cloid_for_key("same execution intent"))
    assert first == str(cloid_for_key("same execution intent"))
    assert first != str(cloid_for_key("different intent"))
    assert first.startswith("0x") and len(first) == 34


class _ProtectionFailureExchange(_PartialFillExchange):
    def order(self, *_: Any, **__: Any) -> dict[str, Any]:
        raise TimeoutError("protection timeout")

    def market_close(self, *_: Any, **__: Any) -> dict[str, Any]:
        return {"status": "ok"}


class _ProtectionAndCloseFailureExchange(_ProtectionFailureExchange):
    def market_close(self, *_: Any, **__: Any) -> dict[str, Any]:
        raise TimeoutError("close outcome unknown")


class _ProtectionReplacementExchange:
    def __init__(self, *, fail_new: bool = False) -> None:
        self.fail_new = fail_new
        self.calls: list[str] = []

    def order(self, *_: Any, **__: Any) -> dict[str, Any]:
        self.calls.append("order")
        if self.fail_new:
            raise TimeoutError("new protection timeout")
        return {"status": "ok"}

    def cancel(self, *_: Any, **__: Any) -> dict[str, Any]:
        self.calls.append("cancel")
        return {"status": "ok"}


def test_protection_failure_confirms_emergency_close_and_latches_lock(
    app_paths, monkeypatch
) -> None:
    gateway = _gateway(app_paths)
    gateway.exchange = _ProtectionFailureExchange()
    gateway.info = SimpleNamespace()
    monkeypatch.setattr(
        gateway,
        "reconcile",
        lambda: AccountSnapshot(10_000, 10_000, (), (), {"BTC": 60_000}),
    )
    with pytest.raises(ExchangeError, match="fechada imediatamente"):
        gateway.place_entry_with_protection(
            _signal(),
            size=1,
            stop_price=59_400,
            take_profit_price=60_600,
            slippage=0.01,
        )
    assert gateway.store.lock("unprotected_position")["locked"] is True


def test_emergency_close_failure_latches_lock_before_network_attempt(app_paths) -> None:
    gateway = _gateway(app_paths)
    gateway.exchange = _ProtectionAndCloseFailureExchange()
    gateway.info = SimpleNamespace()
    with pytest.raises(ExchangeError, match="lock persistente"):
        gateway.place_entry_with_protection(
            _signal(),
            size=1,
            stop_price=59_400,
            take_profit_price=60_600,
            slippage=0.01,
        )
    lock = gateway.store.lock("unprotected_position")
    assert lock["locked"] is True
    assert lock["payload"]["emergency_close"] == "pending"


@pytest.mark.parametrize("fail_new", [False, True])
def test_replacement_never_cancels_old_protection_before_new_pair(
    app_paths, monkeypatch, fail_new: bool
) -> None:
    gateway = _gateway(app_paths)
    exchange = _ProtectionReplacementExchange(fail_new=fail_new)
    gateway.exchange = exchange
    gateway.info = SimpleNamespace()
    monkeypatch.setattr(
        gateway,
        "snapshot",
        lambda: AccountSnapshot(
            10_000,
            10_000,
            (
                {"coin": "BTC", "oid": 1, "reduceOnly": True},
                {"coin": "BTC", "oid": 2, "reduceOnly": True},
            ),
            (),
            {"BTC": 60_000},
        ),
    )
    if fail_new:
        with pytest.raises(TimeoutError, match="new protection"):
            gateway.replace_protection(
                "BTC",
                side=Side.LONG,
                size=1,
                stop_price=59_400,
                take_profit_price=60_600,
            )
        assert exchange.calls == ["order"]
    else:
        gateway.replace_protection(
            "BTC",
            side=Side.LONG,
            size=1,
            stop_price=59_400,
            take_profit_price=60_600,
        )
        assert exchange.calls == ["order", "order", "cancel", "cancel"]


def test_adapter_has_no_transfer_or_withdraw_surface() -> None:
    forbidden = {"withdraw", "transfer", "usd_transfer", "spot_transfer"}
    assert not forbidden.intersection(dir(HyperliquidGateway))
