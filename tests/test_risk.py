from __future__ import annotations

from datetime import UTC, datetime

import pytest
from hypothesis import given
from hypothesis import strategies as st

from orizzonte_desk.models import Position, RiskSnapshot, Side, Signal
from orizzonte_desk.risk import RiskManager, RiskViolation, floor_to_increment


def signal(symbol: str = "BTC", side: Side = Side.LONG) -> Signal:
    return Signal(
        timestamp=datetime.now(UTC),
        symbol=symbol,
        side=side,
        score=0.8,
        probability=0.7,
        entry_reference=100.0,
        stop_distance=2.0,
        atr=1.0,
        regime="bull" if side is Side.LONG else "bear",
    )


def snapshot(**updates: object) -> RiskSnapshot:
    base: dict[str, object] = {
        "budget": 10_000.0,
        "equity": 10_000.0,
        "day_start_equity": 10_000.0,
        "high_water_mark": 10_000.0,
        "open_risk": 0.0,
        "open_notional": 0.0,
        "positions_count": 0,
    }
    base.update(updates)
    return RiskSnapshot.model_validate(base)


def test_sizing_risks_exactly_one_percent(settings) -> None:
    manager = RiskManager(settings.risk)
    sizing = manager.size_order(
        signal(), budget=10_000, equity=10_000, open_notional=0, size_increment=0.001
    )
    assert sizing.risk_usdc == pytest.approx(100)
    assert sizing.size == pytest.approx(50)
    assert sizing.stop_price == pytest.approx(98)
    assert sizing.take_profit_price == pytest.approx(102)
    assert sizing.margin_required == pytest.approx(500)


def test_daily_and_drawdown_locks_latch(settings) -> None:
    manager = RiskManager(settings.risk)
    assert manager.evaluate_locks(snapshot(equity=10_100)).profit_locked
    assert manager.evaluate_locks(snapshot(equity=9_600)).loss_locked
    assert manager.evaluate_locks(snapshot(equity=7_500)).drawdown_locked


def test_no_averaging_down(settings) -> None:
    manager = RiskManager(settings.risk)
    position = Position(
        symbol="BTC",
        side=Side.LONG,
        size=1,
        initial_size=1,
        entry_price=100,
        stop_price=98,
        take_profit_price=102,
        opened_at=datetime.now(UTC),
        initial_risk_usdc=100,
    )
    with pytest.raises(RiskViolation, match="averaging down"):
        manager.assert_new_entry_allowed(signal(), snapshot(), [position])


def test_correlated_altcoin_block(settings) -> None:
    manager = RiskManager(settings.risk)
    position = Position(
        symbol="SOL",
        side=Side.LONG,
        size=1,
        initial_size=1,
        entry_price=100,
        stop_price=98,
        take_profit_price=102,
        opened_at=datetime.now(UTC),
        initial_risk_usdc=100,
    )
    with pytest.raises(RiskViolation, match="redundante"):
        manager.assert_new_entry_allowed(
            signal("XRP"), snapshot(), [position], {("SOL", "XRP"): 0.9}
        )


@given(
    value=st.floats(min_value=0, max_value=1_000_000, allow_nan=False, allow_infinity=False),
    increment=st.sampled_from([0.1, 0.01, 0.001, 0.00001]),
)
def test_floor_increment_never_exceeds_input(value: float, increment: float) -> None:
    result = floor_to_increment(value, increment)
    assert result <= value + 1e-9
    assert result >= 0
