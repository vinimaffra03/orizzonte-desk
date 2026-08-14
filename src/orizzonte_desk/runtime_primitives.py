from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Literal


class RuntimeInvariantError(RuntimeError):
    """A fail-closed runtime invariant was violated."""


RecoveryAction = Literal["submit", "protect", "reuse", "reject_unowned"]
ProtectionFailureAction = Literal["flatten_and_lock", "lock_unconfirmed", "continue"]
DeadManAction = Literal["schedule", "clear", "retain"]


def persist_once(event_id: str, recorder: Callable[[str], bool]) -> bool:
    """Delegate durable idempotency and preserve the recorder's inserted/not-inserted result."""
    if not event_id:
        raise RuntimeInvariantError("An idempotent event requires a stable id")
    return bool(recorder(event_id))


def protection_size_for_fill(
    *,
    requested_size: float,
    observed_filled_size: float,
    normalizer: Callable[[float], float],
) -> float:
    """Never protect more than the quantity authoritatively observed as filled."""
    if requested_size <= 0 or observed_filled_size <= 0:
        return 0.0
    bounded = min(requested_size, observed_filled_size)
    normalized = float(normalizer(bounded))
    if normalized < 0 or normalized > bounded + 1e-12:
        raise RuntimeInvariantError("Filled-quantity normalizer increased exposure")
    return normalized


def deterministic_intent_key(
    *, environment: str, bar: str, symbol: str, side: str, release_id: str
) -> str:
    values = (environment, bar, symbol, side, release_id)
    if any(not value for value in values):
        raise RuntimeInvariantError("Deterministic order intent has an empty binding")
    return ":".join(values)


def timeout_recovery_action(
    *,
    expected_cloid: str,
    tracked_entry_cloid: str | None,
    recovered_position_size: float,
    protection_count: int,
) -> RecoveryAction:
    """Choose the only safe action after an uncertain submit outcome."""
    if recovered_position_size <= 0:
        return "submit"
    if tracked_entry_cloid != expected_cloid:
        return "reject_unowned"
    if protection_count < 2:
        return "protect"
    return "reuse"


def enforce_time_guard(
    *,
    clock_skew_seconds: float | None = None,
    max_clock_skew_seconds: float | None = None,
    data_age_seconds: float | None = None,
    max_data_age_seconds: float | None = None,
) -> None:
    if (
        clock_skew_seconds is not None
        and max_clock_skew_seconds is not None
        and clock_skew_seconds > max_clock_skew_seconds
    ):
        raise RuntimeInvariantError(f"Clock drift excessivo: {clock_skew_seconds:.2f}s")
    if (
        data_age_seconds is not None
        and max_data_age_seconds is not None
        and data_age_seconds > max_data_age_seconds
    ):
        raise RuntimeInvariantError(f"Dados stale: {data_age_seconds:.2f}s")


def protection_failure_action(
    *, protection_confirmed: bool, flatten_confirmed: bool
) -> ProtectionFailureAction:
    if protection_confirmed:
        return "continue"
    if flatten_confirmed:
        return "flatten_and_lock"
    return "lock_unconfirmed"


def protection_kind(order: Mapping[str, object]) -> Literal["sl", "tp"] | None:
    value = str(
        order.get("kind")
        or order.get("tpsl")
        or order.get("orderType")
        or order.get("order_type")
        or ""
    ).lower()
    if value in {"sl", "stop_loss"} or "stop" in value:
        return "sl"
    if value in {"tp", "take_profit"} or "take profit" in value:
        return "tp"
    return None


def has_native_protection_pair(
    position: Mapping[str, object], orders: Sequence[Mapping[str, object]]
) -> bool:
    symbol = str(position.get("coin") or position.get("symbol") or "")
    signed_size = float(str(position.get("szi") or position.get("size") or 0))
    size = abs(signed_size)
    if not symbol or size <= 0:
        return False
    expected_sides = (
        {"a", "ask", "sell", "short", "close long"}
        if signed_size > 0
        else {"b", "bid", "buy", "long", "close short"}
    )
    kinds: set[str] = set()
    for order in orders:
        order_symbol = str(order.get("coin") or order.get("symbol") or "")
        order_size = abs(float(str(order.get("sz") or order.get("size") or 0)))
        side = str(order.get("side") or order.get("dir") or "").lower()
        if (
            order_symbol == symbol
            and bool(order.get("reduceOnly", order.get("reduce_only", False)))
            and order_size + 1e-12 >= size
            and side in expected_sides
            and (kind := protection_kind(order)) is not None
        ):
            kinds.add(kind)
    return kinds == {"sl", "tp"}


def dead_man_action(
    *, status: str, has_positions: bool, pending_entry: bool, positions_protected: bool
) -> DeadManAction:
    if has_positions:
        return "clear" if positions_protected else "retain"
    if status == "running" or pending_entry:
        return "schedule"
    return "retain"
