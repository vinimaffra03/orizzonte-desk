from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from orizzonte_desk import runtime_primitives as primitives
from orizzonte_desk.exchange import cloid_for_key
from orizzonte_desk.models import REQUIRED_TESTNET_SCENARIOS


class ChaosValidationError(RuntimeError):
    """Raised when the deterministic operational chaos gate is incomplete or fails."""


@dataclass(frozen=True)
class TestnetChaosContext:
    lifecycle: Mapping[str, object]


ScenarioProbe = Callable[[TestnetChaosContext], tuple[bool, dict[str, object]]]


class TestnetChaosRunner:
    """Runs deterministic, offline fault probes after the bounded real testnet smoke.

    The probes never call a trading gateway. They exercise fail-closed state transitions and
    bind their complete evidence to the certificate emitted by the real smoke lifecycle.
    """

    def __init__(self, scenarios: Mapping[str, ScenarioProbe] | None = None) -> None:
        self.scenarios = dict(scenarios) if scenarios is not None else _default_scenarios()

    def run(self, context: TestnetChaosContext) -> dict[str, object]:
        required = tuple(REQUIRED_TESTNET_SCENARIOS)
        missing = sorted(set(required) - set(self.scenarios))
        unexpected = sorted(set(self.scenarios) - set(required))
        if missing or unexpected:
            raise ChaosValidationError(
                f"Chaos suite mismatch: missing={missing}, unexpected={unexpected}"
            )
        results: dict[str, bool] = {}
        evidence: dict[str, dict[str, object]] = {}
        hashes: dict[str, str] = {}
        for name in required:
            try:
                passed, scenario_evidence = self.scenarios[name](context)
            except Exception as exc:
                passed = False
                scenario_evidence = {
                    "probe_error": type(exc).__name__,
                    "message": str(exc),
                }
            normalized = _json_normalize(scenario_evidence)
            results[name] = bool(passed)
            evidence[name] = normalized
            hashes[name] = _content_hash(
                {"scenario": name, "passed": bool(passed), "evidence": normalized}
            )
        failed = sorted(name for name, passed in results.items() if not passed)
        if failed:
            raise ChaosValidationError(f"Operational chaos scenarios failed: {failed}")
        payload: dict[str, object] = {
            "schema_version": 1,
            "required_scenarios": list(required),
            "results": results,
            "scenario_hashes": hashes,
            "evidence": evidence,
        }
        return payload | {"report_hash": _content_hash(payload)}

    @staticmethod
    def verify(report: Mapping[str, object]) -> bool:
        payload = dict(report)
        report_hash = str(payload.pop("report_hash", ""))
        if report_hash != _content_hash(payload):
            return False
        required_value = payload.get("required_scenarios")
        if not isinstance(required_value, (list, tuple)):
            return False
        required = tuple(str(item) for item in required_value)
        results = payload.get("results")
        hashes = payload.get("scenario_hashes")
        evidence = payload.get("evidence")
        if (
            required != tuple(REQUIRED_TESTNET_SCENARIOS)
            or not isinstance(results, dict)
            or not isinstance(hashes, dict)
            or not isinstance(evidence, dict)
            or set(results) != set(required)
            or set(hashes) != set(required)
            or set(evidence) != set(required)
        ):
            return False
        for name in required:
            if results[name] is not True or not isinstance(evidence[name], dict):
                return False
            expected = _content_hash({"scenario": name, "passed": True, "evidence": evidence[name]})
            if hashes[name] != expected:
                return False
        return True


def _default_scenarios() -> dict[str, ScenarioProbe]:
    return {
        "partial_fill": _partial_fill,
        "native_sl_tp": _native_sl_tp,
        "duplicate_ws": _duplicate_ws,
        "timeout_after_accept": _timeout_after_accept,
        "clock_drift_stale": _clock_drift_stale,
        "protection_failure_flatten_lock": _protection_failure_flatten_lock,
        "dead_man": _dead_man,
        "empty_account": _empty_account,
    }


def _partial_fill(_: TestnetChaosContext) -> tuple[bool, dict[str, object]]:
    requested = 1.0
    partial_observed = 0.4
    complete_observed = 1.0

    def normalizer(value: float) -> float:
        return round(value, 3)

    partial_protection = primitives.protection_size_for_fill(
        requested_size=requested,
        observed_filled_size=partial_observed,
        normalizer=normalizer,
    )
    complete_protection = primitives.protection_size_for_fill(
        requested_size=requested,
        observed_filled_size=complete_observed,
        normalizer=normalizer,
    )
    evidence: dict[str, object] = {
        "requested": requested,
        "partial_observed": partial_observed,
        "partial_protection": partial_protection,
        "complete_observed": complete_observed,
        "complete_protection": complete_protection,
    }
    return (
        partial_protection == partial_observed
        and complete_protection == complete_observed
        and partial_protection < requested
    ), evidence


def _native_sl_tp(context: TestnetChaosContext) -> tuple[bool, dict[str, object]]:
    raw = context.lifecycle.get("protection_evidence")
    position = raw.get("position") if isinstance(raw, dict) else None
    orders = raw.get("orders") if isinstance(raw, dict) else None
    valid_orders = (
        [item for item in orders if isinstance(item, dict)] if isinstance(orders, list) else []
    )
    pair_confirmed = bool(
        isinstance(position, dict) and primitives.has_native_protection_pair(position, valid_orders)
    )
    duplicate_only_rejected = True
    if isinstance(position, dict) and valid_orders:
        duplicate_only_rejected = not primitives.has_native_protection_pair(
            position,
            [valid_orders[0], dict(valid_orders[0]) | {"oid": "duplicate"}],
        )
    evidence: dict[str, object] = {
        "pair_confirmed": pair_confirmed,
        "duplicate_single_kind_rejected": duplicate_only_rejected,
        "position": position,
        "orders": valid_orders,
    }
    return pair_confirmed and duplicate_only_rejected, evidence


def _duplicate_ws(_: TestnetChaosContext) -> tuple[bool, dict[str, object]]:
    frames = ("fill-42", "fill-42", "fill-42")
    persisted: set[str] = set()

    def recorder(event_id: str) -> bool:
        if event_id in persisted:
            return False
        persisted.add(event_id)
        return True

    applied = sum(primitives.persist_once(event_id, recorder) for event_id in frames)
    evidence = {"received": len(frames), "applied": applied, "fill_ids": sorted(persisted)}
    return applied == 1 and persisted == {"fill-42"}, evidence


def _timeout_after_accept(_: TestnetChaosContext) -> tuple[bool, dict[str, object]]:
    kwargs = {
        "environment": "testnet",
        "bar": "2026-01-01T00:00:00+00:00",
        "symbol": "BTC",
        "side": "long",
        "release_id": "release-chaos",
    }
    first_intent = primitives.deterministic_intent_key(**kwargs)
    retry_intent = primitives.deterministic_intent_key(**kwargs)
    expected_cloid = str(cloid_for_key(first_intent))
    retry_cloid = str(cloid_for_key(retry_intent))
    action = primitives.timeout_recovery_action(
        expected_cloid=expected_cloid,
        tracked_entry_cloid=expected_cloid,
        recovered_position_size=0.1,
        protection_count=0,
    )
    evidence: dict[str, object] = {
        "first_cloid": expected_cloid,
        "retry_cloid": retry_cloid,
        "recovered_position": 0.1,
        "protection_count": 0,
        "recovery_action": action,
    }
    return expected_cloid == retry_cloid and action == "protect", evidence


def _clock_drift_stale(_: TestnetChaosContext) -> tuple[bool, dict[str, object]]:
    policy = {"max_clock_drift_seconds": 5, "max_data_age_seconds": 120}
    injected = {"clock_drift_seconds": 6, "data_age_seconds": 121}
    rejected: dict[str, bool] = {}
    for name, arguments in {
        "clock": {
            "clock_skew_seconds": injected["clock_drift_seconds"],
            "max_clock_skew_seconds": policy["max_clock_drift_seconds"],
        },
        "stale": {
            "data_age_seconds": injected["data_age_seconds"],
            "max_data_age_seconds": policy["max_data_age_seconds"],
        },
    }.items():
        try:
            primitives.enforce_time_guard(**arguments)
            rejected[name] = False
        except primitives.RuntimeInvariantError:
            rejected[name] = True
    return all(rejected.values()), {"policy": policy, "injected": injected, "rejected": rejected}


def _protection_failure_flatten_lock(
    _: TestnetChaosContext,
) -> tuple[bool, dict[str, object]]:
    flattened = primitives.protection_failure_action(
        protection_confirmed=False,
        flatten_confirmed=True,
    )
    unconfirmed = primitives.protection_failure_action(
        protection_confirmed=False,
        flatten_confirmed=False,
    )
    evidence: dict[str, object] = {
        "confirmed_flatten_action": flattened,
        "unconfirmed_flatten_action": unconfirmed,
    }
    return flattened == "flatten_and_lock" and unconfirmed == "lock_unconfirmed", evidence


def _dead_man(context: TestnetChaosContext) -> tuple[bool, dict[str, object]]:
    scheduled = "dead_man" in context.lifecycle
    observed = {
        "flat_running": primitives.dead_man_action(
            status="running",
            has_positions=False,
            pending_entry=False,
            positions_protected=False,
        ),
        "pending_entry": primitives.dead_man_action(
            status="paused",
            has_positions=False,
            pending_entry=True,
            positions_protected=False,
        ),
        "protected_position": primitives.dead_man_action(
            status="paused",
            has_positions=True,
            pending_entry=False,
            positions_protected=True,
        ),
        "unprotected_position": primitives.dead_man_action(
            status="locked",
            has_positions=True,
            pending_entry=False,
            positions_protected=False,
        ),
    }
    expected = {
        "flat_running": "schedule",
        "pending_entry": "schedule",
        "protected_position": "clear",
        "unprotected_position": "retain",
    }
    return scheduled and observed == expected, {
        "gateway_acknowledged": scheduled,
        "observed": observed,
        "expected": expected,
    }


def _empty_account(context: TestnetChaosContext) -> tuple[bool, dict[str, object]]:
    preflight = context.lifecycle.get("preflight")
    final = context.lifecycle.get("final")
    preflight_empty = isinstance(preflight, dict) and preflight.get("empty") is True
    final_empty = (
        isinstance(final, dict)
        and int(final.get("positions", -1)) == 0
        and int(final.get("orders", -1)) == 0
    )
    return preflight_empty and final_empty, {
        "preflight_empty": preflight_empty,
        "final_empty": final_empty,
    }


def _content_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _json_normalize(value: object) -> dict[str, object]:
    normalized = json.loads(json.dumps(value, sort_keys=True, default=str))
    if not isinstance(normalized, dict):
        raise ChaosValidationError("Scenario evidence must be a JSON object")
    return normalized
