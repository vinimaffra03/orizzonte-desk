from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from orizzonte_desk.constants import SYMBOLS
from orizzonte_desk.models import GateResult


def evaluate_gate(
    metrics: dict[str, float],
    by_symbol: dict[str, dict[str, float]],
    stressed_metrics: dict[str, float],
    *,
    dataset_hashes: tuple[str, ...] = (),
    model_hash: str | None = None,
) -> GateResult:
    positive_symbols = sum(by_symbol.get(symbol, {}).get("net_pnl", 0.0) > 0 for symbol in SYMBOLS)
    checks = {
        "sharpe_oos": metrics.get("sharpe", 0.0) >= 1.0,
        "profit_factor": metrics.get("profit_factor", 0.0) >= 1.15,
        "max_drawdown": metrics.get("max_drawdown", 1.0) <= 0.25,
        "positive_symbols": positive_symbols >= 3,
        "stress_expectancy": stressed_metrics.get("expectancy", -1.0) > 0,
        "stress_net_profit": stressed_metrics.get("net_profit", -1.0) > 0,
        "ruin_probability": metrics.get("ruin_probability_50", 1.0) < 0.01,
    }
    reasons = tuple(name for name, passed in checks.items() if not passed)
    numeric = dict(metrics)
    numeric["positive_symbols"] = float(positive_symbols)
    numeric["stress_expectancy"] = stressed_metrics.get("expectancy", 0.0)
    numeric["stress_net_profit"] = stressed_metrics.get("net_profit", 0.0)
    return GateResult(
        passed=all(checks.values()),
        evaluated_at=datetime.now(UTC),
        checks=checks,
        metrics=numeric,
        reasons=reasons,
        dataset_hashes=dataset_hashes,
        model_hash=model_hash,
    )


def save_gate(
    result: GateResult,
    path: Path,
    *,
    release_binding: dict[str, Any] | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = result.model_dump(mode="json")
    if release_binding is not None:
        payload["release_binding"] = release_binding
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_combined_gate(paths: list[Path]) -> dict[str, Any]:
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    bindings = [item.get("release_binding", {}) for item in payloads]
    shared_keys = ("config_fingerprint", "code_hash", "commit_hash")
    shared_invariants_match = (
        bool(bindings)
        and all(all(binding.get(key) for key in shared_keys) for binding in bindings)
        and all(
            all(binding.get(key) == bindings[0].get(key) for key in shared_keys)
            for binding in bindings[1:]
        )
    )
    protocol_pairs = [
        (payload, binding)
        for payload, binding in zip(payloads, bindings, strict=True)
        if binding.get("evaluation_scope") == "training_protocol"
    ]
    candidate_pairs = [
        (payload, binding)
        for payload, binding in zip(payloads, bindings, strict=True)
        if binding.get("evaluation_scope", "candidate") == "candidate"
    ]
    candidate_hashes = {
        str(binding["model_hash"]) for _, binding in candidate_pairs if binding.get("model_hash")
    }
    candidate_model_hash = next(iter(candidate_hashes)) if len(candidate_hashes) == 1 else None
    model_binding_valid = (
        bool(candidate_pairs)
        and candidate_model_hash is not None
        and all(
            payload.get("model_hash") == candidate_model_hash
            and binding.get("model_hash") == candidate_model_hash
            for payload, binding in candidate_pairs
        )
    )
    protocol_binding_valid = bool(protocol_pairs) and all(
        binding.get("protocol_hash")
        and not binding.get("model_hash")
        and not payload.get("model_hash")
        for payload, binding in protocol_pairs
    )
    bindings_match = shared_invariants_match and model_binding_valid and protocol_binding_valid
    protocol_hashes = sorted({str(binding["protocol_hash"]) for _, binding in protocol_pairs})
    return {
        "passed": (
            bool(payloads)
            and all(item.get("passed", False) for item in payloads)
            and bindings_match
        ),
        "gates": payloads,
        "bindings_match": bindings_match,
        "model_hash": candidate_model_hash,
        "release_binding": {
            **({key: bindings[0].get(key) for key in shared_keys} if bindings else {}),
            "model_hash": candidate_model_hash,
            "evaluation_scope": "combined_release",
            "protocol_hashes": protocol_hashes,
            "dataset_hashes": sorted(
                {value for binding in bindings for value in binding.get("dataset_hashes", [])}
            ),
        },
        "evaluated_at": datetime.now(UTC).isoformat(),
    }
