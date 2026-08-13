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


def save_gate(result: GateResult, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_combined_gate(paths: list[Path]) -> dict[str, Any]:
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    return {
        "passed": bool(payloads) and all(item.get("passed", False) for item in payloads),
        "gates": payloads,
        "evaluated_at": datetime.now(UTC).isoformat(),
    }
