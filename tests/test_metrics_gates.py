from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from orizzonte_desk.gates import evaluate_gate, save_gate
from orizzonte_desk.metrics import calculate_metrics
from orizzonte_desk.models import Side, Trade


def test_metrics_and_gate_are_explicit(tmp_path: Path) -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    equity = pd.DataFrame(
        {
            "timestamp": [start + timedelta(days=index) for index in range(200)],
            "equity": [10_000 * (1.001**index) for index in range(200)],
        }
    )
    trades = [
        Trade(
            symbol=symbol,
            side=Side.LONG,
            opened_at=start + timedelta(days=index),
            closed_at=start + timedelta(days=index, hours=4),
            entry_price=100,
            exit_price=101,
            size=10,
            gross_pnl=10,
            net_pnl=9,
            fees=1,
            funding=0,
            slippage=0,
            exit_reason="target",
        )
        for index, symbol in enumerate(["BTC", "ETH", "SOL", "XRP"] * 20)
    ]
    metrics = calculate_metrics(equity, trades, initial_capital=10_000, monte_carlo_samples=100)
    assert metrics.summary["total_return"] > 0
    assert metrics.summary["profit_factor"] > 1
    evaluated_at = equity["timestamp"].max()
    result = evaluate_gate(
        metrics.summary,
        metrics.by_symbol,
        metrics.summary,
        evaluated_at=evaluated_at,
    )
    assert set(result.checks) == {
        "sharpe_oos",
        "profit_factor",
        "max_drawdown",
        "positive_symbols",
        "stress_expectancy",
        "stress_net_profit",
        "ruin_probability",
    }
    first = save_gate(result, tmp_path / "first-gate.json")
    second = save_gate(result, tmp_path / "second-gate.json")
    assert first.read_bytes() == second.read_bytes()
    assert b"\r\n" not in first.read_bytes()
    assert result.evaluated_at == evaluated_at
