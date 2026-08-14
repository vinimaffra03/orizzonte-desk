from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from orizzonte_desk.decision import DecisionPolicy, DecisionPolicySelector
from orizzonte_desk.ml import MetaModelRegistry


def _decision_frame(return_value: float = 0.02) -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=500, freq="h", tz="UTC")
    rows = [(stamp, symbol) for stamp in timestamps for symbol in ("BTC", "ETH", "SOL", "XRP")]
    frame = pd.DataFrame(rows, columns=["timestamp", "symbol"])
    frame["probability"] = np.tile(np.linspace(0.5, 0.99, len(timestamps)), 4)
    frame["realized_return"] = return_value
    return frame.sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def _selector() -> DecisionPolicySelector:
    return DecisionPolicySelector(
        validation_fraction=0.2,
        purge_hours=24,
        quantiles=tuple(value / 100 for value in range(50, 100, 5)),
        min_validation_trades=30,
        bootstrap_samples=1000,
        lcb_quantile=0.05,
        seed=17,
    )


def _event_objective(threshold: float, validation: pd.DataFrame) -> np.ndarray:
    selected = validation[validation["probability"] >= threshold]
    return selected["realized_return"].to_numpy(dtype=float) - 0.002


def test_policy_is_nested_content_addressed_and_writes_full_funnel(tmp_path) -> None:
    selection = _selector().select(
        _decision_frame(),
        model_hash="a" * 64,
        round_trip_cost=0.001,
        evaluator=_event_objective,
    )
    policy = selection.policy

    assert policy.trade_enabled is True
    assert policy.validation_trades >= 30
    assert policy.bootstrap_samples == 1000
    assert policy.cost_multiplier == 2.0
    assert policy.objective == "event_backtest_net_expectancy_r_lcb_p05"
    assert policy.seed == 17
    assert policy.expectancy_lcb_p05 > 0
    assert policy.expectancy_lcb_p05 <= policy.expectancy_p50 <= policy.expectancy_p95
    assert pd.Timestamp(policy.reference_end) < pd.Timestamp(
        policy.validation_start
    ) - pd.Timedelta(hours=24)
    assert DecisionPolicy.from_payload(policy.to_payload()) == policy
    repeated = (
        _selector()
        .select(
            _decision_frame(),
            model_hash="a" * 64,
            round_trip_cost=0.001,
            evaluator=_event_objective,
        )
        .policy
    )
    assert repeated.policy_id == policy.policy_id
    tampered = policy.to_payload()
    tampered["probability_threshold"] += 0.01
    with pytest.raises(RuntimeError, match="Hash"):
        DecisionPolicy.from_payload(tampered)
    artifacts = selection.write(tmp_path)
    assert {path.suffix for path in artifacts.values()} == {".json", ".csv", ".parquet"}
    assert json.loads(artifacts["decision_policy"].read_text())["policy_id"] == policy.policy_id


def test_negative_lcb_produces_explicit_no_trade_policy() -> None:
    policy = (
        _selector()
        .select(
            _decision_frame(return_value=-0.02),
            model_hash="b" * 64,
            round_trip_cost=0.001,
            evaluator=_event_objective,
        )
        .policy
    )

    assert policy.trade_enabled is False
    assert policy.expectancy_lcb_p05 < 0.0
    assert policy.expectancy_lcb_p05 <= policy.expectancy_p50 <= policy.expectancy_p95
    assert policy.no_trade_reason is not None
    assert not policy.apply(np.asarray([0.5, 0.99, 1.0])).any()


def test_event_evaluator_counts_executed_trades_not_overlapping_candidates() -> None:
    frame = _decision_frame()

    def risk_constrained_event_evaluator(threshold: float, validation: pd.DataFrame) -> np.ndarray:
        assert len(validation[validation["probability"] >= threshold]) > 2
        # Only two candidates survive overlap/correlation/sizing and become closed trades.
        return np.asarray([0.4, 0.2])

    selection = _selector().select(
        frame,
        model_hash="c" * 64,
        round_trip_cost=0.001,
        evaluator=risk_constrained_event_evaluator,
    )

    assert selection.diagnostics["validation_trades"].eq(2).all()
    assert (selection.diagnostics["validation_candidates"] > 2).all()
    assert selection.policy.trade_enabled is False
    assert selection.policy.validation_trades == 2


def test_registry_objective_uses_event_trades_and_two_x_costs(
    app_paths, settings, monkeypatch
) -> None:
    from orizzonte_desk.backtest import EventBacktester

    calls: list[dict[str, Any]] = []

    def fake_run(self, market, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(kwargs)
        assert kwargs["cost_multiplier"] == 2.0
        assert kwargs["persist"] is False
        assert int(kwargs["enriched_override"]["decision_accepted"].sum()) > 2
        return SimpleNamespace(
            trades=[SimpleNamespace(net_pnl=25.0), SimpleNamespace(net_pnl=-10.0)]
        )

    monkeypatch.setattr(EventBacktester, "run", fake_run)
    timestamps = pd.date_range("2025-01-01", periods=40, freq="h", tz="UTC")
    full = pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": "BTC",
            "signal_raw": 1,
            "close": 100.0,
        }
    )
    validation = full.loc[:, ["timestamp", "symbol"]].copy()
    validation["probability"] = 0.9
    evaluator = MetaModelRegistry(app_paths, settings=settings)._event_threshold_evaluator(
        full,
        seed=7,
        dataset_hash="d" * 64,
    )

    objective = evaluator(0.8, validation)

    assert calls
    assert objective.tolist() == pytest.approx([0.25, -0.1])
