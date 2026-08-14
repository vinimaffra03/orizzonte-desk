from __future__ import annotations

import pandas as pd

from orizzonte_desk.regimes import REGIME_ARMS, RegimeStudy


def _features() -> pd.DataFrame:
    timestamps = pd.date_range("2023-01-02", periods=24 * 7 * 70, freq="h", tz="UTC")
    frame = pd.DataFrame({"timestamp": timestamps, "symbol": "BTC"})
    frame["daily_trend"] = 1.0
    frame["weekly_trend"] = 1.0
    frame["momentum_24h"] = 0.01
    frame["volume_zscore"] = 0.0
    frame["breakout_long"] = True
    frame["breakout_short"] = False
    frame["pullback_long"] = False
    frame["pullback_short"] = False
    frame["signal_raw"] = 1
    frame["close"] = 100.0
    frame["stop_distance"] = 1.0
    frame["risk_fraction"] = frame["stop_distance"] / frame["close"]
    frame["forward_return_24h"] = 0.01
    frame["forward_r_24h"] = frame["forward_return_24h"] / frame["risk_fraction"]
    frame["decision_accepted"] = frame.index % 2 == 0
    return frame


def _study() -> RegimeStudy:
    return RegimeStudy(
        primary_lookback_weeks=26,
        sensitivity_weeks=(13, 26, 52),
        decision_weekday=0,
        decision_hour_utc=0,
        decision_minute_utc=5,
        minimum_trades=30,
        validation_fraction=0.2,
        purge_hours=24,
        bootstrap_samples=100,
        lcb_quantile=0.05,
        seed=9,
    )


def test_formal_regime_study_is_causal_and_challenger_only(tmp_path) -> None:
    result = _study().run(_features(), round_trip_cost=0.001)

    assert result.summary["arms"] == list(REGIME_ARMS)
    assert result.summary["pooled_arm"] == "hybrid"
    assert result.summary["gate_eligible"] is False
    assert result.summary["promotion_eligible"] is False
    weekly = result.decisions[result.decisions["policy"] == "weekly"]
    assert set(weekly["lookback_weeks"]) == {13, 26, 52}
    assert (pd.to_datetime(weekly["decision_at"], utc=True).dt.weekday == 0).all()
    assert (pd.to_datetime(weekly["decision_at"], utc=True).dt.minute == 5).all()
    assert (
        pd.to_datetime(weekly["observable_end"], utc=True)
        <= pd.to_datetime(weekly["decision_at"], utc=True) - pd.Timedelta(hours=24)
    ).all()
    assert (
        pd.to_datetime(weekly["observable_end"], utc=True)
        - pd.to_datetime(weekly["history_start"], utc=True)
        == pd.to_timedelta(weekly["lookback_weeks"] * 7, unit="D")
    ).all()
    expected = {
        "signal_pooled",
        "signal_static",
        "signal_weekly",
        "signal_pooled_ml",
        "signal_static_ml",
        "signal_weekly_ml",
        "signal_hybrid",
        "signal_breakout",
        "signal_pullback",
        "signal_flat",
        "signal_hybrid_ml",
        "signal_breakout_ml",
        "signal_pullback_ml",
        "signal_flat_ml",
    }
    assert expected <= set(result.challenger)
    artifacts = result.write(tmp_path)
    assert {
        "regime-study.json",
        "regime-transitions.csv",
        "weekly-decisions.csv",
        "strategy-ablation.csv",
        "asset-regime-direction-setup-matrix.csv",
    } <= {path.name for path in artifacts.values()}
    assert not result.matrix.empty
    assert result.matrix["gross_expectancy"].eq(1.0).all()
    assert result.matrix["costs_2x"].gt(0).all()
    assert {
        "regime-decisions.parquet",
        "regime-challenger.csv",
        "regime-challenger.parquet",
    } <= {path.name for path in artifacts.values()}


def test_future_outcomes_do_not_change_prior_weekly_decisions() -> None:
    baseline = _features()
    changed = baseline.copy()
    cutoff = baseline["timestamp"].iloc[-24 * 7 * 4]
    changed.loc[changed["timestamp"] >= cutoff, "forward_return_24h"] = -0.5
    changed.loc[changed["timestamp"] >= cutoff, "forward_r_24h"] = -50.0

    first = _study().run(baseline, round_trip_cost=0.001).decisions
    second = _study().run(changed, round_trip_cost=0.001).decisions
    columns = ["decision_at", "lookback_weeks", "regime_state", "selected_arm"]
    first_prior = first[(first["policy"] == "weekly") & (first["decision_at"] < cutoff)][columns]
    second_prior = second[(second["policy"] == "weekly") & (second["decision_at"] < cutoff)][
        columns
    ]
    pd.testing.assert_frame_equal(
        first_prior.reset_index(drop=True), second_prior.reset_index(drop=True)
    )
