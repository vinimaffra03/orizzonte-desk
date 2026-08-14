from __future__ import annotations

import pandas as pd

from orizzonte_desk.backtest import EventBacktester


def test_external_holdout_never_runs_regime_challenger_and_baseline_has_no_gate(
    app_paths, settings
) -> None:
    timestamp = pd.Timestamp("2026-01-01T00:00:00Z")
    rows = []
    for symbol, price in (("BTC", 100.0), ("ETH", 50.0), ("SOL", 20.0), ("XRP", 1.0)):
        rows.append(
            {
                "timestamp": timestamp,
                "symbol": symbol,
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "funding_rate": 0.0,
                "signal_raw": 0,
                "signal": 0,
                "setup_score": 0.0,
                "ml_probability": 0.0,
                "decision_threshold": 0.0,
                "decision_accepted": False,
                "decision_policy_id": "none",
                "stop_distance": 1.0,
                "atr_4h": 1.0,
                "daily_trend": 0.0,
                "label": None,
            }
        )
    enriched = pd.DataFrame(rows)
    market = enriched.loc[
        :, ["timestamp", "symbol", "open", "high", "low", "close", "funding_rate"]
    ]
    market.attrs["dataset_role"] = "external_holdout"

    result = EventBacktester(settings, app_paths).run(
        market,
        source="hyperliquid-mainnet",
        dataset_hash="a" * 64,
        enriched_override=enriched,
        run_stress_suite=False,
    )

    assert not result.gate_path.exists()
    assert "gate" not in result.artifacts
    assert "regime_study" not in result.artifacts
    assert {path.name for path in result.artifacts.values()} >= {
        "funnel-events.parquet",
        "funnel-summary.json",
        "funnel-by-fold.csv",
        "probability-calibration.csv",
    }
