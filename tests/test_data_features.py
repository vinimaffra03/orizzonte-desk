from __future__ import annotations

import json

import pandas as pd
import pytest

from orizzonte_desk.data import DatasetManager, sha256_file
from orizzonte_desk.features import prepare_features
from orizzonte_desk.regimes import build_regime_arms


def test_synthetic_dataset_is_versioned_and_valid(app_paths, settings) -> None:
    manager = DatasetManager(app_paths, settings)
    manifest = manager.generate_synthetic(hours=800, seed=7)
    assert manifest.rows == 800 * 4
    assert manifest.sha256 == sha256_file(
        app_paths.processed_data / f"{manifest.dataset_id}.parquet"
    )
    payload = json.loads((app_paths.manifests / f"{manifest.dataset_id}.json").read_text())
    assert payload["sha256"] == manifest.sha256
    loaded = manager.load(manifest.path)
    assert set(loaded["symbol"]) == {"BTC", "ETH", "SOL", "XRP"}


def test_dataset_rejects_invalid_ohlc(app_paths, settings) -> None:
    manager = DatasetManager(app_paths, settings)
    frame = pd.DataFrame(
        [
            {
                "timestamp": "2026-01-01T00:00:00Z",
                "symbol": "BTC",
                "interval": "1h",
                "open": 100,
                "high": 90,
                "low": 80,
                "close": 95,
                "volume": 1,
                "funding_rate": 0,
            }
        ]
    )
    with pytest.raises(ValueError, match="OHLC"):
        manager.validate_frame(frame)


def test_features_do_not_change_past_when_future_is_appended(app_paths, settings) -> None:
    manager = DatasetManager(app_paths, settings)
    manifest = manager.generate_synthetic(hours=3000, seed=17)
    frame = manager.load(manifest.path)
    cutoff = frame["timestamp"].sort_values().unique()[2500]
    past = frame[frame["timestamp"] <= cutoff]
    full_features = prepare_features(frame, settings.strategy)
    past_features = prepare_features(past, settings.strategy)
    assert full_features["risk_fraction"].equals(
        (full_features["stop_distance"] / full_features["close"]).replace(0, float("nan"))
    )
    assert not build_regime_arms(full_features).empty
    columns = [
        "timestamp",
        "symbol",
        "trend_strength_4h",
        "daily_trend",
        "weekly_trend",
        "signal_raw",
        "risk_fraction",
    ]
    expected = full_features[full_features["timestamp"] <= cutoff][columns].reset_index(drop=True)
    actual = past_features[columns].reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
