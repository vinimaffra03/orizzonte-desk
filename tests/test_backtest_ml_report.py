from __future__ import annotations

import pytest

from orizzonte_desk.backtest import EventBacktester, walk_forward_windows
from orizzonte_desk.data import DatasetManager
from orizzonte_desk.ml import MetaModelRegistry
from orizzonte_desk.reports import generate_report
from orizzonte_desk.strategy import SignalGenerator


def test_backtest_is_deterministic_and_writes_report(app_paths, settings) -> None:
    manager = DatasetManager(app_paths, settings)
    manifest = manager.generate_synthetic(hours=5000, seed=321)
    market = manager.load(manifest.path)
    tester = EventBacktester(settings, app_paths)
    first = tester.run(market, source="synthetic-e2e", dataset_hash=manifest.sha256)
    second = tester.run(market, source="synthetic-e2e", dataset_hash=manifest.sha256)
    assert first.run_id == second.run_id
    assert first.metrics.summary == second.metrics.summary
    assert first.equity["equity"].tolist() == second.equity["equity"].tolist()
    assert first.equity.iloc[-1]["equity"] == first.metrics.summary["final_equity"]
    assert sum(trade.net_pnl for trade in first.trades) == pytest.approx(
        first.metrics.summary["net_profit"]
    )
    report = generate_report(first)
    assert report.exists()
    html = report.read_text(encoding="utf-8")
    assert "ORIZZONTE DESK" in html
    assert "Gate live" in html
    assert not first.gate_path.exists()
    assert "gate" not in first.artifacts


def test_model_candidate_can_be_trained(app_paths, settings) -> None:
    manager = DatasetManager(app_paths, settings)
    manifest = manager.generate_synthetic(hours=7000, seed=777)
    features = SignalGenerator(settings.strategy).enrich(manager.load(manifest.path))
    result = MetaModelRegistry(app_paths).train(features, seed=777)
    assert result.model_path.exists()
    assert result.metadata_path.exists()
    assert 0 <= result.metrics["roc_auc"] <= 1
    assert len(result.model_hash) == 64
    assert result.model_path.read_bytes()
    assert result.decision_policy_path.exists()


def test_walk_forward_windows_are_anchored(app_paths, settings) -> None:
    manager = DatasetManager(app_paths, settings)
    manifest = manager.generate_synthetic(hours=24 * 365 * 3, seed=1)
    frame = manager.load(manifest.path)
    windows = walk_forward_windows(
        frame["timestamp"],
        training_months=18,
        validation_months=3,
        test_months=3,
        step_months=3,
    )
    assert windows
    assert all(item["train_start"] == windows[0]["train_start"] for item in windows)
    assert all(item["train_end"] < item["validation_end"] < item["test_end"] for item in windows)
