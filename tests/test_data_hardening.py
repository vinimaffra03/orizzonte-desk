from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd
import pytest

from orizzonte_desk.data import DatasetManager


class _FundingResponse:
    def __init__(self, payload: list[dict[str, object]]) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> list[dict[str, object]]:
        return self.payload


class _FundingClient:
    def post(self, _url: str, *, json: dict[str, Any]) -> _FundingResponse:
        return _FundingResponse(
            [
                {
                    "coin": json["coin"],
                    "fundingRate": "0.0001",
                    "time": int(json["startTime"]) + 17,
                }
            ]
        )


def test_manifest_is_content_addressed_auditable_and_role_bound(app_paths, settings) -> None:
    manager = DatasetManager(app_paths, settings)
    first = manager.generate_synthetic(hours=500, seed=91)
    second = manager.generate_synthetic(hours=500, seed=91)

    assert first.dataset_id == second.dataset_id
    assert first.role == "development"
    assert first.immutable is True
    assert len(first.config_fingerprint) == 64
    assert first.dataset_id.endswith(first.sha256[:12])
    assert manager.audit_manifest(first)["complete"] is True


def test_hourly_gap_is_rejected_with_audit_details(app_paths, settings) -> None:
    manager = DatasetManager(app_paths, settings)
    manifest = manager.generate_synthetic(hours=500, seed=92)
    frame = manager.load(manifest.path)
    timestamp = frame.loc[frame["symbol"] == "BTC", "timestamp"].iloc[100]
    incomplete = frame[~((frame["symbol"] == "BTC") & (frame["timestamp"] == timestamp))]

    audit = manager.audit_frame(incomplete)
    assert audit["missing_rows"] == 1
    assert audit["gap_examples"] == [f"BTC:{pd.Timestamp(timestamp).isoformat()}"]
    with pytest.raises(ValueError, match="incompleto"):
        manager.validate_frame(incomplete)


def test_load_detects_mutated_immutable_dataset(app_paths, settings) -> None:
    manager = DatasetManager(app_paths, settings)
    manifest = manager.generate_synthetic(hours=500, seed=93)
    path = app_paths.processed_data / f"{manifest.dataset_id}.parquet"
    path.write_bytes(path.read_bytes() + b"tampered")

    with pytest.raises(RuntimeError, match="Checksum"):
        manager.load(manifest.dataset_id)


def test_hyperliquid_funding_is_aligned_to_candle_hour(app_paths, settings) -> None:
    manager = DatasetManager(app_paths, settings)
    manager.client = _FundingClient()  # type: ignore[assignment]
    start = datetime(2026, 1, 1, tzinfo=UTC)

    funding = manager._hyperliquid_funding(
        "https://example.invalid",
        start,
        start + timedelta(hours=1),
    )

    assert len(funding) == 4
    assert funding["timestamp"].eq(pd.Timestamp(start)).all()
    assert funding["funding_rate"].eq(0.0001).all()
