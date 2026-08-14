from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import pytest

from orizzonte_desk.gates import load_combined_gate
from orizzonte_desk.metrics import monte_carlo_summary
from orizzonte_desk.ml import MetaModelRegistry, file_hash, timestamp_holdout_split


def test_timestamp_split_keeps_cross_asset_rows_together_and_embargoed() -> None:
    timestamps = pd.date_range("2024-01-01", periods=100, freq="h", tz="UTC")
    frame = pd.DataFrame(
        [(stamp, symbol) for stamp in timestamps for symbol in ("BTC", "ETH", "SOL", "XRP")],
        columns=["timestamp", "symbol"],
    )
    train, test = timestamp_holdout_split(frame, purge_hours=24)

    assert set(train["timestamp"]).isdisjoint(test["timestamp"])
    assert train["timestamp"].max() < test["timestamp"].min() - pd.Timedelta(hours=24)
    assert train.groupby("timestamp")["symbol"].nunique().eq(4).all()
    assert test.groupby("timestamp")["symbol"].nunique().eq(4).all()


def test_external_holdout_is_never_trainable(app_paths) -> None:
    frame = pd.DataFrame({"signal_raw": [1]})
    frame.attrs["dataset_role"] = "external_holdout"
    with pytest.raises(ValueError, match="external_holdout"):
        MetaModelRegistry(app_paths).train(frame)


def test_promotion_requires_exact_release_binding(app_paths, tmp_path) -> None:
    registry = MetaModelRegistry(app_paths)
    model_id = "model-release-test"
    model_path = app_paths.models / f"{model_id}.joblib"
    joblib.dump({"model": "fixture", "features": (), "release_binding": {}}, model_path)
    model_hash = file_hash(model_path)
    binding = {
        "dataset_hashes": ["a" * 64],
        "config_fingerprint": "b" * 64,
        "code_hash": "c" * 64,
        "commit_hash": "deadbeef",
    }
    (app_paths.models / f"{model_id}.json").write_text(
        json.dumps(
            {
                "model_id": model_id,
                "model_hash": model_hash,
                "status": "candidate",
                "release_binding": binding,
            }
        ),
        encoding="utf-8",
    )
    bad_gate = tmp_path / "bad-gate.json"
    bad_gate.write_text(
        json.dumps(
            {
                "passed": True,
                "model_hash": model_hash,
                "release_binding": {**binding, "model_hash": "wrong"},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="não corresponde"):
        registry.promote(model_id, bad_gate)

    good_gate = tmp_path / "good-gate.json"
    good_gate.write_text(
        json.dumps(
            {
                "passed": True,
                "model_hash": model_hash,
                "release_binding": {
                    **binding,
                    "dataset_hashes": ["a" * 64, "e" * 64],
                    "model_hash": model_hash,
                    "evaluation_scope": "combined_release",
                    "protocol_hashes": ["f" * 64],
                },
            }
        ),
        encoding="utf-8",
    )
    pointer = registry.promote(model_id, good_gate)
    assert pointer["promoted_hash"] == model_hash
    assert len(pointer["gate_hash"]) == 64


def test_combined_gate_rejects_mixed_model_bindings(tmp_path) -> None:
    paths = []
    for index, model_hash in enumerate(("a" * 64, "b" * 64)):
        path = tmp_path / f"gate-{index}.json"
        path.write_text(
            json.dumps(
                {
                    "passed": True,
                    "release_binding": {
                        "model_hash": model_hash,
                        "config_fingerprint": "c" * 64,
                        "code_hash": "d" * 64,
                        "commit_hash": "deadbeef",
                        "dataset_hashes": [str(index) * 64],
                    },
                }
            ),
            encoding="utf-8",
        )
        paths.append(path)
    combined = load_combined_gate(paths)
    assert combined["passed"] is False
    assert combined["bindings_match"] is False


def test_combined_gate_binds_walk_forward_protocol_to_candidate(tmp_path) -> None:
    common = {
        "config_fingerprint": "c" * 64,
        "code_hash": "d" * 64,
        "commit_hash": "deadbeef",
    }
    protocol = tmp_path / "protocol.json"
    protocol.write_text(
        json.dumps(
            {
                "passed": True,
                "release_binding": {
                    **common,
                    "model_hash": None,
                    "evaluation_scope": "training_protocol",
                    "protocol_hash": "p" * 64,
                    "dataset_hashes": ["development"],
                },
            }
        ),
        encoding="utf-8",
    )
    candidate = tmp_path / "candidate.json"
    candidate.write_text(
        json.dumps(
            {
                "passed": True,
                "model_hash": "a" * 64,
                "release_binding": {
                    **common,
                    "model_hash": "a" * 64,
                    "evaluation_scope": "candidate",
                    "dataset_hashes": ["external-holdout"],
                },
            }
        ),
        encoding="utf-8",
    )

    combined = load_combined_gate([protocol, candidate])

    assert combined["passed"] is True
    assert combined["model_hash"] == "a" * 64
    assert combined["release_binding"]["protocol_hashes"] == ["p" * 64]
    assert combined["release_binding"]["dataset_hashes"] == [
        "development",
        "external-holdout",
    ]


def test_block_bootstrap_monte_carlo_is_deterministic_and_complete() -> None:
    returns = np.asarray([0.01, -0.005, 0.003, -0.002] * 20)
    first = monte_carlo_summary(returns, samples=100, seed=7)
    second = monte_carlo_summary(returns, samples=100, seed=7)
    assert first == second
    assert first["mc_final_return_p05"] <= first["mc_final_return_p50"]
    assert first["mc_final_return_p50"] <= first["mc_final_return_p95"]
