from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import pytest

from orizzonte_desk.decision import DecisionPolicy
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
    policy = DecisionPolicy.from_payload(
        {
            "model_hash": model_hash,
            "objective": "event_backtest_net_expectancy_r_lcb_p05",
            "seed": 7,
            "calibration_method": "sigmoid_temporal_purged_3fold",
            "calibration_hash": "9" * 64,
            "release_binding": {
                "dataset_hashes": ["a" * 64],
                "config_fingerprint": "b" * 64,
                "code_hash": "c" * 64,
                "commit_hash": "deadbeef",
            },
            "probability_threshold": 0.7,
            "selected_quantile": 0.8,
            "validation_trades": 30,
            "stressed_expectancy": 0.01,
            "expectancy_lcb_p05": 0.005,
            "expectancy_p50": 0.01,
            "expectancy_p95": 0.02,
            "reference_start": "2024-01-01T00:00:00+00:00",
            "reference_end": "2024-06-01T00:00:00+00:00",
            "validation_start": "2024-06-02T00:00:00+00:00",
            "validation_end": "2024-08-01T00:00:00+00:00",
            "purge_hours": 24,
            "validation_fraction": 0.2,
            "cost_multiplier": 2.0,
            "bootstrap_samples": 1000,
            "lcb_quantile": 0.05,
            "trade_enabled": True,
            "no_trade_reason": None,
        }
    )
    policy_path = app_paths.models / f"{policy.policy_id}.json"
    policy_path.write_text(json.dumps(policy.to_payload()), encoding="utf-8")
    policy_hash = file_hash(policy_path)
    binding = {
        "dataset_hashes": ["a" * 64],
        "config_fingerprint": "b" * 64,
        "code_hash": "c" * 64,
        "commit_hash": "deadbeef",
        "decision_policy_id": policy.policy_id,
        "decision_policy_hash": policy_hash,
    }
    (app_paths.models / f"{model_id}.json").write_text(
        json.dumps(
            {
                "model_id": model_id,
                "model_hash": model_hash,
                "status": "candidate",
                "release_binding": binding,
                "decision_policy_path": str(policy_path),
                "decision_policy_hash": policy_hash,
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
    decision_binding = {
        "decision_policy_id": "decision-final",
        "decision_policy_hash": "e" * 64,
    }
    protocol = tmp_path / "protocol.json"
    protocol.write_text(
        json.dumps(
            {
                "passed": True,
                "evaluated_at": "2026-07-31T00:00:00+00:00",
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
                "evaluated_at": "2026-08-14T00:00:00+00:00",
                "model_hash": "a" * 64,
                "release_binding": {
                    **common,
                    **decision_binding,
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
    assert combined["release_binding"]["decision_policy_id"] == "decision-final"
    assert combined["release_binding"]["decision_policy_hash"] == "e" * 64
    assert combined["release_binding"]["protocol_hashes"] == ["p" * 64]
    assert combined["release_binding"]["dataset_hashes"] == [
        "development",
        "external-holdout",
    ]
    assert combined["evaluated_at"] == "2026-08-14T00:00:00+00:00"
    assert load_combined_gate([candidate, protocol]) == combined


def test_combined_gate_rejects_missing_or_divergent_candidate_policy(tmp_path) -> None:
    common = {
        "config_fingerprint": "c" * 64,
        "code_hash": "d" * 64,
        "commit_hash": "deadbeef",
    }
    protocol = tmp_path / "protocol-policy.json"
    protocol.write_text(
        json.dumps(
            {
                "passed": True,
                "release_binding": {
                    **common,
                    "evaluation_scope": "training_protocol",
                    "protocol_hash": "p" * 64,
                    "model_hash": None,
                },
            }
        ),
        encoding="utf-8",
    )
    candidates = []
    for index, policy_hash in enumerate(("1" * 64, "2" * 64)):
        candidate = tmp_path / f"candidate-policy-{index}.json"
        candidate.write_text(
            json.dumps(
                {
                    "passed": True,
                    "model_hash": "a" * 64,
                    "release_binding": {
                        **common,
                        "evaluation_scope": "candidate",
                        "model_hash": "a" * 64,
                        "decision_policy_id": "decision-final",
                        "decision_policy_hash": policy_hash,
                    },
                }
            ),
            encoding="utf-8",
        )
        candidates.append(candidate)

    divergent = load_combined_gate([protocol, *candidates])
    missing = json.loads(candidates[0].read_text(encoding="utf-8"))
    missing["release_binding"].pop("decision_policy_hash")
    candidates[0].write_text(json.dumps(missing), encoding="utf-8")
    incomplete = load_combined_gate([protocol, candidates[0]])

    assert divergent["passed"] is False
    assert divergent["bindings_match"] is False
    assert incomplete["passed"] is False
    assert incomplete["bindings_match"] is False


def test_block_bootstrap_monte_carlo_is_deterministic_and_complete() -> None:
    returns = np.asarray([0.01, -0.005, 0.003, -0.002] * 20)
    first = monte_carlo_summary(returns, samples=100, seed=7)
    second = monte_carlo_summary(returns, samples=100, seed=7)
    assert first == second
    assert first["mc_final_return_p05"] <= first["mc_final_return_p50"]
    assert first["mc_final_return_p50"] <= first["mc_final_return_p95"]
