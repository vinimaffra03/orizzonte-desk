from __future__ import annotations

import json
from pathlib import Path

import pytest

from orizzonte_desk.config import Settings
from orizzonte_desk.data import stable_fingerprint
from orizzonte_desk.ml import research_code_fingerprint
from orizzonte_desk.release import ReleaseError, ReleaseManager, sha256_path


def _release_ready(app_paths, monkeypatch) -> ReleaseManager:
    app_paths.ensure()
    app_paths.config.parent.mkdir(parents=True, exist_ok=True)
    app_paths.config.write_bytes(
        (Path(__file__).parents[1] / "config" / "settings.toml").read_bytes()
    )
    model = app_paths.models / "promoted.joblib"
    model.write_bytes(b"deterministic-model")
    model_hash = sha256_path(model)
    binding = {
        "dataset_hashes": ["development-dataset"],
        "config_fingerprint": stable_fingerprint(
            Settings.load(app_paths.config).model_dump(mode="json")
        ),
        "code_hash": research_code_fingerprint(),
        "commit_hash": "a" * 40,
    }
    (app_paths.models / "promoted.json").write_text(
        json.dumps(
            {
                "model_id": "model-test",
                "promoted_hash": model_hash,
                "release_binding": binding,
            }
        ),
        encoding="utf-8",
    )
    (app_paths.reports / "live-approval.json").write_text(
        json.dumps(
            {
                "passed": True,
                "model_hash": model_hash,
                "gates": [{"model_hash": model_hash}],
                "release_binding": {
                    **binding,
                    "dataset_hashes": ["development-dataset", "external-holdout"],
                    "model_hash": model_hash,
                    "evaluation_scope": "combined_release",
                    "protocol_hashes": ["protocol-hash"],
                },
            }
        ),
        encoding="utf-8",
    )
    manager = ReleaseManager(app_paths)
    monkeypatch.setattr(manager, "_git_identity", lambda: ("a" * 40, False))
    return manager


def test_release_binds_and_verifies_artifacts(app_paths, monkeypatch) -> None:
    manager = _release_ready(app_paths, monkeypatch)
    built = manager.build()
    assert built.approval_passed and not built.approved
    verified = manager.verify(built.release_id)
    assert verified.verified_at is not None

    with pytest.raises(ReleaseError, match="Confirmação inválida"):
        manager.approve(built.release_id, "APPROVE")
    approved = manager.approve(built.release_id, f"APPROVE RELEASE {built.release_id}")
    assert approved.approved
    assert manager.verify(built.release_id).approved
    assert manager.approved() is not None


def test_release_rejects_unbound_model_and_tampering(app_paths, monkeypatch) -> None:
    manager = _release_ready(app_paths, monkeypatch)
    approval_path = app_paths.reports / "live-approval.json"
    approval_path.write_text(
        json.dumps({"passed": True, "gates": [{"model_hash": "b" * 64}]}),
        encoding="utf-8",
    )
    with pytest.raises(ReleaseError, match="não está vinculado"):
        manager.build()

    manager = _release_ready(app_paths, monkeypatch)
    built = manager.build()
    app_paths.config.write_text("tampered", encoding="utf-8")
    with pytest.raises(ReleaseError, match="config"):
        manager.verify(built.release_id)
