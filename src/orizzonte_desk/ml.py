from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, roc_auc_score

from orizzonte_desk.features import FEATURE_COLUMNS
from orizzonte_desk.paths import AppPaths

RESEARCH_CODE_FILES = (
    "data.py",
    "features.py",
    "strategy.py",
    "ml.py",
    "backtest.py",
    "metrics.py",
    "gates.py",
)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def research_code_fingerprint() -> str:
    """Hash the exact quantitative implementation used by a model/release."""
    digest = hashlib.sha256()
    package = Path(__file__).resolve().parent
    for name in RESEARCH_CODE_FILES:
        digest.update(name.encode())
        digest.update((package / name).read_bytes())
    return digest.hexdigest()


def git_commit_fingerprint(root: Path) -> str:
    """Return the exact commit, marking tracked worktree changes as non-releasable."""
    candidates = (root, Path(__file__).resolve().parents[2])
    last_error: Exception | None = None
    for candidate in dict.fromkeys(path.resolve() for path in candidates):
        try:
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=candidate,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            dirty = subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=candidate,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            last_error = exc
            continue
        return f"{commit}:dirty" if dirty else commit
    raise RuntimeError("Não foi possível vincular a pesquisa a um commit Git") from last_error


def timestamp_holdout_split(
    frame: pd.DataFrame,
    *,
    test_fraction: float = 0.2,
    purge_hours: int = 24,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split whole timestamps, never rows, and purge both sides of the boundary."""
    ordered = frame.copy()
    ordered["timestamp"] = pd.to_datetime(ordered["timestamp"], utc=True)
    ordered = ordered.sort_values(["timestamp", "symbol"])
    unique = pd.DatetimeIndex(ordered["timestamp"].drop_duplicates().sort_values())
    if len(unique) < 10:
        raise ValueError("Poucos timestamps únicos para holdout temporal")
    boundary = unique[max(1, min(len(unique) - 1, int(len(unique) * (1 - test_fraction))))]
    embargo = pd.Timedelta(hours=purge_hours)
    train = ordered[ordered["timestamp"] < boundary - embargo].copy()
    test = ordered[ordered["timestamp"] >= boundary].copy()
    if train.empty or test.empty:
        raise ValueError("Split temporal vazio após purge/embargo")
    if set(train["timestamp"]).intersection(test["timestamp"]):
        raise AssertionError("Timestamp compartilhado entre treino e teste")
    return train, test


def _calibration_folds(
    frame: pd.DataFrame, *, splits: int = 3
) -> list[tuple[np.ndarray, np.ndarray]]:
    timestamps = pd.to_datetime(frame["timestamp"], utc=True)
    unique = pd.DatetimeIndex(timestamps.drop_duplicates().sort_values())
    segment = len(unique) // (splits + 1)
    if segment < 2:
        raise ValueError("Histórico insuficiente para calibração temporal")
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for fold in range(splits):
        validation_start = segment * (fold + 1)
        validation_end = segment * (fold + 2) if fold < splits - 1 else len(unique)
        boundary = unique[validation_start]
        train_mask = timestamps < boundary - pd.Timedelta(hours=24)
        validation_values = unique[validation_start:validation_end]
        validation_mask = timestamps.isin(validation_values)
        train_indices = np.flatnonzero(np.asarray(train_mask))
        validation_indices = np.flatnonzero(np.asarray(validation_mask))
        if train_indices.size and validation_indices.size:
            folds.append((train_indices, validation_indices))
    if len(folds) < 2:
        raise ValueError("Não foi possível criar folds temporais purgados")
    return folds


@dataclass(slots=True)
class TrainingResult:
    model_id: str
    model_path: Path
    metadata_path: Path
    metrics: dict[str, float]
    model_hash: str


class MetaModelRegistry:
    def __init__(self, paths: AppPaths) -> None:
        self.paths = paths
        self.paths.models.mkdir(parents=True, exist_ok=True)

    @property
    def promoted_pointer(self) -> Path:
        return self.paths.models / "promoted.json"

    def train(
        self,
        frame: pd.DataFrame,
        *,
        seed: int = 42017,
        dataset_role: str | None = None,
        dataset_hashes: tuple[str, ...] | None = None,
        config_fingerprint: str | None = None,
        code_hash: str | None = None,
        commit_hash: str | None = None,
    ) -> TrainingResult:
        role = dataset_role or str(frame.attrs.get("dataset_role", "development"))
        if role == "external_holdout":
            raise ValueError("Dataset external_holdout é proibido para treino ou calibração")
        if role != "development":
            raise ValueError(f"Papel de dataset desconhecido: {role}")
        candidates = frame.loc[frame["signal_raw"] != 0].dropna(subset=[*FEATURE_COLUMNS, "label"])
        if len(candidates) < 200:
            raise ValueError(f"Amostra insuficiente para ML: {len(candidates)} sinais; mínimo 200")
        candidates = candidates.sort_values(["timestamp", "symbol"])
        train, test = timestamp_holdout_split(candidates)
        estimator = LGBMClassifier(
            n_estimators=250,
            learning_rate=0.035,
            num_leaves=15,
            max_depth=5,
            min_child_samples=40,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.2,
            reg_lambda=0.8,
            class_weight="balanced",
            random_state=seed,
            verbosity=-1,
        )
        calibrated = CalibratedClassifierCV(
            estimator,
            method="sigmoid",
            cv=_calibration_folds(train),
        )
        calibrated.fit(train.loc[:, FEATURE_COLUMNS], train["label"].astype(int))
        probabilities = calibrated.predict_proba(test.loc[:, FEATURE_COLUMNS])[:, 1]
        metrics = {
            "roc_auc": float(roc_auc_score(test["label"], probabilities))
            if test["label"].nunique() > 1
            else 0.5,
            "brier_score": float(brier_score_loss(test["label"], probabilities)),
            "training_samples": float(len(train)),
            "test_samples": float(len(test)),
            "positive_rate": float(candidates["label"].mean()),
        }
        model_id = datetime.now(UTC).strftime("model-%Y%m%dT%H%M%S%fZ")
        model_path = self.paths.models / f"{model_id}.joblib"
        resolved_dataset_hashes = dataset_hashes or tuple(
            value for value in (str(frame.attrs.get("dataset_hash", "")),) if value
        )
        resolved_config = config_fingerprint or str(frame.attrs.get("config_fingerprint", ""))
        release_binding = {
            "dataset_hashes": sorted(resolved_dataset_hashes),
            "config_fingerprint": resolved_config,
            "code_hash": code_hash or research_code_fingerprint(),
            "commit_hash": commit_hash or git_commit_fingerprint(self.paths.root),
        }
        bundle = {
            "model": calibrated,
            "features": FEATURE_COLUMNS,
            "trained_at": datetime.now(UTC).isoformat(),
            "metrics": metrics,
            "release_binding": release_binding,
        }
        joblib.dump(bundle, model_path, compress=3)
        digest = file_hash(model_path)
        metadata = {
            "model_id": model_id,
            "model_path": str(model_path),
            "model_hash": digest,
            "features": list(FEATURE_COLUMNS),
            "metrics": metrics,
            "status": "candidate",
            "dataset_role": role,
            "release_binding": release_binding,
        }
        metadata_path = self.paths.models / f"{model_id}.json"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return TrainingResult(model_id, model_path, metadata_path, metrics, digest)

    def promote(self, model_id: str, gate_path: Path) -> dict[str, Any]:
        metadata_path = self.paths.models / f"{model_id}.json"
        model_path = self.paths.models / f"{model_id}.joblib"
        if not metadata_path.exists() or not model_path.exists():
            raise FileNotFoundError(f"Modelo candidato não encontrado: {model_id}")
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        if not gate.get("passed", False):
            raise RuntimeError("O modelo não pode ser promovido: gate quantitativo reprovado")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        actual_hash = file_hash(model_path)
        if actual_hash != metadata.get("model_hash"):
            raise RuntimeError("Hash do modelo candidato não confere")
        gate_binding = gate.get("release_binding")
        expected_binding = {
            **metadata.get("release_binding", {}),
            "model_hash": actual_hash,
        }
        if not isinstance(gate_binding, dict):
            raise RuntimeError("Gate não contém release_binding auditável")
        gate_model_hash = gate_binding.get("model_hash")
        top_level_model_matches = gate.get("model_hash", actual_hash) == actual_hash
        invariant_keys = ("config_fingerprint", "code_hash", "commit_hash")
        invariants_match = all(
            gate_binding.get(key) == expected_binding.get(key) for key in invariant_keys
        )
        training_datasets = set(expected_binding.get("dataset_hashes", []))
        evaluated_datasets = set(gate_binding.get("dataset_hashes", []))
        datasets_bound = bool(evaluated_datasets) and training_datasets <= evaluated_datasets
        if (
            gate_model_hash != actual_hash
            or not top_level_model_matches
            or not invariants_match
            or not datasets_bound
            or len(evaluated_datasets) < 2
            or gate_binding.get("evaluation_scope") != "combined_release"
            or not gate_binding.get("protocol_hashes")
        ):
            raise RuntimeError(
                "Gate não corresponde exatamente ao modelo, datasets, configuração e código"
            )
        promoted_model = self.paths.models / "promoted.joblib"
        shutil.copy2(model_path, promoted_model)
        pointer = {
            **metadata,
            "status": "promoted",
            "promoted_at": datetime.now(UTC).isoformat(),
            "gate_path": str(gate_path),
            "gate_hash": file_hash(gate_path),
            "promoted_hash": file_hash(promoted_model),
        }
        self.promoted_pointer.write_text(json.dumps(pointer, indent=2), encoding="utf-8")
        return pointer

    def load_promoted(self) -> dict[str, Any] | None:
        model_path = self.paths.models / "promoted.joblib"
        if not self.promoted_pointer.exists() or not model_path.exists():
            return None
        pointer = json.loads(self.promoted_pointer.read_text(encoding="utf-8"))
        if file_hash(model_path) != pointer["promoted_hash"]:
            raise RuntimeError("Hash do modelo promovido não confere")
        return cast(dict[str, Any], joblib.load(model_path))

    def load_candidate(self, model_id: str) -> dict[str, Any]:
        model_path = self.paths.models / f"{model_id}.joblib"
        metadata_path = self.paths.models / f"{model_id}.json"
        if not model_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(f"Modelo candidato não encontrado: {model_id}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if file_hash(model_path) != metadata.get("model_hash"):
            raise RuntimeError("Hash do modelo candidato não confere")
        return cast(dict[str, Any], joblib.load(model_path))

    def predict(self, frame: pd.DataFrame, bundle: dict[str, Any] | None = None) -> np.ndarray:
        loaded = bundle or self.load_promoted()
        if loaded is None:
            return np.asarray(frame["setup_score"].fillna(0.5).to_numpy(dtype=float))
        model = loaded["model"]
        features = tuple(loaded["features"])
        return np.asarray(model.predict_proba(frame.loc[:, features].fillna(0))[:, 1], dtype=float)
