from __future__ import annotations

import hashlib
import json
import shutil
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
from sklearn.model_selection import TimeSeriesSplit

from orizzonte_desk.features import FEATURE_COLUMNS
from orizzonte_desk.paths import AppPaths


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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

    def train(self, frame: pd.DataFrame, *, seed: int = 42017) -> TrainingResult:
        candidates = frame.loc[frame["signal_raw"] != 0].dropna(subset=[*FEATURE_COLUMNS, "label"])
        if len(candidates) < 200:
            raise ValueError(f"Amostra insuficiente para ML: {len(candidates)} sinais; mínimo 200")
        candidates = candidates.sort_values("timestamp")
        split = int(len(candidates) * 0.8)
        train, test = candidates.iloc[:split], candidates.iloc[split:]
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
            cv=TimeSeriesSplit(n_splits=3),
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
        joblib.dump(
            {
                "model": calibrated,
                "features": FEATURE_COLUMNS,
                "trained_at": datetime.now(UTC).isoformat(),
                "metrics": metrics,
            },
            model_path,
            compress=3,
        )
        digest = file_hash(model_path)
        metadata = {
            "model_id": model_id,
            "model_path": str(model_path),
            "model_hash": digest,
            "features": list(FEATURE_COLUMNS),
            "metrics": metrics,
            "status": "candidate",
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
        promoted_model = self.paths.models / "promoted.joblib"
        shutil.copy2(model_path, promoted_model)
        pointer = {
            **metadata,
            "status": "promoted",
            "promoted_at": datetime.now(UTC).isoformat(),
            "gate_path": str(gate_path),
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

    def predict(self, frame: pd.DataFrame, bundle: dict[str, Any] | None = None) -> np.ndarray:
        loaded = bundle or self.load_promoted()
        if loaded is None:
            return np.asarray(frame["setup_score"].fillna(0.5).to_numpy(dtype=float))
        model = loaded["model"]
        features = tuple(loaded["features"])
        return np.asarray(model.predict_proba(frame.loc[:, features].fillna(0))[:, 1], dtype=float)
