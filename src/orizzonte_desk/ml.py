from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
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

from orizzonte_desk.config import (
    AppConfig,
    BacktestConfig,
    ExecutionConfig,
    ResearchConfig,
    RiskConfig,
    Settings,
    StrategyConfig,
    UniverseConfig,
)
from orizzonte_desk.constants import SYMBOLS
from orizzonte_desk.decision import DecisionPolicy, DecisionPolicySelector, ThresholdEvaluator
from orizzonte_desk.features import FEATURE_COLUMNS
from orizzonte_desk.paths import AppPaths

RESEARCH_CODE_FILES = (
    "data.py",
    "decision.py",
    "features.py",
    "strategy.py",
    "ml.py",
    "backtest.py",
    "metrics.py",
    "gates.py",
    "regimes.py",
    "risk.py",
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
    decision_policy_id: str
    decision_policy_path: Path


class MetaModelRegistry:
    def __init__(
        self,
        paths: AppPaths,
        research: ResearchConfig | None = None,
        execution: ExecutionConfig | None = None,
        *,
        settings: Settings | None = None,
    ) -> None:
        self.paths = paths
        self.research = research or ResearchConfig()
        self.execution = execution or ExecutionConfig()
        if settings is not None:
            self.settings = settings
        elif self.paths.config.exists():
            self.settings = Settings.load(self.paths.config)
        else:
            self.settings = Settings(
                app=AppConfig(),
                universe=UniverseConfig(),
                risk=RiskConfig(),
                strategy=StrategyConfig(),
                execution=self.execution,
                backtest=BacktestConfig(),
                research=self.research,
            )
        self.paths.models.mkdir(parents=True, exist_ok=True)

    def _event_threshold_evaluator(
        self,
        full_frame: pd.DataFrame,
        *,
        seed: int,
        dataset_hash: str,
    ) -> ThresholdEvaluator:
        """Return an event-driven net-R objective for nested threshold selection."""
        from orizzonte_desk.backtest import EventBacktester

        initial_risk = self.settings.backtest.initial_capital * self.settings.risk.risk_per_trade
        ordered = full_frame.copy()
        ordered["timestamp"] = pd.to_datetime(ordered["timestamp"], utc=True)
        closes = ordered.pivot(index="timestamp", columns="symbol", values="close").sort_index()
        returns = closes.apply(np.log).diff()
        correlations: dict[pd.Timestamp, dict[tuple[str, str], float]] = {}
        for left_index, left in enumerate(SYMBOLS):
            for right in SYMBOLS[left_index + 1 :]:
                if left not in returns or right not in returns:
                    continue
                causal = returns[left].rolling(719, min_periods=167).corr(returns[right]).shift(1)
                for timestamp, value in causal.dropna().items():
                    if np.isfinite(value):
                        correlations.setdefault(pd.Timestamp(cast(Any, timestamp)), {})[
                            (left, right)
                        ] = float(value)

        def evaluate(threshold: float, validation: pd.DataFrame) -> np.ndarray:
            validation = validation.loc[:, ["timestamp", "symbol", "probability"]].copy()
            validation["timestamp"] = pd.to_datetime(validation["timestamp"], utc=True)
            validation_start = pd.Timestamp(validation["timestamp"].min())
            validation_end = pd.Timestamp(validation["timestamp"].max())
            warmup_start = validation_start - pd.Timedelta(days=30)
            enriched = full_frame.copy()
            enriched["timestamp"] = pd.to_datetime(enriched["timestamp"], utc=True)
            enriched = enriched[
                (enriched["timestamp"] >= warmup_start) & (enriched["timestamp"] <= validation_end)
            ].copy()
            enriched.drop(columns=["probability"], inplace=True, errors="ignore")
            enriched = enriched.merge(
                validation,
                on=["timestamp", "symbol"],
                how="left",
                validate="one_to_one",
            )
            enriched["ml_probability"] = enriched["probability"].fillna(0.0)
            enriched["decision_threshold"] = threshold
            enriched["decision_policy_id"] = "nested-selection"
            enriched["decision_accepted"] = enriched["probability"].ge(threshold).fillna(False)
            enriched["signal"] = enriched["signal_raw"].where(enriched["decision_accepted"], 0)
            enriched.drop(columns=["probability"], inplace=True)
            enriched.attrs.update(full_frame.attrs)
            result = EventBacktester(self.settings, self.paths).run(
                enriched,
                source=f"decision-inner-q-{threshold:.12f}",
                dataset_hash=dataset_hash,
                cost_multiplier=2.0,
                seed=seed,
                persist=False,
                enriched_override=enriched,
                run_stress_suite=False,
                metrics_monte_carlo_samples=100,
                correlations_override=correlations,
                objective_trades_only=True,
            )
            return np.asarray(
                [trade.net_pnl / initial_risk for trade in result.trades], dtype=float
            )

        return evaluate

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
        candidates = frame.loc[frame["signal_raw"] != 0].dropna(
            subset=[*FEATURE_COLUMNS, "label", "realized_r_24h"]
        )
        minimum_candidates = max(
            200,
            int(
                np.ceil(
                    self.research.threshold_min_validation_trades
                    / self.research.inner_validation_fraction
                    / (1 - self.research.inner_validation_fraction)
                    / min(self.research.threshold_quantiles)
                )
            ),
        )
        if len(candidates) < minimum_candidates:
            raise ValueError(
                f"Amostra insuficiente para ML/DecisionPolicy: {len(candidates)} sinais; "
                f"mínimo {minimum_candidates}"
            )
        candidates = candidates.sort_values(["timestamp", "symbol"])
        development, test = timestamp_holdout_split(
            candidates,
            test_fraction=self.research.inner_validation_fraction,
            purge_hours=self.research.purge_hours,
        )
        train, _ = timestamp_holdout_split(
            development,
            test_fraction=self.research.inner_validation_fraction,
            purge_hours=self.research.purge_hours,
        )
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
            "development_samples": float(len(development)),
            "test_samples": float(len(test)),
            "positive_rate": float(candidates["label"].mean()),
        }
        calibration_method = "sigmoid_temporal_purged_3fold"
        calibration_hash = hashlib.sha256(
            json.dumps(
                {
                    "method": calibration_method,
                    "roc_auc": metrics["roc_auc"],
                    "brier_score": metrics["brier_score"],
                    "training_start": pd.Timestamp(train["timestamp"].min()).isoformat(),
                    "training_end": pd.Timestamp(train["timestamp"].max()).isoformat(),
                    "training_samples": len(train),
                    "features": list(FEATURE_COLUMNS),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
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
            "metrics": metrics,
            "release_binding": release_binding,
        }
        with tempfile.NamedTemporaryFile(
            dir=self.paths.models,
            prefix=".model-",
            suffix=".joblib.tmp",
            delete=False,
        ) as handle:
            provisional_model = Path(handle.name)
        try:
            joblib.dump(bundle, provisional_model, compress=3)
            digest = file_hash(provisional_model)
            model_id = f"model-{digest[:24]}"
            model_path = self.paths.models / f"{model_id}.joblib"
            if model_path.exists():
                if file_hash(model_path) != digest:
                    raise RuntimeError("Colisão de modelo content-addressed detectada")
            else:
                provisional_model.replace(model_path)
        finally:
            provisional_model.unlink(missing_ok=True)
        development_probabilities = calibrated.predict_proba(development.loc[:, FEATURE_COLUMNS])[
            :, 1
        ]
        decision_frame = development.loc[
            :, ["timestamp", "symbol", "realized_r_24h", "stop_distance", "close"]
        ].rename(columns={"realized_r_24h": "realized_return"})
        decision_frame["probability"] = development_probabilities
        selector = DecisionPolicySelector(
            validation_fraction=self.research.inner_validation_fraction,
            purge_hours=self.research.purge_hours,
            quantiles=self.research.threshold_quantiles,
            min_validation_trades=self.research.threshold_min_validation_trades,
            bootstrap_samples=self.research.threshold_bootstrap_samples,
            lcb_quantile=self.research.threshold_lcb_quantile,
            seed=seed,
        )
        round_trip_cost = 2 * (
            self.execution.taker_fee + self.execution.slippage_bps_sol_xrp / 10_000
        )
        decision_frame["cost_r"] = round_trip_cost / (
            decision_frame["stop_distance"] / decision_frame["close"]
        )
        selection = selector.select(
            decision_frame,
            model_hash=digest,
            round_trip_cost=round_trip_cost,
            evaluator=self._event_threshold_evaluator(
                frame,
                seed=seed,
                dataset_hash=(resolved_dataset_hashes[0] if resolved_dataset_hashes else ""),
            ),
            calibration_method=calibration_method,
            calibration_hash=calibration_hash,
            release_binding=release_binding,
        )
        decision_dir = self.paths.models / "decision-policies"
        decision_artifacts = selection.write(decision_dir)
        decision_policy_hash = file_hash(decision_artifacts["decision_policy"])
        release_binding.update(
            {
                "decision_policy_id": selection.policy.policy_id,
                "decision_policy_hash": decision_policy_hash,
            }
        )
        final_decisions = test.loc[
            :, ["timestamp", "symbol", "realized_return_24h", "realized_r_24h"]
        ].copy()
        final_decisions["probability"] = probabilities
        final_decisions["accepted"] = selection.policy.apply(probabilities)
        final_decisions["stage"] = "outer_holdout"
        final_csv = decision_dir / f"{selection.policy.policy_id}-outer-holdout.csv"
        final_parquet = decision_dir / f"{selection.policy.policy_id}-outer-holdout.parquet"
        final_decisions.to_csv(final_csv, index=False, lineterminator="\n")
        final_decisions.to_parquet(final_parquet, compression="zstd", index=False)
        complete_funnel = {
            **selection.funnel,
            "outer_candidates": len(candidates),
            "outer_development": len(development),
            "outer_holdout": len(test),
            "outer_holdout_accepted": int(final_decisions["accepted"].sum()),
        }
        funnel_path = decision_dir / f"{selection.policy.policy_id}-complete-funnel.json"
        funnel_path.write_text(
            json.dumps(complete_funnel, indent=2, sort_keys=True),
            encoding="utf-8",
            newline="\n",
        )
        metrics.update(
            {
                "decision_threshold": selection.policy.probability_threshold,
                "decision_quantile": selection.policy.selected_quantile,
                "decision_validation_trades": float(selection.policy.validation_trades),
                "decision_lcb_p05": selection.policy.expectancy_lcb_p05,
                "decision_outer_trades": float(final_decisions["accepted"].sum()),
            }
        )
        metadata = {
            "model_id": model_id,
            "model_path": str(model_path),
            "model_hash": digest,
            "features": list(FEATURE_COLUMNS),
            "metrics": metrics,
            "status": "candidate",
            "dataset_role": role,
            "release_binding": release_binding,
            "decision_policy_id": selection.policy.policy_id,
            "decision_policy_path": str(decision_artifacts["decision_policy"]),
            "decision_policy_hash": decision_policy_hash,
            "decision_artifacts": {
                **{key: str(value) for key, value in decision_artifacts.items()},
                "outer_holdout_csv": str(final_csv),
                "outer_holdout_parquet": str(final_parquet),
                "complete_funnel": str(funnel_path),
            },
        }
        metadata_path = self.paths.models / f"{model_id}.json"
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8", newline="\n"
        )
        return TrainingResult(
            model_id,
            model_path,
            metadata_path,
            metrics,
            digest,
            selection.policy.policy_id,
            decision_artifacts["decision_policy"],
        )

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
        policy = self.load_decision_policy(model_id)
        if (
            policy is None
            or not policy.trade_enabled
            or policy.expectancy_lcb_p05 <= 0
            or policy.validation_trades < self.research.threshold_min_validation_trades
        ):
            raise RuntimeError("Modelo sem DecisionPolicy elegível não pode ser promovido")
        gate_binding = gate.get("release_binding")
        expected_binding = {
            **metadata.get("release_binding", {}),
            "model_hash": actual_hash,
        }
        if not isinstance(gate_binding, dict):
            raise RuntimeError("Gate não contém release_binding auditável")
        gate_model_hash = gate_binding.get("model_hash")
        top_level_model_matches = gate.get("model_hash", actual_hash) == actual_hash
        invariant_keys = (
            "config_fingerprint",
            "code_hash",
            "commit_hash",
            "decision_policy_id",
            "decision_policy_hash",
        )
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
        self.promoted_pointer.write_text(
            json.dumps(pointer, indent=2, sort_keys=True), encoding="utf-8", newline="\n"
        )
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

    def load_decision_policy(self, model_id: str | None = None) -> DecisionPolicy | None:
        metadata_path = (
            self.paths.models / f"{model_id}.json" if model_id else self.promoted_pointer
        )
        if not metadata_path.exists():
            return None
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        policy_path_value = metadata.get("decision_policy_path")
        if not policy_path_value:
            return None
        policy_path = Path(str(policy_path_value))
        if not policy_path.exists():
            raise RuntimeError("DecisionPolicy vinculada ao modelo não foi encontrada")
        expected_hash = metadata.get("decision_policy_hash")
        if expected_hash and file_hash(policy_path) != expected_hash:
            raise RuntimeError("Hash da DecisionPolicy não confere")
        return DecisionPolicy.from_payload(json.loads(policy_path.read_text(encoding="utf-8")))

    def predict(self, frame: pd.DataFrame, bundle: dict[str, Any] | None = None) -> np.ndarray:
        loaded = bundle or self.load_promoted()
        if loaded is None:
            return np.asarray(frame["setup_score"].fillna(0.5).to_numpy(dtype=float))
        model = loaded["model"]
        features = tuple(loaded["features"])
        return np.asarray(model.predict_proba(frame.loc[:, features].fillna(0))[:, 1], dtype=float)
