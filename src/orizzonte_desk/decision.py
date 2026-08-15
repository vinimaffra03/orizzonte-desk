from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ThresholdEvaluator = Callable[[float, pd.DataFrame], np.ndarray]


def _canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def purged_timestamp_split(
    frame: pd.DataFrame,
    *,
    validation_fraction: float,
    purge_hours: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = frame.copy()
    ordered["timestamp"] = pd.to_datetime(ordered["timestamp"], utc=True)
    ordered = ordered.sort_values(["timestamp", "symbol"])
    timestamps = pd.DatetimeIndex(ordered["timestamp"].drop_duplicates())
    if len(timestamps) < 10:
        raise ValueError("Poucos timestamps para seleção nested da DecisionPolicy")
    split_at = max(1, min(len(timestamps) - 1, int(len(timestamps) * (1 - validation_fraction))))
    validation_start = timestamps[split_at]
    reference = ordered[
        ordered["timestamp"] < validation_start - pd.Timedelta(hours=purge_hours)
    ].copy()
    validation = ordered[ordered["timestamp"] >= validation_start].copy()
    if reference.empty or validation.empty:
        raise ValueError("Seleção nested vazia após purge")
    return reference, validation


def block_bootstrap_lcb(
    values: np.ndarray,
    *,
    samples: int,
    quantile: float,
    seed: int,
) -> float:
    return block_bootstrap_interval(
        values,
        samples=samples,
        lower_quantile=quantile,
        seed=seed,
    )[0]


def block_bootstrap_interval(
    values: np.ndarray,
    *,
    samples: int,
    lower_quantile: float,
    seed: int,
) -> tuple[float, float, float]:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return (float("-inf"), float("-inf"), float("-inf"))
    rng = np.random.default_rng(seed)
    block = max(1, min(10, clean.size // 10))
    means = np.empty(samples, dtype=float)
    for sample in range(samples):
        selected: list[float] = []
        while len(selected) < clean.size:
            start = int(rng.integers(0, max(1, clean.size - block + 1)))
            selected.extend(clean[start : start + block])
        means[sample] = float(np.mean(selected[: clean.size]))
    return (
        float(np.quantile(means, lower_quantile)),
        float(np.quantile(means, 0.50)),
        float(np.quantile(means, 1.0 - lower_quantile)),
    )


@dataclass(frozen=True, slots=True)
class DecisionPolicy:
    policy_id: str
    model_hash: str
    objective: str
    seed: int
    calibration_method: str
    calibration_hash: str
    release_binding: dict[str, Any]
    probability_threshold: float
    selected_quantile: float
    validation_trades: int
    stressed_expectancy: float
    expectancy_lcb_p05: float
    expectancy_p50: float
    expectancy_p95: float
    reference_start: str
    reference_end: str
    validation_start: str
    validation_end: str
    purge_hours: int
    validation_fraction: float
    cost_multiplier: float
    bootstrap_samples: int
    lcb_quantile: float
    trade_enabled: bool
    no_trade_reason: str | None

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> DecisionPolicy:
        content = dict(payload)
        policy_id = str(content.pop("policy_id", ""))
        expected = f"decision-{_canonical_hash(content)[:24]}"
        if policy_id and policy_id != expected:
            raise RuntimeError("Hash da DecisionPolicy não confere")
        return cls(policy_id=expected, **content)

    def to_payload(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "model_hash": self.model_hash,
            "objective": self.objective,
            "seed": self.seed,
            "calibration_method": self.calibration_method,
            "calibration_hash": self.calibration_hash,
            "release_binding": self.release_binding,
            "probability_threshold": self.probability_threshold,
            "selected_quantile": self.selected_quantile,
            "validation_trades": self.validation_trades,
            "stressed_expectancy": self.stressed_expectancy,
            "expectancy_lcb_p05": self.expectancy_lcb_p05,
            "expectancy_p50": self.expectancy_p50,
            "expectancy_p95": self.expectancy_p95,
            "reference_start": self.reference_start,
            "reference_end": self.reference_end,
            "validation_start": self.validation_start,
            "validation_end": self.validation_end,
            "purge_hours": self.purge_hours,
            "validation_fraction": self.validation_fraction,
            "cost_multiplier": self.cost_multiplier,
            "bootstrap_samples": self.bootstrap_samples,
            "lcb_quantile": self.lcb_quantile,
            "trade_enabled": self.trade_enabled,
            "no_trade_reason": self.no_trade_reason,
        }

    def apply(self, probabilities: pd.Series | np.ndarray) -> np.ndarray:
        if not self.trade_enabled:
            return np.zeros(len(probabilities), dtype=bool)
        return np.asarray(probabilities, dtype=float) >= self.probability_threshold


@dataclass(slots=True)
class DecisionSelection:
    policy: DecisionPolicy
    diagnostics: pd.DataFrame
    funnel: dict[str, int]

    def write(self, output_dir: Path) -> dict[str, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        base = output_dir / self.policy.policy_id
        policy_path = base.with_suffix(".json")
        csv_path = output_dir / f"{self.policy.policy_id}-diagnostics.csv"
        parquet_path = output_dir / f"{self.policy.policy_id}-diagnostics.parquet"
        funnel_path = output_dir / f"{self.policy.policy_id}-funnel.json"
        policy_path.write_text(
            json.dumps(self.policy.to_payload(), indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
            newline="\n",
        )
        self.diagnostics.to_csv(csv_path, index=False, lineterminator="\n")
        self.diagnostics.to_parquet(parquet_path, compression="zstd", index=False)
        funnel_path.write_text(
            json.dumps(self.funnel, indent=2, sort_keys=True), encoding="utf-8", newline="\n"
        )
        return {
            "decision_policy": policy_path,
            "decision_diagnostics_csv": csv_path,
            "decision_diagnostics_parquet": parquet_path,
            "decision_funnel": funnel_path,
        }


class DecisionPolicySelector:
    def __init__(
        self,
        *,
        validation_fraction: float,
        purge_hours: int,
        quantiles: tuple[float, ...],
        min_validation_trades: int,
        bootstrap_samples: int,
        lcb_quantile: float,
        seed: int,
    ) -> None:
        self.validation_fraction = validation_fraction
        self.purge_hours = purge_hours
        self.quantiles = quantiles
        self.min_validation_trades = min_validation_trades
        self.bootstrap_samples = bootstrap_samples
        self.lcb_quantile = lcb_quantile
        self.seed = seed

    def select(
        self,
        frame: pd.DataFrame,
        *,
        model_hash: str,
        round_trip_cost: float,
        evaluator: ThresholdEvaluator,
        calibration_method: str = "unspecified",
        calibration_hash: str = "",
        release_binding: dict[str, Any] | None = None,
    ) -> DecisionSelection:
        required = {"timestamp", "symbol", "probability", "realized_return"}
        missing = required - set(frame)
        if missing:
            raise ValueError(f"Colunas ausentes para DecisionPolicy: {sorted(missing)}")
        clean = frame.dropna(subset=["probability", "realized_return"]).copy()
        reference, validation = purged_timestamp_split(
            clean,
            validation_fraction=self.validation_fraction,
            purge_hours=self.purge_hours,
        )
        stressed = validation.copy()
        base_cost_r = (
            stressed["cost_r"].astype(float)
            if "cost_r" in stressed
            else pd.Series(round_trip_cost, index=stressed.index, dtype=float)
        )
        stressed["stressed_return"] = stressed["realized_return"].astype(float) - 2.0 * base_cost_r
        rows: list[dict[str, Any]] = []
        for index, quantile in enumerate(self.quantiles):
            threshold = float(reference["probability"].quantile(quantile))
            selected = stressed[stressed["probability"] >= threshold]
            objective_values = np.asarray(evaluator(threshold, validation.copy()), dtype=float)
            objective_values = objective_values[np.isfinite(objective_values)]
            trades = len(objective_values)
            eligible = trades >= self.min_validation_trades
            expectancy = float(objective_values.mean()) if trades else float("-inf")
            interval = (
                block_bootstrap_interval(
                    objective_values,
                    samples=self.bootstrap_samples,
                    lower_quantile=self.lcb_quantile,
                    seed=self.seed + index,
                )
                if eligible
                else (float("-inf"), float("-inf"), float("-inf"))
            )
            lcb, bootstrap_p50, bootstrap_p95 = interval
            rows.append(
                {
                    "quantile": quantile,
                    "threshold": threshold,
                    "validation_trades": trades,
                    "validation_candidates": len(selected),
                    "stressed_expectancy": expectancy,
                    "expectancy_lcb": lcb,
                    "expectancy_p50": bootstrap_p50,
                    "expectancy_p95": bootstrap_p95,
                    "eligible": eligible,
                    "evaluation_method": "event_backtest_net_r",
                }
            )
        diagnostics = pd.DataFrame(rows)
        eligible_rows = diagnostics[
            diagnostics["eligible"]
            & np.isfinite(diagnostics["expectancy_lcb"])
            & (diagnostics["expectancy_lcb"] > 0)
        ]
        trade_enabled = not eligible_rows.empty
        choice_pool = eligible_rows if trade_enabled else diagnostics
        chosen = choice_pool.sort_values(
            ["expectancy_lcb", "quantile", "threshold"], ascending=[False, False, False]
        ).iloc[0]

        def finite_or_zero(value: Any) -> float:
            numeric = float(value)
            return numeric if np.isfinite(numeric) else 0.0

        payload = {
            "model_hash": model_hash,
            "objective": "event_backtest_net_expectancy_r_lcb_p05",
            "seed": self.seed,
            "calibration_method": calibration_method,
            "calibration_hash": calibration_hash,
            "release_binding": dict(release_binding or {}),
            "probability_threshold": float(chosen["threshold"]),
            "selected_quantile": float(chosen["quantile"]),
            "validation_trades": int(chosen["validation_trades"]),
            "stressed_expectancy": finite_or_zero(chosen["stressed_expectancy"]),
            "expectancy_lcb_p05": finite_or_zero(chosen["expectancy_lcb"]),
            "expectancy_p50": finite_or_zero(chosen["expectancy_p50"]),
            "expectancy_p95": finite_or_zero(chosen["expectancy_p95"]),
            "reference_start": pd.Timestamp(reference["timestamp"].min()).isoformat(),
            "reference_end": pd.Timestamp(reference["timestamp"].max()).isoformat(),
            "validation_start": pd.Timestamp(validation["timestamp"].min()).isoformat(),
            "validation_end": pd.Timestamp(validation["timestamp"].max()).isoformat(),
            "purge_hours": self.purge_hours,
            "validation_fraction": self.validation_fraction,
            "cost_multiplier": 2.0,
            "bootstrap_samples": self.bootstrap_samples,
            "lcb_quantile": self.lcb_quantile,
            "trade_enabled": trade_enabled,
            "no_trade_reason": (
                None if trade_enabled else "no_positive_lcb_threshold_with_minimum_internal_trades"
            ),
        }
        policy = DecisionPolicy.from_payload(payload)
        funnel = {
            "raw_candidates": len(frame),
            "finite_candidates": len(clean),
            "reference_candidates": len(reference),
            "validation_candidates": len(validation),
            "selected_validation_trades": policy.validation_trades,
            "trade_enabled": int(policy.trade_enabled),
        }
        return DecisionSelection(policy=policy, diagnostics=diagnostics, funnel=funnel)
