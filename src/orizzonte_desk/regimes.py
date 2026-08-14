from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

from orizzonte_desk.decision import block_bootstrap_lcb, purged_timestamp_split

REGIME_ARMS = ("hybrid", "breakout", "pullback", "flat")
REGIME_STATES = ("bull", "bear")


def _study_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_regime_arms(features: pd.DataFrame) -> pd.DataFrame:
    """Build the four established setup arms without changing labels, stops or sizing."""
    required = {
        "timestamp",
        "symbol",
        "signal_raw",
        "daily_trend",
        "weekly_trend",
        "momentum_24h",
        "volume_zscore",
        "breakout_long",
        "breakout_short",
        "pullback_long",
        "pullback_short",
        "forward_return_24h",
        "forward_r_24h",
        "risk_fraction",
    }
    missing = required - set(features)
    if missing:
        raise ValueError(f"Colunas ausentes para estudo de regimes: {sorted(missing)}")
    result = features.copy()
    aligned_long = (result["daily_trend"] > 0) & (result["weekly_trend"] > 0)
    aligned_short = (result["daily_trend"] < 0) & (result["weekly_trend"] < 0)
    liquid = result["volume_zscore"].fillna(-2) > -1
    long_timing = (result["momentum_24h"] > 0) & liquid
    short_timing = (result["momentum_24h"] < 0) & liquid
    result["regime_state"] = np.select(
        [aligned_long, aligned_short], ["bull", "bear"], default="mixed"
    )
    result["arm_hybrid"] = result["signal_raw"].astype(int)
    result["arm_breakout"] = np.select(
        [
            aligned_long & result["breakout_long"] & long_timing,
            aligned_short & result["breakout_short"] & short_timing,
        ],
        [1, -1],
        default=0,
    )
    result["arm_pullback"] = np.select(
        [
            aligned_long & result["pullback_long"] & long_timing,
            aligned_short & result["pullback_short"] & short_timing,
        ],
        [1, -1],
        default=0,
    )
    result["arm_flat"] = 0
    return result


@dataclass(slots=True)
class RegimeStudyResult:
    study_id: str
    decisions: pd.DataFrame
    challenger: pd.DataFrame
    summary: dict[str, Any]
    transitions: pd.DataFrame
    matrix: pd.DataFrame
    ablation: pd.DataFrame

    def write(self, output_dir: Path) -> dict[str, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = output_dir / "regime-study.json"
        transitions_csv = output_dir / "regime-transitions.csv"
        decisions_csv = output_dir / "weekly-decisions.csv"
        ablation_csv = output_dir / "strategy-ablation.csv"
        matrix_csv = output_dir / "asset-regime-direction-setup-matrix.csv"
        decisions_parquet = output_dir / "regime-decisions.parquet"
        challenger_csv = output_dir / "regime-challenger.csv"
        challenger_parquet = output_dir / "regime-challenger.parquet"
        summary_path.write_text(json.dumps(self.summary, indent=2), encoding="utf-8")
        self.transitions.to_csv(transitions_csv, index=False)
        self.decisions[self.decisions["policy"] == "weekly"].to_csv(decisions_csv, index=False)
        self.ablation.to_csv(ablation_csv, index=False)
        self.matrix.to_csv(matrix_csv, index=False)
        self.decisions.to_parquet(decisions_parquet, compression="zstd", index=False)
        self.challenger.to_csv(challenger_csv, index=False)
        self.challenger.to_parquet(challenger_parquet, compression="zstd", index=False)
        return {
            "regime_study": summary_path,
            "regime_transitions_csv": transitions_csv,
            "weekly_decisions_csv": decisions_csv,
            "strategy_ablation_csv": ablation_csv,
            "regime_matrix_csv": matrix_csv,
            "regime_decisions_parquet": decisions_parquet,
            "regime_challenger_csv": challenger_csv,
            "regime_challenger_parquet": challenger_parquet,
        }


class RegimeStudy:
    def __init__(
        self,
        *,
        primary_lookback_weeks: int,
        sensitivity_weeks: tuple[int, ...],
        decision_weekday: int,
        decision_hour_utc: int,
        decision_minute_utc: int,
        minimum_trades: int,
        validation_fraction: float,
        purge_hours: int,
        bootstrap_samples: int,
        lcb_quantile: float,
        seed: int,
    ) -> None:
        self.primary_lookback_weeks = primary_lookback_weeks
        self.sensitivity_weeks = sensitivity_weeks
        self.decision_weekday = decision_weekday
        self.decision_hour_utc = decision_hour_utc
        self.decision_minute_utc = decision_minute_utc
        self.minimum_trades = minimum_trades
        self.validation_fraction = validation_fraction
        self.purge_hours = purge_hours
        self.bootstrap_samples = bootstrap_samples
        self.lcb_quantile = lcb_quantile
        self.seed = seed

    def _select_arm(
        self,
        history: pd.DataFrame,
        *,
        regime_state: Literal["bull", "bear"] | None,
        round_trip_cost: float,
        seed_offset: int,
    ) -> tuple[str, float, int]:
        scoped = (
            history if regime_state is None else history[history["regime_state"] == regime_state]
        )
        outcomes: list[tuple[str, float, int]] = []
        for index, arm in enumerate(REGIME_ARMS[:-1]):
            lcb, trades = self._score_arm(
                scoped,
                arm=arm,
                round_trip_cost=round_trip_cost,
                seed_offset=seed_offset + index,
            )
            outcomes.append((arm, lcb, trades))
        eligible = [item for item in outcomes if np.isfinite(item[1]) and item[1] > 0]
        return max(eligible, key=lambda item: (item[1], item[0])) if eligible else ("flat", 0.0, 0)

    def _score_arm(
        self,
        scoped: pd.DataFrame,
        *,
        arm: str,
        round_trip_cost: float,
        seed_offset: int,
    ) -> tuple[float, int]:
        selected = scoped[(scoped[f"arm_{arm}"] != 0) & scoped["forward_r_24h"].notna()]
        risk_fraction = selected["risk_fraction"].astype(float).replace(0.0, np.nan)
        values = (
            (
                selected["forward_r_24h"].astype(float) * selected[f"arm_{arm}"].astype(int)
                - 2.0 * round_trip_cost / risk_fraction
            )
            .dropna()
            .to_numpy(dtype=float)
        )
        trades = len(values)
        lcb = (
            block_bootstrap_lcb(
                values,
                samples=self.bootstrap_samples,
                quantile=self.lcb_quantile,
                seed=self.seed + seed_offset,
            )
            if trades >= self.minimum_trades
            else float("-inf")
        )
        return lcb, trades

    def _first_decision(self, oos_start: pd.Timestamp) -> pd.Timestamp:
        day = oos_start.floor("D")
        decision = day + pd.Timedelta(days=(self.decision_weekday - day.weekday()) % 7)
        decision += pd.Timedelta(hours=self.decision_hour_utc, minutes=self.decision_minute_utc)
        if decision < oos_start:
            decision += pd.Timedelta(days=7)
        return decision

    def run(self, features: pd.DataFrame, *, round_trip_cost: float) -> RegimeStudyResult:
        arms = build_regime_arms(features)
        arms["timestamp"] = pd.to_datetime(arms["timestamp"], utc=True)
        arms = arms.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
        development, oos = purged_timestamp_split(
            arms,
            validation_fraction=self.validation_fraction,
            purge_hours=self.purge_hours,
        )
        oos_start = pd.Timestamp(oos["timestamp"].min())
        oos_end = pd.Timestamp(oos["timestamp"].max())
        rows: list[dict[str, Any]] = []

        pooled_selected_arm, pooled_selected_lcb, pooled_selected_trades = self._select_arm(
            development,
            regime_state=None,
            round_trip_cost=round_trip_cost,
            seed_offset=0,
        )
        pooled_arm = "hybrid"
        pooled_lcb, pooled_trades = self._score_arm(
            development,
            arm=pooled_arm,
            round_trip_cost=round_trip_cost,
            seed_offset=900,
        )
        rows.append(
            {
                "policy": "pooled",
                "decision_at": oos_start,
                "observable_end": pd.Timestamp(development["timestamp"].max()),
                "history_start": pd.Timestamp(development["timestamp"].min()),
                "lookback_weeks": 0,
                "regime_state": "all",
                "selected_arm": pooled_arm,
                "expectancy_lcb": pooled_lcb,
                "trades": pooled_trades,
            }
        )
        rows.append(
            {
                "policy": "pooled_selected",
                "decision_at": oos_start,
                "observable_end": pd.Timestamp(development["timestamp"].max()),
                "history_start": pd.Timestamp(development["timestamp"].min()),
                "lookback_weeks": 0,
                "regime_state": "all",
                "selected_arm": pooled_selected_arm,
                "expectancy_lcb": pooled_selected_lcb,
                "trades": pooled_selected_trades,
            }
        )
        static_map: dict[str, str] = {"mixed": "flat"}
        for state_index, state in enumerate(REGIME_STATES):
            arm, lcb, trades = self._select_arm(
                development,
                regime_state=cast(Literal["bull", "bear"], state),
                round_trip_cost=round_trip_cost,
                seed_offset=10 + state_index * 10,
            )
            static_map[state] = arm
            rows.append(
                {
                    "policy": "static",
                    "decision_at": oos_start,
                    "observable_end": pd.Timestamp(development["timestamp"].max()),
                    "history_start": pd.Timestamp(development["timestamp"].min()),
                    "lookback_weeks": 0,
                    "regime_state": state,
                    "selected_arm": arm,
                    "expectancy_lcb": lcb,
                    "trades": trades,
                }
            )

        weekly_maps: dict[tuple[pd.Timestamp, int], dict[str, str]] = {}
        first_decision = self._first_decision(oos_start)
        decision_times = pd.date_range(first_decision, oos_end, freq="7D")
        for decision_index, decision_at in enumerate(decision_times):
            observable_end = decision_at - pd.Timedelta(hours=24)
            for lookback in self.sensitivity_weeks:
                history_start = observable_end - pd.Timedelta(weeks=lookback)
                history = arms[
                    (arms["timestamp"] >= history_start) & (arms["timestamp"] < observable_end)
                ]
                mapping: dict[str, str] = {"mixed": "flat"}
                for state_index, state in enumerate(REGIME_STATES):
                    arm, lcb, trades = self._select_arm(
                        history,
                        regime_state=cast(Literal["bull", "bear"], state),
                        round_trip_cost=round_trip_cost,
                        seed_offset=1000 + decision_index * 100 + lookback * 10 + state_index,
                    )
                    mapping[state] = arm
                    rows.append(
                        {
                            "policy": "weekly",
                            "decision_at": decision_at,
                            "observable_end": observable_end,
                            "history_start": history_start,
                            "lookback_weeks": lookback,
                            "regime_state": state,
                            "selected_arm": arm,
                            "expectancy_lcb": lcb,
                            "trades": trades,
                        }
                    )
                weekly_maps[(decision_at, lookback)] = mapping

        challenger = oos.loc[:, ["timestamp", "symbol", "regime_state"]].copy()
        challenger["signal_pooled"] = oos[f"arm_{pooled_arm}"].astype(int).to_numpy()
        challenger["signal_static"] = [
            int(oos.iloc[index][f"arm_{static_map[str(state)]}"])
            for index, state in enumerate(oos["regime_state"])
        ]
        for lookback in self.sensitivity_weeks:
            column = f"signal_weekly_{lookback}"
            challenger[column] = 0
            for decision_at in decision_times:
                mapping = weekly_maps[(decision_at, lookback)]
                mask = (challenger["timestamp"] >= decision_at) & (
                    challenger["timestamp"] < decision_at + pd.Timedelta(days=7)
                )
                for state in (*REGIME_STATES, "mixed"):
                    state_mask = mask & (challenger["regime_state"] == state)
                    arm = mapping[state]
                    challenger.loc[state_mask, column] = oos.loc[state_mask, f"arm_{arm}"].astype(
                        int
                    )
        challenger["signal_weekly"] = challenger[f"signal_weekly_{self.primary_lookback_weeks}"]
        for arm in REGIME_ARMS:
            challenger[f"signal_{arm}"] = oos[f"arm_{arm}"].astype(int).to_numpy()
        ml_accept = oos.get("decision_accepted", pd.Series(True, index=oos.index)).astype(bool)
        for policy in (*REGIME_ARMS, "pooled", "static", "weekly"):
            challenger[f"signal_{policy}_ml"] = challenger[f"signal_{policy}"].where(
                ml_accept.to_numpy(), 0
            )
        decisions = pd.DataFrame(rows)
        config = {
            "arms": list(REGIME_ARMS),
            "states": [*REGIME_STATES, "mixed"],
            "primary_lookback_weeks": self.primary_lookback_weeks,
            "sensitivity_weeks": list(self.sensitivity_weeks),
            "decision_schedule_utc": (
                f"weekday={self.decision_weekday} "
                f"{self.decision_hour_utc:02d}:{self.decision_minute_utc:02d}"
            ),
            "minimum_trades": self.minimum_trades,
            "bootstrap_samples": self.bootstrap_samples,
            "lcb_quantile": self.lcb_quantile,
            "cost_multiplier": 2.0,
            "gate_eligible": False,
            "promotion_eligible": False,
        }
        study_id = f"regime-{_study_hash(config)[:24]}"
        summary = {
            "study_id": study_id,
            **config,
            "oos_start": oos_start.isoformat(),
            "oos_end": oos_end.isoformat(),
            "pooled_arm": pooled_arm,
            "pooled_selected_arm": pooled_selected_arm,
            "static_map": static_map,
            "causality": (
                "static/pooled use purged development only; weekly uses observations "
                "ending 24h before each Monday 00:05 UTC decision"
            ),
        }
        transition_mask = challenger.groupby("symbol")["regime_state"].transform(
            lambda values: values.ne(values.shift())
        )
        transitions = challenger.loc[
            transition_mask, ["timestamp", "symbol", "regime_state"]
        ].copy()
        transitions["previous_regime"] = transitions.groupby("symbol")["regime_state"].shift()
        transitions.rename(columns={"regime_state": "new_regime"}, inplace=True)
        matrix_rows: list[dict[str, Any]] = []
        for arm in REGIME_ARMS[:-1]:
            selected = oos[(oos[f"arm_{arm}"] != 0) & oos["forward_r_24h"].notna()].copy()
            if selected.empty:
                continue
            selected["direction"] = np.where(selected[f"arm_{arm}"] > 0, "long", "short")
            selected["base_round_trip_cost_r"] = round_trip_cost / selected["risk_fraction"].astype(
                float
            ).replace(0.0, np.nan)
            selected["stressed_return"] = (
                selected["forward_r_24h"].astype(float) * selected[f"arm_{arm}"].astype(int)
                - 2.0 * selected["base_round_trip_cost_r"]
            )
            selected = selected.dropna(subset=["stressed_return"])
            selected["week"] = selected["timestamp"].dt.strftime("%G-W%V")
            for keys, group in selected.groupby(["symbol", "regime_state", "direction"], sort=True):
                symbol, state, direction = cast(tuple[str, str, str], keys)
                weekly = group.groupby("week")["stressed_return"].mean()
                sensitivity_count = int(
                    (
                        (decisions["policy"] == "weekly")
                        & (decisions["regime_state"] == state)
                        & (decisions["selected_arm"] == arm)
                    ).sum()
                )
                matrix_rows.append(
                    {
                        "asset": symbol,
                        "regime": state,
                        "direction": direction,
                        "setup": arm,
                        "trades": len(group),
                        "gross_expectancy": float(
                            (
                                group["forward_r_24h"].astype(float)
                                * group[f"arm_{arm}"].astype(int)
                            ).mean()
                        ),
                        "expectancy_costs_2x": float(group["stressed_return"].mean()),
                        "costs_2x": float(
                            (2.0 * group["base_round_trip_cost_r"].astype(float)).sum()
                        ),
                        "turnover_events": int(group[f"arm_{arm}"].abs().sum()),
                        "weekly_stability": float((weekly > 0).mean()),
                        "sensitivity_selections": sensitivity_count,
                    }
                )
        matrix = pd.DataFrame(matrix_rows)
        ablation = (
            decisions.groupby(
                ["policy", "lookback_weeks", "regime_state", "selected_arm"],
                dropna=False,
            )
            .agg(
                decisions=("decision_at", "count"),
                expectancy_lcb=("expectancy_lcb", "mean"),
                trades=("trades", "sum"),
            )
            .reset_index()
        )
        return RegimeStudyResult(
            study_id,
            decisions,
            challenger,
            summary,
            transitions,
            matrix,
            ablation,
        )
