from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
import pandas as pd

from orizzonte_desk.config import Settings
from orizzonte_desk.constants import SYMBOLS
from orizzonte_desk.data import stable_fingerprint
from orizzonte_desk.features import FEATURE_COLUMNS, prepare_features
from orizzonte_desk.gates import evaluate_gate, save_gate
from orizzonte_desk.metrics import MetricsBundle, calculate_metrics
from orizzonte_desk.models import Position, RiskSnapshot, Side, Signal, Trade
from orizzonte_desk.paths import AppPaths
from orizzonte_desk.regimes import RegimeStudy
from orizzonte_desk.risk import RiskManager, RiskViolation
from orizzonte_desk.strategy import SignalGenerator


@dataclass(slots=True)
class BacktestResult:
    run_id: str
    trades: list[Trade]
    equity: pd.DataFrame
    metrics: MetricsBundle
    stressed_metrics: MetricsBundle
    gate_path: Path
    artifacts: dict[str, Path]
    stress_results: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass(slots=True)
class _OpenPosition:
    position: Position
    fees: float
    funding: float
    slippage: float
    realized_pnl: float = 0.0
    mae: float = 0.0
    mfe: float = 0.0


class EventBacktester:
    def __init__(self, settings: Settings, paths: AppPaths) -> None:
        self.settings = settings
        self.paths = paths
        self.risk = RiskManager(settings.risk)

    def run(
        self,
        market: pd.DataFrame,
        *,
        source: str,
        dataset_hash: str = "",
        require_promoted_model: bool = False,
        cost_multiplier: float = 1.0,
        signal_delay_hours: int = 1,
        missing_fraction: float = 0.0,
        seed: int | None = None,
        persist: bool = True,
        enriched_override: pd.DataFrame | None = None,
        model_id: str | None = None,
        run_stress_suite: bool = True,
        evaluation_scope: str = "candidate",
        protocol_hash: str | None = None,
        metrics_monte_carlo_samples: int | None = None,
        correlations_override: dict[pd.Timestamp, dict[tuple[str, str], float]] | None = None,
        objective_trades_only: bool = False,
        run_regime_study: bool = False,
    ) -> BacktestResult:
        from orizzonte_desk.ml import MetaModelRegistry

        if evaluation_scope not in {"candidate", "training_protocol"}:
            raise ValueError(f"Escopo de avaliação desconhecido: {evaluation_scope}")
        if evaluation_scope == "training_protocol" and not protocol_hash:
            raise ValueError("Gate de protocolo exige protocol_hash")
        if objective_trades_only and persist:
            raise ValueError("Avaliação trades-only não pode persistir artefatos")
        seed = seed if seed is not None else self.settings.backtest.random_seed
        prepared_market = market.copy()
        if missing_fraction > 0 and enriched_override is None:
            rng = np.random.default_rng(seed)
            keep = rng.random(len(prepared_market)) >= missing_fraction
            prepared_market = prepared_market.loc[keep].copy()
        registry = MetaModelRegistry(
            self.paths,
            self.settings.research,
            self.settings.execution,
            settings=self.settings,
        )
        model_bundle = registry.load_candidate(model_id) if model_id else None
        decision_policy = (
            registry.load_decision_policy(model_id) if model_bundle is not None else None
        )
        generator = SignalGenerator(
            self.settings.strategy,
            registry,
        )
        if enriched_override is None:
            enriched = generator.enrich(
                prepared_market,
                require_promoted_model=require_promoted_model,
                model_bundle=model_bundle,
                model_id=model_id,
                decision_policy=decision_policy,
            )
        else:
            enriched = enriched_override.copy()
            if missing_fraction > 0:
                rng = np.random.default_rng(seed)
                enriched = enriched.loc[rng.random(len(enriched)) >= missing_fraction].copy()
        enriched["timestamp"] = pd.to_datetime(enriched["timestamp"], utc=True)
        enriched = enriched.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
        run_fingerprint = json.dumps(
            {
                "source": source,
                "dataset_hash": dataset_hash,
                "settings": self.settings.model_dump(mode="json"),
                "cost_multiplier": cost_multiplier,
                "signal_delay_hours": signal_delay_hours,
                "missing_fraction": missing_fraction,
                "seed": seed,
                "model_id": model_id,
                "evaluation_scope": evaluation_scope,
                "protocol_hash": protocol_hash,
                "metrics_monte_carlo_samples": metrics_monte_carlo_samples,
                "correlations_override": correlations_override is not None,
                "run_regime_study": run_regime_study,
            },
            sort_keys=True,
        ).encode()
        run_id = f"{source}-{hashlib.sha256(run_fingerprint).hexdigest()[:12]}"
        initial = self.settings.backtest.initial_capital
        cash = initial
        day_start_equity = initial
        high_water_mark = initial
        current_day: date | None = None
        positions: dict[str, _OpenPosition] = {}
        pending: dict[str, tuple[pd.Timestamp, Signal]] = {}
        trades: list[Trade] = []
        equity_rows: list[dict[str, Any]] = []
        price_history: dict[str, list[float]] = {symbol: [] for symbol in SYMBOLS}
        last_prices: dict[str, float] = {}
        profit_locked = False
        loss_locked = False
        drawdown_locked = False

        grouped = enriched.groupby("timestamp", sort=True)
        for timestamp_value, timestamp_frame in cast(Any, grouped):
            timestamp = pd.Timestamp(cast(Any, timestamp_value))
            if current_day != timestamp.date():
                current_day = timestamp.date()
                day_start_equity = self._mark_to_market(cash, positions, last_prices)
                profit_locked = False
                loss_locked = False

            needs_correlations = any(
                symbol in pending and timestamp >= pending[symbol][0] and symbol not in positions
                for symbol in timestamp_frame["symbol"].astype(str)
            )
            correlations = (
                correlations_override.get(timestamp, {})
                if needs_correlations and correlations_override is not None
                else self._correlations(price_history)
                if needs_correlations
                else {}
            )
            for row_value in timestamp_frame.itertuples():
                row: Any = row_value
                symbol = str(row.symbol)
                last_prices[symbol] = float(row.close)
                price_history[symbol].append(float(row.close))
                if len(price_history[symbol]) > 24 * 60:
                    price_history[symbol] = price_history[symbol][-24 * 60 :]

                opened = positions.get(symbol)
                if opened is not None:
                    cash += self._apply_funding(opened, float(row.funding_rate))
                    exit_event = self._evaluate_exit(opened, row, timestamp)
                    if exit_event is not None:
                        exit_price, exit_size, reason = exit_event
                        cash_delta, trade = self._close_leg(
                            opened,
                            exit_price=exit_price,
                            exit_size=exit_size,
                            closed_at=timestamp.to_pydatetime(),
                            reason=reason,
                            cost_multiplier=cost_multiplier,
                        )
                        cash += cash_delta
                        trades.append(trade)
                        if opened.position.size <= 1e-12:
                            positions.pop(symbol, None)

                due = pending.get(symbol)
                if due is not None and timestamp >= due[0] and symbol not in positions:
                    _, signal = due
                    pending.pop(symbol, None)
                    equity_now = self._mark_to_market(cash, positions, last_prices)
                    snapshot = RiskSnapshot(
                        timestamp=timestamp.to_pydatetime(),
                        budget=initial,
                        equity=max(equity_now, 0.01),
                        day_start_equity=max(day_start_equity, 0.01),
                        high_water_mark=max(high_water_mark, 0.01),
                        open_risk=sum(x.position.initial_risk_usdc for x in positions.values()),
                        open_notional=sum(x.position.notional for x in positions.values()),
                        positions_count=len(positions),
                        profit_locked=profit_locked,
                        loss_locked=loss_locked,
                        drawdown_locked=drawdown_locked,
                    )
                    try:
                        self.risk.assert_new_entry_allowed(
                            signal,
                            snapshot,
                            (item.position for item in positions.values()),
                            correlations,
                        )
                        executed = self._open_position(
                            signal,
                            open_price=float(row.open),
                            timestamp=timestamp.to_pydatetime(),
                            equity=equity_now,
                            open_notional=sum(x.position.notional for x in positions.values()),
                            cost_multiplier=cost_multiplier,
                        )
                        # Entry price already includes slippage. Cash is debited only for the fee;
                        # slippage is reported separately but flows through gross PnL.
                        cash -= executed.fees
                        positions[symbol] = executed
                    except RiskViolation:
                        pass

                if int(row.signal) != 0 and symbol not in positions and symbol not in pending:
                    side = Side.LONG if row.signal > 0 else Side.SHORT
                    signal = Signal(
                        timestamp=timestamp.to_pydatetime(),
                        symbol=symbol,
                        side=side,
                        score=float(row.setup_score),
                        probability=float(row.ml_probability),
                        entry_reference=float(row.close),
                        stop_distance=float(row.stop_distance),
                        atr=float(row.atr_4h),
                        regime="bull" if row.daily_trend > 0 else "bear",
                        reasons=("confluência multi-timeframe", "meta-modelo"),
                    )
                    pending[symbol] = (
                        timestamp + pd.Timedelta(hours=signal_delay_hours),
                        signal,
                    )

            equity_now = self._mark_to_market(cash, positions, last_prices)
            high_water_mark = max(high_water_mark, equity_now)
            daily_return = equity_now / max(day_start_equity, 0.01) - 1
            drawdown = equity_now / max(high_water_mark, 0.01) - 1
            profit_locked = profit_locked or daily_return >= self.settings.risk.daily_profit_lock
            loss_locked = loss_locked or daily_return <= -self.settings.risk.daily_loss_limit
            drawdown_locked = drawdown_locked or drawdown <= -self.settings.risk.max_drawdown_limit
            if loss_locked or drawdown_locked:
                for symbol, opened in list(positions.items()):
                    price = last_prices[symbol]
                    cash_delta, trade = self._close_leg(
                        opened,
                        exit_price=price,
                        exit_size=opened.position.size,
                        closed_at=timestamp.to_pydatetime(),
                        reason="daily_loss_lock" if loss_locked else "drawdown_kill_switch",
                        cost_multiplier=cost_multiplier,
                    )
                    cash += cash_delta
                    trades.append(trade)
                    positions.pop(symbol, None)
                pending.clear()
                equity_now = cash
            equity_rows.append(
                {
                    "timestamp": timestamp,
                    "equity": equity_now,
                    "cash": cash,
                    "open_positions": len(positions),
                    "profit_locked": profit_locked,
                    "loss_locked": loss_locked,
                    "drawdown_locked": drawdown_locked,
                }
            )

        if positions and last_prices and equity_rows:
            final_time = pd.Timestamp(equity_rows[-1]["timestamp"]).to_pydatetime()
            for symbol, opened in list(positions.items()):
                cash_delta, trade = self._close_leg(
                    opened,
                    exit_price=last_prices[symbol],
                    exit_size=opened.position.size,
                    closed_at=final_time,
                    reason="end_of_test",
                    cost_multiplier=cost_multiplier,
                )
                cash += cash_delta
                trades.append(trade)
            equity_rows[-1]["equity"] = cash
            equity_rows[-1]["cash"] = cash
            equity_rows[-1]["open_positions"] = 0

        equity_frame = pd.DataFrame(equity_rows)
        if objective_trades_only:
            empty_metrics = MetricsBundle(
                {}, {}, {}, pd.Series(dtype=float), pd.Series(dtype=float)
            )
            return BacktestResult(
                run_id,
                trades,
                equity_frame,
                empty_metrics,
                empty_metrics,
                self.paths.reports / run_id / "gate.json",
                {},
            )
        metrics = calculate_metrics(
            equity_frame,
            trades,
            initial_capital=initial,
            monte_carlo_samples=(
                metrics_monte_carlo_samples
                if metrics_monte_carlo_samples is not None
                else self.settings.backtest.monte_carlo_samples
            ),
            seed=seed,
        )
        stress_results: dict[str, dict[str, float]] = {}
        if not run_stress_suite or cost_multiplier >= 2.0:
            stressed = metrics
        else:
            costs = self.run(
                market,
                source=f"{source}-stress-costs-2x",
                dataset_hash=dataset_hash,
                require_promoted_model=require_promoted_model,
                cost_multiplier=2.0,
                signal_delay_hours=signal_delay_hours,
                seed=seed,
                persist=False,
                enriched_override=enriched,
                model_id=model_id,
                run_stress_suite=False,
            ).metrics
            delayed = self.run(
                market,
                source=f"{source}-stress-delay",
                dataset_hash=dataset_hash,
                cost_multiplier=cost_multiplier,
                signal_delay_hours=max(2, signal_delay_hours + 1),
                seed=seed,
                persist=False,
                enriched_override=enriched,
                model_id=model_id,
                run_stress_suite=False,
            ).metrics
            adverse = enriched.copy()
            adverse["funding_rate"] = np.where(
                adverse["signal_raw"] >= 0,
                adverse["funding_rate"].abs() + 0.0001,
                -(adverse["funding_rate"].abs() + 0.0001),
            )
            adverse_metrics = self.run(
                market,
                source=f"{source}-stress-adverse",
                dataset_hash=dataset_hash,
                cost_multiplier=cost_multiplier,
                signal_delay_hours=signal_delay_hours,
                missing_fraction=max(missing_fraction, 0.005),
                seed=seed,
                persist=False,
                enriched_override=adverse,
                model_id=model_id,
                run_stress_suite=False,
            ).metrics
            perturbed = enriched.copy()
            if "decision_threshold" in perturbed:
                perturbed["signal"] = perturbed["signal_raw"].where(
                    perturbed["ml_probability"] >= perturbed["decision_threshold"] + 0.03,
                    0,
                )
            perturbed_metrics = self.run(
                market,
                source=f"{source}-stress-parameters",
                dataset_hash=dataset_hash,
                cost_multiplier=cost_multiplier,
                signal_delay_hours=signal_delay_hours,
                seed=seed,
                persist=False,
                enriched_override=perturbed,
                model_id=model_id,
                run_stress_suite=False,
            ).metrics
            stressed = costs
            stress_results = {
                "costs_2x": costs.summary,
                "delay_one_candle": delayed.summary,
                "missing_and_adverse_funding": adverse_metrics.summary,
                "parameter_perturbation": perturbed_metrics.summary,
            }
        model_binding: dict[str, Any]
        if evaluation_scope == "training_protocol":
            from orizzonte_desk.ml import git_commit_fingerprint, research_code_fingerprint

            resolved_model_hash = None
            model_binding = {
                "code_hash": research_code_fingerprint(),
                "commit_hash": git_commit_fingerprint(self.paths.root),
            }
        else:
            resolved_model_hash, model_binding = self._model_release(model_id)
        model_hash = resolved_model_hash if evaluation_scope == "candidate" else None
        dataset_hashes = tuple(value for value in (dataset_hash,) if value)
        release_binding = {
            **model_binding,
            "dataset_hashes": sorted(dataset_hashes or model_binding.get("dataset_hashes", [])),
            "config_fingerprint": stable_fingerprint(self.settings.model_dump(mode="json")),
            "model_hash": model_hash,
            "evaluation_scope": evaluation_scope,
        }
        if protocol_hash:
            release_binding["protocol_hash"] = protocol_hash
        gate = evaluate_gate(
            metrics.summary,
            metrics.by_symbol,
            stressed.summary,
            dataset_hashes=dataset_hashes,
            model_hash=model_hash,
        )
        run_dir = self.paths.reports / run_id
        gate_path = run_dir / "gate.json"
        gate_eligible = (
            evaluation_scope == "training_protocol"
            or model_id is not None
            or require_promoted_model
        )
        artifacts: dict[str, Path] = {}
        if persist:
            run_dir.mkdir(parents=True, exist_ok=True)
            equity_path = run_dir / "equity.csv"
            trades_path = run_dir / "trades.csv"
            metrics_path = run_dir / "metrics.json"
            equity_frame.to_csv(equity_path, index=False)
            pd.DataFrame([trade.model_dump(mode="json") for trade in trades]).to_csv(
                trades_path, index=False
            )
            metrics_path.write_text(
                json.dumps(
                    {
                        "summary": metrics.summary,
                        "by_symbol": metrics.by_symbol,
                        "by_direction": metrics.by_direction,
                        "stress": stressed.summary,
                        "stress_suite": stress_results,
                        "source": source,
                        "dataset_hash": dataset_hash,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            if gate_eligible:
                release_binding["gate_eligible"] = True
                save_gate(gate, gate_path, release_binding=release_binding)
            funnel_artifacts = self._write_funnel_artifacts(
                enriched,
                trades,
                run_dir=run_dir,
                model_hash=model_hash,
                dataset_hash=dataset_hash,
            )
            regime_artifacts = (
                self._run_regime_research(
                    market,
                    enriched,
                    source=source,
                    dataset_hash=dataset_hash,
                    seed=seed,
                    run_dir=run_dir,
                )
                if run_regime_study
                and market.attrs.get("dataset_role", "development") == "development"
                else {}
            )
            artifacts = {
                "equity": equity_path,
                "trades": trades_path,
                "metrics": metrics_path,
                **funnel_artifacts,
                **regime_artifacts,
            }
            if gate_eligible:
                artifacts["gate"] = gate_path
        return BacktestResult(
            run_id,
            trades,
            equity_frame,
            metrics,
            stressed,
            gate_path,
            artifacts,
            stress_results,
        )

    def _write_funnel_artifacts(
        self,
        enriched: pd.DataFrame,
        trades: list[Trade],
        *,
        run_dir: Path,
        model_hash: str | None,
        dataset_hash: str,
    ) -> dict[str, Path]:
        candidates = enriched[enriched["signal_raw"] != 0].copy()
        executed = [(trade.symbol, pd.Timestamp(trade.opened_at)) for trade in trades]
        events: list[dict[str, Any]] = []
        stages = (
            "setup_raw",
            "feature_complete",
            "probability",
            "threshold",
            "risk",
            "correlation",
            "sizing",
            "order",
            "fill",
            "protection",
            "exit",
        )
        for row in candidates.to_dict(orient="records"):
            timestamp = pd.Timestamp(row["timestamp"])
            symbol = str(row["symbol"])
            candidate_id = hashlib.sha256(
                f"{dataset_hash}|{timestamp.isoformat()}|{symbol}".encode()
            ).hexdigest()[:24]
            feature_complete = all(pd.notna(row.get(column)) for column in FEATURE_COLUMNS)
            probability_complete = np.isfinite(float(row.get("ml_probability", np.nan)))
            threshold_passed = bool(row.get("decision_accepted", False))
            fill_found = any(
                trade_symbol == symbol
                and timestamp <= opened_at <= timestamp + pd.Timedelta(hours=4)
                for trade_symbol, opened_at in executed
            )
            status_by_stage = {
                "setup_raw": (True, None),
                "feature_complete": (
                    feature_complete,
                    None if feature_complete else "feature_incomplete",
                ),
                "probability": (
                    probability_complete,
                    None if probability_complete else "probability_unavailable",
                ),
                "threshold": (
                    threshold_passed,
                    None if threshold_passed else "decision_policy_rejected",
                ),
            }
            for stage in stages[4:]:
                if not threshold_passed:
                    status_by_stage[stage] = (False, "blocked_by_threshold")
                elif not fill_found:
                    status_by_stage[stage] = (
                        False,
                        "risk_correlation_or_sizing_rejected"
                        if stage in {"risk", "correlation", "sizing"}
                        else "blocked_upstream",
                    )
                else:
                    status_by_stage[stage] = (True, None)
            for stage in stages:
                passed, reason = status_by_stage[stage]
                events.append(
                    {
                        "candidate_id": candidate_id,
                        "timestamp": timestamp,
                        "symbol": symbol,
                        "stage": stage,
                        "passed": passed,
                        "rejection_reason": reason,
                        "fold": row.get("fold", "full"),
                        "policy_id": row.get("decision_policy_id", "deterministic-no-ml"),
                        "model_hash": model_hash,
                        "dataset_hash": dataset_hash,
                        "probability": row.get("ml_probability"),
                        "threshold": row.get("decision_threshold"),
                    }
                )
        event_frame = pd.DataFrame(
            events,
            columns=[
                "candidate_id",
                "timestamp",
                "symbol",
                "stage",
                "passed",
                "rejection_reason",
                "fold",
                "policy_id",
                "model_hash",
                "dataset_hash",
                "probability",
                "threshold",
            ],
        )
        event_path = run_dir / "funnel-events.parquet"
        summary_path = run_dir / "funnel-summary.json"
        fold_path = run_dir / "funnel-by-fold.csv"
        calibration_path = run_dir / "probability-calibration.csv"
        event_frame.to_parquet(event_path, compression="zstd", index=False)
        by_fold = (
            event_frame.groupby(
                ["fold", "stage", "passed", "rejection_reason"],
                dropna=False,
            )
            .size()
            .rename("events")
            .reset_index()
        )
        by_fold.to_csv(fold_path, index=False)
        summary_path.write_text(
            json.dumps(
                {
                    "candidates": len(candidates),
                    "stages": list(stages),
                    "passed_by_stage": {
                        stage: int(((event_frame["stage"] == stage) & event_frame["passed"]).sum())
                        for stage in stages
                    },
                    "model_hash": model_hash,
                    "dataset_hash": dataset_hash,
                    "policy_ids": sorted(
                        str(value) for value in event_frame["policy_id"].dropna().unique()
                    ),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        calibration = candidates.dropna(subset=["ml_probability", "label"]).copy()
        if calibration.empty:
            calibration_frame = pd.DataFrame(
                columns=[
                    "probability_bin",
                    "predicted_probability",
                    "observed_rate",
                    "samples",
                    "fold",
                    "policy_id",
                    "model_hash",
                    "dataset_hash",
                ]
            )
        else:
            calibration["probability_bin"] = pd.cut(
                calibration["ml_probability"],
                bins=np.linspace(0, 1, 11).tolist(),
                include_lowest=True,
            ).astype(str)
            calibration_frame = (
                calibration.groupby("probability_bin", observed=False)
                .agg(
                    predicted_probability=("ml_probability", "mean"),
                    observed_rate=("label", "mean"),
                    samples=("label", "count"),
                )
                .reset_index()
            )
            calibration_frame["fold"] = "full"
            calibration_frame["policy_id"] = (
                str(calibration["decision_policy_id"].iloc[0])
                if "decision_policy_id" in calibration
                else "deterministic-no-ml"
            )
            calibration_frame["model_hash"] = model_hash
            calibration_frame["dataset_hash"] = dataset_hash
        calibration_frame.to_csv(calibration_path, index=False)
        return {
            "funnel_events": event_path,
            "funnel_summary": summary_path,
            "funnel_by_fold": fold_path,
            "probability_calibration": calibration_path,
        }

    def _bootstrap_superiority(
        self,
        challenger: pd.Series,
        baseline: pd.Series,
        *,
        seed: int,
    ) -> float:
        paired = pd.concat([challenger.rename("challenger"), baseline.rename("baseline")], axis=1)
        paired = paired.dropna()
        if paired.empty:
            return 0.0
        differences = (paired["challenger"] - paired["baseline"]).to_numpy(dtype=float)
        rng = np.random.default_rng(seed)
        block = max(1, min(10, len(differences) // 10))
        superior = 0
        for _ in range(self.settings.research.threshold_bootstrap_samples):
            sample: list[float] = []
            while len(sample) < len(differences):
                start = int(rng.integers(0, max(1, len(differences) - block + 1)))
                sample.extend(differences[start : start + block])
            superior += float(np.mean(sample[: len(differences)])) > 0
        return superior / self.settings.research.threshold_bootstrap_samples

    def _run_regime_research(
        self,
        market: pd.DataFrame,
        enriched: pd.DataFrame,
        *,
        source: str,
        dataset_hash: str,
        seed: int,
        run_dir: Path,
    ) -> dict[str, Path]:
        regime_study = RegimeStudy(
            primary_lookback_weeks=self.settings.research.regime_primary_lookback_weeks,
            sensitivity_weeks=self.settings.research.regime_sensitivity_weeks,
            decision_weekday=self.settings.research.weekly_decision_weekday,
            decision_hour_utc=self.settings.research.weekly_decision_hour_utc,
            decision_minute_utc=self.settings.research.weekly_decision_minute_utc,
            minimum_trades=self.settings.research.threshold_min_validation_trades,
            validation_fraction=self.settings.research.inner_validation_fraction,
            purge_hours=self.settings.research.purge_hours,
            bootstrap_samples=self.settings.research.threshold_bootstrap_samples,
            lcb_quantile=self.settings.research.threshold_lcb_quantile,
            seed=seed,
        ).run(
            enriched,
            round_trip_cost=2
            * (
                self.settings.execution.taker_fee
                + self.settings.execution.slippage_bps_sol_xrp / 10_000
            ),
        )
        regime_dir = run_dir / "regimes"
        artifacts = regime_study.write(regime_dir)
        challenger_base = enriched.merge(
            regime_study.challenger.drop(columns=["regime_state"]),
            on=["timestamp", "symbol"],
            how="inner",
        )
        evaluations: dict[str, BacktestResult] = {}
        for policy in ("hybrid", "breakout", "pullback", "flat", "static", "weekly"):
            for ml_suffix in ("", "_ml"):
                name = f"{policy}{ml_suffix or '_no_ml'}"
                challenger_enriched = challenger_base.copy()
                challenger_enriched["signal"] = challenger_enriched[
                    f"signal_{policy}{ml_suffix}"
                ].astype(int)
                evaluations[name] = self.run(
                    market,
                    source=f"{source}-regime-{name}",
                    dataset_hash=dataset_hash,
                    seed=seed,
                    persist=False,
                    enriched_override=challenger_enriched,
                    run_stress_suite=False,
                    metrics_monte_carlo_samples=100,
                )
        policy_ids = set(enriched.get("decision_policy_id", pd.Series(dtype=str)).astype(str))
        comparison_suffix = "_ml" if policy_ids - {"", "none", "deterministic-no-ml"} else "_no_ml"
        baseline_name = f"hybrid{comparison_suffix}"
        weekly_name = f"weekly{comparison_suffix}"
        static_name = f"static{comparison_suffix}"
        if (
            evaluations[static_name].metrics.summary["sharpe"]
            >= evaluations[weekly_name].metrics.summary["sharpe"] - 0.10
        ):
            selected_name = static_name
            selection_reason = "static_preferred_within_0.10_sharpe"
        else:
            selected_name = weekly_name
            selection_reason = "weekly_selector_outside_static_tolerance"
        selected = evaluations[selected_name]
        baseline = evaluations[baseline_name]
        selected_column = f"signal_{selected_name.replace('_no_ml', '').replace('_ml', '')}"
        if selected_name.endswith("_ml"):
            selected_column += "_ml"
        stressed_enriched = challenger_base.copy()
        stressed_enriched["signal"] = stressed_enriched[selected_column].astype(int)
        stressed = self.run(
            market,
            source=f"{source}-regime-{selected_name}-costs-2x",
            dataset_hash=dataset_hash,
            seed=seed,
            persist=False,
            enriched_override=stressed_enriched,
            run_stress_suite=False,
            cost_multiplier=2.0,
            metrics_monte_carlo_samples=100,
        )
        superiority = self._bootstrap_superiority(
            selected.metrics.daily_returns,
            baseline.metrics.daily_returns,
            seed=seed,
        )
        positive_assets = sum(
            selected.metrics.by_symbol.get(symbol, {}).get("net_pnl", 0.0) > 0 for symbol in SYMBOLS
        )
        criteria = {
            "bootstrap_superiority_95": superiority >= 0.95,
            "sharpe_improvement_020": (
                selected.metrics.summary["sharpe"] - baseline.metrics.summary["sharpe"] >= 0.20
            ),
            "profit_factor": selected.metrics.summary["profit_factor"] >= 1.15,
            "expectancy_costs_2x": stressed.metrics.summary["expectancy"] > 0,
            "drawdown_within_2pp": (
                selected.metrics.summary["max_drawdown"]
                <= baseline.metrics.summary["max_drawdown"] + 0.02
            ),
            "three_positive_assets": positive_assets >= 3,
        }
        evaluation_rows: list[dict[str, Any]] = []
        for name, result in evaluations.items():
            evaluation_rows.append(
                {
                    "policy": name,
                    **result.metrics.summary,
                    "selected": name == selected_name,
                }
            )
        ablation_path = artifacts["strategy_ablation_csv"]
        pd.DataFrame(evaluation_rows).to_csv(ablation_path, index=False)
        event_path = regime_dir / "regime-event-driven.json"
        event_path.write_text(
            json.dumps(
                {
                    "selected_policy": selected_name,
                    "official_baseline": baseline_name,
                    "weekly_challenger": weekly_name,
                    "selection_reason": selection_reason,
                    "bootstrap_superiority_probability": superiority,
                    "positive_assets": positive_assets,
                    "criteria": criteria,
                    "research_approved": all(criteria.values()),
                    "gate_eligible": False,
                    "promotion_eligible": False,
                    "costs_2x": stressed.metrics.summary,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        summary = dict(regime_study.summary)
        summary["event_driven_decision"] = json.loads(event_path.read_text(encoding="utf-8"))
        artifacts["regime_study"].write_text(json.dumps(summary, indent=2), encoding="utf-8")
        artifacts["regime_event_driven_json"] = event_path
        return artifacts

    def _open_position(
        self,
        signal: Signal,
        *,
        open_price: float,
        timestamp: datetime,
        equity: float,
        open_notional: float,
        cost_multiplier: float,
    ) -> _OpenPosition:
        slippage_bps = (
            self.settings.execution.slippage_bps_btc_eth
            if signal.symbol in {"BTC", "ETH"}
            else self.settings.execution.slippage_bps_sol_xrp
        ) * cost_multiplier
        execution_price = open_price * (1 + signal.side.sign * slippage_bps / 10_000)
        adjusted_signal = signal.model_copy(update={"entry_reference": execution_price})
        sizing = self.risk.size_order(
            adjusted_signal,
            budget=self.settings.backtest.initial_capital,
            equity=equity,
            open_notional=open_notional,
        )
        fee = sizing.notional * self.settings.execution.taker_fee * cost_multiplier
        slippage = abs(execution_price - open_price) * sizing.size
        position = Position(
            symbol=signal.symbol,
            side=signal.side,
            size=sizing.size,
            initial_size=sizing.size,
            entry_price=execution_price,
            stop_price=sizing.stop_price,
            take_profit_price=sizing.take_profit_price,
            opened_at=timestamp,
            initial_risk_usdc=sizing.risk_usdc,
        )
        return _OpenPosition(position=position, fees=fee, funding=0.0, slippage=slippage)

    def _evaluate_exit(
        self,
        opened: _OpenPosition,
        row: Any,
        timestamp: pd.Timestamp,
    ) -> tuple[float, float, str] | None:
        position = opened.position
        side = position.side.sign
        adverse = (
            (float(row.low) - position.entry_price) * side
            if side > 0
            else (position.entry_price - float(row.high))
        )
        favorable = (
            (float(row.high) - position.entry_price) * side
            if side > 0
            else (position.entry_price - float(row.low))
        )
        opened.mae = min(opened.mae, adverse)
        opened.mfe = max(opened.mfe, favorable)
        stop = position.trailing_price or position.stop_price
        stop_hit = float(row.low) <= stop if side > 0 else float(row.high) >= stop
        if stop_hit:
            return stop, position.size, "stop"
        target_hit = (
            float(row.high) >= position.take_profit_price
            if side > 0
            else float(row.low) <= position.take_profit_price
        )
        if target_hit and not position.partial_taken:
            exit_size = position.initial_size * self.settings.strategy.partial_fraction
            position.partial_taken = True
            position.trailing_price = position.entry_price
            return position.take_profit_price, min(exit_size, position.size), "partial_1r"
        atr = float(row.atr_4h) if np.isfinite(float(row.atr_4h)) else 0.0
        if position.partial_taken and atr > 0:
            candidate = float(row.close) - side * atr * self.settings.strategy.atr_trailing_multiple
            if side > 0:
                position.trailing_price = max(
                    position.trailing_price or position.entry_price, candidate
                )
            else:
                position.trailing_price = min(
                    position.trailing_price or position.entry_price, candidate
                )
        held_hours = (timestamp.to_pydatetime() - position.opened_at).total_seconds() / 3600
        if held_hours >= self.settings.strategy.max_holding_hours:
            return float(row.close), position.size, "time_stop"
        if int(row.signal) == -side:
            return float(row.close), position.size, "regime_reversal"
        return None

    def _close_leg(
        self,
        opened: _OpenPosition,
        *,
        exit_price: float,
        exit_size: float,
        closed_at: datetime,
        reason: str,
        cost_multiplier: float,
    ) -> tuple[float, Trade]:
        position = opened.position
        exit_size = min(exit_size, position.size)
        slippage_bps = (
            self.settings.execution.slippage_bps_btc_eth
            if position.symbol in {"BTC", "ETH"}
            else self.settings.execution.slippage_bps_sol_xrp
        ) * cost_multiplier
        execution_exit = exit_price * (1 - position.side.sign * slippage_bps / 10_000)
        exit_slippage = abs(execution_exit - exit_price) * exit_size
        notional = exit_size * execution_exit
        fee = notional * self.settings.execution.taker_fee * cost_multiplier
        gross = (execution_exit - position.entry_price) * position.side.sign * exit_size
        funding_share = opened.funding * (exit_size / position.size)
        opened.funding -= funding_share
        entry_fee_share = opened.fees * (exit_size / position.initial_size)
        net = gross - fee - funding_share - entry_fee_share
        position.size -= exit_size
        trade = Trade(
            symbol=position.symbol,
            side=position.side,
            opened_at=position.opened_at,
            closed_at=closed_at,
            entry_price=position.entry_price,
            exit_price=execution_exit,
            size=exit_size,
            gross_pnl=gross,
            net_pnl=net,
            fees=fee + entry_fee_share,
            funding=funding_share,
            slippage=opened.slippage * (exit_size / position.initial_size) + exit_slippage,
            exit_reason=reason,
            mae=opened.mae,
            mfe=opened.mfe,
        )
        # Funding was already applied to cash on each hourly event.
        return gross - fee, trade

    @staticmethod
    def _apply_funding(opened: _OpenPosition, funding_rate: float) -> float:
        payment = opened.position.notional * funding_rate * opened.position.side.sign
        opened.funding += payment
        return -payment

    @staticmethod
    def _mark_to_market(
        cash: float,
        positions: dict[str, _OpenPosition],
        prices: dict[str, float],
    ) -> float:
        unrealized = sum(
            (prices.get(symbol, opened.position.entry_price) - opened.position.entry_price)
            * opened.position.side.sign
            * opened.position.size
            for symbol, opened in positions.items()
        )
        return cash + unrealized

    @staticmethod
    def _correlations(history: dict[str, list[float]]) -> dict[tuple[str, str], float]:
        result: dict[tuple[str, str], float] = {}
        for index, left in enumerate(SYMBOLS):
            for right in SYMBOLS[index + 1 :]:
                length = min(len(history[left]), len(history[right]), 24 * 30)
                if length < 24 * 7:
                    continue
                left_returns = np.diff(np.log(history[left][-length:]))
                right_returns = np.diff(np.log(history[right][-length:]))
                correlation = np.corrcoef(left_returns, right_returns)[0, 1]
                if np.isfinite(correlation):
                    result[(left, right)] = float(correlation)
        return result

    def _model_release(self, model_id: str | None) -> tuple[str | None, dict[str, Any]]:
        pointer = (
            self.paths.models / f"{model_id}.json"
            if model_id
            else self.paths.models / "promoted.json"
        )
        if not pointer.exists():
            from orizzonte_desk.ml import git_commit_fingerprint, research_code_fingerprint

            return None, {
                "code_hash": research_code_fingerprint(),
                "commit_hash": git_commit_fingerprint(self.paths.root),
            }
        payload = json.loads(pointer.read_text(encoding="utf-8"))
        model_hash = payload.get("model_hash") or payload.get("promoted_hash")
        return cast(str | None, model_hash), cast(
            dict[str, Any], payload.get("release_binding", {})
        )


def walk_forward_windows(
    timestamps: pd.Series,
    *,
    training_months: int,
    validation_months: int,
    test_months: int,
    step_months: int,
) -> list[dict[str, pd.Timestamp]]:
    values = pd.to_datetime(timestamps, utc=True).sort_values()
    start = values.min().normalize()
    end = values.max().normalize()
    windows: list[dict[str, pd.Timestamp]] = []
    cursor = start
    while True:
        train_end = cursor + pd.DateOffset(months=training_months)
        validation_end = train_end + pd.DateOffset(months=validation_months)
        test_end = validation_end + pd.DateOffset(months=test_months)
        if test_end > end:
            break
        windows.append(
            {
                "train_start": start,
                "train_end": train_end,
                "validation_end": validation_end,
                "test_end": test_end,
            }
        )
        cursor += pd.DateOffset(months=step_months)
    return windows


class WalkForwardEvaluator:
    """Anchored walk-forward training with a purged/embargoed external OOS stream."""

    def __init__(self, settings: Settings, paths: AppPaths) -> None:
        self.settings = settings
        self.paths = paths

    def run(
        self,
        market: pd.DataFrame,
        *,
        source: str,
        dataset_hash: str,
        run_regime_study: bool = False,
    ) -> BacktestResult:
        from orizzonte_desk.ml import MetaModelRegistry

        if market.attrs.get("dataset_role") == "external_holdout":
            raise ValueError("Walk-forward não pode treinar no external_holdout")
        features = prepare_features(market, self.settings.strategy)
        windows = walk_forward_windows(
            features["timestamp"],
            training_months=self.settings.backtest.training_months,
            validation_months=self.settings.backtest.validation_months,
            test_months=self.settings.backtest.test_months,
            step_months=self.settings.backtest.step_months,
        )
        if not windows:
            raise ValueError("Histórico insuficiente para walk-forward de 18+3+3 meses")
        registry = MetaModelRegistry(
            self.paths,
            self.settings.research,
            self.settings.execution,
            settings=self.settings,
        )
        oos_pieces: list[pd.DataFrame] = []
        fold_manifest: list[dict[str, Any]] = []
        embargo = pd.Timedelta(hours=24)
        for index, window in enumerate(windows):
            fit_end = window["validation_end"] - embargo
            test_start = window["validation_end"] + embargo
            fit = features[
                (features["timestamp"] >= window["train_start"]) & (features["timestamp"] < fit_end)
            ]
            test = features[
                (features["timestamp"] >= test_start) & (features["timestamp"] < window["test_end"])
            ].copy()
            if test.empty:
                continue
            trained = registry.train(
                fit,
                seed=self.settings.backtest.random_seed + index,
                dataset_role="development",
                dataset_hashes=(dataset_hash,),
                config_fingerprint=stable_fingerprint(self.settings.model_dump(mode="json")),
            )
            bundle = joblib.load(trained.model_path)
            decision_policy = registry.load_decision_policy(trained.model_id)
            if decision_policy is None:
                raise RuntimeError("Fold walk-forward sem DecisionPolicy")
            candidates = test["signal_raw"] != 0
            test["ml_probability"] = 0.0
            if candidates.any():
                test.loc[candidates, "ml_probability"] = registry.predict(
                    test.loc[candidates], bundle
                )
            test["decision_threshold"] = decision_policy.probability_threshold
            test["decision_policy_id"] = decision_policy.policy_id
            test["decision_accepted"] = False
            if candidates.any():
                test.loc[candidates, "decision_accepted"] = decision_policy.apply(
                    test.loc[candidates, "ml_probability"]
                )
            test["signal"] = test["signal_raw"].where(test["decision_accepted"], 0)
            oos_pieces.append(test)
            fold_manifest.append(
                {
                    "fold": index + 1,
                    **{key: value.isoformat() for key, value in window.items()},
                    "fit_end_purged": fit_end.isoformat(),
                    "test_start_embargoed": test_start.isoformat(),
                    "model_id": trained.model_id,
                    "model_hash": trained.model_hash,
                    "validation_metrics": trained.metrics,
                    "decision_policy_id": decision_policy.policy_id,
                    "decision_trade_enabled": decision_policy.trade_enabled,
                    "decision_lcb_p05": decision_policy.expectancy_lcb_p05,
                    "oos_rows": len(test),
                }
            )
        if not oos_pieces:
            raise RuntimeError("Nenhum fold OOS foi produzido")
        oos = (
            pd.concat(oos_pieces, ignore_index=True)
            .sort_values(["timestamp", "symbol"])
            .drop_duplicates(["timestamp", "symbol"], keep="first")
        )
        oos.attrs.update(market.attrs)
        protocol_hash = stable_fingerprint(
            {
                "anchored": True,
                "purge_hours": 24,
                "embargo_hours": 24,
                "dataset_hash": dataset_hash,
                "folds": fold_manifest,
            }
        )
        result = EventBacktester(self.settings, self.paths).run(
            market,
            source=f"{source}-walkforward-oos",
            dataset_hash=dataset_hash,
            enriched_override=oos,
            evaluation_scope="training_protocol",
            protocol_hash=protocol_hash,
            run_regime_study=run_regime_study,
        )
        manifest_path = result.gate_path.parent / "walk-forward.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "anchored": True,
                    "purge_hours": 24,
                    "embargo_hours": 24,
                    "protocol_hash": protocol_hash,
                    "folds": fold_manifest,
                    "oos_start": pd.Timestamp(oos["timestamp"].min()).isoformat(),
                    "oos_end": pd.Timestamp(oos["timestamp"].max()).isoformat(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        result.artifacts["walk_forward"] = manifest_path
        return result
