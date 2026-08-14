from __future__ import annotations

from typing import Any, Literal, cast

import pandas as pd

from orizzonte_desk.config import StrategyConfig
from orizzonte_desk.decision import DecisionPolicy
from orizzonte_desk.features import prepare_features
from orizzonte_desk.ml import MetaModelRegistry
from orizzonte_desk.models import Side, Signal


class SignalGenerator:
    def __init__(
        self,
        config: StrategyConfig,
        registry: MetaModelRegistry | None = None,
    ) -> None:
        self.config = config
        self.registry = registry

    def enrich(
        self,
        market: pd.DataFrame,
        *,
        require_promoted_model: bool = False,
        model_bundle: dict[str, Any] | None = None,
        model_id: str | None = None,
        decision_policy: DecisionPolicy | None = None,
    ) -> pd.DataFrame:
        features = prepare_features(market, self.config)
        candidates = features["signal_raw"] != 0
        features["ml_probability"] = 0.0
        features["decision_accepted"] = False
        features["decision_threshold"] = 0.0
        features["decision_policy_id"] = "none"
        loaded_bundle = model_bundle
        if loaded_bundle is None and self.registry is not None:
            loaded_bundle = self.registry.load_promoted()
        if candidates.any():
            if require_promoted_model and loaded_bundle is None:
                raise RuntimeError("Nenhum modelo promovido disponível")
            if self.registry is None or loaded_bundle is None:
                probabilities = features.loc[candidates, "setup_score"].to_numpy()
                accepted = pd.Series(True, index=features.index[candidates])
                policy = None
            else:
                probabilities = self.registry.predict(features.loc[candidates], loaded_bundle)
                policy = decision_policy or self.registry.load_decision_policy(model_id)
                if policy is None:
                    raise RuntimeError("Modelo ML sem DecisionPolicy vinculada")
                accepted = pd.Series(policy.apply(probabilities), index=features.index[candidates])
            features.loc[candidates, "ml_probability"] = probabilities
            features.loc[candidates, "decision_accepted"] = accepted
            features.loc[candidates, "decision_threshold"] = (
                policy.probability_threshold if policy is not None else 0.0
            )
            features.loc[candidates, "decision_policy_id"] = (
                policy.policy_id if policy is not None else "deterministic-no-ml"
            )
        features["decision_accepted"] = features["decision_accepted"].fillna(False).astype(bool)
        features["signal"] = features["signal_raw"].where(features["decision_accepted"], 0)
        features.attrs.update(market.attrs)
        return features

    def latest(self, market: pd.DataFrame, *, require_promoted_model: bool = False) -> list[Signal]:
        enriched = self.enrich(market, require_promoted_model=require_promoted_model)
        latest = enriched.sort_values("timestamp").groupby("symbol", as_index=False).tail(1)
        output: list[Signal] = []
        records = cast(list[dict[str, Any]], latest.to_dict(orient="records"))
        for row in records:
            if int(row["signal"]) == 0:
                continue
            regime: Literal["bull", "bear", "neutral"] = (
                "bull"
                if float(row["daily_trend"]) > 0
                else "bear"
                if float(row["daily_trend"]) < 0
                else "neutral"
            )
            reasons = (
                "regime semanal/diário alinhado",
                "setup 4h confirmado",
                "timing 1h confirmado",
                "meta-modelo aprovado",
            )
            output.append(
                Signal(
                    timestamp=pd.Timestamp(row["timestamp"]).to_pydatetime(),
                    symbol=str(row["symbol"]),
                    side=Side.LONG if float(row["signal"]) > 0 else Side.SHORT,
                    score=float(row["setup_score"]),
                    probability=float(row["ml_probability"]),
                    entry_reference=float(row["close"]),
                    stop_distance=float(row["stop_distance"]),
                    atr=float(row["atr_4h"]),
                    regime=regime,
                    reasons=reasons,
                )
            )
        return output
