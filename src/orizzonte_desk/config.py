from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from orizzonte_desk.constants import SYMBOLS
from orizzonte_desk.paths import AppPaths


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class AppConfig(FrozenModel):
    locale: str = "pt_BR"
    timezone: Literal["UTC"] = "UTC"
    host: Literal["127.0.0.1"] = "127.0.0.1"
    port: int = Field(default=8790, ge=1024, le=65535)
    minimum_free_gb: float = Field(default=20.0, ge=1.0)


class UniverseConfig(FrozenModel):
    symbols: tuple[str, ...] = SYMBOLS
    base_interval: Literal["1h"] = "1h"
    setup_interval: Literal["4h"] = "4h"
    regime_intervals: tuple[Literal["1d", "1w"], ...] = ("1d", "1w")

    @field_validator("symbols")
    @classmethod
    def fixed_universe(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(item.upper() for item in value)
        if normalized != SYMBOLS:
            raise ValueError(f"O universo é fixo e deve ser {SYMBOLS}.")
        return normalized


class RiskConfig(FrozenModel):
    leverage: Literal[10] = 10
    isolated_margin: Literal[True] = True
    risk_per_trade: float = Field(default=0.01, gt=0, le=0.01)
    aggregate_risk: float = Field(default=0.02, gt=0, le=0.02)
    max_positions: int = Field(default=2, ge=1, le=2)
    max_notional_multiple: float = Field(default=10.0, gt=0, le=10.0)
    correlation_threshold: float = Field(default=0.75, ge=0.5, le=0.95)
    daily_profit_lock: float = Field(default=0.01, gt=0)
    daily_loss_limit: float = Field(default=0.04, gt=0, le=0.04)
    max_drawdown_limit: float = Field(default=0.25, gt=0, le=0.25)
    margin_reserve: float = Field(default=0.20, ge=0.1, le=0.5)


class StrategyConfig(FrozenModel):
    fast_ema: int = Field(default=20, ge=5)
    slow_ema: int = Field(default=50, ge=20)
    donchian_window: int = Field(default=20, ge=10)
    atr_window: int = Field(default=14, ge=5)
    atr_stop_multiple: float = Field(default=1.5, ge=1.0)
    atr_trailing_multiple: float = Field(default=2.0, ge=1.0)
    partial_take_profit_r: float = Field(default=1.0, gt=0)
    partial_fraction: float = Field(default=0.5, gt=0, lt=1)
    max_holding_hours: int = Field(default=240, ge=24)
    ml_probability_threshold: float = Field(default=0.58, ge=0.5, le=0.9)

    @model_validator(mode="after")
    def validate_emas(self) -> StrategyConfig:
        if self.fast_ema >= self.slow_ema:
            raise ValueError("fast_ema deve ser menor que slow_ema")
        return self


class ExecutionConfig(FrozenModel):
    environment: Literal["paper", "testnet", "mainnet"] = "testnet"
    slippage_bps_btc_eth: float = Field(default=15.0, ge=0)
    slippage_bps_sol_xrp: float = Field(default=25.0, ge=0)
    taker_fee: float = Field(default=0.00045, ge=0)
    maker_fee: float = Field(default=0.00015, ge=0)
    dead_man_timeout_seconds: int = Field(default=30, ge=10)
    stale_data_seconds: int = Field(default=120, ge=30)


class BacktestConfig(FrozenModel):
    initial_capital: float = Field(default=10000.0, gt=0)
    start_date: str = "2021-01-01"
    training_months: int = Field(default=18, ge=12)
    validation_months: int = Field(default=3, ge=1)
    test_months: int = Field(default=3, ge=1)
    step_months: int = Field(default=3, ge=1)
    monte_carlo_samples: int = Field(default=2000, ge=100)
    random_seed: int = 42017


class Settings(FrozenModel):
    app: AppConfig
    universe: UniverseConfig
    risk: RiskConfig
    strategy: StrategyConfig
    execution: ExecutionConfig
    backtest: BacktestConfig

    @classmethod
    def load(cls, path: Path | None = None) -> Settings:
        config_path = path or AppPaths.discover().config
        with config_path.open("rb") as handle:
            return cls.model_validate(tomllib.load(handle))
