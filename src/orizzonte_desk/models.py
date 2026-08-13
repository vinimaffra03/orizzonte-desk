from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from orizzonte_desk.constants import SYMBOLS


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Environment(StrEnum):
    PAPER = "paper"
    TESTNET = "testnet"
    MAINNET = "mainnet"


class Side(StrEnum):
    LONG = "long"
    SHORT = "short"

    @property
    def sign(self) -> int:
        return 1 if self is Side.LONG else -1


class AgentStatus(StrEnum):
    DISARMED = "disarmed"
    ARMED = "armed"
    RUNNING = "running"
    PAUSED = "paused"
    LOCKED = "locked"


class Candle(StrictModel):
    timestamp: datetime
    symbol: str
    interval: str = "1h"
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: float = Field(ge=0)
    funding_rate: float = 0.0

    @field_validator("symbol")
    @classmethod
    def symbol_is_allowed(cls, value: str) -> str:
        normalized = value.upper()
        if normalized not in SYMBOLS:
            raise ValueError(f"Símbolo fora do universo: {normalized}")
        return normalized


class Signal(StrictModel):
    timestamp: datetime
    symbol: str
    side: Side
    score: float = Field(ge=0, le=1)
    probability: float = Field(ge=0, le=1)
    entry_reference: float = Field(gt=0)
    stop_distance: float = Field(gt=0)
    atr: float = Field(gt=0)
    regime: Literal["bull", "bear", "neutral"]
    reasons: tuple[str, ...] = ()

    @field_validator("symbol")
    @classmethod
    def signal_symbol_is_allowed(cls, value: str) -> str:
        normalized = value.upper()
        if normalized not in SYMBOLS:
            raise ValueError(f"Símbolo fora do universo: {normalized}")
        return normalized


class Position(StrictModel):
    symbol: str
    side: Side
    size: float = Field(gt=0)
    entry_price: float = Field(gt=0)
    stop_price: float = Field(gt=0)
    take_profit_price: float = Field(gt=0)
    opened_at: datetime
    initial_size: float = Field(gt=0)
    initial_risk_usdc: float = Field(gt=0)
    partial_taken: bool = False
    trailing_price: float | None = None
    order_id: str | None = None

    @property
    def notional(self) -> float:
        return self.size * self.entry_price


class Trade(StrictModel):
    symbol: str
    side: Side
    opened_at: datetime
    closed_at: datetime
    entry_price: float
    exit_price: float
    size: float
    gross_pnl: float
    net_pnl: float
    fees: float
    funding: float
    slippage: float
    exit_reason: str
    mae: float = 0.0
    mfe: float = 0.0


class RiskSnapshot(StrictModel):
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    budget: float = Field(gt=0)
    equity: float = Field(gt=0)
    day_start_equity: float = Field(gt=0)
    high_water_mark: float = Field(gt=0)
    open_risk: float = Field(ge=0)
    open_notional: float = Field(ge=0)
    positions_count: int = Field(ge=0)
    profit_locked: bool = False
    loss_locked: bool = False
    drawdown_locked: bool = False


class GateResult(StrictModel):
    passed: bool
    evaluated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    checks: dict[str, bool]
    metrics: dict[str, float]
    reasons: tuple[str, ...] = ()
    dataset_hashes: tuple[str, ...] = ()
    model_hash: str | None = None


class AgentState(StrictModel):
    status: AgentStatus = AgentStatus.DISARMED
    environment: Environment = Environment.PAPER
    budget_usdc: float | None = None
    account_address: str | None = None
    armed_at: datetime | None = None
    last_heartbeat: datetime | None = None
    profit_locked: bool = False
    loss_locked: bool = False
    drawdown_locked: bool = False
    metadata: dict[str, Any] = {}
