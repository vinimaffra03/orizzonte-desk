from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from orizzonte_desk.constants import SYMBOLS

REQUIRED_TESTNET_SCENARIOS = (
    "partial_fill",
    "native_sl_tp",
    "duplicate_ws",
    "timeout_after_accept",
    "clock_drift_stale",
    "protection_failure_flatten_lock",
    "dead_man",
    "empty_account",
)


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


class TestnetCertificate(StrictModel):
    schema_version: Literal[2] = 2
    certificate_id: str
    environment: Literal["testnet"] = "testnet"
    release_id: str
    model_hash: str
    gates_hash: str
    account_address: str
    wallet_address: str
    evidence_hashes: tuple[str, ...]
    required_scenarios: tuple[str, ...]
    scenario_results: dict[str, bool]
    scenario_hashes: dict[str, str]
    chaos_report_hash: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def validate_chaos_gate(self) -> TestnetCertificate:
        required = tuple(REQUIRED_TESTNET_SCENARIOS)
        if self.required_scenarios != required:
            raise ValueError("Testnet certificate has an incomplete required scenario set")
        if set(self.scenario_results) != set(required) or not all(self.scenario_results.values()):
            raise ValueError("Every required testnet scenario must pass")
        if set(self.scenario_hashes) != set(required) or any(
            not _is_sha256(value) for value in self.scenario_hashes.values()
        ):
            raise ValueError("Every required testnet scenario needs a SHA-256 evidence hash")
        if not _is_sha256(self.chaos_report_hash):
            raise ValueError("Testnet chaos report needs a SHA-256 hash")
        return self

    @classmethod
    def build(
        cls,
        *,
        release_id: str,
        model_hash: str,
        gates_hash: str,
        account_address: str,
        wallet_address: str,
        evidence_hashes: tuple[str, ...],
        required_scenarios: tuple[str, ...],
        scenario_results: dict[str, bool],
        scenario_hashes: dict[str, str],
        chaos_report_hash: str,
        created_at: datetime | None = None,
    ) -> TestnetCertificate:
        timestamp = created_at or datetime.now(UTC)
        normalized_evidence = tuple(sorted(set(evidence_hashes)))
        payload: dict[str, Any] = {
            "schema_version": 2,
            "environment": "testnet",
            "release_id": release_id,
            "model_hash": model_hash,
            "gates_hash": gates_hash,
            "account_address": account_address.lower(),
            "wallet_address": wallet_address.lower(),
            "evidence_hashes": normalized_evidence,
            "required_scenarios": required_scenarios,
            "scenario_results": scenario_results,
            "scenario_hashes": scenario_hashes,
            "chaos_report_hash": chaos_report_hash,
            "created_at": timestamp.isoformat().replace("+00:00", "Z"),
        }
        certificate_id = f"testnet-{_content_hash(payload)}"
        return cls(
            certificate_id=certificate_id,
            release_id=release_id,
            model_hash=model_hash,
            gates_hash=gates_hash,
            account_address=account_address.lower(),
            wallet_address=wallet_address.lower(),
            evidence_hashes=normalized_evidence,
            required_scenarios=required_scenarios,
            scenario_results=scenario_results,
            scenario_hashes=scenario_hashes,
            chaos_report_hash=chaos_report_hash,
            created_at=timestamp,
        )

    def verify_content_address(self) -> bool:
        payload = self.model_dump(mode="json", exclude={"certificate_id"})
        payload["evidence_hashes"] = sorted(payload["evidence_hashes"])
        return self.certificate_id == f"testnet-{_content_hash(payload)}"


class MainnetAuthorization(StrictModel):
    schema_version: Literal[1] = 1
    authorization_id: str
    environment: Literal["mainnet"] = "mainnet"
    release_id: str
    certificate_id: str
    model_hash: str
    gates_hash: str
    git_commit: str
    config_sha256: str
    config_fingerprint: str
    code_hash: str
    account_address: str
    wallet_address: str
    budget_usdc: float = Field(gt=0, le=500)
    issued_at: datetime
    expires_at: datetime


def _content_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value.lower())
