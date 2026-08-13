from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import ROUND_DOWN, Decimal

from orizzonte_desk.config import RiskConfig
from orizzonte_desk.constants import ALTCOINS, SYMBOLS
from orizzonte_desk.models import Position, RiskSnapshot, Signal


class RiskViolation(RuntimeError):
    """Raised when an order would violate a hard risk invariant."""


@dataclass(frozen=True, slots=True)
class OrderSizing:
    size: float
    notional: float
    margin_required: float
    risk_usdc: float
    stop_price: float
    take_profit_price: float


def floor_to_increment(value: float, increment: float) -> float:
    if increment <= 0:
        raise ValueError("Incremento deve ser positivo")
    units = (Decimal(str(value)) / Decimal(str(increment))).to_integral_value(rounding=ROUND_DOWN)
    return float(units * Decimal(str(increment)))


class RiskManager:
    def __init__(self, config: RiskConfig) -> None:
        self.config = config

    def size_order(
        self,
        signal: Signal,
        *,
        budget: float,
        equity: float,
        open_notional: float,
        size_increment: float = 1e-5,
    ) -> OrderSizing:
        if signal.symbol not in SYMBOLS:
            raise RiskViolation(f"Ativo não permitido: {signal.symbol}")
        if budget <= 0 or equity <= 0:
            raise RiskViolation("Orçamento e equity devem ser positivos")
        capital_base = min(budget, equity)
        risk_usdc = capital_base * self.config.risk_per_trade
        raw_size = risk_usdc / signal.stop_distance
        size = floor_to_increment(raw_size, size_increment)
        if size <= 0:
            raise RiskViolation("Tamanho calculado abaixo da precisão mínima")
        notional = size * signal.entry_reference
        remaining_cap = budget * self.config.max_notional_multiple - open_notional
        if remaining_cap <= 0:
            raise RiskViolation("Teto nocional agregado esgotado")
        if notional > remaining_cap:
            size = floor_to_increment(remaining_cap / signal.entry_reference, size_increment)
            notional = size * signal.entry_reference
            risk_usdc = size * signal.stop_distance
        margin_required = notional / self.config.leverage
        available_margin = equity * (1 - self.config.margin_reserve)
        if margin_required > available_margin:
            size = floor_to_increment(
                available_margin * self.config.leverage / signal.entry_reference,
                size_increment,
            )
            notional = size * signal.entry_reference
            margin_required = notional / self.config.leverage
            risk_usdc = size * signal.stop_distance
        if size <= 0 or risk_usdc <= 0:
            raise RiskViolation("Margem insuficiente para uma ordem válida")
        stop_price = signal.entry_reference - signal.side.sign * signal.stop_distance
        take_profit_price = signal.entry_reference + signal.side.sign * signal.stop_distance
        if stop_price <= 0 or take_profit_price <= 0:
            raise RiskViolation("Stop ou alvo resultou em preço inválido")
        return OrderSizing(
            size=size,
            notional=notional,
            margin_required=margin_required,
            risk_usdc=risk_usdc,
            stop_price=stop_price,
            take_profit_price=take_profit_price,
        )

    def evaluate_locks(self, snapshot: RiskSnapshot) -> RiskSnapshot:
        daily_return = snapshot.equity / snapshot.day_start_equity - 1
        drawdown = snapshot.equity / snapshot.high_water_mark - 1
        return snapshot.model_copy(
            update={
                "profit_locked": snapshot.profit_locked
                or daily_return >= self.config.daily_profit_lock,
                "loss_locked": snapshot.loss_locked
                or daily_return <= -self.config.daily_loss_limit,
                "drawdown_locked": snapshot.drawdown_locked
                or drawdown <= -self.config.max_drawdown_limit,
            }
        )

    def assert_new_entry_allowed(
        self,
        signal: Signal,
        snapshot: RiskSnapshot,
        positions: Iterable[Position],
        correlations: dict[tuple[str, str], float] | None = None,
    ) -> None:
        current = list(positions)
        evaluated = self.evaluate_locks(snapshot)
        if evaluated.profit_locked:
            raise RiskViolation("Meta diária atingida; novas entradas bloqueadas até 00:00 UTC")
        if evaluated.loss_locked:
            raise RiskViolation("Stop diário atingido; sessão bloqueada")
        if evaluated.drawdown_locked:
            raise RiskViolation("Kill switch de drawdown ativado")
        if len(current) >= self.config.max_positions:
            raise RiskViolation("Máximo de posições simultâneas atingido")
        if any(item.symbol == signal.symbol for item in current):
            raise RiskViolation("Aumento de posição e averaging down são proibidos")
        total_risk = sum(item.initial_risk_usdc for item in current)
        if total_risk >= snapshot.budget * self.config.aggregate_risk:
            raise RiskViolation("Limite de risco agregado atingido")
        correlations = correlations or {}
        for item in current:
            pair = (
                min(item.symbol, signal.symbol),
                max(item.symbol, signal.symbol),
            )
            correlation = abs(correlations.get(pair, 0.0))
            if (
                correlation > self.config.correlation_threshold
                and item.side is signal.side
                and item.symbol in ALTCOINS
                and signal.symbol in ALTCOINS
            ):
                raise RiskViolation(
                    f"Exposição altcoin redundante: correlação {correlation:.2f} em {pair}"
                )

    @staticmethod
    def utc_day_changed(previous: datetime, current: datetime) -> bool:
        return previous.astimezone(UTC).date() != current.astimezone(UTC).date()
