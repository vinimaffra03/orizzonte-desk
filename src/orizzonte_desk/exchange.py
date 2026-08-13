from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any, Protocol, cast

from eth_account import Account
from hyperliquid.exchange import Exchange
from hyperliquid.info import Info
from hyperliquid.utils import constants
from hyperliquid.utils.types import Cloid

from orizzonte_desk.constants import SYMBOLS
from orizzonte_desk.models import Environment, Side, Signal


class ExchangeError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class AccountSnapshot:
    equity: float
    withdrawable: float
    open_orders: tuple[dict[str, Any], ...]
    positions: tuple[dict[str, Any], ...]
    mids: dict[str, float]


class TradingGateway(Protocol):
    def snapshot(self) -> AccountSnapshot: ...

    def place_entry_with_protection(
        self,
        signal: Signal,
        *,
        size: float,
        stop_price: float,
        take_profit_price: float,
        slippage: float,
    ) -> dict[str, Any]: ...

    def cancel_all(self) -> list[dict[str, Any]]: ...

    def flatten_all(self, *, slippage: float = 0.01) -> list[dict[str, Any]]: ...

    def schedule_dead_man(self, timeout_seconds: int) -> dict[str, Any]: ...


def client_order_id() -> Cloid:
    return Cloid.from_str(f"0x{uuid.uuid4().hex}")


class HyperliquidGateway:
    """Narrow trading-only adapter. It intentionally exposes no transfer or withdrawal API."""

    def __init__(
        self,
        *,
        secret_key: str,
        account_address: str,
        environment: Environment,
    ) -> None:
        if environment not in {Environment.TESTNET, Environment.MAINNET}:
            raise ValueError("HyperliquidGateway requer testnet ou mainnet")
        if not account_address.startswith("0x") or len(account_address) != 42:
            raise ValueError("Endereço principal inválido")
        base_url = (
            constants.MAINNET_API_URL
            if environment is Environment.MAINNET
            else constants.TESTNET_API_URL
        )
        wallet = Account.from_key(secret_key)
        self.environment = environment
        self.account_address = account_address.lower()
        self.info = Info(base_url, skip_ws=True, timeout=15)
        self.exchange = Exchange(
            wallet,
            base_url,
            account_address=self.account_address,
            timeout=15,
        )

    def snapshot(self) -> AccountSnapshot:
        state = self.info.user_state(self.account_address)
        orders = self.info.open_orders(self.account_address)
        positions = tuple(
            item["position"]
            for item in state.get("assetPositions", [])
            if abs(float(item["position"].get("szi", 0))) > 0
        )
        summary = state.get("marginSummary", {})
        mids = {key: float(value) for key, value in self.info.all_mids().items() if key in SYMBOLS}
        return AccountSnapshot(
            equity=float(summary.get("accountValue", 0)),
            withdrawable=float(state.get("withdrawable", 0)),
            open_orders=tuple(orders),
            positions=positions,
            mids=mids,
        )

    def place_entry_with_protection(
        self,
        signal: Signal,
        *,
        size: float,
        stop_price: float,
        take_profit_price: float,
        slippage: float,
    ) -> dict[str, Any]:
        if signal.symbol not in SYMBOLS:
            raise ExchangeError(f"Ativo não permitido: {signal.symbol}")
        if size <= 0:
            raise ExchangeError("Tamanho deve ser positivo")
        leverage_response = self.exchange.update_leverage(10, signal.symbol, is_cross=False)
        self._assert_ok(leverage_response, "configuração de margem isolada 10x")
        is_buy = signal.side is Side.LONG
        entry = self.exchange.market_open(
            signal.symbol,
            is_buy,
            size,
            slippage=slippage,
            cloid=client_order_id(),
        )
        self._assert_ok(entry, "ordem de entrada")
        protections: list[dict[str, Any]] = []
        try:
            stop = self.exchange.order(
                signal.symbol,
                not is_buy,
                size,
                stop_price,
                {"trigger": {"triggerPx": stop_price, "isMarket": True, "tpsl": "sl"}},
                reduce_only=True,
                cloid=client_order_id(),
            )
            self._assert_ok(stop, "stop loss reduce-only")
            protections.append(stop)
            take_profit = self.exchange.order(
                signal.symbol,
                not is_buy,
                size,
                take_profit_price,
                {
                    "trigger": {
                        "triggerPx": take_profit_price,
                        "isMarket": True,
                        "tpsl": "tp",
                    }
                },
                reduce_only=True,
                cloid=client_order_id(),
            )
            self._assert_ok(take_profit, "take profit reduce-only")
            protections.append(take_profit)
        except Exception as exc:
            self.exchange.market_close(signal.symbol, sz=size, slippage=max(slippage, 0.02))
            raise ExchangeError("Proteção nativa falhou; posição fechada imediatamente") from exc
        return {
            "leverage": leverage_response,
            "entry": entry,
            "protections": protections,
            "isolated": True,
            "leverage_value": 10,
        }

    def cancel_all(self) -> list[dict[str, Any]]:
        snapshot = self.snapshot()
        responses: list[dict[str, Any]] = []
        for order in snapshot.open_orders:
            response = self.exchange.cancel(str(order["coin"]), int(order["oid"]))
            self._assert_ok(response, "cancelamento")
            responses.append(response)
        return responses

    def flatten_all(self, *, slippage: float = 0.01) -> list[dict[str, Any]]:
        self.cancel_all()
        responses: list[dict[str, Any]] = []
        for position in self.snapshot().positions:
            symbol = str(position["coin"])
            if symbol not in SYMBOLS:
                raise ExchangeError(f"Posição manual fora do universo detectada: {symbol}")
            response = self.exchange.market_close(symbol, slippage=slippage)
            self._assert_ok(response, "flatten")
            responses.append(response)
        return responses

    def schedule_dead_man(self, timeout_seconds: int) -> dict[str, Any]:
        if timeout_seconds < 10:
            raise ValueError("Dead man's switch deve ser de pelo menos 10 segundos")
        cancel_at = int((time.time() + timeout_seconds) * 1000)
        response = self.exchange.schedule_cancel(cancel_at)
        self._assert_ok(response, "dead man's switch")
        return cast(dict[str, Any], response)

    @staticmethod
    def _assert_ok(response: dict[str, Any], action: str) -> None:
        if response.get("status") != "ok":
            raise ExchangeError(f"Falha em {action}: {response}")
        statuses = response.get("response", {}).get("data", {}).get("statuses", [])
        errors = [
            status.get("error")
            for status in statuses
            if isinstance(status, dict) and "error" in status
        ]
        if errors:
            raise ExchangeError(f"Falha em {action}: {errors}")


class PaperGateway:
    def __init__(self, initial_equity: float = 10_000.0) -> None:
        self.equity = initial_equity
        self.orders: list[dict[str, Any]] = []
        self.positions: list[dict[str, Any]] = []
        self.mids: dict[str, float] = {}

    def snapshot(self) -> AccountSnapshot:
        return AccountSnapshot(
            equity=self.equity,
            withdrawable=self.equity,
            open_orders=tuple(self.orders),
            positions=tuple(self.positions),
            mids=dict(self.mids),
        )

    def place_entry_with_protection(
        self,
        signal: Signal,
        *,
        size: float,
        stop_price: float,
        take_profit_price: float,
        slippage: float,
    ) -> dict[str, Any]:
        position = {
            "coin": signal.symbol,
            "szi": str(size * signal.side.sign),
            "entryPx": str(signal.entry_reference),
            "leverage": {"type": "isolated", "value": 10},
        }
        self.positions.append(position)
        protections = [
            {"coin": signal.symbol, "kind": "sl", "price": stop_price, "reduceOnly": True},
            {"coin": signal.symbol, "kind": "tp", "price": take_profit_price, "reduceOnly": True},
        ]
        self.orders.extend(protections)
        return {"status": "ok", "position": position, "protections": protections}

    def cancel_all(self) -> list[dict[str, Any]]:
        canceled = list(self.orders)
        self.orders.clear()
        return canceled

    def flatten_all(self, *, slippage: float = 0.01) -> list[dict[str, Any]]:
        closed = list(self.positions)
        self.positions.clear()
        self.orders.clear()
        return closed

    def schedule_dead_man(self, timeout_seconds: int) -> dict[str, Any]:
        return {"status": "ok", "paper": True, "timeout_seconds": timeout_seconds}
