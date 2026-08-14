from __future__ import annotations

import email.utils
import hashlib
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import ROUND_DOWN, Decimal
from typing import Any, ClassVar, Protocol, cast

from eth_account import Account
from hyperliquid.exchange import Exchange
from hyperliquid.info import Info
from hyperliquid.utils import constants
from hyperliquid.utils.types import Cloid

from orizzonte_desk.constants import SYMBOLS
from orizzonte_desk.models import Environment, Side, Signal
from orizzonte_desk.storage import StateStore


class ExchangeError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class AccountSnapshot:
    equity: float
    withdrawable: float
    open_orders: tuple[dict[str, Any], ...]
    positions: tuple[dict[str, Any], ...]
    mids: dict[str, float]


@dataclass(frozen=True, slots=True)
class AssetMetadata:
    symbol: str
    size_decimals: int
    size_increment: float
    max_price_decimals: int

    def normalize_size(self, value: float) -> float:
        return _floor(value, self.size_increment)

    def normalize_price(self, value: float) -> float:
        if value <= 0:
            raise ExchangeError("Preço deve ser positivo")
        # Hyperliquid prices have at most 5 significant figures and 6-szDecimals decimals.
        significant = float(f"{value:.5g}")
        quantum = Decimal(1).scaleb(-self.max_price_decimals)
        return float(Decimal(str(significant)).quantize(quantum, rounding=ROUND_DOWN))


class TradingGateway(Protocol):
    def snapshot(self) -> AccountSnapshot: ...

    def asset_metadata(self, symbol: str) -> AssetMetadata: ...

    def reconcile(self) -> AccountSnapshot: ...

    def place_entry_with_protection(
        self,
        signal: Signal,
        *,
        size: float,
        stop_price: float,
        take_profit_price: float,
        slippage: float,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]: ...

    def reduce_position(
        self, symbol: str, *, size: float, slippage: float, reason: str
    ) -> dict[str, Any]: ...

    def replace_protection(
        self,
        symbol: str,
        *,
        side: Side,
        size: float,
        stop_price: float,
        take_profit_price: float,
    ) -> list[dict[str, Any]]: ...

    def cancel_all(self) -> list[dict[str, Any]]: ...

    def flatten_all(self, *, slippage: float = 0.01) -> list[dict[str, Any]]: ...

    def schedule_dead_man(self, timeout_seconds: int) -> dict[str, Any]: ...


def _floor(value: float, increment: float) -> float:
    units = (Decimal(str(value)) / Decimal(str(increment))).to_integral_value(rounding=ROUND_DOWN)
    return float(units * Decimal(str(increment)))


def client_order_id() -> Cloid:
    return Cloid.from_str(f"0x{uuid.uuid4().hex}")


def cloid_for_key(key: str) -> Cloid:
    return Cloid.from_str(f"0x{hashlib.sha256(key.encode('utf-8')).hexdigest()[:32]}")


def _cloid_text(value: Cloid) -> str:
    return str(value)


class HyperliquidGateway:
    """Trading-only adapter. Transfer, withdrawal and account mutation are not exposed."""

    def __init__(
        self,
        *,
        secret_key: str,
        account_address: str,
        environment: Environment,
        store: StateStore | None = None,
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
        self.wallet_address = wallet.address.lower()
        self.info = Info(base_url, skip_ws=True, timeout=15)
        self.exchange = Exchange(
            wallet,
            base_url,
            account_address=self.account_address,
            timeout=15,
        )
        self.store = store
        self._metadata: dict[str, AssetMetadata] = {}

    def asset_metadata(self, symbol: str) -> AssetMetadata:
        normalized = symbol.upper()
        if normalized not in SYMBOLS:
            raise ExchangeError(f"Ativo não permitido: {normalized}")
        if not self._metadata:
            universe = self.info.meta().get("universe", [])
            for item in universe:
                name = str(item.get("name", "")).upper()
                if name not in SYMBOLS:
                    continue
                size_decimals = int(item["szDecimals"])
                self._metadata[name] = AssetMetadata(
                    symbol=name,
                    size_decimals=size_decimals,
                    size_increment=10.0 ** (-size_decimals),
                    max_price_decimals=max(0, 6 - size_decimals),
                )
        if normalized not in self._metadata:
            raise ExchangeError(f"Metadados indisponíveis para {normalized}")
        return self._metadata[normalized]

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
        snapshot = AccountSnapshot(
            equity=float(summary.get("accountValue", 0)),
            withdrawable=float(state.get("withdrawable", 0)),
            open_orders=tuple(orders),
            positions=positions,
            mids=mids,
        )
        if self.store:
            self._persist_snapshot(snapshot)
        return snapshot

    def reconcile(self) -> AccountSnapshot:
        """REST is authoritative after reconnect/restart; fill insertion is idempotent."""
        snapshot = self.snapshot()
        if self.store:
            for fill in self.info.user_fills(self.account_address):
                self._persist_fill(cast(dict[str, Any], fill))
            self.store.set(
                "exchange_reconciliation",
                {
                    "environment": self.environment.value,
                    "completed_at": datetime.now(UTC).isoformat(),
                    "positions": len(snapshot.positions),
                    "orders": len(snapshot.open_orders),
                },
            )
        return snapshot

    def preflight(self, *, require_empty: bool = True) -> dict[str, Any]:
        """Read-only venue/account validation. It never signs or submits an action."""
        if self.wallet_address == self.account_address:
            raise ExchangeError("Use uma API wallet exclusiva; não use a chave da conta principal")
        role = self.info.user_role(self.account_address)
        role_name = str(role.get("role", role.get("type", ""))).lower()
        if role_name and role_name not in {"user", "master"}:
            raise ExchangeError(f"Role da conta principal não suportada: {role_name}")
        agents = self.info.extra_agents(self.account_address)
        registered = _addresses(agents)
        if registered and self.wallet_address not in registered:
            raise ExchangeError("API wallet não está registrada como agente da conta principal")
        snapshot = self.snapshot()
        if require_empty and (snapshot.positions or snapshot.open_orders):
            raise ExchangeError("Conta possui posições ou ordens preexistentes")
        for symbol in SYMBOLS:
            self.asset_metadata(symbol)
        fees = self.info.user_fees(self.account_address)
        skew = self._clock_skew_seconds()
        if skew > 5:
            raise ExchangeError(f"Clock drift excessivo: {skew:.2f}s")
        return {
            "environment": self.environment.value,
            "account_address": self.account_address,
            "wallet_address": self.wallet_address,
            "account_role": role_name or "unknown",
            "clock_skew_seconds": skew,
            "equity": snapshot.equity,
            "empty": not snapshot.positions and not snapshot.open_orders,
            "fees": fees,
            "symbols": list(SYMBOLS),
        }

    def place_entry_with_protection(
        self,
        signal: Signal,
        *,
        size: float,
        stop_price: float,
        take_profit_price: float,
        slippage: float,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        metadata = self.asset_metadata(signal.symbol)
        normalized_size = metadata.normalize_size(size)
        normalized_stop = metadata.normalize_price(stop_price)
        normalized_take_profit = metadata.normalize_price(take_profit_price)
        if normalized_size <= 0:
            raise ExchangeError("Tamanho abaixo da precisão mínima")
        is_buy = signal.side is Side.LONG
        entry_cloid = cloid_for_key(idempotency_key) if idempotency_key else client_order_id()
        entry_cloid_text = _cloid_text(entry_cloid)
        if self.store and self.store.order(entry_cloid_text):
            tracked = next(
                (
                    item["payload"]
                    for item in self.store.positions()
                    if item["symbol"] == signal.symbol
                ),
                None,
            )
            recovered = self.reconcile()
            position = next(
                (item for item in recovered.positions if item.get("coin") == signal.symbol), None
            )
            owned = isinstance(tracked, dict) and tracked.get("entryCloid") == entry_cloid_text
            if position and not owned:
                raise ExchangeError(
                    "Posição do símbolo não pertence ao CLOID idempotente; retry recusado"
                )
            if position and owned:
                recovered_size = metadata.normalize_size(abs(float(position.get("szi", 0))))
                protections = [
                    item
                    for item in recovered.open_orders
                    if item.get("coin") == signal.symbol
                    and item.get("reduceOnly")
                    and float(item.get("sz", item.get("size", recovered_size))) >= recovered_size
                ]
                if len(protections) < 2:
                    protections = self.replace_protection(
                        signal.symbol,
                        side=signal.side,
                        size=recovered_size,
                        stop_price=normalized_stop,
                        take_profit_price=normalized_take_profit,
                    )
                return {
                    "status": "ok",
                    "idempotent_replay": True,
                    "entry_cloid": entry_cloid_text,
                    "filled_size": recovered_size,
                    "average_price": float(position.get("entryPx", signal.entry_reference)),
                    "protections": protections,
                    "isolated": True,
                    "leverage_value": 10,
                }
        leverage_response = self.exchange.update_leverage(10, signal.symbol, is_cross=False)
        self._assert_ok(leverage_response, "configuração de margem isolada 10x")
        if self.store:
            self.store.upsert_order(
                entry_cloid_text,
                symbol=signal.symbol,
                side=signal.side.value,
                status="submitting",
                payload={"requested_size": normalized_size, "kind": "entry"},
            )
        entry = self.exchange.market_open(
            signal.symbol,
            is_buy,
            normalized_size,
            slippage=slippage,
            cloid=entry_cloid,
        )
        self._assert_ok(entry, "ordem de entrada")
        filled_size, average_price, exchange_order_id = self._entry_fill(entry)
        if filled_size <= 0:
            # A market response can omit totals; the authoritative position snapshot is safe here
            # because increasing/averaging a pre-existing symbol is forbidden by the engine.
            current = next(
                (item for item in self.snapshot().positions if item.get("coin") == signal.symbol),
                None,
            )
            filled_size = abs(float(current.get("szi", 0))) if current else 0.0
            average_price = float(current.get("entryPx", 0)) if current else 0.0
        filled_size = metadata.normalize_size(min(filled_size, normalized_size))
        if filled_size <= 0:
            if self.store:
                self.store.upsert_order(
                    entry_cloid_text,
                    symbol=signal.symbol,
                    side=signal.side.value,
                    status="unfilled",
                    payload=cast(dict[str, Any], entry),
                    exchange_order_id=exchange_order_id,
                )
            raise ExchangeError("Entrada não confirmou quantidade preenchida; proteção não enviada")
        if self.store:
            self.store.upsert_order(
                entry_cloid_text,
                symbol=signal.symbol,
                side=signal.side.value,
                status="filled" if filled_size >= normalized_size else "partially_filled",
                payload={"response": entry, "filled_size": filled_size},
                exchange_order_id=exchange_order_id,
            )
            self.store.record_fill(
                f"entry:{entry_cloid_text}:{filled_size}",
                symbol=signal.symbol,
                size=filled_size,
                price=average_price or signal.entry_reference,
                payload={"synthetic_from_order_response": True, "response": entry},
                client_order_id=entry_cloid_text,
                exchange_order_id=exchange_order_id,
            )
        try:
            protections = self._place_protections(
                signal.symbol,
                side=signal.side,
                size=filled_size,
                stop_price=normalized_stop,
                take_profit_price=normalized_take_profit,
            )
        except Exception as exc:
            if self.store:
                self.store.latch_lock(
                    "unprotected_position",
                    reason="Falha ao instalar proteção nativa",
                    payload={
                        "symbol": signal.symbol,
                        "filled_size": filled_size,
                        "protection_error": str(exc),
                        "emergency_close": "pending",
                    },
                )
            try:
                close_response = self.exchange.market_close(
                    signal.symbol, sz=filled_size, slippage=max(slippage, 0.02)
                )
                self._assert_ok(cast(dict[str, Any], close_response), "fechamento emergencial")
                residual = next(
                    (
                        item
                        for item in self.reconcile().positions
                        if item.get("coin") == signal.symbol
                    ),
                    None,
                )
            except Exception as close_exc:
                raise ExchangeError(
                    "Proteção falhou e o fechamento emergencial não foi confirmado; "
                    "lock persistente acionado"
                ) from close_exc
            if residual:
                raise ExchangeError(
                    "Proteção falhou e fechamento emergencial deixou posição residual"
                ) from exc
            raise ExchangeError("Proteção nativa falhou; posição fechada imediatamente") from exc
        return {
            "leverage": leverage_response,
            "entry": entry,
            "entry_cloid": entry_cloid_text,
            "filled_size": filled_size,
            "average_price": average_price,
            "protections": protections,
            "isolated": True,
            "leverage_value": 10,
        }

    def reduce_position(
        self, symbol: str, *, size: float, slippage: float, reason: str
    ) -> dict[str, Any]:
        metadata = self.asset_metadata(symbol)
        normalized = metadata.normalize_size(size)
        if normalized <= 0:
            raise ExchangeError("Redução abaixo da precisão mínima")
        response = self.exchange.market_close(symbol, sz=normalized, slippage=slippage)
        self._assert_ok(response, f"redução ({reason})")
        return cast(dict[str, Any], response)

    def replace_protection(
        self,
        symbol: str,
        *,
        side: Side,
        size: float,
        stop_price: float,
        take_profit_price: float,
    ) -> list[dict[str, Any]]:
        old_orders = [
            order
            for order in self.snapshot().open_orders
            if str(order.get("coin")) == symbol and bool(order.get("reduceOnly", False))
        ]
        metadata = self.asset_metadata(symbol)
        # Keep the prior SL/TP live until the complete replacement pair is acknowledged. If
        # either new order fails, the old native protections remain in place.
        protections = self._place_protections(
            symbol,
            side=side,
            size=metadata.normalize_size(size),
            stop_price=metadata.normalize_price(stop_price),
            take_profit_price=metadata.normalize_price(take_profit_price),
        )
        for order in old_orders:
            try:
                response = self.exchange.cancel(symbol, int(order["oid"]))
                self._assert_ok(response, "cancelamento de proteção antiga")
            except Exception as exc:
                # Duplicate reduce-only protection is safer than a naked position. Reconciliation
                # will observe it and a later maintenance cycle can retry the cancellation.
                if self.store:
                    self.store.event(
                        "protection",
                        "Proteção antiga permaneceu após substituição segura",
                        level="WARNING",
                        payload={"symbol": symbol, "oid": order.get("oid"), "error": str(exc)},
                    )
        return protections

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
        if self.store:
            self.store.replace_positions([])
        return responses

    def schedule_dead_man(self, timeout_seconds: int) -> dict[str, Any]:
        if timeout_seconds < 10:
            raise ValueError("Dead man's switch deve ser de pelo menos 10 segundos")
        cancel_at = int((time.time() + timeout_seconds) * 1000)
        response = self.exchange.schedule_cancel(cancel_at)
        self._assert_ok(response, "dead man's switch")
        return cast(dict[str, Any], response)

    def _place_protections(
        self,
        symbol: str,
        *,
        side: Side,
        size: float,
        stop_price: float,
        take_profit_price: float,
    ) -> list[dict[str, Any]]:
        if size <= 0:
            raise ExchangeError("Proteção requer quantidade preenchida positiva")
        is_buy = side is Side.LONG
        definitions = (("sl", stop_price), ("tp", take_profit_price))
        protections: list[dict[str, Any]] = []
        for kind, price in definitions:
            cloid = client_order_id()
            response = self.exchange.order(
                symbol,
                not is_buy,
                size,
                price,
                {"trigger": {"triggerPx": price, "isMarket": True, "tpsl": kind}},
                reduce_only=True,
                cloid=cloid,
            )
            self._assert_ok(response, f"{kind} reduce-only")
            protections.append(cast(dict[str, Any], response))
            if self.store:
                self.store.upsert_order(
                    _cloid_text(cloid),
                    symbol=symbol,
                    side=(Side.SHORT if is_buy else Side.LONG).value,
                    status="open",
                    payload={"kind": kind, "size": size, "price": price, "response": response},
                )
        return protections

    def _persist_snapshot(self, snapshot: AccountSnapshot) -> None:
        assert self.store is not None
        self.store.replace_positions(list(snapshot.positions))
        for order in snapshot.open_orders:
            cloid = str(order.get("cloid") or f"oid:{order.get('oid')}")
            side = str(order.get("side") or order.get("dir") or "unknown").lower()
            self.store.upsert_order(
                cloid,
                symbol=str(order.get("coin", "")),
                side=side,
                status="open",
                payload=order,
                exchange_order_id=str(order["oid"]) if order.get("oid") is not None else None,
            )

    def _persist_fill(self, fill: dict[str, Any]) -> bool:
        assert self.store is not None
        fill_id = str(
            fill.get("tid")
            or fill.get("hash")
            or f"{fill.get('oid')}:{fill.get('time')}:{fill.get('sz')}:{fill.get('px')}"
        )
        return self.store.record_fill(
            fill_id,
            symbol=str(fill.get("coin", "")),
            size=abs(float(fill.get("sz", 0))),
            price=float(fill.get("px", 0)),
            fee=abs(float(fill.get("fee", 0))),
            payload=fill,
            client_order_id=str(fill["cloid"]) if fill.get("cloid") else None,
            exchange_order_id=str(fill["oid"]) if fill.get("oid") is not None else None,
            filled_at=_timestamp_text(fill.get("time")),
        )

    @staticmethod
    def _entry_fill(response: dict[str, Any]) -> tuple[float, float, str | None]:
        statuses = response.get("response", {}).get("data", {}).get("statuses", [])
        total = 0.0
        weighted = 0.0
        oid: str | None = None
        for status in statuses:
            if not isinstance(status, dict) or not isinstance(status.get("filled"), dict):
                continue
            fill = status["filled"]
            size = abs(float(fill.get("totalSz", 0)))
            price = float(fill.get("avgPx", 0))
            total += size
            weighted += size * price
            if fill.get("oid") is not None:
                oid = str(fill["oid"])
        return total, weighted / total if total else 0.0, oid

    def _clock_skew_seconds(self) -> float:
        response = self.info.session.head(self.info.base_url, timeout=self.info.timeout)
        response.raise_for_status()
        header = response.headers.get("Date")
        if not header:
            raise ExchangeError("Servidor não forneceu relógio HTTP para o preflight")
        server_time = email.utils.parsedate_to_datetime(header).astimezone(UTC)
        return abs((datetime.now(UTC) - server_time).total_seconds())

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
    """Persistent local fill simulator sharing the live account/order shapes."""

    DEFAULT_METADATA: ClassVar[dict[str, AssetMetadata]] = {
        "BTC": AssetMetadata("BTC", 5, 1e-5, 1),
        "ETH": AssetMetadata("ETH", 4, 1e-4, 2),
        "SOL": AssetMetadata("SOL", 2, 1e-2, 4),
        "XRP": AssetMetadata("XRP", 1, 1e-1, 5),
    }

    def __init__(
        self,
        initial_equity: float = 10_000.0,
        store: StateStore | None = None,
        taker_fee: float = 0.00045,
    ) -> None:
        self.store = store
        self.taker_fee = taker_fee
        self.orders: list[dict[str, Any]]
        self.positions: list[dict[str, Any]]
        self.mids: dict[str, float]
        persisted = store.get("paper_ledger") if store else None
        if isinstance(persisted, dict):
            self.equity = float(persisted.get("cash", initial_equity))
            self.orders = list(persisted.get("orders", []))
            self.positions = list(persisted.get("positions", []))
            self.mids = {str(k): float(v) for k, v in persisted.get("mids", {}).items()}
        else:
            self.equity = initial_equity
            self.orders = []
            self.positions = []
            self.mids = {}
            self._save()

    def asset_metadata(self, symbol: str) -> AssetMetadata:
        try:
            return self.DEFAULT_METADATA[symbol.upper()]
        except KeyError as exc:
            raise ExchangeError(f"Ativo não permitido: {symbol}") from exc

    def snapshot(self) -> AccountSnapshot:
        unrealized = sum(self._unrealized(item) for item in self.positions)
        return AccountSnapshot(
            equity=self.equity + unrealized,
            withdrawable=self.equity,
            open_orders=tuple(item for item in self.orders if item.get("status") == "open"),
            positions=tuple(self.positions),
            mids=dict(self.mids),
        )

    def reconcile(self) -> AccountSnapshot:
        # Reloading the durable ledger provides restart semantics equivalent to REST resync.
        if self.store:
            persisted = self.store.get("paper_ledger")
            if isinstance(persisted, dict):
                self.equity = float(persisted["cash"])
                self.orders = list(persisted["orders"])
                self.positions = list(persisted["positions"])
                self.mids = {str(k): float(v) for k, v in persisted["mids"].items()}
        return self.snapshot()

    def place_entry_with_protection(
        self,
        signal: Signal,
        *,
        size: float,
        stop_price: float,
        take_profit_price: float,
        slippage: float,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        metadata = self.asset_metadata(signal.symbol)
        normalized = metadata.normalize_size(size)
        if normalized <= 0:
            raise ExchangeError("Tamanho abaixo da precisão mínima")
        entry_price = signal.entry_reference * (1 + signal.side.sign * slippage)
        entry_cloid = _cloid_text(
            cloid_for_key(idempotency_key) if idempotency_key else client_order_id()
        )
        existing = next(
            (item for item in self.positions if item.get("entryCloid") == entry_cloid), None
        )
        if existing:
            protections = [item for item in self.orders if item.get("coin") == signal.symbol]
            return {
                "status": "ok",
                "idempotent_replay": True,
                "entry_cloid": entry_cloid,
                "filled_size": abs(float(existing["szi"])),
                "average_price": float(existing["entryPx"]),
                "position": existing,
                "protections": protections,
                "isolated": True,
                "leverage_value": 10,
            }
        if any(item["coin"] == signal.symbol for item in self.positions):
            raise ExchangeError("Aumento de posição e averaging down são proibidos")
        position = {
            "coin": signal.symbol,
            "szi": str(normalized * signal.side.sign),
            "entryPx": str(entry_price),
            "positionValue": str(normalized * entry_price),
            "leverage": {"type": "isolated", "value": 10},
            "openedAt": signal.timestamp.isoformat(),
            "initialSize": normalized,
            "initialRisk": normalized * signal.stop_distance,
            "stopPx": metadata.normalize_price(stop_price),
            "takeProfitPx": metadata.normalize_price(take_profit_price),
            "partialTaken": False,
            "bestPx": entry_price,
            "entryCloid": entry_cloid,
            "regime": signal.regime,
        }
        fee = normalized * entry_price * self.taker_fee
        self.equity -= fee
        self.positions.append(position)
        protections = self._paper_protections(position)
        self.orders.extend(protections)
        self.mids[signal.symbol] = entry_price
        if self.store:
            self.store.upsert_order(
                entry_cloid,
                symbol=signal.symbol,
                side=signal.side.value,
                status="filled",
                payload={"kind": "entry", "filled_size": normalized},
            )
            self.store.record_fill(
                f"paper:{entry_cloid}",
                symbol=signal.symbol,
                size=normalized,
                price=entry_price,
                fee=fee,
                payload={"paper": True, "direction": "entry"},
                client_order_id=entry_cloid,
                filled_at=signal.timestamp.isoformat(),
            )
            self.store.upsert_position(signal.symbol, position)
            for order in protections:
                self.store.upsert_order(
                    str(order["cloid"]),
                    symbol=signal.symbol,
                    side=str(order["side"]),
                    status="open",
                    payload=order,
                )
        self._save()
        return {
            "status": "ok",
            "entry_cloid": entry_cloid,
            "filled_size": normalized,
            "average_price": entry_price,
            "position": position,
            "protections": protections,
            "isolated": True,
            "leverage_value": 10,
        }

    def update_market(
        self, symbol: str, price: float, *, funding_rate: float = 0.0, event_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Apply one unique paper mark/funding event and emulate native trigger fills."""
        if event_id and self.store and self.store.get(f"paper_market_event:{event_id}"):
            return []
        self.mids[symbol] = price
        closed: list[dict[str, Any]] = []
        position = next((item for item in self.positions if item["coin"] == symbol), None)
        if position:
            signed_size = float(position["szi"])
            funding = -(signed_size * price * funding_rate)
            self.equity += funding
            side = Side.LONG if signed_size > 0 else Side.SHORT
            stop_hit = (
                price <= float(position["stopPx"])
                if side is Side.LONG
                else price >= float(position["stopPx"])
            )
            tp_hit = (
                price >= float(position["takeProfitPx"])
                if side is Side.LONG
                else price <= float(position["takeProfitPx"])
            )
            if stop_hit or tp_hit:
                closed.append(
                    self.reduce_position(
                        symbol,
                        size=abs(signed_size),
                        slippage=0.0,
                        reason="native_stop" if stop_hit else "native_take_profit",
                    )
                )
        if event_id and self.store:
            self.store.set(f"paper_market_event:{event_id}", True)
        self._save()
        return closed

    def reduce_position(
        self, symbol: str, *, size: float, slippage: float, reason: str
    ) -> dict[str, Any]:
        position = next((item for item in self.positions if item["coin"] == symbol), None)
        if not position:
            raise ExchangeError(f"Sem posição paper em {symbol}")
        metadata = self.asset_metadata(symbol)
        requested = metadata.normalize_size(size)
        current = abs(float(position["szi"]))
        closed_size = min(requested, current)
        if closed_size <= 0:
            raise ExchangeError("Redução paper inválida")
        side = Side.LONG if float(position["szi"]) > 0 else Side.SHORT
        mark = self.mids.get(symbol, float(position["entryPx"]))
        exit_price = mark * (1 - side.sign * slippage)
        pnl = side.sign * (exit_price - float(position["entryPx"])) * closed_size
        fee = exit_price * closed_size * self.taker_fee
        self.equity += pnl - fee
        remaining = metadata.normalize_size(current - closed_size)
        fill_id = f"paper-close:{position['entryCloid']}:{current}:{remaining}"
        if remaining > 0:
            position["szi"] = str(remaining * side.sign)
            position["positionValue"] = str(remaining * exit_price)
            position["partialTaken"] = True
            self._cancel_paper_protections(symbol)
            protections = self._paper_protections(position)
            self.orders.extend(protections)
            self._persist_paper_protections(protections)
            if self.store:
                self.store.upsert_position(symbol, position)
        else:
            self.positions.remove(position)
            self._cancel_paper_protections(symbol)
            if self.store:
                self.store.remove_position(symbol)
        if self.store:
            self.store.record_fill(
                fill_id,
                symbol=symbol,
                size=closed_size,
                price=exit_price,
                fee=fee,
                payload={"paper": True, "direction": "exit", "reason": reason, "pnl": pnl},
            )
        self._save()
        return {
            "status": "ok",
            "symbol": symbol,
            "closed_size": closed_size,
            "remaining_size": remaining,
            "price": exit_price,
            "pnl": pnl,
            "fee": fee,
            "reason": reason,
        }

    def replace_protection(
        self,
        symbol: str,
        *,
        side: Side,
        size: float,
        stop_price: float,
        take_profit_price: float,
    ) -> list[dict[str, Any]]:
        position = next((item for item in self.positions if item["coin"] == symbol), None)
        if not position:
            raise ExchangeError(f"Sem posição paper em {symbol}")
        position["stopPx"] = self.asset_metadata(symbol).normalize_price(stop_price)
        position["takeProfitPx"] = self.asset_metadata(symbol).normalize_price(take_profit_price)
        self._cancel_paper_protections(symbol)
        protections = self._paper_protections(position, size=size, side=side)
        self.orders.extend(protections)
        if self.store:
            self.store.upsert_position(symbol, position)
            self._persist_paper_protections(protections)
        self._save()
        return protections

    def cancel_all(self) -> list[dict[str, Any]]:
        canceled = list(self.orders)
        if self.store:
            for order in canceled:
                self.store.upsert_order(
                    str(order["cloid"]),
                    symbol=str(order["coin"]),
                    side=str(order["side"]),
                    status="canceled",
                    payload=order | {"status": "canceled"},
                )
        self.orders.clear()
        self._save()
        return canceled

    def flatten_all(self, *, slippage: float = 0.01) -> list[dict[str, Any]]:
        closed: list[dict[str, Any]] = []
        for position in list(self.positions):
            closed.append(
                self.reduce_position(
                    str(position["coin"]),
                    size=abs(float(position["szi"])),
                    slippage=slippage,
                    reason="flatten",
                )
            )
        self.orders.clear()
        self._save()
        return closed

    def schedule_dead_man(self, timeout_seconds: int) -> dict[str, Any]:
        return {"status": "ok", "paper": True, "timeout_seconds": timeout_seconds}

    def _paper_protections(
        self,
        position: dict[str, Any],
        *,
        size: float | None = None,
        side: Side | None = None,
    ) -> list[dict[str, Any]]:
        actual_side = side or (Side.LONG if float(position["szi"]) > 0 else Side.SHORT)
        actual_size = size if size is not None else abs(float(position["szi"]))
        return [
            {
                "coin": position["coin"],
                "cloid": _cloid_text(client_order_id()),
                "kind": kind,
                "price": float(position[price_key]),
                "size": actual_size,
                "side": (Side.SHORT if actual_side is Side.LONG else Side.LONG).value,
                "reduceOnly": True,
                "status": "open",
            }
            for kind, price_key in (("sl", "stopPx"), ("tp", "takeProfitPx"))
        ]

    def _cancel_paper_protections(self, symbol: str) -> None:
        canceled = [item for item in self.orders if item.get("coin") == symbol]
        self.orders = [item for item in self.orders if item.get("coin") != symbol]
        if self.store:
            for order in canceled:
                self.store.upsert_order(
                    str(order["cloid"]),
                    symbol=symbol,
                    side=str(order["side"]),
                    status="canceled",
                    payload=order | {"status": "canceled"},
                )

    def _persist_paper_protections(self, protections: list[dict[str, Any]]) -> None:
        if not self.store:
            return
        for order in protections:
            self.store.upsert_order(
                str(order["cloid"]),
                symbol=str(order["coin"]),
                side=str(order["side"]),
                status="open",
                payload=order,
            )

    def _unrealized(self, position: dict[str, Any]) -> float:
        signed = float(position["szi"])
        mark = self.mids.get(str(position["coin"]), float(position["entryPx"]))
        return (mark - float(position["entryPx"])) * signed

    def _save(self) -> None:
        if self.store:
            self.store.set(
                "paper_ledger",
                {
                    "cash": self.equity,
                    "orders": self.orders,
                    "positions": self.positions,
                    "mids": self.mids,
                    "updated_at": datetime.now(UTC).isoformat(),
                },
            )


def _addresses(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if (
                key.lower() in {"address", "agentaddress", "agent"}
                and isinstance(item, str)
                and item.startswith("0x")
                and len(item) == 42
            ):
                found.add(item.lower())
            found.update(_addresses(item))
    elif isinstance(value, list):
        for item in value:
            found.update(_addresses(item))
    return found


def _timestamp_text(value: Any) -> str:
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value) / 1000, tz=UTC).isoformat()
    if isinstance(value, str) and value:
        return value
    return datetime.now(UTC).isoformat()
