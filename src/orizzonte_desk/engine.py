from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any, cast

import httpx
import pandas as pd

from orizzonte_desk.config import Settings
from orizzonte_desk.constants import MAINNET_API_URL, SYMBOLS, TESTNET_API_URL
from orizzonte_desk.controller import AgentController
from orizzonte_desk.exchange import cloid_for_key
from orizzonte_desk.ml import MetaModelRegistry
from orizzonte_desk.models import AgentStatus, Environment, Position, RiskSnapshot, Side
from orizzonte_desk.paths import AppPaths
from orizzonte_desk.risk import RiskManager, RiskViolation
from orizzonte_desk.storage import StateStore
from orizzonte_desk.strategy import SignalGenerator


class TradingEngine:
    def __init__(
        self,
        paths: AppPaths,
        settings: Settings,
        store: StateStore,
        controller: AgentController,
    ) -> None:
        self.paths = paths
        self.settings = settings
        self.store = store
        self.controller = controller
        self.risk = RiskManager(settings.risk)
        self.generator = SignalGenerator(settings.strategy, MetaModelRegistry(paths))
        self.client = httpx.Client(timeout=20)
        self._market_cache: dict[Environment, tuple[datetime, pd.DataFrame]] = {}

    def tick(self) -> dict[str, Any]:
        state = self.store.agent_state()
        if state.status is not AgentStatus.RUNNING:
            return {"acted": False, "reason": f"state:{state.status}"}
        if state.budget_usdc is None:
            raise RuntimeError("Agente rodando sem orçamento")
        if state.environment is not Environment.PAPER:
            self._assert_stream_healthy()
        market = self._market_window(state.environment)
        newest = pd.to_datetime(market["timestamp"], utc=True).max()
        now = pd.Timestamp.now(tz="UTC")
        if (now - newest).total_seconds() > self.settings.execution.stale_data_seconds + 3600:
            raise RuntimeError(f"Market data stale: {newest.isoformat()}")
        metadata = dict(state.metadata)
        last_signal_bar = metadata.get("last_signal_bar")
        gateway = self.controller.gateway(state.environment)
        updater = getattr(gateway, "update_market", None)
        if callable(updater):
            for row in latest_rows(market):
                updater(
                    str(row["symbol"]),
                    float(row["close"]),
                    funding_rate=float(row.get("funding_rate", 0)),
                    event_id=f"{row['symbol']}:{pd.Timestamp(row['timestamp']).isoformat()}",
                )
        account = gateway.reconcile()
        self._audit_position_protections(gateway, account.positions)
        management_actions = self._manage_positions(gateway, account.positions, market)
        if management_actions:
            account = gateway.reconcile()
        if last_signal_bar == newest.isoformat():
            return {
                "acted": bool(management_actions),
                "reason": "bar_already_processed",
                "actions": management_actions,
            }
        equity = account.equity
        day_key = now.strftime("%Y-%m-%d")
        if metadata.get("risk_day") != day_key:
            metadata.update(
                {
                    "risk_day": day_key,
                    "day_start_equity": equity,
                    "profit_locked": False,
                    "loss_locked": False,
                }
            )
        high_water = max(float(metadata.get("high_water_mark", equity)), equity)
        metadata["high_water_mark"] = high_water
        latest_market = (
            market.sort_values("timestamp")
            .groupby("symbol", as_index=False)
            .tail(1)
            .loc[:, ["timestamp", "symbol", "close", "funding_rate"]]
        )
        metadata["market"] = [
            {
                "timestamp": pd.Timestamp(row["timestamp"]).isoformat(),
                "symbol": str(row["symbol"]),
                "close": float(row["close"]),
                "funding_rate": float(row["funding_rate"]),
            }
            for row in latest_market.to_dict(orient="records")
        ]
        metadata["positions"] = list(account.positions)
        metadata["orders"] = list(account.open_orders)
        snapshot = RiskSnapshot(
            budget=state.budget_usdc,
            equity=max(equity, 0.01),
            day_start_equity=max(float(metadata.get("day_start_equity", equity)), 0.01),
            high_water_mark=max(high_water, 0.01),
            open_risk=len(account.positions)
            * state.budget_usdc
            * self.settings.risk.risk_per_trade,
            open_notional=sum(
                abs(float(item.get("positionValue", 0))) for item in account.positions
            ),
            positions_count=len(account.positions),
            profit_locked=bool(metadata.get("profit_locked", False)),
            loss_locked=bool(metadata.get("loss_locked", False)),
            drawdown_locked=bool(metadata.get("drawdown_locked", False)),
        )
        evaluated = self.risk.evaluate_locks(snapshot)
        metadata.update(
            {
                "profit_locked": evaluated.profit_locked,
                "loss_locked": evaluated.loss_locked,
                "drawdown_locked": evaluated.drawdown_locked,
                "last_equity": equity,
                "last_signal_bar": newest.isoformat(),
            }
        )
        if evaluated.loss_locked or evaluated.drawdown_locked:
            if evaluated.loss_locked:
                self.store.latch_lock(
                    "daily_loss",
                    reason="Stop diário atingido",
                    payload={"risk_day": day_key, "equity": equity},
                )
            if evaluated.drawdown_locked:
                self.store.latch_lock(
                    "drawdown",
                    reason="Drawdown máximo atingido",
                    payload={"high_water_mark": high_water, "equity": equity},
                )
            gateway.flatten_all(slippage=0.02)
            locked = state.model_copy(
                update={
                    "status": AgentStatus.LOCKED,
                    "loss_locked": evaluated.loss_locked,
                    "drawdown_locked": evaluated.drawdown_locked,
                    "metadata": metadata,
                }
            )
            self.store.save_agent_state(locked)
            self.store.event(
                "risk",
                "Flatten automático por limite de risco",
                level="CRITICAL",
                payload=metadata,
            )
            return {"acted": True, "action": "flatten"}
        current_positions = self._positions(account.positions, state.budget_usdc)
        correlations = self._latest_correlations(market)
        signals = sorted(
            self.generator.latest(
                market,
                require_promoted_model=state.environment is not Environment.PAPER,
            ),
            key=lambda item: (item.probability, item.score),
            reverse=True,
        )
        metadata["signals"] = [signal.model_dump(mode="json") for signal in signals]
        # Persist the bar boundary before the first network call. A timeout can never cause the
        # next tick to generate a fresh CLOID for the same intent.
        self.store.save_agent_state(state.model_copy(update={"metadata": metadata}))
        actions: list[dict[str, Any]] = []
        open_notional = evaluated.open_notional
        for signal in signals:
            try:
                self.risk.assert_new_entry_allowed(
                    signal,
                    evaluated,
                    current_positions,
                    correlations,
                )
                sizing = self.risk.size_order(
                    signal,
                    budget=state.budget_usdc,
                    equity=equity,
                    open_notional=open_notional,
                    size_increment=gateway.asset_metadata(signal.symbol).size_increment,
                )
                slippage = (
                    self.settings.execution.slippage_bps_btc_eth
                    if signal.symbol in {"BTC", "ETH"}
                    else self.settings.execution.slippage_bps_sol_xrp
                ) / 10_000
                intent = (
                    f"{state.environment.value}:{newest.isoformat()}:{signal.symbol}:"
                    f"{signal.side.value}:{state.metadata.get('release_id', 'paper')}"
                )
                intent_cloid = str(cloid_for_key(intent))
                self.store.upsert_order(
                    intent_cloid,
                    symbol=signal.symbol,
                    side=signal.side.value,
                    status="intent",
                    payload={"bar": newest.isoformat(), "intent": intent},
                )
                response = gateway.place_entry_with_protection(
                    signal,
                    size=sizing.size,
                    stop_price=sizing.stop_price,
                    take_profit_price=sizing.take_profit_price,
                    slippage=slippage,
                    idempotency_key=intent,
                )
                self._assert_position_protected(gateway, signal.symbol)
                actions.append({"symbol": signal.symbol, "side": signal.side, "response": response})
                self.store.event(
                    "execution",
                    f"Entrada {signal.side.value} {signal.symbol}",
                    payload={
                        "size": sizing.size,
                        "stop": sizing.stop_price,
                        "take_profit": sizing.take_profit_price,
                        "probability": signal.probability,
                    },
                )
                filled_size = float(response.get("filled_size", sizing.size))
                average_price = float(response.get("average_price") or signal.entry_reference)
                open_notional += filled_size * average_price
                tracked_position = Position(
                    symbol=signal.symbol,
                    side=signal.side,
                    size=filled_size,
                    initial_size=filled_size,
                    entry_price=average_price,
                    stop_price=sizing.stop_price,
                    take_profit_price=sizing.take_profit_price,
                    opened_at=datetime.now(UTC),
                    initial_risk_usdc=filled_size * signal.stop_distance,
                    order_id=str(response.get("entry_cloid") or "") or None,
                )
                self.store.upsert_position(
                    signal.symbol,
                    {
                        "coin": signal.symbol,
                        "szi": str(filled_size * signal.side.sign),
                        "entryPx": str(average_price),
                        "stopPx": sizing.stop_price,
                        "takeProfitPx": sizing.take_profit_price,
                        "openedAt": tracked_position.opened_at.isoformat(),
                        "initialSize": filled_size,
                        "initialRisk": tracked_position.initial_risk_usdc,
                        "partialTaken": False,
                        "regime": signal.regime,
                        "entryCloid": response.get("entry_cloid"),
                    },
                )
                current_positions.append(tracked_position)
                if len(current_positions) >= self.settings.risk.max_positions:
                    break
            except RiskViolation as exc:
                self.store.event(
                    "risk",
                    f"Sinal {signal.symbol} rejeitado",
                    payload={"reason": str(exc)},
                )
            except Exception as exc:
                residual: object = None
                try:
                    residual = gateway.reconcile().positions
                    gateway.flatten_all(slippage=0.03)
                    residual = gateway.reconcile().positions
                except Exception as flatten_exc:
                    residual = {"reconcile_or_flatten_error": str(flatten_exc)}
                self.store.latch_lock(
                    "unprotected_position",
                    reason="Falha de execução; sessão bloqueada",
                    payload={"symbol": signal.symbol, "error": str(exc), "residual": residual},
                )
                locked = state.model_copy(
                    update={
                        "status": AgentStatus.LOCKED,
                        "metadata": metadata
                        | {"execution_error": str(exc), "last_signal_bar": newest.isoformat()},
                    }
                )
                self.store.save_agent_state(locked)
                self.store.event(
                    "execution",
                    "Falha de execução; flatten e lock acionados",
                    level="CRITICAL",
                    payload={"symbol": signal.symbol, "error": str(exc)},
                )
                return {"acted": True, "action": "execution_failure_lock", "error": str(exc)}
        updated = state.model_copy(
            update={
                "last_heartbeat": datetime.now(UTC),
                "profit_locked": evaluated.profit_locked,
                "loss_locked": evaluated.loss_locked,
                "drawdown_locked": evaluated.drawdown_locked,
                "metadata": metadata,
            }
        )
        self.store.save_agent_state(updated)
        return {
            "acted": bool(actions or management_actions),
            "actions": management_actions + actions,
            "signals": len(signals),
        }

    def _assert_stream_healthy(self) -> None:
        stream = self.store.get("market_stream")
        if not isinstance(stream, dict) or not stream.get("connected"):
            self.store.latch_lock("connectivity", reason="WebSocket desconectado")
            raise RuntimeError("Streaming da Hyperliquid desconectado")
        last_event = stream.get("last_event")
        if not isinstance(last_event, str):
            raise RuntimeError("Streaming sem timestamp de atividade")
        age = (datetime.now(UTC) - datetime.fromisoformat(last_event)).total_seconds()
        if age > self.settings.execution.stale_data_seconds:
            self.store.latch_lock(
                "connectivity",
                reason="Streaming stale",
                payload={"age_seconds": age},
            )
            raise RuntimeError(f"Streaming stale há {age:.1f}s")

    def _manage_positions(
        self,
        gateway: Any,
        raw_positions: tuple[dict[str, Any], ...],
        market: pd.DataFrame,
    ) -> list[dict[str, Any]]:
        tracked = {item["symbol"]: item["payload"] for item in self.store.positions()}
        latest = {str(item["symbol"]): item for item in latest_rows(market)}
        actions: list[dict[str, Any]] = []
        now = datetime.now(UTC)
        for raw in raw_positions:
            symbol = str(raw.get("coin", ""))
            if symbol not in SYMBOLS or symbol not in latest:
                continue
            item = tracked.get(symbol, {}) | raw
            signed_size = float(item.get("szi", 0))
            size = abs(signed_size)
            entry = float(item.get("entryPx", 0))
            if size <= 0 or entry <= 0:
                continue
            side = Side.LONG if signed_size > 0 else Side.SHORT
            mark = float(latest[symbol]["close"])
            opened_raw = item.get("openedAt")
            opened = datetime.fromisoformat(str(opened_raw)) if opened_raw else now
            if opened.tzinfo is None:
                opened = opened.replace(tzinfo=UTC)
            if (now - opened).total_seconds() >= self.settings.strategy.max_holding_hours * 3600:
                response = gateway.reduce_position(
                    symbol, size=size, slippage=0.02, reason="time_stop"
                )
                actions.append({"symbol": symbol, "action": "time_stop", "response": response})
                continue
            initial_size = float(item.get("initialSize", size))
            initial_risk = float(item.get("initialRisk", 0))
            stop_distance = initial_risk / initial_size if initial_size > 0 else 0
            favorable = side.sign * (mark - entry)
            partial_taken = bool(item.get("partialTaken", False))
            if stop_distance > 0 and favorable >= stop_distance and not partial_taken:
                partial_size = gateway.asset_metadata(symbol).normalize_size(
                    size * self.settings.strategy.partial_fraction
                )
                if 0 < partial_size < size:
                    response = gateway.reduce_position(
                        symbol, size=partial_size, slippage=0.02, reason="partial_1r"
                    )
                    remaining = gateway.asset_metadata(symbol).normalize_size(size - partial_size)
                    take_profit = float(item.get("takeProfitPx", entry + side.sign * stop_distance))
                    protections = gateway.replace_protection(
                        symbol,
                        side=side,
                        size=remaining,
                        stop_price=entry,
                        take_profit_price=take_profit,
                    )
                    item.update(
                        {
                            "szi": str(remaining * side.sign),
                            "partialTaken": True,
                            "stopPx": entry,
                        }
                    )
                    self.store.upsert_position(symbol, item)
                    actions.append(
                        {
                            "symbol": symbol,
                            "action": "partial_1r_breakeven",
                            "response": response,
                            "protections": protections,
                        }
                    )
                    continue
            symbol_rows = market.loc[market["symbol"] == symbol].sort_values("timestamp").tail(20)
            previous = symbol_rows["close"].shift(1)
            true_range = pd.concat(
                [
                    symbol_rows["high"] - symbol_rows["low"],
                    (symbol_rows["high"] - previous).abs(),
                    (symbol_rows["low"] - previous).abs(),
                ],
                axis=1,
            ).max(axis=1)
            atr = float(true_range.tail(self.settings.strategy.atr_window).mean())
            old_stop = float(item.get("stopPx", entry - side.sign * max(stop_distance, 1e-9)))
            candidate = mark - side.sign * atr * self.settings.strategy.atr_trailing_multiple
            new_stop = max(old_stop, candidate) if side is Side.LONG else min(old_stop, candidate)
            improves = new_stop > old_stop if side is Side.LONG else new_stop < old_stop
            if improves and side.sign * (new_stop - entry) >= 0:
                take_profit = float(
                    item.get("takeProfitPx", entry + side.sign * max(stop_distance, atr))
                )
                protections = gateway.replace_protection(
                    symbol,
                    side=side,
                    size=size,
                    stop_price=new_stop,
                    take_profit_price=take_profit,
                )
                item["stopPx"] = new_stop
                self.store.upsert_position(symbol, item)
                actions.append(
                    {"symbol": symbol, "action": "atr_trailing", "protections": protections}
                )
        return actions

    def _audit_position_protections(
        self, gateway: Any, raw_positions: tuple[dict[str, Any], ...]
    ) -> None:
        tracked = {item["symbol"]: item["payload"] for item in self.store.positions()}
        snapshot = gateway.snapshot()
        for raw in raw_positions:
            symbol = str(raw.get("coin", ""))
            local = tracked.get(symbol, {})
            if not local.get("entryCloid"):
                gateway.flatten_all(slippage=0.03)
                self.store.latch_lock(
                    "unprotected_position",
                    reason="Posição sem ownership CLOID detectada",
                    payload={"symbol": symbol},
                )
                raise RuntimeError(f"Posição não atribuída ao agente: {symbol}")
            size = abs(float(raw.get("szi", 0)))
            protections = [
                item
                for item in snapshot.open_orders
                if item.get("coin") == symbol
                and bool(item.get("reduceOnly", False))
                and float(item.get("sz", item.get("size", size))) >= size
            ]
            if len(protections) >= 2:
                continue
            side = Side.LONG if float(raw.get("szi", 0)) > 0 else Side.SHORT
            try:
                gateway.replace_protection(
                    symbol,
                    side=side,
                    size=size,
                    stop_price=float(local["stopPx"]),
                    take_profit_price=float(local["takeProfitPx"]),
                )
                self._assert_position_protected(gateway, symbol)
            except Exception as exc:
                gateway.flatten_all(slippage=0.03)
                self.store.latch_lock(
                    "unprotected_position",
                    reason="Proteção ausente não pôde ser restaurada",
                    payload={"symbol": symbol, "error": str(exc)},
                )
                raise RuntimeError(f"Posição sem proteção confirmada: {symbol}") from exc

    @staticmethod
    def _assert_position_protected(gateway: Any, symbol: str) -> None:
        snapshot = gateway.snapshot()
        position = next((item for item in snapshot.positions if item.get("coin") == symbol), None)
        protections = [
            item
            for item in snapshot.open_orders
            if item.get("coin") == symbol and bool(item.get("reduceOnly", False))
        ]
        isolated = (
            isinstance(position, dict)
            and str(position.get("leverage", {}).get("type", "")) == "isolated"
        )
        leverage = int(position.get("leverage", {}).get("value", 0)) if position else 0
        if not position or len(protections) < 2 or not isolated or leverage != 10:
            gateway.flatten_all(slippage=0.02)
            raise RuntimeError(
                f"Entrada {symbol} sem confirmação de margem isolada 10x e duas proteções"
            )

    def _market_window(self, environment: Environment) -> pd.DataFrame:
        base_url = MAINNET_API_URL if environment is Environment.MAINNET else TESTNET_API_URL
        if environment is Environment.PAPER:
            base_url = MAINNET_API_URL
        end = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
        cached = self._market_cache.get(environment)
        if cached and cached[0] == end:
            return cached[1].copy(deep=True)
        start = end - timedelta(hours=5000)
        rows: list[dict[str, Any]] = []
        for symbol in SYMBOLS:
            candles_response = self.client.post(
                f"{base_url}/info",
                json={
                    "type": "candleSnapshot",
                    "req": {
                        "coin": symbol,
                        "interval": "1h",
                        "startTime": int(start.timestamp() * 1000),
                        "endTime": int(end.timestamp() * 1000),
                    },
                },
            )
            candles_response.raise_for_status()
            funding = self._funding_history(
                base_url,
                symbol,
                start=end - timedelta(hours=24 * 30 + 2),
                end=end,
            )
            rows.extend(
                {
                    "timestamp": (timestamp := pd.to_datetime(item["t"], unit="ms", utc=True)),
                    "symbol": symbol,
                    "interval": "1h",
                    "open": float(item["o"]),
                    "high": float(item["h"]),
                    "low": float(item["l"]),
                    "close": float(item["c"]),
                    "volume": float(item["v"]),
                    "funding_rate": funding.get(timestamp.floor("h"), 0.0),
                }
                for item in candles_response.json()
            )
        if not rows:
            raise RuntimeError("Sem dados de mercado")
        frame = pd.DataFrame(rows)
        self._market_cache[environment] = (end, frame.copy(deep=True))
        return frame

    def _funding_history(
        self,
        base_url: str,
        symbol: str,
        *,
        start: datetime,
        end: datetime,
    ) -> dict[pd.Timestamp, float]:
        """Fetch the rolling feature window using the official paginated info endpoint."""
        cursor = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)
        rates: dict[pd.Timestamp, float] = {}
        while cursor <= end_ms:
            response = self.client.post(
                f"{base_url}/info",
                json={
                    "type": "fundingHistory",
                    "coin": symbol,
                    "startTime": cursor,
                    "endTime": end_ms,
                },
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, list):
                raise RuntimeError(f"Funding inválido para {symbol}")
            if not payload:
                break
            timestamps: list[int] = []
            for item in payload:
                timestamp_ms = int(item["time"])
                timestamps.append(timestamp_ms)
                hour = pd.to_datetime(timestamp_ms, unit="ms", utc=True).floor("h")
                rates[hour] = float(item["fundingRate"])
            next_cursor = max(timestamps) + 1
            if len(payload) < 500 or next_cursor <= cursor:
                break
            cursor = next_cursor
        if not rates:
            raise RuntimeError(f"Funding histórico ausente para {symbol}")
        latest = max(rates)
        if (pd.Timestamp(end) - latest).total_seconds() > 2 * 3600:
            raise RuntimeError(f"Funding histórico stale para {symbol}: {latest.isoformat()}")
        return rates

    @staticmethod
    def _positions(raw: tuple[dict[str, Any], ...], budget: float) -> list[Position]:
        output: list[Position] = []
        for item in raw:
            size = float(item.get("szi", 0))
            if not size or str(item.get("coin")) not in SYMBOLS:
                continue
            entry = float(item.get("entryPx") or 0)
            liquidation = float(item.get("liquidationPx") or entry * (0.9 if size > 0 else 1.1))
            risk = min(budget * 0.01, abs(entry - liquidation) * abs(size))
            output.append(
                Position(
                    symbol=str(item["coin"]),
                    side=Side.LONG if size > 0 else Side.SHORT,
                    size=abs(size),
                    initial_size=abs(size),
                    entry_price=entry,
                    stop_price=entry - (1 if size > 0 else -1) * max(entry * 0.005, 1e-9),
                    take_profit_price=entry + (1 if size > 0 else -1) * max(entry * 0.005, 1e-9),
                    opened_at=datetime.now(UTC),
                    initial_risk_usdc=max(risk, 0.01),
                )
            )
        return output

    @staticmethod
    def _latest_correlations(market: pd.DataFrame) -> dict[tuple[str, str], float]:
        pivot = market.pivot(index="timestamp", columns="symbol", values="close").sort_index()
        corr = pivot.pct_change().tail(24 * 30).corr()
        result: dict[tuple[str, str], float] = {}
        for index, left in enumerate(SYMBOLS):
            for right in SYMBOLS[index + 1 :]:
                value = cast(Any, corr.loc[left, right])
                if pd.notna(value):
                    result[(left, right)] = float(value)
        return result


def latest_rows(market: pd.DataFrame) -> list[dict[str, Any]]:
    return cast(
        list[dict[str, Any]],
        market.sort_values("timestamp")
        .groupby("symbol", as_index=False)
        .tail(1)
        .to_dict(orient="records"),
    )
