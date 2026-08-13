from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any, cast

import httpx
import pandas as pd

from orizzonte_desk.config import Settings
from orizzonte_desk.constants import MAINNET_API_URL, SYMBOLS, TESTNET_API_URL
from orizzonte_desk.controller import AgentController
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

    def tick(self) -> dict[str, Any]:
        state = self.store.agent_state()
        if state.status is not AgentStatus.RUNNING:
            return {"acted": False, "reason": f"state:{state.status}"}
        if state.budget_usdc is None:
            raise RuntimeError("Agente rodando sem orçamento")
        market = self._market_window(state.environment)
        newest = pd.to_datetime(market["timestamp"], utc=True).max()
        now = pd.Timestamp.now(tz="UTC")
        if (now - newest).total_seconds() > self.settings.execution.stale_data_seconds + 3600:
            raise RuntimeError(f"Market data stale: {newest.isoformat()}")
        metadata = dict(state.metadata)
        last_signal_bar = metadata.get("last_signal_bar")
        if last_signal_bar == newest.isoformat():
            return {"acted": False, "reason": "bar_already_processed"}
        gateway = self.controller.gateway(state.environment)
        account = gateway.snapshot()
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
        actions: list[dict[str, Any]] = []
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
                    open_notional=evaluated.open_notional,
                    size_increment=1e-6,
                )
                slippage = (
                    self.settings.execution.slippage_bps_btc_eth
                    if signal.symbol in {"BTC", "ETH"}
                    else self.settings.execution.slippage_bps_sol_xrp
                ) / 10_000
                response = gateway.place_entry_with_protection(
                    signal,
                    size=sizing.size,
                    stop_price=sizing.stop_price,
                    take_profit_price=sizing.take_profit_price,
                    slippage=slippage,
                )
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
                current_positions.append(
                    Position(
                        symbol=signal.symbol,
                        side=signal.side,
                        size=sizing.size,
                        initial_size=sizing.size,
                        entry_price=signal.entry_reference,
                        stop_price=sizing.stop_price,
                        take_profit_price=sizing.take_profit_price,
                        opened_at=datetime.now(UTC),
                        initial_risk_usdc=sizing.risk_usdc,
                    )
                )
                if len(current_positions) >= self.settings.risk.max_positions:
                    break
            except RiskViolation as exc:
                self.store.event(
                    "risk",
                    f"Sinal {signal.symbol} rejeitado",
                    payload={"reason": str(exc)},
                )
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
        return {"acted": bool(actions), "actions": actions, "signals": len(signals)}

    def _market_window(self, environment: Environment) -> pd.DataFrame:
        base_url = MAINNET_API_URL if environment is Environment.MAINNET else TESTNET_API_URL
        if environment is Environment.PAPER:
            base_url = MAINNET_API_URL
        end = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
        start = end - timedelta(hours=5000)
        rows: list[dict[str, Any]] = []
        for symbol in SYMBOLS:
            response = self.client.post(
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
            response.raise_for_status()
            rows.extend(
                {
                    "timestamp": pd.to_datetime(item["t"], unit="ms", utc=True),
                    "symbol": symbol,
                    "interval": "1h",
                    "open": float(item["o"]),
                    "high": float(item["h"]),
                    "low": float(item["l"]),
                    "close": float(item["c"]),
                    "volume": float(item["v"]),
                    "funding_rate": 0.0,
                }
                for item in response.json()
            )
        if not rows:
            raise RuntimeError("Sem dados de mercado")
        return pd.DataFrame(rows)

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
