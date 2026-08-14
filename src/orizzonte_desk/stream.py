from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from typing import Any

import websockets

from orizzonte_desk.constants import MAINNET_WS_URL, SYMBOLS, TESTNET_WS_URL
from orizzonte_desk.models import Environment
from orizzonte_desk.storage import StateStore


class HyperliquidStream:
    """Resilient market/user stream; REST engine snapshots are the resync source of truth."""

    def __init__(self, store: StateStore) -> None:
        self.store = store

    async def run(self) -> None:
        delay = 1.0
        while True:
            state = self.store.agent_state()
            environment = state.environment
            endpoint = TESTNET_WS_URL if environment is Environment.TESTNET else MAINNET_WS_URL
            account = state.account_address
            try:
                self.store.set(
                    "market_stream",
                    {
                        "connected": False,
                        "status": "connecting",
                        "environment": environment.value,
                        "updated_at": datetime.now(UTC).isoformat(),
                    },
                )
                async with websockets.connect(
                    endpoint,
                    ping_interval=20,
                    ping_timeout=20,
                    close_timeout=5,
                    max_queue=2048,
                ) as socket:
                    subscriptions: list[dict[str, Any]] = [
                        {"type": "candle", "coin": symbol, "interval": "1h"} for symbol in SYMBOLS
                    ]
                    if account:
                        subscriptions.extend(
                            [
                                {"type": "orderUpdates", "user": account},
                                {"type": "userFills", "user": account},
                                {"type": "userEvents", "user": account},
                            ]
                        )
                    for subscription in subscriptions:
                        await socket.send(
                            json.dumps({"method": "subscribe", "subscription": subscription})
                        )
                    self.store.event(
                        "websocket",
                        f"Streaming conectado em {environment.value}",
                        payload={"subscriptions": len(subscriptions)},
                    )
                    self.store.set(
                        "market_stream",
                        {
                            "connected": True,
                            "status": "connected",
                            "environment": environment.value,
                            "connected_at": datetime.now(UTC).isoformat(),
                            "last_event": datetime.now(UTC).isoformat(),
                        },
                    )
                    self.store.clear_lock("connectivity")
                    delay = 1.0
                    while True:
                        try:
                            raw = await asyncio.wait_for(socket.recv(), timeout=30)
                        except TimeoutError:
                            current = self.store.agent_state()
                            if (
                                current.environment is not environment
                                or current.account_address != account
                            ):
                                break
                            await socket.ping()
                            continue
                        message = json.loads(raw)
                        self._handle(message)
                        current = self.store.agent_state()
                        if (
                            current.environment is not environment
                            or current.account_address != account
                        ):
                            break
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.store.set(
                    "market_stream",
                    {
                        "connected": False,
                        "status": "reconnecting",
                        "environment": environment.value,
                        "last_error": str(exc),
                        "updated_at": datetime.now(UTC).isoformat(),
                    },
                )
                self.store.event(
                    "websocket",
                    "Streaming desconectado; reconexão agendada",
                    level="ERROR",
                    payload={"error": str(exc), "delay_seconds": delay},
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2, 30.0)

    def _handle(self, message: dict[str, Any]) -> None:
        channel = str(message.get("channel", "unknown"))
        now = datetime.now(UTC).isoformat()
        self.store.set(
            "market_stream",
            {
                "connected": True,
                "status": "connected",
                "last_event": now,
                "channel": channel,
            },
        )
        processed = 0
        if channel == "orderUpdates":
            updates = message.get("data", [])
            if isinstance(updates, dict):
                updates = updates.get("orders", updates.get("updates", [updates]))
            if isinstance(updates, list):
                for update in updates:
                    if isinstance(update, dict):
                        self._handle_order(update)
                        processed += 1
        if channel in {"userFills", "userEvents"}:
            data = message.get("data", {})
            fills: object = data
            is_snapshot = False
            if isinstance(data, dict):
                fills = data.get("fills", [])
                is_snapshot = bool(data.get("isSnapshot", False))
            if isinstance(fills, list):
                for fill in fills:
                    if isinstance(fill, dict) and self._handle_fill(fill):
                        processed += 1
            if is_snapshot:
                self.store.set(
                    "exchange_stream_snapshot",
                    {"received_at": now, "fills": len(fills) if isinstance(fills, list) else 0},
                )
        if channel in {"orderUpdates", "userFills", "userEvents", "notification"}:
            self.store.event(
                "exchange_event",
                f"Evento recebido: {channel}",
                payload={"data": message.get("data"), "processed": processed},
            )

    def _handle_order(self, update: dict[str, Any]) -> None:
        order = update.get("order", update)
        if not isinstance(order, dict):
            return
        exchange_order_id = order.get("oid")
        cloid = str(order.get("cloid") or f"oid:{exchange_order_id}")
        symbol = str(order.get("coin", ""))
        if symbol not in SYMBOLS:
            self.store.event(
                "exchange_event",
                "Atualização de ordem fora do universo ignorada",
                level="WARNING",
                payload={"symbol": symbol, "cloid": cloid},
            )
            return
        side = str(order.get("side") or order.get("dir") or "unknown").lower()
        status = str(update.get("status") or order.get("status") or "open").lower()
        self.store.upsert_order(
            cloid,
            symbol=symbol,
            side=side,
            status=status,
            payload=update,
            exchange_order_id=str(exchange_order_id) if exchange_order_id is not None else None,
        )

    def _handle_fill(self, fill: dict[str, Any]) -> bool:
        symbol = str(fill.get("coin", ""))
        size = abs(float(fill.get("sz", 0)))
        price = float(fill.get("px", 0))
        if symbol not in SYMBOLS or size <= 0 or price <= 0:
            return False
        fill_id = str(
            fill.get("tid")
            or fill.get("hash")
            or f"{fill.get('oid')}:{fill.get('time')}:{size}:{price}"
        )
        value = fill.get("time")
        if isinstance(value, (int, float)):
            filled_at = datetime.fromtimestamp(float(value) / 1000, tz=UTC).isoformat()
        else:
            filled_at = str(value or datetime.now(UTC).isoformat())
        return self.store.record_fill(
            fill_id,
            symbol=symbol,
            size=size,
            price=price,
            fee=abs(float(fill.get("fee", 0))),
            payload=fill,
            client_order_id=str(fill["cloid"]) if fill.get("cloid") else None,
            exchange_order_id=str(fill["oid"]) if fill.get("oid") is not None else None,
            filled_at=filled_at,
        )
