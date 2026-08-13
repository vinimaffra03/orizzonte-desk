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
            {"connected": True, "last_event": now, "channel": channel},
        )
        if channel in {"orderUpdates", "userFills", "userEvents", "notification"}:
            self.store.event(
                "exchange_event",
                f"Evento recebido: {channel}",
                payload={"data": message.get("data")},
            )
