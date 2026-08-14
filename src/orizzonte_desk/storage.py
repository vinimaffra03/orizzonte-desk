from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from orizzonte_desk.models import AgentState

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS kv_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    level TEXT NOT NULL,
    category TEXT NOT NULL,
    message TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS orders (
    client_order_id TEXT PRIMARY KEY,
    exchange_order_id TEXT,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    status TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS fills (
    fill_id TEXT PRIMARY KEY,
    client_order_id TEXT,
    exchange_order_id TEXT,
    symbol TEXT NOT NULL,
    size REAL NOT NULL,
    price REAL NOT NULL,
    fee REAL NOT NULL DEFAULT 0,
    payload TEXT NOT NULL,
    filled_at TEXT NOT NULL,
    received_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS positions (
    symbol TEXT PRIMARY KEY,
    payload TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS research_runs (
    run_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    manifest_path TEXT,
    metrics TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_fills_timestamp ON fills(filled_at DESC);
CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);
"""


class StateStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def initialize(self) -> None:
        with self._lock, self.connect() as connection:
            connection.executescript(SCHEMA)
        if self.get("agent_state") is None:
            self.set("agent_state", AgentState().model_dump(mode="json"))

    def set(self, key: str, value: Any) -> None:
        now = datetime.now(UTC).isoformat()
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True)
        with self._lock, self.connect() as connection:
            connection.execute(
                """
                INSERT INTO kv_state(key, value, updated_at) VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
                """,
                (key, encoded, now),
            )

    def get(self, key: str) -> Any | None:
        with self._lock, self.connect() as connection:
            row = connection.execute("SELECT value FROM kv_state WHERE key=?", (key,)).fetchone()
        return json.loads(row["value"]) if row else None

    def agent_state(self) -> AgentState:
        value = self.get("agent_state")
        return AgentState.model_validate(value or {})

    def save_agent_state(self, state: AgentState) -> None:
        self.set("agent_state", state.model_dump(mode="json"))

    def event(
        self,
        category: str,
        message: str,
        *,
        level: str = "INFO",
        payload: dict[str, Any] | None = None,
    ) -> None:
        with self._lock, self.connect() as connection:
            connection.execute(
                "INSERT INTO events(timestamp, level, category, message, payload) VALUES (?, ?, ?, ?, ?)",
                (
                    datetime.now(UTC).isoformat(),
                    level,
                    category,
                    message,
                    json.dumps(payload or {}, ensure_ascii=False, sort_keys=True),
                ),
            )

    def recent_events(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock, self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(row) | {"payload": json.loads(row["payload"])} for row in rows]

    def upsert_order(
        self,
        client_order_id: str,
        *,
        symbol: str,
        side: str,
        status: str,
        payload: dict[str, Any],
        exchange_order_id: str | None = None,
    ) -> None:
        """Persist the newest exchange view while keeping CLOID as the idempotency key."""
        now = datetime.now(UTC).isoformat()
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        with self._lock, self.connect() as connection:
            connection.execute(
                """
                INSERT INTO orders(
                    client_order_id, exchange_order_id, symbol, side, status,
                    payload, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(client_order_id) DO UPDATE SET
                    exchange_order_id=COALESCE(excluded.exchange_order_id, orders.exchange_order_id),
                    symbol=excluded.symbol,
                    side=excluded.side,
                    status=excluded.status,
                    payload=excluded.payload,
                    updated_at=excluded.updated_at
                """,
                (
                    client_order_id,
                    exchange_order_id,
                    symbol,
                    side,
                    status,
                    encoded,
                    now,
                    now,
                ),
            )

    def order(self, client_order_id: str) -> dict[str, Any] | None:
        with self._lock, self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM orders WHERE client_order_id=?", (client_order_id,)
            ).fetchone()
        return self._decode_payload_row(row) if row else None

    def orders(self, *, open_only: bool = False) -> list[dict[str, Any]]:
        query = "SELECT * FROM orders"
        parameters: tuple[Any, ...] = ()
        if open_only:
            query += " WHERE status NOT IN (?, ?, ?, ?, ?)"
            parameters = ("filled", "canceled", "cancelled", "rejected", "closed")
        query += " ORDER BY created_at"
        with self._lock, self.connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [self._decode_payload_row(row) for row in rows]

    def record_fill(
        self,
        fill_id: str,
        *,
        symbol: str,
        size: float,
        price: float,
        payload: dict[str, Any],
        client_order_id: str | None = None,
        exchange_order_id: str | None = None,
        fee: float = 0.0,
        filled_at: str | None = None,
    ) -> bool:
        """Insert a fill once. False means a replayed REST/WS duplicate."""
        if not fill_id or size <= 0 or price <= 0:
            raise ValueError("Fill requer id, tamanho e preço positivos")
        received_at = datetime.now(UTC).isoformat()
        with self._lock, self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO fills(
                    fill_id, client_order_id, exchange_order_id, symbol, size,
                    price, fee, payload, filled_at, received_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    fill_id,
                    client_order_id,
                    exchange_order_id,
                    symbol,
                    size,
                    price,
                    fee,
                    json.dumps(payload, ensure_ascii=False, sort_keys=True),
                    filled_at or received_at,
                    received_at,
                ),
            )
        return cursor.rowcount == 1

    def fills(self, limit: int = 500) -> list[dict[str, Any]]:
        with self._lock, self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM fills ORDER BY filled_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._decode_payload_row(row) for row in rows]

    def upsert_position(self, symbol: str, payload: dict[str, Any]) -> None:
        now = datetime.now(UTC).isoformat()
        with self._lock, self.connect() as connection:
            connection.execute(
                """
                INSERT INTO positions(symbol, payload, updated_at) VALUES (?, ?, ?)
                ON CONFLICT(symbol) DO UPDATE SET
                    payload=excluded.payload, updated_at=excluded.updated_at
                """,
                (symbol, json.dumps(payload, ensure_ascii=False, sort_keys=True), now),
            )

    def remove_position(self, symbol: str) -> None:
        with self._lock, self.connect() as connection:
            connection.execute("DELETE FROM positions WHERE symbol=?", (symbol,))

    def positions(self) -> list[dict[str, Any]]:
        with self._lock, self.connect() as connection:
            rows = connection.execute("SELECT * FROM positions ORDER BY symbol").fetchall()
        return [self._decode_payload_row(row) for row in rows]

    def replace_positions(self, positions: list[dict[str, Any]]) -> None:
        """Atomically apply a REST account snapshot and remove stale local positions."""
        now = datetime.now(UTC).isoformat()
        symbols = {str(item["coin"]) for item in positions}
        with self._lock, self.connect() as connection:
            for item in positions:
                existing = connection.execute(
                    "SELECT payload FROM positions WHERE symbol=?", (str(item["coin"]),)
                ).fetchone()
                merged = json.loads(existing["payload"]) if existing else {}
                merged.update(item)
                connection.execute(
                    """
                    INSERT INTO positions(symbol, payload, updated_at) VALUES (?, ?, ?)
                    ON CONFLICT(symbol) DO UPDATE SET
                        payload=excluded.payload, updated_at=excluded.updated_at
                    """,
                    (
                        str(item["coin"]),
                        json.dumps(merged, ensure_ascii=False, sort_keys=True),
                        now,
                    ),
                )
            if symbols:
                placeholders = ",".join("?" for _ in symbols)
                connection.execute(
                    f"DELETE FROM positions WHERE symbol NOT IN ({placeholders})",
                    tuple(sorted(symbols)),
                )
            else:
                connection.execute("DELETE FROM positions")

    def latch_lock(self, name: str, *, reason: str, payload: dict[str, Any] | None = None) -> None:
        self.set(
            f"risk_lock:{name}",
            {
                "locked": True,
                "reason": reason,
                "payload": payload or {},
                "latched_at": datetime.now(UTC).isoformat(),
            },
        )

    def lock(self, name: str) -> dict[str, Any] | None:
        value = self.get(f"risk_lock:{name}")
        return value if isinstance(value, dict) else None

    def clear_lock(self, name: str) -> None:
        with self._lock, self.connect() as connection:
            connection.execute("DELETE FROM kv_state WHERE key=?", (f"risk_lock:{name}",))

    @staticmethod
    def _decode_payload_row(row: sqlite3.Row) -> dict[str, Any]:
        decoded = dict(row)
        decoded["payload"] = json.loads(decoded["payload"])
        return decoded
