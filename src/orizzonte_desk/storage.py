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
