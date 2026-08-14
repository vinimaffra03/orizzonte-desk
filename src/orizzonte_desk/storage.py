from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from orizzonte_desk.models import AgentState, Environment, MainnetAuthorization, TestnetCertificate

LATEST_SCHEMA_VERSION = 3
TERMINAL_ORDER_STATUSES = ("filled", "canceled", "cancelled", "rejected", "closed")

COMMON_SCHEMA = """
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
CREATE TABLE IF NOT EXISTS research_runs (
    run_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    manifest_path TEXT,
    metrics TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL,
    backup_path TEXT
);
"""

SCOPED_SCHEMA = """
CREATE TABLE IF NOT EXISTS orders (
    environment TEXT NOT NULL,
    account_address TEXT NOT NULL,
    client_order_id TEXT NOT NULL,
    exchange_order_id TEXT,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    status TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(environment, account_address, client_order_id)
);
CREATE TABLE IF NOT EXISTS fills (
    environment TEXT NOT NULL,
    account_address TEXT NOT NULL,
    fill_id TEXT NOT NULL,
    client_order_id TEXT,
    exchange_order_id TEXT,
    symbol TEXT NOT NULL,
    size REAL NOT NULL,
    price REAL NOT NULL,
    fee REAL NOT NULL DEFAULT 0,
    payload TEXT NOT NULL,
    filled_at TEXT NOT NULL,
    received_at TEXT NOT NULL,
    PRIMARY KEY(environment, account_address, fill_id)
);
CREATE TABLE IF NOT EXISTS positions (
    environment TEXT NOT NULL,
    account_address TEXT NOT NULL,
    symbol TEXT NOT NULL,
    payload TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(environment, account_address, symbol)
);
CREATE TABLE IF NOT EXISTS testnet_certificates (
    certificate_id TEXT PRIMARY KEY,
    release_id TEXT NOT NULL,
    account_address TEXT NOT NULL,
    wallet_address TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS mainnet_authorizations (
    authorization_id TEXT PRIMARY KEY,
    token_hash TEXT NOT NULL UNIQUE,
    release_id TEXT NOT NULL,
    certificate_id TEXT NOT NULL,
    model_hash TEXT NOT NULL,
    gates_hash TEXT NOT NULL,
    git_commit TEXT NOT NULL,
    config_sha256 TEXT NOT NULL,
    config_fingerprint TEXT NOT NULL,
    code_hash TEXT NOT NULL,
    account_address TEXT NOT NULL,
    wallet_address TEXT NOT NULL,
    budget_usdc REAL NOT NULL,
    issued_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    consumed_at TEXT,
    consumed_by_session TEXT,
    revoked_at TEXT,
    payload TEXT NOT NULL,
    FOREIGN KEY(certificate_id) REFERENCES testnet_certificates(certificate_id)
);
"""

INDEX_SCHEMA = """
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_fills_scope_timestamp
    ON fills(environment, account_address, filled_at DESC);
CREATE INDEX IF NOT EXISTS idx_orders_scope_status
    ON orders(environment, account_address, status);
CREATE INDEX IF NOT EXISTS idx_authorizations_expiry
    ON mainnet_authorizations(expires_at, consumed_at);
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
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def initialize(self) -> None:
        with self._lock:
            backup_path = self._backup_before_migration()
            connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
            connection.row_factory = sqlite3.Row
            try:
                connection.execute("PRAGMA foreign_keys=ON")
                connection.execute("PRAGMA journal_mode=WAL")
                connection.execute("BEGIN IMMEDIATE")
                self._execute_script(connection, COMMON_SCHEMA)
                current = int(connection.execute("PRAGMA user_version").fetchone()[0])
                if current < LATEST_SCHEMA_VERSION:
                    self._migrate_scoped_tables(connection)
                    self._execute_script(connection, SCOPED_SCHEMA)
                    self._execute_script(connection, INDEX_SCHEMA)
                    connection.execute(
                        "INSERT OR REPLACE INTO schema_migrations(version, applied_at, backup_path) "
                        "VALUES (?, ?, ?)",
                        (
                            LATEST_SCHEMA_VERSION,
                            datetime.now(UTC).isoformat(),
                            str(backup_path) if backup_path else None,
                        ),
                    )
                    connection.execute(f"PRAGMA user_version={LATEST_SCHEMA_VERSION}")
                else:
                    self._execute_script(connection, SCOPED_SCHEMA)
                    self._execute_script(connection, INDEX_SCHEMA)
                auth_columns = {
                    str(row["name"])
                    for row in connection.execute("PRAGMA table_info(mainnet_authorizations)")
                }
                required_auth_columns = {
                    "revoked_at": "TEXT",
                    "git_commit": "TEXT",
                    "config_sha256": "TEXT",
                    "config_fingerprint": "TEXT",
                    "code_hash": "TEXT",
                }
                for column, definition in required_auth_columns.items():
                    if column not in auth_columns:
                        connection.execute(
                            f"ALTER TABLE mainnet_authorizations ADD COLUMN {column} {definition}"
                        )
                connection.execute("COMMIT")
            except Exception:
                if connection.in_transaction:
                    connection.execute("ROLLBACK")
                raise
            finally:
                connection.close()
        if self.get("agent_state") is None:
            self.set("agent_state", AgentState().model_dump(mode="json"))

    def _backup_before_migration(self) -> Path | None:
        if not self.path.is_file() or self.path.stat().st_size == 0:
            return None
        with sqlite3.connect(self.path, timeout=30) as source:
            version = int(source.execute("PRAGMA user_version").fetchone()[0])
            tables = source.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchone()[0]
            if version >= LATEST_SCHEMA_VERSION or not tables:
                return None
            timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
            destination_path = self.path.with_name(
                f"{self.path.stem}.pre-v{LATEST_SCHEMA_VERSION}-{timestamp}.bak"
            )
            with sqlite3.connect(destination_path) as destination:
                source.backup(destination)
        return destination_path

    @staticmethod
    def _execute_script(connection: sqlite3.Connection, script: str) -> None:
        for statement in script.split(";"):
            if statement.strip():
                connection.execute(statement)

    def _migrate_scoped_tables(self, connection: sqlite3.Connection) -> None:
        for table in ("orders", "fills", "positions"):
            if not self._table_exists(connection, table):
                continue
            columns = {
                str(row["name"]) for row in connection.execute(f"PRAGMA table_info({table})")
            }
            if {"environment", "account_address"} <= columns:
                continue
            legacy = f"{table}_legacy_v1"
            connection.execute(f"ALTER TABLE {table} RENAME TO {legacy}")
            self._execute_script(connection, SCOPED_SCHEMA)
            legacy_columns = [
                str(row["name"]) for row in connection.execute(f"PRAGMA table_info({legacy})")
            ]
            joined = ", ".join(legacy_columns)
            connection.execute(
                f"INSERT INTO {table}(environment, account_address, {joined}) "
                f"SELECT 'paper', 'paper', {joined} FROM {legacy}"
            )
            connection.execute(f"DROP TABLE {legacy}")

    @staticmethod
    def _table_exists(connection: sqlite3.Connection, table: str) -> bool:
        return (
            connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
            ).fetchone()
            is not None
        )

    def set(self, key: str, value: Any) -> None:
        now = datetime.now(UTC).isoformat()
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True)
        with self._lock, self.connect() as connection:
            connection.execute(
                "INSERT INTO kv_state(key, value, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                (key, encoded, now),
            )

    def get(self, key: str) -> Any | None:
        with self._lock, self.connect() as connection:
            row = connection.execute("SELECT value FROM kv_state WHERE key=?", (key,)).fetchone()
        return json.loads(row["value"]) if row else None

    def agent_state(self) -> AgentState:
        return AgentState.model_validate(self.get("agent_state") or {})

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
        environment: Environment | str = Environment.PAPER,
        account_address: str = "paper",
    ) -> None:
        env, account = _scope(environment, account_address)
        now = datetime.now(UTC).isoformat()
        with self._lock, self.connect() as connection:
            connection.execute(
                """
                INSERT INTO orders(
                    environment, account_address, client_order_id, exchange_order_id,
                    symbol, side, status, payload, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(environment, account_address, client_order_id) DO UPDATE SET
                    exchange_order_id=COALESCE(excluded.exchange_order_id, orders.exchange_order_id),
                    symbol=excluded.symbol, side=excluded.side, status=excluded.status,
                    payload=excluded.payload, updated_at=excluded.updated_at
                """,
                (
                    env,
                    account,
                    client_order_id,
                    exchange_order_id,
                    symbol,
                    side,
                    status,
                    json.dumps(payload, ensure_ascii=False, sort_keys=True),
                    now,
                    now,
                ),
            )

    def order(
        self,
        client_order_id: str,
        *,
        environment: Environment | str = Environment.PAPER,
        account_address: str = "paper",
    ) -> dict[str, Any] | None:
        env, account = _scope(environment, account_address)
        with self._lock, self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM orders WHERE environment=? AND account_address=? "
                "AND client_order_id=?",
                (env, account, client_order_id),
            ).fetchone()
        return self._decode_payload_row(row) if row else None

    def orders(
        self,
        *,
        open_only: bool = False,
        environment: Environment | str = Environment.PAPER,
        account_address: str = "paper",
    ) -> list[dict[str, Any]]:
        env, account = _scope(environment, account_address)
        query = "SELECT * FROM orders WHERE environment=? AND account_address=?"
        parameters: list[Any] = [env, account]
        if open_only:
            query += " AND status NOT IN (?, ?, ?, ?, ?)"
            parameters.extend(TERMINAL_ORDER_STATUSES)
        query += " ORDER BY created_at"
        with self._lock, self.connect() as connection:
            rows = connection.execute(query, tuple(parameters)).fetchall()
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
        environment: Environment | str = Environment.PAPER,
        account_address: str = "paper",
    ) -> bool:
        if not fill_id or size <= 0 or price <= 0:
            raise ValueError("Fill requer id, tamanho e preço positivos")
        env, account = _scope(environment, account_address)
        received_at = datetime.now(UTC).isoformat()
        with self._lock, self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO fills(
                    environment, account_address, fill_id, client_order_id,
                    exchange_order_id, symbol, size, price, fee, payload, filled_at, received_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    env,
                    account,
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

    def fills(
        self,
        limit: int = 500,
        *,
        environment: Environment | str = Environment.PAPER,
        account_address: str = "paper",
    ) -> list[dict[str, Any]]:
        env, account = _scope(environment, account_address)
        with self._lock, self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM fills WHERE environment=? AND account_address=? "
                "ORDER BY filled_at DESC LIMIT ?",
                (env, account, limit),
            ).fetchall()
        return [self._decode_payload_row(row) for row in rows]

    def upsert_position(
        self,
        symbol: str,
        payload: dict[str, Any],
        *,
        environment: Environment | str = Environment.PAPER,
        account_address: str = "paper",
    ) -> None:
        env, account = _scope(environment, account_address)
        now = datetime.now(UTC).isoformat()
        with self._lock, self.connect() as connection:
            connection.execute(
                """
                INSERT INTO positions(environment, account_address, symbol, payload, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(environment, account_address, symbol) DO UPDATE SET
                    payload=excluded.payload, updated_at=excluded.updated_at
                """,
                (env, account, symbol, json.dumps(payload, sort_keys=True), now),
            )

    def remove_position(
        self,
        symbol: str,
        *,
        environment: Environment | str = Environment.PAPER,
        account_address: str = "paper",
    ) -> None:
        env, account = _scope(environment, account_address)
        with self._lock, self.connect() as connection:
            connection.execute(
                "DELETE FROM positions WHERE environment=? AND account_address=? AND symbol=?",
                (env, account, symbol),
            )

    def positions(
        self,
        *,
        environment: Environment | str = Environment.PAPER,
        account_address: str = "paper",
    ) -> list[dict[str, Any]]:
        env, account = _scope(environment, account_address)
        with self._lock, self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM positions WHERE environment=? AND account_address=? ORDER BY symbol",
                (env, account),
            ).fetchall()
        return [self._decode_payload_row(row) for row in rows]

    def replace_positions(
        self,
        positions: list[dict[str, Any]],
        *,
        environment: Environment | str = Environment.PAPER,
        account_address: str = "paper",
    ) -> None:
        env, account = _scope(environment, account_address)
        now = datetime.now(UTC).isoformat()
        symbols = {str(item["coin"]) for item in positions}
        with self._lock, self.connect() as connection:
            for item in positions:
                symbol = str(item["coin"])
                existing = connection.execute(
                    "SELECT payload FROM positions WHERE environment=? AND account_address=? "
                    "AND symbol=?",
                    (env, account, symbol),
                ).fetchone()
                merged = json.loads(existing["payload"]) if existing else {}
                merged.update(item)
                connection.execute(
                    """
                    INSERT INTO positions(environment, account_address, symbol, payload, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(environment, account_address, symbol) DO UPDATE SET
                        payload=excluded.payload, updated_at=excluded.updated_at
                    """,
                    (env, account, symbol, json.dumps(merged, sort_keys=True), now),
                )
            if symbols:
                placeholders = ",".join("?" for _ in symbols)
                connection.execute(
                    "DELETE FROM positions WHERE environment=? AND account_address=? "
                    f"AND symbol NOT IN ({placeholders})",
                    (env, account, *sorted(symbols)),
                )
            else:
                connection.execute(
                    "DELETE FROM positions WHERE environment=? AND account_address=?",
                    (env, account),
                )

    def save_testnet_certificate(self, certificate: TestnetCertificate) -> None:
        with self._lock, self.connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO testnet_certificates(
                    certificate_id, release_id, account_address, wallet_address,
                    payload, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    certificate.certificate_id,
                    certificate.release_id,
                    certificate.account_address,
                    certificate.wallet_address,
                    certificate.model_dump_json(),
                    certificate.created_at.isoformat(),
                ),
            )

    def testnet_certificate(self, certificate_id: str) -> TestnetCertificate | None:
        with self._lock, self.connect() as connection:
            row = connection.execute(
                "SELECT payload FROM testnet_certificates WHERE certificate_id=?",
                (certificate_id,),
            ).fetchone()
        if not row:
            return None
        try:
            return TestnetCertificate.model_validate_json(row["payload"])
        except ValueError:
            return None

    def latest_testnet_certificate(self) -> TestnetCertificate | None:
        with self._lock, self.connect() as connection:
            row = connection.execute(
                "SELECT payload FROM testnet_certificates ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
        if not row:
            return None
        try:
            return TestnetCertificate.model_validate_json(row["payload"])
        except ValueError:
            return None

    def save_mainnet_authorization(
        self, authorization: MainnetAuthorization, *, token_hash: str
    ) -> None:
        with self._lock, self.connect() as connection:
            connection.execute(
                """
                INSERT INTO mainnet_authorizations(
                    authorization_id, token_hash, release_id, certificate_id, model_hash,
                    gates_hash, git_commit, config_sha256, config_fingerprint, code_hash,
                    account_address, wallet_address, budget_usdc, issued_at, expires_at,
                    consumed_at, consumed_by_session, revoked_at, payload
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, ?)
                """,
                (
                    authorization.authorization_id,
                    token_hash,
                    authorization.release_id,
                    authorization.certificate_id,
                    authorization.model_hash,
                    authorization.gates_hash,
                    authorization.git_commit,
                    authorization.config_sha256,
                    authorization.config_fingerprint,
                    authorization.code_hash,
                    authorization.account_address,
                    authorization.wallet_address,
                    authorization.budget_usdc,
                    authorization.issued_at.isoformat(),
                    authorization.expires_at.isoformat(),
                    authorization.model_dump_json(),
                ),
            )

    def authorization(self, authorization_id: str) -> dict[str, Any] | None:
        with self._lock, self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM mainnet_authorizations WHERE authorization_id=?",
                (authorization_id,),
            ).fetchone()
        return dict(row) | {"payload": json.loads(row["payload"])} if row else None

    def consume_mainnet_authorization(
        self,
        authorization_id: str,
        *,
        token_hash: str,
        session_id: str,
        release_id: str,
        certificate_id: str,
        model_hash: str,
        gates_hash: str,
        git_commit: str,
        config_sha256: str,
        config_fingerprint: str,
        code_hash: str,
        account_address: str,
        wallet_address: str,
        budget_usdc: float,
        now: datetime | None = None,
    ) -> MainnetAuthorization:
        timestamp = now or datetime.now(UTC)
        expected = {
            "token_hash": token_hash,
            "release_id": release_id,
            "certificate_id": certificate_id,
            "model_hash": model_hash,
            "gates_hash": gates_hash,
            "git_commit": git_commit,
            "config_sha256": config_sha256,
            "config_fingerprint": config_fingerprint,
            "code_hash": code_hash,
            "account_address": account_address.lower(),
            "wallet_address": wallet_address.lower(),
        }
        with self._lock, self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM mainnet_authorizations WHERE authorization_id=?",
                (authorization_id,),
            ).fetchone()
            if not row:
                raise ValueError("Autorização mainnet inexistente")
            if row["consumed_at"]:
                raise ValueError("Autorização mainnet já consumida")
            if row["revoked_at"]:
                raise ValueError("Autorização mainnet revogada")
            if datetime.fromisoformat(str(row["expires_at"])) <= timestamp:
                raise ValueError("Autorização mainnet expirada")
            for key, value in expected.items():
                actual = str(row[key]).lower() if "address" in key else str(row[key])
                if actual != str(value):
                    raise ValueError(f"Binding mainnet divergente: {key}")
            if abs(float(row["budget_usdc"]) - budget_usdc) > 1e-9:
                raise ValueError("Binding mainnet divergente: budget_usdc")
            authorization = MainnetAuthorization.model_validate_json(row["payload"])
            for key, value in expected.items():
                if key == "token_hash":
                    continue
                actual = getattr(authorization, key)
                if "address" in key:
                    actual = str(actual).lower()
                if actual != value:
                    raise ValueError(f"Payload mainnet divergente: {key}")
            if abs(authorization.budget_usdc - budget_usdc) > 1e-9:
                raise ValueError("Payload mainnet divergente: budget_usdc")
            cursor = connection.execute(
                "UPDATE mainnet_authorizations SET consumed_at=?, consumed_by_session=? "
                "WHERE authorization_id=? AND consumed_at IS NULL",
                (timestamp.isoformat(), session_id, authorization_id),
            )
            if cursor.rowcount != 1:
                raise ValueError("Autorização mainnet sofreu consumo concorrente")
        return authorization

    def revoke_mainnet_authorization(self, authorization_id: str) -> bool:
        with self._lock, self.connect() as connection:
            cursor = connection.execute(
                "UPDATE mainnet_authorizations SET revoked_at=? "
                "WHERE authorization_id=? AND consumed_at IS NULL AND revoked_at IS NULL",
                (datetime.now(UTC).isoformat(), authorization_id),
            )
        return cursor.rowcount == 1

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


def _scope(environment: Environment | str, account_address: str) -> tuple[str, str]:
    env = environment.value if isinstance(environment, Environment) else str(environment)
    account = account_address.lower() if account_address else "paper"
    return env, account
