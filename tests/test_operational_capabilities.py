from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from eth_account import Account
from fastapi.testclient import TestClient
from pydantic import ValidationError

import orizzonte_desk.secrets as secret_module
from orizzonte_desk import runtime_primitives as primitives
from orizzonte_desk.chaos import ChaosValidationError
from orizzonte_desk.chaos import (
    TestnetChaosContext as ChaosContext,
)
from orizzonte_desk.chaos import (
    TestnetChaosRunner as ChaosRunner,
)
from orizzonte_desk.controller import AgentController, ControlError
from orizzonte_desk.daemon import _positions_are_protected, create_app
from orizzonte_desk.exchange import AccountSnapshot, AssetMetadata, HyperliquidGateway
from orizzonte_desk.models import (
    REQUIRED_TESTNET_SCENARIOS,
    AgentState,
    AgentStatus,
    Environment,
    MainnetAuthorization,
)
from orizzonte_desk.models import TestnetCertificate as Certificate
from orizzonte_desk.release import ReleaseManifest
from orizzonte_desk.secrets import DPAPICapabilityStore, EnvironmentSecretManager, SecretStoreError
from orizzonte_desk.storage import LATEST_SCHEMA_VERSION, StateStore
from orizzonte_desk.stream import HyperliquidStream


def _legacy_database(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE orders (
                client_order_id TEXT PRIMARY KEY, exchange_order_id TEXT, symbol TEXT NOT NULL,
                side TEXT NOT NULL, status TEXT NOT NULL, payload TEXT NOT NULL,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            );
            CREATE TABLE fills (
                fill_id TEXT PRIMARY KEY, client_order_id TEXT, exchange_order_id TEXT,
                symbol TEXT NOT NULL, size REAL NOT NULL, price REAL NOT NULL,
                fee REAL NOT NULL, payload TEXT NOT NULL, filled_at TEXT NOT NULL,
                received_at TEXT NOT NULL
            );
            CREATE TABLE positions (
                symbol TEXT PRIMARY KEY, payload TEXT NOT NULL, updated_at TEXT NOT NULL
            );
            """
        )
        connection.execute(
            "INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("c1", "7", "BTC", "long", "open", "{}", "t1", "t1"),
        )
        connection.execute(
            "INSERT INTO fills VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("f1", "c1", "7", "BTC", 0.1, 60_000, 1, "{}", "t1", "t1"),
        )
        connection.execute(
            "INSERT INTO positions VALUES (?, ?, ?)",
            ("BTC", '{"coin":"BTC","szi":"0.1"}', "t1"),
        )


def _logical_digest(path: Path) -> str:
    payload: dict[str, list[tuple[object, ...]]] = {}
    with sqlite3.connect(path) as connection:
        for table in ("orders", "fills", "positions"):
            payload[table] = connection.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall()
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def test_versioned_migration_backs_up_and_preserves_legacy_rows(tmp_path: Path) -> None:
    database = tmp_path / "state" / "orizzonte.db"
    _legacy_database(database)
    before = _logical_digest(database)
    store = StateStore(database)
    store.initialize()

    backups = list(database.parent.glob(f"orizzonte.pre-v{LATEST_SCHEMA_VERSION}-*.bak"))
    assert len(backups) == 1
    assert _logical_digest(backups[0]) == before
    assert [item["client_order_id"] for item in store.orders()] == ["c1"]
    assert [item["fill_id"] for item in store.fills()] == ["f1"]
    assert [item["symbol"] for item in store.positions()] == ["BTC"]
    with sqlite3.connect(database) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == LATEST_SCHEMA_VERSION
        migration = connection.execute(
            "SELECT backup_path FROM schema_migrations WHERE version=?",
            (LATEST_SCHEMA_VERSION,),
        ).fetchone()
    assert Path(migration[0]) == backups[0]

    restored = tmp_path / "restored.db"
    shutil.copy2(backups[0], restored)
    assert _logical_digest(restored) == before


def test_execution_state_is_isolated_by_environment_and_account(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state.db")
    store.initialize()
    for environment, account in (
        (Environment.TESTNET, "0x" + "1" * 40),
        (Environment.MAINNET, "0x" + "2" * 40),
    ):
        store.upsert_order(
            "same-cloid",
            symbol="BTC",
            side="long",
            status="open",
            payload={"environment": environment.value},
            environment=environment,
            account_address=account,
        )
        store.record_fill(
            "same-fill",
            symbol="BTC",
            size=0.1,
            price=60_000,
            payload={"environment": environment.value},
            environment=environment,
            account_address=account,
        )
        store.upsert_position(
            "BTC",
            {"coin": "BTC", "environment": environment.value},
            environment=environment,
            account_address=account,
        )
    assert (
        store.orders(environment=Environment.TESTNET, account_address="0x" + "1" * 40)[0][
            "payload"
        ]["environment"]
        == "testnet"
    )
    assert (
        store.orders(environment=Environment.MAINNET, account_address="0x" + "2" * 40)[0][
            "payload"
        ]["environment"]
        == "mainnet"
    )
    assert store.orders(environment=Environment.MAINNET, account_address="0x" + "1" * 40) == []


def _certificate() -> Certificate:
    results = {name: True for name in REQUIRED_TESTNET_SCENARIOS}
    hashes = {
        name: hashlib.sha256(f"scenario:{name}".encode()).hexdigest()
        for name in REQUIRED_TESTNET_SCENARIOS
    }
    return Certificate.build(
        release_id="release-1",
        model_hash="a" * 64,
        gates_hash="b" * 64,
        account_address="0x" + "1" * 40,
        wallet_address="0x" + "2" * 40,
        evidence_hashes=("c" * 64, "d" * 64),
        required_scenarios=REQUIRED_TESTNET_SCENARIOS,
        scenario_results=results,
        scenario_hashes=hashes,
        chaos_report_hash="e" * 64,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def test_certificate_is_content_addressed_and_tamper_evident() -> None:
    certificate = _certificate()
    assert certificate.verify_content_address() is True
    assert certificate.model_copy(update={"model_hash": "f" * 64}).verify_content_address() is False
    tampered_results = certificate.scenario_results | {"duplicate_ws": False}
    assert (
        certificate.model_copy(
            update={"scenario_results": tampered_results}
        ).verify_content_address()
        is False
    )
    with pytest.raises(ValidationError, match="Every required testnet scenario must pass"):
        Certificate.model_validate(
            certificate.model_dump()
            | {
                "scenario_results": {
                    name: True for name in REQUIRED_TESTNET_SCENARIOS if name != "duplicate_ws"
                }
            }
        )


def test_chaos_runner_requires_every_scenario_and_detects_evidence_tamper() -> None:
    context = ChaosContext(
        lifecycle={
            "preflight": {"empty": True},
            "dead_man": {"status": "ok"},
            "entry": {"protections": [{"status": "ok"}, {"status": "ok"}]},
            "after_entry": {"positions": 1, "orders": 2},
            "protection_evidence": {
                "position": {"coin": "BTC", "szi": "0.1"},
                "orders": [
                    {
                        "coin": "BTC",
                        "side": "A",
                        "sz": "0.1",
                        "reduceOnly": True,
                        "orderType": "Stop Market",
                    },
                    {
                        "coin": "BTC",
                        "side": "A",
                        "sz": "0.1",
                        "reduceOnly": True,
                        "orderType": "Take Profit Market",
                    },
                ],
            },
            "final": {"positions": 0, "orders": 0},
        }
    )
    runner = ChaosRunner()
    report = runner.run(context)
    assert runner.verify(report) is True
    missing = dict(runner.scenarios)
    missing.pop("duplicate_ws")
    with pytest.raises(ChaosValidationError, match=r"missing=.*duplicate_ws"):
        ChaosRunner(missing).run(context)
    tampered = json.loads(json.dumps(report))
    tampered["evidence"]["partial_fill"]["filled"] = 99
    assert runner.verify(tampered) is False


def test_chaos_mutations_fail_gate_and_cannot_create_certificate(
    tmp_path: Path, monkeypatch
) -> None:
    context = ChaosContext(
        lifecycle={
            "preflight": {"empty": True},
            "dead_man": {"status": "ok"},
            "entry": {"protections": [{"status": "ok"}, {"status": "ok"}]},
            "after_entry": {"positions": 1, "orders": 2},
            "protection_evidence": {
                "position": {"coin": "BTC", "szi": "0.1"},
                "orders": [
                    {
                        "coin": "BTC",
                        "side": "A",
                        "sz": "0.1",
                        "reduceOnly": True,
                        "orderType": "Stop Market",
                    },
                    {
                        "coin": "BTC",
                        "side": "A",
                        "sz": "0.1",
                        "reduceOnly": True,
                        "orderType": "Take Profit Market",
                    },
                ],
            },
            "final": {"positions": 0, "orders": 0},
        }
    )
    store = StateStore(tmp_path / "state.db")
    store.initialize()
    mutations = (
        (
            "protection_size_for_fill",
            lambda **kwargs: kwargs["requested_size"],
        ),
        ("persist_once", lambda _event_id, _recorder: True),
        ("timeout_recovery_action", lambda **_kwargs: "submit"),
        ("enforce_time_guard", lambda **_kwargs: None),
        ("protection_failure_action", lambda **_kwargs: "continue"),
        ("has_native_protection_pair", lambda *_args, **_kwargs: True),
        ("dead_man_action", lambda **_kwargs: "clear"),
    )
    for primitive_name, unsafe_replacement in mutations:
        with monkeypatch.context() as patcher:
            patcher.setattr(primitives, primitive_name, unsafe_replacement)
            with pytest.raises(ChaosValidationError, match="failed"):
                ChaosRunner().run(context)
        assert store.latest_testnet_certificate() is None


def test_smoke_mutation_refuses_certificate_before_persistence(
    app_paths, settings, monkeypatch
) -> None:
    store = StateStore(app_paths.database)
    store.initialize()
    controller = AgentController(app_paths, settings, store)
    gateway = object.__new__(HyperliquidGateway)
    gateway.environment = Environment.TESTNET
    flat = AccountSnapshot(
        equity=100,
        withdrawable=100,
        positions=(),
        open_orders=(),
        mids={"BTC": 60_000},
    )
    opened = AccountSnapshot(
        equity=100,
        withdrawable=100,
        positions=({"coin": "BTC", "szi": "0.001"},),
        open_orders=(
            {
                "coin": "BTC",
                "side": "A",
                "sz": "0.001",
                "reduceOnly": True,
                "orderType": "Stop Market",
            },
            {
                "coin": "BTC",
                "side": "A",
                "sz": "0.001",
                "reduceOnly": True,
                "orderType": "Take Profit Market",
            },
        ),
        mids={"BTC": 60_000},
    )
    reconciliations = iter((opened, opened, flat))
    monkeypatch.setattr(controller, "gateway", lambda _environment: gateway)
    monkeypatch.setattr(
        controller,
        "testnet_preflight",
        lambda: {
            "empty": True,
            "account_address": "0x" + "1" * 40,
            "wallet_address": "0x" + "2" * 40,
        },
    )
    monkeypatch.setattr(gateway, "snapshot", lambda: flat)
    monkeypatch.setattr(
        gateway,
        "asset_metadata",
        lambda _symbol: AssetMetadata(
            symbol="BTC",
            size_decimals=5,
            size_increment=0.00001,
            max_price_decimals=1,
        ),
    )
    monkeypatch.setattr(gateway, "schedule_dead_man", lambda _seconds: {"status": "ok"})
    monkeypatch.setattr(
        gateway,
        "place_entry_with_protection",
        lambda *_args, **_kwargs: {"protections": [{"status": "ok"}, {"status": "ok"}]},
    )
    monkeypatch.setattr(gateway, "reconcile", lambda: next(reconciliations))
    monkeypatch.setattr(gateway, "flatten_all", lambda **_kwargs: [{"status": "ok"}])
    monkeypatch.setattr(primitives, "enforce_time_guard", lambda **_kwargs: None)

    with pytest.raises(ControlError, match="chaos gate"):
        controller.testnet_smoke(
            budget_usdc=25,
            confirmation="TESTNET SMOKE 25.00",
        )
    assert store.latest_testnet_certificate() is None


def test_mainnet_authorization_consumption_is_atomic_one_shot_and_bound(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state.db")
    store.initialize()
    certificate = _certificate()
    store.save_testnet_certificate(certificate)
    issued = datetime.now(UTC)
    authorization = MainnetAuthorization(
        authorization_id="a" * 32,
        release_id=certificate.release_id,
        certificate_id=certificate.certificate_id,
        model_hash=certificate.model_hash,
        gates_hash=certificate.gates_hash,
        git_commit="1" * 40,
        config_sha256="2" * 64,
        config_fingerprint="3" * 64,
        code_hash="4" * 64,
        account_address="0x" + "3" * 40,
        wallet_address="0x" + "4" * 40,
        budget_usdc=500,
        issued_at=issued,
        expires_at=issued + timedelta(minutes=15),
    )
    store.save_mainnet_authorization(authorization, token_hash="token-hash")
    kwargs = {
        "token_hash": "token-hash",
        "session_id": "session-1",
        "release_id": authorization.release_id,
        "certificate_id": authorization.certificate_id,
        "model_hash": authorization.model_hash,
        "gates_hash": authorization.gates_hash,
        "git_commit": authorization.git_commit,
        "config_sha256": authorization.config_sha256,
        "config_fingerprint": authorization.config_fingerprint,
        "code_hash": authorization.code_hash,
        "account_address": authorization.account_address,
        "wallet_address": authorization.wallet_address,
        "budget_usdc": authorization.budget_usdc,
    }
    with pytest.raises(ValueError, match="config_sha256"):
        store.consume_mainnet_authorization(
            authorization.authorization_id,
            **(kwargs | {"config_sha256": "f" * 64}),
        )
    assert store.authorization(authorization.authorization_id)["consumed_at"] is None
    tampered_payload = authorization.model_dump(mode="json") | {"code_hash": "f" * 64}
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "UPDATE mainnet_authorizations SET payload=? WHERE authorization_id=?",
            (json.dumps(tampered_payload), authorization.authorization_id),
        )
    with pytest.raises(ValueError, match="Payload mainnet divergente: code_hash"):
        store.consume_mainnet_authorization(authorization.authorization_id, **kwargs)
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "UPDATE mainnet_authorizations SET payload=? WHERE authorization_id=?",
            (authorization.model_dump_json(), authorization.authorization_id),
        )
    assert store.consume_mainnet_authorization(authorization.authorization_id, **kwargs)
    with pytest.raises(ValueError, match="consumida"):
        store.consume_mainnet_authorization(authorization.authorization_id, **kwargs)
    with pytest.raises(ValidationError):
        MainnetAuthorization.model_validate(authorization.model_dump() | {"budget_usdc": 501})


def test_environment_wallets_generate_rotate_and_never_reuse(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(secret_module, "protect", lambda data, *, entropy: data)
    monkeypatch.setattr(secret_module, "unprotect", lambda data, *, entropy: data)
    manager = EnvironmentSecretManager(tmp_path)
    account = "0x" + "1" * 40
    first_key = Account.create().key.hex()
    second_key = Account.create().key.hex()
    created = manager.generate(
        Environment.TESTNET,
        secret_key=first_key,
        account_address=account,
    )
    assert created["configured"] is True
    assert "secret_key" not in created
    rotated = manager.rotate(
        Environment.TESTNET,
        secret_key=second_key,
        account_address=account,
    )
    assert rotated["previous"]["fingerprint"] != rotated["current"]["fingerprint"]
    with pytest.raises(SecretStoreError, match="reutilizada"):
        manager.rotate(
            Environment.TESTNET,
            secret_key=first_key,
            account_address=account,
        )
    assert manager.path_for(Environment.TESTNET) != manager.path_for(Environment.MAINNET)


def test_dpapi_capability_never_exposes_plaintext_token(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(secret_module, "protect", lambda data, *, entropy: data)
    monkeypatch.setattr(secret_module, "unprotect", lambda data, *, entropy: data)
    authorization = MainnetAuthorization(
        authorization_id="f" * 32,
        release_id="release-1",
        certificate_id=_certificate().certificate_id,
        model_hash="a" * 64,
        gates_hash="b" * 64,
        git_commit="1" * 40,
        config_sha256="2" * 64,
        config_fingerprint="3" * 64,
        code_hash="4" * 64,
        account_address="0x" + "3" * 40,
        wallet_address="0x" + "4" * 40,
        budget_usdc=100,
        issued_at=datetime.now(UTC),
        expires_at=datetime.now(UTC) + timedelta(minutes=15),
    )
    capabilities = DPAPICapabilityStore(tmp_path)
    token_hash = capabilities.issue(authorization)
    assert len(token_hash) == 64
    assert "token" not in capabilities.status(authorization.authorization_id)
    assert capabilities.load(authorization.authorization_id)["token_hash"] == token_hash
    capabilities.delete(authorization.authorization_id)
    assert capabilities.status(authorization.authorization_id)["available"] is False


def test_controller_consumes_dpapi_capability_once_with_all_bindings(
    app_paths, settings, monkeypatch
) -> None:
    monkeypatch.setattr(secret_module, "protect", lambda data, *, entropy: data)
    monkeypatch.setattr(secret_module, "unprotect", lambda data, *, entropy: data)
    store = StateStore(app_paths.database)
    store.initialize()
    controller = AgentController(app_paths, settings, store)
    certificate = _certificate()
    store.save_testnet_certificate(certificate)
    gate_path = app_paths.reports / "authorization-gate.json"
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    gate_path.write_text(
        json.dumps(
            {
                "release_binding": {
                    "commit_hash": "a" * 40,
                    "config_fingerprint": "3" * 64,
                    "code_hash": "4" * 64,
                }
            }
        ),
        encoding="utf-8",
    )
    release = ReleaseManifest(
        release_id=certificate.release_id,
        created_at=datetime.now(UTC),
        git_commit="a" * 40,
        artifacts={
            "model": {"sha256": certificate.model_hash, "path": "model"},
            "config": {"sha256": "2" * 64, "path": "config"},
            "research_approval": {
                "sha256": certificate.gates_hash,
                "path": str(gate_path),
            },
        },
        approval_passed=True,
        approved=True,
    )
    issued = datetime.now(UTC)
    authorization = MainnetAuthorization(
        authorization_id="e" * 32,
        release_id=release.release_id,
        certificate_id=certificate.certificate_id,
        model_hash=certificate.model_hash,
        gates_hash=certificate.gates_hash,
        git_commit=release.git_commit,
        config_sha256="2" * 64,
        config_fingerprint="3" * 64,
        code_hash="4" * 64,
        account_address="0x" + "3" * 40,
        wallet_address="0x" + "4" * 40,
        budget_usdc=100,
        issued_at=issued,
        expires_at=issued + timedelta(minutes=15),
    )
    token_hash = controller.capabilities.issue(authorization)
    store.save_mainnet_authorization(authorization, token_hash=token_hash)
    store.set("pending_mainnet_authorization", {"authorization_id": authorization.authorization_id})

    capability = controller.capabilities.load(authorization.authorization_id)
    with monkeypatch.context() as patcher:
        patcher.setattr(
            controller.capabilities,
            "load",
            lambda _authorization_id: capability | {"config_sha256": "f" * 64},
        )
        with pytest.raises(ControlError, match="Binding DPAPI divergente: config_sha256"):
            controller._consume_mainnet_capability(
                release=release,
                budget_usdc=100,
                account_address=authorization.account_address,
                wallet_address=authorization.wallet_address,
                session_id="tampered-session",
            )
    assert store.authorization(authorization.authorization_id)["consumed_at"] is None
    consumed = controller._consume_mainnet_capability(
        release=release,
        budget_usdc=100,
        account_address=authorization.account_address,
        wallet_address=authorization.wallet_address,
        session_id="session-1",
    )
    assert consumed.authorization_id == authorization.authorization_id
    assert store.authorization(authorization.authorization_id)["consumed_at"] is not None
    assert controller.capabilities.status(authorization.authorization_id)["available"] is False
    with pytest.raises(ControlError, match="ausente"):
        controller._consume_mainnet_capability(
            release=release,
            budget_usdc=100,
            account_address=authorization.account_address,
            wallet_address=authorization.wallet_address,
            session_id="session-2",
        )


def test_mainnet_restart_pauses_and_state_exposes_operational_safety_metadata(
    app_paths, settings, monkeypatch
) -> None:
    account = "0x" + "3" * 40
    store = StateStore(app_paths.database)
    store.initialize()
    store.save_agent_state(
        AgentState(
            status=AgentStatus.RUNNING,
            environment=Environment.MAINNET,
            account_address=account,
            budget_usdc=100,
        )
    )
    store.upsert_position(
        "BTC",
        {"coin": "BTC", "szi": "0.01"},
        environment=Environment.MAINNET,
        account_address=account,
    )
    for client_order_id, kind in (("sl-1", "stop_loss"), ("tp-1", "take_profit")):
        store.upsert_order(
            client_order_id,
            symbol="BTC",
            side="short",
            status="open",
            payload={"kind": kind, "reduceOnly": True},
            environment=Environment.MAINNET,
            account_address=account,
        )

    async def idle_stream(_: HyperliquidStream) -> None:
        await asyncio.Event().wait()

    monkeypatch.setattr(HyperliquidStream, "run", idle_stream)
    with TestClient(create_app(app_paths, settings)) as client:
        state = client.get("/state").json()
    assert state["status"] == "paused"
    assert state["metadata"]["requires_mainnet_reauthorization"] is True
    assert state["metadata"]["testnet_certificate"]["available"] is False
    assert state["metadata"]["mainnet_authorization"]["locked"] is True
    management = state["metadata"]["protection_management"]
    assert management["mode"] == "manage_only"
    assert management["protected_symbols"] == ["BTC"]
    assert management["unprotected_symbols"] == []


def test_dead_man_clears_only_for_sized_opposite_sl_and_tp_pair() -> None:
    position = {"coin": "BTC", "szi": "0.1"}
    stop = {
        "coin": "BTC",
        "side": "A",
        "sz": "0.1",
        "reduceOnly": True,
        "orderType": "Stop Market",
    }
    take_profit = stop | {"orderType": "Take Profit Market"}

    def snapshot(*orders: dict[str, object]) -> AccountSnapshot:
        return AccountSnapshot(
            equity=100,
            withdrawable=100,
            positions=(position,),
            open_orders=orders,
            mids={"BTC": 60_000},
        )

    assert _positions_are_protected(snapshot(stop, take_profit)) is True
    assert _positions_are_protected(snapshot(stop, stop | {"oid": 2})) is False
    assert _positions_are_protected(snapshot(stop, take_profit | {"side": "B"})) is False
    assert _positions_are_protected(snapshot(stop, take_profit | {"sz": "0.09"})) is False
