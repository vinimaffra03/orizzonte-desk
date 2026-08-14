from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import subprocess
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from orizzonte_desk.config import OperationsConfig
from orizzonte_desk.ops import (
    TASK_DAEMON,
    TASK_WATCHDOG,
    OperationsError,
    OperationsManager,
    configure_rotating_logging,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FakeScheduler:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.installed: set[str] = set()

    def __call__(self, command: Any) -> subprocess.CompletedProcess[str]:
        call = tuple(str(item) for item in command)
        self.calls.append(call)
        action = call[1]
        name = call[call.index("/TN") + 1]
        if action == "/Create":
            self.installed.add(name)
            return subprocess.CompletedProcess(call, 0, "created", "")
        if action == "/Delete":
            existed = name in self.installed
            self.installed.discard(name)
            return subprocess.CompletedProcess(call, 0 if existed else 1, "deleted", "")
        installed = name in self.installed
        return subprocess.CompletedProcess(
            call, 0 if installed else 1, "ready" if installed else "", "missing"
        )


def prepare_executables(root: Path) -> None:
    scripts = root / ".venv" / "Scripts"
    scripts.mkdir(parents=True)
    (scripts / "python.exe").write_bytes(b"fixture")
    (scripts / "orizzonte.exe").write_bytes(b"fixture")


def test_scheduler_requires_explicit_confirmation_and_uses_project_runtime(
    app_paths: Any,
) -> None:
    prepare_executables(app_paths.root)
    scheduler = FakeScheduler()
    manager = OperationsManager(
        app_paths,
        OperationsConfig(),
        runner=scheduler,
        platform_name="nt",
    )

    with pytest.raises(OperationsError, match="INSTALL ORIZZONTE TASKS"):
        manager.install_tasks("yes")
    assert scheduler.calls == []

    installed = manager.install_tasks("INSTALL ORIZZONTE TASKS")
    assert {item.name for item in installed if item.installed} == {TASK_DAEMON, TASK_WATCHDOG}
    create_calls = [call for call in scheduler.calls if call[1] == "/Create"]
    assert len(create_calls) == 2
    actions = [call[call.index("/TR") + 1] for call in create_calls]
    assert all(str(app_paths.root) in action for action in actions)
    assert all("$env:TEMP=" in action and "$env:TMP=" in action for action in actions)
    assert all("$env:UV_CACHE_DIR=" in action for action in actions)
    assert all("-WindowStyle Hidden" in action for action in actions)
    assert any("orizzonte_desk.ops" in " ".join(call) for call in create_calls)

    removed = manager.uninstall_tasks("REMOVE ORIZZONTE TASKS")
    assert not any(item.installed for item in removed)


def test_scheduler_status_is_read_only_and_non_windows_fails_closed(app_paths: Any) -> None:
    scheduler = FakeScheduler()
    manager = OperationsManager(
        app_paths,
        OperationsConfig(),
        runner=scheduler,
        platform_name="posix",
    )
    with pytest.raises(OperationsError, match="Windows Task Scheduler"):
        manager.task_status()
    assert scheduler.calls == []


def test_watchdog_locks_on_stale_heartbeat_without_real_state_mutation(app_paths: Any) -> None:
    now = datetime(2026, 8, 14, 12, tzinfo=UTC)
    reasons: list[str] = []

    def probe() -> dict[str, Any]:
        return {
            "status": "ok",
            "agent": {
                "status": "running",
                "last_heartbeat": (now - timedelta(seconds=91)).isoformat(),
            },
        }

    manager = OperationsManager(
        app_paths,
        OperationsConfig(watchdog_interval_seconds=30),
        probe=probe,
        fail_closed_action=lambda reason: not reasons.append(reason),
    )
    result = manager.watchdog_once(now=now)
    assert not result.healthy and result.fail_closed
    assert "stale" in result.reason
    assert reasons and not app_paths.database.exists()


def test_watchdog_healthy_and_unconfirmed_lock_paths(app_paths: Any) -> None:
    now = datetime(2026, 8, 14, 12, tzinfo=UTC)
    healthy = OperationsManager(
        app_paths,
        OperationsConfig(),
        probe=lambda: {
            "status": "ok",
            "agent": {"status": "running", "last_heartbeat": now.isoformat()},
        },
        fail_closed_action=lambda _: pytest.fail("fail-closed should not run"),
    ).watchdog_once(now=now)
    assert healthy.healthy and not healthy.fail_closed

    manager = OperationsManager(
        app_paths,
        OperationsConfig(),
        probe=lambda: (_ for _ in ()).throw(TimeoutError("offline")),
        fail_closed_action=lambda _: False,
    )
    with pytest.raises(OperationsError, match="bloqueio local não confirmado"):
        manager.watchdog_once(now=now)


def test_rotating_logging_is_local_and_not_duplicated(app_paths: Any) -> None:
    logger_name = f"orizzonte-test-{id(app_paths)}"
    logger = configure_rotating_logging(app_paths, OperationsConfig(), logger_name=logger_name)
    same_logger = configure_rotating_logging(app_paths, OperationsConfig(), logger_name=logger_name)
    handlers = [handler for handler in logger.handlers if hasattr(handler, "baseFilename")]
    assert same_logger is logger and len(handlers) == 1
    assert Path(handlers[0].baseFilename).parent == app_paths.logs.resolve()
    assert handlers[0].formatter is not None
    assert handlers[0].formatter.converter is time.gmtime
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    logging.Logger.manager.loggerDict.pop(logger_name, None)


def prepare_backup_source(app_paths: Any) -> None:
    app_paths.ensure()
    app_paths.config.parent.mkdir(parents=True, exist_ok=True)
    app_paths.config.write_text("[app]\nhost='127.0.0.1'\n", encoding="utf-8")
    with sqlite3.connect(app_paths.database) as connection:
        connection.execute("CREATE TABLE ledger(id INTEGER PRIMARY KEY, value TEXT)")
        connection.execute("INSERT INTO ledger(value) VALUES ('safe')")
    model = app_paths.models / "promoted.joblib"
    model.write_bytes(b"model")
    pointer = app_paths.models / "promoted.json"
    pointer.write_text('{"model":"fixture"}', encoding="utf-8")
    approval = app_paths.reports / "live-approval.json"
    approval.parent.mkdir(parents=True, exist_ok=True)
    approval.write_text('{"passed":true}', encoding="utf-8")
    release_dir = app_paths.reports / "releases" / "release-test"
    release_dir.mkdir(parents=True)
    release_manifest = release_dir / "manifest.json"
    release_manifest.write_text(
        json.dumps(
            {
                "release_id": "release-test",
                "artifacts": {
                    "config": {
                        "path": str(app_paths.config.resolve()),
                        "sha256": sha256(app_paths.config),
                    },
                    "model": {"path": str(model.resolve()), "sha256": sha256(model)},
                    "model_pointer": {"path": str(pointer.resolve()), "sha256": sha256(pointer)},
                    "research_approval": {
                        "path": str(approval.resolve()),
                        "sha256": sha256(approval),
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    approved = app_paths.reports / "releases" / "approved.json"
    approved.write_text(
        json.dumps(
            {
                "release_id": "release-test",
                "manifest": str(release_manifest.resolve()),
                "manifest_sha256": sha256(release_manifest),
            }
        ),
        encoding="utf-8",
    )
    app_paths.secret_file.write_bytes(b"must-not-be-backed-up")


def test_backup_and_restore_dry_run_are_local_verified_and_non_mutating(
    app_paths: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepare_backup_source(app_paths)
    source_hashes = {path: sha256(path) for path in (app_paths.database, app_paths.config)}
    minimums: list[float] = []
    monkeypatch.setattr(
        type(app_paths), "assert_free_space", lambda _self, minimum: minimums.append(minimum)
    )
    manager = OperationsManager(app_paths, OperationsConfig(), platform_name="posix")

    backup = manager.backup()
    assert minimums == [20.0]
    assert backup.release_present and backup.files >= 8
    backup_path = Path(backup.path)
    manifest_text = (backup_path / "manifest.json").read_text(encoding="utf-8")
    assert ".secrets" not in manifest_text and "hyperliquid.dpapi" not in manifest_text

    result = manager.restore_dry_run(backup.backup_id)
    assert result.valid and result.sqlite_integrity == "ok" and result.release_present
    assert source_hashes == {path: sha256(path) for path in source_hashes}
    with sqlite3.connect(app_paths.database) as connection:
        assert connection.execute("SELECT value FROM ledger").fetchone() == ("safe",)


def test_restore_dry_run_rejects_tampering_and_path_traversal(app_paths: Any) -> None:
    prepare_backup_source(app_paths)
    manager = OperationsManager(app_paths, OperationsConfig(), platform_name="posix")
    backup = manager.backup()
    (Path(backup.path) / "config" / "settings.toml").write_text("tampered", encoding="utf-8")
    with pytest.raises(OperationsError, match="Checksum divergente"):
        manager.restore_dry_run(backup.backup_id)
    with pytest.raises(OperationsError, match="inválido ou inexistente"):
        manager.restore_dry_run("../outside")


def test_backup_retention_deletes_only_old_backup_directories(app_paths: Any) -> None:
    prepare_backup_source(app_paths)
    backup_root = app_paths.root / "backups"
    backup_root.mkdir()
    for index in range(7):
        (backup_root / f"backup-2026010{index}T000000000000Z").mkdir()
    unrelated = backup_root / "keep-me"
    unrelated.mkdir()
    manager = OperationsManager(
        app_paths,
        OperationsConfig(backup_retention=7),
        platform_name="posix",
    )
    result = manager.backup()
    assert len(result.removed_by_retention) == 1
    assert unrelated.is_dir()
    assert len(list(backup_root.glob("backup-*"))) == 7
