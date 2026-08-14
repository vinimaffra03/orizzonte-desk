from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import sqlite3
import subprocess
import time
import tomllib
from collections.abc import Callable, Sequence
from contextlib import closing
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Protocol, cast

import httpx

from orizzonte_desk.config import OperationsConfig, Settings
from orizzonte_desk.models import AgentStatus
from orizzonte_desk.paths import AppPaths
from orizzonte_desk.storage import StateStore

TASK_DAEMON = r"\OrizzonteDesk\Daemon"
TASK_WATCHDOG = r"\OrizzonteDesk\Watchdog"
TASK_NAMES = (TASK_DAEMON, TASK_WATCHDOG)


class OperationsError(RuntimeError):
    """An operational action could not be proven safe or complete."""


class CommandRunner(Protocol):
    def __call__(self, command: Sequence[str]) -> subprocess.CompletedProcess[str]: ...


class WatchdogProbe(Protocol):
    def __call__(self) -> dict[str, Any]: ...


class FailClosedAction(Protocol):
    def __call__(self, reason: str) -> bool: ...


@dataclass(frozen=True, slots=True)
class TaskStatus:
    name: str
    installed: bool
    detail: str


@dataclass(frozen=True, slots=True)
class WatchdogResult:
    healthy: bool
    fail_closed: bool
    reason: str
    checked_at: str


@dataclass(frozen=True, slots=True)
class BackupResult:
    backup_id: str
    path: str
    files: int
    release_present: bool
    removed_by_retention: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RestoreDryRunResult:
    backup_id: str
    valid: bool
    files_checked: int
    sqlite_integrity: str
    release_present: bool
    checked_at: str


def _run_command(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, check=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sqlite_snapshot(source: Path, destination: Path) -> None:
    source_uri = f"file:{source.resolve().as_posix()}?mode=ro"
    with (
        closing(sqlite3.connect(source_uri, uri=True)) as source_connection,
        closing(sqlite3.connect(destination)) as destination_connection,
    ):
        source_connection.backup(destination_connection)
        result = destination_connection.execute("PRAGMA integrity_check").fetchone()
        if not result or result[0] != "ok":
            raise OperationsError(f"Snapshot SQLite inválido: {result}")


def configure_rotating_logging(
    paths: AppPaths,
    operations: OperationsConfig,
    *,
    logger_name: str = "orizzonte_desk",
) -> logging.Logger:
    """Configure one project-local rotating handler without duplicating it."""
    paths.logs.mkdir(parents=True, exist_ok=True)
    log_path = (paths.logs / "orizzonte.log").resolve()
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    for handler in logger.handlers:
        if isinstance(handler, RotatingFileHandler) and Path(handler.baseFilename) == log_path:
            return logger
    handler = RotatingFileHandler(
        log_path,
        maxBytes=operations.log_max_bytes,
        backupCount=operations.log_backup_count,
        encoding="utf-8",
        delay=True,
    )
    formatter = logging.Formatter(
        "%(asctime)sZ %(levelname)s %(name)s %(message)s", "%Y-%m-%dT%H:%M:%S"
    )
    formatter.converter = time.gmtime
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


class LoopbackProbe:
    def __init__(self, base_url: str = "http://127.0.0.1:8790", timeout: float = 3.0) -> None:
        self.base_url = base_url
        self.timeout = timeout

    def __call__(self) -> dict[str, Any]:
        response = httpx.get(f"{self.base_url}/health", timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise OperationsError("Health payload inválido")
        return cast(dict[str, Any], payload)


class LocalFailClosed:
    """Persist a lock so a recovered daemon cannot resume unattended."""

    def __init__(self, paths: AppPaths) -> None:
        self.paths = paths

    def __call__(self, reason: str) -> bool:
        try:
            store = StateStore(self.paths.database)
            store.initialize()
            state = store.agent_state()
            if state.status in {AgentStatus.RUNNING, AgentStatus.ARMED, AgentStatus.PAUSED}:
                store.save_agent_state(state.model_copy(update={"status": AgentStatus.LOCKED}))
            store.latch_lock("connectivity", reason=reason, payload={"source": "ops_watchdog"})
            store.event(
                "watchdog",
                "Watchdog aplicou bloqueio fail-closed",
                level="CRITICAL",
                payload={"reason": reason},
            )
            return True
        except Exception:
            return False


class OperationsManager:
    """Explicit local operations; construction itself has no side effects."""

    def __init__(
        self,
        paths: AppPaths,
        operations: OperationsConfig,
        *,
        runner: CommandRunner = _run_command,
        probe: WatchdogProbe | None = None,
        fail_closed_action: FailClosedAction | None = None,
        platform_name: str | None = None,
        sqlite_snapshot: Callable[[Path, Path], None] = _sqlite_snapshot,
    ) -> None:
        self.paths = paths
        self.operations = operations
        self.runner = runner
        self.probe = probe or LoopbackProbe()
        self.fail_closed_action = fail_closed_action or LocalFailClosed(paths)
        self.platform_name = platform_name or os.name
        self.sqlite_snapshot = sqlite_snapshot
        self.backup_root = paths.root / "backups"

    def install_tasks(self, confirmation: str) -> tuple[TaskStatus, ...]:
        """Install only after the exact operator confirmation."""
        expected = "INSTALL ORIZZONTE TASKS"
        if confirmation != expected:
            raise OperationsError(f"Confirmação inválida. Digite exatamente: {expected}")
        self._require_windows()
        python = self.paths.root / ".venv" / "Scripts" / "python.exe"
        executable = self.paths.root / ".venv" / "Scripts" / "orizzonte.exe"
        missing = [str(path) for path in (python, executable) if not path.is_file()]
        if missing:
            raise OperationsError(f"Executáveis locais ausentes: {', '.join(missing)}")
        daemon_action = self._powershell_task_action(executable, "daemon")
        watchdog_action = self._powershell_task_action(
            python,
            "-m",
            "orizzonte_desk.ops",
            "--root",
            str(self.paths.root),
            "watchdog",
        )
        commands = (
            (
                "schtasks.exe",
                "/Create",
                "/TN",
                TASK_DAEMON,
                "/TR",
                daemon_action,
                "/SC",
                "ONLOGON",
                "/RL",
                "LIMITED",
                "/F",
            ),
            (
                "schtasks.exe",
                "/Create",
                "/TN",
                TASK_WATCHDOG,
                "/TR",
                watchdog_action,
                "/SC",
                "ONLOGON",
                "/DELAY",
                "0000:30",
                "/RL",
                "LIMITED",
                "/F",
            ),
        )
        for command in commands:
            result = self.runner(command)
            if result.returncode != 0:
                raise OperationsError(result.stderr.strip() or f"Falha ao instalar {command[3]}")
        return self.task_status()

    def task_status(self) -> tuple[TaskStatus, ...]:
        self._require_windows()
        statuses: list[TaskStatus] = []
        for name in TASK_NAMES:
            result = self.runner(("schtasks.exe", "/Query", "/TN", name, "/FO", "LIST", "/V"))
            detail = (result.stdout if result.returncode == 0 else result.stderr).strip()
            statuses.append(TaskStatus(name, result.returncode == 0, detail))
        return tuple(statuses)

    def uninstall_tasks(self, confirmation: str) -> tuple[TaskStatus, ...]:
        expected = "REMOVE ORIZZONTE TASKS"
        if confirmation != expected:
            raise OperationsError(f"Confirmação inválida. Digite exatamente: {expected}")
        self._require_windows()
        for name in TASK_NAMES:
            result = self.runner(("schtasks.exe", "/Delete", "/TN", name, "/F"))
            # schtasks returns 1 when an already absent task is deleted; absence is the goal.
            if result.returncode not in {0, 1}:
                raise OperationsError(result.stderr.strip() or f"Falha ao remover {name}")
        return self.task_status()

    def watchdog_once(self, *, now: datetime | None = None) -> WatchdogResult:
        checked_at = (now or datetime.now(UTC)).astimezone(UTC)
        try:
            payload = self.probe()
            if payload.get("status") != "ok":
                raise OperationsError("daemon retornou health diferente de ok")
            agent = payload.get("agent", {})
            if not isinstance(agent, dict):
                raise OperationsError("health sem estado do agente")
            heartbeat = agent.get("last_heartbeat")
            if agent.get("status") == AgentStatus.RUNNING.value:
                if not isinstance(heartbeat, str):
                    raise OperationsError("agente running sem heartbeat")
                heartbeat_at = datetime.fromisoformat(heartbeat.replace("Z", "+00:00"))
                age = (checked_at - heartbeat_at.astimezone(UTC)).total_seconds()
                if age > self.operations.watchdog_interval_seconds * 3:
                    raise OperationsError(f"heartbeat stale há {age:.1f}s")
            return WatchdogResult(True, False, "healthy", checked_at.isoformat())
        except Exception as exc:
            reason = f"Watchdog: {exc}"
            applied = self.fail_closed_action(reason)
            if not applied:
                raise OperationsError(f"{reason}; bloqueio local não confirmado") from exc
            return WatchdogResult(False, True, reason, checked_at.isoformat())

    def watchdog_forever(self) -> None:
        logger = configure_rotating_logging(
            self.paths, self.operations, logger_name="orizzonte.ops"
        )
        while True:
            try:
                result = self.watchdog_once()
                logger.log(logging.INFO if result.healthy else logging.CRITICAL, result.reason)
            except Exception:
                logger.exception("Watchdog não conseguiu confirmar fail-closed")
            time.sleep(self.operations.watchdog_interval_seconds)

    def backup(self) -> BackupResult:
        self.paths.assert_free_space(20.0)
        self._validate_backup_root()
        required = {
            "state/orizzonte.db": self.paths.database,
            "config/settings.toml": self.paths.config,
        }
        missing = [str(path) for path in required.values() if not path.is_file()]
        if missing:
            raise OperationsError(f"Backup recusado; arquivos ausentes: {', '.join(missing)}")
        now = datetime.now(UTC)
        backup_id = now.strftime("backup-%Y%m%dT%H%M%S%fZ")
        staging = self.backup_root / f".{backup_id}.staging"
        destination = self.backup_root / backup_id
        staging.mkdir(parents=True, exist_ok=False)
        artifacts: list[dict[str, Any]] = []
        release_present = False
        try:
            database_destination = staging / "state" / "orizzonte.db"
            database_destination.parent.mkdir(parents=True)
            self.sqlite_snapshot(self.paths.database, database_destination)
            self._record(artifacts, staging, database_destination, self.paths.database)
            config_destination = staging / "config" / "settings.toml"
            config_destination.parent.mkdir(parents=True)
            shutil.copy2(self.paths.config, config_destination)
            self._record(artifacts, staging, config_destination, self.paths.config)
            release_present = self._copy_release(staging, artifacts)
            manifest = {
                "schema_version": 1,
                "backup_id": backup_id,
                "created_at": now.isoformat(),
                "project_root": str(self.paths.root.resolve()),
                "release_present": release_present,
                "artifacts": artifacts,
            }
            (staging / "manifest.json").write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            self._promote_staging(staging, destination)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        removed = self._apply_retention()
        return BackupResult(backup_id, str(destination), len(artifacts), release_present, removed)

    def restore_dry_run(self, backup_id: str) -> RestoreDryRunResult:
        """Validate a backup without opening any project destination for writing."""
        directory = (self.backup_root / backup_id).resolve()
        if directory.parent != self.backup_root.resolve() or not directory.is_dir():
            raise OperationsError(f"Backup inválido ou inexistente: {backup_id}")
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            raise OperationsError("Manifesto do backup ausente")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("backup_id") != backup_id or manifest.get("schema_version") != 1:
            raise OperationsError("Identidade ou schema do backup inválido")
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, list):
            raise OperationsError("Lista de artefatos inválida")
        checked = 0
        for artifact in artifacts:
            if not isinstance(artifact, dict):
                raise OperationsError("Entrada de artefato inválida")
            relative = Path(str(artifact.get("path", "")))
            candidate = (directory / relative).resolve()
            if not candidate.is_relative_to(directory) or not candidate.is_file():
                raise OperationsError(f"Artefato ausente ou fora do backup: {relative}")
            if _sha256(candidate) != artifact.get("sha256"):
                raise OperationsError(f"Checksum divergente: {relative}")
            checked += 1
        config_path = directory / "config" / "settings.toml"
        with config_path.open("rb") as handle:
            tomllib.load(handle)
        database = directory / "state" / "orizzonte.db"
        database_uri = f"file:{database.resolve().as_posix()}?mode=ro&immutable=1"
        with closing(sqlite3.connect(database_uri, uri=True)) as connection:
            row = connection.execute("PRAGMA integrity_check").fetchone()
        integrity = str(row[0]) if row else "missing"
        if integrity != "ok":
            raise OperationsError(f"Integridade SQLite falhou: {integrity}")
        return RestoreDryRunResult(
            backup_id=backup_id,
            valid=True,
            files_checked=checked,
            sqlite_integrity=integrity,
            release_present=bool(manifest.get("release_present")),
            checked_at=datetime.now(UTC).isoformat(),
        )

    def _copy_release(self, staging: Path, artifacts: list[dict[str, Any]]) -> bool:
        pointer = self.paths.reports / "releases" / "approved.json"
        if not pointer.is_file():
            return False
        payload = json.loads(pointer.read_text(encoding="utf-8"))
        manifest_path = Path(str(payload.get("manifest", ""))).resolve()
        self._assert_project_file(manifest_path)
        if _sha256(manifest_path) != payload.get("manifest_sha256"):
            raise OperationsError("Checksum do manifesto aprovado diverge do ponteiro")
        release_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        release_id = str(release_manifest.get("release_id", ""))
        if not release_id or payload.get("release_id") != release_id:
            raise OperationsError("Ponteiro de release inconsistente")
        files: dict[str, Path] = {
            "release/approved.json": pointer,
            "release/manifest.json": manifest_path,
        }
        release_artifacts = release_manifest.get("artifacts", {})
        if not isinstance(release_artifacts, dict):
            raise OperationsError("Artefatos da release inválidos")
        for name, item in release_artifacts.items():
            if not isinstance(item, dict):
                raise OperationsError(f"Artefato de release inválido: {name}")
            source = Path(str(item.get("path", ""))).resolve()
            self._assert_project_file(source)
            if _sha256(source) != item.get("sha256"):
                raise OperationsError(f"Artefato da release alterado: {name}")
            suffix = "".join(source.suffixes)
            files[f"release/artifacts/{name}{suffix}"] = source
        for relative, source in files.items():
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            self._record(artifacts, staging, destination, source)
        return True

    def _record(
        self, artifacts: list[dict[str, Any]], staging: Path, destination: Path, source: Path
    ) -> None:
        artifacts.append(
            {
                "path": destination.relative_to(staging).as_posix(),
                "source": str(source.resolve()),
                "sha256": _sha256(destination),
                "size": destination.stat().st_size,
            }
        )

    def _apply_retention(self) -> tuple[str, ...]:
        backups = sorted(
            (path for path in self.backup_root.glob("backup-*") if path.is_dir()),
            key=lambda path: path.name,
            reverse=True,
        )
        removed: list[str] = []
        for path in backups[self.operations.backup_retention :]:
            resolved = path.resolve()
            if resolved.parent != self.backup_root.resolve():
                raise OperationsError(f"Retenção recusou caminho inesperado: {resolved}")
            shutil.rmtree(resolved)
            removed.append(path.name)
        return tuple(removed)

    @staticmethod
    def _promote_staging(staging: Path, destination: Path) -> None:
        """Commit a Windows backup by publishing its manifest last."""
        destination.mkdir(parents=False, exist_ok=False)
        try:
            for child in staging.iterdir():
                if child.name != "manifest.json":
                    shutil.move(str(child), destination / child.name)
            shutil.move(str(staging / "manifest.json"), destination / "manifest.json")
            staging.rmdir()
        except Exception:
            shutil.rmtree(destination, ignore_errors=True)
            raise

    def _assert_project_file(self, path: Path) -> None:
        root = self.paths.root.resolve()
        if (
            not path.is_relative_to(root)
            or not path.is_file()
            or self.paths.secrets.resolve() in path.parents
        ):
            raise OperationsError(f"Release referencia arquivo inseguro: {path}")

    def _validate_backup_root(self) -> None:
        root = self.paths.root.resolve()
        backup = self.backup_root.resolve()
        if not backup.is_relative_to(root) or backup == root:
            raise OperationsError("Diretório de backup precisa permanecer dentro do projeto")
        if self.platform_name == "nt" and root.drive.upper() != "D:":
            raise OperationsError("Backups operacionais são permitidos somente no disco D:")
        self.backup_root.mkdir(parents=True, exist_ok=True)

    def _powershell_task_action(self, executable: Path, *arguments: str) -> str:
        """Build a Task Scheduler action with every writable runtime path pinned to D:."""

        def quoted(value: str | Path) -> str:
            return "'" + str(value).replace("'", "''") + "'"

        assignments = ";".join(
            f"$env:{name}={quoted(value)}"
            for name, value in self.paths.runtime_environment().items()
        )
        invocation = " ".join(("&", quoted(executable), *(quoted(item) for item in arguments)))
        command = f"{assignments};{invocation}"
        return (
            "powershell.exe -NoProfile -NonInteractive -WindowStyle Hidden "
            f'-ExecutionPolicy Bypass -Command "{command}"'
        )

    def _require_windows(self) -> None:
        if self.platform_name != "nt":
            raise OperationsError("Windows Task Scheduler está disponível somente no Windows")


def _print_json(value: object) -> None:
    if hasattr(value, "__dataclass_fields__"):
        payload: object = asdict(cast(Any, value))
    elif isinstance(value, tuple):
        payload = [
            asdict(cast(Any, item)) if hasattr(item, "__dataclass_fields__") else item
            for item in value
        ]
    else:
        payload = value
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m orizzonte_desk.ops")
    parser.add_argument("--root", type=Path, default=None)
    subparsers = parser.add_subparsers(dest="command", required=True)
    install = subparsers.add_parser("tasks-install")
    install.add_argument("--confirm", required=True)
    subparsers.add_parser("tasks-status")
    uninstall = subparsers.add_parser("tasks-uninstall")
    uninstall.add_argument("--confirm", required=True)
    subparsers.add_parser("backup")
    dry_run = subparsers.add_parser("restore-dry-run")
    dry_run.add_argument("backup_id")
    subparsers.add_parser("watchdog")
    arguments = parser.parse_args(argv)
    paths = AppPaths(arguments.root.resolve()) if arguments.root else AppPaths.discover()
    settings = Settings.load(paths.config)
    manager = OperationsManager(paths, settings.operations)
    if arguments.command == "tasks-install":
        _print_json(manager.install_tasks(arguments.confirm))
    elif arguments.command == "tasks-status":
        _print_json(manager.task_status())
    elif arguments.command == "tasks-uninstall":
        _print_json(manager.uninstall_tasks(arguments.confirm))
    elif arguments.command == "backup":
        _print_json(manager.backup())
    elif arguments.command == "restore-dry-run":
        _print_json(manager.restore_dry_run(arguments.backup_id))
    else:
        manager.watchdog_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
