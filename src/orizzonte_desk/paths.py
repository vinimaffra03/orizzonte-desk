from __future__ import annotations

import os
import shutil
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from orizzonte_desk.constants import DEFAULT_HOME_WINDOWS


def default_home() -> Path:
    configured = os.environ.get("ORIZZONTE_HOME")
    if configured:
        return Path(configured).expanduser().resolve()
    if os.name == "nt":
        return Path(DEFAULT_HOME_WINDOWS)
    return Path("/app") if Path("/app").exists() else Path.cwd().resolve()


@dataclass(frozen=True, slots=True)
class AppPaths:
    root: Path

    @classmethod
    def discover(cls) -> AppPaths:
        return cls(default_home())

    @property
    def config(self) -> Path:
        return self.root / "config" / "settings.toml"

    @property
    def cache(self) -> Path:
        return self.root / ".cache"

    @property
    def temp(self) -> Path:
        return self.root / ".tmp"

    @property
    def secrets(self) -> Path:
        return self.root / ".secrets"

    @property
    def raw_data(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def processed_data(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def manifests(self) -> Path:
        return self.root / "data" / "manifests"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def state(self) -> Path:
        return self.root / "state"

    @property
    def database(self) -> Path:
        return self.state / "orizzonte.db"

    @property
    def secret_file(self) -> Path:
        return self.secrets / "hyperliquid.dpapi"

    def ensure(self) -> None:
        for path in (
            self.cache,
            self.temp,
            self.secrets,
            self.raw_data,
            self.processed_data,
            self.manifests,
            self.models,
            self.reports,
            self.logs,
            self.state,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def free_gb(self) -> float:
        return shutil.disk_usage(self.root).free / (1024**3)

    def assert_free_space(self, minimum_gb: float) -> None:
        free = self.free_gb()
        if free < minimum_gb:
            raise RuntimeError(
                f"Espaço insuficiente em {self.root.drive or self.root}: "
                f"{free:.2f} GB livres; mínimo {minimum_gb:.2f} GB."
            )

    def runtime_environment(self) -> dict[str, str]:
        return {
            "ORIZZONTE_HOME": str(self.root),
            "UV_CACHE_DIR": str(self.cache / "uv"),
            "UV_PYTHON_INSTALL_DIR": str(self.root / ".runtime" / "python"),
            "TEMP": str(self.temp),
            "TMP": str(self.temp),
        }

    def cleanup_temp(self, *, max_age_days: int = 7) -> int:
        """Remove only stale files inside the project-owned temporary directory."""
        self.temp.mkdir(parents=True, exist_ok=True)
        cutoff = time.time() - max_age_days * 86_400
        removed = 0
        for path in self.temp.rglob("*"):
            if path.is_file() and path.stat().st_mtime < cutoff:
                path.unlink()
                removed += 1
        for directory in sorted(
            (item for item in self.temp.rglob("*") if item.is_dir()),
            key=lambda item: len(item.parts),
            reverse=True,
        ):
            with suppress(OSError):
                directory.rmdir()
        return removed
