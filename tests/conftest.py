from __future__ import annotations

from pathlib import Path

import pytest

from orizzonte_desk.config import Settings
from orizzonte_desk.paths import AppPaths


def pytest_configure(config: pytest.Config) -> None:
    """Ensure the repository-local pytest parent exists on a clean clone."""
    (config.rootpath / ".tmp").mkdir(parents=True, exist_ok=True)


@pytest.fixture()
def settings() -> Settings:
    root = Path(__file__).parents[1]
    return Settings.load(root / "config" / "settings.toml")


@pytest.fixture()
def app_paths(tmp_path: Path) -> AppPaths:
    paths = AppPaths(tmp_path)
    paths.ensure()
    return paths
