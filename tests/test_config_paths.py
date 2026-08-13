from __future__ import annotations

from pathlib import Path

import pytest

from orizzonte_desk.config import Settings
from orizzonte_desk.paths import AppPaths


def test_settings_lock_universe_and_risk(settings: Settings) -> None:
    assert settings.universe.symbols == ("BTC", "ETH", "SOL", "XRP")
    assert settings.risk.leverage == 10
    assert settings.risk.isolated_margin is True
    assert settings.risk.risk_per_trade == 0.01
    assert settings.app.host == "127.0.0.1"


def test_paths_keep_runtime_under_root(tmp_path: Path) -> None:
    paths = AppPaths(tmp_path)
    paths.ensure()
    environment = paths.runtime_environment()
    for value in environment.values():
        assert Path(value).is_relative_to(tmp_path)
    assert paths.database.parent == tmp_path / "state"


def test_free_space_guard(app_paths: AppPaths) -> None:
    app_paths.assert_free_space(0.000001)
    with pytest.raises(RuntimeError, match="Espaço insuficiente"):
        app_paths.assert_free_space(10**9)
