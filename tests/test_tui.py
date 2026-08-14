from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest
from textual.widgets import DataTable

from orizzonte_desk.tui import (
    DaemonClient,
    DaemonUnavailable,
    Metric,
    OrizzonteTUI,
    _format_threshold,
)


class StubDaemonClient(DaemonClient):
    def __init__(self, state: dict[str, Any], events: list[dict[str, Any]] | None = None) -> None:
        self.state = state
        self.events = events or []

    def fetch(self) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        return self.state, self.events


class OfflineDaemonClient(DaemonClient):
    def __init__(self) -> None:
        pass

    def fetch(self) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        raise DaemonUnavailable("daemon indisponível em http://127.0.0.1:8790")


def prepare_config(paths: Any) -> None:
    source = Path(__file__).parents[1] / "config" / "settings.toml"
    paths.config.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, paths.config)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (144, 40)])
async def test_tui_is_usable_at_supported_terminal_sizes(
    app_paths: Any, size: tuple[int, int]
) -> None:
    prepare_config(app_paths)
    state = {
        "status": "paused",
        "environment": "testnet",
        "budget_usdc": 100.0,
        "last_heartbeat": "2026-08-14T00:00:00Z",
        "metadata": {
            "last_equity": 101.0,
            "day_start_equity": 100.0,
            "high_water_mark": 102.0,
            "stream_status": "healthy",
            "reconciliation_status": "ok",
            "protection_status": "protected",
            "release_id": "release-abc",
            "market": [
                {
                    "timestamp": "2026-08-14T00:00:00Z",
                    "symbol": "BTC",
                    "close": 60_000,
                    "funding_rate": 0.0001,
                    "regime_1d": "bull",
                    "regime_1w": "bull",
                    "volatility_24h": 0.025,
                }
            ],
            "signals": [
                {
                    "timestamp": "2026-08-14T00:00:00Z",
                    "symbol": "BTC",
                    "side": "long",
                    "regime": "bull",
                    "strategy": "breakout",
                    "score": 0.8,
                    "probability": 0.7,
                }
            ],
            "positions": [],
            "decision_policy": {
                "strategy": "hybrid",
                "regime": "bull",
                "probability_threshold": 0.61,
                "decision_at": "2026-08-10T00:05:00Z",
                "evidence_end": "2026-08-09T23:59:59Z",
            },
            "testnet_certificate": {
                "certificate_id": "cert-abc",
                "available": True,
                "valid": True,
            },
            "mainnet_authorization": {
                "authorization_id": "auth-abc",
                "available": True,
            },
            "protection_management": {"status": "managed"},
        },
    }
    app = OrizzonteTUI(app_paths, StubDaemonClient(state))

    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        assert app.size.width == size[0]
        assert app.size.height == size[1]
        assert app.query_one("#m-state", Metric).value == "PAUSED"
        operations = str(app.query_one("#operations-content").render())
        assert "ONLINE" in operations
        assert "release-abc" in operations
        assert "cert-abc" in operations
        assert "DISPONÍVEL" in operations
        assert "MANAGED" in operations
        assert "HYBRID / BULL / 61.0%" in operations
        assert "2026-08-10T00:05:00Z" in operations
        assert len(app.query_one("#signal-table", DataTable).columns) == 10


@pytest.mark.asyncio
async def test_tui_fails_closed_when_daemon_is_offline(app_paths: Any) -> None:
    prepare_config(app_paths)
    app = OrizzonteTUI(app_paths, OfflineDaemonClient())

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        assert app.query_one("#m-state", Metric).value == "DAEMON OFFLINE"
        operations = str(app.query_one("#operations-content").render())
        assert "Controles bloqueados" in operations


def test_missing_decision_policy_never_falls_back_to_fixed_threshold() -> None:
    assert _format_threshold(None) == "—"
