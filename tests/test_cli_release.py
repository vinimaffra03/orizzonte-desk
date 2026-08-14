from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from click import unstyle
from typer.testing import CliRunner

import orizzonte_desk.cli as cli_module


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


def test_status_reads_state_from_daemon(runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cli_module,
        "daemon_request",
        lambda method, endpoint, **kwargs: {
            "status": "disarmed",
            "environment": "paper",
            "metadata": {},
        },
    )

    result = runner.invoke(cli_module.app, ["status"])

    assert result.exit_code == 0
    assert "disarmed" in result.stdout


def test_release_verify_delegates_to_daemon(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, str, dict[str, Any] | None]] = []

    def fake_request(
        method: str,
        endpoint: str,
        *,
        payload: dict[str, Any] | None = None,
        timeout: float = 10.0,
    ) -> dict[str, Any]:
        calls.append((method, endpoint, payload))
        assert timeout == 30.0
        return {"release_id": "release-abc", "verified": True}

    monkeypatch.setattr(cli_module, "daemon_request", fake_request)

    result = runner.invoke(cli_module.app, ["release", "verify", "release-abc"])

    assert result.exit_code == 0
    assert calls == [("POST", "/release/verify", {"release_id": "release-abc"})]
    assert "release-abc" in result.stdout


def test_release_approve_requires_exact_confirmation(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    called = False

    def fake_request(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(cli_module, "daemon_request", fake_request)

    result = runner.invoke(
        cli_module.app,
        ["release", "approve", "release-abc", "--confirm", "WRONG"],
    )

    assert result.exit_code == 1
    assert "APPROVE RELEASE release-abc" in result.stdout
    assert not called


def test_testnet_smoke_is_explicit_and_never_selects_mainnet(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, str, dict[str, Any] | None]] = []

    def fake_request(
        method: str,
        endpoint: str,
        *,
        payload: dict[str, Any] | None = None,
        timeout: float = 10.0,
    ) -> dict[str, Any]:
        calls.append((method, endpoint, payload))
        assert timeout == 120.0
        return {"environment": "testnet", "passed": True}

    monkeypatch.setattr(cli_module, "daemon_request", fake_request)

    result = runner.invoke(
        cli_module.app,
        [
            "testnet",
            "smoke",
            "--budget-usdc",
            "25",
            "--confirm",
            "TESTNET SMOKE 25.00",
        ],
    )

    assert result.exit_code == 0
    assert calls == [
        (
            "POST",
            "/testnet/smoke",
            {"budget_usdc": 25.0, "confirmation": "TESTNET SMOKE 25.00"},
        )
    ]
    assert "mainnet" not in str(calls).lower()


def test_new_command_groups_are_documented_by_help(runner: CliRunner) -> None:
    result = runner.invoke(cli_module.app, ["--help"])

    assert result.exit_code == 0
    assert "testnet" in result.stdout
    assert "release" in result.stdout


def test_unsupported_daemon_capability_fails_closed(
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    app_paths: Any,
    settings: Any,
) -> None:
    monkeypatch.setattr(cli_module, "daemon_settings", lambda: (app_paths, settings))

    def not_found(*args: Any, **kwargs: Any) -> httpx.Response:
        request = httpx.Request("POST", "http://127.0.0.1:8790/release/build")
        return httpx.Response(404, request=request, json={"detail": "Not Found"})

    monkeypatch.setattr(cli_module.httpx, "request", not_found)

    result = runner.invoke(cli_module.app, ["release", "build"])

    assert result.exit_code == 1
    assert "nenhuma ação foi executada" in result.stdout


def test_backtest_model_id_reaches_event_backtester(
    monkeypatch: pytest.MonkeyPatch,
    app_paths: Any,
    settings: Any,
) -> None:
    manifest = SimpleNamespace(
        dataset_id="dataset-short",
        source="synthetic",
        sha256="a" * 64,
        path="ignored.parquet",
    )
    market = cli_module.pd.DataFrame(
        {"timestamp": cli_module.pd.date_range("2026-01-01", periods=48, freq="h", tz="UTC")}
    )
    captured: dict[str, Any] = {}
    expected = object()

    class FakeDatasetManager:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def list_manifests(self) -> list[Any]:
            return [manifest]

        def load(self, _path: str) -> Any:
            return market

    class FakeBacktester:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def run(self, _market: Any, **kwargs: Any) -> object:
            captured.update(kwargs)
            return expected

    monkeypatch.setattr(cli_module, "context", lambda: (app_paths, settings, None, None))
    monkeypatch.setattr(cli_module, "DatasetManager", FakeDatasetManager)
    monkeypatch.setattr(cli_module, "EventBacktester", FakeBacktester)
    monkeypatch.setattr(cli_module, "generate_report", lambda result: Path("report.html"))

    result, report = cli_module.run_backtest(
        dataset_id="dataset-short",
        model_id="model-candidate-123",
    )

    assert result is expected
    assert report == Path("report.html")
    assert captured["model_id"] == "model-candidate-123"


def test_backtest_run_help_exposes_model_id(runner: CliRunner) -> None:
    result = runner.invoke(cli_module.app, ["backtest", "run", "--help"])

    assert result.exit_code == 0
    assert "--model-id" in unstyle(result.stdout)
