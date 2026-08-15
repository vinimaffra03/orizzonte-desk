from __future__ import annotations

import json
import os
import platform
import shutil
import sys
from dataclasses import asdict, is_dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, NoReturn, cast

import httpx
import pandas as pd
import typer
import uvicorn
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from orizzonte_desk.backtest import EventBacktester, WalkForwardEvaluator
from orizzonte_desk.config import Settings
from orizzonte_desk.constants import APP_NAME, LIVE_CONFIRMATION_PREFIX
from orizzonte_desk.controller import AgentController
from orizzonte_desk.daemon import create_app
from orizzonte_desk.data import DatasetManager, DatasetManifest, stable_fingerprint
from orizzonte_desk.gates import load_combined_gate
from orizzonte_desk.ml import (
    MetaModelRegistry,
    git_commit_fingerprint,
    research_code_fingerprint,
)
from orizzonte_desk.models import AgentStatus, Environment
from orizzonte_desk.ops import OperationsError, OperationsManager
from orizzonte_desk.paths import AppPaths
from orizzonte_desk.reports import generate_report, open_report
from orizzonte_desk.secrets import EnvironmentSecretManager
from orizzonte_desk.storage import StateStore
from orizzonte_desk.strategy import SignalGenerator
from orizzonte_desk.tui import run_tui

console = Console()
app = typer.Typer(
    name="orizzonte",
    help="Orizzonte Desk — research, risk and execution for BTC/ETH/SOL/XRP perps.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)
data_app = typer.Typer(help="Datasets versionados e validação.")
research_app = typer.Typer(help="Treino, avaliação e promoção manual do meta-modelo.")
backtest_app = typer.Typer(help="Backtests, stress e gate live.")
report_app = typer.Typer(help="Relatórios quantitativos.")
paper_app = typer.Typer(help="Controle do paper trading.")
live_app = typer.Typer(help="Controle explícito de testnet/mainnet.")
secret_app = typer.Typer(help="Cofre DPAPI local para a API wallet.")
testnet_app = typer.Typer(help="Preflight, smoke e reconciliação controlados no testnet.")
release_app = typer.Typer(help="Build, verificação e aprovação de releases imutáveis.")
mainnet_app = typer.Typer(help="Capability local, efêmera e de uso único para mainnet.")
ops_app = typer.Typer(help="Task Scheduler, watchdog e backups locais no disco D:.")
app.add_typer(data_app, name="data")
app.add_typer(research_app, name="research")
app.add_typer(backtest_app, name="backtest")
app.add_typer(report_app, name="report")
app.add_typer(paper_app, name="paper")
app.add_typer(live_app, name="live")
app.add_typer(secret_app, name="secret")
app.add_typer(testnet_app, name="testnet")
app.add_typer(release_app, name="release")
app.add_typer(mainnet_app, name="mainnet")
app.add_typer(ops_app, name="ops")


class DataSource(StrEnum):
    SYNTHETIC = "synthetic"
    BINANCE = "binance"
    HYPERLIQUID = "hyperliquid"


def context() -> tuple[AppPaths, Settings, StateStore, AgentController]:
    paths = AppPaths.discover()
    settings = Settings.load(paths.config)
    paths.ensure()
    store = StateStore(paths.database)
    store.initialize()
    return paths, settings, store, AgentController(paths, settings, store)


def fail(message: str, code: int = 1) -> NoReturn:
    console.print(f"[bold red]ERRO[/] {message}")
    raise typer.Exit(code)


def daemon_settings() -> tuple[AppPaths, Settings]:
    """Load the loopback daemon address without opening the local state database."""
    paths = AppPaths.discover()
    return paths, Settings.load(paths.config)


def daemon_request(
    method: str,
    endpoint: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout: float = 10.0,
) -> Any:
    """Call the local daemon and fail closed on unavailable/unsupported capabilities."""
    _, settings = daemon_settings()
    base_url = f"http://{settings.app.host}:{settings.app.port}"
    try:
        response = httpx.request(
            method,
            f"{base_url}{endpoint}",
            json=payload,
            timeout=timeout,
        )
    except httpx.RequestError as exc:
        fail(
            "Daemon local indisponível em "
            f"{base_url}. Inicie `orizzonte daemon` e tente novamente ({exc.__class__.__name__})."
        )
    if not response.is_success:
        detail: str
        try:
            body = response.json()
            detail = str(body.get("detail", body)) if isinstance(body, dict) else str(body)
        except ValueError:
            detail = response.text.strip() or "sem detalhes"
        if response.status_code == 404:
            fail(
                f"O daemon não oferece a capacidade {endpoint}. "
                "Atualize o runtime antes de prosseguir; nenhuma ação foi executada."
            )
        fail(f"Daemon recusou {endpoint} (HTTP {response.status_code}): {detail}")
    try:
        return response.json()
    except ValueError:
        fail(f"Resposta inválida do daemon para {endpoint}; nenhuma confirmação foi aceita.")


def daemon_state() -> dict[str, Any]:
    state = daemon_request("GET", "/state")
    if not isinstance(state, dict):
        fail("Estado inválido recebido do daemon")
    return state


def print_state(state: Any) -> None:
    payload = state.model_dump(mode="json") if hasattr(state, "model_dump") else state
    table = Table(show_header=False, border_style="bright_black")
    table.add_column("Campo", style="dim")
    table.add_column("Valor", style="cyan")
    for key, value in payload.items():
        table.add_row(
            str(key),
            json.dumps(value, ensure_ascii=False)
            if isinstance(value, (dict, list))
            else str(value),
        )
    console.print(table)


@app.command("init")
def initialize() -> None:
    """Cria diretórios locais e inicializa o banco WAL."""
    paths, _, store, _ = context()
    paths.assert_free_space(20)
    removed = paths.cleanup_temp(max_age_days=7)
    store.initialize()
    for key, value in paths.runtime_environment().items():
        os.environ[key] = value
    console.print(
        Panel.fit(
            f"[bold cyan]{APP_NAME} inicializado[/]\nRaiz: {paths.root}\nLivre: {paths.free_gb():.2f} GB\nTemporários expirados: {removed}",
            border_style="cyan",
        )
    )


@app.command()
def doctor() -> None:
    """Valida armazenamento, configuração, ferramentas e conectividade read-only."""
    paths, settings, _, _ = context()
    checks: list[tuple[str, bool, str]] = []
    checks.append(("Raiz no disco D:", paths.root.drive.upper() == "D:", str(paths.root)))
    checks.append(
        (
            "Espaço mínimo",
            paths.free_gb() >= settings.app.minimum_free_gb,
            f"{paths.free_gb():.2f} GB",
        )
    )
    checks.append(("Python 3.11", sys.version_info[:2] == (3, 11), platform.python_version()))
    checks.append(("Configuração", paths.config.exists(), str(paths.config)))
    checks.append(("SQLite WAL", paths.database.exists(), str(paths.database)))
    wallets = EnvironmentSecretManager(paths.secrets)
    configured_wallets = [
        environment.value
        for environment in (Environment.TESTNET, Environment.MAINNET)
        if wallets.path_for(environment).exists()
    ]
    checks.append(
        (
            "Cofres DPAPI separados",
            bool(configured_wallets),
            ", ".join(configured_wallets) if configured_wallets else "opcionais até testnet/live",
        )
    )
    checks.append(
        (
            "Modelo promovido",
            (paths.models / "promoted.json").exists(),
            "necessário para testnet/live",
        )
    )
    try:
        response = httpx.post(
            "https://api.hyperliquid.xyz/info",
            json={"type": "allMids"},
            timeout=8,
        )
        checks.append(("Hyperliquid API", response.is_success, f"HTTP {response.status_code}"))
    except Exception as exc:
        checks.append(("Hyperliquid API", False, str(exc)))
    table = Table("Check", "Status", "Detalhe", border_style="bright_black")
    for name, passed, detail in checks:
        table.add_row(name, "[green]OK[/]" if passed else "[yellow]ATENÇÃO[/]", detail)
    console.print(table)
    if not all(
        passed
        for name, passed, _ in checks
        if name not in {"Cofres DPAPI separados", "Modelo promovido"}
    ):
        raise typer.Exit(1)


@app.command()
def tui() -> None:
    """Abre o dashboard Bloomberg-style no terminal."""
    run_tui()


@app.command()
def daemon() -> None:
    """Inicia o daemon local em 127.0.0.1."""
    paths, settings, _, _ = context()
    uvicorn.run(
        create_app(paths, settings),
        host=settings.app.host,
        port=settings.app.port,
        log_level="info",
    )


@app.command()
def status() -> None:
    """Mostra o estado operacional fornecido pelo daemon local."""
    print_state(daemon_state())


@app.command()
def positions() -> None:
    state = daemon_state()
    metadata = state.get("metadata", {})
    console.print_json(data=list(metadata.get("positions", [])))


@app.command()
def orders() -> None:
    state = daemon_state()
    metadata = state.get("metadata", {})
    console.print_json(data=list(metadata.get("orders", metadata.get("open_orders", []))))


@app.command()
def risk() -> None:
    _, settings = daemon_settings()
    state = daemon_state()
    table = Table("Limite", "Valor", border_style="bright_black")
    values = {
        "Alavancagem": f"{settings.risk.leverage}× isolada",
        "Risco por trade": f"{settings.risk.risk_per_trade:.2%}",
        "Risco agregado": f"{settings.risk.aggregate_risk:.2%}",
        "Meta diária": f"{settings.risk.daily_profit_lock:.2%}",
        "Stop diário": f"-{settings.risk.daily_loss_limit:.2%}",
        "Kill switch": f"-{settings.risk.max_drawdown_limit:.2%}",
        "Orçamento armado": str(state.get("budget_usdc") or "—"),
    }
    for key, value in values.items():
        table.add_row(key, value)
    console.print(table)


@app.command()
def logs(limit: int = typer.Option(100, min=1, max=500)) -> None:
    events = daemon_request("GET", f"/events?limit={limit}")
    if not isinstance(events, list):
        fail("Lista de eventos inválida recebida do daemon")
    for event in reversed(events):
        console.print(
            f"[dim]{event['timestamp']}[/] [cyan]{event['level']:<8}[/] "
            f"[bold]{event['category']}[/] {event['message']}"
        )


def require_wallet_environment(environment: Environment) -> None:
    if environment is Environment.PAPER:
        fail("Paper não possui API wallet; escolha testnet ou mainnet")


@secret_app.command("generate")
def secret_generate(
    environment: Environment = typer.Option(..., "--environment"),
    account_address: str = typer.Option(..., prompt=True),
) -> None:
    """Gera localmente uma API wallet nova e salva somente o segredo DPAPI."""
    require_wallet_environment(environment)
    print_state(
        daemon_request(
            "POST",
            f"/internal/secrets/{environment.value}/generate",
            payload={"account_address": account_address},
        )
    )


@secret_app.command("verify")
def secret_verify(environment: Environment = typer.Option(..., "--environment")) -> None:
    require_wallet_environment(environment)
    print_state(daemon_request("POST", f"/internal/secrets/{environment.value}/verify"))


@secret_app.command("rotate")
def secret_rotate(
    environment: Environment = typer.Option(..., "--environment"),
    account_address: str = typer.Option(..., prompt=True),
    confirm: str = typer.Option(..., prompt=True),
) -> None:
    """Rotaciona para uma chave gerada localmente e proíbe reutilizar fingerprints antigas."""
    require_wallet_environment(environment)
    expected = f"ROTATE {environment.value.upper()} API WALLET"
    if confirm != expected:
        fail(f"Confirmação inválida. Digite exatamente: {expected}")
    print_state(
        daemon_request(
            "POST",
            f"/internal/secrets/{environment.value}/rotate",
            payload={"account_address": account_address},
        )
    )


@secret_app.command("status")
def secret_status(environment: Environment = typer.Option(..., "--environment")) -> None:
    require_wallet_environment(environment)
    print_state(daemon_request("GET", f"/internal/secrets/{environment.value}/status"))


@data_app.command("sync")
def data_sync(
    source: DataSource = typer.Option(DataSource.SYNTHETIC),
    start: str = typer.Option("2021-01-01"),
    hours: int = typer.Option(16_000, min=500),
    environment: Environment = typer.Option(Environment.MAINNET),
) -> None:
    paths, settings, _, _ = context()
    manager = DatasetManager(paths, settings)
    with console.status(f"Sincronizando {source.value}..."):
        if source is DataSource.SYNTHETIC:
            manifest = manager.generate_synthetic(hours=hours)
        elif source is DataSource.BINANCE:
            manifest = manager.sync_binance(start=start)
        else:
            if environment is Environment.PAPER:
                environment = Environment.MAINNET
            manifest = manager.sync_hyperliquid(environment=environment.value)  # type: ignore[arg-type]
    console.print_json(data=manifest.model_dump(mode="json"))


def latest_manifest(paths: AppPaths, source_contains: str | None = None) -> DatasetManifest:
    manifests = DatasetManager(paths, Settings.load(paths.config)).list_manifests()
    if source_contains:
        manifests = [item for item in manifests if source_contains in item.source]
    if not manifests:
        fail("Nenhum dataset encontrado. Execute `orizzonte data sync`.")
    return manifests[0]


@data_app.command("validate")
def data_validate(dataset_id: str | None = None) -> None:
    paths, settings, _, _ = context()
    manager = DatasetManager(paths, settings)
    manifest = (
        latest_manifest(paths)
        if dataset_id is None
        else next(
            (item for item in manager.list_manifests() if item.dataset_id == dataset_id), None
        )
    )
    if manifest is None:
        fail(f"Manifesto não encontrado: {dataset_id}")
    frame = manager.validate_frame(manager.load(manifest.path))
    console.print(
        f"[green]VÁLIDO[/] {manifest.dataset_id}: {len(frame):,} linhas, SHA {manifest.sha256[:12]}"
    )


@data_app.command("status")
def data_status() -> None:
    paths, settings, _, _ = context()
    table = Table("Dataset", "Fonte", "Período", "Linhas", "SHA", border_style="bright_black")
    for item in DatasetManager(paths, settings).list_manifests():
        table.add_row(
            item.dataset_id,
            item.source,
            f"{item.start:%Y-%m-%d} → {item.end:%Y-%m-%d}",
            f"{item.rows:,}",
            item.sha256[:10],
        )
    console.print(table)


@research_app.command("train")
def research_train(dataset_id: str | None = None) -> None:
    paths, settings, _, _ = context()
    manager = DatasetManager(paths, settings)
    manifest = (
        latest_manifest(paths)
        if dataset_id is None
        else next(
            (item for item in manager.list_manifests() if item.dataset_id == dataset_id), None
        )
    )
    if manifest is None:
        fail(f"Dataset não encontrado: {dataset_id}")
    with console.status("Gerando features e treinando LightGBM calibrado..."):
        enriched = SignalGenerator(settings.strategy).enrich(manager.load(manifest.path))
        result = MetaModelRegistry(paths, settings.research, settings.execution).train(
            enriched,
            seed=settings.backtest.random_seed,
            dataset_role=manifest.role,
            dataset_hashes=(manifest.sha256,),
            config_fingerprint=stable_fingerprint(settings.model_dump(mode="json")),
            code_hash=research_code_fingerprint(),
            commit_hash=git_commit_fingerprint(paths.root),
        )
    console.print_json(
        data={
            "model_id": result.model_id,
            "hash": result.model_hash,
            "decision_policy_id": result.decision_policy_id,
            **result.metrics,
        }
    )


@research_app.command("evaluate")
def research_evaluate() -> None:
    paths, _, _, _ = context()
    table = Table("Modelo", "ROC AUC", "Brier", "Amostras", "Status", border_style="bright_black")
    promoted = (
        json.loads((paths.models / "promoted.json").read_text(encoding="utf-8"))["model_id"]
        if (paths.models / "promoted.json").exists()
        else None
    )
    for metadata_path in sorted(paths.models.glob("model-*.json"), reverse=True):
        item = json.loads(metadata_path.read_text(encoding="utf-8"))
        metrics = item["metrics"]
        table.add_row(
            item["model_id"],
            f"{metrics['roc_auc']:.3f}",
            f"{metrics['brier_score']:.3f}",
            f"{metrics['training_samples']:.0f}",
            "[green]PROMOVIDO[/]" if item["model_id"] == promoted else "candidato",
        )
    console.print(table)


@research_app.command("promote")
def research_promote(
    model_id: str, gate: Path = typer.Option(..., exists=True, dir_okay=False)
) -> None:
    paths, settings, _, _ = context()
    pointer = MetaModelRegistry(paths, settings.research, settings.execution).promote(
        model_id, gate
    )
    console.print_json(data=pointer)


def manifest_by_id(
    manager: DatasetManager,
    paths: AppPaths,
    dataset_id: str | None,
) -> DatasetManifest:
    manifest = (
        latest_manifest(paths)
        if dataset_id is None
        else next(
            (item for item in manager.list_manifests() if item.dataset_id == dataset_id),
            None,
        )
    )
    if manifest is None:
        fail(f"Dataset não encontrado: {dataset_id}")
    return manifest


@research_app.command("diagnose")
def research_diagnose(
    dataset_id: str = typer.Option(..., "--dataset-id"),
    model_id: str | None = typer.Option(None, "--model-id"),
) -> None:
    """Exporta telemetria completa do funil; baseline sem ML nunca produz gate."""
    paths, settings, _, _ = context()
    manager = DatasetManager(paths, settings)
    manifest = manifest_by_id(manager, paths, dataset_id)
    if manifest.role == "external_holdout":
        fail(
            "Diagnóstico/challenger é proibido no external_holdout; "
            "use uma única vez `backtest run --model-id`"
        )
    market = manager.load(manifest.path)
    registry = MetaModelRegistry(paths, settings.research, settings.execution)
    if model_id:
        bundle = registry.load_candidate(model_id)
        policy = registry.load_decision_policy(model_id)
        enriched = SignalGenerator(settings.strategy, registry).enrich(
            market,
            model_bundle=bundle,
            model_id=model_id,
            decision_policy=policy,
        )
    else:
        enriched = SignalGenerator(settings.strategy).enrich(market)
    with console.status("Gerando funil quantitativo e calibração..."):
        result = EventBacktester(settings, paths).run(
            market,
            source=f"{manifest.source}-diagnose",
            dataset_hash=manifest.sha256,
            enriched_override=enriched,
            model_id=model_id,
            run_stress_suite=False,
        )
    console.print_json(
        data={
            "run_id": result.run_id,
            "baseline_without_ml": model_id is None,
            "gate_eligible": model_id is not None,
            "artifacts": {
                key: str(value)
                for key, value in result.artifacts.items()
                if key.startswith("funnel_") or key == "probability_calibration"
            },
        }
    )


@research_app.command("regimes")
def research_regimes(dataset_id: str = typer.Option(..., "--dataset-id")) -> None:
    """Executa a ablação formal dev-only; o resultado é challenger sem promoção."""
    paths, settings, _, _ = context()
    manager = DatasetManager(paths, settings)
    manifest = manifest_by_id(manager, paths, dataset_id)
    if manifest.role != "development":
        fail("Estudo de regimes aceita somente dataset development")
    with console.status("Executando walk-forward e ablação event-driven de regimes..."):
        result, report = run_backtest(dataset_id=dataset_id, include_regime_study=True)
    regime_artifacts = {
        key: str(value)
        for key, value in result.artifacts.items()
        if key.startswith("regime_") or key in {"strategy_ablation_csv", "weekly_decisions_csv"}
    }
    if not regime_artifacts:
        fail("O run não produziu artefatos formais de regime")
    console.print_json(
        data={
            "run_id": result.run_id,
            "challenger_only": True,
            "gate_eligible": False,
            "report": str(report),
            "artifacts": regime_artifacts,
        }
    )


def run_backtest(
    *,
    dataset_id: str | None,
    source_contains: str | None = None,
    stress_only: bool = False,
    model_id: str | None = None,
    include_regime_study: bool = False,
) -> tuple[Any, Path]:
    paths, settings, _, _ = context()
    manager = DatasetManager(paths, settings)
    manifest = (
        latest_manifest(paths, source_contains)
        if dataset_id is None
        else next(
            (item for item in manager.list_manifests() if item.dataset_id == dataset_id), None
        )
    )
    if manifest is None:
        fail(f"Dataset não encontrado: {dataset_id}")
    if getattr(manifest, "role", "development") == "external_holdout" and model_id is None:
        fail("External holdout exige --model-id para vincular o gate ao candidato exato")
    with console.status("Executando simulador event-driven e Monte Carlo..."):
        market = manager.load(manifest.path)
        span_days = (
            pd.to_datetime(market["timestamp"], utc=True).max()
            - pd.to_datetime(market["timestamp"], utc=True).min()
        ).days
        if span_days >= 730 and "hyperliquid" not in manifest.source and not stress_only:
            result = WalkForwardEvaluator(settings, paths).run(
                market,
                source=manifest.source,
                dataset_hash=manifest.sha256,
                run_regime_study=include_regime_study,
            )
        else:
            result = EventBacktester(settings, paths).run(
                market,
                source=manifest.source,
                dataset_hash=manifest.sha256,
                model_id=model_id,
                cost_multiplier=2.0 if stress_only else 1.0,
                signal_delay_hours=2 if stress_only else 1,
                missing_fraction=0.005 if stress_only else 0,
                run_regime_study=include_regime_study,
            )
        report = generate_report(result)
    return result, report


@backtest_app.command("run")
def backtest_run(
    dataset_id: str | None = None,
    model_id: str | None = typer.Option(
        None,
        "--model-id",
        help="Avalia um modelo candidato específico e vincula seu hash ao gate.",
    ),
) -> None:
    result, report = run_backtest(dataset_id=dataset_id, model_id=model_id)
    console.print_json(
        data={"run_id": result.run_id, **result.metrics.summary, "report": str(report)}
    )


@backtest_app.command("stress")
def backtest_stress(
    dataset_id: str | None = None,
    model_id: str | None = typer.Option(
        None,
        "--model-id",
        help="Avalia o stress de um modelo candidato específico.",
    ),
) -> None:
    result, report = run_backtest(dataset_id=dataset_id, stress_only=True, model_id=model_id)
    console.print_json(
        data={"run_id": result.run_id, **result.metrics.summary, "report": str(report)}
    )


@backtest_app.command("compare")
def backtest_compare(
    long_gate: Path | None = typer.Option(None, exists=True, dir_okay=False),
    hyperliquid_gate: Path | None = typer.Option(None, exists=True, dir_okay=False),
) -> None:
    paths, _, _, _ = context()

    def gate_order(path: Path) -> tuple[str, str]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return str(payload.get("evaluated_at", "")), path.parent.name

    gates = sorted(paths.reports.glob("*/gate.json"), key=gate_order, reverse=True)
    if long_gate is None:
        long_gate = next((item for item in gates if "binance" in item.parent.name), None)
    if hyperliquid_gate is None:
        hyperliquid_gate = next((item for item in gates if "hyperliquid" in item.parent.name), None)
    if long_gate is None or hyperliquid_gate is None:
        fail("São necessários um gate Binance longo e um gate Hyperliquid recente")
    combined = load_combined_gate([long_gate, hyperliquid_gate])
    output = paths.reports / "live-approval.json"
    output.write_text(
        json.dumps(combined, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )
    console.print_json(data={**combined, "path": str(output)})


def latest_report(paths: AppPaths) -> Path:
    reports = sorted(
        paths.reports.glob("*/report.html"), key=lambda item: item.stat().st_mtime, reverse=True
    )
    if not reports:
        fail("Nenhum relatório encontrado")
    return reports[0]


@report_app.command("latest")
def report_latest() -> None:
    paths, _, _, _ = context()
    console.print(str(latest_report(paths)))


@report_app.command("open")
def report_open() -> None:
    paths, _, _, _ = context()
    path = latest_report(paths)
    open_report(path)
    console.print(f"Aberto: {path}")


@report_app.command("export")
def report_export(destination: Path) -> None:
    paths, _, _, _ = context()
    source = latest_report(paths).parent
    destination = destination.resolve()
    if destination.exists():
        fail("Destino já existe; exportação não sobrescreve arquivos")
    shutil.copytree(source, destination)
    console.print(f"[green]Exportado[/] {destination}")


def control_action(action: str) -> None:
    print_state(daemon_request("POST", f"/control/{action}"))


@paper_app.command("start")
def paper_start(budget_usdc: float = typer.Option(10_000.0, min=1)) -> None:
    confirmation = f"{LIVE_CONFIRMATION_PREFIX} PAPER {budget_usdc:.2f}"
    daemon_request(
        "POST",
        "/control/arm",
        payload={
            "environment": Environment.PAPER.value,
            "budget_usdc": budget_usdc,
            "confirmation": confirmation,
        },
    )
    print_state(daemon_request("POST", "/control/start"))


@paper_app.command("pause")
def paper_pause() -> None:
    control_action("pause")


@paper_app.command("stop")
def paper_stop() -> None:
    state = daemon_state()
    if state.get("status") == AgentStatus.RUNNING.value:
        daemon_request("POST", "/control/pause")
    metadata = state.get("metadata", {})
    if metadata.get("positions"):
        daemon_request("POST", "/control/flatten")
    print_state(daemon_request("POST", "/control/disarm"))


@live_app.command("arm")
def live_arm(
    budget_usdc: float = typer.Option(..., min=1),
    environment: Environment = typer.Option(Environment.TESTNET),
    confirm: str = typer.Option(..., prompt=True),
) -> None:
    if environment is Environment.PAPER:
        fail("Use `orizzonte paper start` para paper trading")
    print_state(
        daemon_request(
            "POST",
            "/control/arm",
            payload={
                "environment": environment.value,
                "budget_usdc": budget_usdc,
                "confirmation": confirm,
            },
        )
    )


@live_app.command("start")
def live_start() -> None:
    control_action("start")


@live_app.command("pause")
def live_pause() -> None:
    control_action("pause")


@live_app.command("resume")
def live_resume() -> None:
    control_action("start")


@live_app.command("flatten")
def live_flatten(confirm: str = typer.Option(..., prompt=True)) -> None:
    if confirm != "FLATTEN":
        fail("Confirmação inválida; digite FLATTEN")
    control_action("flatten")


@live_app.command("disarm")
def live_disarm() -> None:
    control_action("disarm")


@testnet_app.command("preflight")
def testnet_preflight() -> None:
    """Valida release, conta, conectividade e bloqueios sem enviar ordens."""
    print_state(daemon_request("POST", "/testnet/preflight", timeout=30.0))


@testnet_app.command("smoke")
def testnet_smoke(
    budget_usdc: float = typer.Option(25.0, min=1),
    confirm: str = typer.Option(..., prompt=True),
) -> None:
    """Solicita ao daemon o smoke test controlado; nunca usa mainnet."""
    expected = f"TESTNET SMOKE {budget_usdc:.2f}"
    if confirm != expected:
        fail(f"Confirmação inválida. Digite exatamente: {expected}")
    print_state(
        daemon_request(
            "POST",
            "/testnet/smoke",
            payload={"budget_usdc": budget_usdc, "confirmation": confirm},
            timeout=120.0,
        )
    )


@testnet_app.command("reconcile")
def testnet_reconcile() -> None:
    """Reconcilia estado persistido com snapshots do testnet."""
    print_state(daemon_request("POST", "/testnet/reconcile", timeout=30.0))


@testnet_app.command("certificate")
def testnet_certificate() -> None:
    """Mostra o certificado content-addressed; não executa smoke nem envia ordens."""
    print_state(daemon_request("GET", "/internal/testnet/certificate/status"))


@release_app.command("build")
def release_build() -> None:
    """Constrói no daemon um manifesto de release imutável e não aprovado."""
    print_state(daemon_request("POST", "/release/build", timeout=30.0))


@release_app.command("verify")
def release_verify(release_id: str) -> None:
    """Recalcula hashes e vínculos de uma release sem habilitar mainnet."""
    print_state(
        daemon_request(
            "POST",
            "/release/verify",
            payload={"release_id": release_id},
            timeout=30.0,
        )
    )


@release_app.command("approve")
def release_approve(
    release_id: str,
    confirm: str = typer.Option(..., prompt=True),
) -> None:
    """Aprova manualmente uma release verificada; não inicia mainnet."""
    expected = f"APPROVE RELEASE {release_id}"
    if confirm != expected:
        fail(f"Confirmação inválida. Digite exatamente: {expected}")
    print_state(
        daemon_request(
            "POST",
            "/release/approve",
            payload={"release_id": release_id, "confirmation": confirm},
            timeout=30.0,
        )
    )


@mainnet_app.command("authorize")
def mainnet_authorize(
    budget_usdc: float = typer.Option(..., min=0.01, max=500.0),
    confirm: str = typer.Option(..., prompt=True),
) -> None:
    """Emite capability DPAPI de 15 minutos; não arma nem envia ordens."""
    print_state(
        daemon_request(
            "POST",
            "/internal/mainnet/authorize",
            payload={"budget_usdc": budget_usdc, "confirmation": confirm},
            timeout=30.0,
        )
    )


@mainnet_app.command("status")
def mainnet_status() -> None:
    print_state(daemon_request("GET", "/internal/mainnet/authorization/status"))


@mainnet_app.command("revoke")
def mainnet_revoke(
    authorization_id: str,
    confirm: str = typer.Option(..., prompt=True),
) -> None:
    expected = f"REVOKE MAINNET {authorization_id}"
    if confirm != expected:
        fail(f"Confirmação inválida. Digite exatamente: {expected}")
    print_state(
        daemon_request(
            "POST",
            "/internal/mainnet/authorization/revoke",
            payload={"authorization_id": authorization_id},
        )
    )


def print_operation(value: object) -> None:
    if isinstance(value, tuple):
        payload: object = [
            asdict(cast(Any, item)) if is_dataclass(item) else item for item in value
        ]
    elif is_dataclass(value):
        payload = asdict(cast(Any, value))
    else:
        payload = value
    console.print_json(data=payload)


def operations_manager() -> OperationsManager:
    paths = AppPaths.discover()
    settings = Settings.load(paths.config)
    paths.ensure()
    return OperationsManager(paths, settings.operations)


def operation_call(action: str, *args: object) -> None:
    try:
        result = getattr(operations_manager(), action)(*args)
    except OperationsError as exc:
        fail(str(exc))
    print_operation(result)


@ops_app.command("install")
def ops_install(confirm: str = typer.Option(..., prompt=True)) -> None:
    operation_call("install_tasks", confirm)


@ops_app.command("status")
def ops_status() -> None:
    operation_call("task_status")


@ops_app.command("backup")
def ops_backup() -> None:
    operation_call("backup")


@ops_app.command("restore-dry-run")
def ops_restore_dry_run(backup_id: str) -> None:
    operation_call("restore_dry_run", backup_id)


@ops_app.command("uninstall")
def ops_uninstall(confirm: str = typer.Option(..., prompt=True)) -> None:
    operation_call("uninstall_tasks", confirm)
