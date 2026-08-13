from __future__ import annotations

import json
from typing import ClassVar

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import BindingType
from textual.containers import Grid
from textual.reactive import reactive
from textual.widgets import (
    DataTable,
    Footer,
    Header,
    RichLog,
    Static,
    TabbedContent,
    TabPane,
)

from orizzonte_desk.config import Settings
from orizzonte_desk.models import AgentStatus
from orizzonte_desk.paths import AppPaths
from orizzonte_desk.storage import StateStore


class Metric(Static):
    value = reactive("—")

    def __init__(
        self,
        label: str,
        value: str = "—",
        *,
        classes: str = "",
        id: str | None = None,
    ) -> None:
        super().__init__(classes=f"metric {classes}".strip(), id=id)
        self.label_text = label
        self.value = value

    def render(self) -> Text:
        content = Text()
        content.append(self.label_text.upper() + "\n", style="dim")
        content.append(self.value, style="bold")
        return content


class OrizzonteTUI(App[None]):
    CSS = """
    Screen { background: #050b10; color: #d5e7f0; }
    Header { background: #091722; color: #00d9ff; }
    Footer { background: #091722; color: #7892a5; }
    #brand { height: 3; padding: 1 2; background: #07131c; color: #00d9ff; text-style: bold; letter-spacing: 2; border-bottom: solid #17374b; }
    #ticker { height: 3; padding: 1 2; background: #0a1a25; color: #ffb000; border-bottom: solid #17374b; }
    #metrics { height: 7; grid-size: 6 1; grid-gutter: 1; padding: 1 2; }
    .metric { height: 5; background: #0b1924; border: solid #17374b; padding: 1 2; }
    .positive { color: #34d399; } .negative { color: #fb7185; } .warning { color: #ffb000; }
    TabbedContent { margin: 0 2 1 2; }
    TabPane { background: #071018; padding: 1; }
    DataTable { background: #071018; color: #d5e7f0; }
    RichLog { background: #071018; border: solid #17374b; }
    #statusbar { height: 3; padding: 1 2; background: #0b1924; color: #7892a5; border-top: solid #17374b; }
    @media (max-width: 100) { #metrics { grid-size: 3 2; height: 12; } }
    """
    BINDINGS: ClassVar[list[BindingType]] = [
        ("q", "quit", "Sair"),
        ("r", "refresh", "Atualizar"),
        ("p", "pause_hint", "Pausar"),
        ("f", "flatten_hint", "Flatten"),
        ("l", "focus_logs", "Logs"),
    ]

    def __init__(self, paths: AppPaths | None = None) -> None:
        super().__init__()
        self.paths = paths or AppPaths.discover()
        self.settings = Settings.load(self.paths.config)
        self.store = StateStore(self.paths.database)
        self.store.initialize()
        self.title = "Orizzonte Desk"
        self.sub_title = "BTC · ETH · SOL · XRP | UTC"

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static("ORIZZONTE DESK  //  QUANTITATIVE PERPETUALS SYSTEM", id="brand")
        yield Static("BTC  —    ETH  —    SOL  —    XRP  —", id="ticker")
        with Grid(id="metrics"):
            yield Metric("Estado", id="m-state")
            yield Metric("Ambiente", id="m-env")
            yield Metric("Orçamento", id="m-budget")
            yield Metric("Equity", id="m-equity")
            yield Metric("PnL diário", id="m-daily")
            yield Metric("Drawdown", id="m-dd")
        with TabbedContent(initial="market"):
            with TabPane("MERCADO", id="market"):
                yield DataTable(id="market-table", zebra_stripes=True)
            with TabPane("SINAIS", id="signals"):
                yield DataTable(id="signal-table", zebra_stripes=True)
            with TabPane("POSIÇÕES / ORDENS", id="positions"):
                yield DataTable(id="position-table", zebra_stripes=True)
            with TabPane("RISCO", id="risk"):
                yield Static(id="risk-content")
            with TabPane("PESQUISA", id="research"):
                yield DataTable(id="research-table", zebra_stripes=True)
            with TabPane("LOGS", id="logs"):
                yield RichLog(id="event-log", highlight=True, markup=True)
        yield Static(id="statusbar")
        yield Footer()

    def on_mount(self) -> None:
        self._setup_tables()
        self.refresh_data()
        self.set_interval(2.0, self.refresh_data)

    def _setup_tables(self) -> None:
        self.query_one("#market-table", DataTable).add_columns(
            "ATIVO", "MID", "REGIME 1D", "REGIME 1W", "VOL 24H", "FUNDING"
        )
        self.query_one("#signal-table", DataTable).add_columns(
            "HORA UTC", "ATIVO", "LADO", "SCORE", "PROB.", "STATUS"
        )
        self.query_one("#position-table", DataTable).add_columns(
            "ATIVO", "LADO", "TAMANHO", "ENTRADA", "STOP", "ALVO", "PNL"
        )
        self.query_one("#research-table", DataTable).add_columns(
            "RUN", "SHARPE", "PF", "MAX DD", "STRESS", "GATE"
        )

    def refresh_data(self) -> None:
        state = self.store.agent_state()
        metadata = state.metadata
        self.query_one("#m-state", Metric).value = state.status.value.upper()
        self.query_one("#m-env", Metric).value = state.environment.value.upper()
        self.query_one("#m-budget", Metric).value = (
            f"{state.budget_usdc:,.2f} USDC" if state.budget_usdc else "—"
        )
        equity = float(metadata.get("last_equity", metadata.get("preflight_equity", 0)))
        day_start = float(metadata.get("day_start_equity", equity or 1))
        high_water = float(metadata.get("high_water_mark", equity or 1))
        self.query_one("#m-equity", Metric).value = f"{equity:,.2f}" if equity else "—"
        self.query_one("#m-daily", Metric).value = (
            f"{(equity / day_start - 1) * 100:+.2f}%" if equity and day_start else "—"
        )
        self.query_one("#m-dd", Metric).value = (
            f"{(equity / high_water - 1) * 100:.2f}%" if equity and high_water else "—"
        )
        market_rows = metadata.get("market", [])
        market_table = self.query_one("#market-table", DataTable)
        market_table.clear()
        ticker_values: list[str] = []
        for row in market_rows:
            price = float(row.get("close", 0))
            funding = float(row.get("funding_rate", 0))
            market_table.add_row(
                str(row.get("symbol", "—")),
                f"{price:,.6g}",
                "—",
                "—",
                "—",
                f"{funding:.4%}",
            )
            ticker_values.append(f"{row.get('symbol')}  {price:,.6g}")
        if ticker_values:
            self.query_one("#ticker", Static).update("    |    ".join(ticker_values))

        signal_table = self.query_one("#signal-table", DataTable)
        signal_table.clear()
        for signal in metadata.get("signals", []):
            signal_table.add_row(
                str(signal.get("timestamp", "—")),
                str(signal.get("symbol", "—")),
                str(signal.get("side", "—")).upper(),
                f"{float(signal.get('score', 0)):.2f}",
                f"{float(signal.get('probability', 0)):.1%}",
                "APROVADO",
            )

        position_table = self.query_one("#position-table", DataTable)
        position_table.clear()
        for position in metadata.get("positions", []):
            size = float(position.get("szi", 0))
            entry = float(position.get("entryPx") or 0)
            position_table.add_row(
                str(position.get("coin", "—")),
                "LONG" if size > 0 else "SHORT",
                f"{abs(size):,.6g}",
                f"{entry:,.6g}",
                "nativo",
                "nativo",
                f"{float(position.get('unrealizedPnl', 0)):+.2f}",
            )

        research_table = self.query_one("#research-table", DataTable)
        research_table.clear()
        metric_files = sorted(
            self.paths.reports.glob("*/metrics.json"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )[:20]
        for metric_file in metric_files:
            try:
                payload = json.loads(metric_file.read_text(encoding="utf-8"))
                summary = payload["summary"]
                gate_file = metric_file.parent / "gate.json"
                gate = (
                    json.loads(gate_file.read_text(encoding="utf-8")) if gate_file.exists() else {}
                )
                research_table.add_row(
                    metric_file.parent.name,
                    f"{summary.get('sharpe', 0):.2f}",
                    f"{summary.get('profit_factor', 0):.2f}",
                    f"{summary.get('max_drawdown', 0):.1%}",
                    f"{payload.get('stress', {}).get('net_profit', 0):+.2f}",
                    "OK" if gate.get("passed") else "FALHA",
                )
            except (OSError, ValueError, KeyError):
                continue
        risk_lines = [
            "[b #ffb000]LIMITES ATIVOS[/]",
            "Alavancagem: [b]10× isolada[/]",
            "Risco/trade: [b]1,00%[/]  ·  Agregado: [b]2,00%[/]",
            "Meta diária: [#34d399]+1,00%[/]  ·  Stop diário: [#fb7185]−4,00%[/]",
            "Kill switch: [#fb7185]−25,00% desde high-water mark[/]",
            f"Profit lock: {metadata.get('profit_locked', state.profit_locked)}",
            f"Loss lock: {metadata.get('loss_locked', state.loss_locked)}",
            f"Drawdown lock: {metadata.get('drawdown_locked', state.drawdown_locked)}",
        ]
        self.query_one("#risk-content", Static).update("\n".join(risk_lines))
        status_color = {
            AgentStatus.RUNNING: "#34d399",
            AgentStatus.LOCKED: "#fb7185",
            AgentStatus.PAUSED: "#ffb000",
        }.get(state.status, "#7892a5")
        heartbeat = state.last_heartbeat.isoformat() if state.last_heartbeat else "nunca"
        self.query_one("#statusbar", Static).update(
            f"[{status_color}]● {state.status.value.upper()}[/]  |  heartbeat {heartbeat}  |  "
            f"storage {self.paths.root}  |  {self.paths.free_gb():.1f} GB livres"
        )
        event_log = self.query_one("#event-log", RichLog)
        event_log.clear()
        for event in reversed(self.store.recent_events(100)):
            color = (
                "red"
                if event["level"] in {"ERROR", "CRITICAL"}
                else "yellow"
                if event["level"] == "WARNING"
                else "cyan"
            )
            event_log.write(
                f"[{color}]{event['timestamp']} {event['level']:<8}[/] {event['category']}: {event['message']}"
            )

    def action_refresh(self) -> None:
        self.refresh_data()

    def action_pause_hint(self) -> None:
        self.notify("Use `orizzonte live pause` em outro terminal.", severity="warning")

    def action_flatten_hint(self) -> None:
        self.notify("Use `orizzonte live flatten --confirm FLATTEN`.", severity="error")

    def action_focus_logs(self) -> None:
        self.query_one("#event-log", RichLog).focus()


def run_tui() -> None:
    OrizzonteTUI().run()
