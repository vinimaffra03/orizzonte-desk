from __future__ import annotations

import json
from typing import Any, ClassVar

import httpx
from rich.text import Text
from textual import events
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
from orizzonte_desk.paths import AppPaths


class DaemonUnavailable(RuntimeError):
    """The loopback daemon did not provide a complete operational snapshot."""


class DaemonClient:
    def __init__(self, settings: Settings, *, timeout: float = 1.5) -> None:
        self.base_url = f"http://{settings.app.host}:{settings.app.port}"
        self.timeout = timeout

    def fetch(self) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        try:
            with httpx.Client(base_url=self.base_url, timeout=self.timeout) as client:
                health_response = client.get("/health")
                state_response = client.get("/state")
                events_response = client.get("/events", params={"limit": 100})
                for response in (health_response, state_response, events_response):
                    response.raise_for_status()
                health = health_response.json()
                state = state_response.json()
                events = events_response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise DaemonUnavailable(
                f"daemon indisponível em {self.base_url}: {exc.__class__.__name__}"
            ) from exc
        if not isinstance(health, dict) or health.get("status") != "ok":
            raise DaemonUnavailable("health check do daemon não está OK")
        if not isinstance(state, dict) or not isinstance(events, list):
            raise DaemonUnavailable("snapshot inválido recebido do daemon")
        return state, [event for event in events if isinstance(event, dict)]


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
    #brand { height: 2; padding: 0 2; background: #07131c; color: #00d9ff; text-style: bold; border-bottom: solid #17374b; }
    #ticker { height: 2; padding: 0 2; background: #0a1a25; color: #ffb000; border-bottom: solid #17374b; }
    #metrics { height: 5; grid-size: 6 1; grid-gutter: 1; padding: 1 2 0 2; }
    .metric { height: 3; background: #0b1924; border: solid #17374b; padding: 0 1; }
    .positive { color: #34d399; } .negative { color: #fb7185; } .warning { color: #ffb000; }
    TabbedContent { margin: 0 2 1 2; }
    TabPane { background: #071018; padding: 1; }
    DataTable { background: #071018; color: #d5e7f0; }
    RichLog { background: #071018; border: solid #17374b; }
    #statusbar { height: 2; padding: 0 2; background: #0b1924; color: #7892a5; border-top: solid #17374b; }
    #operations-content { padding: 1 2; }
    .narrow #brand, .narrow #ticker { padding: 0 1; }
    .narrow #metrics { grid-size: 3 2; grid-gutter: 0 1; height: 7; padding: 1 1 0 1; }
    .narrow TabbedContent { margin: 0 1; }
    .narrow #statusbar { padding: 0 1; }
    """
    BINDINGS: ClassVar[list[BindingType]] = [
        ("q", "quit", "Sair"),
        ("r", "refresh", "Atualizar"),
        ("p", "pause_hint", "Pausar"),
        ("f", "flatten_hint", "Flatten"),
        ("l", "focus_logs", "Logs"),
    ]

    def __init__(
        self,
        paths: AppPaths | None = None,
        daemon_client: DaemonClient | None = None,
    ) -> None:
        super().__init__()
        self.paths = paths or AppPaths.discover()
        self.settings = Settings.load(self.paths.config)
        self.daemon_client = daemon_client or DaemonClient(self.settings)
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
            with TabPane("SISTEMA", id="system"):
                yield Static(id="operations-content")
            with TabPane("PESQUISA", id="research"):
                yield DataTable(id="research-table", zebra_stripes=True)
            with TabPane("LOGS", id="logs"):
                yield RichLog(id="event-log", highlight=True, markup=True)
        yield Static(id="statusbar")
        yield Footer()

    def on_mount(self) -> None:
        self._apply_responsive_layout(self.size.width)
        self._setup_tables()
        self.refresh_data()
        self.set_interval(2.0, self.refresh_data)

    def on_resize(self, event: events.Resize) -> None:
        self._apply_responsive_layout(event.size.width)

    def _apply_responsive_layout(self, width: int) -> None:
        self.screen.set_class(width <= 100, "narrow")

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
        try:
            state, events = self.daemon_client.fetch()
        except DaemonUnavailable as exc:
            self.query_one("#m-state", Metric).value = "DAEMON OFFLINE"
            self.query_one("#operations-content", Static).update(
                "[b #fb7185]DAEMON OFFLINE[/]\n"
                f"{exc}\n\n"
                "Controles bloqueados. Inicie `orizzonte daemon`; nenhum fallback escreve no SQLite."
            )
            self.query_one("#statusbar", Static).update(
                "[#fb7185]● D:OFF[/] | W:? | R:PENDENTE | P:PENDENTE | REL:NÃO APROVADA"
            )
            event_log = self.query_one("#event-log", RichLog)
            event_log.clear()
            event_log.write(f"[red]{exc}[/]")
            return

        metadata_value = state.get("metadata", {})
        metadata = metadata_value if isinstance(metadata_value, dict) else {}
        status = str(state.get("status", "unknown"))
        self.query_one("#m-state", Metric).value = status.upper()
        self.query_one("#m-env", Metric).value = str(state.get("environment", "—")).upper()
        budget = state.get("budget_usdc")
        self.query_one("#m-budget", Metric).value = f"{float(budget):,.2f} USDC" if budget else "—"
        equity = float(metadata.get("last_equity", metadata.get("preflight_equity", 0)) or 0)
        day_start = float(metadata.get("day_start_equity", equity or 1))
        high_water = float(metadata.get("high_water_mark", equity or 1))
        self.query_one("#m-equity", Metric).value = f"{equity:,.2f}" if equity else "—"
        self.query_one("#m-daily", Metric).value = (
            f"{(equity / day_start - 1) * 100:+.2f}%" if equity and day_start else "—"
        )
        self.query_one("#m-dd", Metric).value = (
            f"{(equity / high_water - 1) * 100:.2f}%" if equity and high_water else "—"
        )
        market_value = metadata.get("market", [])
        market_rows = market_value if isinstance(market_value, list) else []
        market_table = self.query_one("#market-table", DataTable)
        market_table.clear()
        ticker_values: list[str] = []
        for row in market_rows:
            if not isinstance(row, dict):
                continue
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
        signal_value = metadata.get("signals", [])
        for signal in signal_value if isinstance(signal_value, list) else []:
            if not isinstance(signal, dict):
                continue
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
        positions_value = metadata.get("positions", [])
        positions = positions_value if isinstance(positions_value, list) else []
        for position in positions:
            if not isinstance(position, dict):
                continue
            size = float(position.get("szi", position.get("size", 0)) or 0)
            entry = float(position.get("entryPx", position.get("entry_price", 0)) or 0)
            position_table.add_row(
                str(position.get("coin", position.get("symbol", "—"))),
                "LONG" if size > 0 else "SHORT",
                f"{abs(size):,.6g}",
                f"{entry:,.6g}",
                "nativo",
                "nativo",
                f"{float(position.get('unrealizedPnl', position.get('unrealized_pnl', 0)) or 0):+.2f}",
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
            f"Profit lock: {metadata.get('profit_locked', state.get('profit_locked', False))}",
            f"Loss lock: {metadata.get('loss_locked', state.get('loss_locked', False))}",
            f"Drawdown lock: {metadata.get('drawdown_locked', state.get('drawdown_locked', False))}",
        ]
        self.query_one("#risk-content", Static).update("\n".join(risk_lines))

        stream_status = str(
            metadata.get("stream_status", metadata.get("websocket_status", "DESCONHECIDO"))
        ).upper()
        reconciliation_value = metadata.get(
            "reconciliation_status", metadata.get("reconciliation", "PENDENTE")
        )
        if isinstance(reconciliation_value, dict):
            reconciliation_value = reconciliation_value.get("status", "PENDENTE")
        reconciliation = str(reconciliation_value).upper()
        protection_value = metadata.get("protection_status")
        if protection_value is None:
            protection = "SEM POSIÇÃO" if not positions else "PENDENTE"
        else:
            protection = str(protection_value).upper()
        release_id = str(
            metadata.get("release_id") or metadata.get("approved_release_id") or "NÃO APROVADA"
        )
        operations = [
            "[b #00d9ff]SAÚDE OPERACIONAL[/]",
            "Daemon: [#34d399]ONLINE[/]",
            f"Stream: [b]{stream_status}[/]",
            f"Reconciliação: [b]{reconciliation}[/]",
            f"Proteções nativas: [b]{protection}[/]",
            f"Release aprovada: [b]{release_id}[/]",
            "Mainnet: [#fb7185]BLOQUEADA NESTA ENTREGA[/]",
        ]
        self.query_one("#operations-content", Static).update("\n".join(operations))
        status_color = {
            "running": "#34d399",
            "locked": "#fb7185",
            "paused": "#ffb000",
        }.get(status, "#7892a5")
        heartbeat = str(state.get("last_heartbeat") or "nunca")
        self.query_one("#statusbar", Static).update(
            f"[{status_color}]● {status.upper()}[/] | D:ON | W:{stream_status} | "
            f"R:{reconciliation} | P:{protection} | REL:{release_id} | HB:{heartbeat}"
        )
        event_log = self.query_one("#event-log", RichLog)
        event_log.clear()
        for event in reversed(events):
            level = str(event.get("level", "INFO"))
            color = (
                "red"
                if level in {"ERROR", "CRITICAL"}
                else "yellow"
                if level == "WARNING"
                else "cyan"
            )
            event_log.write(
                f"[{color}]{event.get('timestamp', '—')} {level:<8}[/] "
                f"{event.get('category', 'event')}: {event.get('message', '')}"
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
