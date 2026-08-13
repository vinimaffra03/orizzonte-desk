from __future__ import annotations

import json
import webbrowser
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from jinja2 import Environment, StrictUndefined

from orizzonte_desk.backtest import BacktestResult

REPORT_TEMPLATE = """<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Orizzonte Desk — {{ run_id }}</title>
  <style>
    :root { color-scheme: dark; --bg:#071018; --panel:#0b1924; --line:#183347; --cyan:#00d9ff; --amber:#ffb000; --green:#34d399; --red:#fb7185; --text:#d8e7f0; --muted:#7892a5; }
    * { box-sizing:border-box } body { margin:0; background:var(--bg); color:var(--text); font:14px Inter,Segoe UI,sans-serif; }
    header { display:flex; justify-content:space-between; align-items:end; padding:24px 32px; border-bottom:1px solid var(--line); background:#08131c; }
    h1 { margin:0; letter-spacing:.12em; color:var(--cyan); font-size:22px } h2 { color:var(--amber); font-size:14px; text-transform:uppercase; letter-spacing:.12em }
    .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:12px; padding:20px 32px; }
    .card,.panel { background:var(--panel); border:1px solid var(--line); border-radius:7px; padding:14px; box-shadow:0 8px 24px #0004; }
    .label { color:var(--muted); text-transform:uppercase; font-size:11px; letter-spacing:.08em } .value { font:600 24px ui-monospace,Consolas,monospace; margin-top:8px }
    .panel { margin:0 32px 20px; overflow:auto } table { width:100%; border-collapse:collapse; font-family:ui-monospace,Consolas,monospace } th,td { padding:8px; border-bottom:1px solid var(--line); text-align:right } th:first-child,td:first-child{text-align:left}
    .ok{color:var(--green)} .bad{color:var(--red)} footer{padding:20px 32px;color:var(--muted)}
  </style>
</head>
<body>
<header><div><div class="label">Relatório quantitativo reproduzível</div><h1>ORIZZONTE DESK</h1></div><div>{{ run_id }}</div></header>
<section class="grid">
{% for label, value in cards %}<article class="card"><div class="label">{{ label }}</div><div class="value">{{ value }}</div></article>{% endfor %}
</section>
<section class="panel"><h2>Equity & Drawdown</h2>{{ chart }}</section>
<section class="panel"><h2>Gate live</h2><div class="{{ 'ok' if gate.passed else 'bad' }}">{{ 'APROVADO' if gate.passed else 'REPROVADO' }}</div><table><tbody>{% for name, passed in gate.checks.items() %}<tr><td>{{ name }}</td><td class="{{ 'ok' if passed else 'bad' }}">{{ 'OK' if passed else 'FALHA' }}</td></tr>{% endfor %}</tbody></table></section>
<section class="panel"><h2>Por ativo</h2><table><thead><tr><th>Ativo</th><th>PnL</th><th>Trades</th><th>Win rate</th><th>Profit factor</th></tr></thead><tbody>{% for symbol, row in by_symbol.items() %}<tr><td>{{ symbol }}</td><td>{{ '%.2f'|format(row.net_pnl) }}</td><td>{{ row.trades|int }}</td><td>{{ '%.1f%%'|format(row.win_rate*100) }}</td><td>{{ '%.2f'|format(row.profit_factor) }}</td></tr>{% endfor %}</tbody></table></section>
<section class="panel"><h2>Manifesto</h2><pre>{{ manifest }}</pre></section>
<footer>O alvo de 1% ao dia é uma meta de pesquisa, não garantia de retorno. Resultados históricos não asseguram resultados futuros.</footer>
</body></html>"""


def _percentage(value: float) -> str:
    return f"{value * 100:.2f}%"


def generate_report(result: BacktestResult) -> Path:
    output_dir = result.gate_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = result.metrics.summary
    gate = json.loads(result.gate_path.read_text(encoding="utf-8"))
    equity = result.equity.copy()
    equity["timestamp"] = pd.to_datetime(equity["timestamp"], utc=True)
    drawdown = equity["equity"] / equity["equity"].cummax() - 1
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=equity["timestamp"], y=equity["equity"], name="Equity", line={"color": "#00d9ff"}
        )
    )
    figure.add_trace(
        go.Scatter(
            x=equity["timestamp"],
            y=drawdown,
            name="Drawdown",
            yaxis="y2",
            line={"color": "#fb7185"},
        )
    )
    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b1924",
        plot_bgcolor="#071018",
        height=440,
        margin={"l": 45, "r": 45, "t": 25, "b": 35},
        yaxis2={"overlaying": "y", "side": "right", "tickformat": ".0%"},
        legend={"orientation": "h"},
    )
    cards = [
        ("Retorno líquido", _percentage(metrics["total_return"])),
        ("Sharpe", f"{metrics['sharpe']:.2f}"),
        ("Sortino", f"{metrics['sortino']:.2f}"),
        ("Max drawdown", _percentage(metrics["max_drawdown"])),
        ("Profit factor", f"{metrics['profit_factor']:.2f}"),
        ("Taxa de acerto", _percentage(metrics["win_rate"])),
        ("Dias ≥ 1%", _percentage(metrics["days_hit_1pct"])),
        ("Risco de ruína 50%", _percentage(metrics["ruin_probability_50"])),
    ]
    manifest = {
        "run_id": result.run_id,
        "metrics": metrics,
        "stress": result.stressed_metrics.summary,
        "artifacts": {key: str(value) for key, value in result.artifacts.items()},
    }
    html = (
        Environment(undefined=StrictUndefined, autoescape=True)
        .from_string(REPORT_TEMPLATE)
        .render(
            run_id=result.run_id,
            cards=cards,
            chart=figure.to_html(full_html=False, include_plotlyjs="cdn"),
            gate=gate,
            by_symbol=result.metrics.by_symbol,
            manifest=json.dumps(manifest, indent=2, ensure_ascii=False),
        )
    )
    output = output_dir / "report.html"
    output.write_text(html, encoding="utf-8")
    (output_dir / "summary.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return output


def open_report(path: Path) -> None:
    webbrowser.open(path.resolve().as_uri())
