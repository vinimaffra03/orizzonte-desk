from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from orizzonte_desk.models import Trade


def _safe_ratio(numerator: float, denominator: float, default: float = 0.0) -> float:
    return float(numerator / denominator) if denominator not in {0.0, -0.0} else default


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peaks = equity.cummax()
    drawdowns = equity / peaks - 1
    return float(abs(drawdowns.min()))


def historical_var_cvar(returns: pd.Series, alpha: float = 0.05) -> tuple[float, float]:
    clean = returns.dropna()
    if clean.empty:
        return 0.0, 0.0
    threshold = float(clean.quantile(alpha))
    tail = clean[clean <= threshold]
    return abs(threshold), abs(float(tail.mean())) if not tail.empty else abs(threshold)


def monte_carlo_ruin_probability(
    trade_returns: np.ndarray,
    *,
    samples: int = 2000,
    seed: int = 42017,
    ruin_fraction: float = 0.5,
) -> float:
    values = np.asarray(trade_returns, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 1.0
    rng = np.random.default_rng(seed)
    block = max(1, min(10, values.size // 10))
    ruined = 0
    for _ in range(samples):
        sampled: list[float] = []
        while len(sampled) < values.size:
            start = int(rng.integers(0, max(1, values.size - block + 1)))
            sampled.extend(values[start : start + block])
        curve = np.cumprod(1 + np.clip(sampled[: values.size], -0.99, 10))
        if curve.min(initial=1.0) <= 1 - ruin_fraction:
            ruined += 1
    return ruined / samples


@dataclass(slots=True)
class MetricsBundle:
    summary: dict[str, float]
    by_symbol: dict[str, dict[str, float]]
    by_direction: dict[str, dict[str, float]]
    daily_returns: pd.Series
    drawdown: pd.Series


def calculate_metrics(
    equity_frame: pd.DataFrame,
    trades: list[Trade],
    *,
    initial_capital: float,
    monte_carlo_samples: int = 2000,
    seed: int = 42017,
) -> MetricsBundle:
    if equity_frame.empty:
        raise ValueError("Curva de equity vazia")
    curve = equity_frame.copy()
    curve["timestamp"] = pd.to_datetime(curve["timestamp"], utc=True)
    curve = curve.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    equity = curve.set_index("timestamp")["equity"].astype(float)
    daily_equity = equity.resample("1D").last().dropna()
    daily_returns = daily_equity.pct_change().dropna()
    annualization = np.sqrt(365)
    mean_return = float(daily_returns.mean()) if not daily_returns.empty else 0.0
    volatility = float(daily_returns.std(ddof=1)) if len(daily_returns) > 1 else 0.0
    downside = daily_returns[daily_returns < 0]
    downside_vol = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
    sharpe = _safe_ratio(mean_return, volatility) * annualization
    sortino = _safe_ratio(mean_return, downside_vol) * annualization
    dd = equity / equity.cummax() - 1
    maximum_dd = abs(float(dd.min()))
    days = max(1, (equity.index.max() - equity.index.min()).days)
    total_return = float(equity.iloc[-1] / initial_capital - 1)
    annual_return = float((equity.iloc[-1] / initial_capital) ** (365 / days) - 1)
    calmar = _safe_ratio(annual_return, maximum_dd)
    omega = _safe_ratio(
        float(daily_returns[daily_returns > 0].sum()),
        abs(float(daily_returns[daily_returns < 0].sum())),
    )
    var_95, cvar_95 = historical_var_cvar(daily_returns)
    trade_frame = pd.DataFrame([trade.model_dump(mode="json") for trade in trades])
    if trade_frame.empty:
        trade_frame = pd.DataFrame(
            columns=[
                "symbol",
                "side",
                "net_pnl",
                "gross_pnl",
                "fees",
                "funding",
                "slippage",
                "mae",
                "mfe",
            ]
        )
    wins = trade_frame[trade_frame["net_pnl"] > 0]
    losses = trade_frame[trade_frame["net_pnl"] < 0]
    gross_profit = float(wins["net_pnl"].sum())
    gross_loss = abs(float(losses["net_pnl"].sum()))
    profit_factor = (
        gross_profit / gross_loss if gross_loss > 0 else 999.0 if gross_profit > 0 else 0.0
    )
    trade_returns = trade_frame["net_pnl"].to_numpy(dtype=float) / initial_capital
    ruin_probability = monte_carlo_ruin_probability(
        trade_returns,
        samples=monte_carlo_samples,
        seed=seed,
    )
    summary = {
        "initial_capital": initial_capital,
        "final_equity": float(equity.iloc[-1]),
        "net_profit": float(equity.iloc[-1] - initial_capital),
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_volatility": volatility * annualization,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "omega": omega,
        "max_drawdown": maximum_dd,
        "var_95": var_95,
        "cvar_95": cvar_95,
        "profit_factor": profit_factor,
        "expectancy": float(trade_frame["net_pnl"].mean()) if not trade_frame.empty else 0.0,
        "win_rate": _safe_ratio(float(len(wins)), float(len(trade_frame))),
        "trades": float(len(trade_frame)),
        "fees": float(trade_frame["fees"].sum()),
        "funding": float(trade_frame["funding"].sum()),
        "slippage": float(trade_frame["slippage"].sum()),
        "mae_mean": float(trade_frame["mae"].mean()) if not trade_frame.empty else 0.0,
        "mfe_mean": float(trade_frame["mfe"].mean()) if not trade_frame.empty else 0.0,
        "days_hit_1pct": float((daily_returns >= 0.01).mean()) if not daily_returns.empty else 0.0,
        "ruin_probability_50": ruin_probability,
    }

    def grouped(column: str) -> dict[str, dict[str, float]]:
        output: dict[str, dict[str, float]] = {}
        if trade_frame.empty:
            return output
        for key, group in trade_frame.groupby(column):
            group_wins = group[group["net_pnl"] > 0]
            group_losses = group[group["net_pnl"] < 0]
            output[str(key)] = {
                "net_pnl": float(group["net_pnl"].sum()),
                "trades": float(len(group)),
                "win_rate": _safe_ratio(float(len(group_wins)), float(len(group))),
                "profit_factor": (
                    float(group_wins["net_pnl"].sum()) / abs(float(group_losses["net_pnl"].sum()))
                    if abs(float(group_losses["net_pnl"].sum())) > 0
                    else 999.0
                    if float(group_wins["net_pnl"].sum()) > 0
                    else 0.0
                ),
            }
        return output

    return MetricsBundle(summary, grouped("symbol"), grouped("side"), daily_returns, dd)
