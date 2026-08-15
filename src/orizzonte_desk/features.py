from __future__ import annotations

import numpy as np
import pandas as pd

from orizzonte_desk.config import StrategyConfig

FEATURE_COLUMNS = (
    "return_1h",
    "return_24h",
    "volatility_24h",
    "atr_pct_1h",
    "momentum_24h",
    "volume_zscore",
    "funding_zscore",
    "relative_strength_btc",
    "correlation_btc",
    "trend_strength_4h",
    "distance_fast_ema_4h",
    "daily_trend",
    "weekly_trend",
    "setup_score",
)


def _atr(frame: pd.DataFrame, window: int) -> pd.Series:
    previous_close = frame["close"].shift(1)
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous_close).abs(),
            (frame["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()


def _resample_ohlcv(frame: pd.DataFrame, rule: str) -> pd.DataFrame:
    return (
        frame.resample(rule, label="right", closed="left")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
                "funding_rate": "sum",
            }
        )
        .dropna(subset=["open", "high", "low", "close"])
    )


def _zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=max(5, window // 3)).mean()
    std = series.rolling(window, min_periods=max(5, window // 3)).std().replace(0, np.nan)
    return ((series - mean) / std).clip(-8, 8)


def _symbol_features(frame: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    symbol = str(frame["symbol"].iloc[0])
    base = frame.sort_values("timestamp").set_index("timestamp").copy()
    base.index = pd.to_datetime(base.index, utc=True)
    base["return_1h"] = base["close"].pct_change()
    base["return_24h"] = base["close"].pct_change(24)
    base["volatility_24h"] = base["return_1h"].rolling(24, min_periods=12).std()
    base["atr_1h"] = _atr(base, config.atr_window)
    base["atr_pct_1h"] = base["atr_1h"] / base["close"]
    base["momentum_24h"] = base["close"] / base["close"].shift(24) - 1
    base["volume_zscore"] = _zscore(np.log1p(base["volume"]), 72)
    base["funding_zscore"] = _zscore(base["funding_rate"].fillna(0), 24 * 30)

    four_hour = _resample_ohlcv(base, "4h")
    four_hour["ema_fast"] = four_hour["close"].ewm(span=config.fast_ema, adjust=False).mean()
    four_hour["ema_slow"] = four_hour["close"].ewm(span=config.slow_ema, adjust=False).mean()
    four_hour["atr_4h"] = _atr(four_hour, config.atr_window)
    four_hour["donchian_high"] = four_hour["high"].rolling(config.donchian_window).max().shift(1)
    four_hour["donchian_low"] = four_hour["low"].rolling(config.donchian_window).min().shift(1)
    four_hour["trend_strength_4h"] = (
        (four_hour["ema_fast"] - four_hour["ema_slow"]) / four_hour["atr_4h"]
    ).clip(-8, 8)
    four_hour["distance_fast_ema_4h"] = (
        (four_hour["close"] - four_hour["ema_fast"]) / four_hour["atr_4h"]
    ).clip(-8, 8)
    four_hour["breakout_long"] = four_hour["close"] > four_hour["donchian_high"]
    four_hour["breakout_short"] = four_hour["close"] < four_hour["donchian_low"]
    four_hour["pullback_long"] = (
        (four_hour["ema_fast"] > four_hour["ema_slow"])
        & (four_hour["low"] <= four_hour["ema_fast"])
        & (four_hour["close"] > four_hour["ema_fast"])
    )
    four_hour["pullback_short"] = (
        (four_hour["ema_fast"] < four_hour["ema_slow"])
        & (four_hour["high"] >= four_hour["ema_fast"])
        & (four_hour["close"] < four_hour["ema_fast"])
    )

    daily = _resample_ohlcv(base, "1D")
    daily_fast = daily["close"].ewm(span=config.fast_ema, adjust=False).mean()
    daily_slow = daily["close"].ewm(span=config.slow_ema, adjust=False).mean()
    daily["daily_trend"] = np.sign(daily_fast - daily_slow)

    weekly = _resample_ohlcv(base, "7D")
    weekly_fast = weekly["close"].ewm(span=max(8, config.fast_ema // 2), adjust=False).mean()
    weekly_slow = weekly["close"].ewm(span=max(20, config.slow_ema // 2), adjust=False).mean()
    weekly["weekly_trend"] = np.sign(weekly_fast - weekly_slow)

    four_columns = [
        "atr_4h",
        "trend_strength_4h",
        "distance_fast_ema_4h",
        "breakout_long",
        "breakout_short",
        "pullback_long",
        "pullback_short",
    ]
    base = base.join(four_hour[four_columns].reindex(base.index, method="ffill"))
    base = base.join(daily[["daily_trend"]].reindex(base.index, method="ffill"))
    base = base.join(weekly[["weekly_trend"]].reindex(base.index, method="ffill"))
    long_regime = (base["daily_trend"] > 0) & (base["weekly_trend"] > 0)
    short_regime = (base["daily_trend"] < 0) & (base["weekly_trend"] < 0)
    long_timing = (base["momentum_24h"] > 0) & (base["volume_zscore"] > -1.0)
    short_timing = (base["momentum_24h"] < 0) & (base["volume_zscore"] > -1.0)
    base["signal_raw"] = np.select(
        [
            long_regime & (base["breakout_long"] | base["pullback_long"]) & long_timing,
            short_regime & (base["breakout_short"] | base["pullback_short"]) & short_timing,
        ],
        [1, -1],
        default=0,
    )
    confluence = (
        (base["daily_trend"] == base["weekly_trend"]).astype(float)
        + np.tanh(base["trend_strength_4h"].abs().fillna(0))
        + np.tanh(base["momentum_24h"].abs().fillna(0) * 10)
        + (base["volume_zscore"].fillna(-2) > -1).astype(float)
    )
    base["setup_score"] = (confluence / 4).clip(0, 1)
    base["stop_distance"] = np.maximum(
        base["atr_4h"] * config.atr_stop_multiple,
        base["close"] * 0.005,
    )
    forward_return = base["close"].shift(-24) / base["close"] - 1
    base["forward_return_24h"] = forward_return
    base["realized_return_24h"] = forward_return * base["signal_raw"]
    risk_fraction = (base["stop_distance"] / base["close"]).replace(0, np.nan)
    base["risk_fraction"] = risk_fraction
    base["forward_r_24h"] = forward_return / risk_fraction
    base["realized_r_24h"] = base["realized_return_24h"] / risk_fraction
    base["label"] = (
        forward_return * base["signal_raw"] > np.maximum(base["atr_pct_1h"] * 1.5, 0.003)
    ).astype(float)
    base.loc[(base["signal_raw"] == 0) | forward_return.isna(), "label"] = np.nan
    base["symbol"] = symbol
    return base.reset_index()


def prepare_features(frame: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    required = {"timestamp", "symbol", "open", "high", "low", "close", "volume", "funding_rate"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Colunas ausentes para features: {sorted(missing)}")
    prepared = frame.copy()
    prepared["timestamp"] = pd.to_datetime(prepared["timestamp"], utc=True)
    pieces = [_symbol_features(group, config) for _, group in prepared.groupby("symbol", sort=True)]
    result = pd.concat(pieces, ignore_index=True).sort_values(["timestamp", "symbol"])

    close_pivot = result.pivot(index="timestamp", columns="symbol", values="close").sort_index()
    btc_return = close_pivot["BTC"].pct_change(24, fill_method=None)
    relation_pieces: list[pd.DataFrame] = []
    for symbol in result["symbol"].unique():
        symbol_return = close_pivot[symbol].pct_change(24, fill_method=None)
        relation_pieces.append(
            pd.DataFrame(
                {
                    "timestamp": close_pivot.index,
                    "symbol": symbol,
                    "relative_strength_btc": symbol_return - btc_return,
                    "correlation_btc": close_pivot[symbol]
                    .pct_change(fill_method=None)
                    .rolling(24 * 30, min_periods=24 * 7)
                    .corr(close_pivot["BTC"].pct_change(fill_method=None)),
                }
            )
        )
    relations = pd.concat(relation_pieces, ignore_index=True)
    result = result.merge(relations, on=["timestamp", "symbol"], how="left")
    result.replace([np.inf, -np.inf], np.nan, inplace=True)
    result = result.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    result.attrs.update(frame.attrs)
    return result
