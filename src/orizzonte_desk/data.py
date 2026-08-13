from __future__ import annotations

import hashlib
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import httpx
import numpy as np
import pandas as pd
import polars as pl
from pydantic import BaseModel, ConfigDict

from orizzonte_desk.config import Settings
from orizzonte_desk.constants import (
    BINANCE_FUTURES_URL,
    MAINNET_API_URL,
    SYMBOLS,
    TESTNET_API_URL,
)
from orizzonte_desk.paths import AppPaths

DATA_COLUMNS = (
    "timestamp",
    "symbol",
    "interval",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "funding_rate",
)


class DatasetManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    source: str
    symbols: tuple[str, ...]
    interval: str
    start: datetime
    end: datetime
    rows: int
    sha256: str
    path: str
    created_at: datetime
    parameters: dict[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_timestamp(value: str | datetime) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    return stamp.tz_convert("UTC")


class DatasetManager:
    def __init__(
        self,
        paths: AppPaths,
        settings: Settings,
        client: httpx.Client | None = None,
    ) -> None:
        self.paths = paths
        self.settings = settings
        self.paths.ensure()
        self.client = client or httpx.Client(timeout=30.0)

    def _write_dataset(
        self,
        frame: pd.DataFrame,
        *,
        source: str,
        parameters: dict[str, Any],
    ) -> DatasetManifest:
        self.paths.assert_free_space(self.settings.app.minimum_free_gb)
        validated = self.validate_frame(frame)
        dataset_id = (
            f"{source}-{validated['timestamp'].min():%Y%m%d}-{validated['timestamp'].max():%Y%m%d}"
        )
        output = self.paths.processed_data / f"{dataset_id}.parquet"
        arrow_frame = validated.copy()
        arrow_frame["timestamp"] = arrow_frame["timestamp"].dt.tz_convert("UTC")
        pl.from_pandas(arrow_frame).write_parquet(output, compression="zstd", statistics=True)
        digest = sha256_file(output)
        manifest = DatasetManifest(
            dataset_id=dataset_id,
            source=source,
            symbols=tuple(sorted(validated["symbol"].unique())),
            interval=str(validated["interval"].iloc[0]),
            start=validated["timestamp"].min().to_pydatetime(),
            end=validated["timestamp"].max().to_pydatetime(),
            rows=len(validated),
            sha256=digest,
            path=str(output),
            created_at=datetime.now(UTC),
            parameters=parameters,
        )
        manifest_path = self.paths.manifests / f"{dataset_id}.json"
        manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        return manifest

    def validate_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = set(DATA_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(f"Colunas ausentes no dataset: {sorted(missing)}")
        result = frame.loc[:, DATA_COLUMNS].copy()
        result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True)
        result["symbol"] = result["symbol"].str.upper()
        unknown = set(result["symbol"].unique()) - set(SYMBOLS)
        if unknown:
            raise ValueError(f"Ativos fora do universo: {sorted(unknown)}")
        numeric = ["open", "high", "low", "close", "volume", "funding_rate"]
        result[numeric] = result[numeric].apply(pd.to_numeric, errors="raise")
        if result.duplicated(["timestamp", "symbol"]).any():
            raise ValueError("Dataset contém timestamps duplicados por ativo")
        if (result[["open", "high", "low", "close"]] <= 0).any().any():
            raise ValueError("Dataset contém preços não positivos")
        invalid_ohlc = (result["high"] < result[["open", "close", "low"]].max(axis=1)) | (
            result["low"] > result[["open", "close", "high"]].min(axis=1)
        )
        if invalid_ohlc.any():
            raise ValueError("Dataset contém candles OHLC inválidos")
        return result.sort_values(["timestamp", "symbol"]).reset_index(drop=True)

    def load(self, path_or_id: str | Path) -> pd.DataFrame:
        path = Path(path_or_id)
        if not path.exists():
            path = self.paths.processed_data / f"{path_or_id}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Dataset não encontrado: {path_or_id}")
        return (
            pl.read_parquet(path)
            .to_pandas()
            .assign(timestamp=lambda data: pd.to_datetime(data["timestamp"], utc=True))
        )

    def list_manifests(self) -> list[DatasetManifest]:
        manifests: list[DatasetManifest] = []
        for path in self.paths.manifests.glob("*.json"):
            manifests.append(DatasetManifest.model_validate_json(path.read_text(encoding="utf-8")))
        return sorted(manifests, key=lambda item: item.created_at, reverse=True)

    def sync_binance(
        self,
        start: str | datetime = "2021-01-01",
        end: str | datetime | None = None,
    ) -> DatasetManifest:
        start_ts = _utc_timestamp(start)
        end_ts = _utc_timestamp(end or datetime.now(UTC).replace(minute=0, second=0, microsecond=0))
        rows: list[dict[str, Any]] = []
        for symbol in SYMBOLS:
            cursor = int(start_ts.timestamp() * 1000)
            end_ms = int(end_ts.timestamp() * 1000)
            market_symbol = f"{symbol}USDT"
            while cursor < end_ms:
                self.paths.assert_free_space(self.settings.app.minimum_free_gb)
                response = self.client.get(
                    f"{BINANCE_FUTURES_URL}/fapi/v1/klines",
                    params={
                        "symbol": market_symbol,
                        "interval": "1h",
                        "startTime": cursor,
                        "endTime": end_ms,
                        "limit": 1500,
                    },
                )
                response.raise_for_status()
                batch = response.json()
                if not batch:
                    break
                for item in batch:
                    rows.append(
                        {
                            "timestamp": pd.to_datetime(item[0], unit="ms", utc=True),
                            "symbol": symbol,
                            "interval": "1h",
                            "open": float(item[1]),
                            "high": float(item[2]),
                            "low": float(item[3]),
                            "close": float(item[4]),
                            "volume": float(item[5]),
                            "funding_rate": 0.0,
                        }
                    )
                next_cursor = int(batch[-1][0]) + 3_600_000
                if next_cursor <= cursor:
                    raise RuntimeError("Paginação da Binance não avançou")
                cursor = next_cursor
                time.sleep(0.03)
        frame = pd.DataFrame(rows)
        if frame.empty:
            raise RuntimeError("A Binance não retornou candles")
        funding = self._binance_funding(start_ts, end_ts)
        if not funding.empty:
            frame = frame.merge(
                funding, on=["timestamp", "symbol"], how="left", suffixes=("", "_api")
            )
            frame["funding_rate"] = frame.pop("funding_rate_api").fillna(frame["funding_rate"])
        return self._write_dataset(
            frame,
            source="binance-usdm",
            parameters={"start": start_ts.isoformat(), "end": end_ts.isoformat(), "interval": "1h"},
        )

    def _binance_funding(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for symbol in SYMBOLS:
            cursor = int(start.timestamp() * 1000)
            end_ms = int(end.timestamp() * 1000)
            while cursor < end_ms:
                response = self.client.get(
                    f"{BINANCE_FUTURES_URL}/fapi/v1/fundingRate",
                    params={
                        "symbol": f"{symbol}USDT",
                        "startTime": cursor,
                        "endTime": end_ms,
                        "limit": 1000,
                    },
                )
                response.raise_for_status()
                batch = response.json()
                if not batch:
                    break
                for item in batch:
                    rows.append(
                        {
                            "timestamp": pd.to_datetime(item["fundingTime"], unit="ms", utc=True),
                            "symbol": symbol,
                            "funding_rate": float(item["fundingRate"]),
                        }
                    )
                next_cursor = int(batch[-1]["fundingTime"]) + 1
                if next_cursor <= cursor:
                    break
                cursor = next_cursor
                if len(batch) < 1000:
                    break
        return pd.DataFrame(rows)

    def sync_hyperliquid(
        self,
        *,
        environment: Literal["mainnet", "testnet"] = "mainnet",
        lookback_hours: int = 5000,
    ) -> DatasetManifest:
        base_url = MAINNET_API_URL if environment == "mainnet" else TESTNET_API_URL
        end = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
        start = end - timedelta(hours=min(lookback_hours, 5000))
        rows: list[dict[str, Any]] = []
        for symbol in SYMBOLS:
            response = self.client.post(
                f"{base_url}/info",
                json={
                    "type": "candleSnapshot",
                    "req": {
                        "coin": symbol,
                        "interval": "1h",
                        "startTime": int(start.timestamp() * 1000),
                        "endTime": int(end.timestamp() * 1000),
                    },
                },
            )
            response.raise_for_status()
            for item in response.json():
                rows.append(
                    {
                        "timestamp": pd.to_datetime(item["t"], unit="ms", utc=True),
                        "symbol": symbol,
                        "interval": "1h",
                        "open": float(item["o"]),
                        "high": float(item["h"]),
                        "low": float(item["l"]),
                        "close": float(item["c"]),
                        "volume": float(item["v"]),
                        "funding_rate": 0.0,
                    }
                )
        if not rows:
            raise RuntimeError("A Hyperliquid não retornou candles")
        frame = pd.DataFrame(rows)
        funding = self._hyperliquid_funding(base_url, start, end)
        if not funding.empty:
            frame = frame.merge(
                funding,
                on=["timestamp", "symbol"],
                how="left",
                suffixes=("", "_api"),
            )
            frame["funding_rate"] = frame.pop("funding_rate_api").fillna(frame["funding_rate"])
        return self._write_dataset(
            frame,
            source=f"hyperliquid-{environment}",
            parameters={"lookback_hours": lookback_hours, "interval": "1h"},
        )

    def _hyperliquid_funding(
        self,
        base_url: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        end_ms = int(end.timestamp() * 1000)
        for symbol in SYMBOLS:
            cursor = int(start.timestamp() * 1000)
            while cursor < end_ms:
                response = self.client.post(
                    f"{base_url}/info",
                    json={
                        "type": "fundingHistory",
                        "coin": symbol,
                        "startTime": cursor,
                        "endTime": end_ms,
                    },
                )
                response.raise_for_status()
                batch = response.json()
                if not batch:
                    break
                for item in batch:
                    rows.append(
                        {
                            "timestamp": pd.to_datetime(item["time"], unit="ms", utc=True),
                            "symbol": symbol,
                            "funding_rate": float(item["fundingRate"]),
                        }
                    )
                next_cursor = max(int(item["time"]) for item in batch) + 1
                if next_cursor <= cursor:
                    break
                cursor = next_cursor
                if len(batch) < 500:
                    break
        return pd.DataFrame(rows)

    def generate_synthetic(
        self,
        *,
        hours: int = 16_000,
        seed: int = 42017,
    ) -> DatasetManifest:
        if hours < 500:
            raise ValueError("O dataset sintético deve ter pelo menos 500 horas")
        rng = np.random.default_rng(seed)
        end = pd.Timestamp.now(tz="UTC").floor("h")
        timestamps = pd.date_range(end=end, periods=hours, freq="h", tz="UTC")
        common = rng.normal(0.00002, 0.008, hours)
        starts = {"BTC": 30_000.0, "ETH": 1_800.0, "SOL": 40.0, "XRP": 0.5}
        betas = {"BTC": 0.8, "ETH": 1.0, "SOL": 1.25, "XRP": 1.1}
        all_rows: list[pd.DataFrame] = []
        for symbol in SYMBOLS:
            innovation = rng.normal(0, 0.006 * betas[symbol], hours)
            cycle = 0.0004 * np.sin(np.arange(hours) / (24 * 45))
            returns = np.clip(common * betas[symbol] + innovation + cycle, -0.12, 0.12)
            close = starts[symbol] * np.exp(np.cumsum(returns))
            open_price = np.r_[close[0], close[:-1]]
            intrabar = np.abs(rng.normal(0.004, 0.003, hours))
            high = np.maximum(open_price, close) * (1 + intrabar)
            low = np.minimum(open_price, close) * np.maximum(0.1, 1 - intrabar)
            volume = rng.lognormal(8 if symbol in {"BTC", "ETH"} else 9, 0.7, hours)
            funding = np.where(np.arange(hours) % 8 == 0, np.clip(returns * 0.02, -0.001, 0.001), 0)
            all_rows.append(
                pd.DataFrame(
                    {
                        "timestamp": timestamps,
                        "symbol": symbol,
                        "interval": "1h",
                        "open": open_price,
                        "high": high,
                        "low": low,
                        "close": close,
                        "volume": volume,
                        "funding_rate": funding,
                    }
                )
            )
        return self._write_dataset(
            pd.concat(all_rows, ignore_index=True),
            source="synthetic",
            parameters={"hours": hours, "seed": seed},
        )
