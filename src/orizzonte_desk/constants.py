from __future__ import annotations

from typing import Final

APP_NAME: Final = "Orizzonte Desk"
SYMBOLS: Final[tuple[str, ...]] = ("BTC", "ETH", "SOL", "XRP")
ALTCOINS: Final[frozenset[str]] = frozenset({"ETH", "SOL", "XRP"})
DEFAULT_HOME_WINDOWS: Final = r"D:\orizzonte desk"
DEFAULT_PORT: Final = 8790
MAINNET_API_URL: Final = "https://api.hyperliquid.xyz"
TESTNET_API_URL: Final = "https://api.hyperliquid-testnet.xyz"
MAINNET_WS_URL: Final = "wss://api.hyperliquid.xyz/ws"
TESTNET_WS_URL: Final = "wss://api.hyperliquid-testnet.xyz/ws"
BINANCE_FUTURES_URL: Final = "https://fapi.binance.com"
LIVE_CONFIRMATION_PREFIX: Final = "ORIZZONTE LIVE"
