from __future__ import annotations

from datetime import UTC, datetime

from orizzonte_desk.exchange import PaperGateway, client_order_id
from orizzonte_desk.models import Side, Signal


def test_client_order_ids_are_valid_and_unique() -> None:
    first = str(client_order_id())
    second = str(client_order_id())
    assert first.startswith("0x") and len(first) == 34
    assert first != second


def test_paper_entry_has_native_shape_protections() -> None:
    gateway = PaperGateway()
    signal = Signal(
        timestamp=datetime.now(UTC),
        symbol="ETH",
        side=Side.SHORT,
        score=0.8,
        probability=0.7,
        entry_reference=2000,
        stop_distance=50,
        atr=30,
        regime="bear",
    )
    response = gateway.place_entry_with_protection(
        signal,
        size=1,
        stop_price=2050,
        take_profit_price=1950,
        slippage=0.001,
    )
    assert response["status"] == "ok"
    assert all(item["reduceOnly"] for item in response["protections"])
    assert gateway.snapshot().positions[0]["leverage"] == {"type": "isolated", "value": 10}
