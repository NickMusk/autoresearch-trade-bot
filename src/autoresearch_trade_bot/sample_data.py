from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, Iterable, Sequence

from .models import Bar


def make_series(symbol: str, closes: Sequence[float]) -> list[Bar]:
    start = datetime(2025, 1, 1, 0, 0, 0)
    bars = []
    for index, close in enumerate(closes):
        bars.append(
            Bar(
                symbol=symbol,
                timestamp=start + timedelta(minutes=index * 5),
                open=close,
                high=close,
                low=close,
                close=close,
                volume=1000.0,
            )
        )
    return bars


def build_demo_dataset(
    symbols: Iterable[str] = ("BTCUSDT", "ETHUSDT", "SOLUSDT"),
) -> Dict[str, list[Bar]]:
    closes_by_symbol = {
        "BTCUSDT": [100, 101, 103, 106, 110, 114],
        "ETHUSDT": [100, 99, 98, 96, 95, 94],
        "SOLUSDT": [100, 100, 101, 100, 101, 100],
    }
    return {
        symbol: make_series(symbol, closes_by_symbol[symbol])
        for symbol in symbols
        if symbol in closes_by_symbol
    }
