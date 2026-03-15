from __future__ import annotations

from typing import Mapping, Protocol, Sequence

from .models import Bar


class HistoricalDataSource(Protocol):
    """Interface for validated historical bars."""

    def load_bars(self, symbols: Sequence[str], timeframe: str) -> Mapping[str, Sequence[Bar]]:
        ...


class RealtimeBarSource(Protocol):
    """Interface for a future paper/shadow market data stream."""

    def subscribe_bars(self, symbols: Sequence[str], timeframe: str) -> None:
        ...
