from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Dict, Mapping, Protocol, Sequence

from .models import Bar


class Strategy(Protocol):
    def target_weights(self, history_by_symbol: Mapping[str, Sequence[Bar]]) -> Dict[str, float]:
        ...


@dataclass(frozen=True)
class CrossSectionalMomentumStrategy:
    lookback_bars: int = 12
    top_k: int = 1
    gross_target: float = 1.0

    def target_weights(self, history_by_symbol: Mapping[str, Sequence[Bar]]) -> Dict[str, float]:
        if not history_by_symbol:
            return {}

        ready_scores = []
        for symbol, history in history_by_symbol.items():
            if len(history) <= self.lookback_bars:
                continue
            score = self._normalized_momentum(history)
            ready_scores.append((symbol, score))

        if len(ready_scores) < self.top_k * 2:
            return {}

        ranked = sorted(ready_scores, key=lambda item: item[1], reverse=True)
        longs = ranked[: self.top_k]
        shorts = ranked[-self.top_k :]

        side_gross = self.gross_target / 2.0
        long_weight = side_gross / len(longs)
        short_weight = -side_gross / len(shorts)

        weights: Dict[str, float] = {}
        for symbol, _score in longs:
            weights[symbol] = long_weight
        for symbol, _score in shorts:
            weights[symbol] = short_weight
        return weights

    def _normalized_momentum(self, history: Sequence[Bar]) -> float:
        recent = history[-(self.lookback_bars + 1) :]
        start_close = recent[0].close
        end_close = recent[-1].close
        raw_return = (end_close / start_close) - 1.0

        one_bar_returns = []
        for index in range(1, len(recent)):
            previous = recent[index - 1].close
            current = recent[index].close
            one_bar_returns.append((current / previous) - 1.0)

        volatility = statistics.pstdev(one_bar_returns)
        if math.isclose(volatility, 0.0):
            return raw_return
        return raw_return / volatility
