from __future__ import annotations

from dataclasses import dataclass
import math
import statistics
from typing import Dict, Mapping, Sequence

from autoresearch_trade_bot.models import Bar
from autoresearch_trade_bot.strategy import Strategy

TRAIN_CONFIG = {
    "funding_penalty_weight": 0.0,
    "gross_target": 0.5,
    "lookback_bars": 24,
    "min_cross_sectional_spread": 0.0,
    "min_signal_strength": 0.0,
    "ranking_mode": "risk_adjusted",
    "regime_lookback_bars": 36,
    "regime_threshold": 0.015,
    "reversal_bias_weight": 0.0,
    "top_k": 1,
    "use_regime_filter": False,
    "volatility_floor": 0.0,
}
STRATEGY_NAME = "configurable-momentum"


@dataclass(frozen=True)
class ConfigurableMomentumStrategy:
    lookback_bars: int
    top_k: int
    gross_target: float
    ranking_mode: str
    use_regime_filter: bool
    regime_lookback_bars: int
    regime_threshold: float
    min_signal_strength: float
    min_cross_sectional_spread: float
    volatility_floor: float
    reversal_bias_weight: float
    funding_penalty_weight: float

    def target_weights(self, history_by_symbol: Mapping[str, Sequence[Bar]]) -> Dict[str, float]:
        if not history_by_symbol:
            return {}
        if self.use_regime_filter and not self._passes_regime_filter(history_by_symbol):
            return {}
        ready_scores = []
        for symbol, history in history_by_symbol.items():
            if len(history) <= self.lookback_bars:
                continue
            score = self._score(history)
            if abs(score) < self.min_signal_strength:
                continue
            ready_scores.append((symbol, score))
        if len(ready_scores) < self.top_k * 2:
            return {}
        ranked = sorted(ready_scores, key=lambda item: item[1], reverse=True)
        cross_sectional_spread = ranked[0][1] - ranked[-1][1]
        if cross_sectional_spread < self.min_cross_sectional_spread:
            return {}
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

    def _score(self, history: Sequence[Bar]) -> float:
        recent = history[-(self.lookback_bars + 1) :]
        start_close = recent[0].close
        end_close = recent[-1].close
        raw_return = (end_close / start_close) - 1.0
        one_bar_returns = []
        for index in range(1, len(recent)):
            previous = recent[index - 1].close
            current = recent[index].close
            one_bar_returns.append((current / previous) - 1.0)
        volatility = max(statistics.pstdev(one_bar_returns), self.volatility_floor)
        if math.isclose(volatility, 0.0):
            base_signal = raw_return
        else:
            base_signal = raw_return if self.ranking_mode == "raw_return" else raw_return / volatility
        last_bar_return = one_bar_returns[-1] if one_bar_returns else 0.0
        funding_penalty = self.funding_penalty_weight * recent[-1].funding_rate
        reversal_penalty = self.reversal_bias_weight * last_bar_return
        return base_signal - funding_penalty - reversal_penalty

    def _passes_regime_filter(
        self,
        history_by_symbol: Mapping[str, Sequence[Bar]],
    ) -> bool:
        anchor_symbol = "BTCUSDT" if "BTCUSDT" in history_by_symbol else next(iter(history_by_symbol))
        history = history_by_symbol[anchor_symbol]
        if len(history) <= self.regime_lookback_bars:
            return True
        recent = history[-(self.regime_lookback_bars + 1) :]
        regime_return = (recent[-1].close / recent[0].close) - 1.0
        return abs(regime_return) >= self.regime_threshold


def build_strategy(_dataset_spec=None) -> Strategy:
    return ConfigurableMomentumStrategy(
        lookback_bars=int(TRAIN_CONFIG["lookback_bars"]),
        top_k=int(TRAIN_CONFIG["top_k"]),
        gross_target=float(TRAIN_CONFIG["gross_target"]),
        ranking_mode=str(TRAIN_CONFIG["ranking_mode"]),
        use_regime_filter=bool(TRAIN_CONFIG["use_regime_filter"]),
        regime_lookback_bars=int(TRAIN_CONFIG["regime_lookback_bars"]),
        regime_threshold=float(TRAIN_CONFIG["regime_threshold"]),
        min_signal_strength=float(TRAIN_CONFIG["min_signal_strength"]),
        min_cross_sectional_spread=float(TRAIN_CONFIG["min_cross_sectional_spread"]),
        volatility_floor=float(TRAIN_CONFIG["volatility_floor"]),
        reversal_bias_weight=float(TRAIN_CONFIG["reversal_bias_weight"]),
        funding_penalty_weight=float(TRAIN_CONFIG["funding_penalty_weight"]),
    )
