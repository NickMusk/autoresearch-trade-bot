from __future__ import annotations

from autoresearch_trade_bot.strategy import CrossSectionalMomentumStrategy

STRATEGY_NAME = "mom-lb24-top1-gross0.5"


def build_strategy():
    return CrossSectionalMomentumStrategy(
        lookback_bars=24,
        top_k=1,
        gross_target=0.5,
    )
