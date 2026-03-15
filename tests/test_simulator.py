from __future__ import annotations

import unittest
from datetime import datetime

from autoresearch_trade_bot.config import ExperimentConfig, PromotionGate, RiskLimits
from autoresearch_trade_bot.models import Bar
from autoresearch_trade_bot.sample_data import build_demo_dataset, make_series
from autoresearch_trade_bot.simulator import BacktestEngine
from autoresearch_trade_bot.strategy import CrossSectionalMomentumStrategy


class BacktestEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ExperimentConfig(
            name="baseline",
            universe=("BTCUSDT", "ETHUSDT", "SOLUSDT"),
            fee_rate=0.0,
            bars_per_year=12,
            risk_limits=RiskLimits(
                max_gross_leverage=1.0,
                max_symbol_weight=0.5,
                max_turnover_per_step=1.0,
            ),
            promotion_gate=PromotionGate(),
        )
        self.data = build_demo_dataset()

    def test_run_is_deterministic_for_same_inputs(self) -> None:
        engine = BacktestEngine(self.config)
        strategy = CrossSectionalMomentumStrategy(lookback_bars=2, top_k=1, gross_target=1.0)

        first = engine.run(self.data, strategy)
        second = engine.run(self.data, strategy)

        self.assertEqual(first.equity_curve, second.equity_curve)
        self.assertEqual(first.weight_history, second.weight_history)
        self.assertEqual(first.metrics, second.metrics)

    def test_rejects_misaligned_series(self) -> None:
        engine = BacktestEngine(self.config)
        strategy = CrossSectionalMomentumStrategy(lookback_bars=2, top_k=1, gross_target=1.0)
        misaligned = dict(self.data)
        misaligned["SOLUSDT"] = make_series("SOLUSDT", [100, 101, 102, 103, 104, 105])
        misaligned["SOLUSDT"][0] = Bar(
            symbol="SOLUSDT",
            timestamp=datetime(2025, 1, 1, 0, 1, 0),
            open=100,
            high=100,
            low=100,
            close=100,
            volume=1000.0,
        )

        with self.assertRaises(ValueError):
            engine.run(misaligned, strategy)


if __name__ == "__main__":
    unittest.main()
