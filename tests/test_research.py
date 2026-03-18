from __future__ import annotations

import unittest

from autoresearch_trade_bot.config import ExperimentConfig, PromotionGate, RiskLimits
from autoresearch_trade_bot.research import ResearchEvaluator
from autoresearch_trade_bot.sample_data import build_demo_dataset
from autoresearch_trade_bot.strategy import CrossSectionalMomentumStrategy


class NoTradeStrategy:
    def target_weights(self, _history_by_symbol):
        return {}


class ResearchEvaluatorTests(unittest.TestCase):
    def test_accepts_a_profitable_and_stable_strategy(self) -> None:
        config = ExperimentConfig(
            name="acceptance",
            universe=("BTCUSDT", "ETHUSDT", "SOLUSDT"),
            fee_rate=0.0,
            bars_per_year=12,
            risk_limits=RiskLimits(
                max_gross_leverage=1.0,
                max_symbol_weight=0.5,
                max_turnover_per_step=1.0,
            ),
            promotion_gate=PromotionGate(
                min_total_return=0.01,
                min_sharpe=0.5,
                max_drawdown=0.10,
                max_average_turnover=1.0,
            ),
        )
        evaluator = ResearchEvaluator(config)
        strategy = CrossSectionalMomentumStrategy(lookback_bars=2, top_k=1, gross_target=1.0)
        bars_by_symbol = build_demo_dataset()

        result = evaluator.evaluate(bars_by_symbol, strategy)

        self.assertTrue(result.accepted)
        self.assertGreater(result.metrics.total_return, 0.01)
        self.assertGreater(result.metrics.sharpe, 0.5)

    def test_rejects_when_gate_is_too_strict(self) -> None:
        config = ExperimentConfig(
            name="rejection",
            universe=("BTCUSDT", "ETHUSDT", "SOLUSDT"),
            fee_rate=0.0,
            bars_per_year=12,
            risk_limits=RiskLimits(),
            promotion_gate=PromotionGate(
                min_total_return=0.20,
                min_sharpe=10.0,
                max_drawdown=0.01,
                max_average_turnover=0.01,
            ),
        )
        evaluator = ResearchEvaluator(config)
        strategy = CrossSectionalMomentumStrategy(lookback_bars=2, top_k=1, gross_target=1.0)
        bars_by_symbol = build_demo_dataset()

        result = evaluator.evaluate(bars_by_symbol, strategy)

        self.assertFalse(result.accepted)
        self.assertIn("total_return_below_gate", result.rejection_reasons)
        self.assertIn("sharpe_below_gate", result.rejection_reasons)
        self.assertIn("turnover_above_gate", result.rejection_reasons)

    def test_rejects_zero_trade_run_explicitly(self) -> None:
        config = ExperimentConfig(
            name="no-trade",
            universe=("BTCUSDT", "ETHUSDT", "SOLUSDT"),
            fee_rate=0.0,
            bars_per_year=12,
            risk_limits=RiskLimits(),
            promotion_gate=PromotionGate(
                min_total_return=0.0,
                min_sharpe=1.0,
                max_drawdown=0.20,
                max_average_turnover=1.0,
            ),
        )
        evaluator = ResearchEvaluator(config)
        bars_by_symbol = build_demo_dataset()

        result = evaluator.evaluate(bars_by_symbol, NoTradeStrategy())

        self.assertFalse(result.accepted)
        self.assertEqual(result.metrics.nonzero_turnover_steps, 0)
        self.assertIn("no_trades_executed", result.rejection_reasons)


if __name__ == "__main__":
    unittest.main()
