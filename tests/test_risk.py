from __future__ import annotations

import unittest

from autoresearch_trade_bot.config import RiskLimits
from autoresearch_trade_bot.risk import RiskManager


class RiskManagerTests(unittest.TestCase):
    def test_apply_enforces_symbol_gross_and_turnover_limits(self) -> None:
        manager = RiskManager(
            RiskLimits(
                max_gross_leverage=0.8,
                max_symbol_weight=0.4,
                max_turnover_per_step=0.5,
            )
        )

        constrained = manager.apply(
            target_weights={"BTCUSDT": 0.7, "ETHUSDT": -0.6},
            current_weights={"BTCUSDT": 0.0, "ETHUSDT": 0.0},
        )

        self.assertAlmostEqual(constrained["BTCUSDT"], 0.25)
        self.assertAlmostEqual(constrained["ETHUSDT"], -0.25)
        self.assertAlmostEqual(sum(abs(weight) for weight in constrained.values()), 0.5)


if __name__ == "__main__":
    unittest.main()
