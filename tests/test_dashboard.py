from __future__ import annotations

import unittest

from autoresearch_trade_bot.dashboard import build_dashboard_snapshot


class DashboardSnapshotTests(unittest.TestCase):
    def test_snapshot_exposes_metrics_and_blockers(self) -> None:
        snapshot = build_dashboard_snapshot()

        self.assertEqual(snapshot.phase, "Render-deployed research kernel")
        self.assertFalse(snapshot.research_rollout_ready)
        self.assertTrue(snapshot.accepted_for_paper)
        self.assertIn("Historical Binance ingestion is not implemented yet.", snapshot.research_blockers)
        self.assertIn("sharpe", snapshot.baseline_metrics)
        self.assertGreater(snapshot.baseline_metrics["total_return"], 0.01)


if __name__ == "__main__":
    unittest.main()
