from __future__ import annotations

import os
import tempfile
import unittest

from autoresearch_trade_bot.dashboard import build_dashboard_snapshot, build_family_dashboard_data
from autoresearch_trade_bot.state import FilesystemResearchStateStore, ResearchStatusSnapshot


class DashboardSnapshotTests(unittest.TestCase):
    def test_snapshot_exposes_metrics_and_blockers(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            snapshot = build_dashboard_snapshot(storage_root=tempdir)

            self.assertEqual(snapshot.phase, "Render-deployed research kernel")
            self.assertFalse(snapshot.research_rollout_ready)
            self.assertTrue(snapshot.accepted_for_paper)
            self.assertIn("No validated Binance dataset has been materialized into local storage yet.", snapshot.research_blockers)
            self.assertIn("sharpe", snapshot.baseline_metrics)
            self.assertGreater(snapshot.baseline_metrics["total_return"], 0.01)

    def test_snapshot_prefers_persisted_status_path(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            store = FilesystemResearchStateStore(tempdir)
            store.save_snapshot(
                ResearchStatusSnapshot(
                    mission="mission",
                    phase="Continuous research worker active",
                    research_rollout_ready=True,
                    research_blockers=[],
                    baseline_strategy="variant search",
                    promotion_gate={"min_sharpe": 1.0},
                    baseline_metrics={"score": 1.23, "sharpe": 2.0},
                    accepted_for_paper=True,
                    next_milestones=["none"],
                    loop_state="holding",
                    latest_dataset_id="dataset-1",
                    latest_cycle_completed_at="2026-03-15T00:05:00+00:00",
                    last_processed_bar="2026-03-15T00:00:00+00:00",
                    recent_acceptance_rate=1.0,
                    consecutive_failures=0,
                    multi_window_summary={"aggregate_score": 1.2, "acceptance_rate": 0.75},
                    leaderboard=[],
                )
            )
            previous_value = os.environ.get("AUTORESEARCH_STATUS_PATH")
            os.environ["AUTORESEARCH_STATUS_PATH"] = str(store.latest_status_path)
            try:
                snapshot = build_dashboard_snapshot(storage_root=tempdir)
            finally:
                if previous_value is None:
                    os.environ.pop("AUTORESEARCH_STATUS_PATH", None)
                else:
                    os.environ["AUTORESEARCH_STATUS_PATH"] = previous_value

            self.assertEqual(snapshot.phase, "Continuous research worker active")
            self.assertTrue(snapshot.research_rollout_ready)
            self.assertEqual(snapshot.loop_state, "holding")
            self.assertEqual(snapshot.multi_window_summary["aggregate_score"], 1.2)

    def test_family_dashboard_data_includes_wave_tabs_from_local_family_repos(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = tempdir
            snapshot = build_dashboard_snapshot(storage_root=tempdir)

            family_root = os.path.join(root, ".autoresearch", "family_repos", "mean_reversion", ".autoresearch", "runs")
            os.makedirs(family_root, exist_ok=True)
            FilesystemResearchStateStore(family_root).save_snapshot(
                ResearchStatusSnapshot(
                    mission="family mission",
                    phase="Family worker active",
                    research_rollout_ready=False,
                    research_blockers=["waiting for keep"],
                    baseline_strategy="mean reversion",
                    promotion_gate={"min_sharpe": 1.0},
                    baseline_metrics={"score": 3.21, "sharpe": 2.1},
                    accepted_for_paper=False,
                    next_milestones=["find family winner"],
                    loop_state="searching",
                    latest_dataset_id="dataset-1",
                    latest_cycle_completed_at="2026-03-19T08:00:00+00:00",
                    last_processed_bar="2026-03-19T07:55:00+00:00",
                    recent_acceptance_rate=0.2,
                    consecutive_failures=0,
                    multi_window_summary={},
                    leaderboard=[{"strategy_name": "mr-v1", "score": 3.21, "accepted": False}],
                    current_best_strategy_name="mr-v1",
                )
            )
            previous_value = os.environ.get("AUTORESEARCH_FAMILY_REPOS_ROOT")
            os.environ["AUTORESEARCH_FAMILY_REPOS_ROOT"] = os.path.join(root, ".autoresearch", "family_repos")
            try:
                payload = build_family_dashboard_data(snapshot)
            finally:
                if previous_value is None:
                    os.environ.pop("AUTORESEARCH_FAMILY_REPOS_ROOT", None)
                else:
                    os.environ["AUTORESEARCH_FAMILY_REPOS_ROOT"] = previous_value

            family_tabs = payload["family_tabs"]
            self.assertEqual(len(family_tabs), 3)
            mean_reversion_tab = next(item for item in family_tabs if item["family_id"] == "mean_reversion")
            self.assertIsNotNone(mean_reversion_tab["snapshot"])
            self.assertEqual(mean_reversion_tab["snapshot"]["phase"], "Family worker active")


if __name__ == "__main__":
    unittest.main()
