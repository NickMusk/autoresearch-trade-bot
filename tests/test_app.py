from __future__ import annotations

import unittest

from autoresearch_trade_bot.app import render_dashboard


class AppRenderTests(unittest.TestCase):
    def test_render_dashboard_includes_family_tabs(self) -> None:
        html = render_dashboard(
            {
                "primary_snapshot": {
                    "mission": "mission",
                    "phase": "active",
                    "research_rollout_ready": False,
                    "research_blockers": ["blocker"],
                    "baseline_strategy": "baseline",
                    "promotion_gate": {"min_sharpe": 1.0},
                    "baseline_metrics": {"score": 1.2},
                    "accepted_for_paper": False,
                    "next_milestones": ["milestone"],
                    "loop_state": "searching",
                    "latest_dataset_id": "dataset",
                    "latest_cycle_completed_at": "2026-03-19T09:00:00+00:00",
                    "last_processed_bar": "2026-03-19T08:55:00+00:00",
                    "recent_acceptance_rate": 0.1,
                    "consecutive_failures": 0,
                    "multi_window_summary": {},
                    "leaderboard": [],
                    "latest_decision": {"decision": "discard_screen", "candidate_score": 1.0, "baseline_score": 2.0},
                    "latest_candidate_summary": {},
                    "current_best_strategy_name": "winner",
                    "latest_kept_summary": {},
                },
                "family_tabs": [
                    {
                        "family_id": "mean_reversion",
                        "label": "Mean Reversion",
                        "snapshot": {
                            "phase": "family active",
                            "loop_state": "searching",
                            "current_best_strategy_name": "mr-v1",
                            "baseline_metrics": {"score": 3.2},
                            "latest_decision": {"decision": "keep"},
                            "latest_cycle_completed_at": "2026-03-19T09:00:00+00:00",
                            "leaderboard": [{"strategy_name": "mr-v1", "score": 3.2, "accepted": False}],
                            "research_blockers": ["none yet"],
                        },
                    },
                    {
                        "family_id": "ema_trend",
                        "label": "Ema Trend",
                        "snapshot": None,
                    },
                ],
            }
        )

        self.assertIn("Strategy Families", html)
        self.assertIn("data-family='mean_reversion'", html)
        self.assertIn("Mean Reversion Leaderboard", html)
        self.assertIn("No snapshot published for this family yet.", html)


if __name__ == "__main__":
    unittest.main()
