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
                    "current_best_ready_for_paper": True,
                    "current_best_validation_pass_rate": 0.75,
                    "current_best_validated_for_rollout": True,
                    "latest_cycle_rollout_ready": False,
                    "next_milestones": ["milestone"],
                    "loop_state": "searching",
                    "latest_dataset_id": "dataset",
                    "latest_cycle_completed_at": "2026-03-19T09:00:00+00:00",
                    "last_processed_bar": "2026-03-19T08:55:00+00:00",
                    "recent_acceptance_rate": 0.1,
                    "evaluation_acceptance_rate": 0.1,
                    "generation_validity_rate": 0.8,
                    "mutation_win_rate": 0.1,
                    "consecutive_failures": 0,
                    "multi_window_summary": {
                        "current_best_validation_pass_rate": 0.75,
                        "current_best_validated_for_rollout": True,
                    },
                    "leaderboard": [],
                    "latest_decision": {"decision": "discard_screen", "candidate_score": 1.0, "baseline_score": 2.0},
                    "latest_candidate_summary": {},
                    "current_best_validation_summary": {
                        "validation_pass_rate": 0.75,
                        "validated_for_rollout": True,
                    },
                    "research_champion_summary": {
                        "strategy_name": "winner",
                        "research_score": 4.2,
                    },
                    "rollout_champion_summary": {
                        "strategy_name": "stable-winner",
                        "research_score": 3.8,
                        "validation_summary": {
                            "validation_pass_rate": 0.75,
                            "validated_for_rollout": True,
                        },
                    },
                    "rollout_candidate_shortlist": [
                        {
                            "strategy_name": "stable-winner",
                            "validation_pass_rate": 0.75,
                            "validated_for_rollout": True,
                        }
                    ],
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
                            "current_best_ready_for_paper": False,
                            "current_best_validated_for_rollout": False,
                            "research_champion_summary": {"strategy_name": "mr-v1"},
                            "rollout_champion_summary": {},
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
        self.assertIn("generation_validity_rate", html)
        self.assertIn("Research Champion Paper Ready", html)
        self.assertIn("Research Champion Validated", html)
        self.assertIn("Rollout Champion", html)
        self.assertIn("Rollout Shortlist", html)
        self.assertIn("current_best_validation_pass_rate", html)
        self.assertIn("mutation_win_rate", html)
        self.assertIn("latest_cycle_rollout_ready", html)


if __name__ == "__main__":
    unittest.main()
