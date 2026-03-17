from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from autoresearch_trade_bot.config import DataConfig, LLMWorkerConfig, ResearchTargetGate
from autoresearch_trade_bot.llm_worker import LLMAutoresearchWorker
from autoresearch_trade_bot.state import FilesystemResearchStateStore


def make_fake_decision(
    *,
    decision: str = "discard_screen",
    score: float = -10.0,
    ready_for_paper: bool = False,
    failure_reason: str = "",
):
    report = SimpleNamespace(
        provider_name="llm",
        model_name="fake-gpt",
        ready_for_paper=ready_for_paper,
        average_metrics={
            "total_return": -0.1,
            "sharpe": -1.0,
            "max_drawdown": 0.2,
            "average_turnover": 0.1,
            "bars_processed": 100,
        },
        research_score=score,
        gate_failures=[] if ready_for_paper else ["acceptance_rate_below_gate"],
        failure_reason=failure_reason,
        strategy_name="configurable-momentum",
    )
    return SimpleNamespace(
        decision=decision,
        stage="screen" if decision == "discard_screen" else "full",
        mutation_label="llm-test",
        baseline_score=-5.0,
        candidate_score=score,
        kept_commit="abc123" if decision == "keep" else None,
        report=report,
    )


class LLMAutoresearchWorkerTests(unittest.TestCase):
    def test_run_cycle_creates_campaign_status_and_history(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            temp_path = Path(tempdir)
            campaign_calls: list[str] = []

            def fake_prepare_campaign(**kwargs):
                campaign_name = kwargs["campaign_name"]
                campaign_calls.append(campaign_name)
                campaign_path = Path(kwargs["campaigns_root"]) / f"{campaign_name}.json"
                campaign_path.parent.mkdir(parents=True, exist_ok=True)
                campaign_path.write_text("{}", encoding="utf-8")
                Path(kwargs["active_pointer_path"]).write_text(
                    str(campaign_path.resolve()),
                    encoding="utf-8",
                )
                return campaign_path

            config = LLMWorkerConfig(
                repo_url="https://github.com/example/repo.git",
                symbols=("BTCUSDT", "ETHUSDT"),
                repo_root=str(temp_path / "repo"),
                campaigns_root=str(temp_path / "campaigns"),
                active_campaign_path=str(temp_path / "active_campaign.txt"),
                worktrees_root=str(temp_path / "worktrees"),
                state_root=str(temp_path / "state"),
                artifact_root=str(temp_path / "artifacts"),
                results_path=str(temp_path / "results.tsv"),
                cycle_interval_seconds=3600,
                campaign_refresh_interval_seconds=86400,
                max_mutations_per_cycle=1,
                data_config=DataConfig(
                    exchange="bybit",
                    market="linear",
                    timeframe="5m",
                    storage_root=str(temp_path / "data"),
                ),
                target_gate=ResearchTargetGate(),
            )
            worker = LLMAutoresearchWorker(
                config=config,
                state_store=FilesystemResearchStateStore(config.state_root),
                llm_runner=lambda **_kwargs: [make_fake_decision()],
                campaign_preparer=fake_prepare_campaign,
                now_fn=lambda: datetime(2026, 3, 17, 12, 0, tzinfo=timezone.utc),
                sleep_fn=lambda _seconds: None,
            )
            worker._ensure_repo_ready = lambda: None  # type: ignore[method-assign]

            result = worker.run_cycle()

            self.assertEqual(len(campaign_calls), 1)
            self.assertEqual(result.campaign_id, campaign_calls[0])
            snapshot = FilesystemResearchStateStore(config.state_root).load_snapshot()
            self.assertIsNotNone(snapshot)
            assert snapshot is not None
            self.assertEqual(snapshot.phase, "LLM autoresearch worker active")
            self.assertEqual(snapshot.loop_state, "searching")
            self.assertEqual(snapshot.latest_dataset_id, campaign_calls[0])
            self.assertEqual(snapshot.multi_window_summary["latest_decision"]["decision"], "discard_screen")
            history = FilesystemResearchStateStore(config.state_root).load_history()
            self.assertEqual(len(history), 1)

    def test_run_cycle_reuses_active_campaign_until_refresh_due(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            temp_path = Path(tempdir)
            campaign_calls: list[str] = []

            def fake_prepare_campaign(**kwargs):
                campaign_calls.append(kwargs["campaign_name"])
                campaign_path = Path(kwargs["campaigns_root"]) / f"{kwargs['campaign_name']}.json"
                campaign_path.parent.mkdir(parents=True, exist_ok=True)
                campaign_path.write_text("{}", encoding="utf-8")
                Path(kwargs["active_pointer_path"]).write_text(
                    str(campaign_path.resolve()),
                    encoding="utf-8",
                )
                return campaign_path

            config = LLMWorkerConfig(
                repo_url="https://github.com/example/repo.git",
                repo_root=str(temp_path / "repo"),
                campaigns_root=str(temp_path / "campaigns"),
                active_campaign_path=str(temp_path / "active_campaign.txt"),
                worktrees_root=str(temp_path / "worktrees"),
                state_root=str(temp_path / "state"),
                artifact_root=str(temp_path / "artifacts"),
                results_path=str(temp_path / "results.tsv"),
                cycle_interval_seconds=3600,
                campaign_refresh_interval_seconds=86400,
                data_config=DataConfig(
                    exchange="bybit",
                    market="linear",
                    timeframe="5m",
                    storage_root=str(temp_path / "data"),
                ),
            )
            store = FilesystemResearchStateStore(config.state_root)
            worker = LLMAutoresearchWorker(
                config=config,
                state_store=store,
                llm_runner=lambda **_kwargs: [make_fake_decision()],
                campaign_preparer=fake_prepare_campaign,
                now_fn=lambda: datetime(2026, 3, 17, 12, 0, tzinfo=timezone.utc),
                sleep_fn=lambda _seconds: None,
            )
            worker._ensure_repo_ready = lambda: None  # type: ignore[method-assign]
            worker.run_cycle()
            self.assertEqual(len(campaign_calls), 1)

            second_worker = LLMAutoresearchWorker(
                config=config,
                state_store=store,
                llm_runner=lambda **_kwargs: [make_fake_decision(score=-9.0)],
                campaign_preparer=fake_prepare_campaign,
                now_fn=lambda: datetime(2026, 3, 17, 13, 5, tzinfo=timezone.utc),
                sleep_fn=lambda _seconds: None,
            )
            second_worker._ensure_repo_ready = lambda: None  # type: ignore[method-assign]
            second_worker.run_cycle()
            self.assertEqual(len(campaign_calls), 1)

    def test_run_forever_persists_degraded_snapshot_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            temp_path = Path(tempdir)
            sleep_calls: list[float] = []

            def fake_prepare_campaign(**kwargs):
                campaign_path = Path(kwargs["campaigns_root"]) / "campaign.json"
                campaign_path.parent.mkdir(parents=True, exist_ok=True)
                campaign_path.write_text("{}", encoding="utf-8")
                Path(kwargs["active_pointer_path"]).write_text(
                    str(campaign_path.resolve()),
                    encoding="utf-8",
                )
                return campaign_path

            config = LLMWorkerConfig(
                repo_url="https://github.com/example/repo.git",
                repo_root=str(temp_path / "repo"),
                campaigns_root=str(temp_path / "campaigns"),
                active_campaign_path=str(temp_path / "active_campaign.txt"),
                worktrees_root=str(temp_path / "worktrees"),
                state_root=str(temp_path / "state"),
                artifact_root=str(temp_path / "artifacts"),
                results_path=str(temp_path / "results.tsv"),
                cycle_interval_seconds=1,
                failure_cooldown_seconds=123,
                max_cycles=1,
                data_config=DataConfig(
                    exchange="bybit",
                    market="linear",
                    timeframe="5m",
                    storage_root=str(temp_path / "data"),
                ),
            )
            worker = LLMAutoresearchWorker(
                config=config,
                state_store=FilesystemResearchStateStore(config.state_root),
                llm_runner=lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("openai_http_error:429")),
                campaign_preparer=fake_prepare_campaign,
                now_fn=lambda: datetime(2026, 3, 17, 12, 0, tzinfo=timezone.utc),
                sleep_fn=lambda seconds: sleep_calls.append(seconds),
            )
            worker._ensure_repo_ready = lambda: None  # type: ignore[method-assign]
            worker.run_forever()

            snapshot = FilesystemResearchStateStore(config.state_root).load_snapshot()
            self.assertIsNotNone(snapshot)
            assert snapshot is not None
            self.assertEqual(snapshot.loop_state, "degraded")
            self.assertEqual(snapshot.consecutive_failures, 1)
            self.assertIn("openai_http_error:429", snapshot.research_blockers[0])
            self.assertEqual(sleep_calls, [123.0])


if __name__ == "__main__":
    unittest.main()
