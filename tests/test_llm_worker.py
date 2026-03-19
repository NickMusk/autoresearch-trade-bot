from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from autoresearch_trade_bot.config import DataConfig, LLMWorkerConfig, ResearchTargetGate
from autoresearch_trade_bot.llm_worker import LLMAutoresearchWorker, publisher_from_env
from autoresearch_trade_bot.state import FilesystemResearchStateStore


def make_fake_decision(
    *,
    decision: str = "discard_screen",
    score: float = -10.0,
    ready_for_paper: bool = False,
    failure_reason: str = "",
    candidate_metrics: dict | None = None,
    baseline_metrics: dict | None = None,
    candidate_gate_failures: list[str] | None = None,
    baseline_train_config: dict | None = None,
    candidate_train_config: dict | None = None,
):
    resolved_candidate_metrics = candidate_metrics or {
        "total_return": -0.1,
        "sharpe": -1.0,
        "max_drawdown": 0.2,
        "average_turnover": 0.1,
        "bars_processed": 100,
    }
    resolved_baseline_metrics = baseline_metrics or {
        "total_return": 0.05,
        "sharpe": 1.25,
        "max_drawdown": 0.08,
        "average_turnover": 0.04,
        "bars_processed": 100,
    }
    baseline_report = SimpleNamespace(
        provider_name="baseline",
        model_name="",
        ready_for_paper=False,
        average_metrics=resolved_baseline_metrics,
        research_score=4.5,
        gate_failures=["sharpe_below_gate"],
        failure_reason="",
        strategy_name="baseline-train-head",
        train_config=baseline_train_config or {
            "lookback_bars": 24,
            "top_k": 1,
            "gross_target": 0.5,
            "ranking_mode": "risk_adjusted",
        },
    )
    report = SimpleNamespace(
        provider_name="llm",
        model_name="fake-gpt",
        ready_for_paper=ready_for_paper,
        average_metrics=resolved_candidate_metrics,
        research_score=score,
        gate_failures=(
            []
            if ready_for_paper
            else list(candidate_gate_failures or ["acceptance_rate_below_gate"])
        ),
        failure_reason=failure_reason,
        strategy_name="configurable-momentum",
        train_config=candidate_train_config or {
            "lookback_bars": 12,
            "top_k": 1,
            "gross_target": 0.5,
            "ranking_mode": "risk_adjusted",
        },
    )
    return SimpleNamespace(
        decision=decision,
        stage="screen" if decision == "discard_screen" else "full",
        mutation_label="llm-test",
        baseline_score=-5.0,
        candidate_score=score,
        kept_commit="abc123" if decision == "keep" else None,
        baseline_report=baseline_report,
        report=report,
    )


class LLMAutoresearchWorkerTests(unittest.TestCase):
    def _git(self, cwd: Path, *args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    def test_zero_cycle_interval_is_treated_as_continuous_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            temp_path = Path(tempdir)
            config = LLMWorkerConfig(
                repo_url="https://github.com/example/repo.git",
                repo_root=str(temp_path / "repo"),
                campaigns_root=str(temp_path / "campaigns"),
                active_campaign_path=str(temp_path / "active_campaign.txt"),
                worktrees_root=str(temp_path / "worktrees"),
                state_root=str(temp_path / "state"),
                artifact_root=str(temp_path / "artifacts"),
                results_path=str(temp_path / "results.tsv"),
                cycle_interval_seconds=0,
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
                llm_runner=lambda **_kwargs: [make_fake_decision()],
                campaign_preparer=lambda **kwargs: Path(kwargs["campaigns_root"]) / "campaign.json",
                now_fn=lambda: datetime(2026, 3, 17, 12, 0, tzinfo=timezone.utc),
                sleep_fn=lambda _seconds: None,
            )
            checkpoint = worker.state_store.load_llm_checkpoint()
            due = worker._is_cycle_due(datetime(2026, 3, 17, 12, 0, tzinfo=timezone.utc), checkpoint)
            self.assertTrue(due)
            self.assertEqual(
                worker._seconds_until_next_cycle(
                    datetime(2026, 3, 17, 12, 0, tzinfo=timezone.utc),
                    checkpoint,
                ),
                0.0,
            )

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
            self.assertEqual(snapshot.latest_decision["decision"], "discard_screen")
            self.assertEqual(snapshot.current_best_strategy_name, "baseline-train-head")
            history = FilesystemResearchStateStore(config.state_root).load_history()
            self.assertEqual(len(history), 1)

    def test_run_forever_in_continuous_mode_still_respects_failure_cooldown(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            temp_path = Path(tempdir)
            sleep_calls: list[float] = []
            run_calls = {"count": 0}

            def fake_runner(**_kwargs):
                run_calls["count"] += 1
                if run_calls["count"] == 1:
                    raise RuntimeError("openai_timeout")
                return [make_fake_decision()]

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
                cycle_interval_seconds=0,
                timeout_failure_cooldown_seconds=180,
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
                llm_runner=fake_runner,
                campaign_preparer=fake_prepare_campaign,
                now_fn=lambda: datetime(2026, 3, 17, 12, 0, tzinfo=timezone.utc),
                sleep_fn=lambda seconds: sleep_calls.append(seconds),
            )
            worker._ensure_repo_ready = lambda: None  # type: ignore[method-assign]

            worker.run_forever()

            self.assertEqual(run_calls["count"], 1)
            self.assertEqual(sleep_calls, [180.0])

    def test_success_snapshot_uses_baseline_report_metrics_not_latest_candidate_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            temp_path = Path(tempdir)

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
                llm_runner=lambda **_kwargs: [
                    make_fake_decision(
                        score=-2.0,
                        candidate_metrics={
                            "total_return": 0.0,
                            "sharpe": 0.0,
                            "max_drawdown": 0.0,
                            "average_turnover": 0.0,
                            "bars_processed": 2015,
                        },
                        baseline_metrics={
                            "total_return": 0.07,
                            "sharpe": 1.75,
                            "max_drawdown": 0.09,
                            "average_turnover": 0.03,
                            "bars_processed": 2015,
                        },
                    )
                ],
                campaign_preparer=fake_prepare_campaign,
                now_fn=lambda: datetime(2026, 3, 17, 12, 0, tzinfo=timezone.utc),
                sleep_fn=lambda _seconds: None,
            )
            worker._ensure_repo_ready = lambda: None  # type: ignore[method-assign]

            worker.run_cycle()

            snapshot = FilesystemResearchStateStore(config.state_root).load_snapshot()
            self.assertIsNotNone(snapshot)
            assert snapshot is not None
            self.assertEqual(snapshot.baseline_metrics["total_return"], 0.07)
            self.assertEqual(snapshot.baseline_metrics["sharpe"], 1.75)
            self.assertEqual(snapshot.baseline_metrics["score"], 4.5)
            self.assertEqual(snapshot.latest_candidate_summary["research_score"], -2.0)

    def test_success_snapshot_marks_no_trade_candidate_explicitly(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            temp_path = Path(tempdir)

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
                llm_runner=lambda **_kwargs: [
                    make_fake_decision(
                        score=0.0,
                        candidate_metrics={
                            "total_return": 0.0,
                            "sharpe": 0.0,
                            "max_drawdown": 0.0,
                            "average_turnover": 0.0,
                            "bars_processed": 2015,
                            "nonzero_turnover_steps": 0,
                        },
                        candidate_gate_failures=["no_trades_executed", "sharpe_below_gate"],
                    )
                ],
                campaign_preparer=fake_prepare_campaign,
                now_fn=lambda: datetime(2026, 3, 17, 12, 0, tzinfo=timezone.utc),
                sleep_fn=lambda _seconds: None,
            )
            worker._ensure_repo_ready = lambda: None  # type: ignore[method-assign]

            worker.run_cycle()

            snapshot = FilesystemResearchStateStore(config.state_root).load_snapshot()
            self.assertIsNotNone(snapshot)
            assert snapshot is not None
            self.assertIn(
                "Latest LLM candidate produced no trades on the evaluation window.",
                snapshot.research_blockers,
            )

    def test_success_snapshot_includes_latest_kept_config_diff(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            temp_path = Path(tempdir)

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
                llm_runner=lambda **_kwargs: [
                    make_fake_decision(
                        decision="keep",
                        score=5.25,
                        ready_for_paper=False,
                        baseline_train_config={
                            "lookback_bars": 24,
                            "top_k": 1,
                            "gross_target": 0.5,
                            "ranking_mode": "risk_adjusted",
                        },
                        candidate_train_config={
                            "lookback_bars": 24,
                            "top_k": 1,
                            "gross_target": 0.25,
                            "ranking_mode": "risk_adjusted",
                        },
                    )
                ],
                campaign_preparer=fake_prepare_campaign,
                now_fn=lambda: datetime(2026, 3, 17, 12, 0, tzinfo=timezone.utc),
                sleep_fn=lambda _seconds: None,
            )
            worker._ensure_repo_ready = lambda: None  # type: ignore[method-assign]

            worker.run_cycle()

            snapshot = FilesystemResearchStateStore(config.state_root).load_snapshot()
            self.assertIsNotNone(snapshot)
            assert snapshot is not None
            self.assertEqual(snapshot.latest_kept_summary["strategy_name"], "configurable-momentum")
            self.assertEqual(snapshot.latest_kept_summary["baseline_strategy_name"], "baseline-train-head")
            self.assertEqual(snapshot.latest_kept_summary["score_delta"], 0.75)
            self.assertEqual(
                snapshot.latest_kept_summary["config_diff"],
                [
                    {
                        "key": "gross_target",
                        "before": 0.5,
                        "after": 0.25,
                    }
                ],
            )

    def test_run_cycle_loads_leaderboard_from_worktree_results_when_configured_results_path_is_stale(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            temp_path = Path(tempdir)

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
                results_path=str(temp_path / "separate-results.tsv"),
                data_config=DataConfig(
                    exchange="bybit",
                    market="linear",
                    timeframe="5m",
                    storage_root=str(temp_path / "data"),
                ),
            )
            worktree_results = temp_path / "worktrees" / "codex-autoresearch-crypto" / "results.tsv"
            worktree_results.parent.mkdir(parents=True, exist_ok=True)
            worktree_results.write_text(
                "\t".join(
                    [
                        "recorded_at",
                        "campaign_id",
                        "decision",
                        "stage",
                        "mutation_label",
                        "provider_name",
                        "model_name",
                        "prompt_id",
                        "run_id",
                        "campaign_name",
                        "strategy_name",
                        "git_branch",
                        "git_commit",
                        "parent_commit",
                        "train_sha1",
                        "candidate_sha1",
                        "train_config_json",
                        "baseline_score",
                        "delta_score",
                        "research_score",
                        "acceptance_rate",
                        "average_total_return",
                        "average_sharpe",
                        "average_max_drawdown",
                        "worst_max_drawdown",
                        "average_turnover",
                        "ready_for_paper",
                        "gate_failures_json",
                        "runtime_seconds",
                        "failure_reason",
                        "proposal_artifact_path",
                        "artifact_path",
                    ]
                )
                + "\n"
                + "\t".join(
                    [
                        "2026-03-17T12:00:00+00:00",
                        "campaign",
                        "keep",
                        "full",
                        "llm-kept",
                        "llm",
                        "fake-gpt",
                        "",
                        "run-1",
                        "campaign",
                        "configurable-momentum",
                        "codex/autoresearch-crypto",
                        "abc123",
                        "parent123",
                        "train123",
                        "candidate123",
                        "{}",
                        "4.500000",
                        "0.750000",
                        "5.250000",
                        "1.000000",
                        "0.080000",
                        "1.500000",
                        "0.090000",
                        "0.090000",
                        "0.030000",
                        "false",
                        "[]",
                        "1.500000",
                        "",
                        "",
                        "",
                    ]
                )
                + "\n",
                encoding="utf-8",
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

            worker.run_cycle()

            snapshot = FilesystemResearchStateStore(config.state_root).load_snapshot()
            self.assertIsNotNone(snapshot)
            assert snapshot is not None
            self.assertTrue(snapshot.leaderboard)
            self.assertEqual(snapshot.leaderboard[0]["strategy_name"], "llm-kept")

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

    def test_run_forever_classifies_openai_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            temp_path = Path(tempdir)
            sleep_calls: list[float] = []

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
                failure_cooldown_seconds=45,
                timeout_failure_cooldown_seconds=12,
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
                llm_runner=lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("openai_timeout")),
                campaign_preparer=lambda **_kwargs: temp_path / "campaign.json",
                now_fn=lambda: datetime(2026, 3, 17, 12, 0, tzinfo=timezone.utc),
                sleep_fn=lambda seconds: sleep_calls.append(seconds),
            )
            worker._ensure_repo_ready = lambda: None  # type: ignore[method-assign]
            worker._ensure_active_campaign = lambda **_kwargs: (temp_path / "campaign.json", False)  # type: ignore[method-assign]

            worker.run_forever()

            snapshot = FilesystemResearchStateStore(config.state_root).load_snapshot()
            self.assertIsNotNone(snapshot)
            assert snapshot is not None
            self.assertEqual(snapshot.loop_state, "degraded")
            self.assertIn("OpenAI request timed out", snapshot.research_blockers[0])
            self.assertEqual(sleep_calls, [12.0])

    def test_failure_snapshot_carries_forward_last_healthy_state(self) -> None:
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
                timeout_failure_cooldown_seconds=15,
                max_cycles=1,
                data_config=DataConfig(
                    exchange="bybit",
                    market="linear",
                    timeframe="5m",
                    storage_root=str(temp_path / "data"),
                ),
            )
            store = FilesystemResearchStateStore(config.state_root)
            healthy_worker = LLMAutoresearchWorker(
                config=config,
                state_store=store,
                llm_runner=lambda **_kwargs: [make_fake_decision(score=1.25)],
                campaign_preparer=fake_prepare_campaign,
                now_fn=lambda: datetime(2026, 3, 17, 12, 0, tzinfo=timezone.utc),
                sleep_fn=lambda _seconds: None,
            )
            healthy_worker._ensure_repo_ready = lambda: None  # type: ignore[method-assign]
            healthy_worker.run_cycle()

            failing_worker = LLMAutoresearchWorker(
                config=config,
                state_store=store,
                llm_runner=lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("openai_timeout")),
                campaign_preparer=fake_prepare_campaign,
                now_fn=lambda: datetime(2026, 3, 17, 12, 12, tzinfo=timezone.utc),
                sleep_fn=lambda seconds: sleep_calls.append(seconds),
            )
            failing_worker._ensure_repo_ready = lambda: None  # type: ignore[method-assign]
            failing_worker.run_forever()

            snapshot = store.load_snapshot()
            self.assertIsNotNone(snapshot)
            assert snapshot is not None
            self.assertEqual(snapshot.loop_state, "degraded")
            self.assertIn("OpenAI request timed out", snapshot.research_blockers[0])
            self.assertEqual(snapshot.latest_dataset_id, "campaign")
            self.assertEqual(snapshot.baseline_metrics["score"], 4.5)
            self.assertEqual(
                snapshot.multi_window_summary["latest_decision"]["mutation_label"],
                "llm-test",
            )
            self.assertEqual(sleep_calls, [15.0])

    def test_persist_snapshot_suppresses_publish_timeout_for_degraded_status(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            temp_path = Path(tempdir)
            config = LLMWorkerConfig(
                repo_url="https://github.com/example/repo.git",
                repo_root=str(temp_path / "repo"),
                campaigns_root=str(temp_path / "campaigns"),
                active_campaign_path=str(temp_path / "active_campaign.txt"),
                worktrees_root=str(temp_path / "worktrees"),
                state_root=str(temp_path / "state"),
                artifact_root=str(temp_path / "artifacts"),
                results_path=str(temp_path / "results.tsv"),
                data_config=DataConfig(
                    exchange="bybit",
                    market="linear",
                    timeframe="5m",
                    storage_root=str(temp_path / "data"),
                ),
            )

            class FailingPublisher:
                def publish_json(self, filename, payload, message):  # type: ignore[no-untyped-def]
                    raise RuntimeError("github_publish_timeout")

            worker = LLMAutoresearchWorker(
                config=config,
                state_store=FilesystemResearchStateStore(config.state_root),
                publisher=FailingPublisher(),  # type: ignore[arg-type]
                now_fn=lambda: datetime(2026, 3, 17, 12, 0, tzinfo=timezone.utc),
                sleep_fn=lambda _seconds: None,
            )

            snapshot = worker._build_failure_snapshot(  # type: ignore[attr-defined]
                checkpoint=worker.state_store.load_llm_checkpoint(),
                failure_message="GitHub status publish timed out",
                previous_snapshot=None,
            )
            worker._persist_snapshot(  # type: ignore[attr-defined]
                snapshot,
                message="Publish degraded LLM autoresearch status",
                suppress_publish_errors=True,
            )

            stored_snapshot = FilesystemResearchStateStore(config.state_root).load_snapshot()
            self.assertIsNotNone(stored_snapshot)
            assert stored_snapshot is not None
            self.assertIn("GitHub status publish timed out", stored_snapshot.research_blockers[0])

    def test_ensure_repo_ready_seeds_configured_family_branch(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            workspace = Path(tempdir)
            source_repo = workspace / "source"
            source_repo.mkdir()
            self._git(source_repo, "init")
            self._git(source_repo, "config", "user.email", "bot@example.com")
            self._git(source_repo, "config", "user.name", "Bot")
            (source_repo / "train.py").write_text(
                "\n".join(
                    [
                        "from __future__ import annotations",
                        "TRAIN_CONFIG = {'lookback_bars': 24, 'top_k': 1, 'gross_target': 0.5}",
                        'STRATEGY_NAME = "baseline"',
                        'STRATEGY_FAMILY = "momentum"',
                        "",
                        "def build_strategy(_dataset_spec=None):",
                        "    return None",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            self._git(source_repo, "add", "train.py")
            self._git(source_repo, "commit", "-m", "Initial baseline")
            self._git(source_repo, "branch", "-M", "main")

            config = LLMWorkerConfig(
                repo_url=str(source_repo),
                repo_root=str(workspace / "repo"),
                branch_name="codex/family-mean-reversion",
                strategy_family="mean_reversion",
                campaigns_root=str(workspace / "campaigns"),
                active_campaign_path=str(workspace / "active_campaign.txt"),
                worktrees_root=str(workspace / "worktrees"),
                state_root=str(workspace / "state"),
                artifact_root=str(workspace / "artifacts"),
                results_path=str(workspace / "results.tsv"),
                data_config=DataConfig(
                    exchange="bybit",
                    market="linear",
                    timeframe="5m",
                    storage_root=str(workspace / "data"),
                ),
            )
            worker = LLMAutoresearchWorker(
                config=config,
                state_store=FilesystemResearchStateStore(config.state_root),
                llm_runner=lambda **_kwargs: [make_fake_decision()],
                campaign_preparer=lambda **kwargs: Path(kwargs["campaigns_root"]) / "campaign.json",
            )

            worker._ensure_repo_ready()

            repo_root = Path(config.repo_root)
            self.assertEqual(self._git(repo_root, "branch", "--show-current"), "main")
            self._git(repo_root, "checkout", config.branch_name)
            train_text = (repo_root / "train.py").read_text(encoding="utf-8")
            self.assertIn('STRATEGY_FAMILY = "mean_reversion"', train_text)

    def test_publisher_from_env_uses_family_specific_status_namespace(self) -> None:
        previous = {
            key: os.environ.get(key)
            for key in (
                "GITHUB_TOKEN",
                "AUTORESEARCH_STATUS_GITHUB_REPO",
                "AUTORESEARCH_STATUS_GITHUB_BRANCH",
                "AUTORESEARCH_STATUS_GITHUB_PATH",
            )
        }
        os.environ["GITHUB_TOKEN"] = "token"
        os.environ["AUTORESEARCH_STATUS_GITHUB_REPO"] = "NickMusk/autoresearch-trade-bot"
        os.environ["AUTORESEARCH_STATUS_GITHUB_BRANCH"] = "render-state"
        os.environ["AUTORESEARCH_STATUS_GITHUB_PATH"] = "status/ema_trend"
        try:
            publisher = publisher_from_env()
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

        self.assertIsNotNone(publisher)
        assert publisher is not None
        self.assertEqual(publisher.base_path, "status/ema_trend")


if __name__ == "__main__":
    unittest.main()
