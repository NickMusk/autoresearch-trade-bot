from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import autoresearch_trade_bot.llm_worker as llm_worker_module
from autoresearch_trade_bot.config import DataConfig, LLMWorkerConfig, ResearchTargetGate
from autoresearch_trade_bot.llm_worker import LLMAutoresearchWorker, publisher_from_env
from autoresearch_trade_bot.state import (
    CycleSummary,
    FilesystemResearchStateStore,
    LLMWorkerCheckpoint,
    ResearchStatusSnapshot,
)


def make_fake_decision(
    *,
    decision: str = "discard_screen",
    score: float = -10.0,
    candidate_acceptance_rate: float = 0.25,
    baseline_acceptance_rate: float = 0.75,
    ready_for_paper: bool = False,
    baseline_ready_for_paper: bool = False,
    failure_reason: str = "",
    candidate_metrics: dict | None = None,
    baseline_metrics: dict | None = None,
    candidate_gate_failures: list[str] | None = None,
    baseline_gate_failures: list[str] | None = None,
    baseline_train_config: dict | None = None,
    candidate_train_config: dict | None = None,
    baseline_strategy_name: str = "baseline-train-head",
    candidate_strategy_name: str = "configurable-momentum",
    baseline_git_commit: str = "baseline123",
    candidate_git_commit: str = "candidate123",
    baseline_train_sha1: str = "baseline-train123",
    candidate_train_sha1: str = "candidate-train123",
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
        ready_for_paper=baseline_ready_for_paper,
        average_metrics=resolved_baseline_metrics,
        research_score=4.5,
        acceptance_rate=baseline_acceptance_rate,
        gate_failures=(
            []
            if baseline_ready_for_paper
            else list(baseline_gate_failures or ["sharpe_below_gate"])
        ),
        failure_reason="",
        strategy_name=baseline_strategy_name,
        git_commit=baseline_git_commit,
        train_sha1=baseline_train_sha1,
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
        acceptance_rate=candidate_acceptance_rate,
        gate_failures=(
            []
            if ready_for_paper
            else list(candidate_gate_failures or ["acceptance_rate_below_gate"])
        ),
        failure_reason=failure_reason,
        strategy_name=candidate_strategy_name,
        git_commit=candidate_git_commit,
        train_sha1=candidate_train_sha1,
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


def make_fake_validation_report(
    *,
    validation_pass_rate: float = 0.75,
    paper_ready: bool = True,
    gate_failures: list[str] | None = None,
    average_metrics: dict | None = None,
):
    resolved_gate_failures = (
        list(gate_failures or ["sharpe_below_gate"])
        if not paper_ready
        else list(gate_failures or [])
    )
    return SimpleNamespace(
        campaign_id="validation-campaign",
        campaign_name="validation-campaign",
        strategy_name="validated-current-best",
        git_commit="baseline123",
        train_sha1="train123",
        acceptance_rate=validation_pass_rate,
        windows_passed=3,
        total_windows=3,
        gate_failures=resolved_gate_failures,
        average_metrics=average_metrics
        or {
            "total_return": 0.08,
            "sharpe": 1.5,
            "max_drawdown": 0.09,
            "average_turnover": 0.03,
            "bars_processed": 100,
        },
        artifact_path="validation/report.json",
    )


def make_fake_current_best_validator(**report_kwargs):
    return lambda **_kwargs: make_fake_validation_report(**report_kwargs)


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
                current_best_validator=make_fake_current_best_validator(),
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
            campaign_calls: list[dict] = []

            def fake_prepare_campaign(**kwargs):
                campaign_name = kwargs["campaign_name"]
                campaign_calls.append(dict(kwargs))
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
                current_best_validator=make_fake_current_best_validator(
                    validation_pass_rate=0.4,
                    paper_ready=True,
                ),
                now_fn=lambda: datetime(2026, 3, 17, 12, 0, tzinfo=timezone.utc),
                sleep_fn=lambda _seconds: None,
            )
            worker._ensure_repo_ready = lambda: None  # type: ignore[method-assign]

            result = worker.run_cycle()

            self.assertEqual(len(campaign_calls), 3)
            self.assertIn("holdout", campaign_calls[1]["campaign_name"])
            self.assertIn("rollout", campaign_calls[2]["campaign_name"])
            self.assertEqual(campaign_calls[0]["window_days"], 30)
            self.assertEqual(campaign_calls[1]["window_days"], 30)
            self.assertEqual(campaign_calls[1]["window_count"], 3)
            self.assertEqual(campaign_calls[2]["window_days"], 30)
            self.assertEqual(campaign_calls[2]["window_count"], 12)

    def test_run_cycle_records_completion_time_at_end_of_cycle(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            temp_path = Path(tempdir)
            timestamps = [
                datetime(2026, 3, 17, 12, 0, tzinfo=timezone.utc),
                datetime(2026, 3, 17, 12, 4, tzinfo=timezone.utc),
            ]

            def fake_now() -> datetime:
                if len(timestamps) > 1:
                    return timestamps.pop(0)
                return timestamps[0]

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
                campaign_preparer=lambda **kwargs: Path(kwargs["campaigns_root"]) / "campaign.json",
                current_best_validator=make_fake_current_best_validator(
                    validation_pass_rate=0.4,
                    paper_ready=True,
                ),
                now_fn=fake_now,
                sleep_fn=lambda _seconds: None,
            )
            worker._ensure_repo_ready = lambda: None  # type: ignore[method-assign]

            result = worker.run_cycle()
            checkpoint = worker.state_store.load_llm_checkpoint()

            self.assertEqual(
                result.completed_at,
                datetime(2026, 3, 17, 12, 4, tzinfo=timezone.utc),
            )
            self.assertEqual(
                checkpoint.last_cycle_completed_at,
                datetime(2026, 3, 17, 12, 4, tzinfo=timezone.utc),
            )
            history = FilesystemResearchStateStore(config.state_root).load_history()
            self.assertEqual(len(history), 1)

    def test_success_snapshot_distinguishes_current_best_readiness_from_latest_cycle_rollout(self) -> None:
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
                recent_results_limit=5,
                data_config=DataConfig(
                    exchange="bybit",
                    market="linear",
                    timeframe="5m",
                    storage_root=str(temp_path / "data"),
                ),
            )
            store = FilesystemResearchStateStore(config.state_root)
            store.save_history(
                [
                    CycleSummary(
                        cycle_id="old-1",
                        dataset_id="campaign",
                        completed_at=datetime(2026, 3, 17, 11, 0, tzinfo=timezone.utc),
                        last_processed_bar=datetime(2026, 3, 17, 10, 55, tzinfo=timezone.utc),
                        accepted=False,
                        score=1.0,
                        strategy_name="older",
                        params={"decision": "discard_screen"},
                        metrics={},
                    ),
                    CycleSummary(
                        cycle_id="old-2",
                        dataset_id="campaign",
                        completed_at=datetime(2026, 3, 17, 11, 10, tzinfo=timezone.utc),
                        last_processed_bar=datetime(2026, 3, 17, 11, 5, tzinfo=timezone.utc),
                        accepted=False,
                        score=1.0,
                        strategy_name="older",
                        params={"decision": "discard_full"},
                        metrics={},
                    ),
                ]
            )
            worker = LLMAutoresearchWorker(
                config=config,
                state_store=store,
                llm_runner=lambda **_kwargs: [
                    make_fake_decision(
                        decision="discard_screen",
                        ready_for_paper=True,
                        baseline_ready_for_paper=True,
                        score=4.0,
                    )
                ],
                campaign_preparer=fake_prepare_campaign,
                current_best_validator=make_fake_current_best_validator(
                    validation_pass_rate=0.4,
                    paper_ready=True,
                ),
                now_fn=lambda: datetime(2026, 3, 17, 12, 0, tzinfo=timezone.utc),
                sleep_fn=lambda _seconds: None,
            )
            worker._ensure_repo_ready = lambda: None  # type: ignore[method-assign]

            worker.run_cycle()

            snapshot = store.load_snapshot()
            self.assertIsNotNone(snapshot)
            assert snapshot is not None
            self.assertTrue(snapshot.current_best_ready_for_paper)
            self.assertEqual(snapshot.current_best_fast_validation_pass_rate, 0.4)
            self.assertFalse(snapshot.current_best_fast_holdout_passed)
            self.assertEqual(snapshot.current_best_validation_pass_rate, 0.0)
            self.assertFalse(snapshot.latest_cycle_rollout_ready)
            self.assertFalse(snapshot.research_rollout_ready)
            self.assertIn(
                "still has not cleared the fast holdout screen",
                " ".join(snapshot.research_blockers),
            )

    def test_current_best_can_be_validated_for_rollout_without_recent_keep(self) -> None:
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
                fast_validation_campaigns_root=str(temp_path / "validation_campaigns"),
                fast_validation_active_campaign_path=str(temp_path / "validation_active_campaign.txt"),
                rollout_validation_campaigns_root=str(temp_path / "rollout_validation_campaigns"),
                rollout_validation_active_campaign_path=str(temp_path / "rollout_validation_active_campaign.txt"),
                worktrees_root=str(temp_path / "worktrees"),
                state_root=str(temp_path / "state"),
                artifact_root=str(temp_path / "artifacts"),
                results_path=str(temp_path / "results.tsv"),
                recent_results_limit=5,
                data_config=DataConfig(
                    exchange="bybit",
                    market="linear",
                    timeframe="5m",
                    storage_root=str(temp_path / "data"),
                ),
            )
            store = FilesystemResearchStateStore(config.state_root)
            store.save_history(
                [
                    CycleSummary(
                        cycle_id="old-1",
                        dataset_id="campaign",
                        completed_at=datetime(2026, 3, 17, 11, 0, tzinfo=timezone.utc),
                        last_processed_bar=datetime(2026, 3, 17, 10, 55, tzinfo=timezone.utc),
                        accepted=False,
                        score=1.0,
                        strategy_name="older",
                        params={"decision": "discard_screen"},
                        metrics={},
                    ),
                    CycleSummary(
                        cycle_id="old-2",
                        dataset_id="campaign",
                        completed_at=datetime(2026, 3, 17, 11, 10, tzinfo=timezone.utc),
                        last_processed_bar=datetime(2026, 3, 17, 11, 5, tzinfo=timezone.utc),
                        accepted=False,
                        score=1.0,
                        strategy_name="older",
                        params={"decision": "discard_full"},
                        metrics={},
                    ),
                ]
            )
            worker = LLMAutoresearchWorker(
                config=config,
                state_store=store,
                llm_runner=lambda **_kwargs: [
                    make_fake_decision(
                        decision="discard_screen",
                        ready_for_paper=True,
                        baseline_ready_for_paper=True,
                        score=4.0,
                    )
                ],
                campaign_preparer=fake_prepare_campaign,
                current_best_validator=make_fake_current_best_validator(
                    validation_pass_rate=0.75,
                    paper_ready=True,
                ),
                now_fn=lambda: datetime(2026, 3, 17, 12, 0, tzinfo=timezone.utc),
                sleep_fn=lambda _seconds: None,
            )
            worker._ensure_repo_ready = lambda: None  # type: ignore[method-assign]

            worker.run_cycle()

            snapshot = store.load_snapshot()
            self.assertIsNotNone(snapshot)
            assert snapshot is not None
            self.assertFalse(snapshot.latest_cycle_rollout_ready)
            self.assertTrue(snapshot.current_best_ready_for_paper)
            self.assertTrue(snapshot.current_best_fast_holdout_passed)
            self.assertTrue(snapshot.current_best_validated_for_rollout)
            self.assertTrue(snapshot.research_rollout_ready)
            self.assertTrue(snapshot.accepted_for_paper)
            self.assertEqual(snapshot.current_best_fast_validation_pass_rate, 0.75)
            self.assertEqual(snapshot.current_best_validation_pass_rate, 0.75)
            self.assertEqual(snapshot.mutation_win_rate, 0.0)

    def test_rollout_champion_can_differ_from_research_champion(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            temp_path = Path(tempdir)

            def fake_prepare_campaign(**kwargs):
                campaign_path = Path(kwargs["campaigns_root"]) / f"{kwargs['campaign_name']}.json"
                campaign_path.parent.mkdir(parents=True, exist_ok=True)
                campaign_path.write_text("{}", encoding="utf-8")
                Path(kwargs["active_pointer_path"]).write_text(
                    str(campaign_path.resolve()),
                    encoding="utf-8",
                )
                return campaign_path

            def fake_current_best_validator(*, candidate_sha1: str, parent_commit: str, **_kwargs):
                if candidate_sha1 == "stable-train":
                    return make_fake_validation_report(validation_pass_rate=0.9, paper_ready=True)
                if candidate_sha1 == "unstable-train":
                    return make_fake_validation_report(validation_pass_rate=0.2, paper_ready=True)
                return make_fake_validation_report(validation_pass_rate=0.0, paper_ready=False)

            config = LLMWorkerConfig(
                repo_url="https://github.com/example/repo.git",
                repo_root=str(temp_path / "repo"),
                campaigns_root=str(temp_path / "campaigns"),
                active_campaign_path=str(temp_path / "active_campaign.txt"),
                fast_validation_campaigns_root=str(temp_path / "validation_campaigns"),
                fast_validation_active_campaign_path=str(temp_path / "validation_active_campaign.txt"),
                rollout_validation_campaigns_root=str(temp_path / "rollout_validation_campaigns"),
                rollout_validation_active_campaign_path=str(temp_path / "rollout_validation_active_campaign.txt"),
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
            worktree_root = Path(config.worktrees_root) / "codex-autoresearch-crypto"
            worktree_root.mkdir(parents=True, exist_ok=True)
            (worktree_root / "train.py").write_text("TRAIN_CONFIG = {}\n", encoding="utf-8")
            store = FilesystemResearchStateStore(config.state_root)

            stable_worker = LLMAutoresearchWorker(
                config=config,
                state_store=store,
                llm_runner=lambda **_kwargs: [
                    make_fake_decision(
                        decision="keep",
                        score=5.0,
                        ready_for_paper=True,
                        baseline_ready_for_paper=True,
                        candidate_strategy_name="stable-rollout",
                        candidate_git_commit="stable-commit",
                        candidate_train_sha1="stable-train",
                    )
                ],
                campaign_preparer=fake_prepare_campaign,
                current_best_validator=fake_current_best_validator,
                now_fn=lambda: datetime(2026, 3, 17, 12, 0, tzinfo=timezone.utc),
                sleep_fn=lambda _seconds: None,
            )
            stable_worker._ensure_repo_ready = lambda: None  # type: ignore[method-assign]
            stable_worker._candidate_train_path = lambda **_kwargs: worktree_root / "train.py"  # type: ignore[method-assign]
            stable_worker.run_cycle()

            unstable_worker = LLMAutoresearchWorker(
                config=config,
                state_store=store,
                llm_runner=lambda **_kwargs: [
                    make_fake_decision(
                        decision="keep",
                        score=9.0,
                        ready_for_paper=True,
                        baseline_ready_for_paper=True,
                        baseline_strategy_name="stable-rollout",
                        baseline_git_commit="stable-commit",
                        baseline_train_sha1="stable-train",
                        candidate_strategy_name="high-score-unstable",
                        candidate_git_commit="unstable-commit",
                        candidate_train_sha1="unstable-train",
                    )
                ],
                campaign_preparer=fake_prepare_campaign,
                current_best_validator=fake_current_best_validator,
                now_fn=lambda: datetime(2026, 3, 17, 12, 30, tzinfo=timezone.utc),
                sleep_fn=lambda _seconds: None,
            )
            unstable_worker._ensure_repo_ready = lambda: None  # type: ignore[method-assign]
            unstable_worker._candidate_train_path = lambda **_kwargs: worktree_root / "train.py"  # type: ignore[method-assign]
            unstable_worker.run_cycle()

            snapshot = store.load_snapshot()
            self.assertIsNotNone(snapshot)
            assert snapshot is not None
            self.assertEqual(snapshot.research_champion_summary["strategy_name"], "high-score-unstable")
            self.assertEqual(snapshot.rollout_champion_summary["strategy_name"], "stable-rollout")
            self.assertTrue(snapshot.research_rollout_ready)
            self.assertTrue(snapshot.accepted_for_paper)
            self.assertFalse(snapshot.latest_cycle_rollout_ready)

    def test_current_best_validation_cache_reuses_same_strategy_and_campaign(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            temp_path = Path(tempdir)
            validator_calls = {"count": 0}

            def fake_prepare_campaign(**kwargs):
                campaign_path = Path(kwargs["campaigns_root"]) / "campaign.json"
                campaign_path.parent.mkdir(parents=True, exist_ok=True)
                campaign_path.write_text("{}", encoding="utf-8")
                Path(kwargs["active_pointer_path"]).write_text(
                    str(campaign_path.resolve()),
                    encoding="utf-8",
                )
                return campaign_path

            def fake_current_best_validator(**_kwargs):
                validator_calls["count"] += 1
                return make_fake_validation_report(
                    validation_pass_rate=0.8,
                    paper_ready=True,
                )

            config = LLMWorkerConfig(
                repo_url="https://github.com/example/repo.git",
                repo_root=str(temp_path / "repo"),
                campaigns_root=str(temp_path / "campaigns"),
                active_campaign_path=str(temp_path / "active_campaign.txt"),
                fast_validation_campaigns_root=str(temp_path / "validation_campaigns"),
                fast_validation_active_campaign_path=str(temp_path / "validation_active_campaign.txt"),
                fast_validation_refresh_interval_seconds=86_400,
                rollout_validation_campaigns_root=str(temp_path / "rollout_validation_campaigns"),
                rollout_validation_active_campaign_path=str(temp_path / "rollout_validation_active_campaign.txt"),
                rollout_validation_refresh_interval_seconds=86_400,
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
            store = FilesystemResearchStateStore(config.state_root)
            worker = LLMAutoresearchWorker(
                config=config,
                state_store=store,
                llm_runner=lambda **_kwargs: [
                    make_fake_decision(
                        decision="discard_screen",
                        ready_for_paper=True,
                        baseline_ready_for_paper=True,
                    )
                ],
                campaign_preparer=fake_prepare_campaign,
                current_best_validator=fake_current_best_validator,
                now_fn=lambda: datetime(2026, 3, 17, 12, 0, tzinfo=timezone.utc),
                sleep_fn=lambda _seconds: None,
            )
            worker._ensure_repo_ready = lambda: None  # type: ignore[method-assign]
            worker.run_cycle()

            second_worker = LLMAutoresearchWorker(
                config=config,
                state_store=store,
                llm_runner=lambda **_kwargs: [
                    make_fake_decision(
                        decision="discard_full",
                        ready_for_paper=True,
                        baseline_ready_for_paper=True,
                    )
                ],
                campaign_preparer=fake_prepare_campaign,
                current_best_validator=fake_current_best_validator,
                now_fn=lambda: datetime(2026, 3, 17, 12, 30, tzinfo=timezone.utc),
                sleep_fn=lambda _seconds: None,
            )
            second_worker._ensure_repo_ready = lambda: None  # type: ignore[method-assign]
            second_worker.run_cycle()

            self.assertEqual(validator_calls["count"], 2)

    def test_acceptance_rate_ignores_validation_errors_but_tracks_generation_validity(self) -> None:
        history = [
            SimpleNamespace(accepted=False, params={"decision": "discard_error"}),
            SimpleNamespace(accepted=False, params={"decision": "discard_screen"}),
            SimpleNamespace(accepted=True, params={"decision": "keep"}),
        ]
        self.assertEqual(LLMAutoresearchWorker._evaluation_acceptance_rate(history), 0.5)
        self.assertEqual(LLMAutoresearchWorker._generation_validity_rate(history), 2 / 3)

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
                current_best_validator=make_fake_current_best_validator(),
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
                current_best_validator=make_fake_current_best_validator(),
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
                current_best_validator=make_fake_current_best_validator(),
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
                current_best_validator=make_fake_current_best_validator(
                    validation_pass_rate=0.8,
                    paper_ready=True,
                ),
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
            self.assertEqual(snapshot.latest_decision["candidate_acceptance_rate"], 0.25)
            self.assertEqual(
                snapshot.latest_decision["candidate_average_metrics"]["total_return"],
                -0.1,
            )
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
                current_best_validator=make_fake_current_best_validator(),
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
                current_best_validator=make_fake_current_best_validator(),
                now_fn=lambda: datetime(2026, 3, 17, 12, 0, tzinfo=timezone.utc),
                sleep_fn=lambda _seconds: None,
            )
            worker._ensure_repo_ready = lambda: None  # type: ignore[method-assign]
            worker.run_cycle()
            self.assertEqual(len(campaign_calls), 3)

            second_worker = LLMAutoresearchWorker(
                config=config,
                state_store=store,
                llm_runner=lambda **_kwargs: [make_fake_decision(score=-9.0)],
                campaign_preparer=fake_prepare_campaign,
                current_best_validator=make_fake_current_best_validator(),
                now_fn=lambda: datetime(2026, 3, 17, 13, 5, tzinfo=timezone.utc),
                sleep_fn=lambda _seconds: None,
            )
            second_worker._ensure_repo_ready = lambda: None  # type: ignore[method-assign]
            second_worker.run_cycle()
            self.assertEqual(len(campaign_calls), 3)

    def test_run_forever_persists_transient_snapshot_on_first_failure(self) -> None:
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
            self.assertEqual(snapshot.loop_state, "transient_error")
            self.assertEqual(snapshot.phase, "LLM autoresearch worker transient error")
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
            self.assertEqual(snapshot.loop_state, "transient_error")
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
                current_best_validator=make_fake_current_best_validator(),
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
            self.assertEqual(snapshot.loop_state, "transient_error")
            self.assertIn("OpenAI request timed out", snapshot.research_blockers[0])
            self.assertEqual(snapshot.latest_dataset_id, "campaign")
            self.assertEqual(snapshot.baseline_metrics["score"], 4.5)
            self.assertEqual(
                snapshot.multi_window_summary["latest_decision"]["mutation_label"],
                "llm-test",
            )
            self.assertEqual(sleep_calls, [15.0])

    def test_failure_snapshot_becomes_degraded_after_threshold_failures(self) -> None:
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
                degraded_failure_threshold=3,
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
                now_fn=lambda: datetime(2026, 3, 17, 12, 0, tzinfo=timezone.utc),
                sleep_fn=lambda _seconds: None,
            )

            snapshot = worker._build_failure_snapshot(  # type: ignore[attr-defined]
                checkpoint=worker.state_store.load_llm_checkpoint().__class__(consecutive_failures=3),
                failure_message="OpenAI request timed out",
                previous_snapshot=None,
            )

            self.assertEqual(snapshot.loop_state, "degraded")
            self.assertEqual(snapshot.phase, "LLM autoresearch worker degraded")

    def test_failure_snapshot_clears_stale_rollout_ready_but_keeps_champion_certification(self) -> None:
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
            worker = LLMAutoresearchWorker(
                config=config,
                state_store=FilesystemResearchStateStore(config.state_root),
                now_fn=lambda: datetime(2026, 3, 17, 12, 0, tzinfo=timezone.utc),
                sleep_fn=lambda _seconds: None,
            )
            previous_snapshot = ResearchStatusSnapshot(
                mission="mission",
                phase="active",
                research_rollout_ready=True,
                research_blockers=[],
                baseline_strategy="baseline",
                promotion_gate={"min_sharpe": 1.0},
                baseline_metrics={"score": 4.5},
                accepted_for_paper=True,
                current_best_ready_for_paper=True,
                current_best_fast_validation_pass_rate=1.0,
                current_best_fast_holdout_passed=True,
                current_best_validation_pass_rate=0.8,
                current_best_validated_for_rollout=True,
                latest_cycle_rollout_ready=True,
                next_milestones=[],
                loop_state="holding",
                latest_dataset_id="dataset",
                latest_cycle_completed_at="2026-03-17T11:55:00+00:00",
                last_processed_bar="2026-03-17T11:50:00+00:00",
                recent_acceptance_rate=0.5,
                evaluation_acceptance_rate=0.5,
                generation_validity_rate=1.0,
                mutation_win_rate=0.5,
                consecutive_failures=0,
                current_best_fast_validation_summary={"passes_stage_gate": True},
                current_best_validation_summary={"validated_for_rollout": True},
                research_champion_summary={"strategy_name": "research-champion"},
                rollout_champion_summary={
                    "strategy_name": "rollout-champion",
                    "rollout_validation_summary": {"validated_for_rollout": True},
                },
                rollout_candidate_shortlist=[],
                multi_window_summary={},
                leaderboard=[],
                latest_decision={"decision": "keep"},
                latest_candidate_summary={},
                current_best_strategy_name="research-champion",
                latest_kept_summary={},
            )

            snapshot = worker._build_failure_snapshot(  # type: ignore[attr-defined]
                checkpoint=LLMWorkerCheckpoint(consecutive_failures=1),
                failure_message="OpenAI request timed out",
                previous_snapshot=previous_snapshot,
            )

            self.assertFalse(snapshot.research_rollout_ready)
            self.assertTrue(snapshot.current_best_validated_for_rollout)
            self.assertTrue(snapshot.current_best_ready_for_paper)
            self.assertTrue(snapshot.accepted_for_paper)
            self.assertFalse(snapshot.latest_cycle_rollout_ready)

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
            with patch("autoresearch_trade_bot.llm_worker._log_runtime_event") as mocked_log:
                worker._persist_snapshot(  # type: ignore[attr-defined]
                    snapshot,
                    message="Publish degraded LLM autoresearch status",
                    suppress_publish_errors=True,
                )

            stored_snapshot = FilesystemResearchStateStore(config.state_root).load_snapshot()
            self.assertIsNotNone(stored_snapshot)
            assert stored_snapshot is not None
            self.assertIn("GitHub status publish timed out", stored_snapshot.research_blockers[0])
            mocked_log.assert_called()
            self.assertIn("LLM worker status publish failed", mocked_log.call_args.args[0])

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
                current_best_validator=make_fake_current_best_validator(),
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

    def test_main_bootstraps_history_on_start_when_enabled(self) -> None:
        previous = dict(os.environ)
        try:
            os.environ["AUTORESEARCH_LLM_BOOTSTRAP_HISTORY_ON_START"] = "1"
            with patch(
                "autoresearch_trade_bot.llm_worker.maybe_install_history_dataset_from_env"
            ) as mocked_install:
                mocked_install.return_value = SimpleNamespace(
                    manifest_path=None,
                    installed=False,
                )
                with patch("autoresearch_trade_bot.history_refresh.history_refresh_config_from_env") as mocked_config:
                    mocked_config.return_value = SimpleNamespace(
                        full_lookback_days=365,
                        bootstrap_skip_open_interest=True,
                    )
                    with patch("autoresearch_trade_bot.history_refresh.run_history_refresh_once") as mocked_run:
                        with patch.object(llm_worker_module, "llm_worker_config_from_env") as mocked_worker_config:
                            mocked_worker_config.return_value = LLMWorkerConfig(
                                repo_url="https://example.com/repo.git"
                            )
                            with patch("autoresearch_trade_bot.llm_worker.FilesystemResearchStateStore"):
                                worker_double = SimpleNamespace(run_forever=lambda: None)
                                with patch(
                                    "autoresearch_trade_bot.llm_worker.LLMAutoresearchWorker",
                                    return_value=worker_double,
                                ):
                                    llm_worker_module.main()

            self.assertEqual(mocked_run.call_count, 1)
            kwargs = mocked_run.call_args.kwargs
            self.assertEqual(kwargs["lookback_days"], 365)
            self.assertTrue(kwargs["skip_open_interest"])
        finally:
            os.environ.clear()
            os.environ.update(previous)

    def test_main_skips_bootstrap_when_dataset_installer_has_manifest(self) -> None:
        previous = dict(os.environ)
        try:
            os.environ["AUTORESEARCH_LLM_BOOTSTRAP_HISTORY_ON_START"] = "1"
            with patch(
                "autoresearch_trade_bot.llm_worker.maybe_install_history_dataset_from_env"
            ) as mocked_install:
                mocked_install.return_value = SimpleNamespace(
                    manifest_path=Path("/tmp/autoresearch/data/manifest.json"),
                    installed=False,
                )
                with patch("autoresearch_trade_bot.history_refresh.run_history_refresh_once") as mocked_run:
                    with patch.object(llm_worker_module, "llm_worker_config_from_env") as mocked_worker_config:
                        mocked_worker_config.return_value = LLMWorkerConfig(
                            repo_url="https://example.com/repo.git"
                        )
                        with patch("autoresearch_trade_bot.llm_worker.FilesystemResearchStateStore"):
                            worker_double = SimpleNamespace(run_forever=lambda: None)
                            with patch(
                                "autoresearch_trade_bot.llm_worker.LLMAutoresearchWorker",
                                return_value=worker_double,
                            ):
                                llm_worker_module.main()

            self.assertEqual(mocked_run.call_count, 0)
            self.assertEqual(mocked_install.call_count, 1)
        finally:
            os.environ.clear()
            os.environ.update(previous)

    def test_main_emits_startup_checkpoints(self) -> None:
        previous = dict(os.environ)
        try:
            with patch(
                "autoresearch_trade_bot.llm_worker.maybe_install_history_dataset_from_env"
            ) as mocked_install:
                mocked_install.return_value = SimpleNamespace(
                    manifest_path=Path("/tmp/autoresearch/data/manifest.json"),
                    installed=False,
                )
                with patch.object(llm_worker_module, "llm_worker_config_from_env") as mocked_worker_config:
                    mocked_worker_config.return_value = LLMWorkerConfig(
                        repo_url="https://example.com/repo.git",
                        data_config=DataConfig(
                            exchange="binance",
                            market="usdm_futures",
                            timeframe="5m",
                            storage_root="/tmp/data",
                            local_only=False,
                        ),
                    )
                    with patch("autoresearch_trade_bot.llm_worker.FilesystemResearchStateStore"):
                        worker_double = SimpleNamespace(run_forever=lambda: None)
                        with patch(
                            "autoresearch_trade_bot.llm_worker.LLMAutoresearchWorker",
                            return_value=worker_double,
                        ):
                            with patch("builtins.print") as mocked_print:
                                llm_worker_module.main()
        finally:
            os.environ.clear()
            os.environ.update(previous)

        printed = [" ".join(str(arg) for arg in call.args) for call in mocked_print.call_args_list]
        self.assertIn("LLM worker startup: checking history dataset install", printed)
        self.assertTrue(
            any("LLM worker startup: dataset install complete" in line for line in printed)
        )
        self.assertIn("LLM worker startup: installed manifest present, skipping bootstrap", printed)
        self.assertIn("LLM worker startup: loading worker config", printed)
        self.assertTrue(
            any("LLM worker startup: config loaded (exchange=binance, market=usdm_futures" in line for line in printed)
        )
        self.assertIn("LLM worker startup: initializing state store and worker", printed)
        self.assertIn("LLM worker startup: entering run loop", printed)

    def test_run_forever_emits_cycle_checkpoints(self) -> None:
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
                max_cycles=1,
                data_config=DataConfig(
                    exchange="binance",
                    market="usdm_futures",
                    timeframe="5m",
                    storage_root=str(temp_path / "data"),
                ),
            )
            worker = LLMAutoresearchWorker(
                config=config,
                state_store=FilesystemResearchStateStore(config.state_root),
                now_fn=lambda: datetime(2026, 3, 23, 6, 30, tzinfo=timezone.utc),
                sleep_fn=lambda _seconds: None,
            )
            with patch.object(worker, "run_cycle", return_value=SimpleNamespace()) as mocked_cycle:
                with patch("builtins.print") as mocked_print:
                    worker.run_forever()

        printed = [" ".join(str(arg) for arg in call.args) for call in mocked_print.call_args_list]
        self.assertEqual(mocked_cycle.call_count, 1)
        self.assertTrue(any("LLM worker loop: starting cycle at" in line for line in printed))
        self.assertTrue(any("LLM worker loop: cycle completed at" in line for line in printed))

    def test_llm_worker_config_from_env_supports_local_only_history_mode(self) -> None:
        previous = dict(os.environ)
        try:
            os.environ["AUTORESEARCH_LLM_LOCAL_DATA_ONLY"] = "1"
            os.environ["AUTORESEARCH_LLM_DATA_ROOT"] = "/tmp/autoresearch/llm/data"
            with patch.object(llm_worker_module, "_discover_repo_url", return_value="https://example.com/repo.git"):
                config = llm_worker_module.llm_worker_config_from_env()
        finally:
            os.environ.clear()
            os.environ.update(previous)

        self.assertTrue(config.data_config.local_only)
        self.assertEqual(
            config.data_config.history_readiness_state_path,
            "/tmp/autoresearch/llm/history_refresh_state.json",
        )

    def test_llm_worker_config_from_env_supports_disabling_open_interest(self) -> None:
        previous = dict(os.environ)
        try:
            os.environ["AUTORESEARCH_INCLUDE_OPEN_INTEREST"] = "0"
            with patch.object(llm_worker_module, "_discover_repo_url", return_value="https://example.com/repo.git"):
                config = llm_worker_module.llm_worker_config_from_env()
        finally:
            os.environ.clear()
            os.environ.update(previous)

        self.assertFalse(config.data_config.include_open_interest)

    def test_llm_worker_config_from_env_falls_back_to_default_repo_url(self) -> None:
        previous = dict(os.environ)
        try:
            os.environ.pop("AUTORESEARCH_LLM_REPO_URL", None)
            with patch.object(
                llm_worker_module,
                "_discover_repo_url",
                return_value=llm_worker_module.DEFAULT_LLM_REPO_URL,
            ):
                config = llm_worker_module.llm_worker_config_from_env()
        finally:
            os.environ.clear()
            os.environ.update(previous)

        self.assertEqual(config.repo_url, llm_worker_module.DEFAULT_LLM_REPO_URL)

    def test_llm_worker_config_from_env_uses_expanded_recent_results_default(self) -> None:
        previous = dict(os.environ)
        try:
            os.environ.pop("AUTORESEARCH_LLM_RECENT_RESULTS_LIMIT", None)
            with patch.object(
                llm_worker_module,
                "_discover_repo_url",
                return_value=llm_worker_module.DEFAULT_LLM_REPO_URL,
            ):
                config = llm_worker_module.llm_worker_config_from_env()
        finally:
            os.environ.clear()
            os.environ.update(previous)

        self.assertEqual(config.recent_results_limit, 12)

    def test_discover_repo_url_falls_back_to_default_when_git_lookup_fails(self) -> None:
        with patch(
            "autoresearch_trade_bot.llm_worker.subprocess.run",
            side_effect=subprocess.CalledProcessError(2, ["git", "remote", "get-url", "origin"]),
        ):
            repo_url = llm_worker_module._discover_repo_url()

        self.assertEqual(repo_url, llm_worker_module.DEFAULT_LLM_REPO_URL)


if __name__ == "__main__":
    unittest.main()
