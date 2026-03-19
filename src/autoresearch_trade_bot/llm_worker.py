from __future__ import annotations

import json
import os
import subprocess
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Sequence

from .autoresearch import (
    diff_train_configs,
    ensure_git_identity,
    prepare_campaign,
    resolve_campaign_path,
    slugify,
)
from .config import DataConfig, LLMWorkerConfig, ResearchTargetGate
from .datasets import timeframe_to_timedelta
from .family_wave import ensure_family_branch_seeded
from .mutations import load_recent_results, run_llm_mutation_campaign
from .state import (
    CycleSummary,
    FilesystemResearchStateStore,
    GitHubStatusPublisher,
    LeaderboardEntry,
    LLMWorkerCheckpoint,
    ResearchStatusSnapshot,
)


@dataclass(frozen=True)
class LLMCycleResult:
    cycle_id: str
    campaign_id: str
    campaign_path: str
    completed_at: datetime
    latest_closed_bar: datetime
    refreshed_campaign: bool
    decisions: list[dict[str, object]]
    recent_acceptance_rate: float
    rollout_ready: bool
    baseline_report: dict[str, object]
    latest_report: dict[str, object]


class LLMAutoresearchWorker:
    def __init__(
        self,
        config: LLMWorkerConfig,
        state_store: FilesystemResearchStateStore,
        publisher: GitHubStatusPublisher | None = None,
        now_fn: Callable[[], datetime] | None = None,
        sleep_fn: Callable[[float], None] | None = None,
        llm_runner: Callable[..., list] = run_llm_mutation_campaign,
        campaign_preparer: Callable[..., Path] = prepare_campaign,
    ) -> None:
        self.config = config
        self.state_store = state_store
        self.publisher = publisher
        self.now_fn = now_fn or (lambda: datetime.now(timezone.utc))
        self.sleep_fn = sleep_fn or time.sleep
        self.llm_runner = llm_runner
        self.campaign_preparer = campaign_preparer

    def run_forever(self) -> None:
        cycle_attempts = 0
        while self.config.max_cycles is None or cycle_attempts < self.config.max_cycles:
            checkpoint = self.state_store.load_llm_checkpoint()
            now = self._ensure_utc(self.now_fn())
            if not self._is_cycle_due(now, checkpoint):
                self.sleep_fn(self._seconds_until_next_cycle(now, checkpoint))
                continue
            try:
                self.run_cycle(now=now)
            except Exception as exc:  # noqa: BLE001
                failure_message = self._normalize_failure_message(exc)
                previous_snapshot = self.state_store.load_snapshot()
                failed_checkpoint = LLMWorkerCheckpoint(
                    last_cycle_completed_at=checkpoint.last_cycle_completed_at,
                    last_campaign_refreshed_at=checkpoint.last_campaign_refreshed_at,
                    consecutive_failures=checkpoint.consecutive_failures + 1,
                )
                self.state_store.save_llm_checkpoint(failed_checkpoint)
                snapshot = self._build_failure_snapshot(
                    checkpoint=failed_checkpoint,
                    failure_message=failure_message,
                    previous_snapshot=previous_snapshot,
                )
                self._persist_snapshot(
                    snapshot,
                    message="Publish degraded LLM autoresearch status",
                    suppress_publish_errors=True,
                )
                self.sleep_fn(self._failure_cooldown_seconds(failure_message))
            finally:
                cycle_attempts += 1

    def run_cycle(self, now: datetime | None = None) -> LLMCycleResult:
        cycle_now = self._ensure_utc(now or self.now_fn())
        latest_closed_bar = self.latest_closed_bar(cycle_now)
        checkpoint = self.state_store.load_llm_checkpoint()

        self._ensure_repo_ready()
        campaign_path, refreshed_campaign = self._ensure_active_campaign(
            latest_closed_bar=latest_closed_bar,
            checkpoint=checkpoint,
        )
        decisions = self.llm_runner(
            campaign_path=campaign_path,
            repo_root=self.config.repo_root,
            branch_name=self.config.branch_name,
            model_name=self.config.model_name,
            max_mutations=self.config.max_mutations_per_cycle,
            worktrees_root=self.config.worktrees_root,
            recent_results_limit=self.config.recent_results_limit,
            openai_timeout_seconds=self.config.openai_timeout_seconds,
            openai_max_retries=self.config.openai_max_retries,
            openai_retry_backoff_seconds=self.config.openai_retry_backoff_seconds,
        )
        latest_decision = decisions[-1]
        cycle_result = self._record_cycle(
            campaign_path=campaign_path,
            latest_closed_bar=latest_closed_bar,
            cycle_now=cycle_now,
            refreshed_campaign=refreshed_campaign,
            latest_decision=latest_decision,
        )
        updated_checkpoint = LLMWorkerCheckpoint(
            last_cycle_completed_at=cycle_now,
            last_campaign_refreshed_at=(
                latest_closed_bar if refreshed_campaign else checkpoint.last_campaign_refreshed_at
            ),
            consecutive_failures=0,
        )
        self.state_store.save_llm_checkpoint(updated_checkpoint)
        snapshot = self._build_success_snapshot(cycle_result, updated_checkpoint)
        self._persist_snapshot(snapshot, message=f"Publish LLM autoresearch status for {cycle_result.cycle_id}")
        return cycle_result

    def latest_closed_bar(self, now: datetime) -> datetime:
        resolved_now = self._ensure_utc(now)
        step = timeframe_to_timedelta(self.config.data_config.timeframe)
        epoch_seconds = int(resolved_now.timestamp())
        step_seconds = int(step.total_seconds())
        floored_seconds = epoch_seconds - (epoch_seconds % step_seconds)
        return datetime.fromtimestamp(floored_seconds, tz=timezone.utc)

    def _is_cycle_due(self, now: datetime, checkpoint: LLMWorkerCheckpoint) -> bool:
        if self.config.cycle_interval_seconds <= 0:
            return True
        if checkpoint.last_cycle_completed_at is None:
            return True
        elapsed = now - checkpoint.last_cycle_completed_at
        return elapsed.total_seconds() >= self.config.cycle_interval_seconds

    def _seconds_until_next_cycle(self, now: datetime, checkpoint: LLMWorkerCheckpoint) -> float:
        if self.config.cycle_interval_seconds <= 0:
            return 0.0
        if checkpoint.last_cycle_completed_at is None:
            return 1.0
        next_due = checkpoint.last_cycle_completed_at + timedelta(
            seconds=self.config.cycle_interval_seconds
        )
        return max((next_due - now).total_seconds(), 1.0)

    def _ensure_active_campaign(
        self,
        *,
        latest_closed_bar: datetime,
        checkpoint: LLMWorkerCheckpoint,
    ) -> tuple[Path, bool]:
        active_pointer = Path(self.config.active_campaign_path)
        refresh_due = (
            checkpoint.last_campaign_refreshed_at is None
            or (latest_closed_bar - checkpoint.last_campaign_refreshed_at).total_seconds()
            >= self.config.campaign_refresh_interval_seconds
            or not active_pointer.exists()
        )
        if not refresh_due:
            resolved_path = resolve_campaign_path(pointer_path=active_pointer)
            if not resolved_path.exists():
                refresh_due = True
        if not refresh_due:
            return (resolved_path, False)

        campaign_name = (
            f"render-{self.config.data_config.exchange}-{self.config.data_config.market}-"
            f"{self.config.data_config.timeframe}-{latest_closed_bar.strftime('%Y%m%dT%H%M%SZ')}"
        )
        campaign_path = self.campaign_preparer(
            campaign_name=campaign_name,
            exchange=self.config.data_config.exchange,
            market=self.config.data_config.market,
            timeframe=self.config.data_config.timeframe,
            symbols=self.config.symbols,
            anchor_end=latest_closed_bar,
            window_days=self.config.history_window_days,
            window_count=self.config.campaign_window_count,
            storage_root=self.config.data_config.storage_root,
            campaigns_root=self.config.campaigns_root,
            active_pointer_path=self.config.active_campaign_path,
            target_gate=self.config.target_gate,
        )
        return campaign_path, True

    def _ensure_repo_ready(self) -> None:
        repo_root = Path(self.config.repo_root)
        repo_root.parent.mkdir(parents=True, exist_ok=True)
        if not (repo_root / ".git").exists():
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--branch",
                    "main",
                    "--single-branch",
                    self.config.repo_url,
                    str(repo_root),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            ensure_git_identity(repo_root)
        else:
            subprocess.run(
                ["git", "checkout", "main"],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                ["git", "pull", "--ff-only", "origin", "main"],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            )
        ensure_git_identity(repo_root)
        ensure_family_branch_seeded(
            repo_root=repo_root,
            strategy_family=self.config.strategy_family,
            branch_name=self.config.branch_name,
            base_ref="main",
        )

    def _record_cycle(
        self,
        *,
        campaign_path: Path,
        latest_closed_bar: datetime,
        cycle_now: datetime,
        refreshed_campaign: bool,
        latest_decision,
    ) -> LLMCycleResult:
        recent_results = self._load_recent_results(limit=50)
        history = self.state_store.load_history()
        summary = CycleSummary(
            cycle_id=uuid.uuid4().hex[:12],
            dataset_id=Path(campaign_path).stem,
            completed_at=cycle_now,
            last_processed_bar=latest_closed_bar,
            accepted=latest_decision.decision == "keep",
            score=float(latest_decision.candidate_score),
            strategy_name=str(latest_decision.mutation_label),
            params={
                "decision": latest_decision.decision,
                "provider_name": latest_decision.report.provider_name,
                "model_name": latest_decision.report.model_name,
            },
            metrics={
                **latest_decision.report.average_metrics,
                "research_score": latest_decision.report.research_score,
            },
        )
        recent_history = (history + [summary])[-max(self.config.recent_results_limit, 10) :]
        self.state_store.save_history(recent_history)
        recent_acceptance_rate = self._acceptance_rate(recent_history)

        report = latest_decision.report
        baseline_report = latest_decision.baseline_report
        rollout_ready = (
            latest_decision.decision == "keep"
            and report.ready_for_paper
            and recent_acceptance_rate >= self.config.target_gate.min_acceptance_rate
        )
        leaderboard = self._build_leaderboard(recent_results)
        self.state_store.save_leaderboard(
            [LeaderboardEntry.from_dict(item) for item in leaderboard]
        )
        return LLMCycleResult(
            cycle_id=summary.cycle_id,
            campaign_id=Path(campaign_path).stem,
            campaign_path=str(campaign_path),
            completed_at=cycle_now,
            latest_closed_bar=latest_closed_bar,
            refreshed_campaign=refreshed_campaign,
            decisions=[
                {
                    "decision": latest_decision.decision,
                    "stage": latest_decision.stage,
                    "mutation_label": latest_decision.mutation_label,
                    "baseline_score": latest_decision.baseline_score,
                    "candidate_score": latest_decision.candidate_score,
                    "kept_commit": latest_decision.kept_commit,
                    "failure_reason": latest_decision.report.failure_reason,
                    "provider_name": latest_decision.report.provider_name,
                    "model_name": latest_decision.report.model_name,
                    "gate_failures": latest_decision.report.gate_failures,
                }
            ],
            recent_acceptance_rate=recent_acceptance_rate,
            rollout_ready=rollout_ready,
            baseline_report={
                "strategy_name": baseline_report.strategy_name,
                "research_score": baseline_report.research_score,
                "ready_for_paper": baseline_report.ready_for_paper,
                "average_metrics": baseline_report.average_metrics,
                "train_config": baseline_report.train_config,
                "gate_failures": baseline_report.gate_failures,
                "provider_name": baseline_report.provider_name,
                "model_name": baseline_report.model_name,
                "failure_reason": baseline_report.failure_reason,
            },
            latest_report={
                "strategy_name": report.strategy_name,
                "research_score": report.research_score,
                "ready_for_paper": report.ready_for_paper,
                "average_metrics": report.average_metrics,
                "train_config": report.train_config,
                "gate_failures": report.gate_failures,
                "provider_name": report.provider_name,
                "model_name": report.model_name,
                "failure_reason": report.failure_reason,
            },
        )

    def _build_success_snapshot(
        self,
        cycle_result: LLMCycleResult,
        checkpoint: LLMWorkerCheckpoint,
    ) -> ResearchStatusSnapshot:
        report = cycle_result.latest_report
        baseline_report = cycle_result.baseline_report
        baseline_metrics = baseline_report["average_metrics"]
        latest_decision = cycle_result.decisions[-1]
        latest_kept_summary = {}
        if latest_decision["decision"] == "keep":
            config_diff = diff_train_configs(
                dict(baseline_report.get("train_config", {})),
                dict(report.get("train_config", {})),
            )
            latest_kept_summary = {
                "strategy_name": report["strategy_name"],
                "baseline_strategy_name": baseline_report["strategy_name"],
                "score_delta": round(
                    float(report["research_score"]) - float(baseline_report["research_score"]),
                    6,
                ),
                "config_diff": config_diff,
            }
        blockers = []
        if latest_decision["decision"] == "discard_screen":
            blockers.append("Latest LLM candidate underperformed the baseline on the screen window.")
        elif latest_decision["decision"] == "discard_full":
            blockers.append("Latest LLM candidate passed screen but underperformed on the full campaign.")
        elif latest_decision["decision"] == "discard_error":
            blockers.append(f"Latest LLM candidate failed validation or execution: {report['failure_reason']}.")
        elif latest_decision["decision"] == "skip_duplicate":
            blockers.append("Latest LLM candidate was skipped because it duplicated an existing train.py state.")
        if "no_trades_executed" in latest_decision["gate_failures"]:
            blockers.append("Latest LLM candidate produced no trades on the evaluation window.")
        if latest_decision["gate_failures"]:
            blockers.extend(
                f"Latest kept candidate still fails gate: {failure}."
                for failure in latest_decision["gate_failures"]
            )
        if not blockers and not cycle_result.rollout_ready:
            blockers.append("No kept LLM candidate is ready for rollout yet.")

        return ResearchStatusSnapshot(
            mission="Continuously mutate train.py with an LLM against a frozen crypto campaign and keep only score improvements.",
            phase="LLM autoresearch worker active",
            research_rollout_ready=cycle_result.rollout_ready,
            research_blockers=blockers,
            baseline_strategy="Karpathy-style single-file strategy mutation on train.py using frozen historical campaigns.",
            promotion_gate={
                "min_total_return": self.config.target_gate.min_total_return,
                "min_sharpe": self.config.target_gate.min_sharpe,
                "max_drawdown": self.config.target_gate.max_drawdown,
                "min_acceptance_rate": self.config.target_gate.min_acceptance_rate,
            },
            baseline_metrics={
                "total_return": round(float(baseline_metrics.get("total_return", 0.0)), 4),
                "sharpe": round(float(baseline_metrics.get("sharpe", 0.0)), 4),
                "max_drawdown": round(float(baseline_metrics.get("max_drawdown", 0.0)), 4),
                "average_turnover": round(float(baseline_metrics.get("average_turnover", 0.0)), 4),
                "bars_processed": int(baseline_metrics.get("bars_processed", 0)),
                "score": round(float(baseline_report["research_score"]), 4),
            },
            accepted_for_paper=cycle_result.rollout_ready,
            next_milestones=[
                "Keep finding an LLM-generated candidate that survives full campaign evaluation.",
                "Accumulate multiple kept improvements in the research branch.",
                "Promote only candidates that satisfy the hard rollout gates.",
            ],
            loop_state="holding" if cycle_result.rollout_ready else "searching",
            latest_dataset_id=cycle_result.campaign_id,
            latest_cycle_completed_at=cycle_result.completed_at.isoformat(),
            last_processed_bar=cycle_result.latest_closed_bar.isoformat(),
            recent_acceptance_rate=cycle_result.recent_acceptance_rate,
            consecutive_failures=checkpoint.consecutive_failures,
            multi_window_summary={
                "campaign_path": cycle_result.campaign_path,
                "campaign_refreshed_this_cycle": cycle_result.refreshed_campaign,
                "latest_decision": latest_decision,
                "baseline_strategy_name": baseline_report["strategy_name"],
                "baseline_score": baseline_report["research_score"],
                "model_name": report["model_name"],
                "provider_name": report["provider_name"],
            },
            latest_decision=dict(latest_decision),
            latest_candidate_summary={
                "strategy_name": report["strategy_name"],
                "research_score": report["research_score"],
                "ready_for_paper": report["ready_for_paper"],
                "average_metrics": dict(report["average_metrics"]),
                "train_config": dict(report.get("train_config", {})),
                "gate_failures": list(report["gate_failures"]),
                "provider_name": report["provider_name"],
                "model_name": report["model_name"],
                "failure_reason": report["failure_reason"],
            },
            current_best_strategy_name=str(baseline_report["strategy_name"]),
            latest_kept_summary=latest_kept_summary,
            leaderboard=[
                {
                    "strategy_name": str(item["strategy_name"]),
                    "params": dict(item["params"]),
                    "accepted": bool(item["accepted"]),
                    "score": float(item["score"]),
                    "metrics": dict(item["metrics"]),
                    "rejection_reasons": list(item["rejection_reasons"]),
                }
                for item in self._build_leaderboard(self._load_recent_results(limit=20))
            ],
        )

    def _build_failure_snapshot(
        self,
        *,
        checkpoint: LLMWorkerCheckpoint,
        failure_message: str,
        previous_snapshot: ResearchStatusSnapshot | None,
    ) -> ResearchStatusSnapshot:
        baseline_strategy = (
            previous_snapshot.baseline_strategy
            if previous_snapshot is not None
            else "Karpathy-style single-file strategy mutation on train.py using frozen historical campaigns."
        )
        baseline_metrics = (
            dict(previous_snapshot.baseline_metrics)
            if previous_snapshot is not None
            else {
                "total_return": 0.0,
                "sharpe": 0.0,
                "max_drawdown": 0.0,
                "average_turnover": 0.0,
                "bars_processed": 0,
                "score": 0.0,
            }
        )
        latest_dataset_id = (
            previous_snapshot.latest_dataset_id if previous_snapshot is not None else ""
        )
        latest_cycle_completed_at = (
            previous_snapshot.latest_cycle_completed_at
            if previous_snapshot is not None
            else (
                checkpoint.last_cycle_completed_at.isoformat()
                if checkpoint.last_cycle_completed_at is not None
                else None
            )
        )
        last_processed_bar = (
            previous_snapshot.last_processed_bar if previous_snapshot is not None else None
        )
        recent_acceptance_rate = (
            previous_snapshot.recent_acceptance_rate if previous_snapshot is not None else 0.0
        )
        multi_window_summary = (
            dict(previous_snapshot.multi_window_summary)
            if previous_snapshot is not None
            else {}
        )
        leaderboard = list(previous_snapshot.leaderboard) if previous_snapshot is not None else []
        latest_decision = (
            dict(previous_snapshot.latest_decision)
            if previous_snapshot is not None and previous_snapshot.latest_decision is not None
            else None
        )
        latest_candidate_summary = (
            dict(previous_snapshot.latest_candidate_summary)
            if previous_snapshot is not None
            else {}
        )
        current_best_strategy_name = (
            previous_snapshot.current_best_strategy_name if previous_snapshot is not None else None
        )
        latest_kept_summary = (
            dict(previous_snapshot.latest_kept_summary)
            if previous_snapshot is not None
            else {}
        )
        return ResearchStatusSnapshot(
            mission="Continuously mutate train.py with an LLM against a frozen crypto campaign and keep only score improvements.",
            phase="LLM autoresearch worker degraded",
            research_rollout_ready=False,
            research_blockers=[failure_message],
            baseline_strategy=baseline_strategy,
            promotion_gate={
                "min_total_return": self.config.target_gate.min_total_return,
                "min_sharpe": self.config.target_gate.min_sharpe,
                "max_drawdown": self.config.target_gate.max_drawdown,
                "min_acceptance_rate": self.config.target_gate.min_acceptance_rate,
            },
            baseline_metrics=baseline_metrics,
            accepted_for_paper=False,
            next_milestones=[
                "Recover the worker from the latest API, repo, or campaign failure.",
            ],
            loop_state="degraded",
            latest_dataset_id=latest_dataset_id,
            latest_cycle_completed_at=latest_cycle_completed_at,
            last_processed_bar=last_processed_bar,
            recent_acceptance_rate=recent_acceptance_rate,
            consecutive_failures=checkpoint.consecutive_failures,
            multi_window_summary=multi_window_summary,
            leaderboard=leaderboard,
            latest_decision=latest_decision,
            latest_candidate_summary=latest_candidate_summary,
            current_best_strategy_name=current_best_strategy_name,
            latest_kept_summary=latest_kept_summary,
        )

    def _persist_snapshot(
        self,
        snapshot: ResearchStatusSnapshot,
        message: str,
        *,
        suppress_publish_errors: bool = False,
    ) -> None:
        self.state_store.save_snapshot(snapshot)
        if self.publisher is not None:
            try:
                self.publisher.publish_json(
                    "latest_status.json",
                    snapshot.to_dict(),
                    message=message,
                )
                leaderboard = {"leaderboard": list(snapshot.leaderboard)}
                self.publisher.publish_json(
                    "leaderboard.json",
                    leaderboard,
                    message=message,
                )
            except Exception:
                if not suppress_publish_errors:
                    raise

    def _resolved_results_path(self) -> Path:
        worktree_results = (
            Path(self.config.worktrees_root) / slugify(self.config.branch_name) / "results.tsv"
        )
        configured_results = Path(self.config.results_path)
        if worktree_results.exists():
            return worktree_results
        return configured_results

    def _load_recent_results(self, *, limit: int) -> list[dict[str, str]]:
        return load_recent_results(self._resolved_results_path(), limit=limit)

    @staticmethod
    def _build_leaderboard(rows: Sequence[dict[str, str]]) -> list[dict[str, object]]:
        scored_rows = []
        for row in rows:
            try:
                score = float(row.get("research_score", "") or 0.0)
            except ValueError:
                continue
            scored_rows.append(
                {
                    "strategy_name": row.get("mutation_label", "unknown"),
                    "params": {
                        "decision": row.get("decision", ""),
                        "provider_name": row.get("provider_name", ""),
                        "model_name": row.get("model_name", ""),
                        "stage": row.get("stage", ""),
                    },
                    "accepted": row.get("decision") == "keep",
                    "score": score,
                    "metrics": {
                        "total_return": float(row.get("average_total_return", "") or 0.0),
                        "sharpe": float(row.get("average_sharpe", "") or 0.0),
                        "max_drawdown": float(row.get("average_max_drawdown", "") or 0.0),
                        "average_turnover": float(row.get("average_turnover", "") or 0.0),
                    },
                    "rejection_reasons": [
                        row.get("failure_reason", "")
                    ]
                    if row.get("failure_reason")
                    else [],
                }
            )
        return sorted(scored_rows, key=lambda item: item["score"], reverse=True)[:5]

    @staticmethod
    def _acceptance_rate(history: Sequence[CycleSummary]) -> float:
        if not history:
            return 0.0
        accepted = sum(1 for item in history if item.accepted)
        return accepted / len(history)

    @staticmethod
    def _normalize_failure_message(exc: Exception) -> str:
        raw = str(exc).strip()
        if raw == "openai_timeout":
            return "OpenAI request timed out"
        if raw == "github_publish_timeout":
            return "GitHub status publish timed out"
        if not raw:
            return type(exc).__name__
        return raw

    def _failure_cooldown_seconds(self, failure_message: str) -> float:
        if failure_message in {"OpenAI request timed out", "GitHub status publish timed out"}:
            return float(self.config.timeout_failure_cooldown_seconds)
        return float(self.config.failure_cooldown_seconds)

    @staticmethod
    def _ensure_utc(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)


def llm_worker_config_from_env() -> LLMWorkerConfig:
    data_root = os.environ.get("AUTORESEARCH_LLM_DATA_ROOT", "/var/data/autoresearch/llm/data")
    state_root = os.environ.get("AUTORESEARCH_LLM_STATE_ROOT", "/var/data/autoresearch/llm/state")
    artifact_root = os.environ.get("AUTORESEARCH_LLM_ARTIFACT_ROOT", "/var/data/autoresearch/llm/artifacts")
    repo_root = os.environ.get("AUTORESEARCH_LLM_REPO_ROOT", "/var/data/autoresearch/llm/repo")
    campaigns_root = os.environ.get("AUTORESEARCH_LLM_CAMPAIGNS_ROOT", "/var/data/autoresearch/llm/campaigns")
    active_campaign_path = os.environ.get(
        "AUTORESEARCH_LLM_ACTIVE_CAMPAIGN_PATH",
        "/var/data/autoresearch/llm/active_campaign.txt",
    )
    worktrees_root = os.environ.get("AUTORESEARCH_LLM_WORKTREES_ROOT", "/var/data/autoresearch/llm/worktrees")
    results_path = os.environ.get("AUTORESEARCH_LLM_RESULTS_PATH", "/var/data/autoresearch/llm/results.tsv")
    symbols = tuple(
        symbol.strip().upper()
        for symbol in os.environ.get(
            "AUTORESEARCH_SYMBOLS",
            "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT",
        ).split(",")
        if symbol.strip()
    )
    repo_url = os.environ.get("AUTORESEARCH_LLM_REPO_URL") or _discover_repo_url()
    max_cycles_raw = os.environ.get("AUTORESEARCH_LLM_MAX_CYCLES")
    max_cycles = int(max_cycles_raw) if max_cycles_raw else None
    return LLMWorkerConfig(
        repo_url=repo_url,
        symbols=symbols,
        repo_root=repo_root,
        branch_name=os.environ.get("AUTORESEARCH_LLM_BRANCH_NAME", "codex/autoresearch-crypto"),
        strategy_family=os.environ.get("AUTORESEARCH_STRATEGY_FAMILY", "momentum"),
        model_name=os.environ.get("AUTORESEARCH_LLM_MODEL_NAME", "gpt-5-mini"),
        cycle_interval_seconds=int(os.environ.get("AUTORESEARCH_LLM_CYCLE_INTERVAL_SECONDS", "3600")),
        campaign_refresh_interval_seconds=int(
            os.environ.get("AUTORESEARCH_LLM_CAMPAIGN_REFRESH_INTERVAL_SECONDS", "86400")
        ),
        failure_cooldown_seconds=int(
            os.environ.get("AUTORESEARCH_LLM_FAILURE_COOLDOWN_SECONDS", "900")
        ),
        timeout_failure_cooldown_seconds=int(
            os.environ.get("AUTORESEARCH_LLM_TIMEOUT_FAILURE_COOLDOWN_SECONDS", "180")
        ),
        openai_timeout_seconds=float(
            os.environ.get("AUTORESEARCH_LLM_OPENAI_TIMEOUT_SECONDS", "120")
        ),
        openai_max_retries=int(
            os.environ.get("AUTORESEARCH_LLM_OPENAI_MAX_RETRIES", "3")
        ),
        openai_retry_backoff_seconds=float(
            os.environ.get("AUTORESEARCH_LLM_OPENAI_RETRY_BACKOFF_SECONDS", "2.0")
        ),
        max_mutations_per_cycle=int(
            os.environ.get("AUTORESEARCH_LLM_MAX_MUTATIONS_PER_CYCLE", "1")
        ),
        recent_results_limit=int(os.environ.get("AUTORESEARCH_LLM_RECENT_RESULTS_LIMIT", "5")),
        history_window_days=int(os.environ.get("AUTORESEARCH_LLM_HISTORY_WINDOW_DAYS", "7")),
        campaign_window_count=int(os.environ.get("AUTORESEARCH_LLM_CAMPAIGN_WINDOW_COUNT", "1")),
        campaigns_root=campaigns_root,
        active_campaign_path=active_campaign_path,
        worktrees_root=worktrees_root,
        state_root=state_root,
        artifact_root=artifact_root,
        results_path=results_path,
        max_cycles=max_cycles,
        data_config=DataConfig(
            exchange=os.environ.get("AUTORESEARCH_EXCHANGE", "bybit"),
            market=os.environ.get("AUTORESEARCH_MARKET", "linear"),
            timeframe=os.environ.get("AUTORESEARCH_TIMEFRAME", "5m"),
            storage_root=data_root,
            strict_validation=True,
            max_batch_size=int(os.environ.get("AUTORESEARCH_MAX_BATCH_SIZE", "1000")),
            request_timeout_seconds=int(
                os.environ.get("AUTORESEARCH_REQUEST_TIMEOUT_SECONDS", "30")
            ),
        ),
        target_gate=ResearchTargetGate(
            min_total_return=float(
                os.environ.get("AUTORESEARCH_TARGET_MIN_TOTAL_RETURN", "0.0")
            ),
            min_sharpe=float(os.environ.get("AUTORESEARCH_TARGET_MIN_SHARPE", "1.0")),
            max_drawdown=float(os.environ.get("AUTORESEARCH_TARGET_MAX_DRAWDOWN", "0.20")),
            min_acceptance_rate=float(
                os.environ.get("AUTORESEARCH_TARGET_MIN_ACCEPTANCE_RATE", "0.60")
            ),
        ),
    )


def publisher_from_env() -> GitHubStatusPublisher | None:
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    repo = os.environ.get("AUTORESEARCH_STATUS_GITHUB_REPO")
    if not token or not repo:
        return None
    return GitHubStatusPublisher(
        token=token,
        repo=repo,
        branch=os.environ.get("AUTORESEARCH_STATUS_GITHUB_BRANCH", "render-state"),
        base_path=os.environ.get("AUTORESEARCH_STATUS_GITHUB_PATH", "status"),
    )


def _discover_repo_url() -> str:
    completed = subprocess.run(
        ["git", "remote", "get-url", "origin"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def main() -> None:
    config = llm_worker_config_from_env()
    state_store = FilesystemResearchStateStore(config.state_root)
    worker = LLMAutoresearchWorker(
        config=config,
        state_store=state_store,
        publisher=publisher_from_env(),
    )
    print(
        "Starting LLM autoresearch worker for %s on %s with model %s"
        % (",".join(config.symbols), config.data_config.timeframe, config.model_name)
    )
    worker.run_forever()


if __name__ == "__main__":
    main()
