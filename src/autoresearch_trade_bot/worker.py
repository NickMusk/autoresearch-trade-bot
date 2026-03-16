from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from .config import DataConfig, ResearchTargetGate, WorkerConfig
from .data import (
    HistoricalDatasetMaterializer,
    ManifestHistoricalDataSource,
    find_covering_manifest,
    find_latest_reusable_manifest,
)
from .datasets import DatasetSpec, timeframe_to_timedelta
from .experiments import (
    MultiWindowVariantReport,
    StrategyVariant,
    build_rolling_window_specs,
    evaluate_variant_across_windows,
    run_variant_search,
)
from .state import (
    CycleSummary,
    FilesystemResearchStateStore,
    GitHubStatusPublisher,
    LeaderboardEntry,
    ResearchStatusSnapshot,
    WorkerCheckpoint,
)


@dataclass(frozen=True)
class ResearchCycleResult:
    cycle_id: str
    dataset_id: str
    manifest_path: str
    last_processed_bar: datetime
    completed_at: datetime
    best_entry: LeaderboardEntry
    leaderboard: list[LeaderboardEntry]
    multi_window_reports: list[MultiWindowVariantReport]
    ran_multi_window_validation: bool
    recent_acceptance_rate: float
    rollout_ready: bool


class ContinuousResearchWorker:
    def __init__(
        self,
        worker_config: WorkerConfig,
        state_store: FilesystemResearchStateStore,
        materializer: HistoricalDatasetMaterializer | None = None,
        publisher: GitHubStatusPublisher | None = None,
        now_fn: Callable[[], datetime] | None = None,
        sleep_fn: Callable[[float], None] | None = None,
    ) -> None:
        self.worker_config = worker_config
        self.state_store = state_store
        self.materializer = materializer or HistoricalDatasetMaterializer.for_exchange(
            worker_config.data_config
        )
        self.publisher = publisher
        self.now_fn = now_fn or (lambda: datetime.now(timezone.utc))
        self.sleep_fn = sleep_fn or time.sleep

    def run_forever(self) -> None:
        cycles_run = 0
        while self.worker_config.max_cycles is None or cycles_run < self.worker_config.max_cycles:
            now = self.now_fn()
            latest_closed_bar = self.latest_closed_bar(now)
            checkpoint = self.state_store.load_checkpoint()
            if checkpoint.last_processed_bar == latest_closed_bar:
                self.sleep_fn(self.seconds_until_next_cycle(now))
                continue
            try:
                self.run_cycle(now)
                cycles_run += 1
            except Exception as exc:  # noqa: BLE001
                failed_checkpoint = WorkerCheckpoint(
                    last_processed_bar=checkpoint.last_processed_bar,
                    last_multi_window_bar=checkpoint.last_multi_window_bar,
                    consecutive_failures=checkpoint.consecutive_failures + 1,
                    last_cycle_completed_at=checkpoint.last_cycle_completed_at,
                )
                self.state_store.save_checkpoint(failed_checkpoint)
                snapshot = self._build_failure_snapshot(
                    checkpoint=failed_checkpoint,
                    failure_message=str(exc),
                )
                self._persist_status(snapshot, message="Persist failed research status")
                self.sleep_fn(float(self.worker_config.failure_cooldown_seconds))

    def run_cycle(self, now: datetime | None = None) -> ResearchCycleResult:
        cycle_now = self._ensure_utc(now or self.now_fn())
        previous_checkpoint = self.state_store.load_checkpoint()
        latest_closed_bar = self.latest_closed_bar(cycle_now)
        spec = self._build_dataset_spec(latest_closed_bar)
        manifest_path = self._materialize_or_reuse_dataset(spec)
        cycle_result = self._evaluate_cycle(
            spec,
            manifest_path,
            latest_closed_bar,
            cycle_now,
            previous_checkpoint,
        )
        checkpoint = WorkerCheckpoint(
            last_processed_bar=latest_closed_bar,
            last_multi_window_bar=(
                latest_closed_bar
                if cycle_result.ran_multi_window_validation
                else previous_checkpoint.last_multi_window_bar
            ),
            consecutive_failures=0,
            last_cycle_completed_at=cycle_result.completed_at,
        )
        self.state_store.save_checkpoint(checkpoint)
        snapshot = self._build_success_snapshot(cycle_result, checkpoint)
        self._persist_cycle_artifact(cycle_result)
        self._persist_status(
            snapshot,
            leaderboard=cycle_result.leaderboard,
            message=f"Publish research status for {cycle_result.cycle_id}",
        )
        return cycle_result

    def latest_closed_bar(self, now: datetime) -> datetime:
        resolved_now = self._ensure_utc(now)
        step = timeframe_to_timedelta(self.worker_config.timeframe)
        epoch_seconds = int(resolved_now.timestamp())
        step_seconds = int(step.total_seconds())
        floored_seconds = epoch_seconds - (epoch_seconds % step_seconds)
        return datetime.fromtimestamp(floored_seconds, tz=timezone.utc)

    def seconds_until_next_cycle(self, now: datetime) -> float:
        resolved_now = self._ensure_utc(now)
        step = timeframe_to_timedelta(self.worker_config.timeframe)
        next_bar = self.latest_closed_bar(resolved_now) + step
        target = next_bar + timedelta(seconds=self.worker_config.poll_buffer_seconds)
        return max((target - resolved_now).total_seconds(), 1.0)

    def _build_dataset_spec(self, latest_closed_bar: datetime) -> DatasetSpec:
        start = latest_closed_bar - timedelta(days=self.worker_config.history_window_days)
        return DatasetSpec(
            exchange=self.worker_config.data_config.exchange,
            market=self.worker_config.data_config.market,
            timeframe=self.worker_config.timeframe,
            start=start,
            end=latest_closed_bar,
            symbols=self.worker_config.symbols,
        )

    def _materialize_or_reuse_dataset(self, spec: DatasetSpec) -> Path:
        manifest_path = (
            Path(self.worker_config.data_config.storage_root)
            / spec.exchange
            / spec.market
            / spec.timeframe
            / spec.dataset_id
            / "manifest.json"
        )
        if manifest_path.exists():
            return manifest_path
        covering_manifest = find_covering_manifest(
            self.worker_config.data_config.storage_root,
            spec,
            self.materializer.store,
        )
        if covering_manifest is not None:
            return covering_manifest
        reusable_manifest = find_latest_reusable_manifest(
            self.worker_config.data_config.storage_root,
            spec,
            self.materializer.store,
        )
        if reusable_manifest is not None:
            dataset = self.materializer.materialize_incremental(spec, reusable_manifest)
            return dataset.manifest_path
        dataset = self.materializer.materialize(spec)
        return dataset.manifest_path

    def _evaluate_cycle(
        self,
        spec: DatasetSpec,
        manifest_path: Path,
        latest_closed_bar: datetime,
        cycle_now: datetime,
        checkpoint: WorkerCheckpoint,
    ) -> ResearchCycleResult:
        loader = ManifestHistoricalDataSource.from_manifest_path(manifest_path)
        bars = loader.load_bars(spec)
        reports = run_variant_search(
            dataset_spec=spec,
            bars_by_symbol=bars,
            storage_root=str(loader.storage_root),
            max_variants=self.worker_config.max_variants_per_cycle,
        )
        leaderboard = [
            LeaderboardEntry(
                strategy_name=report.variant.name,
                params=report.variant.to_dict(),
                accepted=report.accepted,
                score=report.score,
                metrics=report.metrics,
                rejection_reasons=report.rejection_reasons,
            )
            for report in reports
        ]
        best_entry = leaderboard[0]
        ran_multi_window_validation = self._should_run_multi_window_validation(
            checkpoint=checkpoint,
            latest_closed_bar=latest_closed_bar,
        )
        if ran_multi_window_validation:
            multi_window_reports = self._run_multi_window_validation(
                anchor_spec=spec,
                loader=loader,
                leaderboard=leaderboard,
            )
        else:
            multi_window_reports = self._load_previous_multi_window_reports()
        best_multi_window_report = multi_window_reports[0] if multi_window_reports else None
        history = self.state_store.load_history()
        new_summary = CycleSummary(
            cycle_id=spec.dataset_id,
            dataset_id=spec.dataset_id,
            completed_at=cycle_now,
            last_processed_bar=latest_closed_bar,
            accepted=best_multi_window_report.acceptance_rate >= self.worker_config.target_gate.min_acceptance_rate
            if best_multi_window_report is not None
            else best_entry.accepted,
            score=best_multi_window_report.aggregate_score
            if best_multi_window_report is not None
            else best_entry.score,
            strategy_name=best_multi_window_report.variant.name
            if best_multi_window_report is not None
            else best_entry.strategy_name,
            params=best_multi_window_report.variant.to_dict()
            if best_multi_window_report is not None
            else best_entry.params,
            metrics=best_multi_window_report.average_metrics
            if best_multi_window_report is not None
            else best_entry.metrics,
        )
        recent_history = (history + [new_summary])[-self.worker_config.recent_cycles_for_acceptance :]
        self.state_store.save_history(recent_history)
        recent_acceptance_rate = self._acceptance_rate(recent_history)
        rollout_ready = self._is_rollout_ready(
            best_entry,
            best_multi_window_report,
            recent_acceptance_rate,
        )
        return ResearchCycleResult(
            cycle_id=spec.dataset_id,
            dataset_id=spec.dataset_id,
            manifest_path=str(manifest_path),
            last_processed_bar=latest_closed_bar,
            completed_at=cycle_now,
            best_entry=best_entry,
            leaderboard=leaderboard[:5],
            multi_window_reports=multi_window_reports,
            ran_multi_window_validation=ran_multi_window_validation,
            recent_acceptance_rate=recent_acceptance_rate,
            rollout_ready=rollout_ready,
        )

    def _build_success_snapshot(
        self,
        cycle_result: ResearchCycleResult,
        checkpoint: WorkerCheckpoint,
    ) -> ResearchStatusSnapshot:
        target_gate = self.worker_config.target_gate
        blockers = []
        best_multi_window_report = (
            cycle_result.multi_window_reports[0] if cycle_result.multi_window_reports else None
        )
        if not cycle_result.best_entry.accepted:
            blockers.append("No current strategy variant passes the promotion gate on the latest window.")
        if cycle_result.recent_acceptance_rate < target_gate.min_acceptance_rate:
            blockers.append("Recent acceptance rate is still below the rollout threshold.")
        if best_multi_window_report is not None:
            if best_multi_window_report.average_metrics["sharpe"] < target_gate.min_sharpe:
                blockers.append("Sharpe is below the rollout target.")
            if best_multi_window_report.worst_max_drawdown > target_gate.max_drawdown:
                blockers.append("Worst drawdown across windows is above the rollout target.")
            if best_multi_window_report.average_metrics["total_return"] < target_gate.min_total_return:
                blockers.append("Total return is below the rollout target.")
        else:
            if cycle_result.best_entry.metrics["sharpe"] < target_gate.min_sharpe:
                blockers.append("Sharpe is below the rollout target.")
            if cycle_result.best_entry.metrics["max_drawdown"] > target_gate.max_drawdown:
                blockers.append("Max drawdown is above the rollout target.")
            if cycle_result.best_entry.metrics["total_return"] < target_gate.min_total_return:
                blockers.append("Total return is below the rollout target.")

        return ResearchStatusSnapshot(
            mission="Continuously search, test, and rank crypto strategies before paper or live rollout.",
            phase="Continuous research worker active",
            research_rollout_ready=cycle_result.rollout_ready,
            research_blockers=blockers,
            baseline_strategy="Market-neutral cross-sectional momentum variant search on liquid perpetuals.",
            promotion_gate={
                "min_total_return": target_gate.min_total_return,
                "min_sharpe": target_gate.min_sharpe,
                "max_drawdown": target_gate.max_drawdown,
                "min_acceptance_rate": target_gate.min_acceptance_rate,
            },
            baseline_metrics={
                "total_return": round(float(cycle_result.best_entry.metrics["total_return"]), 4),
                "sharpe": round(float(cycle_result.best_entry.metrics["sharpe"]), 4),
                "max_drawdown": round(float(cycle_result.best_entry.metrics["max_drawdown"]), 4),
                "average_turnover": round(float(cycle_result.best_entry.metrics["average_turnover"]), 4),
                "bars_processed": int(cycle_result.best_entry.metrics["bars_processed"]),
                "score": round(float(cycle_result.best_entry.score), 4),
            },
            accepted_for_paper=cycle_result.best_entry.accepted,
            next_milestones=[
                "Promote only the variants that stay above target gate across recent windows.",
                "Add realtime paper or shadow replay on the same engine boundaries.",
                "Increase candidate breadth only after a stable multi-window winner appears.",
            ],
            loop_state="holding" if cycle_result.rollout_ready else "searching",
            latest_dataset_id=cycle_result.dataset_id,
            latest_cycle_completed_at=cycle_result.completed_at.isoformat(),
            last_processed_bar=cycle_result.last_processed_bar.isoformat(),
            recent_acceptance_rate=round(cycle_result.recent_acceptance_rate, 4),
            consecutive_failures=checkpoint.consecutive_failures,
            multi_window_summary=self._build_multi_window_summary(
                cycle_result.multi_window_reports,
                checkpoint.last_multi_window_bar,
                cycle_result.ran_multi_window_validation,
            ),
            leaderboard=[entry.to_dict() for entry in cycle_result.leaderboard],
        )

    def _build_failure_snapshot(
        self,
        checkpoint: WorkerCheckpoint,
        failure_message: str,
    ) -> ResearchStatusSnapshot:
        latest_snapshot = self.state_store.load_snapshot()
        baseline_metrics = latest_snapshot.baseline_metrics if latest_snapshot else {}
        leaderboard = latest_snapshot.leaderboard if latest_snapshot else []
        return ResearchStatusSnapshot(
            mission="Continuously search, test, and rank crypto strategies before paper or live rollout.",
            phase="Continuous research worker degraded",
            research_rollout_ready=False,
            research_blockers=[f"Worker cycle failed: {failure_message}"],
            baseline_strategy="Market-neutral cross-sectional momentum variant search on liquid perpetuals.",
            promotion_gate={
                "min_total_return": self.worker_config.target_gate.min_total_return,
                "min_sharpe": self.worker_config.target_gate.min_sharpe,
                "max_drawdown": self.worker_config.target_gate.max_drawdown,
                "min_acceptance_rate": self.worker_config.target_gate.min_acceptance_rate,
            },
            baseline_metrics=baseline_metrics,
            accepted_for_paper=False,
            next_milestones=[
                "Recover the worker and resume the next closed-bar cycle.",
                "Preserve the last good leaderboard while the failure is investigated.",
            ],
            loop_state="degraded",
            latest_dataset_id=latest_snapshot.latest_dataset_id if latest_snapshot else "",
            latest_cycle_completed_at=(
                checkpoint.last_cycle_completed_at.isoformat()
                if checkpoint.last_cycle_completed_at is not None
                else None
            ),
            last_processed_bar=(
                checkpoint.last_processed_bar.isoformat()
                if checkpoint.last_processed_bar is not None
                else None
            ),
            recent_acceptance_rate=(
                latest_snapshot.recent_acceptance_rate if latest_snapshot else 0.0
            ),
            consecutive_failures=checkpoint.consecutive_failures,
            multi_window_summary=(
                latest_snapshot.multi_window_summary if latest_snapshot else {}
            ),
            leaderboard=leaderboard,
        )

    def _persist_cycle_artifact(self, cycle_result: ResearchCycleResult) -> None:
        artifact_dir = Path(self.worker_config.artifact_root) / "research-cycles"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        path = artifact_dir / f"{cycle_result.cycle_id}.json"
        payload = {
            "cycle_id": cycle_result.cycle_id,
            "dataset_id": cycle_result.dataset_id,
            "manifest_path": cycle_result.manifest_path,
            "completed_at": cycle_result.completed_at.isoformat(),
            "last_processed_bar": cycle_result.last_processed_bar.isoformat(),
            "recent_acceptance_rate": cycle_result.recent_acceptance_rate,
            "rollout_ready": cycle_result.rollout_ready,
            "ran_multi_window_validation": cycle_result.ran_multi_window_validation,
            "multi_window_reports": [
                report.to_dict() for report in cycle_result.multi_window_reports
            ],
            "leaderboard": [entry.to_dict() for entry in cycle_result.leaderboard],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _persist_status(
        self,
        snapshot: ResearchStatusSnapshot,
        leaderboard: list[LeaderboardEntry] | None = None,
        message: str = "Publish research status",
    ) -> None:
        self.state_store.save_snapshot(snapshot)
        if leaderboard is not None:
            self.state_store.save_leaderboard(leaderboard)
        if self.publisher is not None:
            self.publisher.publish_json(
                "latest_status.json",
                snapshot.to_dict(),
                message=message,
            )
            if leaderboard is not None:
                self.publisher.publish_json(
                    "leaderboard.json",
                    {"leaderboard": [entry.to_dict() for entry in leaderboard]},
                    message=message,
                )

    def _is_rollout_ready(
        self,
        entry: LeaderboardEntry,
        multi_window_report: MultiWindowVariantReport | None,
        recent_acceptance_rate: float,
    ) -> bool:
        gate = self.worker_config.target_gate
        if multi_window_report is not None:
            return (
                entry.accepted
                and multi_window_report.average_metrics["total_return"] >= gate.min_total_return
                and multi_window_report.average_metrics["sharpe"] >= gate.min_sharpe
                and multi_window_report.worst_max_drawdown <= gate.max_drawdown
                and multi_window_report.acceptance_rate >= gate.min_acceptance_rate
                and recent_acceptance_rate >= gate.min_acceptance_rate
            )
        return (
            entry.accepted
            and entry.metrics["total_return"] >= gate.min_total_return
            and entry.metrics["sharpe"] >= gate.min_sharpe
            and entry.metrics["max_drawdown"] <= gate.max_drawdown
            and recent_acceptance_rate >= gate.min_acceptance_rate
        )

    def _run_multi_window_validation(
        self,
        anchor_spec: DatasetSpec,
        loader: ManifestHistoricalDataSource,
        leaderboard: list[LeaderboardEntry],
    ) -> list[MultiWindowVariantReport]:
        candidate_entries = leaderboard[: self.worker_config.multi_window_top_candidates]
        if not candidate_entries:
            return []

        window_specs = build_rolling_window_specs(
            anchor_spec=anchor_spec,
            window_count=self.worker_config.multi_window_count,
        )
        window_bars = {}
        for spec in window_specs:
            manifest_path = self._materialize_or_reuse_dataset(spec)
            window_loader = (
                loader
                if manifest_path == Path(loader.manifest_path)
                else ManifestHistoricalDataSource.from_manifest_path(manifest_path)
            )
            window_bars[spec.dataset_id] = window_loader.load_bars(spec)

        reports = []
        for entry in candidate_entries:
            reports.append(
                evaluate_variant_across_windows(
                    variant=self._variant_from_entry(entry),
                    window_specs=window_specs,
                    bars_by_dataset_id=window_bars,
                    storage_root=str(loader.storage_root),
                )
            )
        return sorted(reports, key=lambda item: item.aggregate_score, reverse=True)

    @staticmethod
    def _variant_from_entry(entry: LeaderboardEntry):
        params = entry.params
        return StrategyVariant(
            name=str(params["name"]),
            lookback_bars=int(params["lookback_bars"]),
            top_k=int(params["top_k"]),
            gross_target=float(params["gross_target"]),
        )

    def _build_multi_window_summary(
        self,
        reports: list[MultiWindowVariantReport],
        last_validated_bar: datetime | None,
        ran_this_cycle: bool,
    ) -> dict[str, object]:
        if not reports:
            return {}
        best = reports[0]
        payload = {
            "best_variant": best.variant.to_dict(),
            "aggregate_score": round(best.aggregate_score, 4),
            "acceptance_rate": round(best.acceptance_rate, 4),
            "average_metrics": {
                key: round(float(value), 4) if isinstance(value, float) else value
                for key, value in best.average_metrics.items()
            },
            "worst_max_drawdown": round(best.worst_max_drawdown, 4),
            "windows_passed": best.windows_passed,
            "total_windows": best.total_windows,
            "validation_interval_bars": self.worker_config.multi_window_validation_interval_bars,
            "ran_this_cycle": ran_this_cycle,
            "reports": [report.to_dict() for report in reports],
        }
        if last_validated_bar is not None:
            payload["last_validated_bar"] = last_validated_bar.isoformat()
        return payload

    def _should_run_multi_window_validation(
        self,
        checkpoint: WorkerCheckpoint,
        latest_closed_bar: datetime,
    ) -> bool:
        if checkpoint.last_multi_window_bar is None:
            return True
        step = timeframe_to_timedelta(self.worker_config.timeframe)
        elapsed = latest_closed_bar - checkpoint.last_multi_window_bar
        elapsed_bars = int(elapsed.total_seconds() // step.total_seconds())
        return elapsed_bars >= self.worker_config.multi_window_validation_interval_bars

    def _load_previous_multi_window_reports(self) -> list[MultiWindowVariantReport]:
        snapshot = self.state_store.load_snapshot()
        if snapshot is None:
            return []
        reports = snapshot.multi_window_summary.get("reports", [])
        return [MultiWindowVariantReport.from_dict(item) for item in reports]

    @staticmethod
    def _acceptance_rate(history: list[CycleSummary]) -> float:
        if not history:
            return 0.0
        accepted = sum(1 for item in history if item.accepted)
        return accepted / len(history)

    @staticmethod
    def _ensure_utc(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)


def worker_config_from_env() -> WorkerConfig:
    timeframe = os.environ.get("AUTORESEARCH_TIMEFRAME", "5m")
    data_root = os.environ.get("AUTORESEARCH_DATA_ROOT", "data")
    state_root = os.environ.get("AUTORESEARCH_STATE_ROOT", "state")
    artifact_root = os.environ.get("AUTORESEARCH_ARTIFACT_ROOT", "artifacts")
    symbols = tuple(
        symbol.strip().upper()
        for symbol in os.environ.get(
            "AUTORESEARCH_SYMBOLS",
            "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT",
        ).split(",")
        if symbol.strip()
    )
    max_cycles_raw = os.environ.get("AUTORESEARCH_MAX_CYCLES")
    max_cycles = int(max_cycles_raw) if max_cycles_raw else None
    return WorkerConfig(
        symbols=symbols,
        timeframe=timeframe,
        history_window_days=int(os.environ.get("AUTORESEARCH_HISTORY_WINDOW_DAYS", "7")),
        max_variants_per_cycle=int(os.environ.get("AUTORESEARCH_MAX_VARIANTS", "12")),
        multi_window_count=int(os.environ.get("AUTORESEARCH_MULTI_WINDOW_COUNT", "4")),
        multi_window_top_candidates=int(
            os.environ.get("AUTORESEARCH_MULTI_WINDOW_TOP_CANDIDATES", "3")
        ),
        multi_window_validation_interval_bars=int(
            os.environ.get("AUTORESEARCH_MULTI_WINDOW_INTERVAL_BARS", "12")
        ),
        recent_cycles_for_acceptance=int(
            os.environ.get("AUTORESEARCH_RECENT_CYCLES_FOR_ACCEPTANCE", "12")
        ),
        poll_buffer_seconds=int(os.environ.get("AUTORESEARCH_POLL_BUFFER_SECONDS", "15")),
        failure_cooldown_seconds=int(
            os.environ.get("AUTORESEARCH_FAILURE_COOLDOWN_SECONDS", "90")
        ),
        state_root=state_root,
        artifact_root=artifact_root,
        max_cycles=max_cycles,
        data_config=DataConfig(
            exchange=os.environ.get("AUTORESEARCH_EXCHANGE", "binance"),
            market=os.environ.get("AUTORESEARCH_MARKET", "usdm_futures"),
            timeframe=timeframe,
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
            max_drawdown=float(
                os.environ.get("AUTORESEARCH_TARGET_MAX_DRAWDOWN", "0.20")
            ),
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


def main() -> None:
    worker_config = worker_config_from_env()
    state_store = FilesystemResearchStateStore(worker_config.state_root)
    worker = ContinuousResearchWorker(
        worker_config=worker_config,
        state_store=state_store,
        publisher=publisher_from_env(),
    )
    print(
        "Starting continuous research worker for %s on %s"
        % (",".join(worker_config.symbols), worker_config.timeframe)
    )
    worker.run_forever()


if __name__ == "__main__":
    main()
