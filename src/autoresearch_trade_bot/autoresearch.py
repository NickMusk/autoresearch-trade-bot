from __future__ import annotations

import csv
import hashlib
import importlib.util
import inspect
import json
import os
import re
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .config import DataConfig, ResearchTargetGate
from .data import (
    HistoricalDatasetMaterializer,
    ManifestHistoricalDataSource,
    ensure_dataset_manifest,
)
from .datasets import DatasetSpec, ensure_utc
from .experiments import WindowEvaluationReport, build_baseline_experiment_config, build_rolling_window_specs
from .research import ResearchEvaluator, compute_research_score
from .strategy_families import (
    FAMILY_MOMENTUM,
    deterministic_mutation_specs,
    normalize_train_config as normalize_family_train_config,
    render_train_file as render_family_train_file,
)
from .strategy import Strategy


RESULTS_TSV_COLUMNS = (
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
)

DEFAULT_CAMPAIGNS_ROOT = ".autoresearch/campaigns"
DEFAULT_ACTIVE_CAMPAIGN = ".autoresearch/active_campaign.txt"
DEFAULT_WORKTREE_ROOT = ".autoresearch/worktrees"
DEFAULT_GIT_USER_NAME = "Autoresearch Bot"
DEFAULT_GIT_USER_EMAIL = "autoresearch-bot@local"


@dataclass(frozen=True)
class FrozenResearchWindow:
    dataset_id: str
    manifest_path: str
    start: str
    end: str

    def to_dict(self) -> dict[str, str]:
        return {
            "dataset_id": self.dataset_id,
            "manifest_path": self.manifest_path,
            "start": self.start,
            "end": self.end,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, str]) -> "FrozenResearchWindow":
        return cls(
            dataset_id=str(payload["dataset_id"]),
            manifest_path=str(payload["manifest_path"]),
            start=str(payload["start"]),
            end=str(payload["end"]),
        )


@dataclass(frozen=True)
class AutoresearchCampaign:
    campaign_id: str
    name: str
    exchange: str
    market: str
    timeframe: str
    symbols: tuple[str, ...]
    storage_root: str
    target_gate: ResearchTargetGate
    windows: tuple[FrozenResearchWindow, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "name": self.name,
            "exchange": self.exchange,
            "market": self.market,
            "timeframe": self.timeframe,
            "symbols": list(self.symbols),
            "storage_root": self.storage_root,
            "target_gate": {
                "min_total_return": self.target_gate.min_total_return,
                "min_sharpe": self.target_gate.min_sharpe,
                "max_drawdown": self.target_gate.max_drawdown,
                "min_acceptance_rate": self.target_gate.min_acceptance_rate,
            },
            "windows": [window.to_dict() for window in self.windows],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AutoresearchCampaign":
        gate = payload["target_gate"]
        return cls(
            campaign_id=str(payload.get("campaign_id", payload["name"])),
            name=str(payload["name"]),
            exchange=str(payload["exchange"]),
            market=str(payload["market"]),
            timeframe=str(payload["timeframe"]),
            symbols=tuple(str(symbol) for symbol in payload["symbols"]),
            storage_root=str(payload["storage_root"]),
            target_gate=ResearchTargetGate(
                min_total_return=float(gate["min_total_return"]),
                min_sharpe=float(gate["min_sharpe"]),
                max_drawdown=float(gate["max_drawdown"]),
                min_acceptance_rate=float(gate["min_acceptance_rate"]),
            ),
            windows=tuple(
                FrozenResearchWindow.from_dict(item) for item in payload["windows"]
            ),
        )


@dataclass(frozen=True)
class AutoresearchRunReport:
    run_id: str
    recorded_at: str
    campaign_id: str
    campaign_name: str
    strategy_name: str
    git_branch: str
    git_commit: str
    parent_commit: str
    train_file: str
    train_sha1: str
    stage: str
    mutation_label: str
    baseline_score: float | None
    delta_score: float | None
    research_score: float
    acceptance_rate: float
    average_metrics: dict[str, float]
    worst_max_drawdown: float
    ready_for_paper: bool
    gate_failures: list[str]
    windows_passed: int
    total_windows: int
    window_reports: list[WindowEvaluationReport]
    runtime_seconds: float
    artifact_path: str
    provider_name: str = "manual"
    model_name: str = ""
    prompt_id: str = ""
    candidate_sha1: str = ""
    train_config: dict[str, Any] = field(default_factory=dict)
    failure_reason: str = ""
    proposal_artifact_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "recorded_at": self.recorded_at,
            "campaign_id": self.campaign_id,
            "campaign_name": self.campaign_name,
            "strategy_name": self.strategy_name,
            "git_branch": self.git_branch,
            "git_commit": self.git_commit,
            "parent_commit": self.parent_commit,
            "train_file": self.train_file,
            "train_sha1": self.train_sha1,
            "stage": self.stage,
            "mutation_label": self.mutation_label,
            "provider_name": self.provider_name,
            "model_name": self.model_name,
            "prompt_id": self.prompt_id,
            "baseline_score": self.baseline_score,
            "delta_score": self.delta_score,
            "candidate_sha1": self.candidate_sha1,
            "train_config": dict(self.train_config),
            "research_score": self.research_score,
            "acceptance_rate": self.acceptance_rate,
            "average_metrics": dict(self.average_metrics),
            "worst_max_drawdown": self.worst_max_drawdown,
            "ready_for_paper": self.ready_for_paper,
            "gate_failures": list(self.gate_failures),
            "windows_passed": self.windows_passed,
            "total_windows": self.total_windows,
            "window_reports": [report.to_dict() for report in self.window_reports],
            "runtime_seconds": self.runtime_seconds,
            "failure_reason": self.failure_reason,
            "proposal_artifact_path": self.proposal_artifact_path,
            "artifact_path": self.artifact_path,
        }


@dataclass(frozen=True)
class GitAutoresearchDecision:
    decision: str
    stage: str
    mutation_label: str
    baseline_score: float
    candidate_score: float
    kept_commit: str | None
    baseline_report: AutoresearchRunReport
    report: AutoresearchRunReport


@dataclass(frozen=True)
class DeterministicMutationProposal:
    label: str
    config_updates: dict[str, Any]
    commit_message: str


@dataclass(frozen=True)
class MutationProposal:
    label: str
    candidate_text: str
    commit_message: str
    provider_name: str = "manual"
    model_name: str = ""
    prompt_id: str = ""
    notes: str = ""
    proposal_artifact_path: str = ""
    candidate_sha1: str = ""


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip()).strip("-").lower()
    return slug or "campaign"


def _report_can_be_promoted(report: AutoresearchRunReport) -> bool:
    failures = set(report.gate_failures)
    if "no_trades_executed" in failures:
        return False
    if "drawdown_above_gate" in failures:
        return False
    if report.acceptance_rate < 0.10:
        return False
    if {"total_return_below_gate", "sharpe_below_gate"}.issubset(failures):
        return False
    average_metrics = report.average_metrics
    if float(average_metrics.get("total_return", 0.0)) <= -0.05:
        return False
    if float(average_metrics.get("sharpe", 0.0)) <= -0.5:
        return False
    if float(average_metrics.get("max_drawdown", 0.0)) > 0.35:
        return False
    return True


def diff_train_configs(
    baseline_config: Mapping[str, Any],
    candidate_config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    all_keys = sorted(set(baseline_config) | set(candidate_config))
    for key in all_keys:
        baseline_value = baseline_config.get(key)
        candidate_value = candidate_config.get(key)
        if baseline_value != candidate_value:
            changes.append(
                {
                    "key": key,
                    "before": baseline_value,
                    "after": candidate_value,
                }
            )
    return changes


def _sha1_text(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def campaign_path_for_name(
    campaign_name: str,
    campaigns_root: str | Path = DEFAULT_CAMPAIGNS_ROOT,
) -> Path:
    return Path(campaigns_root) / f"{slugify(campaign_name)}.json"


def active_campaign_path(
    path: str | Path = DEFAULT_ACTIVE_CAMPAIGN,
) -> Path:
    return Path(path)


def set_active_campaign(
    campaign_path: str | Path,
    pointer_path: str | Path = DEFAULT_ACTIVE_CAMPAIGN,
) -> Path:
    pointer = active_campaign_path(pointer_path)
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(str(Path(campaign_path).resolve()), encoding="utf-8")
    return pointer


def resolve_campaign_path(
    campaign_path: str | Path | None = None,
    *,
    campaign_name: str | None = None,
    campaigns_root: str | Path = DEFAULT_CAMPAIGNS_ROOT,
    pointer_path: str | Path = DEFAULT_ACTIVE_CAMPAIGN,
) -> Path:
    if campaign_path is not None:
        return Path(campaign_path)
    if campaign_name is not None:
        return campaign_path_for_name(campaign_name, campaigns_root=campaigns_root)
    pointer = active_campaign_path(pointer_path)
    if pointer.exists():
        return Path(pointer.read_text(encoding="utf-8").strip())
    raise FileNotFoundError("no active autoresearch campaign configured")


def build_campaign_specs(
    *,
    exchange: str,
    market: str,
    timeframe: str,
    symbols: Sequence[str],
    anchor_end: datetime,
    window_days: int,
    window_count: int,
) -> list[DatasetSpec]:
    anchor_end_utc = ensure_utc(anchor_end)
    anchor_spec = DatasetSpec(
        exchange=exchange,
        market=market,
        timeframe=timeframe,
        start=anchor_end_utc - timedelta(days=window_days),
        end=anchor_end_utc,
        symbols=tuple(symbols),
    )
    return build_rolling_window_specs(anchor_spec, window_count)


def prepare_campaign(
    *,
    campaign_name: str,
    exchange: str,
    market: str,
    timeframe: str,
    symbols: Sequence[str],
    anchor_end: datetime,
    window_days: int,
    window_count: int,
    storage_root: str = "data",
    campaign_path: str | Path | None = None,
    campaigns_root: str | Path = DEFAULT_CAMPAIGNS_ROOT,
    set_active: bool = True,
    active_pointer_path: str | Path = DEFAULT_ACTIVE_CAMPAIGN,
    target_gate: ResearchTargetGate | None = None,
    local_data_only: bool = False,
    history_readiness_state_path: str | Path | None = None,
    include_open_interest: bool = True,
    materializer: HistoricalDatasetMaterializer | None = None,
) -> Path:
    target = target_gate or ResearchTargetGate()
    data_config = DataConfig(
        exchange=exchange,
        market=market,
        timeframe=timeframe,
        storage_root=storage_root,
        default_start=ensure_utc(anchor_end) - timedelta(days=window_days),
        default_end=ensure_utc(anchor_end),
        local_only=local_data_only,
        include_open_interest=include_open_interest,
        history_readiness_state_path=(
            str(history_readiness_state_path) if history_readiness_state_path is not None else None
        ),
    )
    resolved_materializer = materializer or HistoricalDatasetMaterializer.for_exchange(data_config)
    windows = []
    for spec in build_campaign_specs(
        exchange=exchange,
        market=market,
        timeframe=timeframe,
        symbols=symbols,
        anchor_end=anchor_end,
        window_days=window_days,
        window_count=window_count,
    ):
        dataset_manifest_path = ensure_dataset_manifest(
            storage_root=storage_root,
            spec=spec,
            materializer=resolved_materializer,
            local_only=local_data_only,
            readiness_state_path=history_readiness_state_path,
        )
        windows.append(
            FrozenResearchWindow(
                dataset_id=spec.dataset_id,
                manifest_path=str(dataset_manifest_path),
                start=spec.start.isoformat(),
                end=spec.end.isoformat(),
            )
        )

    campaign = AutoresearchCampaign(
        campaign_id=slugify(campaign_name),
        name=campaign_name,
        exchange=exchange,
        market=market,
        timeframe=timeframe,
        symbols=tuple(symbols),
        storage_root=storage_root,
        target_gate=target,
        windows=tuple(windows),
    )
    path = resolve_campaign_path(
        campaign_path,
        campaign_name=campaign_name if campaign_path is None else None,
        campaigns_root=campaigns_root,
        pointer_path=active_pointer_path,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(campaign.to_dict(), indent=2), encoding="utf-8")
    if set_active:
        set_active_campaign(path, pointer_path=active_pointer_path)
    return path


def load_campaign(
    campaign_path: str | Path | None = None,
    *,
    campaign_name: str | None = None,
    campaigns_root: str | Path = DEFAULT_CAMPAIGNS_ROOT,
    pointer_path: str | Path = DEFAULT_ACTIVE_CAMPAIGN,
) -> AutoresearchCampaign:
    path = resolve_campaign_path(
        campaign_path,
        campaign_name=campaign_name,
        campaigns_root=campaigns_root,
        pointer_path=pointer_path,
    )
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return AutoresearchCampaign.from_dict(payload)


def evaluate_train_file(
    *,
    campaign_path: str | Path | None = None,
    campaign_name: str | None = None,
    train_path: str | Path = "train.py",
    artifact_root: str | Path = ".autoresearch/runs",
    stage: str = "full",
    mutation_label: str = "manual",
    provider_name: str = "manual",
    model_name: str = "",
    prompt_id: str = "",
    baseline_score: float | None = None,
    parent_commit: str = "",
    candidate_sha1: str = "",
    proposal_artifact_path: str = "",
    window_limit: int | None = None,
    recorded_at: datetime | None = None,
) -> AutoresearchRunReport:
    started_at = time.perf_counter()
    campaign = load_campaign(campaign_path, campaign_name=campaign_name)
    train_file = Path(train_path)
    train_file_resolved = train_file.resolve()
    module = _load_train_module(train_file)
    strategy_builder = _load_strategy_builder(module)
    strategy_name = str(getattr(module, "STRATEGY_NAME", train_file.stem))
    strategy_family = str(getattr(module, "STRATEGY_FAMILY", FAMILY_MOMENTUM))
    train_config = normalize_train_config(
        getattr(module, "TRAIN_CONFIG", {}),
        strategy_family=strategy_family,
    )
    train_sha1 = _git(["hash-object", str(train_file_resolved)], cwd=train_file_resolved.parent)
    git_branch = _safe_git(["branch", "--show-current"], cwd=train_file_resolved.parent)
    git_commit = _safe_git(["rev-parse", "HEAD"], cwd=train_file_resolved.parent)
    timestamp = ensure_utc(recorded_at or datetime.now(timezone.utc))

    window_reports = []
    selected_windows = campaign.windows[:window_limit] if window_limit is not None else campaign.windows
    for window in selected_windows:
        loader = ManifestHistoricalDataSource.from_manifest_path(window.manifest_path)
        manifest = loader.manifest
        spec = DatasetSpec(
            exchange=manifest.exchange,
            market=manifest.market,
            timeframe=manifest.timeframe,
            start=manifest.start,
            end=manifest.end,
            symbols=manifest.symbols,
        )
        bars = loader.load_bars(spec)
        evaluator = ResearchEvaluator(
            build_baseline_experiment_config(spec, storage_root=str(loader.storage_root))
        )
        strategy = _build_strategy(strategy_builder, spec)
        result = evaluator.evaluate(bars, strategy)
        window_reports.append(
            WindowEvaluationReport(
                dataset_id=spec.dataset_id,
                window_start=spec.start.isoformat(),
                window_end=spec.end.isoformat(),
                accepted=result.accepted,
                score=result.score,
                metrics={
                    "total_return": result.metrics.total_return,
                    "sharpe": result.metrics.sharpe,
                    "max_drawdown": result.metrics.max_drawdown,
                    "average_turnover": result.metrics.average_turnover,
                    "bars_processed": result.metrics.bars_processed,
                    "nonzero_turnover_steps": result.metrics.nonzero_turnover_steps,
                },
                rejection_reasons=list(result.rejection_reasons),
            )
        )

    aggregated = _aggregate_reports(window_reports)
    gate_failures = _gate_failures(campaign.target_gate, aggregated)
    ready_for_paper = not gate_failures
    run_id = uuid.uuid4().hex[:12]
    artifact_path = Path(artifact_root) / f"{run_id}.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    report = AutoresearchRunReport(
        run_id=run_id,
        recorded_at=timestamp.isoformat(),
        campaign_id=campaign.campaign_id,
        campaign_name=campaign.name,
        strategy_name=strategy_name,
        git_branch=git_branch,
        git_commit=git_commit,
        parent_commit=parent_commit,
        train_file=str(train_file),
        train_sha1=train_sha1,
        stage=stage,
        mutation_label=mutation_label,
        provider_name=provider_name,
        model_name=model_name,
        prompt_id=prompt_id,
        baseline_score=baseline_score,
        delta_score=(
            aggregated["research_score"] - baseline_score
            if baseline_score is not None
            else None
        ),
        candidate_sha1=candidate_sha1,
        train_config=train_config,
        research_score=aggregated["research_score"],
        acceptance_rate=aggregated["acceptance_rate"],
        average_metrics=aggregated["average_metrics"],
        worst_max_drawdown=aggregated["worst_max_drawdown"],
        ready_for_paper=ready_for_paper,
        gate_failures=gate_failures,
        windows_passed=aggregated["windows_passed"],
        total_windows=aggregated["total_windows"],
        window_reports=window_reports,
        runtime_seconds=round(time.perf_counter() - started_at, 6),
        proposal_artifact_path=proposal_artifact_path,
        artifact_path=str(artifact_path),
    )
    artifact_path.write_text(
        json.dumps(report.to_dict(), indent=2),
        encoding="utf-8",
    )
    return report


def append_results_row(
    *,
    results_path: str | Path,
    report: AutoresearchRunReport,
    decision: str,
) -> Path:
    path = Path(results_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "recorded_at": report.recorded_at,
        "campaign_id": report.campaign_id,
        "decision": decision,
        "stage": report.stage,
        "mutation_label": report.mutation_label,
        "provider_name": report.provider_name,
        "model_name": report.model_name,
        "prompt_id": report.prompt_id,
        "run_id": report.run_id,
        "campaign_name": report.campaign_name,
        "strategy_name": report.strategy_name,
        "git_branch": report.git_branch,
        "git_commit": report.git_commit,
        "parent_commit": report.parent_commit,
        "train_sha1": report.train_sha1,
        "candidate_sha1": report.candidate_sha1,
        "train_config_json": json.dumps(report.train_config, sort_keys=True),
        "baseline_score": (
            f"{report.baseline_score:.6f}" if report.baseline_score is not None else ""
        ),
        "delta_score": (
            f"{report.delta_score:.6f}" if report.delta_score is not None else ""
        ),
        "research_score": f"{report.research_score:.6f}",
        "acceptance_rate": f"{report.acceptance_rate:.6f}",
        "average_total_return": f"{report.average_metrics['total_return']:.6f}",
        "average_sharpe": f"{report.average_metrics['sharpe']:.6f}",
        "average_max_drawdown": f"{report.average_metrics['max_drawdown']:.6f}",
        "worst_max_drawdown": f"{report.worst_max_drawdown:.6f}",
        "average_turnover": f"{report.average_metrics['average_turnover']:.6f}",
        "ready_for_paper": str(report.ready_for_paper).lower(),
        "gate_failures_json": json.dumps(report.gate_failures, sort_keys=True),
        "runtime_seconds": f"{report.runtime_seconds:.6f}",
        "failure_reason": report.failure_reason,
        "proposal_artifact_path": report.proposal_artifact_path,
        "artifact_path": report.artifact_path,
    }
    should_write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULTS_TSV_COLUMNS, delimiter="\t")
        if should_write_header:
            writer.writeheader()
        writer.writerow(row)
    return path


class GitAutoresearchRunner:
    def __init__(
        self,
        *,
        repo_root: str | Path,
        branch_name: str,
        evaluator: Callable[..., AutoresearchRunReport] = evaluate_train_file,
        train_relpath: str = "train.py",
        results_relpath: str = "results.tsv",
        artifact_relpath: str = ".autoresearch/runs",
        worktrees_root: str | Path = DEFAULT_WORKTREE_ROOT,
        screen_window_count: int = 1,
    ) -> None:
        self.repo_root = Path(repo_root)
        self.branch_name = branch_name
        self.evaluator = evaluator
        self.train_relpath = train_relpath
        self.results_relpath = results_relpath
        self.artifact_relpath = artifact_relpath
        self.worktrees_root = Path(worktrees_root)
        self.screen_window_count = screen_window_count
        self._baseline_cache: dict[tuple[str, str, str, int | None], AutoresearchRunReport] = {}
        self._seen_candidate_sha1s: set[str] | None = None

    @property
    def worktree_root(self) -> Path:
        return self.worktrees_root / slugify(self.branch_name)

    @property
    def train_path(self) -> Path:
        return self.worktree_root / self.train_relpath

    @property
    def results_path(self) -> Path:
        return self.worktree_root / self.results_relpath

    @property
    def artifact_root(self) -> Path:
        return self.worktree_root / self.artifact_relpath

    def ensure_worktree(self) -> None:
        self.worktree_root.parent.mkdir(parents=True, exist_ok=True)
        ensure_git_identity(self.repo_root)
        if self.worktree_root.exists():
            return
        branch_exists = self._git(["branch", "--list", self.branch_name]).strip()
        if branch_exists:
            self._git(["worktree", "add", str(self.worktree_root), self.branch_name])
        else:
            self._git(["worktree", "add", "-b", self.branch_name, str(self.worktree_root), "HEAD"])

    def evaluate_current(self, *, campaign_path: str | Path) -> AutoresearchRunReport:
        self.ensure_worktree()
        return self.evaluator(
            campaign_path=campaign_path,
            train_path=self.train_path,
            artifact_root=self.artifact_root,
            mutation_label="baseline",
        )

    def apply_candidate(
        self,
        *,
        campaign_path: str | Path,
        candidate_text: str,
        commit_message: str,
        mutation_label: str = "candidate",
    ) -> GitAutoresearchDecision:
        self.ensure_worktree()
        baseline_report = self.evaluate_current(campaign_path=campaign_path)
        original_text = self.train_path.read_text(encoding="utf-8")
        self.train_path.write_text(candidate_text, encoding="utf-8")
        try:
            candidate_report = self.evaluator(
                campaign_path=campaign_path,
                train_path=self.train_path,
                artifact_root=self.artifact_root,
                mutation_label=mutation_label,
                parent_commit=baseline_report.git_commit,
                baseline_score=baseline_report.research_score,
            )
            if (
                candidate_report.research_score > baseline_report.research_score
                and _report_can_be_promoted(candidate_report)
            ):
                self._git_in_worktree(["add", self.train_relpath])
                self._git_in_worktree(["commit", "-m", commit_message])
                kept_commit = self._git_in_worktree(["rev-parse", "HEAD"]).strip()
                candidate_report = replace(candidate_report, git_commit=kept_commit)
                decision = "keep"
            else:
                self.train_path.write_text(original_text, encoding="utf-8")
                kept_commit = None
                decision = "discard"
            append_results_row(
                results_path=self.results_path,
                report=candidate_report,
                decision=decision,
            )
            return GitAutoresearchDecision(
                decision=decision,
                stage=candidate_report.stage,
                mutation_label=mutation_label,
                baseline_score=baseline_report.research_score,
                candidate_score=candidate_report.research_score,
                kept_commit=kept_commit,
                baseline_report=baseline_report,
                report=candidate_report,
            )
        except Exception:
            self.train_path.write_text(original_text, encoding="utf-8")
            raise

    def apply_candidate_staged(
        self,
        *,
        campaign_path: str | Path,
        candidate_text: str,
        commit_message: str,
        mutation_label: str,
    ) -> GitAutoresearchDecision:
        proposal = MutationProposal(
            label=mutation_label,
            candidate_text=candidate_text,
            commit_message=commit_message,
        )
        return self.apply_mutation_proposal_staged(
            campaign_path=campaign_path,
            proposal=proposal,
        )

    def apply_mutation_proposal_staged(
        self,
        *,
        campaign_path: str | Path,
        proposal: MutationProposal,
        validator: Callable[[str], tuple[bool, str]] | None = None,
    ) -> GitAutoresearchDecision:
        self.ensure_worktree()
        baseline_screen = self._evaluate_baseline_cached(
            campaign_path=campaign_path,
            stage="screen",
            window_limit=self.screen_window_count,
        )
        baseline_full = self._evaluate_baseline_cached(
            campaign_path=campaign_path,
            stage="full",
            window_limit=None,
        )
        candidate_sha1 = proposal.candidate_sha1 or _sha1_text(proposal.candidate_text)
        if candidate_sha1 in self._load_seen_candidate_sha1s():
            duplicate_report = self._build_failure_report(
                campaign_path=campaign_path,
                stage="validation",
                mutation_label=proposal.label,
                provider_name=proposal.provider_name,
                model_name=proposal.model_name,
                prompt_id=proposal.prompt_id,
                baseline_score=baseline_full.research_score,
                parent_commit=baseline_full.git_commit,
                candidate_sha1=candidate_sha1,
                failure_reason="duplicate_candidate",
                proposal_artifact_path=proposal.proposal_artifact_path,
            )
            append_results_row(
                results_path=self.results_path,
                report=duplicate_report,
                decision="skip_duplicate",
            )
            return GitAutoresearchDecision(
                decision="skip_duplicate",
                stage=duplicate_report.stage,
                mutation_label=proposal.label,
                baseline_score=baseline_full.research_score,
                candidate_score=duplicate_report.research_score,
                kept_commit=None,
                baseline_report=baseline_full,
                report=duplicate_report,
            )

        original_text = self.train_path.read_text(encoding="utf-8")
        if validator is not None:
            is_valid, failure_reason = validator(proposal.candidate_text)
            if not is_valid:
                invalid_report = self._build_failure_report(
                    campaign_path=campaign_path,
                    stage="validation",
                    mutation_label=proposal.label,
                    provider_name=proposal.provider_name,
                    model_name=proposal.model_name,
                    prompt_id=proposal.prompt_id,
                    baseline_score=baseline_full.research_score,
                    parent_commit=baseline_full.git_commit,
                    candidate_sha1=candidate_sha1,
                    failure_reason=failure_reason,
                    proposal_artifact_path=proposal.proposal_artifact_path,
                )
                append_results_row(
                    results_path=self.results_path,
                    report=invalid_report,
                    decision="discard_error",
                )
                return GitAutoresearchDecision(
                    decision="discard_error",
                    stage=invalid_report.stage,
                    mutation_label=proposal.label,
                    baseline_score=baseline_full.research_score,
                    candidate_score=invalid_report.research_score,
                    kept_commit=None,
                    baseline_report=baseline_full,
                    report=invalid_report,
                )
        self.train_path.write_text(proposal.candidate_text, encoding="utf-8")
        try:
            screen_report = self.evaluator(
                campaign_path=campaign_path,
                train_path=self.train_path,
                artifact_root=self.artifact_root,
                stage="screen",
                mutation_label=proposal.label,
                provider_name=proposal.provider_name,
                model_name=proposal.model_name,
                prompt_id=proposal.prompt_id,
                baseline_score=baseline_screen.research_score,
                parent_commit=baseline_full.git_commit,
                candidate_sha1=candidate_sha1,
                proposal_artifact_path=proposal.proposal_artifact_path,
                window_limit=self.screen_window_count,
            )
            if screen_report.research_score <= baseline_screen.research_score:
                self.train_path.write_text(original_text, encoding="utf-8")
                append_results_row(
                    results_path=self.results_path,
                    report=screen_report,
                    decision="discard_screen",
                )
                return GitAutoresearchDecision(
                    decision="discard_screen",
                    stage="screen",
                    mutation_label=proposal.label,
                    baseline_score=baseline_screen.research_score,
                    candidate_score=screen_report.research_score,
                    kept_commit=None,
                    baseline_report=baseline_full,
                    report=screen_report,
                )

            full_report = self.evaluator(
                campaign_path=campaign_path,
                train_path=self.train_path,
                artifact_root=self.artifact_root,
                stage="full",
                mutation_label=proposal.label,
                provider_name=proposal.provider_name,
                model_name=proposal.model_name,
                prompt_id=proposal.prompt_id,
                baseline_score=baseline_full.research_score,
                parent_commit=baseline_full.git_commit,
                candidate_sha1=candidate_sha1,
                proposal_artifact_path=proposal.proposal_artifact_path,
            )
            if (
                full_report.research_score > baseline_full.research_score
                and _report_can_be_promoted(full_report)
            ):
                self._git_in_worktree(["add", self.train_relpath])
                self._git_in_worktree(["commit", "-m", proposal.commit_message])
                kept_commit = self._git_in_worktree(["rev-parse", "HEAD"]).strip()
                full_report = replace(full_report, git_commit=kept_commit)
                decision = "keep"
                self._remember_candidate_sha1(candidate_sha1)
            else:
                self.train_path.write_text(original_text, encoding="utf-8")
                kept_commit = None
                decision = "discard_full"
            append_results_row(
                results_path=self.results_path,
                report=screen_report,
                decision="screen_pass",
            )
            append_results_row(
                results_path=self.results_path,
                report=full_report,
                decision=decision,
            )
            return GitAutoresearchDecision(
                decision=decision,
                stage="full",
                mutation_label=proposal.label,
                baseline_score=baseline_full.research_score,
                candidate_score=full_report.research_score,
                kept_commit=kept_commit,
                baseline_report=baseline_full,
                report=full_report,
            )
        except Exception as exc:
            self.train_path.write_text(original_text, encoding="utf-8")
            failed_report = self._build_failure_report(
                campaign_path=campaign_path,
                stage="validation",
                mutation_label=proposal.label,
                provider_name=proposal.provider_name,
                model_name=proposal.model_name,
                prompt_id=proposal.prompt_id,
                baseline_score=baseline_full.research_score,
                parent_commit=baseline_full.git_commit,
                candidate_sha1=candidate_sha1,
                failure_reason=format_failure_reason(exc),
                proposal_artifact_path=proposal.proposal_artifact_path,
            )
            append_results_row(
                results_path=self.results_path,
                report=failed_report,
                decision="discard_error",
            )
            return GitAutoresearchDecision(
                decision="discard_error",
                stage=failed_report.stage,
                mutation_label=proposal.label,
                baseline_score=baseline_full.research_score,
                candidate_score=failed_report.research_score,
                kept_commit=None,
                baseline_report=baseline_full,
                report=failed_report,
            )

    def remove_worktree(self) -> None:
        if self.worktree_root.exists():
            self._git(["worktree", "remove", str(self.worktree_root)])

    def _git(self, args: Sequence[str]) -> str:
        return _git(args, cwd=self.repo_root)

    def _git_in_worktree(self, args: Sequence[str]) -> str:
        return _git(args, cwd=self.worktree_root)

    def _evaluate_baseline_cached(
        self,
        *,
        campaign_path: str | Path,
        stage: str,
        window_limit: int | None,
    ) -> AutoresearchRunReport:
        head_commit = _safe_git(["rev-parse", "HEAD"], cwd=self.worktree_root)
        cache_key = (str(Path(campaign_path)), head_commit, stage, window_limit)
        cached = self._baseline_cache.get(cache_key)
        if cached is not None:
            return cached
        report = self.evaluator(
            campaign_path=campaign_path,
            train_path=self.train_path,
            artifact_root=self.artifact_root,
            stage=stage,
            mutation_label="baseline",
            provider_name="baseline",
            model_name="",
            prompt_id="",
            candidate_sha1=_sha1_text(self.train_path.read_text(encoding="utf-8")),
            proposal_artifact_path="",
            window_limit=window_limit,
        )
        self._baseline_cache[cache_key] = report
        return report

    def _build_failure_report(
        self,
        *,
        campaign_path: str | Path,
        stage: str,
        mutation_label: str,
        provider_name: str,
        model_name: str,
        prompt_id: str,
        baseline_score: float,
        parent_commit: str,
        candidate_sha1: str,
        failure_reason: str,
        proposal_artifact_path: str,
    ) -> AutoresearchRunReport:
        campaign = load_campaign(campaign_path)
        run_id = uuid.uuid4().hex[:12]
        artifact_path = self.artifact_root / f"{run_id}.json"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        report = AutoresearchRunReport(
            run_id=run_id,
            recorded_at=datetime.now(timezone.utc).isoformat(),
            campaign_id=campaign.campaign_id,
            campaign_name=campaign.name,
            strategy_name="invalid-candidate",
            git_branch=_safe_git(["branch", "--show-current"], cwd=self.worktree_root),
            git_commit=_safe_git(["rev-parse", "HEAD"], cwd=self.worktree_root),
            parent_commit=parent_commit,
            train_file=str(self.train_path),
            train_sha1="",
            stage=stage,
            mutation_label=mutation_label,
            provider_name=provider_name,
            model_name=model_name,
            prompt_id=prompt_id,
            baseline_score=baseline_score,
            delta_score=None,
            candidate_sha1=candidate_sha1,
            train_config={},
            research_score=float("-inf"),
            acceptance_rate=0.0,
            average_metrics={
                "total_return": 0.0,
                "sharpe": 0.0,
                "max_drawdown": 1.0,
                "average_turnover": 0.0,
                "bars_processed": 0,
                "nonzero_turnover_steps": 0,
            },
            worst_max_drawdown=1.0,
            ready_for_paper=False,
            gate_failures=[failure_reason],
            windows_passed=0,
            total_windows=len(campaign.windows),
            window_reports=[],
            runtime_seconds=0.0,
            failure_reason=failure_reason,
            proposal_artifact_path=proposal_artifact_path,
            artifact_path=str(artifact_path),
        )
        artifact_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        return report

    def _load_seen_candidate_sha1s(self) -> set[str]:
        if self._seen_candidate_sha1s is not None:
            return self._seen_candidate_sha1s
        seen: set[str] = set()
        if self.results_path.exists():
            with self.results_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle, delimiter="\t")
                for row in reader:
                    candidate_sha1 = (row.get("candidate_sha1") or "").strip()
                    if candidate_sha1:
                        seen.add(candidate_sha1)
        self._seen_candidate_sha1s = seen
        return seen

    def _remember_candidate_sha1(self, candidate_sha1: str) -> None:
        self._load_seen_candidate_sha1s().add(candidate_sha1)


class DeterministicTrainMutator:
    def generate(self, train_path: str | Path) -> list[DeterministicMutationProposal]:
        module = _load_train_module(Path(train_path))
        if not hasattr(module, "TRAIN_CONFIG"):
            raise AttributeError("train.py must define TRAIN_CONFIG for deterministic mutations")
        strategy_family = str(getattr(module, "STRATEGY_FAMILY", FAMILY_MOMENTUM))
        config = normalize_train_config(
            dict(getattr(module, "TRAIN_CONFIG")),
            strategy_family=strategy_family,
        )
        proposals: list[DeterministicMutationProposal] = []
        for item in deterministic_mutation_specs(strategy_family, config):
            label = str(item["label"])
            config_updates = dict(item["config_updates"])
            proposals.append(
                DeterministicMutationProposal(
                    label=label,
                    config_updates=config_updates,
                    commit_message=f"Mutate train.py: {label}",
                )
            )
        return proposals


def normalize_train_config(
    train_config: Mapping[str, Any],
    *,
    strategy_family: str = FAMILY_MOMENTUM,
) -> dict[str, Any]:
    return normalize_family_train_config(train_config, strategy_family=strategy_family)


def render_train_file(
    train_config: Mapping[str, Any],
    *,
    strategy_family: str = FAMILY_MOMENTUM,
) -> str:
    return render_family_train_file(train_config, strategy_family=strategy_family)


def run_deterministic_mutation_campaign(
    *,
    campaign_path: str | Path,
    repo_root: str | Path,
    branch_name: str,
    max_mutations: int = 8,
    worktrees_root: str | Path = DEFAULT_WORKTREE_ROOT,
) -> list[GitAutoresearchDecision]:
    runner = GitAutoresearchRunner(
        repo_root=repo_root,
        branch_name=branch_name,
        worktrees_root=worktrees_root,
    )
    runner.ensure_worktree()
    mutator = DeterministicTrainMutator()
    decisions = []
    for proposal in mutator.generate(runner.train_path)[:max_mutations]:
        module = _load_train_module(runner.train_path)
        strategy_family = str(getattr(module, "STRATEGY_FAMILY", FAMILY_MOMENTUM))
        config = dict(getattr(module, "TRAIN_CONFIG"))
        config.update(proposal.config_updates)
        candidate_text = render_train_file(config, strategy_family=strategy_family)
        decisions.append(
            runner.apply_candidate_staged(
                campaign_path=campaign_path,
                candidate_text=candidate_text,
                commit_message=proposal.commit_message,
                mutation_label=proposal.label,
            )
        )
    return decisions


def _aggregate_reports(window_reports: Sequence[WindowEvaluationReport]) -> dict[str, Any]:
    total_windows = len(window_reports)
    if total_windows == 0:
        raise ValueError("window_reports must not be empty")
    windows_passed = sum(1 for report in window_reports if report.accepted)
    return {
        "research_score": sum(report.score for report in window_reports) / total_windows,
        "acceptance_rate": windows_passed / total_windows,
        "average_metrics": {
            "total_return": sum(float(report.metrics["total_return"]) for report in window_reports)
            / total_windows,
            "sharpe": sum(float(report.metrics["sharpe"]) for report in window_reports)
            / total_windows,
            "max_drawdown": sum(float(report.metrics["max_drawdown"]) for report in window_reports)
            / total_windows,
            "average_turnover": sum(
                float(report.metrics["average_turnover"]) for report in window_reports
            )
            / total_windows,
            "bars_processed": int(
                round(
                    sum(int(report.metrics["bars_processed"]) for report in window_reports)
                    / total_windows
                )
            ),
            "nonzero_turnover_steps": int(
                round(
                    sum(int(report.metrics.get("nonzero_turnover_steps", 0)) for report in window_reports)
                    / total_windows
                )
            ),
        },
        "worst_max_drawdown": max(
            float(report.metrics["max_drawdown"]) for report in window_reports
        ),
        "windows_passed": windows_passed,
        "total_windows": total_windows,
    }


def _gate_failures(
    gate: ResearchTargetGate,
    aggregated: Mapping[str, Any],
) -> list[str]:
    failures = []
    metrics = aggregated["average_metrics"]
    if int(metrics.get("nonzero_turnover_steps", 0)) == 0:
        failures.append("no_trades_executed")
    if metrics["total_return"] < gate.min_total_return:
        failures.append("total_return_below_gate")
    if metrics["sharpe"] < gate.min_sharpe:
        failures.append("sharpe_below_gate")
    if aggregated["worst_max_drawdown"] > gate.max_drawdown:
        failures.append("drawdown_above_gate")
    if aggregated["acceptance_rate"] < gate.min_acceptance_rate:
        failures.append("acceptance_rate_below_gate")
    return failures


def _load_train_module(train_path: Path):
    spec = importlib.util.spec_from_file_location(
        f"autoresearch_train_{uuid.uuid4().hex}",
        train_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load train file: {train_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_strategy_builder(module) -> Callable[..., Strategy]:
    if not hasattr(module, "build_strategy"):
        raise AttributeError("train.py must define build_strategy")
    builder = getattr(module, "build_strategy")
    if not callable(builder):
        raise TypeError("build_strategy must be callable")
    return builder


def _build_strategy(
    builder: Callable[..., Strategy],
    dataset_spec: DatasetSpec,
) -> Strategy:
    signature = inspect.signature(builder)
    if len(signature.parameters) == 0:
        return builder()
    return builder(dataset_spec)


def ensure_git_identity(repo_root: str | Path) -> None:
    root = Path(repo_root)
    user_name = _safe_git(["config", "--local", "--get", "user.name"], cwd=root)
    user_email = _safe_git(["config", "--local", "--get", "user.email"], cwd=root)
    desired_name = os.environ.get("AUTORESEARCH_GIT_USER_NAME", DEFAULT_GIT_USER_NAME)
    desired_email = os.environ.get("AUTORESEARCH_GIT_USER_EMAIL", DEFAULT_GIT_USER_EMAIL)
    if not user_name:
        _git(["config", "--local", "user.name", desired_name], cwd=root)
    if not user_email:
        _git(["config", "--local", "user.email", desired_email], cwd=root)


def format_failure_reason(exc: Exception) -> str:
    if isinstance(exc, subprocess.CalledProcessError):
        command = (
            " ".join(str(part) for part in exc.cmd)
            if isinstance(exc.cmd, Sequence) and not isinstance(exc.cmd, (str, bytes))
            else str(exc.cmd)
        )
        stderr = (exc.stderr or exc.stdout or "").strip().replace("\n", " ")
        if stderr:
            stderr = re.sub(r"\s+", " ", stderr)[:240]
            return f"candidate_error:CalledProcessError:cmd={command}:rc={exc.returncode}:stderr={stderr}"
        return f"candidate_error:CalledProcessError:cmd={command}:rc={exc.returncode}"
    return f"candidate_error:{type(exc).__name__}"


def _git(args: Sequence[str], cwd: Path) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _safe_git(args: Sequence[str], cwd: Path) -> str:
    try:
        return _git(args, cwd)
    except subprocess.CalledProcessError:
        return ""
