from __future__ import annotations

import csv
import importlib.util
import inspect
import json
import subprocess
import time
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .config import DataConfig, ResearchTargetGate
from .data import HistoricalDatasetMaterializer, ManifestHistoricalDataSource
from .datasets import DatasetSpec, ensure_utc
from .experiments import WindowEvaluationReport, build_baseline_experiment_config, build_rolling_window_specs
from .research import ResearchEvaluator
from .strategy import Strategy


RESULTS_TSV_COLUMNS = (
    "recorded_at",
    "decision",
    "run_id",
    "campaign_name",
    "strategy_name",
    "git_branch",
    "git_commit",
    "train_sha1",
    "research_score",
    "acceptance_rate",
    "average_total_return",
    "average_sharpe",
    "average_max_drawdown",
    "worst_max_drawdown",
    "average_turnover",
    "ready_for_paper",
    "artifact_path",
)


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
    campaign_name: str
    strategy_name: str
    git_branch: str
    git_commit: str
    train_file: str
    train_sha1: str
    research_score: float
    acceptance_rate: float
    average_metrics: dict[str, float]
    worst_max_drawdown: float
    ready_for_paper: bool
    gate_failures: list[str]
    windows_passed: int
    total_windows: int
    window_reports: list[WindowEvaluationReport]
    artifact_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "recorded_at": self.recorded_at,
            "campaign_name": self.campaign_name,
            "strategy_name": self.strategy_name,
            "git_branch": self.git_branch,
            "git_commit": self.git_commit,
            "train_file": self.train_file,
            "train_sha1": self.train_sha1,
            "research_score": self.research_score,
            "acceptance_rate": self.acceptance_rate,
            "average_metrics": dict(self.average_metrics),
            "worst_max_drawdown": self.worst_max_drawdown,
            "ready_for_paper": self.ready_for_paper,
            "gate_failures": list(self.gate_failures),
            "windows_passed": self.windows_passed,
            "total_windows": self.total_windows,
            "window_reports": [report.to_dict() for report in self.window_reports],
            "artifact_path": self.artifact_path,
        }


@dataclass(frozen=True)
class GitAutoresearchDecision:
    decision: str
    baseline_score: float
    candidate_score: float
    kept_commit: str | None
    report: AutoresearchRunReport


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
    campaign_path: str | Path = ".autoresearch/campaign.json",
    target_gate: ResearchTargetGate | None = None,
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
        dataset = resolved_materializer.materialize(spec)
        windows.append(
            FrozenResearchWindow(
                dataset_id=spec.dataset_id,
                manifest_path=str(dataset.manifest_path),
                start=spec.start.isoformat(),
                end=spec.end.isoformat(),
            )
        )

    campaign = AutoresearchCampaign(
        name=campaign_name,
        exchange=exchange,
        market=market,
        timeframe=timeframe,
        symbols=tuple(symbols),
        storage_root=storage_root,
        target_gate=target,
        windows=tuple(windows),
    )
    path = Path(campaign_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(campaign.to_dict(), indent=2), encoding="utf-8")
    return path


def load_campaign(campaign_path: str | Path) -> AutoresearchCampaign:
    payload = json.loads(Path(campaign_path).read_text(encoding="utf-8"))
    return AutoresearchCampaign.from_dict(payload)


def evaluate_train_file(
    *,
    campaign_path: str | Path,
    train_path: str | Path = "train.py",
    artifact_root: str | Path = ".autoresearch/runs",
    recorded_at: datetime | None = None,
) -> AutoresearchRunReport:
    started_at = time.perf_counter()
    campaign = load_campaign(campaign_path)
    train_file = Path(train_path)
    module = _load_train_module(train_file)
    strategy_builder = _load_strategy_builder(module)
    strategy_name = str(getattr(module, "STRATEGY_NAME", train_file.stem))
    train_sha1 = _git(["hash-object", str(train_file)], cwd=train_file.parent.resolve())
    git_branch = _safe_git(["branch", "--show-current"], cwd=train_file.parent.resolve())
    git_commit = _safe_git(["rev-parse", "HEAD"], cwd=train_file.parent.resolve())
    timestamp = ensure_utc(recorded_at or datetime.now(timezone.utc))

    window_reports = []
    for window in campaign.windows:
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
        campaign_name=campaign.name,
        strategy_name=strategy_name,
        git_branch=git_branch,
        git_commit=git_commit,
        train_file=str(train_file),
        train_sha1=train_sha1,
        research_score=aggregated["research_score"],
        acceptance_rate=aggregated["acceptance_rate"],
        average_metrics=aggregated["average_metrics"],
        worst_max_drawdown=aggregated["worst_max_drawdown"],
        ready_for_paper=ready_for_paper,
        gate_failures=gate_failures,
        windows_passed=aggregated["windows_passed"],
        total_windows=aggregated["total_windows"],
        window_reports=window_reports,
        artifact_path=str(artifact_path),
    )
    artifact_path.write_text(
        json.dumps(
            {
                **report.to_dict(),
                "runtime_seconds": round(time.perf_counter() - started_at, 6),
            },
            indent=2,
        ),
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
        "decision": decision,
        "run_id": report.run_id,
        "campaign_name": report.campaign_name,
        "strategy_name": report.strategy_name,
        "git_branch": report.git_branch,
        "git_commit": report.git_commit,
        "train_sha1": report.train_sha1,
        "research_score": f"{report.research_score:.6f}",
        "acceptance_rate": f"{report.acceptance_rate:.6f}",
        "average_total_return": f"{report.average_metrics['total_return']:.6f}",
        "average_sharpe": f"{report.average_metrics['sharpe']:.6f}",
        "average_max_drawdown": f"{report.average_metrics['max_drawdown']:.6f}",
        "worst_max_drawdown": f"{report.worst_max_drawdown:.6f}",
        "average_turnover": f"{report.average_metrics['average_turnover']:.6f}",
        "ready_for_paper": str(report.ready_for_paper).lower(),
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
    ) -> None:
        self.repo_root = Path(repo_root)
        self.branch_name = branch_name
        self.evaluator = evaluator
        self.train_relpath = train_relpath
        self.results_relpath = results_relpath
        self.artifact_relpath = artifact_relpath

    @property
    def train_path(self) -> Path:
        return self.repo_root / self.train_relpath

    @property
    def results_path(self) -> Path:
        return self.repo_root / self.results_relpath

    @property
    def artifact_root(self) -> Path:
        return self.repo_root / self.artifact_relpath

    def ensure_branch(self) -> None:
        branch_exists = self._git(["branch", "--list", self.branch_name]).strip()
        if branch_exists:
            self._git(["checkout", self.branch_name])
        else:
            self._git(["checkout", "-b", self.branch_name])

    def evaluate_current(self, *, campaign_path: str | Path) -> AutoresearchRunReport:
        return self.evaluator(
            campaign_path=campaign_path,
            train_path=self.train_path,
            artifact_root=self.artifact_root,
        )

    def apply_candidate(
        self,
        *,
        campaign_path: str | Path,
        candidate_text: str,
        commit_message: str,
    ) -> GitAutoresearchDecision:
        self.ensure_branch()
        baseline_report = self.evaluate_current(campaign_path=campaign_path)
        original_text = self.train_path.read_text(encoding="utf-8")
        self.train_path.write_text(candidate_text, encoding="utf-8")
        try:
            candidate_report = self.evaluator(
                campaign_path=campaign_path,
                train_path=self.train_path,
                artifact_root=self.artifact_root,
            )
            if candidate_report.research_score > baseline_report.research_score:
                self._git(["add", self.train_relpath])
                self._git(["commit", "-m", commit_message])
                kept_commit = self._git(["rev-parse", "HEAD"]).strip()
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
                baseline_score=baseline_report.research_score,
                candidate_score=candidate_report.research_score,
                kept_commit=kept_commit,
                report=candidate_report,
            )
        except Exception:
            self.train_path.write_text(original_text, encoding="utf-8")
            raise

    def _git(self, args: Sequence[str]) -> str:
        return _git(args, cwd=self.repo_root)


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
