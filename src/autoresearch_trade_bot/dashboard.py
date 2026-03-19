from __future__ import annotations

from dataclasses import asdict
import os
from pathlib import Path
from .config import ExperimentConfig, PromotionGate, RiskLimits
from .experiments import discover_latest_manifest, run_baseline_from_manifest_path
from .research import ResearchEvaluator
from .sample_data import build_demo_dataset
from .state import (
    ResearchStatusSnapshot,
    load_status_snapshot_from_path,
    load_status_snapshot_from_url,
)
from .strategy import CrossSectionalMomentumStrategy

DashboardSnapshot = ResearchStatusSnapshot
WAVE1_FAMILY_IDS = ("mean_reversion", "ema_trend", "volatility_breakout")


def default_experiment_config() -> ExperimentConfig:
    return ExperimentConfig(
        name="baseline-dashboard-demo",
        universe=("BTCUSDT", "ETHUSDT", "SOLUSDT"),
        fee_rate=0.0,
        bars_per_year=12,
        risk_limits=RiskLimits(
            max_gross_leverage=1.0,
            max_symbol_weight=0.5,
            max_turnover_per_step=1.0,
        ),
        promotion_gate=PromotionGate(
            min_total_return=0.01,
            min_sharpe=0.50,
            max_drawdown=0.10,
            max_average_turnover=1.0,
        ),
    )


def build_dashboard_snapshot(storage_root: str | None = None) -> DashboardSnapshot:
    status_path = os.environ.get("AUTORESEARCH_STATUS_PATH")
    if status_path:
        persisted_snapshot = load_status_snapshot_from_path(status_path)
        if persisted_snapshot is not None:
            return persisted_snapshot

    status_url = os.environ.get("AUTORESEARCH_STATUS_URL")
    if status_url:
        remote_snapshot = load_status_snapshot_from_url(status_url)
        if remote_snapshot is not None:
            return remote_snapshot

    resolved_storage_root = storage_root or os.environ.get("AUTORESEARCH_DATA_ROOT", "data")
    latest_manifest = discover_latest_manifest(resolved_storage_root)
    if latest_manifest is not None:
        real_snapshot = _build_snapshot_from_local_manifest(latest_manifest)
        if real_snapshot is not None:
            return real_snapshot

    return _build_demo_snapshot()


def build_family_dashboard_data(snapshot: DashboardSnapshot) -> dict[str, object]:
    family_snapshots = load_family_snapshots()
    families = []
    for family_id in WAVE1_FAMILY_IDS:
        family_snapshot = family_snapshots.get(family_id)
        families.append(
            {
                "family_id": family_id,
                "label": family_id.replace("_", " ").title(),
                "snapshot": family_snapshot.to_dict() if family_snapshot is not None else None,
            }
        )
    return {
        "primary_snapshot": snapshot.to_dict(),
        "family_tabs": families,
    }


def load_family_snapshots() -> dict[str, ResearchStatusSnapshot]:
    snapshots: dict[str, ResearchStatusSnapshot] = {}
    for family_id in WAVE1_FAMILY_IDS:
        snapshot = None
        status_url = _family_status_url(family_id)
        if status_url:
            snapshot = load_status_snapshot_from_url(status_url, timeout_seconds=5)
        if snapshot is None:
            snapshot_path = _family_status_path(family_id)
            if snapshot_path is not None:
                snapshot = load_status_snapshot_from_path(snapshot_path)
        if snapshot is not None:
            snapshots[family_id] = snapshot
    return snapshots


def _family_status_url(family_id: str) -> str | None:
    explicit_key = f"AUTORESEARCH_FAMILY_STATUS_URL_{family_id.upper()}"
    explicit_url = os.environ.get(explicit_key)
    if explicit_url:
        return explicit_url
    template = os.environ.get("AUTORESEARCH_FAMILY_STATUS_URL_TEMPLATE", "").strip()
    if not template:
        return None
    return template.format(family_id=family_id)


def _family_status_path(family_id: str) -> Path | None:
    root = os.environ.get("AUTORESEARCH_FAMILY_REPOS_ROOT")
    if not root:
        return None
    return Path(root) / family_id / ".autoresearch" / "runs" / "latest_status.json"


def _build_demo_snapshot() -> DashboardSnapshot:
    config = default_experiment_config()
    strategy = CrossSectionalMomentumStrategy(lookback_bars=2, top_k=1, gross_target=1.0)
    evaluator = ResearchEvaluator(config)
    result = evaluator.evaluate(build_demo_dataset(config.universe), strategy)

    return DashboardSnapshot(
        mission="Find, test, and rank crypto strategies before paper or live rollout.",
        phase="Render-deployed research kernel",
        research_rollout_ready=False,
        research_blockers=[
            "Realtime paper or shadow market data stream is not implemented yet.",
            "Experiment persistence and walk-forward validation are not implemented yet.",
            "No validated Binance dataset has been materialized into local storage yet.",
        ],
        baseline_strategy="Market-neutral cross-sectional momentum on liquid perpetuals.",
        promotion_gate=asdict(config.promotion_gate),
        baseline_metrics={
            "total_return": round(result.metrics.total_return, 4),
            "sharpe": round(result.metrics.sharpe, 4),
            "max_drawdown": round(result.metrics.max_drawdown, 4),
            "average_turnover": round(result.metrics.average_turnover, 4),
            "bars_processed": result.metrics.bars_processed,
            "score": round(result.score, 4),
        },
        accepted_for_paper=result.accepted,
        next_milestones=[
            "Materialize the first Binance historical dataset and pin a manifest.",
            "Add walk-forward evaluation and experiment persistence.",
            "Add realtime paper or shadow replay using the same engine boundaries.",
        ],
        loop_state="idle",
        latest_dataset_id="",
        latest_cycle_completed_at=None,
        last_processed_bar=None,
        recent_acceptance_rate=0.0,
        consecutive_failures=0,
        multi_window_summary={},
        leaderboard=[],
    )


def _build_snapshot_from_local_manifest(manifest_path: Path) -> DashboardSnapshot | None:
    try:
        report = run_baseline_from_manifest_path(manifest_path)
    except RuntimeError:
        return None

    return DashboardSnapshot(
        mission="Find, test, and rank crypto strategies before paper or live rollout.",
        phase="Local validated Binance dataset available",
        research_rollout_ready=False,
        research_blockers=[
            "Realtime paper or shadow market data stream is not implemented yet.",
            "Experiment persistence and walk-forward validation are not implemented yet.",
        ],
        baseline_strategy="Market-neutral cross-sectional momentum on liquid perpetuals.",
        promotion_gate=asdict(default_experiment_config().promotion_gate),
        baseline_metrics={
            "total_return": round(float(report.metrics["total_return"]), 4),
            "sharpe": round(float(report.metrics["sharpe"]), 4),
            "max_drawdown": round(float(report.metrics["max_drawdown"]), 4),
            "average_turnover": round(float(report.metrics["average_turnover"]), 4),
            "bars_processed": int(report.metrics["bars_processed"]),
            "score": round(report.score, 4),
        },
        accepted_for_paper=report.accepted,
        next_milestones=[
            "Persist this baseline report and compare it across walk-forward windows.",
            "Add realtime paper or shadow replay using the same engine boundaries.",
            "Expand the universe after validating data quality and run stability.",
        ],
        loop_state="manual",
        latest_dataset_id=report.dataset_id,
        latest_cycle_completed_at=None,
        last_processed_bar=None,
        recent_acceptance_rate=1.0 if report.accepted else 0.0,
        consecutive_failures=0,
        multi_window_summary={},
        leaderboard=[],
    )
