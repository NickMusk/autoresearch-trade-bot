from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List

from .config import ExperimentConfig, PromotionGate, RiskLimits
from .research import ResearchEvaluator
from .sample_data import build_demo_dataset
from .strategy import CrossSectionalMomentumStrategy


@dataclass(frozen=True)
class DashboardSnapshot:
    mission: str
    phase: str
    research_rollout_ready: bool
    research_blockers: List[str]
    baseline_strategy: str
    promotion_gate: Dict[str, float]
    baseline_metrics: Dict[str, float]
    accepted_for_paper: bool
    next_milestones: List[str]


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


def build_dashboard_snapshot() -> DashboardSnapshot:
    config = default_experiment_config()
    strategy = CrossSectionalMomentumStrategy(lookback_bars=2, top_k=1, gross_target=1.0)
    evaluator = ResearchEvaluator(config)
    result = evaluator.evaluate(build_demo_dataset(config.universe), strategy)

    return DashboardSnapshot(
        mission="Find, test, and rank crypto strategies before paper or live rollout.",
        phase="Render-deployed research kernel",
        research_rollout_ready=False,
        research_blockers=[
            "Historical Binance ingestion is not implemented yet.",
            "Realtime paper or shadow market data stream is not implemented yet.",
            "Experiment persistence and walk-forward validation are not implemented yet.",
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
            "Add Binance historical ingestion and dataset validators.",
            "Add walk-forward evaluation and experiment persistence.",
            "Add realtime paper or shadow replay using the same engine boundaries.",
        ],
    )
