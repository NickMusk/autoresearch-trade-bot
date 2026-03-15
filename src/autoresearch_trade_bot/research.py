from __future__ import annotations

from .config import ExperimentConfig
from .models import ExperimentResult
from .simulator import BacktestEngine
from .strategy import Strategy


class ResearchEvaluator:
    """Scores a strategy and decides whether it is allowed into the next stage."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.engine = BacktestEngine(config)

    def evaluate(self, bars_by_symbol, strategy: Strategy) -> ExperimentResult:
        simulation = self.engine.run(bars_by_symbol, strategy)
        metrics = simulation.metrics

        rejection_reasons = []
        gate = self.config.promotion_gate
        if metrics.total_return < gate.min_total_return:
            rejection_reasons.append("total_return_below_gate")
        if metrics.sharpe < gate.min_sharpe:
            rejection_reasons.append("sharpe_below_gate")
        if metrics.max_drawdown > gate.max_drawdown:
            rejection_reasons.append("drawdown_above_gate")
        if metrics.average_turnover > gate.max_average_turnover:
            rejection_reasons.append("turnover_above_gate")

        score = (
            metrics.sharpe
            + (metrics.total_return * 5.0)
            - (metrics.max_drawdown * 4.0)
            - metrics.average_turnover
        )
        return ExperimentResult(
            accepted=not rejection_reasons,
            score=score,
            metrics=metrics,
            rejection_reasons=rejection_reasons,
        )
