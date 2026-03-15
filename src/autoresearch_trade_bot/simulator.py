from __future__ import annotations

import math
import statistics
from typing import Dict, Mapping, Sequence

from .config import ExperimentConfig
from .models import Bar, SimulationMetrics, SimulationResult
from .risk import RiskManager
from .strategy import Strategy


class BacktestEngine:
    """Deterministic bar-by-bar simulator used for research and paper parity."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.risk_manager = RiskManager(config.risk_limits)

    def run(
        self,
        bars_by_symbol: Mapping[str, Sequence[Bar]],
        strategy: Strategy,
    ) -> SimulationResult:
        self._validate_inputs(bars_by_symbol)

        symbols = tuple(sorted(bars_by_symbol))
        bar_count = len(next(iter(bars_by_symbol.values())))

        equity_curve = [self.config.initial_capital]
        step_returns = []
        turnovers = []
        weight_history = []
        current_weights: Dict[str, float] = {}

        for step_index in range(1, bar_count):
            realized_return = 0.0
            for symbol in symbols:
                previous_bar = bars_by_symbol[symbol][step_index - 1]
                current_bar = bars_by_symbol[symbol][step_index]
                asset_return = (current_bar.close / previous_bar.close) - 1.0
                funding_drag = current_weights.get(symbol, 0.0) * current_bar.funding_rate
                realized_return += current_weights.get(symbol, 0.0) * asset_return
                realized_return -= funding_drag

            visible_history = {
                symbol: bars_by_symbol[symbol][: step_index + 1] for symbol in symbols
            }
            proposed_weights = strategy.target_weights(visible_history)
            target_weights = self.risk_manager.apply(proposed_weights, current_weights)

            turnover = self._turnover(current_weights, target_weights)
            fee_drag = turnover * self.config.fee_rate
            step_return = realized_return - fee_drag

            next_equity = equity_curve[-1] * (1.0 + step_return)
            equity_curve.append(next_equity)
            step_returns.append(step_return)
            turnovers.append(turnover)
            weight_history.append(dict(target_weights))
            current_weights = target_weights

        metrics = self._compute_metrics(equity_curve, step_returns, turnovers)
        return SimulationResult(
            equity_curve=equity_curve,
            step_returns=step_returns,
            turnovers=turnovers,
            weight_history=weight_history,
            metrics=metrics,
        )

    def _compute_metrics(
        self,
        equity_curve: Sequence[float],
        step_returns: Sequence[float],
        turnovers: Sequence[float],
    ) -> SimulationMetrics:
        total_return = (equity_curve[-1] / equity_curve[0]) - 1.0

        if len(step_returns) > 1:
            mean_return = statistics.fmean(step_returns)
            volatility = statistics.pstdev(step_returns)
            if math.isclose(volatility, 0.0):
                sharpe = 0.0
            else:
                sharpe = math.sqrt(self.config.bars_per_year) * (mean_return / volatility)
        else:
            sharpe = 0.0

        max_drawdown = 0.0
        running_peak = equity_curve[0]
        for equity in equity_curve:
            running_peak = max(running_peak, equity)
            drawdown = 1.0 - (equity / running_peak)
            max_drawdown = max(max_drawdown, drawdown)

        average_turnover = statistics.fmean(turnovers) if turnovers else 0.0
        return SimulationMetrics(
            total_return=total_return,
            sharpe=sharpe,
            max_drawdown=max_drawdown,
            average_turnover=average_turnover,
            bars_processed=len(step_returns),
        )

    def _validate_inputs(self, bars_by_symbol: Mapping[str, Sequence[Bar]]) -> None:
        if not bars_by_symbol:
            raise ValueError("bars_by_symbol must not be empty")

        lengths = {symbol: len(bars) for symbol, bars in bars_by_symbol.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError("all symbols must have the same bar count")

        expected_timestamps = [bar.timestamp for bar in next(iter(bars_by_symbol.values()))]
        for symbol, bars in bars_by_symbol.items():
            timestamps = [bar.timestamp for bar in bars]
            if timestamps != expected_timestamps:
                raise ValueError(
                    "all symbols must be aligned by timestamp; misaligned series: %s"
                    % symbol
                )

    @staticmethod
    def _turnover(
        current_weights: Mapping[str, float],
        target_weights: Mapping[str, float],
    ) -> float:
        all_symbols = set(current_weights) | set(target_weights)
        return sum(
            abs(target_weights.get(symbol, 0.0) - current_weights.get(symbol, 0.0))
            for symbol in all_symbols
        )
