from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List


@dataclass(frozen=True)
class Bar:
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    funding_rate: float = 0.0
    open_interest: float = 0.0


@dataclass(frozen=True)
class SimulationMetrics:
    total_return: float
    sharpe: float
    max_drawdown: float
    average_turnover: float
    bars_processed: int


@dataclass(frozen=True)
class SimulationResult:
    equity_curve: List[float]
    step_returns: List[float]
    turnovers: List[float]
    weight_history: List[Dict[str, float]]
    metrics: SimulationMetrics


@dataclass(frozen=True)
class ExperimentResult:
    accepted: bool
    score: float
    metrics: SimulationMetrics
    rejection_reasons: List[str] = field(default_factory=list)
