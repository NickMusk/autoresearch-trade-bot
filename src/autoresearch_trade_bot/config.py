from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Tuple


@dataclass(frozen=True)
class RiskLimits:
    max_gross_leverage: float = 1.0
    max_symbol_weight: float = 0.10
    max_turnover_per_step: float = 0.40


@dataclass(frozen=True)
class PromotionGate:
    min_total_return: float = 0.00
    min_sharpe: float = 0.50
    max_drawdown: float = 0.20
    max_average_turnover: float = 0.40


@dataclass(frozen=True)
class DataConfig:
    exchange: str = "binance"
    market: str = "usdm_futures"
    timeframe: str = "5m"
    storage_root: str = "data"
    strict_validation: bool = True
    max_batch_size: int = 1000
    request_timeout_seconds: int = 30
    default_start: datetime = field(
        default_factory=lambda: datetime(2025, 1, 1, tzinfo=timezone.utc)
    )
    default_end: datetime = field(
        default_factory=lambda: datetime(2025, 1, 2, tzinfo=timezone.utc)
    )


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    universe: Tuple[str, ...]
    initial_capital: float = 1.0
    fee_rate: float = 0.0005
    bars_per_year: int = 365 * 24 * 12
    data_config: DataConfig = field(default_factory=DataConfig)
    risk_limits: RiskLimits = field(default_factory=RiskLimits)
    promotion_gate: PromotionGate = field(default_factory=PromotionGate)
