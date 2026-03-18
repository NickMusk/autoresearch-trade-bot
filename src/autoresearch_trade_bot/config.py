from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Tuple


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


@dataclass(frozen=True)
class ResearchTargetGate:
    min_total_return: float = 0.00
    min_sharpe: float = 1.00
    max_drawdown: float = 0.20
    min_acceptance_rate: float = 0.60


@dataclass(frozen=True)
class WorkerConfig:
    symbols: Tuple[str, ...] = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT")
    timeframe: str = "5m"
    history_window_days: int = 7
    max_variants_per_cycle: int = 12
    multi_window_count: int = 4
    multi_window_top_candidates: int = 3
    multi_window_validation_interval_bars: int = 12
    recent_cycles_for_acceptance: int = 12
    poll_buffer_seconds: int = 15
    failure_cooldown_seconds: int = 90
    state_root: str = "state"
    artifact_root: str = "artifacts"
    max_cycles: Optional[int] = None
    data_config: DataConfig = field(default_factory=DataConfig)
    target_gate: ResearchTargetGate = field(default_factory=ResearchTargetGate)


@dataclass(frozen=True)
class LLMWorkerConfig:
    repo_url: str
    symbols: Tuple[str, ...] = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT")
    repo_root: str = "repo"
    branch_name: str = "codex/autoresearch-crypto"
    model_name: str = "gpt-5-mini"
    cycle_interval_seconds: int = 3600
    campaign_refresh_interval_seconds: int = 86400
    failure_cooldown_seconds: int = 900
    timeout_failure_cooldown_seconds: int = 180
    openai_timeout_seconds: float = 120.0
    openai_max_retries: int = 3
    openai_retry_backoff_seconds: float = 2.0
    max_mutations_per_cycle: int = 1
    recent_results_limit: int = 5
    history_window_days: int = 7
    campaign_window_count: int = 1
    campaigns_root: str = "campaigns"
    active_campaign_path: str = "active_campaign.txt"
    worktrees_root: str = "worktrees"
    state_root: str = "state"
    artifact_root: str = "artifacts"
    results_path: str = "results.tsv"
    max_cycles: Optional[int] = None
    data_config: DataConfig = field(default_factory=DataConfig)
    target_gate: ResearchTargetGate = field(default_factory=ResearchTargetGate)
