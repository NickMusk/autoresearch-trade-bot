"""Core research kernel for the autoresearch trading system."""

from .config import (
    DataConfig,
    ExperimentConfig,
    PromotionGate,
    ResearchTargetGate,
    RiskLimits,
    WorkerConfig,
)
from .data import (
    DataValidationError,
    HistoricalDatasetMaterializer,
    ManifestHistoricalDataSource,
)
from .dashboard import DashboardSnapshot, build_dashboard_snapshot, default_experiment_config
from .datasets import DatasetManifest, DatasetSpec, ValidatedDataset, ValidationIssue
from .experiments import (
    BaselineRunReport,
    VariantRunReport,
    build_baseline_experiment_config,
    build_strategy_variants,
    discover_latest_manifest,
    run_baseline_from_manifest_path,
    run_variant_search,
)
from .models import Bar, ExperimentResult, SimulationMetrics, SimulationResult
from .research import ResearchEvaluator
from .risk import RiskManager
from .simulator import BacktestEngine
from .state import (
    CycleSummary,
    FilesystemResearchStateStore,
    LeaderboardEntry,
    ResearchStatusSnapshot,
    WorkerCheckpoint,
)
from .storage import PyArrowParquetDatasetStore
from .strategy import CrossSectionalMomentumStrategy, Strategy
from .worker import ContinuousResearchWorker

__all__ = [
    "Bar",
    "BacktestEngine",
    "BaselineRunReport",
    "CrossSectionalMomentumStrategy",
    "ContinuousResearchWorker",
    "CycleSummary",
    "DataConfig",
    "DataValidationError",
    "DashboardSnapshot",
    "DatasetManifest",
    "DatasetSpec",
    "ExperimentConfig",
    "ExperimentResult",
    "FilesystemResearchStateStore",
    "HistoricalDatasetMaterializer",
    "LeaderboardEntry",
    "ManifestHistoricalDataSource",
    "PromotionGate",
    "PyArrowParquetDatasetStore",
    "ResearchStatusSnapshot",
    "ResearchTargetGate",
    "ResearchEvaluator",
    "RiskLimits",
    "RiskManager",
    "SimulationMetrics",
    "SimulationResult",
    "Strategy",
    "VariantRunReport",
    "ValidatedDataset",
    "ValidationIssue",
    "WorkerCheckpoint",
    "WorkerConfig",
    "build_baseline_experiment_config",
    "build_strategy_variants",
    "build_dashboard_snapshot",
    "default_experiment_config",
    "discover_latest_manifest",
    "run_baseline_from_manifest_path",
    "run_variant_search",
]
