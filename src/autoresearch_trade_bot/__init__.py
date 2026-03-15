"""Core research kernel for the autoresearch trading system."""

from .config import DataConfig, ExperimentConfig, PromotionGate, RiskLimits
from .data import (
    DataValidationError,
    HistoricalDatasetMaterializer,
    ManifestHistoricalDataSource,
)
from .dashboard import DashboardSnapshot, build_dashboard_snapshot, default_experiment_config
from .datasets import DatasetManifest, DatasetSpec, ValidatedDataset, ValidationIssue
from .experiments import (
    BaselineRunReport,
    build_baseline_experiment_config,
    discover_latest_manifest,
    run_baseline_from_manifest_path,
)
from .models import Bar, ExperimentResult, SimulationMetrics, SimulationResult
from .research import ResearchEvaluator
from .risk import RiskManager
from .simulator import BacktestEngine
from .storage import PyArrowParquetDatasetStore
from .strategy import CrossSectionalMomentumStrategy, Strategy

__all__ = [
    "Bar",
    "BacktestEngine",
    "BaselineRunReport",
    "CrossSectionalMomentumStrategy",
    "DataConfig",
    "DataValidationError",
    "DashboardSnapshot",
    "DatasetManifest",
    "DatasetSpec",
    "ExperimentConfig",
    "ExperimentResult",
    "HistoricalDatasetMaterializer",
    "ManifestHistoricalDataSource",
    "PromotionGate",
    "PyArrowParquetDatasetStore",
    "ResearchEvaluator",
    "RiskLimits",
    "RiskManager",
    "SimulationMetrics",
    "SimulationResult",
    "Strategy",
    "ValidatedDataset",
    "ValidationIssue",
    "build_baseline_experiment_config",
    "build_dashboard_snapshot",
    "default_experiment_config",
    "discover_latest_manifest",
    "run_baseline_from_manifest_path",
]
