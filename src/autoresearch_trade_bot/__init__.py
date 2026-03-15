"""Core research kernel for the autoresearch trading system."""

from .config import ExperimentConfig, PromotionGate, RiskLimits
from .dashboard import DashboardSnapshot, build_dashboard_snapshot, default_experiment_config
from .models import Bar, ExperimentResult, SimulationMetrics, SimulationResult
from .research import ResearchEvaluator
from .risk import RiskManager
from .simulator import BacktestEngine
from .strategy import CrossSectionalMomentumStrategy, Strategy

__all__ = [
    "Bar",
    "BacktestEngine",
    "CrossSectionalMomentumStrategy",
    "DashboardSnapshot",
    "ExperimentConfig",
    "ExperimentResult",
    "PromotionGate",
    "ResearchEvaluator",
    "RiskLimits",
    "RiskManager",
    "SimulationMetrics",
    "SimulationResult",
    "Strategy",
    "build_dashboard_snapshot",
    "default_experiment_config",
]
