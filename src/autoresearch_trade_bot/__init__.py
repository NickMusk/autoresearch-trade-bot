"""Core research kernel for the autoresearch trading system."""

from .autoresearch import (
    AutoresearchCampaign,
    AutoresearchRunReport,
    DeterministicMutationProposal,
    DeterministicTrainMutator,
    FrozenResearchWindow,
    GitAutoresearchDecision,
    GitAutoresearchRunner,
    MutationProposal,
    append_results_row,
    campaign_path_for_name,
    evaluate_train_file,
    prepare_campaign,
    render_train_file,
    resolve_campaign_path,
    run_deterministic_mutation_campaign,
)
from .config import (
    DataConfig,
    ExperimentConfig,
    LLMWorkerConfig,
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
from .family_wave import FamilyWaveRunResult, prepare_family_repo, run_llm_family_wave
from .models import Bar, ExperimentResult, SimulationMetrics, SimulationResult
from .llm_worker import LLMAutoresearchWorker
from .mutations import (
    DeterministicMutationProvider,
    LLMCompletion,
    LLMMutationProvider,
    MutationContext,
    MutationProvider,
    OpenAIResponsesClient,
    build_llm_mutation_prompt,
    run_llm_mutation_campaign,
    validate_train_candidate_text,
)
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
    "AutoresearchCampaign",
    "AutoresearchRunReport",
    "Bar",
    "BacktestEngine",
    "BaselineRunReport",
    "CrossSectionalMomentumStrategy",
    "ContinuousResearchWorker",
    "CycleSummary",
    "DataConfig",
    "DataValidationError",
    "DashboardSnapshot",
    "DeterministicMutationProposal",
    "DeterministicMutationProvider",
    "DeterministicTrainMutator",
    "DatasetManifest",
    "DatasetSpec",
    "ExperimentConfig",
    "ExperimentResult",
    "FamilyWaveRunResult",
    "FilesystemResearchStateStore",
    "FrozenResearchWindow",
    "GitAutoresearchDecision",
    "GitAutoresearchRunner",
    "HistoricalDatasetMaterializer",
    "LLMAutoresearchWorker",
    "LLMCompletion",
    "LLMMutationProvider",
    "LLMWorkerConfig",
    "LeaderboardEntry",
    "ManifestHistoricalDataSource",
    "MutationProposal",
    "MutationContext",
    "MutationProvider",
    "OpenAIResponsesClient",
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
    "append_results_row",
    "build_llm_mutation_prompt",
    "campaign_path_for_name",
    "evaluate_train_file",
    "prepare_campaign",
    "prepare_family_repo",
    "render_train_file",
    "resolve_campaign_path",
    "run_baseline_from_manifest_path",
    "run_deterministic_mutation_campaign",
    "run_llm_family_wave",
    "run_llm_mutation_campaign",
    "run_variant_search",
    "validate_train_candidate_text",
]
