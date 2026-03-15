from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import DataConfig, ExperimentConfig, PromotionGate, RiskLimits
from .data import ManifestHistoricalDataSource
from .datasets import DatasetSpec, timeframe_to_timedelta
from .research import ResearchEvaluator
from .strategy import CrossSectionalMomentumStrategy


@dataclass(frozen=True)
class BaselineRunReport:
    dataset_id: str
    manifest_path: str
    symbols: tuple[str, ...]
    timeframe: str
    accepted: bool
    score: float
    metrics: dict[str, Any]
    rejection_reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "manifest_path": self.manifest_path,
            "symbols": list(self.symbols),
            "timeframe": self.timeframe,
            "accepted": self.accepted,
            "score": self.score,
            "metrics": self.metrics,
            "rejection_reasons": self.rejection_reasons,
        }


def bars_per_year_for_timeframe(timeframe: str) -> int:
    step = timeframe_to_timedelta(timeframe)
    minutes = step.total_seconds() / 60.0
    return int((365 * 24 * 60) / minutes)


def baseline_strategy_for_symbol_count(symbol_count: int) -> CrossSectionalMomentumStrategy:
    top_k = max(1, min(3, symbol_count // 2))
    return CrossSectionalMomentumStrategy(
        lookback_bars=12,
        top_k=top_k,
        gross_target=1.0,
    )


def build_baseline_experiment_config(spec: DatasetSpec, storage_root: str = "data") -> ExperimentConfig:
    return ExperimentConfig(
        name=f"baseline-{spec.dataset_id}",
        universe=spec.symbols,
        bars_per_year=bars_per_year_for_timeframe(spec.timeframe),
        data_config=DataConfig(
            exchange=spec.exchange,
            market=spec.market,
            timeframe=spec.timeframe,
            storage_root=storage_root,
            default_start=spec.start,
            default_end=spec.end,
        ),
        risk_limits=RiskLimits(
            max_gross_leverage=1.0,
            max_symbol_weight=0.25,
            max_turnover_per_step=1.0,
        ),
        promotion_gate=PromotionGate(
            min_total_return=0.00,
            min_sharpe=0.30,
            max_drawdown=0.35,
            max_average_turnover=1.0,
        ),
    )


def discover_latest_manifest(storage_root: str | Path) -> Path | None:
    root = Path(storage_root)
    manifests = sorted(root.glob("**/manifest.json"), key=lambda item: item.stat().st_mtime)
    if not manifests:
        return None
    return manifests[-1]


def run_baseline_from_manifest_path(
    manifest_path: str | Path,
    output_dir: str | Path | None = None,
) -> BaselineRunReport:
    loader = ManifestHistoricalDataSource.from_manifest_path(manifest_path)
    spec = DatasetSpec(
        exchange=loader.manifest.exchange,
        market=loader.manifest.market,
        timeframe=loader.manifest.timeframe,
        start=loader.manifest.start,
        end=loader.manifest.end,
        symbols=loader.manifest.symbols,
    )
    bars = loader.load_bars(spec)
    config = build_baseline_experiment_config(spec, storage_root=str(loader.storage_root))
    strategy = baseline_strategy_for_symbol_count(len(spec.symbols))
    result = ResearchEvaluator(config).evaluate(bars, strategy)
    report = BaselineRunReport(
        dataset_id=spec.dataset_id,
        manifest_path=str(Path(manifest_path)),
        symbols=spec.symbols,
        timeframe=spec.timeframe,
        accepted=result.accepted,
        score=result.score,
        metrics={
            "total_return": result.metrics.total_return,
            "sharpe": result.metrics.sharpe,
            "max_drawdown": result.metrics.max_drawdown,
            "average_turnover": result.metrics.average_turnover,
            "bars_processed": result.metrics.bars_processed,
        },
        rejection_reasons=list(result.rejection_reasons),
    )
    if output_dir is not None:
        save_baseline_run(report, output_dir)
    return report


def save_baseline_run(report: BaselineRunReport, output_dir: str | Path) -> Path:
    destination_dir = Path(output_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    path = destination_dir / f"{report.dataset_id}.json"
    path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return path
