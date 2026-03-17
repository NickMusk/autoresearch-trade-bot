from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .autoresearch import (
    DEFAULT_ACTIVE_CAMPAIGN,
    DEFAULT_CAMPAIGNS_ROOT,
    GitAutoresearchRunner,
    append_results_row,
    evaluate_train_file,
    prepare_campaign,
    resolve_campaign_path,
    run_deterministic_mutation_campaign,
)
from .config import DataConfig
from .data import HistoricalDatasetMaterializer
from .datasets import DatasetSpec
from .experiments import run_baseline_from_manifest_path
from .mutations import run_llm_mutation_campaign
from .state import FilesystemResearchStateStore
from .worker import ContinuousResearchWorker, worker_config_from_env


def parse_datetime(value: str) -> datetime:
    if "T" not in value:
        value = f"{value}T00:00:00+00:00"
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def cmd_materialize_dataset(args: argparse.Namespace) -> int:
    market = args.market
    exchange = args.exchange
    spec = DatasetSpec(
        exchange=exchange,
        market=market,
        timeframe=args.timeframe,
        start=parse_datetime(args.start),
        end=parse_datetime(args.end),
        symbols=tuple(symbol.strip().upper() for symbol in args.symbols.split(",") if symbol.strip()),
    )
    data_config = DataConfig(
        exchange=exchange,
        market=market,
        timeframe=args.timeframe,
        storage_root=args.storage_root,
        default_start=spec.start,
        default_end=spec.end,
    )
    dataset = HistoricalDatasetMaterializer.for_exchange(data_config).materialize(spec)
    print(dataset.manifest_path)
    return 0


def cmd_run_baseline(args: argparse.Namespace) -> int:
    report = run_baseline_from_manifest_path(args.manifest_path, output_dir=args.output_dir)
    print(json.dumps(report.to_dict(), indent=2))
    return 0


def cmd_run_worker_cycle(_args: argparse.Namespace) -> int:
    worker_config = worker_config_from_env()
    worker = ContinuousResearchWorker(
        worker_config=worker_config,
        state_store=FilesystemResearchStateStore(worker_config.state_root),
    )
    result = worker.run_cycle()
    print(
        json.dumps(
            {
                "cycle_id": result.cycle_id,
                "dataset_id": result.dataset_id,
                "manifest_path": result.manifest_path,
                "rollout_ready": result.rollout_ready,
                "recent_acceptance_rate": result.recent_acceptance_rate,
                "best_entry": result.best_entry.to_dict(),
            },
            indent=2,
        )
    )
    return 0


def cmd_prepare_autoresearch(args: argparse.Namespace) -> int:
    campaign_path = prepare_campaign(
        campaign_name=args.campaign_name,
        exchange=args.exchange,
        market=args.market,
        timeframe=args.timeframe,
        symbols=tuple(
            symbol.strip().upper() for symbol in args.symbols.split(",") if symbol.strip()
        ),
        anchor_end=parse_datetime(args.end),
        window_days=args.window_days,
        window_count=args.window_count,
        storage_root=args.storage_root,
        campaign_path=args.campaign_path,
        campaigns_root=args.campaigns_root,
        active_pointer_path=args.active_campaign_path,
    )
    print(campaign_path)
    return 0


def cmd_eval_autoresearch(args: argparse.Namespace) -> int:
    campaign_path = resolve_campaign_path(
        args.campaign_path,
        campaign_name=args.campaign_name,
        campaigns_root=args.campaigns_root,
        pointer_path=args.active_campaign_path,
    )
    report = evaluate_train_file(
        campaign_path=campaign_path,
        train_path=args.train_path,
        artifact_root=args.artifact_root,
        stage=args.stage,
        mutation_label=args.mutation_label,
    )
    append_results_row(
        results_path=args.results_path,
        report=report,
        decision=args.decision,
    )
    print(json.dumps(report.to_dict(), indent=2))
    return 0


def cmd_apply_autoresearch_candidate(args: argparse.Namespace) -> int:
    campaign_path = resolve_campaign_path(
        args.campaign_path,
        campaign_name=args.campaign_name,
        campaigns_root=args.campaigns_root,
        pointer_path=args.active_campaign_path,
    )
    runner = GitAutoresearchRunner(
        repo_root=args.repo_root,
        branch_name=args.branch_name,
        train_relpath=args.train_path,
        results_relpath=args.results_path,
        artifact_relpath=args.artifact_root,
        worktrees_root=args.worktrees_root,
    )
    decision = runner.apply_candidate_staged(
        campaign_path=campaign_path,
        candidate_text=parse_candidate_text(args.candidate_file),
        commit_message=args.commit_message,
        mutation_label=args.mutation_label,
    )
    print(
        json.dumps(
            {
                "decision": decision.decision,
                "stage": decision.stage,
                "mutation_label": decision.mutation_label,
                "baseline_score": decision.baseline_score,
                "candidate_score": decision.candidate_score,
                "kept_commit": decision.kept_commit,
                "report": decision.report.to_dict(),
            },
            indent=2,
        )
    )
    return 0


def cmd_run_deterministic_autoresearch(args: argparse.Namespace) -> int:
    campaign_path = resolve_campaign_path(
        args.campaign_path,
        campaign_name=args.campaign_name,
        campaigns_root=args.campaigns_root,
        pointer_path=args.active_campaign_path,
    )
    decisions = run_deterministic_mutation_campaign(
        campaign_path=campaign_path,
        repo_root=args.repo_root,
        branch_name=args.branch_name,
        max_mutations=args.max_mutations,
        worktrees_root=args.worktrees_root,
    )
    print(
        json.dumps(
            [
                {
                    "decision": item.decision,
                    "stage": item.stage,
                    "mutation_label": item.mutation_label,
                    "baseline_score": item.baseline_score,
                    "candidate_score": item.candidate_score,
                    "kept_commit": item.kept_commit,
                }
                for item in decisions
            ],
            indent=2,
        )
    )
    return 0


def cmd_run_llm_autoresearch(args: argparse.Namespace) -> int:
    campaign_path = resolve_campaign_path(
        args.campaign_path,
        campaign_name=args.campaign_name,
        campaigns_root=args.campaigns_root,
        pointer_path=args.active_campaign_path,
    )
    decisions = run_llm_mutation_campaign(
        campaign_path=campaign_path,
        repo_root=args.repo_root,
        branch_name=args.branch_name,
        model_name=args.model_name,
        max_mutations=args.max_mutations,
        worktrees_root=args.worktrees_root,
        recent_results_limit=args.recent_results_limit,
    )
    print(
        json.dumps(
            [
                {
                    "decision": item.decision,
                    "stage": item.stage,
                    "mutation_label": item.mutation_label,
                    "baseline_score": item.baseline_score,
                    "candidate_score": item.candidate_score,
                    "kept_commit": item.kept_commit,
                    "failure_reason": item.report.failure_reason,
                    "provider_name": item.report.provider_name,
                    "model_name": item.report.model_name,
                }
                for item in decisions
            ],
            indent=2,
        )
    )
    return 0


def parse_candidate_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Autoresearch trade bot workflow CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    materialize = subparsers.add_parser(
        "materialize-binance",
        help="Fetch, validate, and materialize a Binance historical dataset",
    )
    materialize.add_argument("--symbols", required=True, help="Comma-separated symbols")
    materialize.add_argument("--timeframe", default="5m")
    materialize.add_argument("--start", required=True)
    materialize.add_argument("--end", required=True)
    materialize.add_argument("--storage-root", default="data")
    materialize.set_defaults(
        func=cmd_materialize_dataset,
        exchange="binance",
        market="usdm_futures",
    )

    materialize_bybit = subparsers.add_parser(
        "materialize-bybit",
        help="Fetch, validate, and materialize a Bybit historical dataset",
    )
    materialize_bybit.add_argument("--symbols", required=True, help="Comma-separated symbols")
    materialize_bybit.add_argument("--timeframe", default="5m")
    materialize_bybit.add_argument("--start", required=True)
    materialize_bybit.add_argument("--end", required=True)
    materialize_bybit.add_argument("--storage-root", default="data")
    materialize_bybit.set_defaults(
        func=cmd_materialize_dataset,
        exchange="bybit",
        market="linear",
    )

    baseline = subparsers.add_parser(
        "run-baseline",
        help="Run the baseline strategy against a materialized dataset manifest",
    )
    baseline.add_argument("--manifest-path", required=True)
    baseline.add_argument("--output-dir", default="artifacts/baseline-runs")
    baseline.set_defaults(func=cmd_run_baseline)

    worker_cycle = subparsers.add_parser(
        "run-worker-cycle",
        help="Run one continuous research worker cycle using environment configuration",
    )
    worker_cycle.set_defaults(func=cmd_run_worker_cycle)

    prepare_autoresearch = subparsers.add_parser(
        "prepare-autoresearch",
        help="Freeze a Karpathy-style crypto autoresearch campaign",
    )
    prepare_autoresearch.add_argument("--campaign-name", default="crypto-bybit-5m")
    prepare_autoresearch.add_argument("--exchange", default="bybit")
    prepare_autoresearch.add_argument("--market", default="linear")
    prepare_autoresearch.add_argument("--timeframe", default="5m")
    prepare_autoresearch.add_argument(
        "--symbols",
        default="BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT",
    )
    prepare_autoresearch.add_argument("--window-days", type=int, default=7)
    prepare_autoresearch.add_argument("--window-count", type=int, default=2)
    prepare_autoresearch.add_argument("--end", required=True)
    prepare_autoresearch.add_argument("--storage-root", default="data")
    prepare_autoresearch.add_argument("--campaign-path")
    prepare_autoresearch.add_argument("--campaigns-root", default=DEFAULT_CAMPAIGNS_ROOT)
    prepare_autoresearch.add_argument("--active-campaign-path", default=DEFAULT_ACTIVE_CAMPAIGN)
    prepare_autoresearch.set_defaults(func=cmd_prepare_autoresearch)

    eval_autoresearch = subparsers.add_parser(
        "eval-autoresearch",
        help="Evaluate the current train.py against a frozen autoresearch campaign",
    )
    eval_autoresearch.add_argument("--campaign-path")
    eval_autoresearch.add_argument("--campaign-name")
    eval_autoresearch.add_argument("--campaigns-root", default=DEFAULT_CAMPAIGNS_ROOT)
    eval_autoresearch.add_argument("--active-campaign-path", default=DEFAULT_ACTIVE_CAMPAIGN)
    eval_autoresearch.add_argument("--train-path", default="train.py")
    eval_autoresearch.add_argument("--artifact-root", default=".autoresearch/runs")
    eval_autoresearch.add_argument("--results-path", default="results.tsv")
    eval_autoresearch.add_argument("--decision", default="observed")
    eval_autoresearch.add_argument("--stage", default="full")
    eval_autoresearch.add_argument("--mutation-label", default="manual")
    eval_autoresearch.set_defaults(func=cmd_eval_autoresearch)

    apply_candidate = subparsers.add_parser(
        "apply-autoresearch-candidate",
        help="Apply a candidate train.py mutation on a research branch and keep it only if the score improves",
    )
    apply_candidate.add_argument("--campaign-path")
    apply_candidate.add_argument("--campaign-name")
    apply_candidate.add_argument("--campaigns-root", default=DEFAULT_CAMPAIGNS_ROOT)
    apply_candidate.add_argument("--active-campaign-path", default=DEFAULT_ACTIVE_CAMPAIGN)
    apply_candidate.add_argument("--candidate-file", required=True)
    apply_candidate.add_argument("--repo-root", default=".")
    apply_candidate.add_argument("--branch-name", required=True)
    apply_candidate.add_argument("--train-path", default="train.py")
    apply_candidate.add_argument("--results-path", default="results.tsv")
    apply_candidate.add_argument("--artifact-root", default=".autoresearch/runs")
    apply_candidate.add_argument("--worktrees-root", default=".autoresearch/worktrees")
    apply_candidate.add_argument("--mutation-label", default="manual-candidate")
    apply_candidate.add_argument("--commit-message", required=True)
    apply_candidate.set_defaults(func=cmd_apply_autoresearch_candidate)

    deterministic = subparsers.add_parser(
        "run-deterministic-autoresearch",
        help="Run a staged deterministic mutation batch against the active frozen campaign",
    )
    deterministic.add_argument("--campaign-path")
    deterministic.add_argument("--campaign-name")
    deterministic.add_argument("--campaigns-root", default=DEFAULT_CAMPAIGNS_ROOT)
    deterministic.add_argument("--active-campaign-path", default=DEFAULT_ACTIVE_CAMPAIGN)
    deterministic.add_argument("--repo-root", default=".")
    deterministic.add_argument("--branch-name", required=True)
    deterministic.add_argument("--worktrees-root", default=".autoresearch/worktrees")
    deterministic.add_argument("--max-mutations", type=int, default=8)
    deterministic.set_defaults(func=cmd_run_deterministic_autoresearch)

    llm = subparsers.add_parser(
        "run-llm-autoresearch",
        help="Run a staged LLM mutation batch against the active frozen campaign",
    )
    llm.add_argument("--campaign-path")
    llm.add_argument("--campaign-name")
    llm.add_argument("--campaigns-root", default=DEFAULT_CAMPAIGNS_ROOT)
    llm.add_argument("--active-campaign-path", default=DEFAULT_ACTIVE_CAMPAIGN)
    llm.add_argument("--repo-root", default=".")
    llm.add_argument("--branch-name", required=True)
    llm.add_argument("--worktrees-root", default=".autoresearch/worktrees")
    llm.add_argument("--model-name", default="gpt-5-mini")
    llm.add_argument("--max-mutations", type=int, default=1)
    llm.add_argument("--recent-results-limit", type=int, default=5)
    llm.set_defaults(func=cmd_run_llm_autoresearch)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
