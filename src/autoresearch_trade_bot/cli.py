from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from .config import DataConfig
from .data import HistoricalDatasetMaterializer
from .datasets import DatasetSpec
from .experiments import run_baseline_from_manifest_path


def parse_datetime(value: str) -> datetime:
    if "T" not in value:
        value = f"{value}T00:00:00+00:00"
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def cmd_materialize_binance(args: argparse.Namespace) -> int:
    spec = DatasetSpec(
        exchange="binance",
        market="usdm_futures",
        timeframe=args.timeframe,
        start=parse_datetime(args.start),
        end=parse_datetime(args.end),
        symbols=tuple(symbol.strip().upper() for symbol in args.symbols.split(",") if symbol.strip()),
    )
    data_config = DataConfig(
        exchange="binance",
        market="usdm_futures",
        timeframe=args.timeframe,
        storage_root=args.storage_root,
        default_start=spec.start,
        default_end=spec.end,
    )
    dataset = HistoricalDatasetMaterializer.for_binance(data_config).materialize(spec)
    print(dataset.manifest_path)
    return 0


def cmd_run_baseline(args: argparse.Namespace) -> int:
    report = run_baseline_from_manifest_path(args.manifest_path, output_dir=args.output_dir)
    print(json.dumps(report.to_dict(), indent=2))
    return 0


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
    materialize.set_defaults(func=cmd_materialize_binance)

    baseline = subparsers.add_parser(
        "run-baseline",
        help="Run the baseline strategy against a materialized dataset manifest",
    )
    baseline.add_argument("--manifest-path", required=True)
    baseline.add_argument("--output-dir", default="artifacts/baseline-runs")
    baseline.set_defaults(func=cmd_run_baseline)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
