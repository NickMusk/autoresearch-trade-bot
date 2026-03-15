from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone

from autoresearch_trade_bot.binance import BinanceBarNormalizer
from autoresearch_trade_bot.config import DataConfig
from autoresearch_trade_bot.data import HistoricalDatasetMaterializer
from autoresearch_trade_bot.datasets import DatasetSpec, RawSymbolHistory
from autoresearch_trade_bot.experiments import (
    bars_per_year_for_timeframe,
    build_strategy_variants,
    discover_latest_manifest,
    run_baseline_from_manifest_path,
    run_variant_search,
)
from autoresearch_trade_bot.models import Bar
from autoresearch_trade_bot.storage import PyArrowParquetDatasetStore
from autoresearch_trade_bot.validation import DatasetValidator


class FakeHistoricalClient:
    def __init__(self, payload_by_symbol) -> None:
        self.payload_by_symbol = payload_by_symbol

    def fetch_symbol_history(self, spec: DatasetSpec, symbol: str) -> RawSymbolHistory:
        return self.payload_by_symbol[symbol]


def kline(
    open_time_ms: int,
    open_: float,
    high: float,
    low: float,
    close: float,
    volume: float,
    trade_count: int = 10,
) -> list:
    return [
        open_time_ms,
        str(open_),
        str(high),
        str(low),
        str(close),
        str(volume),
        open_time_ms + (5 * 60 * 1000) - 1,
        "0",
        trade_count,
        "0",
        "0",
        "0",
    ]


def synthetic_bar(symbol: str, timestamp: datetime, close: float) -> Bar:
    return Bar(
        exchange="binance",
        symbol=symbol,
        timeframe="5m",
        timestamp=timestamp,
        close_time=timestamp + timedelta(minutes=5) - timedelta(milliseconds=1),
        open=close,
        high=close * 1.001,
        low=close * 0.999,
        close=close,
        volume=1000.0,
        trade_count=10,
        funding_rate=0.0,
        open_interest=10000.0,
        is_closed=True,
    )


class ExperimentWorkflowTests(unittest.TestCase):
    def test_bars_per_year_for_5m(self) -> None:
        self.assertEqual(bars_per_year_for_timeframe("5m"), 105120)

    def test_build_strategy_variants_caps_count(self) -> None:
        variants = build_strategy_variants(symbol_count=5, max_variants=4)
        self.assertEqual(len(variants), 4)
        self.assertEqual(variants[0].name, "baseline")

    def test_discover_latest_manifest_and_run_baseline(self) -> None:
        start = datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)
        end = start + timedelta(minutes=80)
        spec = DatasetSpec(
            exchange="binance",
            market="usdm_futures",
            timeframe="5m",
            start=start,
            end=end,
            symbols=("BTCUSDT", "ETHUSDT"),
        )
        payload = {
            "BTCUSDT": RawSymbolHistory(
                symbol="BTCUSDT",
                klines=[
                    kline(
                        1735689600000 + (index * 300000),
                        100 + index,
                        101 + index,
                        99 + index,
                        100.5 + index,
                        1000 + (index * 10),
                    )
                    for index in range(16)
                ],
                funding_rates=[],
                open_interest_stats=[],
            ),
            "ETHUSDT": RawSymbolHistory(
                symbol="ETHUSDT",
                klines=[
                    kline(
                        1735689600000 + (index * 300000),
                        100 - (index * 0.8),
                        100.5 - (index * 0.8),
                        99 - (index * 0.8),
                        99.5 - (index * 0.8),
                        900 + (index * 10),
                    )
                    for index in range(16)
                ],
                funding_rates=[],
                open_interest_stats=[],
            ),
        }
        with tempfile.TemporaryDirectory() as tempdir:
            materializer = HistoricalDatasetMaterializer(
                data_config=DataConfig(storage_root=tempdir, strict_validation=True),
                client=FakeHistoricalClient(payload),
                store=PyArrowParquetDatasetStore(tempdir),
                validator=DatasetValidator(),
                normalizer=BinanceBarNormalizer(),
            )
            dataset = materializer.materialize(spec)

            manifest_path = discover_latest_manifest(tempdir)
            self.assertEqual(manifest_path, dataset.manifest_path)

            report = run_baseline_from_manifest_path(manifest_path)
            self.assertEqual(report.dataset_id, spec.dataset_id)
            self.assertGreater(report.metrics["total_return"], 0.0)

    def test_run_variant_search_sorts_by_score(self) -> None:
        start = datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)
        spec = DatasetSpec(
            exchange="binance",
            market="usdm_futures",
            timeframe="5m",
            start=start,
            end=start + timedelta(minutes=300),
            symbols=("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"),
        )
        bars_by_symbol = {}
        for symbol, start_price, direction in (
            ("BTCUSDT", 100.0, 1.4),
            ("ETHUSDT", 80.0, 1.1),
            ("SOLUSDT", 60.0, -0.9),
            ("BNBUSDT", 120.0, -1.2),
        ):
            bars_by_symbol[symbol] = [
                synthetic_bar(
                    symbol=symbol,
                    timestamp=start + timedelta(minutes=5 * index),
                    close=start_price + (direction * index),
                )
                for index in range(60)
            ]

        reports = run_variant_search(spec, bars_by_symbol, max_variants=6)
        self.assertEqual(len(reports), 6)
        self.assertGreaterEqual(reports[0].score, reports[-1].score)
        self.assertEqual(reports[0].dataset_id, spec.dataset_id)


if __name__ == "__main__":
    unittest.main()
