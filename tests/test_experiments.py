from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from autoresearch_trade_bot.binance import BinanceBarNormalizer
from autoresearch_trade_bot.config import DataConfig
from autoresearch_trade_bot.data import HistoricalDatasetMaterializer
from autoresearch_trade_bot.datasets import DatasetSpec, RawSymbolHistory
from autoresearch_trade_bot.experiments import (
    bars_per_year_for_timeframe,
    discover_latest_manifest,
    run_baseline_from_manifest_path,
)
from autoresearch_trade_bot.storage import PyArrowParquetDatasetStore
from autoresearch_trade_bot.validation import DatasetValidator


class FakeHistoricalClient:
    def __init__(self, payload_by_symbol) -> None:
        self.payload_by_symbol = payload_by_symbol

    def fetch_symbol_history(self, spec: DatasetSpec, symbol: str) -> RawSymbolHistory:
        return self.payload_by_symbol[symbol]


def kline(open_time_ms: int, open_: float, high: float, low: float, close: float, volume: float, trade_count: int = 10) -> list:
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


class ExperimentWorkflowTests(unittest.TestCase):
    def test_bars_per_year_for_5m(self) -> None:
        self.assertEqual(bars_per_year_for_timeframe("5m"), 105120)

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
                    kline(1735689600000 + (index * 300000), 100 + index, 101 + index, 99 + index, 100.5 + index, 1000 + (index * 10))
                    for index in range(16)
                ],
                funding_rates=[],
                open_interest_stats=[],
            ),
            "ETHUSDT": RawSymbolHistory(
                symbol="ETHUSDT",
                klines=[
                    kline(1735689600000 + (index * 300000), 100 - (index * 0.8), 100.5 - (index * 0.8), 99 - (index * 0.8), 99.5 - (index * 0.8), 900 + (index * 10))
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


if __name__ == "__main__":
    unittest.main()
