from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from autoresearch_trade_bot.binance import BinanceBarNormalizer
from autoresearch_trade_bot.config import DataConfig, ExperimentConfig
from autoresearch_trade_bot.data import (
    DataValidationError,
    HistoricalDatasetMaterializer,
    ManifestHistoricalDataSource,
    find_covering_manifest,
    find_latest_reusable_manifest,
)
from autoresearch_trade_bot.datasets import DatasetManifest, DatasetSpec, RawSymbolHistory
from autoresearch_trade_bot.research import ResearchEvaluator
from autoresearch_trade_bot.storage import DatasetStore
from autoresearch_trade_bot.strategy import CrossSectionalMomentumStrategy
from autoresearch_trade_bot.validation import DatasetValidator


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


class FakeHistoricalClient:
    def __init__(self, payload_by_symbol) -> None:
        self.payload_by_symbol = payload_by_symbol

    def fetch_symbol_history(self, spec: DatasetSpec, symbol: str) -> RawSymbolHistory:
        return self.payload_by_symbol[symbol]


class SlicingHistoricalClient:
    def __init__(self, payload_by_symbol) -> None:
        self.payload_by_symbol = payload_by_symbol
        self.calls: list[tuple[str, datetime, datetime]] = []

    def fetch_symbol_history(self, spec: DatasetSpec, symbol: str) -> RawSymbolHistory:
        self.calls.append((symbol, spec.start, spec.end))
        source = self.payload_by_symbol[symbol]
        start_ms = int(spec.start.timestamp() * 1000)
        end_ms = int(spec.end.timestamp() * 1000)
        return RawSymbolHistory(
            symbol=symbol,
            klines=[
                row
                for row in source.klines
                if start_ms <= int(row[0]) < end_ms
            ],
            funding_rates=[
                row
                for row in source.funding_rates
                if start_ms <= int(row.get("fundingTime", 0)) < end_ms
            ],
            open_interest_stats=[
                row
                for row in source.open_interest_stats
                if start_ms <= int(row.get("timestamp", 0)) < end_ms
            ],
            mark_price_klines=[
                row
                for row in source.mark_price_klines
                if start_ms <= int(row[0]) < end_ms
            ],
            index_price_klines=[
                row
                for row in source.index_price_klines
                if start_ms <= int(row[0]) < end_ms
            ],
        )


class JsonDatasetStore(DatasetStore):
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def dataset_dir(self, spec: DatasetSpec) -> Path:
        return self.root / spec.exchange / spec.market / spec.timeframe / spec.dataset_id

    def write_bars(self, spec: DatasetSpec, symbol: str, bars) -> Path:
        dataset_dir = self.dataset_dir(spec)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        path = dataset_dir / f"{symbol}.json"
        payload = []
        for bar in bars:
            payload.append(
                {
                    "exchange": bar.exchange,
                    "symbol": bar.symbol,
                    "timeframe": bar.timeframe,
                    "timestamp": bar.timestamp.isoformat(),
                    "close_time": bar.close_time.isoformat() if bar.close_time else None,
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                    "trade_count": bar.trade_count,
                    "funding_rate": bar.funding_rate,
                    "open_interest": bar.open_interest,
                    "mark_price": bar.mark_price,
                    "index_price": bar.index_price,
                    "is_closed": bar.is_closed,
                }
            )
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def read_bars(self, path: Path):
        from autoresearch_trade_bot.models import Bar

        payload = json.loads(path.read_text(encoding="utf-8"))
        return [
            Bar(
                exchange=item["exchange"],
                symbol=item["symbol"],
                timeframe=item["timeframe"],
                timestamp=datetime.fromisoformat(item["timestamp"]),
                close_time=datetime.fromisoformat(item["close_time"])
                if item["close_time"]
                else None,
                open=float(item["open"]),
                high=float(item["high"]),
                low=float(item["low"]),
                close=float(item["close"]),
                volume=float(item["volume"]),
                trade_count=int(item["trade_count"]),
                funding_rate=float(item["funding_rate"]),
                open_interest=float(item["open_interest"]),
                mark_price=item["mark_price"],
                index_price=item["index_price"],
                is_closed=bool(item["is_closed"]),
            )
            for item in payload
        ]

    def write_manifest(self, manifest: DatasetManifest, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
        return path

    def read_manifest(self, path: Path) -> DatasetManifest:
        return DatasetManifest.from_dict(json.loads(path.read_text(encoding="utf-8")))


class DataPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.start = datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)
        self.end = self.start + timedelta(minutes=20)
        self.spec = DatasetSpec(
            exchange="binance",
            market="usdm_futures",
            timeframe="5m",
            start=self.start,
            end=self.end,
            symbols=("BTCUSDT", "ETHUSDT"),
        )

    def test_validator_rejects_duplicate_timestamps(self) -> None:
        raw = RawSymbolHistory(
            symbol="BTCUSDT",
            klines=[
                kline(1735689600000, 100, 101, 99, 100.5, 1000),
                kline(1735689600000, 100.5, 101.5, 100, 101, 1100),
            ],
            funding_rates=[],
            open_interest_stats=[],
        )
        bars = BinanceBarNormalizer().normalize(
            DatasetSpec(
                exchange="binance",
                market="usdm_futures",
                timeframe="5m",
                start=self.start,
                end=self.start + timedelta(minutes=10),
                symbols=("BTCUSDT",),
            ),
            raw,
        )

        report = DatasetValidator().validate(
            DatasetSpec(
                exchange="binance",
                market="usdm_futures",
                timeframe="5m",
                start=self.start,
                end=self.start + timedelta(minutes=10),
                symbols=("BTCUSDT",),
            ),
            {"BTCUSDT": bars},
        )

        self.assertTrue(report.has_errors)
        self.assertTrue(any(issue.code == "non_monotonic_timestamp" for issue in report.issues))

    def test_materialize_and_load_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            payload = {
                "BTCUSDT": RawSymbolHistory(
                    symbol="BTCUSDT",
                    klines=[
                        kline(1735689600000, 100, 101, 99, 100.5, 1000),
                        kline(1735689900000, 100.5, 103, 100, 102.5, 1200),
                        kline(1735690200000, 102.5, 106, 102, 105.5, 1400),
                        kline(1735690500000, 105.5, 110, 105, 109.0, 1600),
                    ],
                    funding_rates=[{"fundingTime": 1735690200000, "fundingRate": "0.0001"}],
                    open_interest_stats=[
                        {"timestamp": 1735689600000, "sumOpenInterest": "2500"},
                        {"timestamp": 1735689900000, "sumOpenInterest": "2550"},
                        {"timestamp": 1735690200000, "sumOpenInterest": "2600"},
                        {"timestamp": 1735690500000, "sumOpenInterest": "2650"},
                    ],
                ),
                "ETHUSDT": RawSymbolHistory(
                    symbol="ETHUSDT",
                    klines=[
                        kline(1735689600000, 100, 100.5, 99, 99.5, 900),
                        kline(1735689900000, 99.5, 99.8, 98.5, 98.9, 950),
                        kline(1735690200000, 98.9, 99.2, 97.5, 97.8, 970),
                        kline(1735690500000, 97.8, 98.0, 96.5, 96.9, 980),
                    ],
                    funding_rates=[],
                    open_interest_stats=[
                        {"timestamp": 1735689600000, "sumOpenInterest": "1800"},
                        {"timestamp": 1735689900000, "sumOpenInterest": "1820"},
                        {"timestamp": 1735690200000, "sumOpenInterest": "1850"},
                        {"timestamp": 1735690500000, "sumOpenInterest": "1875"},
                    ],
                ),
            }
            data_config = DataConfig(storage_root=tempdir, strict_validation=True)
            store = JsonDatasetStore(tempdir)
            materializer = HistoricalDatasetMaterializer(
                data_config=data_config,
                client=FakeHistoricalClient(payload),
                store=store,
                validator=DatasetValidator(),
                normalizer=BinanceBarNormalizer(),
            )

            dataset = materializer.materialize(self.spec)

            loader = ManifestHistoricalDataSource(
                manifest_path=dataset.manifest_path,
                store=store,
                manifest=dataset.manifest,
                storage_root=Path(tempdir),
            )
            loaded = loader.load_bars(self.spec)
            self.assertEqual(set(loaded), {"BTCUSDT", "ETHUSDT"})
            self.assertEqual(len(loaded["BTCUSDT"]), 4)
            self.assertEqual(loaded["BTCUSDT"][2].funding_rate, 0.0001)
            self.assertEqual(loaded["ETHUSDT"][0].open_interest, 1800.0)

            experiment = ExperimentConfig(name="dataset-backed", universe=self.spec.symbols)
            result = ResearchEvaluator(experiment).evaluate(
                loaded,
                CrossSectionalMomentumStrategy(lookback_bars=2, top_k=1, gross_target=1.0),
            )
            self.assertGreater(result.metrics.total_return, 0.0)
            self.assertTrue(dataset.manifest_path.exists())

    def test_materialize_raises_on_gap_when_validation_is_strict(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            payload = {
                "BTCUSDT": RawSymbolHistory(
                    symbol="BTCUSDT",
                    klines=[
                        kline(1735689600000, 100, 101, 99, 100.5, 1000),
                        kline(1735690200000, 102.5, 106, 102, 105.5, 1400),
                    ],
                    funding_rates=[],
                    open_interest_stats=[],
                ),
                "ETHUSDT": RawSymbolHistory(
                    symbol="ETHUSDT",
                    klines=[
                        kline(1735689600000, 100, 100.5, 99, 99.5, 900),
                        kline(1735690200000, 98.9, 99.2, 97.5, 97.8, 970),
                    ],
                    funding_rates=[],
                    open_interest_stats=[],
                ),
            }
            materializer = HistoricalDatasetMaterializer(
                data_config=DataConfig(storage_root=tempdir, strict_validation=True),
                client=FakeHistoricalClient(payload),
                store=JsonDatasetStore(tempdir),
                validator=DatasetValidator(),
                normalizer=BinanceBarNormalizer(),
            )

            with self.assertRaises(DataValidationError):
                materializer.materialize(self.spec)

    def test_materialize_incremental_fetches_only_the_tail(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            full_payload = {
                "BTCUSDT": RawSymbolHistory(
                    symbol="BTCUSDT",
                    klines=[
                        kline(1735689600000 + (index * 300000), 100 + index, 101 + index, 99 + index, 100.5 + index, 1000 + index)
                        for index in range(5)
                    ],
                    funding_rates=[],
                    open_interest_stats=[
                        {"timestamp": 1735689600000 + (index * 300000), "sumOpenInterest": str(2500 + (index * 10))}
                        for index in range(5)
                    ],
                ),
                "ETHUSDT": RawSymbolHistory(
                    symbol="ETHUSDT",
                    klines=[
                        kline(1735689600000 + (index * 300000), 100 - index, 100.5 - index, 99 - index, 99.5 - index, 900 + index)
                        for index in range(5)
                    ],
                    funding_rates=[],
                    open_interest_stats=[
                        {"timestamp": 1735689600000 + (index * 300000), "sumOpenInterest": str(1800 + (index * 10))}
                        for index in range(5)
                    ],
                ),
            }
            client = SlicingHistoricalClient(full_payload)
            store = JsonDatasetStore(tempdir)
            materializer = HistoricalDatasetMaterializer(
                data_config=DataConfig(storage_root=tempdir, strict_validation=True),
                client=client,
                store=store,
                validator=DatasetValidator(),
                normalizer=BinanceBarNormalizer(),
            )
            initial_spec = DatasetSpec(
                exchange="binance",
                market="usdm_futures",
                timeframe="5m",
                start=self.start,
                end=self.start + timedelta(minutes=20),
                symbols=("BTCUSDT", "ETHUSDT"),
            )
            shifted_spec = DatasetSpec(
                exchange="binance",
                market="usdm_futures",
                timeframe="5m",
                start=self.start + timedelta(minutes=5),
                end=self.start + timedelta(minutes=25),
                symbols=("BTCUSDT", "ETHUSDT"),
            )

            initial_dataset = materializer.materialize(initial_spec)
            client.calls.clear()
            shifted_dataset = materializer.materialize_incremental(
                shifted_spec,
                initial_dataset.manifest_path,
            )

            self.assertTrue(shifted_dataset.manifest_path.exists())
            self.assertEqual(len(client.calls), 2)
            self.assertTrue(all(call[1] == initial_spec.end for call in client.calls))
            loader = ManifestHistoricalDataSource(
                manifest_path=shifted_dataset.manifest_path,
                store=store,
                manifest=shifted_dataset.manifest,
                storage_root=Path(tempdir),
            )
            loaded = loader.load_bars(shifted_spec)
            self.assertEqual(len(loaded["BTCUSDT"]), 4)
            self.assertEqual(loaded["BTCUSDT"][0].timestamp, self.start + timedelta(minutes=5))
            self.assertEqual(loaded["BTCUSDT"][-1].timestamp, self.start + timedelta(minutes=20))

    def test_manifest_discovery_prefers_covering_then_latest_reusable(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            store = JsonDatasetStore(tempdir)
            client = SlicingHistoricalClient(
                {
                    "BTCUSDT": RawSymbolHistory(
                        symbol="BTCUSDT",
                        klines=[
                            kline(1735689600000 + (index * 300000), 100 + index, 101 + index, 99 + index, 100.5 + index, 1000 + index)
                            for index in range(6)
                        ],
                    ),
                    "ETHUSDT": RawSymbolHistory(
                        symbol="ETHUSDT",
                        klines=[
                            kline(1735689600000 + (index * 300000), 100 - index, 100.5 - index, 99 - index, 99.5 - index, 900 + index)
                            for index in range(6)
                        ],
                    ),
                }
            )
            materializer = HistoricalDatasetMaterializer(
                data_config=DataConfig(storage_root=tempdir, strict_validation=True),
                client=client,
                store=store,
                validator=DatasetValidator(),
                normalizer=BinanceBarNormalizer(),
            )
            reusable_spec = DatasetSpec(
                exchange="binance",
                market="usdm_futures",
                timeframe="5m",
                start=self.start,
                end=self.start + timedelta(minutes=20),
                symbols=("BTCUSDT", "ETHUSDT"),
            )
            covering_spec = DatasetSpec(
                exchange="binance",
                market="usdm_futures",
                timeframe="5m",
                start=self.start,
                end=self.start + timedelta(minutes=30),
                symbols=("BTCUSDT", "ETHUSDT"),
            )
            requested_spec = DatasetSpec(
                exchange="binance",
                market="usdm_futures",
                timeframe="5m",
                start=self.start + timedelta(minutes=5),
                end=self.start + timedelta(minutes=25),
                symbols=("BTCUSDT", "ETHUSDT"),
            )
            reusable_dataset = materializer.materialize(reusable_spec)
            covering_dataset = materializer.materialize(covering_spec)

            self.assertEqual(
                find_covering_manifest(tempdir, requested_spec, store),
                covering_dataset.manifest_path,
            )
            self.assertEqual(
                find_latest_reusable_manifest(tempdir, requested_spec, store),
                reusable_dataset.manifest_path,
            )


if __name__ == "__main__":
    unittest.main()
