from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from autoresearch_trade_bot.bybit import BybitBarNormalizer
from autoresearch_trade_bot.config import DataConfig
from autoresearch_trade_bot.data import HistoricalDatasetMaterializer, ManifestHistoricalDataSource
from autoresearch_trade_bot.datasets import DatasetManifest, DatasetSpec, RawSymbolHistory
from autoresearch_trade_bot.storage import DatasetStore
from autoresearch_trade_bot.validation import DatasetValidator


class FakeHistoricalClient:
    def __init__(self, payload_by_symbol) -> None:
        self.payload_by_symbol = payload_by_symbol

    def fetch_symbol_history(self, spec: DatasetSpec, symbol: str) -> RawSymbolHistory:
        return self.payload_by_symbol[symbol]


class JsonDatasetStore(DatasetStore):
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def write_bars(self, spec: DatasetSpec, symbol: str, bars) -> Path:
        dataset_dir = self.root / spec.exchange / spec.market / spec.timeframe / spec.dataset_id
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
                mark_price=float(item["mark_price"]) if item["mark_price"] is not None else None,
                index_price=float(item["index_price"]) if item["index_price"] is not None else None,
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


class BybitDataPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.start = datetime(2026, 3, 1, 0, 0, tzinfo=timezone.utc)
        self.end = self.start + timedelta(minutes=15)
        self.start_ms = int(self.start.timestamp() * 1000)
        self.spec = DatasetSpec(
            exchange="bybit",
            market="linear",
            timeframe="5m",
            start=self.start,
            end=self.end,
            symbols=("BTCUSDT", "ETHUSDT"),
        )

    def test_bybit_normalizer_maps_funding_open_interest_and_prices(self) -> None:
        raw = RawSymbolHistory(
            symbol="BTCUSDT",
            klines=[
                [str(self.start_ms), "100", "101", "99", "100.5", "1500", "150000"],
                [str(self.start_ms + 300000), "100.5", "103", "100", "102.5", "1700", "170000"],
                [str(self.start_ms + 600000), "102.5", "106", "102", "105.5", "1900", "190000"],
            ],
            funding_rates=[
                {"fundingRateTimestamp": str(self.start_ms + 600000), "fundingRate": "0.0001"},
            ],
            open_interest_stats=[
                {"timestamp": str(self.start_ms), "openInterest": "2500"},
                {"timestamp": str(self.start_ms + 300000), "openInterest": "2550"},
                {"timestamp": str(self.start_ms + 600000), "openInterest": "2600"},
            ],
            mark_price_klines=[
                [str(self.start_ms), "100", "101", "99", "100.4"],
                [str(self.start_ms + 300000), "100.5", "103", "100", "102.4"],
                [str(self.start_ms + 600000), "102.5", "106", "102", "105.4"],
            ],
            index_price_klines=[
                [str(self.start_ms), "100", "101", "99", "100.3"],
                [str(self.start_ms + 300000), "100.5", "103", "100", "102.3"],
                [str(self.start_ms + 600000), "102.5", "106", "102", "105.3"],
            ],
        )

        bars = BybitBarNormalizer().normalize(self.spec, raw)
        self.assertEqual(len(bars), 3)
        self.assertEqual(bars[0].exchange, "bybit")
        self.assertEqual(bars[2].funding_rate, 0.0001)
        self.assertEqual(bars[1].open_interest, 2550.0)
        self.assertEqual(bars[2].mark_price, 105.4)
        self.assertEqual(bars[2].index_price, 105.3)
        self.assertEqual(
            bars[0].close_time,
            datetime(2026, 3, 1, 0, 4, 59, 999000, tzinfo=timezone.utc),
        )

    def test_materialize_and_load_bybit_dataset(self) -> None:
        payload = {
            "BTCUSDT": RawSymbolHistory(
                symbol="BTCUSDT",
                klines=[
                    [str(self.start_ms), "100", "101", "99", "100.5", "1500", "150000"],
                    [str(self.start_ms + 300000), "100.5", "103", "100", "102.5", "1700", "170000"],
                    [str(self.start_ms + 600000), "102.5", "106", "102", "105.5", "1900", "190000"],
                ],
                funding_rates=[
                    {"fundingRateTimestamp": str(self.start_ms + 600000), "fundingRate": "0.0001"},
                ],
                open_interest_stats=[
                    {"timestamp": str(self.start_ms), "openInterest": "2500"},
                    {"timestamp": str(self.start_ms + 300000), "openInterest": "2550"},
                    {"timestamp": str(self.start_ms + 600000), "openInterest": "2600"},
                ],
                mark_price_klines=[
                    [str(self.start_ms), "100", "101", "99", "100.4"],
                    [str(self.start_ms + 300000), "100.5", "103", "100", "102.4"],
                    [str(self.start_ms + 600000), "102.5", "106", "102", "105.4"],
                ],
                index_price_klines=[
                    [str(self.start_ms), "100", "101", "99", "100.3"],
                    [str(self.start_ms + 300000), "100.5", "103", "100", "102.3"],
                    [str(self.start_ms + 600000), "102.5", "106", "102", "105.3"],
                ],
            ),
            "ETHUSDT": RawSymbolHistory(
                symbol="ETHUSDT",
                klines=[
                    [str(self.start_ms), "80", "80.5", "79", "79.5", "900", "70000"],
                    [str(self.start_ms + 300000), "79.5", "79.8", "78.5", "78.9", "950", "72000"],
                    [str(self.start_ms + 600000), "78.9", "79.2", "77.5", "77.8", "970", "73000"],
                ],
                open_interest_stats=[
                    {"timestamp": str(self.start_ms), "openInterest": "1800"},
                    {"timestamp": str(self.start_ms + 300000), "openInterest": "1820"},
                    {"timestamp": str(self.start_ms + 600000), "openInterest": "1850"},
                ],
                mark_price_klines=[
                    [str(self.start_ms), "80", "80.5", "79", "79.4"],
                    [str(self.start_ms + 300000), "79.5", "79.8", "78.5", "78.8"],
                    [str(self.start_ms + 600000), "78.9", "79.2", "77.5", "77.7"],
                ],
                index_price_klines=[
                    [str(self.start_ms), "80", "80.5", "79", "79.3"],
                    [str(self.start_ms + 300000), "79.5", "79.8", "78.5", "78.7"],
                    [str(self.start_ms + 600000), "78.9", "79.2", "77.5", "77.6"],
                ],
            ),
        }
        with tempfile.TemporaryDirectory() as tempdir:
            data_config = DataConfig(
                exchange="bybit",
                market="linear",
                storage_root=tempdir,
                strict_validation=True,
            )
            materializer = HistoricalDatasetMaterializer(
                data_config=data_config,
                client=FakeHistoricalClient(payload),
                store=JsonDatasetStore(tempdir),
                validator=DatasetValidator(),
                normalizer=BybitBarNormalizer(),
            )
            dataset = materializer.materialize(self.spec)
            loader = ManifestHistoricalDataSource(
                manifest_path=dataset.manifest_path,
                store=JsonDatasetStore(tempdir),
                manifest=dataset.manifest,
                storage_root=Path(tempdir),
            )
            loaded = loader.load_bars(self.spec)
            self.assertEqual(set(loaded), {"BTCUSDT", "ETHUSDT"})
            self.assertEqual(len(loaded["BTCUSDT"]), 3)
            self.assertEqual(loaded["BTCUSDT"][2].funding_rate, 0.0001)
            self.assertEqual(loaded["ETHUSDT"][0].open_interest, 1800.0)
            self.assertEqual(loaded["ETHUSDT"][1].mark_price, 78.8)

    def test_for_exchange_builds_bybit_materializer(self) -> None:
        materializer = HistoricalDatasetMaterializer.for_exchange(
            DataConfig(exchange="bybit", market="linear", storage_root="data")
        )
        self.assertEqual(materializer.client.__class__.__name__, "BybitLinearHistoricalClient")
        self.assertEqual(materializer.normalizer.__class__.__name__, "BybitBarNormalizer")


if __name__ == "__main__":
    unittest.main()
