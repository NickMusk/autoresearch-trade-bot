from __future__ import annotations

import io
import json
import tempfile
import unittest
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from autoresearch_trade_bot.crypto_import import (
    _period_sort_key,
    _row_to_bar,
    _timestamp_to_datetime,
    discover_crypto_data_files,
    import_crypto_data_directory,
)
from autoresearch_trade_bot.data import ManifestHistoricalDataSource
from autoresearch_trade_bot.datasets import DatasetSpec


def sample_row(
    open_time: int,
    open_: str = "100.0",
    high: str = "101.0",
    low: str = "99.0",
    close: str = "100.5",
    volume: str = "10.0",
    close_time: int | None = None,
) -> str:
    resolved_close_time = close_time if close_time is not None else open_time + 299_999_999
    return ",".join(
        [
            str(open_time),
            open_,
            high,
            low,
            close,
            volume,
            str(resolved_close_time),
            "1000.0",
            "42",
            "4.0",
            "400.0",
            "0",
        ]
    )


class CryptoImportTests(unittest.TestCase):
    def test_timestamp_parser_supports_microseconds(self) -> None:
        parsed = _timestamp_to_datetime(1735689600000000)
        self.assertEqual(parsed, datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc))

    def test_row_to_bar_maps_binance_like_kline_schema(self) -> None:
        bar = _row_to_bar(
            row=sample_row(1735689600000000).split(","),
            exchange="binance",
            symbol="BTCUSDT",
            timeframe="5m",
        )
        self.assertEqual(bar.timestamp, datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc))
        self.assertEqual(bar.trade_count, 42)
        self.assertEqual(bar.mark_price, bar.close)
        self.assertEqual(bar.index_price, bar.close)
        self.assertEqual(bar.funding_rate, 0.0)
        self.assertEqual(bar.open_interest, 0.0)

    def test_discover_files_prefers_csv_over_zip_for_same_period(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            (root / "BTCUSDT-5m-2025-01.csv").write_text(sample_row(1735689600000000), encoding="utf-8")
            with zipfile.ZipFile(root / "BTCUSDT-5m-2025-01.zip", "w") as archive:
                archive.writestr("BTCUSDT-5m-2025-01.csv", sample_row(1735689600000000))

            files = discover_crypto_data_files(
                source_root=root,
                symbols=("BTCUSDT",),
                timeframe="5m",
            )

            self.assertEqual(len(files), 1)
            self.assertEqual(files[0].path.suffix, ".csv")

    def test_period_sort_key_accepts_compact_month_names(self) -> None:
        rank, value = _period_sort_key("2026-1")
        self.assertEqual(rank, 1)
        self.assertEqual(value, datetime(2026, 1, 1, tzinfo=timezone.utc))

    def test_import_directory_skips_invalid_zip_and_loads_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            source_root = root / "crypto_data"
            storage_root = root / "storage"
            source_root.mkdir()

            btc_csv = "\n".join(
                [
                    sample_row(1735689600000000, close="100.5"),
                    sample_row(1735689900000000, open_="100.5", high="102.0", low="100.0", close="101.0"),
                ]
            )
            eth_csv = "\n".join(
                [
                    sample_row(1735689600000000, open_="80.0", high="81.0", low="79.0", close="80.5"),
                    sample_row(1735689900000000, open_="80.5", high="82.0", low="80.0", close="81.0"),
                ]
            )
            (source_root / "BTCUSDT-5m-2025-01.csv").write_text(btc_csv, encoding="utf-8")
            (source_root / "ETHUSDT-5m-2025-01.csv").write_text(eth_csv, encoding="utf-8")
            (source_root / "BTCUSDT-5m-2025-02.zip").write_text(
                "<?xml version='1.0'?><error>not a zip</error>",
                encoding="utf-8",
            )

            warnings = io.StringIO()
            datasets = import_crypto_data_directory(
                source_root=source_root,
                storage_root=storage_root,
                symbols=("BTCUSDT", "ETHUSDT"),
                timeframe="5m",
                exchange="binance",
                market="usdm_futures",
                warning_stream=warnings,
            )

            self.assertEqual(len(datasets), 1)
            dataset = datasets[0]
            self.assertTrue(dataset.manifest_path.exists())
            self.assertIn("invalid zip archive", warnings.getvalue())

            loader = ManifestHistoricalDataSource.from_manifest_path(dataset.manifest_path)
            spec = DatasetSpec(
                exchange="binance",
                market="usdm_futures",
                timeframe="5m",
                start=datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc),
                end=datetime(2025, 1, 1, 0, 10, tzinfo=timezone.utc),
                symbols=("BTCUSDT", "ETHUSDT"),
            )
            bars = loader.load_bars(spec)

            self.assertEqual(set(bars), {"BTCUSDT", "ETHUSDT"})
            self.assertEqual(len(bars["BTCUSDT"]), 2)
            self.assertEqual(len(bars["ETHUSDT"]), 2)
            manifest = json.loads(dataset.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(tuple(manifest["symbols"]), ("BTCUSDT", "ETHUSDT"))

    def test_import_directory_splits_discontinuous_ranges_into_multiple_datasets(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            source_root = root / "crypto_data"
            storage_root = root / "storage"
            source_root.mkdir()

            january = "\n".join(
                [
                    sample_row(1735689600000000, close="100.5"),
                    sample_row(1735689900000000, open_="100.5", high="102.0", low="100.0", close="101.0"),
                ]
            )
            march = "\n".join(
                [
                    sample_row(1740787200000000, open_="200.0", high="201.0", low="199.0", close="200.5"),
                    sample_row(1740787500000000, open_="200.5", high="202.0", low="200.0", close="201.0"),
                ]
            )
            for symbol in ("BTCUSDT", "ETHUSDT"):
                (source_root / f"{symbol}-5m-2025-01.csv").write_text(january, encoding="utf-8")
                (source_root / f"{symbol}-5m-2025-03-01.csv").write_text(march, encoding="utf-8")

            datasets = import_crypto_data_directory(
                source_root=source_root,
                storage_root=storage_root,
                symbols=("BTCUSDT", "ETHUSDT"),
                timeframe="5m",
                exchange="binance",
                market="usdm_futures",
            )

            self.assertEqual(len(datasets), 2)
            manifests = [json.loads(dataset.manifest_path.read_text(encoding="utf-8")) for dataset in datasets]
            self.assertEqual(manifests[0]["start"], "2025-01-01T00:00:00+00:00")
            self.assertEqual(manifests[1]["start"], "2025-03-01T00:00:00+00:00")


if __name__ == "__main__":
    unittest.main()
