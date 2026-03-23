from __future__ import annotations

import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from autoresearch_trade_bot.history_dataset_install import (
    DatasetInstallConfig,
    ensure_history_dataset_installed,
)


class HistoryDatasetInstallTests(unittest.TestCase):
    def test_installs_archive_and_writes_readiness_state(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            temp_path = Path(tempdir)
            archive_path = self._build_dataset_archive(temp_path / "source")
            storage_root = temp_path / "storage"
            readiness_state_path = temp_path / "history_refresh_state.json"
            manifest_relative_path = (
                "binance/usdm_futures/5m/"
                "dataset-20250101-20260322/manifest.json"
            )

            result = ensure_history_dataset_installed(
                DatasetInstallConfig(
                    archive_url=archive_path.as_uri(),
                    storage_root=str(storage_root),
                    manifest_relative_path=manifest_relative_path,
                    readiness_state_path=str(readiness_state_path),
                )
            )

            self.assertTrue(result.installed)
            assert result.manifest_path is not None
            self.assertTrue(result.manifest_path.exists())
            state = json.loads(readiness_state_path.read_text(encoding="utf-8"))
            self.assertEqual(state["manifest_path"], str(result.manifest_path))
            self.assertEqual(state["storage_root"], str(storage_root))
            self.assertEqual(state["lookback_days"], 445)
            self.assertEqual(state["source"], "dataset_install")

    def test_existing_manifest_is_no_op_and_still_refreshes_marker(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            temp_path = Path(tempdir)
            storage_root = temp_path / "storage"
            manifest_relative_path = (
                "binance/usdm_futures/5m/"
                "dataset-20250101-20260322/manifest.json"
            )
            manifest_path = storage_root / manifest_relative_path
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(
                json.dumps(
                    {
                        "dataset_id": "dataset-20250101-20260322",
                        "exchange": "binance",
                        "market": "usdm_futures",
                        "timeframe": "5m",
                        "start": "2025-01-01T00:00:00+00:00",
                        "end": "2026-03-22T00:00:00+00:00",
                        "generated_at": "2026-03-23T00:00:00+00:00",
                        "symbols": ["BTCUSDT"],
                        "files": {},
                        "validation_issues": [],
                    }
                ),
                encoding="utf-8",
            )
            readiness_state_path = temp_path / "history_refresh_state.json"

            result = ensure_history_dataset_installed(
                DatasetInstallConfig(
                    archive_url="file:///definitely-not-used.tar.gz",
                    storage_root=str(storage_root),
                    manifest_relative_path=manifest_relative_path,
                    readiness_state_path=str(readiness_state_path),
                )
            )

            self.assertFalse(result.installed)
            self.assertEqual(result.manifest_path, manifest_path)
            state = json.loads(readiness_state_path.read_text(encoding="utf-8"))
            self.assertEqual(state["manifest_path"], str(manifest_path))

    def _build_dataset_archive(self, source_root: Path) -> Path:
        dataset_root = source_root / "binance" / "usdm_futures" / "5m" / "dataset-20250101-20260322"
        dataset_root.mkdir(parents=True, exist_ok=True)
        (dataset_root / "BTCUSDT.parquet").write_bytes(b"PAR1")
        (dataset_root / "ETHUSDT.parquet").write_bytes(b"PAR1")
        (dataset_root / "manifest.json").write_text(
            json.dumps(
                {
                    "dataset_id": "dataset-20250101-20260322",
                    "exchange": "binance",
                    "market": "usdm_futures",
                    "timeframe": "5m",
                    "start": "2025-01-01T00:00:00+00:00",
                    "end": "2026-03-22T00:00:00+00:00",
                    "generated_at": "2026-03-23T00:00:00+00:00",
                    "symbols": ["BTCUSDT", "ETHUSDT"],
                    "files": {},
                    "validation_issues": [],
                }
            ),
            encoding="utf-8",
        )
        archive_path = source_root.parent / "dataset.tar.gz"
        with tarfile.open(archive_path, mode="w:gz") as archive:
            archive.add(source_root / "binance", arcname="binance")
        return archive_path


if __name__ == "__main__":
    unittest.main()
