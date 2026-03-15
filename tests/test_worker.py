from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from autoresearch_trade_bot.config import DataConfig, ResearchTargetGate, WorkerConfig
from autoresearch_trade_bot.datasets import DatasetSpec
from autoresearch_trade_bot.models import Bar
from autoresearch_trade_bot.state import FilesystemResearchStateStore
from autoresearch_trade_bot.worker import ContinuousResearchWorker


def make_bar(
    symbol: str,
    timestamp: datetime,
    close: float,
) -> Bar:
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


def build_bars() -> dict[str, list[Bar]]:
    start = datetime(2026, 3, 14, 0, 0, tzinfo=timezone.utc)
    symbols = {
        "BTCUSDT": 100.0,
        "ETHUSDT": 80.0,
        "SOLUSDT": 60.0,
        "BNBUSDT": 120.0,
        "XRPUSDT": 140.0,
    }
    bars_by_symbol: dict[str, list[Bar]] = {}
    for symbol, start_price in symbols.items():
        bars: list[Bar] = []
        for index in range(60):
            timestamp = start + timedelta(minutes=5 * index)
            if symbol in {"BTCUSDT", "ETHUSDT", "SOLUSDT"}:
                close = start_price + (index * 1.4)
            else:
                close = start_price - (index * 1.1)
            bars.append(make_bar(symbol, timestamp, close))
        bars_by_symbol[symbol] = bars
    return bars_by_symbol


class FakeLoader:
    def __init__(self, bars_by_symbol, storage_root: Path) -> None:
        self._bars_by_symbol = bars_by_symbol
        self.storage_root = storage_root

    def load_bars(self, _spec: DatasetSpec):
        return self._bars_by_symbol


class FakeMaterializer:
    def __init__(self, manifest_path: Path) -> None:
        self.manifest_path = manifest_path
        self.calls = 0

    def materialize(self, _spec: DatasetSpec):
        self.calls += 1
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text("{}", encoding="utf-8")
        return SimpleNamespace(manifest_path=self.manifest_path)


class ContinuousResearchWorkerTests(unittest.TestCase):
    def test_run_cycle_persists_snapshot_history_and_artifact(self) -> None:
        bars_by_symbol = build_bars()
        with tempfile.TemporaryDirectory() as tempdir:
            temp_path = Path(tempdir)
            manifest_path = temp_path / "data" / "binance" / "usdm_futures" / "5m" / "cycle" / "manifest.json"
            worker_config = WorkerConfig(
                symbols=tuple(sorted(bars_by_symbol)),
                timeframe="5m",
                history_window_days=1,
                max_variants_per_cycle=6,
                recent_cycles_for_acceptance=1,
                state_root=str(temp_path / "state"),
                artifact_root=str(temp_path / "artifacts"),
                data_config=DataConfig(storage_root=str(temp_path / "data")),
                target_gate=ResearchTargetGate(
                    min_total_return=0.0,
                    min_sharpe=0.3,
                    max_drawdown=0.35,
                    min_acceptance_rate=1.0,
                ),
            )
            worker = ContinuousResearchWorker(
                worker_config=worker_config,
                state_store=FilesystemResearchStateStore(worker_config.state_root),
                materializer=FakeMaterializer(manifest_path),
                now_fn=lambda: datetime(2026, 3, 15, 0, 5, tzinfo=timezone.utc),
                sleep_fn=lambda _seconds: None,
            )

            with patch(
                "autoresearch_trade_bot.worker.ManifestHistoricalDataSource.from_manifest_path",
                return_value=FakeLoader(bars_by_symbol, temp_path / "data"),
            ):
                result = worker.run_cycle()

            self.assertTrue(result.best_entry.accepted)
            snapshot_path = temp_path / "state" / "latest_status.json"
            history_path = temp_path / "state" / "history.json"
            artifact_path = temp_path / "artifacts" / "research-cycles" / f"{result.cycle_id}.json"
            self.assertTrue(snapshot_path.exists())
            self.assertTrue(history_path.exists())
            self.assertTrue(artifact_path.exists())
            snapshot_text = snapshot_path.read_text(encoding="utf-8")
            self.assertIn('"loop_state": "holding"', snapshot_text)
            self.assertIn('"latest_dataset_id"', snapshot_text)
            self.assertIn('"leaderboard"', snapshot_text)

    def test_latest_closed_bar_and_sleep_align_to_timeframe(self) -> None:
        worker = ContinuousResearchWorker(
            worker_config=WorkerConfig(),
            state_store=FilesystemResearchStateStore(Path(tempfile.gettempdir()) / "unused-state"),
            materializer=FakeMaterializer(Path(tempfile.gettempdir()) / "unused" / "manifest.json"),
            now_fn=lambda: datetime(2026, 3, 15, 12, 3, tzinfo=timezone.utc),
            sleep_fn=lambda _seconds: None,
        )
        now = datetime(2026, 3, 15, 12, 3, tzinfo=timezone.utc)
        self.assertEqual(
            worker.latest_closed_bar(now),
            datetime(2026, 3, 15, 12, 0, tzinfo=timezone.utc),
        )
        self.assertGreater(worker.seconds_until_next_cycle(now), 120.0)


if __name__ == "__main__":
    unittest.main()
