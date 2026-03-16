from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from autoresearch_trade_bot.config import DataConfig, ResearchTargetGate, WorkerConfig
from autoresearch_trade_bot.datasets import DatasetManifest, DatasetSpec
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
    def __init__(self, bars_by_symbol, storage_root: Path, manifest_path: Path | None = None) -> None:
        self._bars_by_symbol = bars_by_symbol
        self.storage_root = storage_root
        self.manifest_path = manifest_path or storage_root / "manifest.json"

    def load_bars(self, _spec: DatasetSpec):
        return self._bars_by_symbol


class FakeMaterializer:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.store = SimpleNamespace(
            read_manifest=lambda path: DatasetManifest.from_dict(
                __import__("json").loads(Path(path).read_text(encoding="utf-8"))
            )
        )
        self.calls = 0

    def materialize(self, spec: DatasetSpec):
        self.calls += 1
        manifest_path = (
            self.root
            / spec.exchange
            / spec.market
            / spec.timeframe
            / spec.dataset_id
            / "manifest.json"
        )
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = DatasetManifest(
            dataset_id=spec.dataset_id,
            exchange=spec.exchange,
            market=spec.market,
            timeframe=spec.timeframe,
            start=spec.start,
            end=spec.end,
            generated_at=datetime.now(timezone.utc),
            symbols=spec.symbols,
            files={},
            validation_issues=[],
        )
        manifest_path.write_text(__import__("json").dumps(manifest.to_dict()), encoding="utf-8")
        return SimpleNamespace(manifest_path=manifest_path)

    def materialize_incremental(self, spec: DatasetSpec, _base_manifest_path: Path):
        return self.materialize(spec)


class ContinuousResearchWorkerTests(unittest.TestCase):
    def test_run_cycle_persists_snapshot_history_and_artifact(self) -> None:
        bars_by_symbol = build_bars()
        with tempfile.TemporaryDirectory() as tempdir:
            temp_path = Path(tempdir)
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
                materializer=FakeMaterializer(temp_path / "data"),
                now_fn=lambda: datetime(2026, 3, 15, 0, 5, tzinfo=timezone.utc),
                sleep_fn=lambda _seconds: None,
            )

            with patch(
                "autoresearch_trade_bot.worker.ManifestHistoricalDataSource.from_manifest_path",
                return_value=FakeLoader(
                    bars_by_symbol,
                    temp_path / "data",
                    manifest_path=temp_path / "data" / "manifest.json",
                ),
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
            self.assertIn('"multi_window_summary"', snapshot_text)
            self.assertIn('"leaderboard"', snapshot_text)

    def test_run_cycle_reuses_previous_multi_window_summary_until_interval_is_due(self) -> None:
        bars_by_symbol = build_bars()
        with tempfile.TemporaryDirectory() as tempdir:
            temp_path = Path(tempdir)
            worker_config = WorkerConfig(
                symbols=tuple(sorted(bars_by_symbol)),
                timeframe="5m",
                history_window_days=1,
                max_variants_per_cycle=6,
                multi_window_validation_interval_bars=12,
                recent_cycles_for_acceptance=2,
                state_root=str(temp_path / "state"),
                artifact_root=str(temp_path / "artifacts"),
                data_config=DataConfig(storage_root=str(temp_path / "data")),
            )
            worker = ContinuousResearchWorker(
                worker_config=worker_config,
                state_store=FilesystemResearchStateStore(worker_config.state_root),
                materializer=FakeMaterializer(temp_path / "data"),
                now_fn=lambda: datetime(2026, 3, 15, 0, 5, tzinfo=timezone.utc),
                sleep_fn=lambda _seconds: None,
            )

            with patch(
                "autoresearch_trade_bot.worker.ManifestHistoricalDataSource.from_manifest_path",
                return_value=FakeLoader(bars_by_symbol, temp_path / "data"),
            ):
                first_result = worker.run_cycle()

            self.assertTrue(first_result.ran_multi_window_validation)

            second_worker = ContinuousResearchWorker(
                worker_config=worker_config,
                state_store=FilesystemResearchStateStore(worker_config.state_root),
                materializer=FakeMaterializer(temp_path / "data"),
                now_fn=lambda: datetime(2026, 3, 15, 0, 10, tzinfo=timezone.utc),
                sleep_fn=lambda _seconds: None,
            )

            with patch(
                "autoresearch_trade_bot.worker.ManifestHistoricalDataSource.from_manifest_path",
                return_value=FakeLoader(bars_by_symbol, temp_path / "data"),
            ), patch.object(
                ContinuousResearchWorker,
                "_run_multi_window_validation",
                side_effect=AssertionError("multi-window should not run before interval is due"),
            ):
                second_result = second_worker.run_cycle()

            self.assertFalse(second_result.ran_multi_window_validation)
            snapshot = FilesystemResearchStateStore(worker_config.state_root).load_snapshot()
            self.assertIsNotNone(snapshot)
            self.assertEqual(snapshot.multi_window_summary["validation_interval_bars"], 12)
            self.assertFalse(snapshot.multi_window_summary["ran_this_cycle"])

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
