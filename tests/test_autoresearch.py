from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from autoresearch_trade_bot.autoresearch import (
    AutoresearchRunReport,
    FrozenResearchWindow,
    GitAutoresearchRunner,
    append_results_row,
    evaluate_train_file,
    prepare_campaign,
)
from autoresearch_trade_bot.binance import BinanceBarNormalizer
from autoresearch_trade_bot.config import DataConfig, ResearchTargetGate
from autoresearch_trade_bot.data import HistoricalDatasetMaterializer
from autoresearch_trade_bot.datasets import DatasetSpec, RawSymbolHistory
from autoresearch_trade_bot.storage import PyArrowParquetDatasetStore
from autoresearch_trade_bot.validation import DatasetValidator


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


class SlicingHistoricalClient:
    def __init__(self, payload_by_symbol) -> None:
        self.payload_by_symbol = payload_by_symbol

    def fetch_symbol_history(self, spec: DatasetSpec, symbol: str) -> RawSymbolHistory:
        source = self.payload_by_symbol[symbol]
        start_ms = int(spec.start.timestamp() * 1000)
        end_ms = int(spec.end.timestamp() * 1000)
        return RawSymbolHistory(
            symbol=symbol,
            klines=[row for row in source.klines if start_ms <= int(row[0]) < end_ms],
            funding_rates=[],
            open_interest_stats=[],
        )


class RecordingMaterializer:
    def __init__(self) -> None:
        self.specs: list[DatasetSpec] = []

    def materialize(self, spec: DatasetSpec):
        self.specs.append(spec)
        path = Path(tempfile.gettempdir()) / f"{spec.dataset_id}.json"
        path.write_text("{}", encoding="utf-8")
        return SimpleNamespace(manifest_path=path)


class AutoresearchTests(unittest.TestCase):
    def test_prepare_campaign_writes_frozen_windows(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            materializer = RecordingMaterializer()
            campaign_path = Path(tempdir) / ".autoresearch" / "campaign.json"
            path = prepare_campaign(
                campaign_name="unit-campaign",
                exchange="bybit",
                market="linear",
                timeframe="5m",
                symbols=("BTCUSDT", "ETHUSDT"),
                anchor_end=datetime(2026, 3, 17, 0, 0, tzinfo=timezone.utc),
                window_days=7,
                window_count=2,
                storage_root=str(Path(tempdir) / "data"),
                campaign_path=campaign_path,
                materializer=materializer,
                target_gate=ResearchTargetGate(),
            )

            self.assertEqual(path, campaign_path)
            payload = json.loads(campaign_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["name"], "unit-campaign")
            self.assertEqual(len(payload["windows"]), 2)
            self.assertEqual(len(materializer.specs), 2)
            self.assertEqual(
                materializer.specs[0].start,
                datetime(2026, 3, 10, 0, 0, tzinfo=timezone.utc),
            )
            self.assertEqual(
                materializer.specs[1].start,
                datetime(2026, 3, 3, 0, 0, tzinfo=timezone.utc),
            )

    def test_evaluate_train_file_writes_artifact_and_results_row(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            temp_path = Path(tempdir)
            start = datetime(2026, 3, 1, 0, 0, tzinfo=timezone.utc)
            payload = {
                "BTCUSDT": RawSymbolHistory(
                    symbol="BTCUSDT",
                    klines=[
                        kline(
                            int((start + timedelta(minutes=5 * index)).timestamp() * 1000),
                            100 + index,
                            101 + index,
                            99 + index,
                            100.5 + index,
                            1000 + index,
                        )
                        for index in range(24)
                    ],
                ),
                "ETHUSDT": RawSymbolHistory(
                    symbol="ETHUSDT",
                    klines=[
                        kline(
                            int((start + timedelta(minutes=5 * index)).timestamp() * 1000),
                            100 - index,
                            100.5 - index,
                            99 - index,
                            99.5 - index,
                            900 + index,
                        )
                        for index in range(24)
                    ],
                ),
            }
            materializer = HistoricalDatasetMaterializer(
                data_config=DataConfig(storage_root=str(temp_path / "data"), strict_validation=True),
                client=SlicingHistoricalClient(payload),
                store=PyArrowParquetDatasetStore(temp_path / "data"),
                validator=DatasetValidator(),
                normalizer=BinanceBarNormalizer(),
            )
            specs = [
                DatasetSpec(
                    exchange="binance",
                    market="usdm_futures",
                    timeframe="5m",
                    start=start,
                    end=start + timedelta(minutes=60),
                    symbols=("BTCUSDT", "ETHUSDT"),
                ),
                DatasetSpec(
                    exchange="binance",
                    market="usdm_futures",
                    timeframe="5m",
                    start=start + timedelta(minutes=60),
                    end=start + timedelta(minutes=120),
                    symbols=("BTCUSDT", "ETHUSDT"),
                ),
            ]
            windows = []
            for spec in specs:
                dataset = materializer.materialize(spec)
                windows.append(
                    FrozenResearchWindow(
                        dataset_id=spec.dataset_id,
                        manifest_path=str(dataset.manifest_path),
                        start=spec.start.isoformat(),
                        end=spec.end.isoformat(),
                    )
                )

            campaign_path = temp_path / ".autoresearch" / "campaign.json"
            campaign_path.parent.mkdir(parents=True, exist_ok=True)
            campaign_path.write_text(
                json.dumps(
                    {
                        "name": "frozen-campaign",
                        "exchange": "binance",
                        "market": "usdm_futures",
                        "timeframe": "5m",
                        "symbols": ["BTCUSDT", "ETHUSDT"],
                        "storage_root": str(temp_path / "data"),
                        "target_gate": {
                            "min_total_return": 0.0,
                            "min_sharpe": 1.0,
                            "max_drawdown": 0.2,
                            "min_acceptance_rate": 0.6,
                        },
                        "windows": [window.to_dict() for window in windows],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            train_path = temp_path / "train.py"
            train_path.write_text(
                "\n".join(
                    [
                        "from autoresearch_trade_bot.strategy import CrossSectionalMomentumStrategy",
                        'STRATEGY_NAME = "unit-train"',
                        "",
                        "def build_strategy():",
                        "    return CrossSectionalMomentumStrategy(lookback_bars=2, top_k=1, gross_target=1.0)",
                    ]
                ),
                encoding="utf-8",
            )

            report = evaluate_train_file(
                campaign_path=campaign_path,
                train_path=train_path,
                artifact_root=temp_path / ".autoresearch" / "runs",
            )
            results_path = append_results_row(
                results_path=temp_path / "results.tsv",
                report=report,
                decision="observed",
            )

            self.assertEqual(report.strategy_name, "unit-train")
            self.assertEqual(report.total_windows, 2)
            self.assertTrue(Path(report.artifact_path).exists())
            self.assertTrue(results_path.exists())
            self.assertIn("research_score", results_path.read_text(encoding="utf-8"))

    def test_git_runner_keeps_improvement_and_discards_regression(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            repo_root = Path(tempdir)
            self._git(repo_root, "init")
            self._git(repo_root, "config", "user.email", "bot@example.com")
            self._git(repo_root, "config", "user.name", "Bot")
            (repo_root / "train.py").write_text("BASELINE = 1\n", encoding="utf-8")
            self._git(repo_root, "add", "train.py")
            self._git(repo_root, "commit", "-m", "Initial train file")

            def fake_evaluator(*, campaign_path, train_path, artifact_root):
                _ = campaign_path
                artifact_dir = Path(artifact_root)
                artifact_dir.mkdir(parents=True, exist_ok=True)
                train_text = Path(train_path).read_text(encoding="utf-8")
                score = 2.0 if "IMPROVE" in train_text else 0.5
                artifact_path = artifact_dir / f"{score}.json"
                artifact_path.write_text("{}", encoding="utf-8")
                return AutoresearchRunReport(
                    run_id=f"run-{score}",
                    recorded_at=datetime.now(timezone.utc).isoformat(),
                    campaign_name="fake-campaign",
                    strategy_name="fake-strategy",
                    git_branch="",
                    git_commit="",
                    train_file=str(train_path),
                    train_sha1="sha",
                    research_score=score,
                    acceptance_rate=0.0,
                    average_metrics={
                        "total_return": score,
                        "sharpe": score,
                        "max_drawdown": 0.1,
                        "average_turnover": 0.1,
                        "bars_processed": 10,
                    },
                    worst_max_drawdown=0.1,
                    ready_for_paper=False,
                    gate_failures=["acceptance_rate_below_gate"],
                    windows_passed=0,
                    total_windows=2,
                    window_reports=[],
                    artifact_path=str(artifact_path),
                )

            runner = GitAutoresearchRunner(
                repo_root=repo_root,
                branch_name="codex/autoresearch-test",
                evaluator=fake_evaluator,
            )

            keep = runner.apply_candidate(
                campaign_path=repo_root / ".autoresearch" / "campaign.json",
                candidate_text="IMPROVE = 1\n",
                commit_message="Keep improvement",
            )
            self.assertEqual(keep.decision, "keep")
            self.assertIsNotNone(keep.kept_commit)
            self.assertEqual((repo_root / "train.py").read_text(encoding="utf-8"), "IMPROVE = 1\n")
            changed_files = self._git(
                repo_root,
                "show",
                "--name-only",
                "--format=",
                "HEAD",
            ).splitlines()
            self.assertEqual(changed_files, ["train.py"])

            discard = runner.apply_candidate(
                campaign_path=repo_root / ".autoresearch" / "campaign.json",
                candidate_text="REGRESS = 1\n",
                commit_message="Discard regression",
            )
            self.assertEqual(discard.decision, "discard")
            self.assertEqual((repo_root / "train.py").read_text(encoding="utf-8"), "IMPROVE = 1\n")
            results_tsv = (repo_root / "results.tsv").read_text(encoding="utf-8")
            self.assertIn("keep", results_tsv)
            self.assertIn("discard", results_tsv)

    @staticmethod
    def _git(repo_root: Path, *args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()


if __name__ == "__main__":
    unittest.main()
