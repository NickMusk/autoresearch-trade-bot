from __future__ import annotations

import ast
import json
import subprocess
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from autoresearch_trade_bot.autoresearch import (
    AutoresearchRunReport,
    DeterministicTrainMutator,
    FrozenResearchWindow,
    GitAutoresearchRunner,
    append_results_row,
    campaign_path_for_name,
    evaluate_train_file,
    prepare_campaign,
    render_train_file,
    resolve_campaign_path,
    run_deterministic_mutation_campaign,
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

    store = SimpleNamespace(read_manifest=lambda _path: None)


class AutoresearchTests(unittest.TestCase):
    def test_prepare_campaign_writes_named_frozen_windows_and_active_pointer(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            temp_path = Path(tempdir)
            materializer = RecordingMaterializer()
            campaigns_root = temp_path / ".autoresearch" / "campaigns"
            active_pointer = temp_path / ".autoresearch" / "active_campaign.txt"
            path = prepare_campaign(
                campaign_name="Bybit 5m Mar17",
                exchange="bybit",
                market="linear",
                timeframe="5m",
                symbols=("BTCUSDT", "ETHUSDT"),
                anchor_end=datetime(2026, 3, 17, 0, 0, tzinfo=timezone.utc),
                window_days=7,
                window_count=2,
                storage_root=str(temp_path / "data"),
                campaigns_root=campaigns_root,
                active_pointer_path=active_pointer,
                materializer=materializer,
                target_gate=ResearchTargetGate(),
            )

            self.assertEqual(path, campaign_path_for_name("Bybit 5m Mar17", campaigns_root))
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["campaign_id"], "bybit-5m-mar17")
            self.assertEqual(len(payload["windows"]), 2)
            self.assertEqual(Path(active_pointer.read_text(encoding="utf-8").strip()), path.resolve())
            self.assertEqual(resolve_campaign_path(None, pointer_path=active_pointer), path.resolve())
            self.assertEqual(len(materializer.specs), 2)

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

            campaign_path = temp_path / ".autoresearch" / "campaigns" / "unit.json"
            campaign_path.parent.mkdir(parents=True, exist_ok=True)
            campaign_path.write_text(
                json.dumps(
                    {
                        "campaign_id": "unit",
                        "name": "unit-campaign",
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
                render_train_file(
                    {
                        "lookback_bars": 2,
                        "top_k": 1,
                        "gross_target": 1.0,
                        "ranking_mode": "raw_return",
                        "use_regime_filter": False,
                        "regime_lookback_bars": 12,
                        "regime_threshold": 0.01,
                        "min_signal_strength": 0.0,
                    }
                ),
                encoding="utf-8",
            )

            report = evaluate_train_file(
                campaign_path=campaign_path,
                train_path=train_path,
                artifact_root=temp_path / ".autoresearch" / "runs",
                stage="screen",
                mutation_label="unit-mutation",
            )
            results_path = append_results_row(
                results_path=temp_path / "results.tsv",
                report=report,
                decision="observed",
            )

            self.assertEqual(report.campaign_id, "unit")
            self.assertEqual(report.stage, "screen")
            self.assertEqual(report.mutation_label, "unit-mutation")
            self.assertEqual(report.total_windows, 2)
            self.assertTrue(Path(report.artifact_path).exists())
            results_text = results_path.read_text(encoding="utf-8")
            self.assertIn("campaign_id", results_text)
            self.assertIn("runtime_seconds", results_text)

    def test_git_runner_uses_worktree_and_staged_keep_discard(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            repo_root = Path(tempdir) / "repo"
            repo_root.mkdir()
            self._git(repo_root, "init")
            self._git(repo_root, "config", "user.email", "bot@example.com")
            self._git(repo_root, "config", "user.name", "Bot")
            (repo_root / "train.py").write_text("BASELINE = 1\n", encoding="utf-8")
            self._git(repo_root, "add", "train.py")
            self._git(repo_root, "commit", "-m", "Initial train file")

            def fake_evaluator(**kwargs):
                train_path = Path(kwargs["train_path"])
                artifact_dir = Path(kwargs["artifact_root"])
                artifact_dir.mkdir(parents=True, exist_ok=True)
                train_text = train_path.read_text(encoding="utf-8")
                if "SCREENBAD" in train_text:
                    score = 0.1
                elif "FULLBAD" in train_text and kwargs.get("stage") == "full":
                    score = 0.4
                elif "GOOD" in train_text and kwargs.get("stage") == "full":
                    score = 2.0
                elif "GOOD" in train_text:
                    score = 1.5
                else:
                    score = 0.5
                artifact_path = artifact_dir / f"{kwargs.get('stage','full')}-{score}.json"
                artifact_path.write_text("{}", encoding="utf-8")
                return AutoresearchRunReport(
                    run_id=f"run-{score}",
                    recorded_at=datetime.now(timezone.utc).isoformat(),
                    campaign_id="fake",
                    campaign_name="fake-campaign",
                    strategy_name="fake-strategy",
                    git_branch="",
                    git_commit="",
                    parent_commit=str(kwargs.get("parent_commit", "")),
                    train_file=str(train_path),
                    train_sha1="sha",
                    stage=str(kwargs.get("stage", "full")),
                    mutation_label=str(kwargs.get("mutation_label", "manual")),
                    baseline_score=kwargs.get("baseline_score"),
                    delta_score=(
                        score - float(kwargs["baseline_score"])
                        if kwargs.get("baseline_score") is not None
                        else None
                    ),
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
                    runtime_seconds=0.01,
                    artifact_path=str(artifact_path),
                )

            runner = GitAutoresearchRunner(
                repo_root=repo_root,
                branch_name="codex/autoresearch-test",
                evaluator=fake_evaluator,
                worktrees_root=repo_root / ".autoresearch" / "worktrees",
            )

            keep = runner.apply_candidate_staged(
                campaign_path=repo_root / "fake-campaign.json",
                candidate_text="GOOD = 1\n",
                commit_message="Keep improvement",
                mutation_label="good-mutation",
            )
            self.assertEqual(keep.decision, "keep")
            self.assertTrue(runner.worktree_root.exists())
            self.assertEqual((repo_root / "train.py").read_text(encoding="utf-8"), "BASELINE = 1\n")
            self.assertEqual(runner.train_path.read_text(encoding="utf-8"), "GOOD = 1\n")

            discard = runner.apply_candidate_staged(
                campaign_path=repo_root / "fake-campaign.json",
                candidate_text="SCREENBAD = 1\n",
                commit_message="Discard on screen",
                mutation_label="screenbad",
            )
            self.assertEqual(discard.decision, "discard_screen")
            self.assertEqual(runner.train_path.read_text(encoding="utf-8"), "GOOD = 1\n")
            self.assertIn("discard_screen", runner.results_path.read_text(encoding="utf-8"))

    def test_deterministic_mutator_and_batch_run(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            repo_root = Path(tempdir) / "repo"
            repo_root.mkdir()
            self._git(repo_root, "init")
            self._git(repo_root, "config", "user.email", "bot@example.com")
            self._git(repo_root, "config", "user.name", "Bot")
            (repo_root / "train.py").write_text(
                render_train_file(
                    {
                        "lookback_bars": 24,
                        "top_k": 1,
                        "gross_target": 0.5,
                        "ranking_mode": "risk_adjusted",
                        "use_regime_filter": False,
                        "regime_lookback_bars": 36,
                        "regime_threshold": 0.015,
                        "min_signal_strength": 0.0,
                    }
                ),
                encoding="utf-8",
            )
            self._git(repo_root, "add", "train.py")
            self._git(repo_root, "commit", "-m", "Initial train file")

            proposals = DeterministicTrainMutator().generate(repo_root / "train.py")
            self.assertTrue(any(item.label.startswith("lookback-") for item in proposals))
            self.assertTrue(any(item.label.startswith("ranking-") for item in proposals))

            def fake_evaluator(**kwargs):
                artifact_dir = Path(kwargs["artifact_root"])
                artifact_dir.mkdir(parents=True, exist_ok=True)
                train_text = Path(kwargs["train_path"]).read_text(encoding="utf-8")
                score = 1.1 if "'lookback_bars': 12" in train_text else 0.5
                artifact_path = artifact_dir / f"{kwargs.get('mutation_label','x')}.json"
                artifact_path.write_text("{}", encoding="utf-8")
                return AutoresearchRunReport(
                    run_id=str(kwargs.get("mutation_label", "x")),
                    recorded_at=datetime.now(timezone.utc).isoformat(),
                    campaign_id="fake",
                    campaign_name="fake-campaign",
                    strategy_name="fake-strategy",
                    git_branch="",
                    git_commit="",
                    parent_commit=str(kwargs.get("parent_commit", "")),
                    train_file=str(kwargs["train_path"]),
                    train_sha1="sha",
                    stage=str(kwargs.get("stage", "full")),
                    mutation_label=str(kwargs.get("mutation_label", "manual")),
                    baseline_score=kwargs.get("baseline_score"),
                    delta_score=None,
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
                    runtime_seconds=0.01,
                    artifact_path=str(artifact_path),
                )

            runner = GitAutoresearchRunner(
                repo_root=repo_root,
                branch_name="codex/autoresearch-batch",
                evaluator=fake_evaluator,
                worktrees_root=repo_root / ".autoresearch" / "worktrees",
            )
            runner.ensure_worktree()
            decisions = []
            for proposal in proposals[:3]:
                module_train = runner.train_path
                if not module_train.exists():
                    module_train.write_text((repo_root / "train.py").read_text(encoding="utf-8"), encoding="utf-8")
                current_config = ast.literal_eval(
                    runner.train_path.read_text(encoding="utf-8")
                    .split("TRAIN_CONFIG = ", 1)[1]
                    .split("\nSTRATEGY_NAME", 1)[0]
                )
                current_config.update(proposal.config_updates)
                decisions.append(
                    runner.apply_candidate_staged(
                        campaign_path=repo_root / "fake-campaign.json",
                        candidate_text=render_train_file(current_config),
                        commit_message=proposal.commit_message,
                        mutation_label=proposal.label,
                    )
                )

            self.assertTrue(any(item.decision == "keep" for item in decisions))

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
