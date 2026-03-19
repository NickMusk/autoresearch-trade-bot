from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from autoresearch_trade_bot.family_wave import (
    default_family_branch_name,
    prepare_family_repo,
    run_llm_family_wave,
)
from autoresearch_trade_bot.strategy_families import (
    FAMILY_EMA_TREND,
    FAMILY_MEAN_REVERSION,
    extract_strategy_family,
)


class FamilyWaveTests(unittest.TestCase):
    def test_prepare_family_repo_seeds_family_branch_without_changing_main_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            workspace = Path(tempdir)
            source_repo = workspace / "source"
            source_repo.mkdir()
            self._git(source_repo, "init")
            self._git(source_repo, "config", "user.email", "bot@example.com")
            self._git(source_repo, "config", "user.name", "Bot")
            (source_repo / "train.py").write_text(
                "\n".join(
                    [
                        "from __future__ import annotations",
                        "TRAIN_CONFIG = {'lookback_bars': 24, 'top_k': 1, 'gross_target': 0.5}",
                        'STRATEGY_NAME = "baseline"',
                        'STRATEGY_FAMILY = "momentum"',
                        "",
                        "def build_strategy(_dataset_spec=None):",
                        "    return None",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            self._git(source_repo, "add", "train.py")
            self._git(source_repo, "commit", "-m", "Initial baseline")
            self._git(source_repo, "branch", "-M", "main")

            family_repo_root, family_branch = prepare_family_repo(
                source_repo_root=source_repo,
                family_repo_root=workspace / "family-repos" / FAMILY_MEAN_REVERSION,
                strategy_family=FAMILY_MEAN_REVERSION,
            )

            self.assertEqual(family_branch, default_family_branch_name(FAMILY_MEAN_REVERSION))
            self.assertEqual(self._git(Path(family_repo_root), "branch", "--show-current"), "main")
            self._git(Path(family_repo_root), "checkout", family_branch)
            train_text = (Path(family_repo_root) / "train.py").read_text(encoding="utf-8")
            self.assertEqual(extract_strategy_family(train_text), FAMILY_MEAN_REVERSION)

    def test_run_llm_family_wave_uses_isolated_repo_roots_per_family(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            workspace = Path(tempdir)
            source_repo = workspace / "source"
            source_repo.mkdir()
            self._git(source_repo, "init")
            self._git(source_repo, "config", "user.email", "bot@example.com")
            self._git(source_repo, "config", "user.name", "Bot")
            (source_repo / "train.py").write_text(
                "\n".join(
                    [
                        "from __future__ import annotations",
                        "TRAIN_CONFIG = {'lookback_bars': 24, 'top_k': 1, 'gross_target': 0.5}",
                        'STRATEGY_NAME = "baseline"',
                        'STRATEGY_FAMILY = "momentum"',
                        "",
                        "def build_strategy(_dataset_spec=None):",
                        "    return None",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            self._git(source_repo, "add", "train.py")
            self._git(source_repo, "commit", "-m", "Initial baseline")
            self._git(source_repo, "branch", "-M", "main")

            calls: list[tuple[str, str, str]] = []

            def fake_runner(**kwargs):  # type: ignore[no-untyped-def]
                calls.append(
                    (
                        str(kwargs["repo_root"]),
                        str(kwargs["branch_name"]),
                        str(kwargs["campaign_path"]),
                    )
                )
                return []

            results = run_llm_family_wave(
                campaign_path="/tmp/campaign.json",
                source_repo_root=source_repo,
                family_repos_root=workspace / "family-repos",
                strategy_families=(FAMILY_MEAN_REVERSION, FAMILY_EMA_TREND),
                max_parallel=2,
                family_runner=fake_runner,
            )

            self.assertEqual(len(results), 2)
            self.assertEqual(len(calls), 2)
            self.assertNotEqual(calls[0][0], calls[1][0])
            self.assertCountEqual(
                [item.strategy_family for item in results],
                [FAMILY_MEAN_REVERSION, FAMILY_EMA_TREND],
            )
            self.assertCountEqual(
                [Path(repo_root).name for repo_root, _, _ in calls],
                [FAMILY_MEAN_REVERSION, FAMILY_EMA_TREND],
            )

    def _git(self, cwd: Path, *args: str) -> str:
        import subprocess

        completed = subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()
