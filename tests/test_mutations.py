from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from autoresearch_trade_bot.autoresearch import (
    AutoresearchRunReport,
    MutationProposal,
    render_train_file,
)
from autoresearch_trade_bot.mutations import (
    LLMCompletion,
    LLMMutationProvider,
    MutationContext,
    build_experiment_memory_summary,
    build_llm_mutation_prompt,
    validate_train_candidate_text,
)


class FakeLLMClient:
    def __init__(self, content: str) -> None:
        self.content = content

    def generate(self, *, system_prompt: str, user_prompt: str) -> LLMCompletion:
        return LLMCompletion(
            content=self.content,
            model_name="fake-gpt",
            prompt_id="resp_fake",
            raw_response={"id": "resp_fake", "model": "fake-gpt"},
        )


class MutationTests(unittest.TestCase):
    def test_validate_train_candidate_text_accepts_current_shape_and_rejects_forbidden_import(self) -> None:
        valid_candidate = render_train_file(
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
        )
        self.assertIn("'funding_penalty_weight': 0.0", valid_candidate)
        self.assertIn("'min_cross_sectional_spread': 0.0", valid_candidate)
        self.assertEqual(validate_train_candidate_text(valid_candidate), (True, ""))

        invalid_candidate = "import os\nTRAIN_CONFIG = {}\nSTRATEGY_NAME = 'x'\ndef build_strategy():\n    return None\n"
        is_valid, failure_reason = validate_train_candidate_text(invalid_candidate)
        self.assertFalse(is_valid)
        self.assertEqual(failure_reason, "forbidden_import:os")

    def test_llm_mutation_provider_writes_proposal_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            temp_path = Path(tempdir)
            candidate_text = render_train_file(
                {
                    "lookback_bars": 12,
                    "top_k": 1,
                    "gross_target": 0.5,
                    "ranking_mode": "risk_adjusted",
                    "use_regime_filter": False,
                    "regime_lookback_bars": 36,
                    "regime_threshold": 0.015,
                    "min_signal_strength": 0.0,
                }
            )
            context = MutationContext(
                campaign_id="unit",
                campaign_name="unit-campaign",
                branch_name="codex/test",
                parent_commit="abc123",
                train_path=temp_path / "train.py",
                results_path=temp_path / "results.tsv",
                artifact_root=temp_path / ".autoresearch" / "runs",
                program_text="Mutate train.py only.",
                current_train_text=render_train_file(
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
                recent_results=(),
            )
            provider = LLMMutationProvider(client=FakeLLMClient(candidate_text))
            proposals = provider.generate(context=context, max_mutations=1)

            self.assertEqual(len(proposals), 1)
            proposal = proposals[0]
            self.assertEqual(proposal.provider_name, "llm")
            self.assertEqual(proposal.model_name, "fake-gpt")
            self.assertTrue(proposal.proposal_artifact_path)
            artifact_payload = json.loads(Path(proposal.proposal_artifact_path).read_text(encoding="utf-8"))
            self.assertEqual(artifact_payload["prompt_id"], "resp_fake")
            self.assertIn("Current train.py to mutate", artifact_payload["user_prompt"])
            self.assertIn("Research memory", artifact_payload["user_prompt"])

    def test_experiment_memory_summary_and_prompt_include_failed_patterns_and_directions(self) -> None:
        current_train = render_train_file(
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
        )
        recent_results = (
            {
                "mutation_label": "llm-a",
                "decision": "discard_screen",
                "stage": "screen",
                "research_score": "-8.5",
                "delta_score": "-3.2",
                "failure_reason": "",
            },
            {
                "mutation_label": "llm-b",
                "decision": "discard_screen",
                "stage": "screen",
                "research_score": "-7.5",
                "delta_score": "-2.2",
                "failure_reason": "",
            },
            {
                "mutation_label": "llm-c",
                "decision": "discard_error",
                "stage": "validation",
                "research_score": "-inf",
                "delta_score": "",
                "failure_reason": "duplicate_candidate",
            },
        )
        summary = build_experiment_memory_summary(recent_results, current_train_text=current_train)
        self.assertIn("Decision mix:", summary)
        self.assertIn("discard_screen=2", summary)
        self.assertIn("duplicate_candidate: 1", summary)
        self.assertIn("min_cross_sectional_spread", summary)
        self.assertIn("funding_penalty_weight", summary)

        context = MutationContext(
            campaign_id="unit",
            campaign_name="unit-campaign",
            branch_name="codex/test",
            parent_commit="abc123",
            train_path=Path("/tmp/train.py"),
            results_path=Path("/tmp/results.tsv"),
            artifact_root=Path("/tmp/artifacts"),
            program_text="Mutate train.py only.",
            current_train_text=current_train,
            recent_results=recent_results,
            experiment_memory_summary=summary,
        )
        system_prompt, user_prompt = build_llm_mutation_prompt(context=context, max_mutations=1)
        self.assertIn("Prefer one substantial research hypothesis", system_prompt)
        self.assertIn("Research memory:", user_prompt)
        self.assertIn("Promising directions:", user_prompt)
        self.assertIn("Recent raw results:", user_prompt)

    def test_runner_discards_invalid_candidate_and_skips_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            repo_root = Path(tempdir) / "repo"
            repo_root.mkdir()
            self._git(repo_root, "init")
            self._git(repo_root, "config", "user.email", "bot@example.com")
            self._git(repo_root, "config", "user.name", "Bot")
            baseline_text = render_train_file(
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
            )
            (repo_root / "train.py").write_text(baseline_text, encoding="utf-8")
            self._git(repo_root, "add", "train.py")
            self._git(repo_root, "commit", "-m", "Initial train file")
            campaign_path = repo_root / "campaign.json"
            campaign_path.write_text(
                json.dumps(
                    {
                        "campaign_id": "unit",
                        "name": "unit-campaign",
                        "exchange": "bybit",
                        "market": "linear",
                        "timeframe": "5m",
                        "symbols": ["BTCUSDT", "ETHUSDT"],
                        "storage_root": str(repo_root / "data"),
                        "target_gate": {
                            "min_total_return": 0.0,
                            "min_sharpe": 1.0,
                            "max_drawdown": 0.2,
                            "min_acceptance_rate": 0.6,
                        },
                        "windows": [],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            def fake_evaluator(**kwargs):
                artifact_dir = Path(kwargs["artifact_root"])
                artifact_dir.mkdir(parents=True, exist_ok=True)
                train_text = Path(kwargs["train_path"]).read_text(encoding="utf-8")
                score = 1.5 if "'lookback_bars': 12" in train_text else 0.5
                artifact_path = artifact_dir / f"{kwargs.get('mutation_label','x')}.json"
                artifact_path.write_text("{}", encoding="utf-8")
                return AutoresearchRunReport(
                    run_id=str(kwargs.get("mutation_label", "x")),
                    recorded_at=datetime.now(timezone.utc).isoformat(),
                    campaign_id="unit",
                    campaign_name="unit-campaign",
                    strategy_name="fake",
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
                    total_windows=1,
                    window_reports=[],
                    runtime_seconds=0.01,
                    artifact_path=str(artifact_path),
                )

            from autoresearch_trade_bot.autoresearch import GitAutoresearchRunner

            runner = GitAutoresearchRunner(
                repo_root=repo_root,
                branch_name="codex/llm-test",
                evaluator=fake_evaluator,
                worktrees_root=repo_root / ".autoresearch" / "worktrees",
            )

            invalid_decision = runner.apply_mutation_proposal_staged(
                campaign_path=campaign_path,
                proposal=MutationProposal(
                    label="bad-import",
                    candidate_text="import os\nTRAIN_CONFIG = {}\nSTRATEGY_NAME = 'x'\ndef build_strategy():\n    return None\n",
                    commit_message="bad import",
                    provider_name="llm",
                    model_name="fake-gpt",
                    prompt_id="resp_bad",
                ),
                validator=validate_train_candidate_text,
            )
            self.assertEqual(invalid_decision.decision, "discard_error")
            self.assertEqual(invalid_decision.report.failure_reason, "forbidden_import:os")

            valid_candidate = render_train_file(
                {
                    "lookback_bars": 12,
                    "top_k": 1,
                    "gross_target": 0.5,
                    "ranking_mode": "risk_adjusted",
                    "use_regime_filter": False,
                    "regime_lookback_bars": 36,
                    "regime_threshold": 0.015,
                    "min_signal_strength": 0.0,
                }
            )
            keep_decision = runner.apply_mutation_proposal_staged(
                campaign_path=campaign_path,
                proposal=MutationProposal(
                    label="good-1",
                    candidate_text=valid_candidate,
                    commit_message="good mutation",
                    provider_name="llm",
                    model_name="fake-gpt",
                    prompt_id="resp_good",
                ),
                validator=validate_train_candidate_text,
            )
            self.assertEqual(keep_decision.decision, "keep")

            duplicate_decision = runner.apply_mutation_proposal_staged(
                campaign_path=campaign_path,
                proposal=MutationProposal(
                    label="good-duplicate",
                    candidate_text=valid_candidate,
                    commit_message="duplicate mutation",
                    provider_name="llm",
                    model_name="fake-gpt",
                    prompt_id="resp_dup",
                ),
                validator=validate_train_candidate_text,
            )
            self.assertEqual(duplicate_decision.decision, "skip_duplicate")
            self.assertEqual(duplicate_decision.report.failure_reason, "duplicate_candidate")

    def test_runner_sets_local_git_identity_before_keep_commit(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            repo_root = Path(tempdir) / "repo"
            repo_root.mkdir()
            self._git(repo_root, "init")
            self._git(repo_root, "config", "user.email", "bot@example.com")
            self._git(repo_root, "config", "user.name", "Bot")
            baseline_text = render_train_file(
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
            )
            (repo_root / "train.py").write_text(baseline_text, encoding="utf-8")
            self._git(repo_root, "add", "train.py")
            self._git(repo_root, "commit", "-m", "Initial train file")
            self._git(repo_root, "config", "--unset", "user.email")
            self._git(repo_root, "config", "--unset", "user.name")

            campaign_path = repo_root / "campaign.json"
            campaign_path.write_text(
                json.dumps(
                    {
                        "campaign_id": "unit",
                        "name": "unit-campaign",
                        "exchange": "bybit",
                        "market": "linear",
                        "timeframe": "5m",
                        "symbols": ["BTCUSDT", "ETHUSDT"],
                        "storage_root": str(repo_root / "data"),
                        "target_gate": {
                            "min_total_return": 0.0,
                            "min_sharpe": 1.0,
                            "max_drawdown": 0.2,
                            "min_acceptance_rate": 0.6,
                        },
                        "windows": [],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            def fake_evaluator(**kwargs):
                artifact_dir = Path(kwargs["artifact_root"])
                artifact_dir.mkdir(parents=True, exist_ok=True)
                train_text = Path(kwargs["train_path"]).read_text(encoding="utf-8")
                score = 1.5 if "'lookback_bars': 12" in train_text else 0.5
                artifact_path = artifact_dir / f"{kwargs.get('mutation_label','x')}.json"
                artifact_path.write_text("{}", encoding="utf-8")
                return AutoresearchRunReport(
                    run_id=str(kwargs.get("mutation_label", "x")),
                    recorded_at=datetime.now(timezone.utc).isoformat(),
                    campaign_id="unit",
                    campaign_name="unit-campaign",
                    strategy_name="fake",
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
                    total_windows=1,
                    window_reports=[],
                    runtime_seconds=0.01,
                    artifact_path=str(artifact_path),
                )

            from autoresearch_trade_bot.autoresearch import GitAutoresearchRunner

            runner = GitAutoresearchRunner(
                repo_root=repo_root,
                branch_name="codex/llm-test-identity",
                evaluator=fake_evaluator,
                worktrees_root=repo_root / ".autoresearch" / "worktrees",
            )

            valid_candidate = render_train_file(
                {
                    "lookback_bars": 12,
                    "top_k": 1,
                    "gross_target": 0.5,
                    "ranking_mode": "risk_adjusted",
                    "use_regime_filter": False,
                    "regime_lookback_bars": 36,
                    "regime_threshold": 0.015,
                    "min_signal_strength": 0.0,
                }
            )
            keep_decision = runner.apply_mutation_proposal_staged(
                campaign_path=campaign_path,
                proposal=MutationProposal(
                    label="good-identity",
                    candidate_text=valid_candidate,
                    commit_message="good mutation",
                    provider_name="llm",
                    model_name="fake-gpt",
                    prompt_id="resp_good",
                ),
                validator=validate_train_candidate_text,
            )

            self.assertEqual(keep_decision.decision, "keep")
            self.assertEqual(
                self._git(repo_root, "config", "--local", "--get", "user.name"),
                "Autoresearch Bot",
            )
            self.assertEqual(
                self._git(repo_root, "config", "--local", "--get", "user.email"),
                "autoresearch-bot@local",
            )

    def test_runner_reports_subprocess_details_on_candidate_error(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            repo_root = Path(tempdir) / "repo"
            repo_root.mkdir()
            self._git(repo_root, "init")
            self._git(repo_root, "config", "user.email", "bot@example.com")
            self._git(repo_root, "config", "user.name", "Bot")
            baseline_text = render_train_file(
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
            )
            (repo_root / "train.py").write_text(baseline_text, encoding="utf-8")
            self._git(repo_root, "add", "train.py")
            self._git(repo_root, "commit", "-m", "Initial train file")

            campaign_path = repo_root / "campaign.json"
            campaign_path.write_text(
                json.dumps(
                    {
                        "campaign_id": "unit",
                        "name": "unit-campaign",
                        "exchange": "bybit",
                        "market": "linear",
                        "timeframe": "5m",
                        "symbols": ["BTCUSDT", "ETHUSDT"],
                        "storage_root": str(repo_root / "data"),
                        "target_gate": {
                            "min_total_return": 0.0,
                            "min_sharpe": 1.0,
                            "max_drawdown": 0.2,
                            "min_acceptance_rate": 0.6,
                        },
                        "windows": [],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            def fake_evaluator(**kwargs):
                artifact_dir = Path(kwargs["artifact_root"])
                artifact_dir.mkdir(parents=True, exist_ok=True)
                artifact_path = artifact_dir / f"{kwargs.get('mutation_label','x')}.json"
                artifact_path.write_text("{}", encoding="utf-8")
                train_text = Path(kwargs["train_path"]).read_text(encoding="utf-8")
                score = 2.0 if "'lookback_bars': 12" in train_text else 0.5
                return AutoresearchRunReport(
                    run_id=str(kwargs.get("mutation_label", "x")),
                    recorded_at=datetime.now(timezone.utc).isoformat(),
                    campaign_id="unit",
                    campaign_name="unit-campaign",
                    strategy_name="fake",
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
                    total_windows=1,
                    window_reports=[],
                    runtime_seconds=0.01,
                    artifact_path=str(artifact_path),
                )

            from autoresearch_trade_bot.autoresearch import GitAutoresearchRunner

            runner = GitAutoresearchRunner(
                repo_root=repo_root,
                branch_name="codex/llm-test-error",
                evaluator=fake_evaluator,
                worktrees_root=repo_root / ".autoresearch" / "worktrees",
            )
            runner.ensure_worktree()
            original_git_in_worktree = runner._git_in_worktree

            def fail_commit(args):  # type: ignore[no-untyped-def]
                if list(args[:2]) == ["commit", "-m"]:
                    raise subprocess.CalledProcessError(
                        128,
                        ["git", "commit", "-m", "good mutation"],
                        stderr="Author identity unknown\nfatal: unable to auto-detect email address",
                    )
                return original_git_in_worktree(args)

            runner._git_in_worktree = fail_commit  # type: ignore[method-assign]

            valid_candidate = render_train_file(
                {
                    "lookback_bars": 12,
                    "top_k": 1,
                    "gross_target": 0.5,
                    "ranking_mode": "risk_adjusted",
                    "use_regime_filter": False,
                    "regime_lookback_bars": 36,
                    "regime_threshold": 0.015,
                    "min_signal_strength": 0.0,
                }
            )
            decision = runner.apply_mutation_proposal_staged(
                campaign_path=campaign_path,
                proposal=MutationProposal(
                    label="good-error",
                    candidate_text=valid_candidate,
                    commit_message="good mutation",
                    provider_name="llm",
                    model_name="fake-gpt",
                    prompt_id="resp_good",
                ),
                validator=validate_train_candidate_text,
            )

            self.assertEqual(decision.decision, "discard_error")
            self.assertIn("cmd=git commit -m good mutation", decision.report.failure_reason)
            self.assertIn("rc=128", decision.report.failure_reason)
            self.assertIn("Author identity unknown", decision.report.failure_reason)

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
