from __future__ import annotations

import json
import io
import socket
import subprocess
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from autoresearch_trade_bot.autoresearch import (
    AutoresearchRunReport,
    MutationProposal,
    render_train_file,
)
from autoresearch_trade_bot.mutations import (
    LLMCompletion,
    LLMMutationProvider,
    MutationContext,
    build_attempt_prompt,
    build_experiment_memory_artifact,
    build_experiment_memory_summary,
    build_llm_mutation_prompt,
    build_mutation_context,
    OpenAIResponsesClient,
    validate_train_candidate_text,
    validate_train_candidate_semantics,
)
from autoresearch_trade_bot.strategy_families import (
    FAMILY_EMA_TREND,
    FAMILY_MEAN_REVERSION,
    FAMILY_VOLATILITY_BREAKOUT,
    deterministic_mutation_specs,
    extract_strategy_family,
    family_mutation_bounds,
    render_train_file as render_family_train_file,
)


class FakeLLMClient:
    def __init__(self, content: str | list[str]) -> None:
        self.content = [content] if isinstance(content, str) else list(content)
        self.calls: list[tuple[str, str]] = []

    def generate(self, *, system_prompt: str, user_prompt: str) -> LLMCompletion:
        self.calls.append((system_prompt, user_prompt))
        content = self.content[min(len(self.calls) - 1, len(self.content) - 1)]
        return LLMCompletion(
            content=content,
            model_name="fake-gpt",
            prompt_id=f"resp_fake_{len(self.calls)}",
            raw_response={"id": f"resp_fake_{len(self.calls)}", "model": "fake-gpt"},
        )


class MutationTests(unittest.TestCase):
    def test_openai_client_retries_retryable_http_error_then_succeeds(self) -> None:
        client = OpenAIResponsesClient(
            api_key="test-key",
            timeout_seconds=1.0,
            max_retries=3,
            retry_backoff_seconds=2.0,
        )
        responses = [
            HTTPError(
                url="https://api.openai.com/v1/responses",
                code=429,
                msg="Too Many Requests",
                hdrs={"Retry-After": "1.5"},
                fp=io.BytesIO(b'{"error":"rate_limited"}'),
            ),
            io.BytesIO(json.dumps({"id": "resp_1", "model": "gpt-5-mini", "output_text": "ok"}).encode("utf-8")),
        ]

        class FakeResponse:
            def __init__(self, payload: bytes) -> None:
                self.payload = payload

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self) -> bytes:
                return self.payload

        side_effects = [responses[0], FakeResponse(responses[1].read())]

        with patch("autoresearch_trade_bot.mutations.urllib.request.urlopen", side_effect=side_effects):
            with patch("autoresearch_trade_bot.mutations.time.sleep") as mocked_sleep:
                completion = client.generate(system_prompt="system", user_prompt="user")

        self.assertEqual(completion.content, "ok")
        mocked_sleep.assert_called_once_with(1.5)

    def test_openai_client_retries_timeout_then_raises_after_budget(self) -> None:
        client = OpenAIResponsesClient(
            api_key="test-key",
            timeout_seconds=1.0,
            max_retries=3,
            retry_backoff_seconds=2.0,
        )

        with patch(
            "autoresearch_trade_bot.mutations.urllib.request.urlopen",
            side_effect=socket.timeout("timed out"),
        ):
            with patch("autoresearch_trade_bot.mutations.time.sleep") as mocked_sleep:
                with self.assertRaisesRegex(RuntimeError, "openai_timeout"):
                    client.generate(system_prompt="system", user_prompt="user")

        self.assertEqual(mocked_sleep.call_args_list[0].args[0], 2.0)
        self.assertEqual(mocked_sleep.call_args_list[1].args[0], 4.0)

    def test_openai_client_does_not_retry_non_retryable_http_error(self) -> None:
        client = OpenAIResponsesClient(
            api_key="test-key",
            timeout_seconds=1.0,
            max_retries=3,
            retry_backoff_seconds=2.0,
        )
        error = HTTPError(
            url="https://api.openai.com/v1/responses",
            code=400,
            msg="Bad Request",
            hdrs={},
            fp=io.BytesIO(b'{"error":"bad_request"}'),
        )

        with patch("autoresearch_trade_bot.mutations.urllib.request.urlopen", side_effect=error):
            with patch("autoresearch_trade_bot.mutations.time.sleep") as mocked_sleep:
                with self.assertRaisesRegex(RuntimeError, r"openai_http_error:400"):
                    client.generate(system_prompt="system", user_prompt="user")

        mocked_sleep.assert_not_called()

    def test_openai_client_retries_retryable_url_timeout_errors(self) -> None:
        client = OpenAIResponsesClient(
            api_key="test-key",
            timeout_seconds=1.0,
            max_retries=2,
            retry_backoff_seconds=3.0,
        )
        timeout_error = URLError(socket.timeout("timed out"))

        with patch("autoresearch_trade_bot.mutations.urllib.request.urlopen", side_effect=timeout_error):
            with patch("autoresearch_trade_bot.mutations.time.sleep") as mocked_sleep:
                with self.assertRaisesRegex(RuntimeError, "openai_timeout"):
                    client.generate(system_prompt="system", user_prompt="user")

        mocked_sleep.assert_called_once_with(3.0)

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
                strategy_family="momentum",
                recent_results=(),
                symbol_count=5,
            )
            provider = LLMMutationProvider(client=FakeLLMClient(candidate_text))
            proposals = provider.generate(context=context, max_mutations=1)

            self.assertEqual(len(proposals), 1)
            proposal = proposals[0]
            self.assertEqual(proposal.provider_name, "llm")
            self.assertEqual(proposal.model_name, "fake-gpt")
            self.assertTrue(proposal.proposal_artifact_path)
            artifact_payload = json.loads(Path(proposal.proposal_artifact_path).read_text(encoding="utf-8"))
            self.assertEqual(artifact_payload["prompt_id"], "resp_fake_1")
            self.assertIn("Current train.py to mutate", artifact_payload["user_prompt"])
            self.assertIn("Research memory", artifact_payload["user_prompt"])

    def test_llm_mutation_provider_generates_two_distinct_proposals_in_one_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            temp_path = Path(tempdir)
            first_candidate = render_train_file(
                {
                    "lookback_bars": 12,
                    "top_k": 1,
                    "gross_target": 0.5,
                    "ranking_mode": "risk_adjusted",
                    "use_regime_filter": False,
                    "regime_lookback_bars": 36,
                    "regime_threshold": 0.015,
                    "min_signal_strength": 0.0,
                    "min_cross_sectional_spread": 0.05,
                }
            )
            second_candidate = render_train_file(
                {
                    "lookback_bars": 24,
                    "top_k": 1,
                    "gross_target": 0.5,
                    "ranking_mode": "risk_adjusted",
                    "use_regime_filter": False,
                    "regime_lookback_bars": 36,
                    "regime_threshold": 0.015,
                    "min_signal_strength": 0.01,
                    "funding_penalty_weight": 10.0,
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
                strategy_family="momentum",
                recent_results=(),
                symbol_count=5,
                experiment_memory_summary="Decision mix: no prior LLM evaluations yet.",
            )
            client = FakeLLMClient([first_candidate, second_candidate])
            provider = LLMMutationProvider(client=client)

            proposals = provider.generate(context=context, max_mutations=2)

            self.assertEqual(len(proposals), 2)
            self.assertNotEqual(proposals[0].candidate_sha1, proposals[1].candidate_sha1)
            self.assertEqual(len(client.calls), 2)
            self.assertIn("attempt_index=1", client.calls[0][1])
            self.assertIn("role_name=exploit_baseline", client.calls[0][1])
            self.assertIn("attempt_index=2", client.calls[1][1])
            self.assertIn("role_name=selectivity_without_no_trade", client.calls[1][1])

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
            {
                "mutation_label": "llm-d",
                "decision": "discard_screen",
                "stage": "screen",
                "research_score": "0.0",
                "delta_score": "-0.8",
                "failure_reason": "",
                "gate_failures_json": "[\"no_trades_executed\",\"sharpe_below_gate\"]",
                "train_config_json": "{\"lookback_bars\":24,\"top_k\":1,\"gross_target\":0.5,\"ranking_mode\":\"risk_adjusted\",\"use_regime_filter\":true,\"regime_lookback_bars\":36,\"regime_threshold\":0.07,\"min_signal_strength\":0.08,\"min_cross_sectional_spread\":0.12,\"volatility_floor\":0.0,\"reversal_bias_weight\":0.0,\"funding_penalty_weight\":0.0}",
            },
        )
        artifact = build_experiment_memory_artifact(recent_results, current_train_text=current_train)
        self.assertIn("dead_zones", artifact)
        self.assertIn("near_wins", artifact)
        self.assertTrue(any(item["trait"] == "regime_filter=on" for item in artifact["dead_zones"]))
        self.assertTrue(any(item["gate_failure"] == "no_trades_executed" for item in artifact["common_gate_failures"]))

        summary = build_experiment_memory_summary(recent_results, current_train_text=current_train)
        self.assertIn("Decision mix:", summary)
        self.assertIn("discard_screen=3", summary)
        self.assertIn("duplicate_candidate: 1", summary)
        self.assertIn("Dead zones:", summary)
        self.assertIn("no_trades_executed", summary)
        self.assertIn("Near wins:", summary)

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
            strategy_family="momentum",
            recent_results=recent_results,
            symbol_count=5,
            experiment_memory_summary=summary,
            experiment_memory_artifact=artifact,
        )
        system_prompt, user_prompt = build_llm_mutation_prompt(context=context, max_mutations=1)
        self.assertIn("Prefer one substantial research hypothesis", system_prompt)
        self.assertIn("Keep the candidate under 48000 bytes", system_prompt)
        self.assertIn("Preserve the existing file structure whenever possible", system_prompt)
        self.assertIn("Hard mutation bounds:", user_prompt)
        self.assertIn("Family template constraints:", user_prompt)
        self.assertIn("paired long/short engine requires top_k between 1 and 2 inclusive", user_prompt)
        self.assertIn("Do not set top_k above 2", user_prompt)
        self.assertIn("Research memory:", user_prompt)
        self.assertIn("Promising directions:", user_prompt)
        self.assertIn("Recent raw results:", user_prompt)
        self.assertIn("Make the smallest full-file edit that expresses the hypothesis", user_prompt)
        self.assertIn("Preserve the family-specific class template", user_prompt)

    def test_family_mutation_bounds_cap_top_k_from_symbol_count(self) -> None:
        bounds = family_mutation_bounds(FAMILY_MEAN_REVERSION, symbol_count=5)
        self.assertEqual(bounds["top_k_min"], 1)
        self.assertEqual(bounds["top_k_max"], 2)
        self.assertIn(
            "reversion_horizon_bars must satisfy 2 <= value < trend_lookback_bars",
            bounds["family_rules"],
        )

    def test_mean_reversion_family_is_extracted_and_validated_with_family_rules(self) -> None:
        current_train = render_family_train_file(
            {
                "lookback_bars": 24,
                "reversion_horizon_bars": 6,
                "ibs_threshold": 0.25,
                "top_k": 1,
                "gross_target": 0.5,
                "reversion_strength_floor": 0.0,
                "volatility_floor": 0.0,
                "use_trend_filter": False,
                "trend_lookback_bars": 48,
            },
            strategy_family=FAMILY_MEAN_REVERSION,
        )
        self.assertEqual(extract_strategy_family(current_train), FAMILY_MEAN_REVERSION)

        risky_candidate = render_family_train_file(
            {
                "lookback_bars": 24,
                "reversion_horizon_bars": 9,
                "ibs_threshold": 0.12,
                "top_k": 1,
                "gross_target": 0.5,
                "reversion_strength_floor": 0.5,
                "volatility_floor": 0.02,
                "use_trend_filter": True,
                "trend_lookback_bars": 72,
            },
            strategy_family=FAMILY_MEAN_REVERSION,
        )
        self.assertEqual(
            validate_train_candidate_semantics(
                risky_candidate,
                current_train_text=current_train,
                symbol_count=5,
            ),
            (False, "likely_no_trade_ibs_floor_stack"),
        )

    def test_family_prompt_includes_family_specific_template_constraints(self) -> None:
        current_train = render_family_train_file(
            {
                "lookback_bars": 24,
                "reversion_horizon_bars": 6,
                "ibs_threshold": 0.25,
                "top_k": 1,
                "gross_target": 0.5,
                "reversion_strength_floor": 0.0,
                "volatility_floor": 0.0,
                "use_trend_filter": False,
                "trend_lookback_bars": 48,
            },
            strategy_family=FAMILY_MEAN_REVERSION,
        )
        context = MutationContext(
            campaign_id="unit",
            campaign_name="unit-campaign",
            branch_name="codex/family-mean-reversion",
            parent_commit="abc123",
            train_path=Path("/tmp/train.py"),
            results_path=Path("/tmp/results.tsv"),
            artifact_root=Path("/tmp/artifacts"),
            program_text="Mutate train.py only.",
            current_train_text=current_train,
            strategy_family=FAMILY_MEAN_REVERSION,
            recent_results=(),
            symbol_count=5,
            experiment_memory_summary="Decision mix: no prior LLM evaluations yet.",
        )
        _system_prompt, user_prompt = build_llm_mutation_prompt(context=context, max_mutations=1)
        self.assertIn("Family template constraints:", user_prompt)
        self.assertIn("Preserve the IBSReversionStrategy template", user_prompt)
        self.assertIn("reversion_horizon_bars", user_prompt)
        self.assertIn("Do not fall back to the generic configurable-momentum template", user_prompt)

    def test_build_mutation_context_uses_explicit_family_override(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            workspace = Path(tempdir)
            repo_root = workspace / "repo"
            repo_root.mkdir()
            (repo_root / "program.md").write_text("Mutate train.py only.", encoding="utf-8")
            train_path = repo_root / "train.py"
            train_path.write_text(
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
                        "min_cross_sectional_spread": 0.0,
                        "volatility_floor": 0.0,
                        "reversal_bias_weight": 0.0,
                        "funding_penalty_weight": 0.0,
                    }
                ),
                encoding="utf-8",
            )
            campaign_path = workspace / "campaign.json"
            campaign_path.write_text(
                json.dumps(
                    {
                        "campaign_id": "unit",
                        "name": "unit-campaign",
                        "exchange": "binance",
                        "market": "usdm_futures",
                        "timeframe": "5m",
                        "storage_root": str(workspace / "data"),
                        "symbols": ["BTCUSDT", "ETHUSDT"],
                        "windows": [],
                        "target_gate": {
                            "min_total_return": 0.0,
                            "min_sharpe": 1.0,
                            "max_drawdown": 0.2,
                            "min_acceptance_rate": 0.6,
                        },
                    }
                ),
                encoding="utf-8",
            )
            results_path = workspace / "results.tsv"
            results_path.write_text("", encoding="utf-8")
            artifact_root = workspace / "artifacts"
            context = build_mutation_context(
                repo_root=repo_root,
                campaign_path=campaign_path,
                branch_name="codex/family-mean-reversion",
                train_path=train_path,
                results_path=results_path,
                artifact_root=artifact_root,
                strategy_family=FAMILY_MEAN_REVERSION,
            )

        self.assertEqual(context.strategy_family, FAMILY_MEAN_REVERSION)
        _system_prompt, user_prompt = build_llm_mutation_prompt(context=context, max_mutations=1)
        self.assertIn("Preserve the IBSReversionStrategy template", user_prompt)
        self.assertIn("ibs_threshold", user_prompt)

    def test_family_specific_memory_guidance_surfaces_dual_momentum_dead_zones(self) -> None:
        current_train = render_family_train_file(
            {
                "gross_target": 0.5,
                "fast_horizon_bars": 12,
                "medium_horizon_bars": 36,
                "slow_horizon_bars": 96,
                "top_k": 1,
                "min_signal_strength": 0.0,
                "absolute_momentum_floor": 0.0,
                "relative_strength_weight": 0.6,
                "use_absolute_filter": True,
                "volatility_floor": 0.0,
            },
            strategy_family=FAMILY_EMA_TREND,
        )
        recent_results = (
            {
                "mutation_label": "dual-a",
                "decision": "discard_full",
                "stage": "full",
                "research_score": "0.0",
                "delta_score": "-0.9",
                "failure_reason": "",
                "gate_failures_json": "[\"no_trades_executed\",\"sharpe_below_gate\"]",
                "train_config_json": json.dumps(
                    {
                        "gross_target": 0.5,
                        "fast_horizon_bars": 16,
                        "medium_horizon_bars": 48,
                        "slow_horizon_bars": 144,
                        "top_k": 1,
                        "min_signal_strength": 0.3,
                        "absolute_momentum_floor": 0.25,
                        "relative_strength_weight": 0.8,
                        "use_absolute_filter": True,
                        "volatility_floor": 0.02,
                    }
                ),
            },
            {
                "mutation_label": "dual-b",
                "decision": "discard_screen",
                "stage": "screen",
                "research_score": "-3.0",
                "delta_score": "-2.1",
                "failure_reason": "",
                "gate_failures_json": "[\"no_trades_executed\"]",
                "train_config_json": json.dumps(
                    {
                        "gross_target": 0.5,
                        "fast_horizon_bars": 16,
                        "medium_horizon_bars": 48,
                        "slow_horizon_bars": 144,
                        "top_k": 1,
                        "min_signal_strength": 0.25,
                        "absolute_momentum_floor": 0.25,
                        "relative_strength_weight": 0.75,
                        "use_absolute_filter": True,
                        "volatility_floor": 0.02,
                    }
                ),
            },
        )
        artifact = build_experiment_memory_artifact(
            recent_results,
            current_train_text=current_train,
            strategy_family=FAMILY_EMA_TREND,
        )
        dead_zone_traits = {item["trait"] for item in artifact["dead_zones"]}
        self.assertIn("absolute_floor=mid", dead_zone_traits)
        self.assertIn("relative_weight=high", dead_zone_traits)
        self.assertIn("absolute_filter=on", dead_zone_traits)
        directions = artifact["promising_directions"]
        self.assertTrue(
            any(
                "Keep absolute_momentum_floor and min_signal_strength mild" in item
                for item in directions
            )
        )
        self.assertTrue(any("Use relative_strength_weight" in item for item in directions))

    def test_deterministic_mutation_specs_cover_family_specific_knobs(self) -> None:
        mean_reversion_specs = deterministic_mutation_specs(
            FAMILY_MEAN_REVERSION,
            {
                "lookback_bars": 24,
                "reversion_horizon_bars": 6,
                "ibs_threshold": 0.25,
                "top_k": 1,
                "gross_target": 0.5,
                "reversion_strength_floor": 0.0,
                "volatility_floor": 0.0,
                "use_trend_filter": False,
                "trend_lookback_bars": 48,
            },
        )
        mean_reversion_keys = {next(iter(item["config_updates"])) for item in mean_reversion_specs}
        self.assertIn("ibs_threshold", mean_reversion_keys)
        self.assertIn("reversion_strength_floor", mean_reversion_keys)
        self.assertIn("volatility_floor", mean_reversion_keys)
        self.assertIn("trend_lookback_bars", mean_reversion_keys)

        ema_specs = deterministic_mutation_specs(
            FAMILY_EMA_TREND,
            {
                "gross_target": 0.5,
                "fast_horizon_bars": 12,
                "medium_horizon_bars": 36,
                "slow_horizon_bars": 96,
                "top_k": 1,
                "min_signal_strength": 0.0,
                "absolute_momentum_floor": 0.0,
                "relative_strength_weight": 0.6,
                "use_absolute_filter": True,
                "volatility_floor": 0.0,
            },
        )
        ema_keys = {next(iter(item["config_updates"])) for item in ema_specs}
        self.assertIn("absolute_momentum_floor", ema_keys)
        self.assertIn("relative_strength_weight", ema_keys)
        self.assertIn("volatility_floor", ema_keys)

        breakout_specs = deterministic_mutation_specs(
            FAMILY_VOLATILITY_BREAKOUT,
            {
                "gross_target": 0.5,
                "channel_bars": 24,
                "atr_lookback_bars": 14,
                "atr_multiplier": 1.0,
                "breakout_buffer": 0.0,
                "top_k": 1,
                "breakout_score_floor": 0.0,
                "use_trend_filter": False,
                "trend_lookback_bars": 72,
            },
        )
        breakout_keys = {next(iter(item["config_updates"])) for item in breakout_specs}
        self.assertIn("breakout_score_floor", breakout_keys)
        self.assertIn("trend_lookback_bars", breakout_keys)

    def test_validate_train_candidate_semantics_rejects_family_specific_no_trade_stacks(self) -> None:
        current_ema = render_family_train_file(
            {
                "gross_target": 0.5,
                "fast_horizon_bars": 12,
                "medium_horizon_bars": 36,
                "slow_horizon_bars": 96,
                "top_k": 1,
                "min_signal_strength": 0.0,
                "absolute_momentum_floor": 0.0,
                "relative_strength_weight": 0.6,
                "use_absolute_filter": True,
                "volatility_floor": 0.0,
            },
            strategy_family=FAMILY_EMA_TREND,
        )
        risky_ema = render_family_train_file(
            {
                "gross_target": 0.5,
                "fast_horizon_bars": 16,
                "medium_horizon_bars": 48,
                "slow_horizon_bars": 144,
                "top_k": 1,
                "min_signal_strength": 0.25,
                "absolute_momentum_floor": 0.25,
                "relative_strength_weight": 0.8,
                "use_absolute_filter": True,
                "volatility_floor": 0.0,
            },
            strategy_family=FAMILY_EMA_TREND,
        )
        self.assertEqual(
            validate_train_candidate_semantics(
                risky_ema,
                current_train_text=current_ema,
                symbol_count=5,
            ),
            (False, "likely_no_trade_confirmation_stack"),
        )

        current_breakout = render_family_train_file(
            {
                "gross_target": 0.5,
                "channel_bars": 24,
                "atr_lookback_bars": 14,
                "atr_multiplier": 1.0,
                "breakout_buffer": 0.0,
                "top_k": 1,
                "breakout_score_floor": 0.0,
                "use_trend_filter": False,
                "trend_lookback_bars": 72,
            },
            strategy_family=FAMILY_VOLATILITY_BREAKOUT,
        )
        risky_breakout = render_family_train_file(
            {
                "gross_target": 0.5,
                "channel_bars": 36,
                "atr_lookback_bars": 14,
                "atr_multiplier": 1.5,
                "breakout_buffer": 0.25,
                "top_k": 1,
                "breakout_score_floor": 0.5,
                "use_trend_filter": False,
                "trend_lookback_bars": 72,
            },
            strategy_family=FAMILY_VOLATILITY_BREAKOUT,
        )
        self.assertEqual(
            validate_train_candidate_semantics(
                risky_breakout,
                current_train_text=current_breakout,
                symbol_count=5,
            ),
            (False, "likely_no_trade_breakout_threshold_stack"),
        )

    def test_validate_train_candidate_semantics_rejects_momentum_template_drift_inside_family_branch(self) -> None:
        current_train = render_family_train_file(
            {
                "lookback_bars": 24,
                "reversion_horizon_bars": 6,
                "ibs_threshold": 0.25,
                "top_k": 1,
                "gross_target": 0.5,
                "reversion_strength_floor": 0.0,
                "volatility_floor": 0.0,
                "use_trend_filter": False,
                "trend_lookback_bars": 48,
            },
            strategy_family=FAMILY_MEAN_REVERSION,
        )
        drifted_candidate = render_train_file(
            {
                "lookback_bars": 24,
                "top_k": 1,
                "gross_target": 0.5,
                "ranking_mode": "risk_adjusted",
                "use_regime_filter": False,
                "regime_lookback_bars": 36,
                "regime_threshold": 0.015,
                "min_signal_strength": 0.02,
                "min_cross_sectional_spread": 0.04,
                "volatility_floor": 0.001,
                "reversal_bias_weight": 0.05,
                "funding_penalty_weight": 0.01,
            }
        ).replace('STRATEGY_FAMILY = "momentum"', f'STRATEGY_FAMILY = "{FAMILY_MEAN_REVERSION}"')
        self.assertEqual(
            validate_train_candidate_semantics(
                drifted_candidate,
                current_train_text=current_train,
                symbol_count=5,
            ),
            (
                False,
                "missing_family_config_keys:ibs_threshold,reversion_horizon_bars,reversion_strength_floor,trend_lookback_bars,use_trend_filter",
            ),
        )

    def test_validate_train_candidate_semantics_rejects_family_template_with_wrong_strategy_class(self) -> None:
        current_train = render_family_train_file(
            {
                "gross_target": 0.5,
                "fast_horizon_bars": 12,
                "medium_horizon_bars": 36,
                "slow_horizon_bars": 96,
                "top_k": 1,
                "min_signal_strength": 0.0,
                "absolute_momentum_floor": 0.0,
                "relative_strength_weight": 0.6,
                "use_absolute_filter": True,
                "volatility_floor": 0.0,
            },
            strategy_family=FAMILY_EMA_TREND,
        )
        drifted_candidate = current_train.replace(
            "class DualMomentumStrategy:",
            "class ConfigurableMomentumStrategy:",
        ).replace(
            "return DualMomentumStrategy(",
            "return ConfigurableMomentumStrategy(",
        )
        self.assertEqual(
            validate_train_candidate_semantics(
                drifted_candidate,
                current_train_text=current_train,
                symbol_count=5,
            ),
            (False, "missing_family_strategy_class:DualMomentumStrategy"),
        )

    def test_validate_train_candidate_semantics_uses_explicit_family_override(self) -> None:
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
                "min_cross_sectional_spread": 0.0,
                "volatility_floor": 0.0,
                "reversal_bias_weight": 0.0,
                "funding_penalty_weight": 0.0,
            }
        )
        family_candidate = render_family_train_file(
            {
                "lookback_bars": 24,
                "reversion_horizon_bars": 6,
                "ibs_threshold": 0.25,
                "top_k": 1,
                "gross_target": 0.5,
                "reversion_strength_floor": 0.0,
                "volatility_floor": 0.0,
                "use_trend_filter": False,
                "trend_lookback_bars": 48,
            },
            strategy_family=FAMILY_MEAN_REVERSION,
        )
        self.assertEqual(
            validate_train_candidate_semantics(
                family_candidate,
                current_train_text=current_train,
                strategy_family=FAMILY_MEAN_REVERSION,
                symbol_count=5,
            ),
            (True, ""),
        )

    def test_attempt_prompt_uses_family_specific_role_specs(self) -> None:
        role_name, prompt = build_attempt_prompt(
            base_user_prompt="base prompt",
            attempt_index=1,
            max_mutations=3,
            strategy_family=FAMILY_EMA_TREND,
        )
        self.assertEqual(role_name, "absolute_vs_relative_balance")
        self.assertIn("role_name=absolute_vs_relative_balance", prompt)

    def test_attempt_prompt_assigns_role_specific_guidance(self) -> None:
        role_name, prompt = build_attempt_prompt(
            base_user_prompt="base prompt",
            attempt_index=2,
            max_mutations=3,
        )
        self.assertEqual(role_name, "alternative_signal_family")
        self.assertIn("role_name=alternative_signal_family", prompt)
        self.assertIn("role_objective=", prompt)

    def test_validate_train_candidate_semantics_rejects_no_op_and_high_no_trade_risk(self) -> None:
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
                "min_cross_sectional_spread": 0.0,
                "volatility_floor": 0.0,
                "reversal_bias_weight": 0.0,
                "funding_penalty_weight": 0.0,
            }
        )
        self.assertEqual(
            validate_train_candidate_semantics(
                current_train,
                current_train_text=current_train,
                symbol_count=5,
            ),
            (False, "no_op_train_config"),
        )

        risky_candidate = render_train_file(
            {
                "lookback_bars": 24,
                "top_k": 1,
                "gross_target": 0.5,
                "ranking_mode": "risk_adjusted",
                "use_regime_filter": True,
                "regime_lookback_bars": 36,
                "regime_threshold": 0.07,
                "min_signal_strength": 0.11,
                "min_cross_sectional_spread": 0.12,
                "volatility_floor": 0.0,
                "reversal_bias_weight": 0.0,
                "funding_penalty_weight": 0.0,
            }
        )
        self.assertEqual(
            validate_train_candidate_semantics(
                risky_candidate,
                current_train_text=current_train,
                symbol_count=5,
            ),
            (False, "likely_no_trade_filter_stack"),
        )

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

    def test_runner_does_not_keep_no_trade_candidate_even_when_score_improves(self) -> None:
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
            baseline_head = self._git(repo_root, "rev-parse", "HEAD")

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
                artifact_path = artifact_dir / f"{kwargs.get('stage', 'full')}-{kwargs.get('mutation_label', 'x')}.json"
                artifact_path.write_text("{}", encoding="utf-8")
                train_text = Path(kwargs["train_path"]).read_text(encoding="utf-8")
                is_candidate = "'lookback_bars': 12" in train_text
                is_screen = kwargs.get("stage") == "screen"
                score = 0.25 if is_candidate else -1.0
                gate_failures = ["no_trades_executed", "acceptance_rate_below_gate"] if is_candidate and not is_screen else ["acceptance_rate_below_gate"]
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
                        "average_turnover": 0.0 if is_candidate else 0.1,
                        "bars_processed": 10,
                        "nonzero_turnover_steps": 0 if is_candidate else 4,
                    },
                    worst_max_drawdown=0.1,
                    ready_for_paper=False,
                    gate_failures=gate_failures,
                    windows_passed=0,
                    total_windows=1,
                    window_reports=[],
                    runtime_seconds=0.01,
                    artifact_path=str(artifact_path),
                )

            from autoresearch_trade_bot.autoresearch import GitAutoresearchRunner

            runner = GitAutoresearchRunner(
                repo_root=repo_root,
                branch_name="codex/llm-no-trade-guard",
                evaluator=fake_evaluator,
                worktrees_root=repo_root / ".autoresearch" / "worktrees",
            )

            no_trade_candidate = render_train_file(
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
                    label="no-trade-better-score",
                    candidate_text=no_trade_candidate,
                    commit_message="should not keep no-trade candidate",
                    provider_name="llm",
                    model_name="fake-gpt",
                    prompt_id="resp_no_trade",
                ),
                validator=validate_train_candidate_text,
            )

            self.assertEqual(decision.decision, "discard_full")
            self.assertIn("no_trades_executed", decision.report.gate_failures)
            self.assertIsNone(decision.kept_commit)
            self.assertEqual(self._git(runner.worktree_root, "rev-parse", "HEAD"), baseline_head)
            self.assertEqual(runner.train_path.read_text(encoding="utf-8"), baseline_text)

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
