from __future__ import annotations

import json
import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from autoresearch_trade_bot.history_refresh import (
    HistoryRefreshConfig,
    history_refresh_config_from_env,
    run_history_refresh_once,
)
from autoresearch_trade_bot.service_runner import run_llm_service


class HistoryRefreshTests(unittest.TestCase):
    def test_history_refresh_config_uses_rollout_horizon_for_bootstrap_default(self) -> None:
        with patch.dict(
            os.environ,
            {
                "AUTORESEARCH_LLM_HISTORY_WINDOW_DAYS": "30",
                "AUTORESEARCH_LLM_CAMPAIGN_WINDOW_COUNT": "1",
                "AUTORESEARCH_LLM_FAST_VALIDATION_WINDOW_DAYS": "30",
                "AUTORESEARCH_LLM_FAST_VALIDATION_WINDOW_COUNT": "3",
                "AUTORESEARCH_LLM_ROLLOUT_VALIDATION_WINDOW_DAYS": "30",
                "AUTORESEARCH_LLM_ROLLOUT_VALIDATION_WINDOW_COUNT": "12",
            },
            clear=False,
        ):
            config = history_refresh_config_from_env()
        self.assertEqual(config.bootstrap_lookback_days, 360)

    def test_run_history_refresh_once_writes_state_and_can_skip_open_interest(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            temp_path = Path(tempdir)
            config = HistoryRefreshConfig(
                exchange="bybit",
                market="linear",
                timeframe="5m",
                symbols=("BTCUSDT", "ETHUSDT"),
                storage_root=str(temp_path / "data"),
                full_lookback_days=365,
                bootstrap_lookback_days=360,
                refresh_interval_seconds=86400,
                state_path=str(temp_path / "state.json"),
                min_request_interval_seconds=0.25,
                rate_limit_max_retries=6,
                rate_limit_backoff_seconds=2.0,
                bootstrap_skip_open_interest=True,
                refresh_skip_open_interest=False,
            )

            captured = {}

            def fake_for_exchange(data_config):
                captured["include_open_interest"] = data_config.include_open_interest
                return SimpleNamespace(store=object())

            manifest_path = temp_path / "data" / "manifest.json"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text("{}", encoding="utf-8")

            with patch(
                "autoresearch_trade_bot.history_refresh.HistoricalDatasetMaterializer.for_exchange",
                side_effect=fake_for_exchange,
            ):
                with patch(
                    "autoresearch_trade_bot.history_refresh.ensure_dataset_manifest",
                    return_value=manifest_path,
                ):
                    returned = run_history_refresh_once(
                        config=config,
                        lookback_days=360,
                        skip_open_interest=True,
                        now_fn=lambda: datetime(2026, 3, 22, 12, 0, tzinfo=timezone.utc),
                    )

            self.assertEqual(returned, manifest_path)
            self.assertFalse(captured["include_open_interest"])
            state = json.loads(Path(config.state_path).read_text(encoding="utf-8"))
            self.assertEqual(state["manifest_path"], str(manifest_path))
            self.assertEqual(state["lookback_days"], 360)
            self.assertTrue(state["skip_open_interest"])


class ServiceRunnerTests(unittest.TestCase):
    def test_run_llm_service_bootstraps_then_spawns_daemon_before_worker(self) -> None:
        popen_calls = []

        class FakeProcess:
            def __init__(self, mode: str, wait_result: int = 0) -> None:
                self.mode = mode
                self._wait_result = wait_result

            def wait(self, timeout: float | None = None) -> int:
                return self._wait_result

            def poll(self):
                return None

            def terminate(self) -> None:
                return None

            def kill(self) -> None:
                return None

        def fake_popen(args, env=None):
            mode = env["AUTORESEARCH_HISTORY_REFRESH_MODE"]
            popen_calls.append(mode)
            return FakeProcess(mode=mode)

        with patch.dict(os.environ, {"AUTORESEARCH_ENABLE_HISTORY_REFRESHER": "1"}, clear=False):
            with patch("autoresearch_trade_bot.service_runner.subprocess.Popen", side_effect=fake_popen):
                with patch("autoresearch_trade_bot.service_runner.llm_worker.main") as mocked_main:
                    run_llm_service()

        self.assertEqual(popen_calls, ["bootstrap", "daemon"])
        self.assertEqual(mocked_main.call_count, 1)


if __name__ == "__main__":
    unittest.main()
