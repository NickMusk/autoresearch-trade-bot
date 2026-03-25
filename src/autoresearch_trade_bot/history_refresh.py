from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from .config import DataConfig
from .data import (
    HistoricalDatasetMaterializer,
    default_history_readiness_state_path,
    ensure_dataset_manifest,
)
from .datasets import DatasetSpec


def _parse_bool(value: str | None, *, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_symbols(raw: str | None) -> tuple[str, ...]:
    if not raw:
        raw = "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT"
    return tuple(symbol.strip().upper() for symbol in raw.split(",") if symbol.strip())


@dataclass(frozen=True)
class HistoryRefreshConfig:
    exchange: str
    market: str
    timeframe: str
    symbols: tuple[str, ...]
    storage_root: str
    full_lookback_days: int
    bootstrap_lookback_days: int
    refresh_interval_seconds: int
    state_path: str
    min_request_interval_seconds: float
    rate_limit_max_retries: int
    rate_limit_backoff_seconds: float
    bootstrap_skip_open_interest: bool
    refresh_skip_open_interest: bool


def history_refresh_config_from_env() -> HistoryRefreshConfig:
    history_window_days = int(os.environ.get("AUTORESEARCH_LLM_HISTORY_WINDOW_DAYS", "30"))
    campaign_window_count = int(os.environ.get("AUTORESEARCH_LLM_CAMPAIGN_WINDOW_COUNT", "1"))
    fast_validation_window_days = int(
        os.environ.get("AUTORESEARCH_LLM_FAST_VALIDATION_WINDOW_DAYS", str(history_window_days))
    )
    fast_validation_window_count = int(
        os.environ.get("AUTORESEARCH_LLM_FAST_VALIDATION_WINDOW_COUNT", "3")
    )
    rollout_validation_window_days = int(
        os.environ.get("AUTORESEARCH_LLM_ROLLOUT_VALIDATION_WINDOW_DAYS", "30")
    )
    rollout_validation_window_count = int(
        os.environ.get("AUTORESEARCH_LLM_ROLLOUT_VALIDATION_WINDOW_COUNT", "12")
    )
    bootstrap_default_days = max(
        history_window_days * max(campaign_window_count, 1),
        fast_validation_window_days * max(fast_validation_window_count, 1),
        rollout_validation_window_days * max(rollout_validation_window_count, 1),
    )
    storage_root = (
        os.environ.get("AUTORESEARCH_LLM_SHARED_DATA_ROOT")
        or os.environ.get("AUTORESEARCH_SHARED_DATA_ROOT")
        or os.environ.get("AUTORESEARCH_LLM_DATA_ROOT")
        or os.environ.get("AUTORESEARCH_DATA_ROOT")
        or "data"
    )
    return HistoryRefreshConfig(
        exchange=os.environ.get("AUTORESEARCH_EXCHANGE", "bybit"),
        market=os.environ.get("AUTORESEARCH_MARKET", "linear"),
        timeframe=os.environ.get("AUTORESEARCH_TIMEFRAME", "5m"),
        symbols=_parse_symbols(os.environ.get("AUTORESEARCH_SYMBOLS")),
        storage_root=storage_root,
        full_lookback_days=int(os.environ.get("AUTORESEARCH_HISTORY_WARM_LOOKBACK_DAYS", "365")),
        bootstrap_lookback_days=int(
            os.environ.get(
                "AUTORESEARCH_HISTORY_BOOTSTRAP_LOOKBACK_DAYS",
                str(bootstrap_default_days),
            )
        ),
        refresh_interval_seconds=int(
            os.environ.get("AUTORESEARCH_HISTORY_REFRESH_INTERVAL_SECONDS", "86400")
        ),
        state_path=os.environ.get(
            "AUTORESEARCH_HISTORY_REFRESH_STATE_PATH",
            str(default_history_readiness_state_path(storage_root)),
        ),
        min_request_interval_seconds=float(
            os.environ.get("AUTORESEARCH_MIN_REQUEST_INTERVAL_SECONDS", "0.25")
        ),
        rate_limit_max_retries=int(
            os.environ.get("AUTORESEARCH_RATE_LIMIT_MAX_RETRIES", "6")
        ),
        rate_limit_backoff_seconds=float(
            os.environ.get("AUTORESEARCH_RATE_LIMIT_BACKOFF_SECONDS", "2.0")
        ),
        bootstrap_skip_open_interest=_parse_bool(
            os.environ.get("AUTORESEARCH_HISTORY_BOOTSTRAP_SKIP_OPEN_INTEREST"),
            default=True,
        ),
        refresh_skip_open_interest=_parse_bool(
            os.environ.get("AUTORESEARCH_HISTORY_REFRESH_SKIP_OPEN_INTEREST"),
            default=False,
        ),
    )


def build_history_spec(
    *,
    config: HistoryRefreshConfig,
    lookback_days: int,
    now: datetime,
) -> DatasetSpec:
    anchor_end = now.astimezone(timezone.utc)
    return DatasetSpec(
        exchange=config.exchange,
        market=config.market,
        timeframe=config.timeframe,
        start=anchor_end - timedelta(days=lookback_days),
        end=anchor_end,
        symbols=config.symbols,
    )


def run_history_refresh_once(
    *,
    config: HistoryRefreshConfig,
    lookback_days: int,
    skip_open_interest: bool,
    now_fn: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> Path:
    spec = build_history_spec(config=config, lookback_days=lookback_days, now=now_fn())
    data_config = DataConfig(
        exchange=config.exchange,
        market=config.market,
        timeframe=config.timeframe,
        storage_root=config.storage_root,
        default_start=spec.start,
        default_end=spec.end,
        min_request_interval_seconds=config.min_request_interval_seconds,
        rate_limit_max_retries=config.rate_limit_max_retries,
        rate_limit_backoff_seconds=config.rate_limit_backoff_seconds,
        include_open_interest=not skip_open_interest,
    )
    materializer = HistoricalDatasetMaterializer.for_exchange(data_config)
    manifest_path = ensure_dataset_manifest(
        storage_root=config.storage_root,
        spec=spec,
        materializer=materializer,
    )
    _write_refresh_state(
        config=config,
        manifest_path=manifest_path,
        lookback_days=lookback_days,
        skip_open_interest=skip_open_interest,
        completed_at=now_fn(),
    )
    return manifest_path


def run_history_refresh_forever(
    *,
    config: HistoryRefreshConfig,
    now_fn: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    sleep_fn: Callable[[float], None] = time.sleep,
) -> None:
    while True:
        run_history_refresh_once(
            config=config,
            lookback_days=config.full_lookback_days,
            skip_open_interest=config.refresh_skip_open_interest,
            now_fn=now_fn,
        )
        sleep_fn(float(config.refresh_interval_seconds))


def _write_refresh_state(
    *,
    config: HistoryRefreshConfig,
    manifest_path: Path,
    lookback_days: int,
    skip_open_interest: bool,
    completed_at: datetime,
) -> None:
    state_path = Path(config.state_path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "manifest_path": str(manifest_path),
        "storage_root": config.storage_root,
        "lookback_days": lookback_days,
        "skip_open_interest": skip_open_interest,
        "completed_at": completed_at.astimezone(timezone.utc).isoformat(),
    }
    state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    mode = os.environ.get("AUTORESEARCH_HISTORY_REFRESH_MODE", "daemon").strip().lower()
    config = history_refresh_config_from_env()
    if mode == "bootstrap":
        run_history_refresh_once(
            config=config,
            lookback_days=config.bootstrap_lookback_days,
            skip_open_interest=config.bootstrap_skip_open_interest,
        )
        return
    if mode == "once":
        run_history_refresh_once(
            config=config,
            lookback_days=config.full_lookback_days,
            skip_open_interest=config.refresh_skip_open_interest,
        )
        return
    run_history_refresh_forever(config=config)


if __name__ == "__main__":
    main()
