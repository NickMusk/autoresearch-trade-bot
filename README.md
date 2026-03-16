# Autoresearch Trade Bot

`v1` is an agentic research system for crypto trading strategies.

The first milestone is intentionally narrow:

- `Bybit-first` continuous research worker, with Binance historical support still available
- deterministic backtest and paper/shadow execution boundaries
- a single baseline strategy: market-neutral cross-sectional momentum
- hard promotion gates before anything is allowed near live capital

The system is designed so that an agent can iterate on `features`, `signal logic`, and `parameters` without being allowed to mutate the execution or risk engine.

## Repository layout

- `docs/v1-architecture.md`: scope, boundaries, and rollout plan
- `src/autoresearch_trade_bot/`: core package
- `tests/`: deterministic tests for the research kernel
- `render.yaml`: Render deployment blueprint for the dashboard app
- `render.yaml`: Render deployment blueprint for the dashboard and continuous worker

## Current status

This repository currently contains the research kernel:

- domain models for bars, experiments, and simulation metrics
- baseline cross-sectional momentum strategy
- deterministic backtest engine
- explicit risk limiter
- promotion gate evaluator
- exchange-specific historical dataset contracts, validation pipeline, and parquet storage interface

The repository now also contains a continuous research worker with:

- bar-aligned `5m` research cycles
- bounded parameter search around the baseline momentum strategy
- persisted `status`, `leaderboard`, `history`, and `checkpoint` artifacts
- optional GitHub-backed status publishing for the Render dashboard

## Dashboard

The repository now includes a minimal read-only dashboard.

- `/`: HTML dashboard with baseline metrics and deployment blockers
- `/health`: health probe
- `/api/status`: JSON snapshot

Run locally:

```bash
PYTHONPATH=src python3 -m autoresearch_trade_bot.app
```

Run one research worker cycle locally:

```bash
PYTHONPATH=src python3 -m autoresearch_trade_bot.cli run-worker-cycle
```

## Render

The dashboard is deployed at [autoresearch-trade-bot.onrender.com](https://autoresearch-trade-bot.onrender.com/).

## Historical Data

The repository now includes Binance and Bybit historical data providers:

- `DatasetSpec` and `DatasetManifest` for reproducible dataset windows
- `HistoricalDatasetMaterializer` for `raw -> normalize -> validate -> store`
- `PyArrowParquetDatasetStore` for parquet-backed symbol datasets
- `ManifestHistoricalDataSource` for loading validated bars back into the simulator

The parquet path requires project dependencies to be installed:

```bash
pip install '.[data]'
```

Materialize a real Binance dataset:

```bash
PYTHONPATH=src python3 -m autoresearch_trade_bot.cli materialize-binance \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT \
  --timeframe 5m \
  --start 2025-01-01 \
  --end 2025-01-08
```

Materialize a real Bybit dataset:

```bash
PYTHONPATH=src python3 -m autoresearch_trade_bot.cli materialize-bybit \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT \
  --timeframe 5m \
  --start 2025-01-01 \
  --end 2025-01-08
```

Run the baseline strategy against a materialized manifest:

```bash
PYTHONPATH=src python3 -m autoresearch_trade_bot.cli run-baseline \
  --manifest-path data/.../manifest.json
```

## Continuous Worker

The Render worker expects:

- `AUTORESEARCH_DATA_ROOT`, `AUTORESEARCH_STATE_ROOT`, `AUTORESEARCH_ARTIFACT_ROOT`
- `AUTORESEARCH_STATUS_GITHUB_REPO`
- `GITHUB_TOKEN` or `GH_TOKEN` if status should be published for the dashboard

The dashboard can read a persisted worker snapshot from:

- `AUTORESEARCH_STATUS_PATH` for local/shared files
- `AUTORESEARCH_STATUS_URL` for a published JSON artifact such as a raw GitHub URL
