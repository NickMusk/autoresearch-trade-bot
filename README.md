# Autoresearch Trade Bot

`v1` is an agentic research system for crypto trading strategies.

The first milestone is intentionally narrow:

- `Bybit-first` frozen-campaign LLM autoresearch worker on Render, with Binance historical support still available
- deterministic backtest and paper/shadow execution boundaries
- a single baseline strategy: market-neutral cross-sectional momentum
- hard promotion gates before anything is allowed near live capital

The repository now also contains a Karpathy-style offline autoresearch core:

- root-level `program.md` for human-owned research policy
- root-level `prepare.py` to freeze named comparable research campaigns
- root-level `train.py` as the single editable strategy surface
- `results.tsv` as the untracked experiment ledger
- dedicated git worktree `keep/discard` semantics through the autoresearch runner
- staged `screen -> full campaign` evaluation before a mutation can be kept
- pluggable mutation providers, including deterministic and LLM-backed candidates

The system is designed so that an agent can iterate on `features`, `signal logic`, and `parameters` without being allowed to mutate the execution or risk engine.

## Repository layout

- `docs/v1-architecture.md`: scope, boundaries, and rollout plan
- `src/autoresearch_trade_bot/`: core package
- `tests/`: deterministic tests for the research kernel
- `render.yaml`: Render deployment blueprint for the dashboard and single LLM worker

## Current status

This repository currently contains the research kernel:

- domain models for bars, experiments, and simulation metrics
- baseline cross-sectional momentum strategy
- deterministic backtest engine
- explicit risk limiter
- promotion gate evaluator
- exchange-specific historical dataset contracts, validation pipeline, and parquet storage interface

The repository now also contains a long-running LLM autoresearch worker with:

- frozen campaign preparation and refresh
- staged `screen -> full` mutation evaluation
- persisted `status`, `leaderboard`, `history`, `checkpoint`, and proposal artifacts
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

Run one legacy variant-search worker cycle locally:

```bash
PYTHONPATH=src python3 -m autoresearch_trade_bot.cli run-worker-cycle
```

Prepare a named frozen autoresearch campaign:

```bash
python3 prepare.py \
  --campaign-name crypto-bybit-5m-mar17 \
  --end 2026-03-17T00:00:00Z
```

Evaluate the current editable `train.py` against the active frozen campaign:

```bash
PYTHONPATH=src python3 -m autoresearch_trade_bot.cli eval-autoresearch \
  --train-path train.py
```

Apply a candidate `train.py` mutation on a research branch and keep it only if the score improves:

```bash
PYTHONPATH=src python3 -m autoresearch_trade_bot.cli apply-autoresearch-candidate \
  --candidate-file /tmp/train_candidate.py \
  --branch-name codex/autoresearch-crypto \
  --worktrees-root .autoresearch/worktrees \
  --commit-message "Try a new train.py mutation"
```

Run the built-in deterministic mutation batch against the active campaign:

```bash
PYTHONPATH=src python3 -m autoresearch_trade_bot.cli run-deterministic-autoresearch \
  --branch-name codex/autoresearch-crypto \
  --worktrees-root .autoresearch/worktrees \
  --max-mutations 8
```

Run an LLM mutation batch against the active campaign:

```bash
OPENAI_API_KEY=... \
PYTHONPATH=src python3 -m autoresearch_trade_bot.cli run-llm-autoresearch \
  --branch-name codex/autoresearch-crypto \
  --worktrees-root .autoresearch/worktrees \
  --model-name gpt-5-mini \
  --max-mutations 1
```

The LLM path keeps the same staged harness:

- candidate validation before backtests
- `screen -> full` evaluation
- `skip_duplicate`, `discard_error`, `discard_screen`, `discard_full`, or `keep`
- prompt/response provenance artifacts under `.autoresearch/runs/proposals/`

## Render

The dashboard is deployed at [autoresearch-trade-bot.onrender.com](https://autoresearch-trade-bot.onrender.com/).

Render now targets a single backend worker model:

- `web`: read-only dashboard
- `worker`: long-running LLM autoresearch loop on a mounted disk

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

The Render LLM worker expects:

- `AUTORESEARCH_LLM_DATA_ROOT`, `AUTORESEARCH_LLM_STATE_ROOT`, `AUTORESEARCH_LLM_ARTIFACT_ROOT`
- `AUTORESEARCH_LLM_REPO_ROOT`, `AUTORESEARCH_LLM_CAMPAIGNS_ROOT`, `AUTORESEARCH_LLM_WORKTREES_ROOT`
- `AUTORESEARCH_LLM_RESULTS_PATH`, `AUTORESEARCH_LLM_MODEL_NAME`, `AUTORESEARCH_LLM_MAX_MUTATIONS_PER_CYCLE`
- `AUTORESEARCH_STATUS_GITHUB_REPO`
- `OPENAI_API_KEY`
- `GITHUB_TOKEN` or `GH_TOKEN` if status should be published for the dashboard

The dashboard can read a persisted worker snapshot from:

- `AUTORESEARCH_STATUS_PATH` for local/shared files
- `AUTORESEARCH_STATUS_URL` for a published JSON artifact such as a raw GitHub URL
