# Repo Map

This file is a compact map of the codebase. It is for navigation, not live status.

## Core Research Loop

- `src/autoresearch_trade_bot/llm_worker.py`
  - Orchestrates the LLM autoresearch loop.
  - Builds campaigns, runs mutations, updates candidate registry, publishes status, and writes snapshots.

- `src/autoresearch_trade_bot/mutations.py`
  - Builds mutation context and prompt payloads.
  - Produces experiment memory summaries from recent results.
  - Runs LLM-based mutation campaigns.

- `src/autoresearch_trade_bot/autoresearch.py`
  - Runs staged strategy evaluation.
  - Applies screen and full campaign decisions.
  - Aggregates metrics and gate failures.

- `src/autoresearch_trade_bot/research.py`
  - Holds the basic research score formula and experiment evaluator.

- `src/autoresearch_trade_bot/rollout.py`
  - Tracks fast validation and rollout validation summaries.
  - Builds research champion and rollout champion summaries.

## Strategy Family System

- `src/autoresearch_trade_bot/strategy_families.py`
  - Defines family profiles, default configs, mutation bounds, deterministic mutation specs, and semantic validators.
  - First place to inspect when families stagnate or overfit.

- `src/autoresearch_trade_bot/family_wave.py`
  - Runs family-focused preparation and mutation workflows.

## Data And Campaigns

- `src/autoresearch_trade_bot/data.py`
  - Historical data loading and manifest-backed data access.

- `src/autoresearch_trade_bot/storage.py`
  - Parquet and manifest dataset storage helpers.

- `src/autoresearch_trade_bot/history_refresh.py`
  - Computes required history coverage and readiness state.

- `src/autoresearch_trade_bot/crypto_import.py`
  - Imports external bulk crypto history into canonical project datasets.

## Runtime And State

- `src/autoresearch_trade_bot/config.py`
  - Default runtime config, including evaluation windows and mutation limits.

- `src/autoresearch_trade_bot/state.py`
  - Snapshot and state serialization helpers.

- Worker state roots
  - `state_root`: published local snapshots, checkpoints, and candidate registry state.
  - `artifact_root`: validation artifacts, prompt artifacts, experiment memory, and materialized sources.

## Dashboard And Status

- `src/autoresearch_trade_bot/app.py`
  - Renders dashboard HTML and API responses.

- `src/autoresearch_trade_bot/dashboard.py`
  - Status freshness helpers and dashboard support logic.

- GitHub status publishing
  - `render-state/status/...`
  - Used as the externally visible operational status source for the dashboard.

## Deployment

- `render.yaml`
  - Declarative deployment intent for Render services and env vars.

- `program.md`
  - Mission and immutable research rules.

- `AGENTS.md`
  - Repo operating contract and memory contract.

## Where To Look First

- Strategy quality is poor or families stagnate:
  - `strategy_families.py`
  - `mutations.py`
  - `autoresearch.py`

- Status looks wrong:
  - `llm_worker.py`
  - `dashboard.py`
  - `app.py`

- Data or history readiness is wrong:
  - `data.py`
  - `storage.py`
  - `history_refresh.py`

- Deploy drift or worker behavior differs from repo config:
  - `render.yaml`
  - `config.py`

## Current Evaluation Topology

As of the current design:

- Search window: `1 x 30d`
- Fast validation: `3 x 30d`
- Rollout validation: `12 x 30d`

Treat these as implementation defaults; change them in code and deployment config together.
