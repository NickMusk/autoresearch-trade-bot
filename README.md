# Autoresearch Trade Bot

`v1` is an agentic research system for crypto trading strategies.

The first milestone is intentionally narrow:

- `Binance-only` historical and realtime interfaces
- deterministic backtest and paper/shadow execution boundaries
- a single baseline strategy: market-neutral cross-sectional momentum
- hard promotion gates before anything is allowed near live capital

The system is designed so that an agent can iterate on `features`, `signal logic`, and `parameters` without being allowed to mutate the execution or risk engine.

## Repository layout

- `docs/v1-architecture.md`: scope, boundaries, and rollout plan
- `src/autoresearch_trade_bot/`: core package
- `tests/`: deterministic tests for the research kernel
- `render.yaml`: Render deployment blueprint for the dashboard app

## Current status

This repository currently contains the research kernel:

- domain models for bars, experiments, and simulation metrics
- baseline cross-sectional momentum strategy
- deterministic backtest engine
- explicit risk limiter
- promotion gate evaluator

Realtime exchange adapters, data persistence, and orchestration are intentionally deferred until the kernel is stable.

## Dashboard

The repository now includes a minimal read-only dashboard.

- `/`: HTML dashboard with baseline metrics and deployment blockers
- `/health`: health probe
- `/api/status`: JSON snapshot

Run locally:

```bash
PYTHONPATH=src python3 -m autoresearch_trade_bot.app
```

## Render

`render.yaml` is included, but a Render deployment still needs a connected git repository or another supported source artifact. The dashboard is deployable in shape, but not source-connected from this workspace yet.
