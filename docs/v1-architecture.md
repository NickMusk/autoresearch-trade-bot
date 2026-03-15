# V1 Architecture

## Mission

Build a research system that can propose, test, rank, and reject crypto strategies before they are allowed into paper, shadow, or live trading.

## One-sentence outcome

Given historical data and realtime market events, the system should answer:

`Is there a strategy that passes our promotion gate right now?`

## Principles

- Keep the research loop cheap enough for many daily iterations.
- Use one deterministic portfolio and execution model across backtest and paper/shadow.
- Let the agent edit only strategy logic, features, and parameters.
- Keep risk controls outside the agent's editable surface.
- Prefer explicit, reproducible experiments over clever automation.

## V1 system boundaries

### In scope

- historical bar-based backtesting
- baseline market-neutral cross-sectional momentum strategy
- deterministic risk and fee model
- experiment scoring and promotion gating
- interfaces for future historical and realtime data sources

### Out of scope

- live order routing
- exchange account management
- multi-exchange smart order execution
- tick-level market making
- reinforcement learning

## Core modules

### `data`

Responsibilities:

- load validated historical bars
- expose a future interface for realtime bar events
- reject malformed, duplicated, or misaligned inputs

Opinionated choice:

- start with `1m` and `5m` bars, plus funding and open interest fields
- defer order book and liquidation data to a later stage

### `strategy`

Responsibilities:

- read market history only
- output target portfolio weights

Constraints:

- no direct access to execution state outside current portfolio weights
- no exchange-specific logic
- no side effects

### `risk`

Responsibilities:

- cap per-symbol exposure
- cap gross leverage
- cap turnover per step

This layer must remain explicit and separate from strategy logic so research cannot cheat by hiding leverage or unrealistic churn.

### `simulator`

Responsibilities:

- replay aligned bars deterministically
- apply returns, fees, funding, and rebalances in a fixed order
- compute metrics needed for promotion

Execution order per bar:

1. realize returns from the prior portfolio
2. ask the strategy for new target weights using current history
3. apply risk limits
4. apply transaction costs
5. carry the new portfolio into the next bar

### `research`

Responsibilities:

- evaluate an experiment
- score the result
- decide `go/no-go` using promotion gates

The agent should eventually operate only through this module.

## Baseline strategy

Use market-neutral cross-sectional momentum on liquid perpetuals:

- universe: top liquid perpetuals by rolling volume
- signal: normalized momentum over a fixed lookback
- construction: long strongest names, short weakest names
- sizing: volatility-aware signal normalization, then risk caps

Why this baseline:

- easier to test than market making
- less dependent on microstructure
- better fit for a fast autoresearch loop

## Promotion gate

No strategy is promoted unless it passes all of these:

- minimum total return
- minimum Sharpe ratio
- maximum drawdown
- maximum average turnover

Later versions should add:

- fee sensitivity
- walk-forward stability
- regime segmentation
- paper/shadow consistency checks

## Data plan

### Historical

Phase 1 fields:

- timestamp
- open/high/low/close
- volume
- funding rate
- open interest

### Realtime

Phase 1 target:

- websocket bar stream into the same strategy and portfolio interfaces
- no separate strategy implementation for paper/shadow

## Rollout plan

1. Stabilize the research kernel with deterministic tests.
2. Add historical ingestion and dataset validation for Binance.
3. Add realtime paper/shadow replay using the same engine boundaries.
4. Only after stable paper/shadow results, discuss live capital rollout.

## Current implementation status

Implemented:

- deterministic research kernel
- dashboard deployment on Render
- dataset spec and manifest contracts
- Binance historical normalization and validation pipeline
- parquet-backed storage interface for validated datasets

Still pending:

- first materialized real Binance dataset
- walk-forward persistence
- realtime paper/shadow ingestion
