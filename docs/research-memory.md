# Research Memory

This file stores durable research lessons. It is not a live run log.

Update this file only when a finding is stable enough that future agents should inherit it by default.

## Durable Learnings

### 1. Short evaluation windows were too noisy

The earlier `7d / 3x7d / 8x14d` topology produced weak robustness signals and encouraged local overfitting.

Current default evaluation topology:

- Search: `1 x 30d`
- Fast validation: `3 x 30d`
- Rollout validation: `12 x 30d`

Implication:

- Prefer slower but more meaningful evaluation over fast, noisy cycles.

### 2. Local historical datasets are the correct production path

Workers should use the canonical local dataset rather than rebuilding long history from exchange APIs at startup.

Implication:

- Treat dataset installation and readiness as infrastructure.
- Do not reintroduce startup-time historical backfills as the normal path.

### 3. Binance-style candle history is sufficient for the current research loop

The current loop works on canonicalized OHLCV history without requiring exchange-native Bybit history semantics.

Implication:

- Data-source consistency matters more than preserving historical exchange branding.
- Keep the internal dataset contract canonical and explicit.

### 4. Open interest is not currently a reliable long-horizon requirement

For the current Binance-based history path, long-horizon open interest caused avoidable failures and did not unlock meaningful research progress.

Implication:

- Keep open-interest dependence disabled unless we have a reliable long-horizon source and a clear research reason to reintroduce it.

### 5. Current family bottleneck is candidate quality, not runtime health

Recent family runs show:

- valid code generation
- low acceptance
- frequent `no_trades_executed`
- repeated failures on Sharpe and acceptance-rate gates

Implication:

- Improve family mutation guidance, family bounds, and anti-no-trade semantics before changing score formula or loosening gates.

### 6. Family workers need stronger family-specific memory

Generic prompt memory is not enough for:

- `mean_reversion`
- `ema_trend`
- `volatility_breakout`

Implication:

- Favor family-specific dead-zone detection and family-specific guidance over generic prompt advice.

## What Does Not Belong Here

Do not add:

- latest incident timelines
- current worker timestamps
- transient Render outages
- one-off bad candidates
- stale environment notes

If a note is only useful for the current day or current deploy, it belongs in runtime status or a runbook, not in this file.
