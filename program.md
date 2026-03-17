# Crypto Autoresearch Program

This repository follows a Karpathy-style research loop.

Rules:

1. `prepare.py` and the harness under `src/autoresearch_trade_bot/` are immutable research infrastructure.
2. `train.py` is the only editable research surface.
3. Optimize only one primary metric: `research_score`.
4. Respect hard gates for reporting and rollout:
   - average `total_return >= 0.0`
   - average `sharpe >= 1.0`
   - worst `max_drawdown <= 0.20`
   - `acceptance_rate >= 0.60`
5. Every candidate strategy must be evaluated on the frozen windows created by `prepare.py`.
6. Keep a candidate only if its `research_score` improves over the current branch head.
7. Record every evaluated candidate in `results.tsv`.
8. Do not modify data ingestion, simulator, risk engine, dashboard, or deployment code while doing autoresearch.

Research intent:

- Find a market-neutral perpetual futures strategy that is robust across frozen windows.
- Prefer simple, explicit strategy logic over clever heuristics.
- Reduce drawdown and turnover while improving `research_score`.
