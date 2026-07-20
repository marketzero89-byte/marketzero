# 03 — Research Methodology

---

## Research Workflow

MarketZero follows a structured research-to-production pipeline with four phases:

```
Hypothesis → Offline Backtest → Paper Trading Validation → Live Deployment
```

No strategy reaches live capital without completing all four phases.

---

## Phase 1 — Hypothesis Formation

Each research cycle begins with a falsifiable hypothesis about a market inefficiency. Examples:

- "EMA crossover signals on 15-minute AAPL bars persist for ≥ 2 bars in BULL regimes"
- "WorldModel ensemble disagreement predicts realised volatility increases 30 minutes ahead"
- "RSI-14 mean reversion on SPY outperforms RSI-7 in RANGING regimes"

Hypotheses are logged in `experiments/` with a timestamp, author, and outcome field that is filled after validation.

---

## Phase 2 — Offline Backtesting

```bash
python main.py backtest --trend bull
```

The synthetic backtester in `trading/backtest_metrics.py` evaluates an equity curve against six metrics:

| Metric | Formula |
|---|---|
| Sharpe | `mean(r) / std(r) × √252` |
| Calmar | `CAGR / abs(MaxDrawdown)` |
| Sortino | `mean(r) / downside_std(r) × √252` |
| Max Drawdown | `max(peak - trough) / peak` |
| Win Rate | `wins / total_trades` |
| Profit Factor | `gross_profit / gross_loss` |

A strategy proceeds to Phase 3 only if: Sharpe ≥ 0.8, MaxDrawdown ≤ 25%, Win Rate ≥ 48%.

---

## Phase 3 — Paper Trading Validation

Validated strategies are added to the agent population and run in paper mode:

```bash
python main.py serve --broker-mode paper --population-size 12
```

Paper mode uses live market prices (Alpaca data feed) but executes no real orders. Performance is logged to `logs/pbt_orchestrator.json`. Validation period: minimum 5 trading days or 10 completed generations.

A strategy graduates to live capital if its live-paper fitness score exceeds 0.3 and its Sharpe exceeds 1.0.

---

## Phase 4 — Live Deployment

```bash
export ALPACA_API_KEY=...
export ALPACA_SECRET_KEY=...
python main.py serve --broker-mode live
```

Live deployment uses the same codebase as paper mode. The only change is `broker_mode = "live"`, which routes orders through the Alpaca REST API instead of the paper simulator.

---

## Experimental Logging

All experiments are tracked in `experiments/`. Each experiment directory contains:

```
experiments/
└── YYYYMMDD_hypothesis_name/
    ├── config.yaml          # hyperparameters used
    ├── results.json         # fitness, Sharpe, drawdown outputs
    ├── equity_curve.csv     # step-by-step equity
    └── notes.md             # qualitative observations
```

---

## Statistical Rigour

- All backtest results report 95% confidence intervals where applicable
- Walk-forward analysis is preferred over single in-sample backtests
- Minimum 252 trading-day sample required for Sharpe significance
- Results in `results/` are never overwritten — each run appends a new timestamped file

---

## Avoiding Research Pitfalls

| Pitfall | Mitigation |
|---|---|
| Overfitting | Live paper validation required before live capital |
| Look-ahead bias | Online evaluation only — no historical lookahead |
| Survivorship bias | Full agent population tracked, including eliminated agents |
| Data snooping | Hypothesis must be logged before backtest is run |
| Regime mismatch | All strategies tested across BULL, BEAR, RANGING, HIGH_VOL regimes |
