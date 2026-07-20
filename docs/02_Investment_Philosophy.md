# 02 — Investment Philosophy

---

## Core Thesis

Markets are partially efficient, but imperfectly so. Short-lived alpha exists in the form of momentum bursts, mean-reversion windows, and volatility dislocations. No single model reliably captures all of these simultaneously. A population of diverse, competing strategies — continuously evolved under selection pressure — approximates a broader coverage of the opportunity surface than any individual strategy can achieve.

MarketZero operationalises this thesis through Population-Based Training.

---

## Principles

### 1. Diversity Over Conviction
We do not bet on a single strategy. We maintain a heterogeneous population of agents (PPO, Dreamer, WorldModel) whose structural differences produce uncorrelated signals. Ensemble aggregation then harvests the wisdom of the crowd while penalising outlier noise.

### 2. Survival of the Fittest — Continuously
Every generation, bottom-quartile agents are replaced or overwritten by top-quartile hyperparameters. Evolution pressure is never paused. Stale strategies do not survive.

### 3. Risk First
Position sizing, drawdown limits, and circuit breakers are hard constraints, not soft guidelines. The fitness function explicitly penalises drawdown. An agent that returns 10% with 25% drawdown scores lower than one returning 7% with 8% drawdown.

### 4. Regime Awareness
A trend-following configuration that works in a BULL regime will underperform in RANGING or HIGH_VOL regimes. The system continuously detects regime state and the signal aggregator adapts weight allocation accordingly.

### 5. No Overfitting to History
Agents are evaluated on live streaming data, not replayed history. Fitness scores reflect real P&L, not backtested returns. This eliminates look-ahead bias and forces genuine generalisation.

---

## What We Are Not

- **Not a black-box neural network** — all agent decisions are interpretable at the hyperparameter level
- **Not a high-frequency trading system** — target holding period is intraday to multi-day
- **Not a purely discretionary system** — all signals are algorithmic; human override is limited to the risk kill-switch

---

## Alpha Sources Targeted

| Source | Mechanism | Primary Agent Type |
|---|---|---|
| Momentum | EMA crossover, trend strength | PPO |
| Mean reversion | Bollinger band extremes, RSI | WorldModel |
| Volatility premium | ATR expansion signals | Dreamer |
| Regime transitions | Regime detector + signal weight shifts | All |
| Ensemble disagreement | WorldModel uncertainty signal | WorldModel |

---

## Capital Allocation Philosophy

Capital is allocated equally across agents at initialisation (`capital_per_agent = $5,000` by default). Elite agents do not receive more capital — they receive more evolutionary influence. This prevents winner-take-all concentration and maintains diversity in the live trading portfolio.
