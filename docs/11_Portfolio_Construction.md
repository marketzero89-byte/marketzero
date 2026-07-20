# 11 — Portfolio Construction

---

## Portfolio Model

MarketZero uses a **flat equal-weight portfolio model** at the agent level combined with **dynamic signal-weighted position sizing** at the execution level.

Each agent is allocated `capital_per_agent` at initialisation. Agents do not compete for capital — they each trade their own allocation independently. The aggregate portfolio equity is the sum of all agent portfolios.

---

## Position Sizing

Position size for each trade is determined by three factors:

```
raw_size    = agent.confidence × agent.max_position_size × capital_per_agent
risk_scaled = raw_size × (1 - drawdown_penalty)
final_size  = min(risk_scaled, max_position_size × portfolio_equity)
```

Where `max_position_size = 0.05` (5% of portfolio per position, default).

### Volatility Scaling

When ATR is available, position size is further scaled:

```
vol_scalar = target_vol / realised_vol
           = 0.01 / (atr / price)

final_size = final_size × clamp(vol_scalar, 0.5, 2.0)
```

This reduces size in high-volatility periods and increases it in low-volatility periods, targeting a consistent portfolio volatility of ~1% per step.

---

## Signal Aggregation → Portfolio Action

The `SignalAggregator` (ENSEMBLE mode) produces a single portfolio-level signal:

```python
# Inputs: list of Signal objects from all agents
aggregate = SignalAggregator.aggregate(signals, mode="ENSEMBLE")

# A trade executes only if all three conditions are met:
if (aggregate.confidence >= confidence_threshold   # default 0.55
    and aggregate.agreement >= 0.55                # >55% agents agree on direction
    and abs(aggregate.direction) > 0.10):          # non-trivial directional view
    broker.submit_order(...)
```

---

## Aggregation Modes

| Mode | Weight Basis | Best For |
|---|---|---|
| `EQUAL` | Uniform | Debugging / baseline |
| `FITNESS` | Rolling fitness score | Stable markets |
| `CONFIDENCE` | Per-step confidence | High-uncertainty regimes |
| `RANK` | Fitness rank (not score) | Robust to outliers |
| `ENSEMBLE` | 60% fitness + 40% confidence | Default (production) |

---

## Multi-Symbol Portfolio

When trading multiple symbols, each symbol is treated as an independent portfolio dimension. The system does not optimise cross-symbol correlation or apply mean-variance optimisation in v1.0.

Each step:
1. All agents produce signals for all symbols (each agent tracks all symbols)
2. Signals are aggregated per symbol independently
3. Position limits are enforced per symbol
4. Total portfolio exposure check: `Σ |position_i| / equity ≤ 0.95`

---

## Cash Management

Uninvested cash sits idle (no cash yield in paper mode). In live mode, Alpaca automatically applies FDIC/SIPC protections on idle cash. Cash balance is tracked in `PortfolioState.cash` and displayed on the dashboard.

Minimum cash buffer: 5% of total capital is always kept in cash to ensure order execution is never blocked by insufficient funds.

---

## Rebalancing

The portfolio does not rebalance on a schedule. Position sizes change only when:
- A new trade signal is executed (increases or reverses a position)
- A stop-loss or take-profit is triggered
- The risk manager forces a position reduction

---

## Default Configuration

| Parameter | Default | Description |
|---|---|---|
| `total_capital` | $120,000 | Total portfolio AUM |
| `capital_per_agent` | $5,000 | Allocation per agent |
| `max_position_size` | 5% | Max single position size |
| `max_portfolio_exposure` | 95% | Max total long + short exposure |
| `max_leverage` | 2× | Maximum gross leverage |
