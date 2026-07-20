# 15 — Performance Analytics

---

## Real-Time Analytics

Performance analytics are computed continuously during live trading and exposed via three channels:

1. **WebSocket dashboard** — equity curve, leaderboard, fitness trends (browser)
2. **JSON logs** — machine-readable per-step records (`logs/pbt_orchestrator.json`)
3. **TensorBoard** — scalar trends over time (`logs/tensorboard/`)

---

## Population-Level Metrics

Computed by `OnlineEvaluator` at each generation end:

| Metric | Description |
|---|---|
| `avg_fitness` | Mean fitness across all agents |
| `best_fitness` | Fitness of top-ranked agent |
| `worst_fitness` | Fitness of bottom-ranked agent |
| `fitness_std` | Standard deviation of fitness (measures diversity) |
| `n_agents` | Active agent count |

A high `fitness_std` indicates a diverse population (healthy). A low `fitness_std` indicates convergence (may need increased mutation strength).

---

## Agent-Level Metrics

Each agent on the leaderboard exposes:

| Field | Type | Description |
|---|---|---|
| `agent_id` | str | Unique agent identifier |
| `agent_type` | str | PPO / Dreamer / WorldModel |
| `fitness` | float | Composite fitness ∈ [-1, 1] |
| `sharpe_ratio` | float | Rolling Sharpe |
| `total_return_pct` | float | Cumulative return % |
| `max_drawdown` | float | Worst peak-to-trough |
| `win_rate` | float | Fraction of profitable trades |
| `n_trades` | int | Total trades executed |

---

## Portfolio-Level Analytics

The portfolio state object provides:

```json
{
  "equity": 125400.00,
  "cash": 95200.00,
  "positions": {"AAPL": 12.5, "SPY": 3.0},
  "total_trades": 47,
  "daily_pnl": 400.00,
  "total_return_pct": 4.5
}
```

---

## Equity Curve Analysis

The dashboard computes rolling analytics from the equity history buffer (last 600 steps):

```javascript
// Annualised volatility (rolling 30-step)
vol = std(returns_30) × √252

// Drawdown from peak
drawdown = (current_equity - peak_equity) / peak_equity × 100
```

Both are displayed in the dashboard header: `vol 0.182 | drawdown -1.23%`

---

## Fitness Evolution Tracking

The fitness chart tracks `best_fitness` and `avg_fitness` per generation. A healthy system shows:

- `best_fitness` trending upward over generations (elite agents improving)
- `avg_fitness` tracking `best_fitness` with a lag (selection pressure working)
- Occasional dips followed by recovery (exploration finding better regions)

If `avg_fitness` flatlines while `best_fitness` improves, the population is converging — increase `mutation_strength` or `population_size`.

---

## Interpreting the Leaderboard

```
Rank  Agent ID        Type      Fitness  Sharpe
 1    PPO_a3f2b1      PPO       +0.847   1.20
 2    Dreamer_9c41    Dreamer   +0.821   1.10
 3    WorldModel_7e2a WorldModel +0.756  0.90
```

- **Rank 1 agent** is the elite — its hyperparameters are never mutated and are copied to bottom-quartile agents at each evolution cycle
- **Sharpe** displayed separately from fitness to distinguish between high-return/high-risk and high-Sharpe/lower-return agents
- **Fitness bar** in the browser dashboard normalises fitness to [0%, 100%] visual width using `(fitness + 1) × 50`

---

## Performance Reports

Monthly, quarterly, and annual reports are stored in `reports/`:

```
reports/
├── monthly/
│   └── YYYY-MM_performance.md
├── quarterly/
│   └── YYYY-QN_performance.md
└── annual/
    └── YYYY_annual_review.md
```

Report template fields: period Sharpe, Calmar, best agent, worst agent, regime distribution, circuit breaker events, net P&L, evolution cycles completed.

---

## Benchmarking

Compare MarketZero performance against:

| Benchmark | Description |
|---|---|
| SPY buy-and-hold | Baseline passive return |
| 60/40 portfolio | Blended equity-bond baseline |
| Momentum factor (MOM) | Pure momentum strategy |
| Random agent | Lower bound — should always beat |

Benchmark comparison utilities are planned for v1.1.
