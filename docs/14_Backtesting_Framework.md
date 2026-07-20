# 14 — Backtesting Framework

---

## Overview

The backtesting framework in `trading/backtest_metrics.py` provides offline evaluation of any equity curve against six performance metrics. It is deliberately lightweight — no order-level simulation, no slippage, no market impact. Its purpose is fast hypothesis screening, not production-grade simulation.

For production-grade historical testing, use the paper trading system with a replay broker (see `07_Market_Simulation.md`).

---

## Running a Backtest

```bash
python main.py backtest --trend bull
python main.py backtest --trend bear
python main.py backtest --trend ranging
```

Output:
```
Backtest Results (bull):
  sharpe_ratio                  : 1.43
  calmar_ratio                  : 0.92
  sortino_ratio                 : 1.87
  max_drawdown_pct              : -8.34
  annual_return_pct             : +12.4
  win_rate                      : 0.543
```

---

## Metrics Reference

### Sharpe Ratio
```
Sharpe = mean(daily_returns) / std(daily_returns) × √252
```
Target: ≥ 1.5. Values below 0.5 indicate the strategy does not compensate for its volatility.

### Calmar Ratio
```
Calmar = CAGR / abs(MaxDrawdown)
CAGR   = (equity_final / equity_initial)^(252/n_days) - 1
```
Target: ≥ 0.8. Rewards strategies that grow steadily without large drawdowns.

### Sortino Ratio
```
Sortino = mean(daily_returns) / downside_std × √252
downside_std = std of returns below zero only
```
Penalises downside volatility only — preferred over Sharpe for asymmetric strategies.

### Maximum Drawdown
```
DD(t) = (equity(t) - peak(t)) / peak(t)
MaxDD  = min(DD(t)) for all t
```
Hard limit: -20% (triggers emergency halt in live mode).

### Win Rate
```
Win Rate = count(r_t > 0) / count(all_trades)
```
Target: ≥ 52%. Note: win rate alone is not sufficient — profit factor also matters.

### Profit Factor
```
Profit Factor = sum(winning_trades) / abs(sum(losing_trades))
```
Target: > 1.3.

---

## BacktestEvaluator API

```python
from trading.backtest_metrics import BacktestEvaluator

evaluator = BacktestEvaluator()

# Pass any list of equity values
equity_curve = [10000.0, 10050.0, 10030.0, ...]
metrics = evaluator.evaluate(equity_curve)

print(metrics.sharpe_ratio)
print(metrics.to_dict())
```

---

## Fitness Function

The PBT fitness function composites multiple metrics into a single [-1, 1] score:

```
Fitness = 0.50 × tanh(Sharpe / 3)
        + 0.20 × tanh(Calmar / 5)
        + 0.15 × tanh(Sortino / 3)
        + 0.10 × tanh(AnnReturn × 5)
        - 0.05 × |MaxDrawdown|
```

Weight rationale:
- Sharpe (50%): primary signal quality measure
- Calmar (20%): drawdown-adjusted growth
- Sortino (15%): downside risk sensitivity
- Annual Return (10%): absolute growth incentive
- Drawdown penalty (5%): explicit capital protection term

---

## Limitations

| Limitation | Impact | Mitigation |
|---|---|---|
| No slippage | Overstates fill quality | Apply 0.05% slippage haircut to paper results |
| No market impact | Overstates scalability | Limit position sizes to < 5% |
| Synthetic prices only | Regime distribution may not match live | Paper trade on live data before going live |
| Single equity curve | No Monte Carlo sensitivity | Run multiple seeds (planned v1.1) |
| No transaction costs | Inflates net return | Deduct estimated $0.005/share commission |

---

## Walk-Forward Analysis

Walk-forward analysis is not automated in v1.0. To run manually:

1. Split historical data into N folds
2. Train on folds 1..k, test on fold k+1
3. Aggregate out-of-sample metrics across all test folds
4. Compare to in-sample metrics — large degradation indicates overfitting

Walk-forward automation is on the v1.1 roadmap.
