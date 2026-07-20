# 10 — Strategy Library

---

## Overview

MarketZero agents do not implement named strategies explicitly. Instead, strategies emerge from the combination of agent type, evolved hyperparameters, and the current market regime. However, certain archetypal strategy patterns emerge reliably in the evolved population.

This document catalogues the observed emergent strategies and their conditions.

---

## Emergent Strategy Archetypes

### 1. Momentum Rider (PPO)
- **Trigger**: EMA_fast > EMA_slow by > 0.5%, RSI > 50
- **Action**: Long with high confidence
- **Best regime**: BULL
- **Typical hyperparams**: `lookback=20`, `rsi_period=14`, `gamma=0.95`
- **Observed Sharpe**: 1.2–1.8

### 2. Mean Reversion Sniper (WorldModel)
- **Trigger**: Price below Bollinger lower band, RSI < 30, ensemble confidence high
- **Action**: Long, small size, short hold
- **Best regime**: RANGING
- **Typical hyperparams**: `bb_std=2.5`, `bb_period=20`, `confidence_threshold=0.7`
- **Observed Sharpe**: 0.8–1.4

### 3. Volatility Harvester (Dreamer)
- **Trigger**: ATR expanding, imagination variance high → abstain; ATR contracting → enter
- **Action**: Long or short depending on imagined trend direction
- **Best regime**: HIGH_VOL (counter-intuitively: enters after vol peaks)
- **Typical hyperparams**: `atr_period=14`, `imagination_horizon=7`
- **Observed Sharpe**: 1.0–1.6

### 4. Trend Follower (PPO / Dreamer)
- **Trigger**: Strong EMA crossover, price above 20-bar SMA
- **Action**: Long, full confidence
- **Best regime**: BULL
- **Risk**: Sharp reversal risk; mitigated by ATR-based stop

### 5. Conservative Cash Holder (WorldModel)
- **Trigger**: Ensemble std high (uncertain), regime = HIGH_VOL or UNKNOWN
- **Action**: Zero direction, zero size → no trade
- **Best regime**: All (as a defensive stance)
- **Note**: This is not a failure state — preserving cash in uncertain regimes is positive expected value

---

## Signal Aggregation Effect on Strategies

The ENSEMBLE aggregator blends signals from all active agents. The resulting aggregate strategy is a weighted mixture of the above archetypes:

```
aggregate_direction = Σ w_i × direction_i × confidence_i / Σ w_i × confidence_i

where w_i = 0.6 × fitness_weight_i + 0.4 × confidence_weight_i
```

In BULL regimes, PPO momentum agents score higher fitness and receive larger weights, shifting the aggregate toward momentum. In RANGING regimes, WorldModel agents tend to score higher (fewer false signals), shifting aggregate toward mean reversion.

---

## Symbol-Specific Behaviour

| Symbol | Dominant emergent strategy | Notes |
|---|---|---|
| AAPL | Momentum / trend following | High intraday momentum |
| SPY | Conservative mean reversion | Lower vol; tight bands |
| GLD | Regime-switching | Gold acts as hedge |
| BTC/USD | Volatility harvesting | Highest σ; Dreamer agents dominate |

---

## Adding a Named Strategy

To hard-code a specific strategy (bypassing evolution):

1. Create a new agent class in `pbt/pbt_agents.py` inheriting `BaseAgent`
2. Override `compute_signal()` with your logic
3. Add the agent type to `agent_type_distribution` in config
4. The agent will still be subject to PBT selection pressure

Example:

```python
class RSIRevertAgent(BaseAgent):
    def compute_signal(self, observation, price):
        rsi = self._compute_rsi(price)
        if rsi < 28:
            return Signal(direction=0.8, confidence=0.75, position_size=0.05)
        elif rsi > 72:
            return Signal(direction=-0.8, confidence=0.75, position_size=0.05)
        return Signal(direction=0.0, confidence=0.0, position_size=0.0)
```
