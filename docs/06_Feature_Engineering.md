# 06 — Feature Engineering

---

## Overview

Feature engineering in MarketZero is agent-local. Each agent maintains its own lookback buffer and computes its own technical indicators using its own hyperparameters. This means two agents observing the same price stream can produce different feature vectors based on their evolved `rsi_period`, `bb_period`, `lookback`, and `atr_period` values.

This deliberate heterogeneity is a core design choice — it prevents the population from converging on a single feature view and maintains signal diversity.

---

## Technical Indicators

### Relative Strength Index (RSI)

```
RS = avg_gain / avg_loss  (over rsi_period bars)
RSI = 100 - (100 / (1 + RS))
```

Hyperparameter range: `rsi_period` ∈ [5, 30] (evolved per agent)

Interpretation in signal logic:
- RSI > 70 → potential overbought (sell signal component)
- RSI < 30 → potential oversold (buy signal component)
- RSI crossing 50 → momentum confirmation

### Bollinger Bands

```
Middle = SMA(close, bb_period)
Upper  = Middle + bb_std × σ(close, bb_period)
Lower  = Middle - bb_std × σ(close, bb_period)
```

Hyperparameters: `bb_period` ∈ [10, 50], `bb_std` ∈ [1.5, 3.0]

### Average True Range (ATR)

```
TR  = max(high - low, |high - prev_close|, |low - prev_close|)
ATR = EMA(TR, atr_period)
```

Hyperparameter: `atr_period` ∈ [7, 21]

ATR is used for position sizing (volatility-scaled) and stop-loss placement.

### EMA Crossover (Regime Detector)

```
EMA_fast = EMA(prices, 9)
EMA_slow = EMA(prices, 21)
trend    = (EMA_fast - EMA_slow) / EMA_slow
```

Used in `RegimeDetector` for BULL/BEAR classification. Not directly exposed to agents.

---

## Observation Vector Construction

Each agent builds its observation vector at every step:

```python
observation = np.array([
    # Raw price history (normalised by current price)
    *[p / current_price for p in price_buffer[-lookback:]],
    # Momentum
    rsi / 100.0,
    # Mean reversion bands (normalised)
    (current_price - bb_lower) / (bb_upper - bb_lower + 1e-8),
    # Volatility
    atr / current_price,
    # Regime encoding (one-hot)
    *regime_one_hot,   # length 5
])
```

Vector length varies per agent based on `lookback` (5–60), giving observation sizes of 12–67.

---

## Regime Encoding

The regime state is one-hot encoded and appended to every agent's observation:

| Regime | Encoding |
|---|---|
| UNKNOWN | [1, 0, 0, 0, 0] |
| BULL | [0, 1, 0, 0, 0] |
| BEAR | [0, 0, 1, 0, 0] |
| RANGING | [0, 0, 0, 1, 0] |
| HIGH_VOL | [0, 0, 0, 0, 1] |

---

## Feature Normalisation

All features are normalised before being passed to agent policy networks:

- Price features: divided by current price (makes them scale-invariant)
- RSI: divided by 100 → [0, 1]
- Bollinger position: `(price - lower) / (upper - lower)` → [0, 1] approximately
- ATR: divided by current price (relative volatility)

---

## Adding New Features

To add a feature to all agents:

1. Compute it in `pbt_agents.py` → `_build_observation()` method
2. Update the observation size constant
3. Retrain or reset population (new observation size is incompatible with old weights)

To add an agent-specific evolved feature parameter:

1. Add to `PBTLiveConfig.hyperparameter_bounds`
2. Initialise randomly in `Agent.__init__`
3. Include in the `get_hyperparams()` / `set_hyperparams()` interface (used by evolution)
