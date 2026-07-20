# 09 — Reinforcement Learning

---

## RL Formulation

MarketZero frames trading as a Markov Decision Process (MDP):

| MDP Component | Definition |
|---|---|
| State `s_t` | Observation vector (price history + indicators + regime) |
| Action `a_t` | `(direction, confidence, position_size)` tuple |
| Reward `r_t` | Step P&L, shaped by Sharpe and drawdown penalty |
| Transition | Next price tick from broker |
| Episode | One generation (`generation_steps` steps) |

---

## PPO Agent

### Policy

The PPO agent uses a linear stochastic policy:

```
logits = W × observation + b
action = tanh(logits + ε),   ε ~ N(0, σ)
```

Where `σ` is an entropy scaling parameter that controls exploration. Higher entropy = more random actions = more exploration.

### Objective

```
L = E[min(r_t × A_t, clip(r_t, 1-ε, 1+ε) × A_t)] - β × H(π)

where:
  r_t = π(a|s) / π_old(a|s)    (probability ratio)
  A_t = advantage estimate
  H(π) = policy entropy
  ε = clip_epsilon (hyperparameter, evolved)
  β = entropy coefficient
```

### Update

```python
advantage = reward - baseline
grad = advantage × (action / confidence) × observation
policy_weights += learning_rate × grad
```

PPO is implemented as a simplified linear version without a value network. The advantage estimate uses a running mean baseline.

---

## Dreamer Agent

The Dreamer agent extends PPO by simulating imagined future trajectories before committing to an action.

### Imagination Rollout

```
Current state s_t
       │
       ▼
Transition model T(s_{t+1} | s_t, a_t)   ← learned linear model
       │
       ▼
Simulate H steps ahead (H = imagination_horizon, default 5)
       │
       ▼
Sum discounted rewards: V = Σ γ^h × r_{t+h}
       │
       ▼
Select action with highest imagined value
```

### Transition Model

```python
# Predicts next state from current state + action
s_next_pred = T_weights @ np.concatenate([s_t, [a_t]])
```

The transition model is updated online:
```
loss_T = ||s_next_pred - s_actual||²
T_weights -= lr_T × ∇loss_T
```

### Confidence

Dreamer confidence is derived from the variance of imagined returns across the rollout:

```
confidence = exp(-var(imagined_returns) / temperature)
```

High variance in imagination → low confidence → reduced position size.

---

## Reward Shaping

Raw step reward is P&L normalised by initial capital. Shaped reward adds:

```
r_shaped = r_pnl
         + α × sharpe_contribution
         - β × abs(drawdown)
         - γ × trade_cost_proxy
```

Default coefficients: α=0.3, β=0.5, γ=0.1

---

## Offline Pre-training

Before live deployment, agents can be pre-trained on synthetic price data:

```bash
python main.py train --n-agents 6 --episodes 100 --symbols AAPL SPY
```

Pre-training uses `PPOTrainer` in `trading/training_ppo.py`. Each episode is a synthetic 252-step GBM price sequence. Pre-trained weights are saved and loaded into the population at `PBTLiveExecutor.initialize_population()` if a checkpoint is specified.

---

## Hyperparameters Evolved Per Agent

| Hyperparameter | Range | Effect |
|---|---|---|
| `learning_rate` | 1e-5 → 1e-2 | Update step size |
| `gamma` | 0.90 → 0.999 | Temporal discount factor |
| `clip_epsilon` | 0.05 → 0.40 | PPO clipping range |
| `confidence_threshold` | 0.50 → 0.90 | Minimum confidence to trade |
| `lookback` | 5 → 60 | Observation window length |
| `rsi_period` | 5 → 30 | RSI indicator period |
| `bb_period` | 10 → 50 | Bollinger band period |
| `bb_std` | 1.5 → 3.0 | Bollinger band width |
| `atr_period` | 7 → 21 | ATR indicator period |

All hyperparameters are evolved jointly by the PBT evolution engine. See `17_PBT_Framework.md`.
