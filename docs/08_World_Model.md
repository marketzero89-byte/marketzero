# 08 — World Model

---

## Overview

The WorldModel agent family uses an ensemble of lightweight predictors to estimate the uncertainty of future price movements. When ensemble members disagree strongly, the agent reduces position size or abstains from trading — a form of epistemic humility that prevents overconfident trades in ambiguous market conditions.

---

## Architecture

```
Price History Buffer
        │
        ▼
  Ensemble of N predictors   (default N=5)
  ┌──────┬──────┬──────┬──────┬──────┐
  │ P1   │ P2   │ P3   │ P4   │ P5   │
  └──────┴──────┴──────┴──────┴──────┘
        │
        ▼
  Mean prediction  → direction signal
  Std of predictions → uncertainty / confidence
        │
        ▼
  confidence = 1 - clamp(std / threshold, 0, 1)
```

Each predictor `Pk` is a linear model with its own independently initialised weights. Diversity in the ensemble comes from different random initialisations and different historical windows.

---

## WorldModel Agent: Key Parameters

| Hyperparameter | Range | Role |
|---|---|---|
| `lookback` | 5–60 | Price history window fed to predictors |
| `learning_rate` | 1e-5–1e-2 | Weight update step size |
| `gamma` | 0.90–0.999 | Temporal discount (used in reward shaping) |
| `confidence_threshold` | 0.50–0.90 | Minimum confidence to emit a signal |
| `n_ensemble` | 3–10 | Number of ensemble predictors |

---

## Uncertainty Estimation

At each step the agent computes:

```python
predictions = [p.predict(obs) for p in self.ensemble]
mean_pred   = np.mean(predictions)       # directional signal: positive = buy
std_pred    = np.std(predictions)        # disagreement = uncertainty

# Normalise uncertainty to [0, 1]
normalised_uncertainty = min(std_pred / self.uncertainty_scale, 1.0)
confidence = 1.0 - normalised_uncertainty
```

If `confidence < confidence_threshold`, the agent emits a zero-direction signal (no trade).

---

## Signal Output

```python
Signal(
    direction   = np.tanh(mean_pred),    # ∈ (-1, 1): -1=full short, +1=full long
    confidence  = confidence,            # ∈ (0, 1)
    position_size = confidence * max_size,
    agent_id    = self.agent_id,
    agent_type  = "WorldModel",
)
```

---

## Online Learning

WorldModel predictors update their weights at each step using gradient descent on the prediction error:

```
loss = (prediction - actual_return)²
grad = 2 × (prediction - actual_return) × observation
weights -= learning_rate × grad
```

This is a simple online linear regression — no neural network, no backpropagation. The simplicity is intentional: WorldModel's edge comes from ensemble diversity, not individual predictor sophistication.

---

## Comparison to PPO and Dreamer

| Feature | PPO | Dreamer | WorldModel |
|---|---|---|---|
| Policy type | Stochastic linear | Imagination rollout | Ensemble disagreement |
| Confidence source | Entropy of action dist. | Planning uncertainty | Predictor std deviation |
| Update method | Policy gradient | Model-based RL | Online gradient descent |
| Latency | Low | Medium | Low |
| Best regime | BULL (trend) | Volatile/complex | RANGING (uncertainty) |

---

## Ensemble Diversification Strategies

Currently all predictors are linear models. Future work (see `22_Research_Roadmap.md`) includes:

- Heterogeneous ensemble: mix linear, polynomial, and small MLP predictors
- Bayesian ensemble: weight predictors by their recent prediction accuracy
- Dropout-based uncertainty: single neural net with MC dropout for uncertainty
