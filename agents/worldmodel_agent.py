"""
WorldModel Agent — Ensemble Disagreement
Uses disagreement among an ensemble of predictors as an exploration bonus.
v1.0: Linear ensemble
v1.2: Heterogeneous ensemble (linear + polynomial + MLP) with MC-Dropout uncertainty
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

N_ACTIONS = 3   # HOLD, BUY, SELL


# ---------------------------------------------------------------------------
# Predictors
# ---------------------------------------------------------------------------

class LinearPredictor:
    """Linear next-state predictor."""

    def __init__(self, state_dim: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.W = rng.standard_normal((state_dim, state_dim)) * 0.01
        self.b = np.zeros(state_dim)
        self.state_dim = state_dim

    def predict(self, state: np.ndarray, action_oh: np.ndarray) -> np.ndarray:
        inp = np.concatenate([state, action_oh])
        W = self.W[:, :len(inp)] if self.W.shape[1] > len(inp) else np.pad(
            self.W, ((0, 0), (0, len(inp) - self.W.shape[1]))
        )
        return W @ inp + self.b

    def update(self, state, action_oh, target, lr):
        inp = np.concatenate([state, action_oh])
        pred = self.predict(state, action_oh)
        err = pred - target
        inp_padded = np.pad(inp, (0, max(0, self.W.shape[1] - len(inp))))[:self.W.shape[1]]
        self.W -= lr * np.outer(err, inp_padded)
        self.b -= lr * err
        return float(np.mean(err ** 2))


class PolynomialPredictor:
    """Degree-2 polynomial feature predictor."""

    def __init__(self, state_dim: int, seed: int = 0):
        # Only use quadratic diagonal terms to keep it tractable
        rng = np.random.default_rng(seed)
        feat_dim = state_dim + N_ACTIONS + state_dim   # linear + action + squared
        self.W = rng.standard_normal((state_dim, feat_dim)) * 0.01
        self.b = np.zeros(state_dim)
        self.state_dim = state_dim

    def _features(self, state, action_oh):
        return np.concatenate([state, action_oh, state ** 2])

    def predict(self, state, action_oh):
        f = self._features(state, action_oh)
        W = self.W[:, :len(f)]
        return W @ f + self.b

    def update(self, state, action_oh, target, lr):
        f = self._features(state, action_oh)
        pred = self.W[:, :len(f)] @ f + self.b
        err = pred - target
        self.W[:, :len(f)] -= lr * np.outer(err, f)
        self.b -= lr * err
        return float(np.mean(err ** 2))


class TinyMLPPredictor:
    """Small MLP predictor with dropout for MC uncertainty."""

    def __init__(self, state_dim: int, hidden: int = 16, seed: int = 0):
        rng = np.random.default_rng(seed)
        inp_dim = state_dim + N_ACTIONS
        self.W1 = rng.standard_normal((hidden, inp_dim)) * np.sqrt(2 / inp_dim)
        self.b1 = np.zeros(hidden)
        self.W2 = rng.standard_normal((state_dim, hidden)) * np.sqrt(2 / hidden)
        self.b2 = np.zeros(state_dim)
        self.state_dim = state_dim
        self.hidden = hidden
        self.dropout_rate = 0.1

    def _relu(self, x):
        return np.maximum(0, x)

    def predict(self, state, action_oh, training=False):
        inp = np.concatenate([state, action_oh])
        h = self._relu(self.W1 @ inp + self.b1)
        if training:
            mask = np.random.binomial(1, 1 - self.dropout_rate, size=h.shape)
            h = h * mask / (1 - self.dropout_rate + 1e-10)
        return self.W2 @ h + self.b2

    def update(self, state, action_oh, target, lr):
        pred = self.predict(state, action_oh, training=True)
        err = pred - target
        inp = np.concatenate([state, action_oh])
        h = self._relu(self.W1 @ inp + self.b1)
        self.W2 -= lr * np.outer(err, h)
        self.b2 -= lr * err
        dh = (self.W2.T @ err) * (h > 0)
        self.W1 -= lr * np.outer(dh, inp)
        self.b1 -= lr * dh
        return float(np.mean(err ** 2))

    def mc_predict(self, state, action_oh, n_samples=10):
        preds = np.array([self.predict(state, action_oh, training=True) for _ in range(n_samples)])
        return preds.mean(axis=0), preds.std(axis=0)


# ---------------------------------------------------------------------------
# WorldModel Agent
# ---------------------------------------------------------------------------

class WorldModelAgent:
    """
    Ensemble-based world model agent.
    Uses disagreement across predictors as an exploration signal.

    Parameters
    ----------
    state_dim : int
    hyperparams : dict  Keys: learning_rate, gamma, ensemble_size, lookback,
                              uncertainty_scale, confidence_threshold
    heterogeneous : bool  Mix predictor types (v1.2)
    """

    def __init__(
        self,
        state_dim: int = 16,
        hyperparams: Optional[Dict] = None,
        heterogeneous: bool = False,
        seed: int = 0,
    ):
        hp = hyperparams or {}
        self.lr                   = float(hp.get("learning_rate", 3e-4))
        self.gamma                = float(hp.get("gamma", 0.99))
        # R-044: ensemble_size from hyperparams [3, 10], not class constant
        self.ensemble_size        = int(np.clip(hp.get("ensemble_size", 5), 3, 10))
        self.lookback             = int(hp.get("lookback", 20))
        self.uncertainty_scale    = float(hp.get("uncertainty_scale", 5.0))  # R-041
        self.confidence_threshold = float(hp.get("confidence_threshold", 0.55))  # R-041/R-045
        self.state_dim    = state_dim
        self.heterogeneous= heterogeneous

        # Build ensemble
        self.ensemble: List = []
        for i in range(self.ensemble_size):
            if heterogeneous:
                if i % 3 == 0:
                    self.ensemble.append(LinearPredictor(state_dim, seed=seed + i))
                elif i % 3 == 1:
                    self.ensemble.append(PolynomialPredictor(state_dim, seed=seed + i))
                else:
                    self.ensemble.append(TinyMLPPredictor(state_dim, seed=seed + i))
            else:
                self.ensemble.append(LinearPredictor(state_dim, seed=seed + i))

        # Action policy: softmax over disagreement-weighted Q values
        rng = np.random.default_rng(seed + 999)
        self._policy_W = rng.standard_normal((N_ACTIONS, state_dim)) * 0.01
        self._policy_b = np.zeros(N_ACTIONS)

        self._buffer: List[Tuple] = []

    # ------------------------------------------------------------------
    # Disagreement = ensemble variance over next-state predictions
    # ------------------------------------------------------------------

    def _one_hot(self, action: int) -> np.ndarray:
        oh = np.zeros(N_ACTIONS)
        oh[action] = 1.0
        return oh

    def disagreement(self, state: np.ndarray, action: int) -> float:
        oh = self._one_hot(action)
        preds = np.array([m.predict(state, oh) for m in self.ensemble])
        return float(np.mean(np.var(preds, axis=0)))

    def bayesian_uncertainty(self, state: np.ndarray, action: int) -> float:
        """MC-Dropout uncertainty from MLP predictors."""
        oh = self._one_hot(action)
        mlp_preds = []
        for m in self.ensemble:
            if isinstance(m, TinyMLPPredictor):
                _, std = m.mc_predict(state, oh)
                mlp_preds.append(float(np.mean(std)))
        return float(np.mean(mlp_preds)) if mlp_preds else self.disagreement(state, action)

    # ------------------------------------------------------------------
    # Policy
    # ------------------------------------------------------------------

    def _softmax(self, x):
        e = np.exp(x - x.max())
        return e / e.sum()

    def act(self, state: np.ndarray) -> Tuple[int, float]:
        logits = self._policy_W @ state + self._policy_b

        # Disagreement bonus: encourage exploring uncertain actions
        bonuses = np.array([self.disagreement(state, a) for a in range(N_ACTIONS)])
        logits += 0.1 * bonuses

        probs    = self._softmax(logits)
        action   = int(np.random.choice(N_ACTIONS, p=probs))

        # R-041: confidence = 1 - min(std/uncertainty_scale, 1.0)
        oh = self._one_hot(action)
        preds = np.array([m.predict(state, oh) for m in self.ensemble])
        std = float(np.mean(np.std(preds, axis=0)))
        confidence = 1.0 - min(std / (self.uncertainty_scale + 1e-10), 1.0)

        # R-041: zero signal when confidence below threshold
        if confidence < self.confidence_threshold:
            return 0, confidence   # HOLD with low confidence

        log_prob = float(np.log(probs[action] + 1e-10))
        return action, log_prob

    def learn(self, observation: np.ndarray, actual_return: float) -> float:
        """
        R-042: Per-step MSE gradient descent on all ensemble models toward actual_return.
        Uses a synthetic next-state target = observation * (1 + actual_return).
        """
        oh_hold = self._one_hot(0)  # use HOLD action as reference for online update
        target = observation * (1.0 + actual_return)
        losses = []
        for model in self.ensemble:
            l = model.update(observation, oh_hold, target, self.lr)
            losses.append(l)
        return float(np.mean(losses)) if losses else 0.0

    def store(self, state, action, reward, next_state) -> None:
        self._buffer.append((state, action, reward, next_state))

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self) -> float:
        if not self._buffer:
            return 0.0

        losses = []
        for state, action, reward, next_state in self._buffer[-128:]:
            oh = self._one_hot(action)
            for model in self.ensemble:
                l = model.update(state, oh, next_state, self.lr)
                losses.append(l)

        # Policy gradient: reward + disagreement bonus
        grad_W = np.zeros_like(self._policy_W)
        grad_b = np.zeros_like(self._policy_b)
        for state, action, reward, _ in self._buffer[-64:]:
            bonus = self.disagreement(state, action)
            adv   = reward + 0.1 * bonus
            logits= self._policy_W @ state + self._policy_b
            probs = self._softmax(logits)

            # R-043: direction = np.tanh(mean_pred) for signal
            oh   = self._one_hot(action)
            preds = np.array([m.predict(state, oh) for m in self.ensemble])
            mean_pred = np.mean(preds)
            direction = float(np.tanh(mean_pred))   # R-043

            d_log = -probs.copy()
            d_log[action] += 1.0
            grad_W += adv * np.outer(d_log, state)
            grad_b += adv * d_log

        n = max(len(self._buffer[-64:]), 1)
        self._policy_W += self.lr * grad_W / n
        self._policy_b += self.lr * grad_b / n

        self._buffer.clear()
        return float(np.mean(losses)) if losses else 0.0
