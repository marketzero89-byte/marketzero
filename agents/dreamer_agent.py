"""
Dreamer Agent — Imagination Rollout World Model
Learns a latent transition model and plans via imagined trajectories.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

N_ACTIONS = 3   # HOLD, BUY, SELL


# ---------------------------------------------------------------------------
# Latent world model (linear transition + reward)
# ---------------------------------------------------------------------------

class LatentWorldModel:
    """
    Compact latent model: s_{t+1} ≈ A*s_t + B*a_t + noise
    Reward model:          r_t ≈ w_r · s_t
    """

    def __init__(self, state_dim: int, latent_dim: int = 8, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.A = rng.standard_normal((latent_dim, state_dim)) * 0.01
        self.B = rng.standard_normal((latent_dim, N_ACTIONS)) * 0.01
        self.C = rng.standard_normal((state_dim, latent_dim)) * 0.01   # decoder
        self.w_r = rng.standard_normal(latent_dim) * 0.01
        self.latent_dim = latent_dim
        self.state_dim = state_dim

        # Online learning buffers
        self._X: List[np.ndarray] = []
        self._Y: List[np.ndarray] = []
        self._R: List[float] = []

    def encode(self, state: np.ndarray) -> np.ndarray:
        return np.tanh(self.A @ state)

    def transition(self, latent: np.ndarray, action_one_hot: np.ndarray) -> np.ndarray:
        return np.tanh(self.A @ self.C @ latent + self.B @ action_one_hot)

    def predict_reward(self, latent: np.ndarray) -> float:
        return float(self.w_r @ latent)

    def one_hot(self, action: int) -> np.ndarray:
        oh = np.zeros(N_ACTIONS)
        oh[action] = 1.0
        return oh

    def store(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray) -> None:
        self._X.append(np.concatenate([state, self.one_hot(action)]))
        self._Y.append(next_state)
        self._R.append(reward)

    def fit(self, lr: float = 1e-3, n_steps: int = 5) -> float:
        """Mini-batch gradient update on stored transitions."""
        if len(self._X) < 4:
            return 0.0
        X = np.array(self._X)
        Y = np.array(self._Y)
        R = np.array(self._R)
        losses = []
        for _ in range(n_steps):
            idx = np.random.choice(len(X), min(32, len(X)), replace=False)
            for i in idx:
                s = X[i, :self.state_dim]
                z = self.encode(s)
                z_pred = self.transition(z, X[i, self.state_dim:])
                s_pred = self.C @ z_pred
                pred_r = self.predict_reward(z)
                loss = float(np.mean((s_pred - Y[i]) ** 2)) + (pred_r - R[i]) ** 2
                losses.append(loss)
                # Gradient descent (simplified finite-diff not needed; use direct)
                err = s_pred - Y[i]
                self.C -= lr * np.outer(err, z_pred)
                self.w_r -= lr * (pred_r - R[i]) * z
        return float(np.mean(losses))


# ---------------------------------------------------------------------------
# Dreamer Agent
# ---------------------------------------------------------------------------

class DreamerAgent:
    """
    Dreamer-style agent.

    Parameters
    ----------
    state_dim : int
    hyperparams : dict  Keys: learning_rate, gamma, imagination_depth, lookback
    """

    def __init__(
        self,
        state_dim: int = 16,
        hyperparams: Optional[Dict] = None,
        seed: int = 0,
    ):
        hp = hyperparams or {}
        self.lr                = float(hp.get("learning_rate", 3e-4))
        self.gamma             = float(hp.get("gamma", 0.99))
        self.imagination_depth = int(hp.get("imagination_depth", 5))   # R-037: spec default H=5
        self.lookback          = int(hp.get("lookback", 20))
        self.state_dim         = state_dim

        self.world_model = LatentWorldModel(state_dim, latent_dim=8, seed=seed)

        rng = np.random.default_rng(seed + 100)
        self._actor_W = rng.standard_normal((N_ACTIONS, 8)) * 0.01
        self._actor_b = np.zeros(N_ACTIONS)

        self._transitions: List[Tuple] = []

    # ------------------------------------------------------------------
    # Imagination rollout
    # ------------------------------------------------------------------

    def _imagine(self, start_state: np.ndarray, depth: int) -> float:
        """Simulate imagined trajectory, return discounted cumulative reward."""
        z = self.world_model.encode(start_state)
        total = 0.0
        discount = 1.0
        for _ in range(depth):
            logits = self._actor_W @ z + self._actor_b
            probs  = self._softmax(logits)
            action = int(np.argmax(probs))
            total += discount * self.world_model.predict_reward(z)
            z = self.world_model.transition(z, self.world_model.one_hot(action))
            discount *= self.gamma
        return total

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max())
        return e / e.sum()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def act(self, state: np.ndarray) -> Tuple[int, float]:
        """Choose action via actor policy in latent space."""
        z = self.world_model.encode(state)
        logits = self._actor_W @ z + self._actor_b
        probs  = self._softmax(logits)
        action = int(np.random.choice(N_ACTIONS, p=probs))
        log_prob = float(np.log(probs[action] + 1e-10))
        return action, log_prob

    def learn(self, observation: np.ndarray, actual_return: float) -> float:
        """
        R-038: Per-step online update — MSE gradient descent on transition model.
        Updates the reward model weights toward the observed actual_return.
        """
        z = self.world_model.encode(observation)
        pred_r = self.world_model.predict_reward(z)
        err    = pred_r - actual_return
        # Gradient of MSE w.r.t. w_r: 2 * err * z
        self.world_model.w_r -= self.lr * err * z
        return float(err ** 2)

    def store(self, state, action, reward, next_state) -> None:
        self.world_model.store(state, action, reward, next_state)
        self._transitions.append((state, action, reward, next_state))

    def update(self) -> float:
        """Fit world model + actor on imagined trajectories."""
        wm_loss = self.world_model.fit(lr=self.lr)

        # Actor update: maximise imagined returns
        if len(self._transitions) >= 4:
            states = [t[0] for t in self._transitions[-64:]]
            grad_W = np.zeros_like(self._actor_W)
            grad_b = np.zeros_like(self._actor_b)
            for s in states:
                z = self.world_model.encode(s)
                imagined_return = self._imagine(s, self.imagination_depth)
                logits = self._actor_W @ z + self._actor_b
                probs  = self._softmax(logits)
                a      = int(np.argmax(probs))
                d_log  = -probs.copy()
                d_log[a] += 1.0
                grad_W += imagined_return * np.outer(d_log, z)
                grad_b += imagined_return * d_log
            n = len(states)
            self._actor_W += self.lr * grad_W / n
            self._actor_b += self.lr * grad_b / n

        return wm_loss

    def clear_buffer(self) -> None:
        self._transitions.clear()
