"""
PPO Agent — Linear + MLP Policy Gradient
v1.0: Linear policy
v1.2: Small MLP (2 hidden layers, 32 units)
"""

from __future__ import annotations

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State / Action definitions
# ---------------------------------------------------------------------------

N_ACTIONS = 3   # 0=HOLD, 1=BUY, 2=SELL


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    log_prob: float = 0.0
    value: float = 0.0


# ---------------------------------------------------------------------------
# Linear policy (v1.0)
# ---------------------------------------------------------------------------

class LinearPolicy:
    """Simple linear softmax policy over a flat state vector."""

    def __init__(self, state_dim: int, n_actions: int = N_ACTIONS, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.W = rng.standard_normal((n_actions, state_dim)) * 0.01
        self.b = np.zeros(n_actions)
        self.state_dim = state_dim
        self.n_actions = n_actions

    def logits(self, state: np.ndarray) -> np.ndarray:
        return self.W @ state + self.b

    def softmax(self, x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max())
        return e / e.sum()

    def action_and_log_prob(self, state: np.ndarray) -> Tuple[int, float]:
        probs = self.softmax(self.logits(state))
        action = int(np.random.choice(self.n_actions, p=probs))
        log_prob = float(np.log(probs[action] + 1e-10))
        return action, log_prob

    def update(self, grad_W: np.ndarray, grad_b: np.ndarray, lr: float) -> None:
        self.W += lr * grad_W
        self.b += lr * grad_b


# ---------------------------------------------------------------------------
# MLP policy (v1.2)
# ---------------------------------------------------------------------------

class MLPPolicy:
    """Two-hidden-layer MLP with 32 units each, pure-numpy for portability."""

    def __init__(
        self,
        state_dim: int,
        n_actions: int = N_ACTIONS,
        hidden: int = 32,
        seed: int = 0,
    ):
        rng = np.random.default_rng(seed)
        scale = lambda fan_in: np.sqrt(2.0 / fan_in)
        self.W1 = rng.standard_normal((hidden, state_dim)) * scale(state_dim)
        self.b1 = np.zeros(hidden)
        self.W2 = rng.standard_normal((hidden, hidden)) * scale(hidden)
        self.b2 = np.zeros(hidden)
        self.W3 = rng.standard_normal((n_actions, hidden)) * scale(hidden)
        self.b3 = np.zeros(n_actions)
        self.n_actions = n_actions
        self.state_dim = state_dim

    def _relu(self, x):
        return np.maximum(0, x)

    def forward(self, state: np.ndarray) -> np.ndarray:
        h1 = self._relu(self.W1 @ state + self.b1)
        h2 = self._relu(self.W2 @ h1 + self.b2)
        logits = self.W3 @ h2 + self.b3
        return logits

    def softmax(self, x):
        e = np.exp(x - x.max())
        return e / e.sum()

    def action_and_log_prob(self, state: np.ndarray) -> Tuple[int, float]:
        probs = self.softmax(self.forward(state))
        action = int(np.random.choice(self.n_actions, p=probs))
        log_prob = float(np.log(probs[action] + 1e-10))
        return action, log_prob


# ---------------------------------------------------------------------------
# PPO Agent
# ---------------------------------------------------------------------------

class PPOAgent:
    """
    Proximal Policy Optimisation agent.

    Parameters
    ----------
    state_dim : int
    hyperparams : dict  Keys: learning_rate, gamma, clip_eps, entropy_coef, lookback
    use_mlp : bool      Use MLP policy (v1.2) instead of linear (v1.0)
    """

    def __init__(
        self,
        state_dim: int = 16,
        hyperparams: Optional[Dict] = None,
        use_mlp: bool = False,
        seed: int = 0,
    ):
        hp = hyperparams or {}
        self.lr            = float(hp.get("learning_rate", 3e-4))
        self.gamma         = float(hp.get("gamma", 0.99))
        self.clip_eps      = float(hp.get("clip_eps", 0.2))
        self.entropy_coef  = float(hp.get("entropy_coef", 0.01))
        self.lookback      = int(hp.get("lookback", 20))
        self.state_dim     = state_dim
        self.use_mlp       = use_mlp

        if use_mlp:
            self.policy = MLPPolicy(state_dim, seed=seed)
        else:
            self.policy = LinearPolicy(state_dim, seed=seed)

        # Value network (linear)
        rng = np.random.default_rng(seed + 1)
        self.value_W = rng.standard_normal(state_dim) * 0.01
        self.value_b = 0.0

        self._buffer: List[Transition] = []

    # ------------------------------------------------------------------
    # Policy interface
    # ------------------------------------------------------------------

    def act(self, state: np.ndarray, noise_std: float = 0.01) -> Tuple[int, float]:
        """
        R-034: Sample action with small Gaussian noise ε on logits for exploration.
        noise_std=0.01 by default; set to 0 for deterministic greedy selection.
        """
        # Inject Gaussian noise into logits before softmax
        if isinstance(self.policy, LinearPolicy):
            logits = self.policy.logits(state) + np.random.randn(N_ACTIONS) * noise_std
            probs = self.policy.softmax(logits)
            action = int(np.random.choice(N_ACTIONS, p=probs))
            log_prob = float(np.log(probs[action] + 1e-10))
            return action, log_prob
        elif isinstance(self.policy, MLPPolicy):
            logits = self.policy.forward(state) + np.random.randn(N_ACTIONS) * noise_std
            probs = self.policy.softmax(logits)
            action = int(np.random.choice(N_ACTIONS, p=probs))
            log_prob = float(np.log(probs[action] + 1e-10))
            return action, log_prob
        return self.policy.action_and_log_prob(state)

    def value(self, state: np.ndarray) -> float:
        return float(self.value_W @ state + self.value_b)

    def store(self, transition: Transition) -> None:
        self._buffer.append(transition)

    def clear_buffer(self) -> None:
        self._buffer.clear()

    # ------------------------------------------------------------------
    # PPO update (simplified REINFORCE + clipped ratio for linear policy)
    # ------------------------------------------------------------------

    def update(self) -> float:
        """
        Run one PPO update over the stored buffer.
        Returns mean policy loss.
        """
        if not self._buffer:
            return 0.0

        states  = np.array([t.state      for t in self._buffer])
        actions = np.array([t.action     for t in self._buffer], dtype=int)
        rewards = np.array([t.reward     for t in self._buffer])
        old_lps = np.array([t.log_prob   for t in self._buffer])

        # Compute returns (Monte Carlo)
        G = 0.0
        returns = np.zeros(len(rewards))
        for i in reversed(range(len(rewards))):
            G = rewards[i] + self.gamma * G
            returns[i] = G

        # Normalise
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Advantages: returns - baseline
        baselines = np.array([self.value(s) for s in states])
        advantages = returns - baselines

        # Gradient update (linear policy only for simplicity)
        if isinstance(self.policy, LinearPolicy):
            grad_W = np.zeros_like(self.policy.W)
            grad_b = np.zeros_like(self.policy.b)
            for i, (s, a, adv, old_lp) in enumerate(
                zip(states, actions, advantages, old_lps)
            ):
                probs   = self.policy.softmax(self.policy.logits(s))
                log_p   = float(np.log(probs[a] + 1e-10))
                ratio   = np.exp(log_p - old_lp)
                clipped = np.clip(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
                obj     = min(ratio * adv, clipped * adv)

                # Policy gradient
                d_logits          = -probs.copy()
                d_logits[a]      += 1.0
                grad_W += obj * np.outer(d_logits, s)
                grad_b += obj * d_logits

                # Entropy bonus
                entropy = -np.sum(probs * np.log(probs + 1e-10))
                grad_W += self.entropy_coef * entropy * np.outer(d_logits, s)

            self.policy.update(grad_W / len(states), grad_b / len(states), self.lr)

        # Value function update
        for s, ret in zip(states, returns):
            v = self.value(s)
            err = ret - v
            self.value_W += self.lr * err * s
            self.value_b += self.lr * err

        loss = float(np.mean(np.abs(advantages)))
        self.clear_buffer()
        return loss

    # ------------------------------------------------------------------
    # Offline pre-training
    # ------------------------------------------------------------------

    def pretrain(self, n_episodes: int = 50, episode_len: int = 100) -> List[float]:
        """
        Offline PPO pre-training using synthetic episodes.
        Returns list of mean losses per episode.
        """
        losses = []
        for ep in range(n_episodes):
            state = np.random.randn(self.state_dim).astype(np.float32)
            ep_reward = 0.0
            for _ in range(episode_len):
                action, lp = self.act(state)
                next_state = state + np.random.randn(self.state_dim) * 0.1
                reward = float(np.random.randn() * 0.1)
                ep_reward += reward
                self.store(Transition(
                    state=state, action=action, reward=reward,
                    next_state=next_state, done=False, log_prob=lp,
                    value=self.value(state),
                ))
                state = next_state
            loss = self.update()
            losses.append(loss)
        logger.info("PPO pre-training done: %d episodes, mean_loss=%.4f", n_episodes, np.mean(losses))
        return losses
