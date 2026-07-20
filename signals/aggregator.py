"""
Signal Aggregation
Combines signals from multiple agents into a single trading decision.
Modes: EQUAL, FITNESS, CONFIDENCE, RANK, ENSEMBLE (60/40), ATTENTION (v1.2)
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

N_ACTIONS = 3   # 0=HOLD, 1=BUY, 2=SELL


class AggMode(str, Enum):
    EQUAL      = "EQUAL"
    FITNESS    = "FITNESS"
    CONFIDENCE = "CONFIDENCE"
    RANK       = "RANK"
    ENSEMBLE   = "ENSEMBLE"
    ATTENTION  = "ATTENTION"


class SignalAggregator:
    """
    Aggregates action distributions from multiple agents.

    Parameters
    ----------
    mode : AggMode
    ensemble_split : float   For ENSEMBLE mode: fraction assigned to top-fitness agent
    """

    def __init__(
        self,
        mode: AggMode = AggMode.ENSEMBLE,
        ensemble_split: float = 0.60,
    ):
        self.mode = mode
        self.ensemble_split = ensemble_split

        # Attention parameters (v1.2): learned weights over agent types
        self._attention_weights: Dict[str, float] = {}
        self._attention_lr: float = 1e-3
        self._attention_history: List[Tuple[str, float]] = []  # (agent_id, reward)

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def aggregate(
        self,
        signals: List[Dict],
        regime: Optional[str] = None,
    ) -> Tuple[int, float]:
        """
        Parameters
        ----------
        signals : list of dicts with keys:
            - agent_id : str
            - agent_type : str
            - action : int
            - log_prob : float
            - fitness : float
            - confidence : float (optional, derived from log_prob if absent)

        regime : str  current market regime (used by ATTENTION mode)

        Returns
        -------
        (final_action: int, final_confidence: float)
        """
        if not signals:
            return 0, 0.0   # HOLD

        # Derive confidence from log_prob if not provided
        for s in signals:
            if "confidence" not in s:
                s["confidence"] = float(np.exp(s.get("log_prob", -1.0)))

        weights = self._compute_weights(signals, regime)

        # R-050: direction = Σ w_i × dir_i × conf_i / Σ w_i × conf_i
        # Map action to direction: BUY=+1, SELL=-1, HOLD=0
        _action_to_dir = {0: 0.0, 1: 1.0, 2: -1.0}
        dirs  = np.array([_action_to_dir.get(s["action"], 0.0) for s in signals])
        confs = np.array([s["confidence"] for s in signals])
        num   = np.sum(weights * dirs  * confs)
        den   = np.sum(weights * confs) + 1e-10
        agg_direction = float(num / den)

        # Map aggregated direction back to discrete action + confidence
        if agg_direction > 0.10:
            final_action = 1   # BUY
        elif agg_direction < -0.10:
            final_action = 2   # SELL
        else:
            final_action = 0   # HOLD
        final_conf = float(abs(agg_direction))
        return final_action, final_conf

    # ------------------------------------------------------------------
    # Weight computation per mode
    # ------------------------------------------------------------------

    def _compute_weights(self, signals: List[Dict], regime: Optional[str]) -> np.ndarray:
        n = len(signals)

        if self.mode == AggMode.EQUAL:
            return np.ones(n) / n

        elif self.mode == AggMode.FITNESS:
            fitnesses = np.array([max(s["fitness"], 0) for s in signals])
            total = fitnesses.sum()
            return fitnesses / total if total > 0 else np.ones(n) / n

        elif self.mode == AggMode.CONFIDENCE:
            confs = np.array([s["confidence"] for s in signals])
            total = confs.sum()
            return confs / total if total > 0 else np.ones(n) / n

        elif self.mode == AggMode.RANK:
            fitnesses = np.array([s["fitness"] for s in signals])
            ranks = fitnesses.argsort().argsort() + 1   # 1-based rank
            total = ranks.sum()
            return ranks / total

        elif self.mode == AggMode.ENSEMBLE:
            # 60/40: best agent gets ensemble_split, rest share remainder
            fitnesses = np.array([s["fitness"] for s in signals])
            best_idx  = int(np.argmax(fitnesses))
            weights   = np.full(n, (1 - self.ensemble_split) / max(n - 1, 1))
            weights[best_idx] = self.ensemble_split
            return weights

        elif self.mode == AggMode.ATTENTION:
            return self._attention_weights_for(signals, regime)

        return np.ones(n) / n

    # ------------------------------------------------------------------
    # Attention (v1.2)
    # ------------------------------------------------------------------

    def _attention_weights_for(
        self, signals: List[Dict], regime: Optional[str]
    ) -> np.ndarray:
        """Learned attention over agent_type × regime."""
        n = len(signals)
        raw = np.array([
            self._attention_weights.get(
                f"{s['agent_type']}_{regime or 'ANY'}", 1.0
            )
            for s in signals
        ])
        # Softmax
        e = np.exp(raw - raw.max())
        return e / e.sum()

    def update_attention(self, signals: List[Dict], reward: float, regime: Optional[str]) -> None:
        """
        Update attention weights based on observed reward.
        Call after the environment returns a reward signal.
        """
        weights = self._attention_weights_for(signals, regime)
        for signal, w in zip(signals, weights):
            key = f"{signal['agent_type']}_{regime or 'ANY'}"
            old = self._attention_weights.get(key, 1.0)
            self._attention_weights[key] = old + self._attention_lr * reward * w
        self._attention_history.append((str([s["agent_id"] for s in signals]), reward))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _action_to_probs(action: int, confidence: float) -> np.ndarray:
        """Convert a discrete action + confidence into a soft probability vector."""
        probs = np.ones(N_ACTIONS) * (1 - confidence) / max(N_ACTIONS - 1, 1)
        probs[action] = confidence
        return np.clip(probs, 0, 1)

    def set_mode(self, mode: AggMode) -> None:
        self.mode = mode
        logger.info("Signal aggregation mode set to %s", mode.value)

    def summary(self) -> Dict:
        return {
            "mode":             self.mode.value,
            "ensemble_split":   self.ensemble_split,
            "attention_keys":   len(self._attention_weights),
        }
