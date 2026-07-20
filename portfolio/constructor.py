"""
Portfolio Construction
Mean-variance optimisation, Kelly criterion sizing, dynamic capital allocation.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mean-Variance Optimisation
# ---------------------------------------------------------------------------

class MeanVarianceOptimiser:
    """
    Markowitz mean-variance optimiser for cross-symbol position sizing.
    Solves: maximise μ·w - λ/2 · w·Σ·w  subject to sum(w)=1, w≥0

    Parameters
    ----------
    risk_aversion : float  λ (higher = more conservative)
    """

    def __init__(self, risk_aversion: float = 3.0):
        self.risk_aversion = risk_aversion

    def optimise(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        max_weight: float = 0.40,
        n_iter: int = 500,
        lr: float = 0.05,
    ) -> np.ndarray:
        """
        Gradient ascent on the mean-variance objective.
        Returns portfolio weights summing to 1.
        """
        n = len(expected_returns)
        w = np.ones(n) / n

        for _ in range(n_iter):
            grad = expected_returns - self.risk_aversion * cov_matrix @ w
            w += lr * grad
            # Project onto simplex with upper bound
            w = self._project_simplex(w, max_weight)

        return w

    @staticmethod
    def _project_simplex(w: np.ndarray, max_w: float) -> np.ndarray:
        """Project onto probability simplex [0, max_w]^n ∩ Δ."""
        w = np.clip(w, 0, max_w)
        total = w.sum()
        return w / (total + 1e-10)

    def compute_portfolio_metrics(
        self, weights: np.ndarray, returns: np.ndarray, cov: np.ndarray
    ) -> Dict:
        port_return = float(weights @ returns)
        port_var    = float(weights @ cov @ weights)
        sharpe = port_return / (np.sqrt(port_var) + 1e-10)
        return {
            "expected_return": round(port_return, 4),
            "volatility":      round(float(np.sqrt(port_var)), 4),
            "sharpe":          round(sharpe, 4),
        }


# ---------------------------------------------------------------------------
# Kelly Criterion
# ---------------------------------------------------------------------------

def kelly_fraction(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    fractional: float = 0.5,
) -> float:
    """
    Compute (fractional) Kelly fraction for a binary bet.

    Parameters
    ----------
    win_rate   : probability of winning
    avg_win    : average gain per win (positive)
    avg_loss   : average loss per loss (positive)
    fractional : scale down full Kelly (0.5 = half-Kelly)

    Returns
    -------
    fraction of capital to risk (clipped [0, 0.25])
    """
    if avg_loss <= 0 or win_rate <= 0:
        return 0.0
    b = avg_win / avg_loss   # odds ratio
    q = 1 - win_rate
    kelly = (b * win_rate - q) / b
    return float(np.clip(kelly * fractional, 0.0, 0.25))


def kelly_position_size(
    equity: float,
    price: float,
    win_rate: float,
    avg_win_pct: float,
    avg_loss_pct: float,
    fractional: float = 0.5,
) -> int:
    """Returns the integer number of shares to buy based on fractional Kelly."""
    f = kelly_fraction(win_rate, avg_win_pct, avg_loss_pct, fractional)
    dollar_amount = equity * f
    return max(0, int(dollar_amount / price))


# ---------------------------------------------------------------------------
# Dynamic Capital Allocation across agents
# ---------------------------------------------------------------------------

class DynamicCapitalAllocator:
    """
    Allocates trading capital across agents proportional to their fitness.

    Parameters
    ----------
    total_capital : float
    min_allocation_pct : float  Minimum share per agent (prevents starvation)
    """

    def __init__(
        self,
        total_capital: float = 100_000.0,
        min_allocation_pct: float = 0.05,
    ):
        self.total_capital      = total_capital
        self.min_allocation_pct = min_allocation_pct
        self._allocations: Dict[str, float] = {}

    def allocate(
        self,
        agents: List[Dict],   # list of {'agent_id': str, 'fitness': float}
    ) -> Dict[str, float]:
        """
        Compute capital allocation per agent.

        Parameters
        ----------
        agents : list of dicts with agent_id and fitness

        Returns
        -------
        dict mapping agent_id → dollar allocation
        """
        if not agents:
            return {}

        n = len(agents)
        min_alloc = self.total_capital * self.min_allocation_pct

        fitnesses = np.array([max(a["fitness"], 0) for a in agents], dtype=float)
        total_fit = fitnesses.sum()

        if total_fit <= 0:
            # Uniform allocation
            allocations = {a["agent_id"]: self.total_capital / n for a in agents}
        else:
            # Proportional minus minimum
            remaining = self.total_capital - min_alloc * n
            remaining = max(remaining, 0)
            props = fitnesses / total_fit
            allocations = {
                a["agent_id"]: min_alloc + remaining * props[i]
                for i, a in enumerate(agents)
            }

        self._allocations = allocations
        total = sum(allocations.values())
        logger.debug("Capital allocated: total=%.0f across %d agents", total, n)
        return allocations

    def allocation(self, agent_id: str) -> float:
        return self._allocations.get(agent_id, 0.0)

    def summary(self) -> List[Dict]:
        return [
            {"agent_id": aid, "capital": round(cap, 2),
             "pct": round(cap / self.total_capital * 100, 1)}
            for aid, cap in sorted(self._allocations.items(),
                                   key=lambda x: -x[1])
        ]


# ---------------------------------------------------------------------------
# Covariance estimation
# ---------------------------------------------------------------------------

def estimate_covariance(
    returns_matrix: np.ndarray,   # shape (n_periods, n_assets)
    method: str = "sample",
) -> np.ndarray:
    """
    Estimate the covariance matrix.

    Parameters
    ----------
    returns_matrix : shape (T, N)
    method : 'sample' | 'shrinkage' (Ledoit-Wolf)
    """
    if method == "shrinkage":
        try:
            from sklearn.covariance import LedoitWolf
            lw = LedoitWolf()
            lw.fit(returns_matrix)
            return lw.covariance_
        except ImportError:
            logger.warning("scikit-learn not available; using sample covariance")

    return np.cov(returns_matrix.T)
