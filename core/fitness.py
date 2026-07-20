"""
Composite Fitness Function
Combines Sharpe, Calmar, Sortino, annual return, drawdown penalty → [-1, 1].
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class FitnessComponents:
    sharpe: float = 0.0
    calmar: float = 0.0
    sortino: float = 0.0
    annual_return: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    composite: float = 0.0


WEIGHTS = {
    "sharpe":        0.30,
    "calmar":        0.20,
    "sortino":       0.20,
    "annual_return": 0.15,
    "win_rate":      0.10,
    "profit_factor": 0.05,
}

# Clamp bounds used during normalisation
_BOUNDS = {
    "sharpe":        (-3.0, 5.0),
    "calmar":        (-2.0, 4.0),
    "sortino":       (-3.0, 6.0),
    "annual_return": (-1.0, 3.0),
    "win_rate":      (0.0, 1.0),
    "profit_factor": (0.0, 5.0),
}


def _normalise(value: float, lo: float, hi: float) -> float:
    """Squash value to [-1, 1] linearly within [lo, hi]."""
    mid = (hi + lo) / 2.0
    half = (hi - lo) / 2.0
    return float(np.clip((value - mid) / half, -1.0, 1.0))


def _safe_div(num: float, den: float, default: float = 0.0) -> float:
    return num / den if abs(den) > 1e-12 else default


def compute_returns(equity_curve: List[float]) -> np.ndarray:
    arr = np.array(equity_curve, dtype=float)
    if len(arr) < 2:
        return np.zeros(1)
    return np.diff(arr) / np.where(arr[:-1] != 0, arr[:-1], 1e-10)


def sharpe_ratio(returns: np.ndarray, risk_free: float = 0.0, periods: int = 252) -> float:
    excess = returns - risk_free / periods
    std = np.std(excess)
    return float(_safe_div(np.mean(excess), std) * np.sqrt(periods))


def sortino_ratio(returns: np.ndarray, risk_free: float = 0.0, periods: int = 252) -> float:
    excess = returns - risk_free / periods
    downside = returns[returns < 0]
    downside_std = np.std(downside) if len(downside) > 0 else 1e-10
    return float(_safe_div(np.mean(excess), downside_std) * np.sqrt(periods))


def max_drawdown(equity_curve: List[float]) -> float:
    arr = np.array(equity_curve, dtype=float)
    peak = np.maximum.accumulate(arr)
    dd = (arr - peak) / np.where(peak != 0, peak, 1e-10)
    return float(np.min(dd))   # negative value


def calmar_ratio(annual_return: float, max_dd: float) -> float:
    return float(_safe_div(annual_return, abs(max_dd)))


def annual_return(equity_curve: List[float], periods_per_year: int = 252) -> float:
    if len(equity_curve) < 2:
        return 0.0
    total_return = equity_curve[-1] / equity_curve[0] - 1.0
    n_years = len(equity_curve) / periods_per_year
    return float((1 + total_return) ** (1.0 / max(n_years, 1e-6)) - 1)


def win_rate(trade_pnls: Optional[List[float]]) -> float:
    if not trade_pnls:
        return 0.5
    wins = sum(1 for p in trade_pnls if p > 0)
    return wins / len(trade_pnls)


def profit_factor(trade_pnls: Optional[List[float]]) -> float:
    if not trade_pnls:
        return 1.0
    gross_win = sum(p for p in trade_pnls if p > 0)
    gross_loss = abs(sum(p for p in trade_pnls if p < 0))
    return _safe_div(gross_win, gross_loss, default=0.0)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_fitness(
    equity_curve: List[float],
    trade_pnls: Optional[List[float]] = None,
    drawdown_penalty_threshold: float = -0.20,
) -> FitnessComponents:
    """
    Compute composite fitness from an equity curve and optional trade P&L list.

    Returns FitnessComponents with .composite in [-1, 1].
    """
    if len(equity_curve) < 5:
        return FitnessComponents(composite=-1.0)

    rets = compute_returns(equity_curve)
    ann_ret = annual_return(equity_curve)
    max_dd = max_drawdown(equity_curve)
    sharpe = sharpe_ratio(rets)
    sortino = sortino_ratio(rets)
    calmar = calmar_ratio(ann_ret, max_dd)
    wr = win_rate(trade_pnls)
    pf = profit_factor(trade_pnls)

    # Normalise each component
    normalised = {
        "sharpe":        _normalise(sharpe, *_BOUNDS["sharpe"]),
        "calmar":        _normalise(calmar, *_BOUNDS["calmar"]),
        "sortino":       _normalise(sortino, *_BOUNDS["sortino"]),
        "annual_return": _normalise(ann_ret, *_BOUNDS["annual_return"]),
        "win_rate":      _normalise(wr, *_BOUNDS["win_rate"]),
        "profit_factor": _normalise(pf, *_BOUNDS["profit_factor"]),
    }

    # Weighted composite
    composite = sum(WEIGHTS[k] * v for k, v in normalised.items())

    # Hard drawdown penalty: if max_dd < threshold, subtract penalty
    if max_dd < drawdown_penalty_threshold:
        severity = (max_dd - drawdown_penalty_threshold) / drawdown_penalty_threshold
        composite -= 0.3 * float(np.clip(severity, 0, 1))

    composite = float(np.clip(composite, -1.0, 1.0))

    return FitnessComponents(
        sharpe=sharpe,
        calmar=calmar,
        sortino=sortino,
        annual_return=ann_ret,
        max_drawdown=max_dd,
        win_rate=wr,
        profit_factor=pf,
        composite=composite,
    )
