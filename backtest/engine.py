"""
Backtesting Engine
Walk-forward analysis, Monte Carlo simulation, out-of-sample reporting.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from core.fitness import compute_fitness, FitnessComponents

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    fold: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    equity_curve: List[float]
    trade_pnls: List[float]
    fitness: FitnessComponents
    elapsed: float = 0.0


@dataclass
class WalkForwardReport:
    n_folds: int
    results: List[BacktestResult]
    oos_sharpe:     float
    oos_calmar:     float
    oos_annual_ret: float
    oos_max_dd:     float
    oos_win_rate:   float
    consistency:    float   # % folds with positive return


@dataclass
class MonteCarloReport:
    n_runs: int
    confidence: float
    sharpe_ci:      Tuple[float, float]
    calmar_ci:      Tuple[float, float]
    annual_ret_ci:  Tuple[float, float]
    max_dd_ci:      Tuple[float, float]
    median_sharpe:  float
    pct_positive:   float


# ---------------------------------------------------------------------------
# Walk-Forward Analysis
# ---------------------------------------------------------------------------

class WalkForwardAnalyser:
    """
    k-fold walk-forward backtesting with out-of-sample reporting.

    Parameters
    ----------
    n_folds : int
    train_pct : float   fraction of each fold used for training
    """

    def __init__(self, n_folds: int = 5, train_pct: float = 0.70):
        self.n_folds   = n_folds
        self.train_pct = train_pct

    def run(
        self,
        prices:      List[float],
        strategy_fn: Callable[[List[float], List[float]], Tuple[List[float], List[float]]],
        # strategy_fn(train_prices, test_prices) → (equity_curve, trade_pnls)
    ) -> WalkForwardReport:
        """
        Run k-fold walk-forward.

        Parameters
        ----------
        prices       : full price series
        strategy_fn  : function that trains on train_prices, returns results on test_prices
        """
        n = len(prices)
        fold_size = n // self.n_folds
        results: List[BacktestResult] = []

        for k in range(self.n_folds):
            fold_start = k * fold_size
            fold_end   = fold_start + fold_size if k < self.n_folds - 1 else n
            fold_prices = prices[fold_start:fold_end]

            train_n = int(len(fold_prices) * self.train_pct)
            train_prices = fold_prices[:train_n]
            test_prices  = fold_prices[train_n:]

            if len(test_prices) < 5:
                logger.warning("Fold %d: insufficient test data, skipping", k)
                continue

            t0 = time.monotonic()
            try:
                equity_curve, trade_pnls = strategy_fn(train_prices, test_prices)
            except Exception as exc:
                logger.error("Fold %d strategy error: %s", k, exc)
                equity_curve = [1.0]
                trade_pnls   = []

            fc = compute_fitness(equity_curve, trade_pnls)
            results.append(BacktestResult(
                fold=k,
                train_start=fold_start,
                train_end=fold_start + train_n,
                test_start=fold_start + train_n,
                test_end=fold_end,
                equity_curve=equity_curve,
                trade_pnls=trade_pnls,
                fitness=fc,
                elapsed=time.monotonic() - t0,
            ))
            logger.info(
                "Fold %d | sharpe=%.3f  calmar=%.3f  ret=%.2f%%  dd=%.2f%%",
                k, fc.sharpe, fc.calmar, fc.annual_return * 100, fc.max_drawdown * 100,
            )

        return self._aggregate(results)

    def _aggregate(self, results: List[BacktestResult]) -> WalkForwardReport:
        if not results:
            return WalkForwardReport(
                n_folds=self.n_folds, results=[], oos_sharpe=0, oos_calmar=0,
                oos_annual_ret=0, oos_max_dd=0, oos_win_rate=0, consistency=0,
            )
        sharpes    = [r.fitness.sharpe        for r in results]
        calmars    = [r.fitness.calmar        for r in results]
        ann_rets   = [r.fitness.annual_return for r in results]
        max_dds    = [r.fitness.max_drawdown  for r in results]
        win_rates  = [r.fitness.win_rate      for r in results]

        return WalkForwardReport(
            n_folds=self.n_folds,
            results=results,
            oos_sharpe=float(np.mean(sharpes)),
            oos_calmar=float(np.mean(calmars)),
            oos_annual_ret=float(np.mean(ann_rets)),
            oos_max_dd=float(np.min(max_dds)),
            oos_win_rate=float(np.mean(win_rates)),
            consistency=float(np.mean([r > 0 for r in ann_rets])),
        )


# ---------------------------------------------------------------------------
# Monte Carlo Simulation
# ---------------------------------------------------------------------------

class MonteCarloSimulator:
    """
    Bootstrap Monte Carlo for confidence intervals on performance metrics.

    Parameters
    ----------
    n_runs : int    number of bootstrap runs
    confidence : float  e.g. 0.95 for 95% CI
    """

    def __init__(self, n_runs: int = 1_000, confidence: float = 0.95):
        self.n_runs     = n_runs
        self.confidence = confidence

    def run(
        self,
        trade_pnls: List[float],
        initial_equity: float = 100_000.0,
    ) -> MonteCarloReport:
        """
        Bootstrap resample trade P&Ls to estimate distribution of metrics.
        """
        if not trade_pnls:
            return self._empty_report()

        rng      = np.random.default_rng(42)
        pnls     = np.array(trade_pnls)
        n_trades = len(pnls)

        sharpes   = []
        calmars   = []
        ann_rets  = []
        max_dds   = []

        for _ in range(self.n_runs):
            sample = rng.choice(pnls, size=n_trades, replace=True)
            equity = [initial_equity]
            for pnl in sample:
                equity.append(equity[-1] + pnl)
            fc = compute_fitness(equity, list(sample))
            sharpes.append(fc.sharpe)
            calmars.append(fc.calmar)
            ann_rets.append(fc.annual_return)
            max_dds.append(fc.max_drawdown)

        alpha = (1 - self.confidence) / 2.0

        def ci(arr):
            a = np.array(arr)
            return float(np.quantile(a, alpha)), float(np.quantile(a, 1 - alpha))

        return MonteCarloReport(
            n_runs=self.n_runs,
            confidence=self.confidence,
            sharpe_ci=ci(sharpes),
            calmar_ci=ci(calmars),
            annual_ret_ci=ci(ann_rets),
            max_dd_ci=ci(max_dds),
            median_sharpe=float(np.median(sharpes)),
            pct_positive=float(np.mean([r > 0 for r in ann_rets])),
        )

    def _empty_report(self) -> MonteCarloReport:
        return MonteCarloReport(
            n_runs=self.n_runs, confidence=self.confidence,
            sharpe_ci=(0.0, 0.0), calmar_ci=(0.0, 0.0),
            annual_ret_ci=(0.0, 0.0), max_dd_ci=(0.0, 0.0),
            median_sharpe=0.0, pct_positive=0.0,
        )


# ---------------------------------------------------------------------------
# Report serialisation
# ---------------------------------------------------------------------------

def save_backtest_report(
    wf: WalkForwardReport,
    mc: Optional[MonteCarloReport],
    path: str | Path,
    *,
    data_source: str = "synthetic",
    data_file: str = "",
    symbol: str = "",
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "data_source": data_source,
        "data_file":   data_file,
        "symbol":      symbol,
        "walk_forward": {
            "n_folds":      wf.n_folds,
            "oos_sharpe":   round(wf.oos_sharpe, 4),
            "oos_calmar":   round(wf.oos_calmar, 4),
            "oos_annual_ret": round(wf.oos_annual_ret, 4),
            "oos_max_dd":   round(wf.oos_max_dd, 4),
            "oos_win_rate": round(wf.oos_win_rate, 4),
            "consistency":  round(wf.consistency, 4),
            "folds": [
                {
                    "fold": r.fold,
                    "sharpe":      round(r.fitness.sharpe, 4),
                    "calmar":      round(r.fitness.calmar, 4),
                    "annual_return": round(r.fitness.annual_return, 4),
                    "max_drawdown":  round(r.fitness.max_drawdown, 4),
                    "win_rate":    round(r.fitness.win_rate, 4),
                    "n_trades":    len(r.trade_pnls),
                    "elapsed":     round(r.elapsed, 2),
                }
                for r in wf.results
            ],
        },
        "monte_carlo": {
            "n_runs":        mc.n_runs,
            "confidence":    mc.confidence,
            "sharpe_ci":     [round(mc.sharpe_ci[0], 4), round(mc.sharpe_ci[1], 4)],
            "calmar_ci":     [round(mc.calmar_ci[0], 4), round(mc.calmar_ci[1], 4)],
            "annual_ret_ci": [round(mc.annual_ret_ci[0], 4), round(mc.annual_ret_ci[1], 4)],
            "max_dd_ci":     [round(mc.max_dd_ci[0], 4), round(mc.max_dd_ci[1], 4)],
            "median_sharpe": round(mc.median_sharpe, 4),
            "pct_positive":  round(mc.pct_positive, 4),
        } if mc else None,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    path.write_text(json.dumps(report, indent=2))
    logger.info("Backtest report saved to %s", path)
