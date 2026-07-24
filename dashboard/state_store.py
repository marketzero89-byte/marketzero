"""
LiveStateStore — thread-safe bridge between PBT engine and dashboard server.

The PBT generation loop (running in a background thread) writes snapshots here
via `update()`.  The FastAPI dashboard reads the latest snapshot via `snapshot()`,
which is called by the state_provider callable passed to `create_app()`.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional


class LiveStateStore:
    """
    A simple in-memory, thread-safe store for the latest PBT system state.

    Fields written by the PBT loop:
        generation, best_fitness, mean_fitness, std_fitness,
        equity, drawdown, regime, halted, n_trades,
        leaderboard, equity_history, fitness_history, recent_trades,
        prices, positions, exploit_count, explore_count, elapsed
    """

    _HISTORY_MAX = 500   # keep last N data-points for chart

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state: Dict[str, Any] = {
            "generation": 0,
            "best_fitness": 0.0,
            "mean_fitness": 0.0,
            "cumulative_best_fitness": 0.0,
            "cumulative_mean_fitness": 0.0,
            "fitness_generations": 0,
            "cumulative_return_pct": 0.0,
            "growth_pct": 0.0,
            "_initial_equity": 0.0,
            "_fitness_generation": -1,
            "std_fitness": 0.0,
            "equity": 0.0,
            "drawdown": 0.0,
            "regime": "UNKNOWN",
            "halted": False,
            "n_trades": 0,
            "cumulative_trades": 0,
            "last_generation_trades": 0,
            "leaderboard": [],
            "agent_leaderboard": [],
            "equity_history": [],
            "fitness_history": [],
            "recent_trades": [],
            "prices": {},
            "positions": {},
            "exploit_count": 0,
            "explore_count": 0,
            "elapsed": 0.0,
            "_started": False,
            "performance": {},
            "risk": {},
            "portfolio_analytics": {},
            "trade_analytics": {},
            "ai_monitoring": {},
            "execution": {},
            "infrastructure": {},
            "market_intelligence": {},
            "ai_evolution": {},
            "notifications": [],
            "alerts_events": [],
            "live_decision_stream": [],
        }

    # ------------------------------------------------------------------
    # Write side (called from PBT thread)
    # ------------------------------------------------------------------

    def update(self, patch: Dict[str, Any]) -> None:
        """Merge *patch* dict into the live state (thread-safe)."""
        with self._lock:
            patch = dict(patch)
            declared_initial_equity = patch.get("initial_equity")
            if (
                self._state["_initial_equity"] <= 0
                and declared_initial_equity is not None
                and float(declared_initial_equity) > 0
            ):
                self._state["_initial_equity"] = float(declared_initial_equity)
            equity = patch.get("equity")
            if equity is not None:
                equity = float(equity)
                if self._state["_initial_equity"] <= 0 and equity > 0:
                    self._state["_initial_equity"] = equity
                initial_equity = self._state["_initial_equity"]
                if initial_equity > 0:
                    cumulative_return = (equity / initial_equity - 1.0) * 100.0
                    self._state["cumulative_return_pct"] = round(cumulative_return, 4)
                    self._state["growth_pct"] = round(cumulative_return, 4)
                    # Live agent snapshots can carry a local return basis. The
                    # dashboard must use the persistent account return instead.
                    patch["total_return_pct"] = round(cumulative_return, 4)

            generation = patch.get("generation")
            if "best_fitness" in patch and "mean_fitness" in patch and generation != self._state["_fitness_generation"]:
                best = float(patch["best_fitness"])
                mean = float(patch["mean_fitness"])
                count = self._state["fitness_generations"]
                self._state["cumulative_best_fitness"] = round(
                    max(self._state["cumulative_best_fitness"], best), 6
                )
                self._state["cumulative_mean_fitness"] = round(
                    ((self._state["cumulative_mean_fitness"] * count) + mean) / (count + 1), 6
                )
                self._state["fitness_generations"] = count + 1
                self._state["_fitness_generation"] = generation

            prev_generation = self._state.get("generation", -1)
            prev_last_trades = self._state.get("last_generation_trades", 0)
            if "generation" in patch and "n_trades" in patch and "cumulative_trades" not in patch:
                if patch["generation"] == prev_generation:
                    delta = patch["n_trades"] - prev_last_trades
                else:
                    delta = patch["n_trades"]
                self._state["cumulative_trades"] = self._state.get("cumulative_trades", 0) + delta
                self._state["last_generation_trades"] = patch["n_trades"]

            self._state.update(patch)
            self._state["cumulative_return_pct"] = round(self._state["cumulative_return_pct"], 4)
            self._state["growth_pct"] = self._state["cumulative_return_pct"]
            self._state["_started"] = True
            # Append to rolling history lists
            eq = patch.get("equity")
            bf = patch.get("best_fitness")
            if eq is not None:
                hist = self._state["equity_history"]
                hist.append(round(eq, 2))
                self._state["equity_history"] = hist[-self._HISTORY_MAX:]
            if bf is not None:
                hist = self._state["fitness_history"]
                hist.append(round(bf, 4))
                self._state["fitness_history"] = hist[-self._HISTORY_MAX:]

    def set_halted(self, halted: bool, reason: str = "") -> None:
        with self._lock:
            self._state["halted"] = halted
            if reason:
                self._state["halt_reason"] = reason

    # ------------------------------------------------------------------
    # Read side (called from dashboard / FastAPI thread)
    # ------------------------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        """Return a shallow copy of the current state (thread-safe)."""
        with self._lock:
            return dict(self._state)

    def as_provider(self):
        """Return a zero-argument callable suitable for create_app(state_provider=...).
        
        This wrapper computes performance metrics from equity_curve and trade_pnls
        if they are present in the snapshot, so the dashboard has full metrics
        even during live generation updates.
        """
        def state_provider_with_metrics():
            from core.metrics_engine import compute_performance_metrics, compute_risk_metrics, compute_periodic_returns, compute_rolling_returns
            
            snap = self.snapshot()
            
            # Ensure all dashboard sections exist even before the first full
            # generation result arrives, so the UI can render consistent panels.
            for key in (
                'performance', 'risk', 'portfolio_analytics', 'trade_analytics',
                'ai_monitoring', 'execution', 'infrastructure', 'market_intelligence',
                'ai_evolution', 'notifications', 'alerts_events', 'live_decision_stream',
                'agent_leaderboard'
            ):
                snap.setdefault(key, [] if key in {'notifications', 'alerts_events', 'live_decision_stream', 'agent_leaderboard'} else {})
            
            # Live state is persisted as equity_history, while some code paths
            # still pass an explicit equity_curve. Support both shapes so the
            # dashboard always gets the same performance/risk payloads.
            equity_curve = snap.get('equity_curve') or snap.get('equity_history') or []
            trade_pnls = snap.get('trade_pnls', [])
            
            if equity_curve:
                try:
                    performance = compute_performance_metrics(equity_curve, trade_pnls)
                    risk = compute_risk_metrics(equity_curve, trade_pnls)
                    
                    # Augment with periodic and rolling returns
                    periodic = compute_periodic_returns(equity_curve)
                    performance['daily_return_pct'] = periodic.get('daily')
                    performance['weekly_return_pct'] = periodic.get('weekly')
                    performance['monthly_return_pct'] = periodic.get('monthly')
                    performance['rolling_returns'] = compute_rolling_returns(equity_curve, window=21)
                    
                    # Add to snapshot
                    snap['performance'] = performance
                    snap['risk'] = risk
                except Exception:
                    # If metric computation fails, just use snapshot as-is
                    pass
            
            return snap
        
        return state_provider_with_metrics
