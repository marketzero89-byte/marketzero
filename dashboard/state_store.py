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
            "std_fitness": 0.0,
            "equity": 0.0,
            "drawdown": 0.0,
            "regime": "UNKNOWN",
            "halted": False,
            "n_trades": 0,
            "cumulative_trades": 0,
            "last_generation_trades": 0,
            "leaderboard": [],
            "equity_history": [],
            "fitness_history": [],
            "recent_trades": [],
            "prices": {},
            "positions": {},
            "exploit_count": 0,
            "explore_count": 0,
            "elapsed": 0.0,
            "_started": False,
        }

    # ------------------------------------------------------------------
    # Write side (called from PBT thread)
    # ------------------------------------------------------------------

    def update(self, patch: Dict[str, Any]) -> None:
        """Merge *patch* dict into the live state (thread-safe)."""
        with self._lock:
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
        """Return a zero-argument callable suitable for create_app(state_provider=...)."""
        return self.snapshot
