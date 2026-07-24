"""
Online Evaluator
Rolling fitness tracker, leaderboard, and population statistics per generation.
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional

import numpy as np

from core.pbt_engine import AgentRecord
from core.fitness import compute_fitness, FitnessComponents

logger = logging.getLogger(__name__)


@dataclass
class AgentStats:
    agent_id: str
    agent_type: str
    fitness_history: List[float] = field(default_factory=list)
    generation_history: List[int] = field(default_factory=list)
    rolling_mean: float = 0.0
    rolling_std: float = 0.0
    peak_fitness: float = -1.0
    current_fitness: float = 0.0


@dataclass
class PopulationSnapshot:
    generation: int
    timestamp: float
    mean_fitness: float
    std_fitness: float
    max_fitness: float
    min_fitness: float
    leaderboard: List[Dict]
    regime: str = "UNKNOWN"


class OnlineEvaluator:
    """
    Wraps the fitness function and maintains rolling statistics across generations.

    Parameters
    ----------
    window : int
        Rolling window size for smoothed fitness computation.
    log_path : str | Path
        JSON-lines log file path.
    """

    def __init__(
        self,
        window: int = 10,
        log_path: str | Path = "logs/evaluator.jsonl",
        tensorboard: bool = False,
    ):
        self.window = window
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        # R-024: Log rotation hint — this file grows unbounded in long runs.
        # For production, use logrotate(8) or Python's logging.handlers.RotatingFileHandler
        # pointed at this path.  Suggested config: maxBytes=50_000_000, backupCount=5.

        self._agent_stats: Dict[str, AgentStats] = {}
        self._snapshots: List[PopulationSnapshot] = []

        self._tb_writer = None
        if tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._tb_writer = SummaryWriter("runs/pbt")
                logger.info("TensorBoard writer initialised")
            except ImportError:
                logger.warning("TensorBoard not available; skipping")

    # ------------------------------------------------------------------
    # Evaluate a single agent given its equity curve + trades
    # ------------------------------------------------------------------

    def evaluate_agent(
        self,
        agent: AgentRecord,
        equity_curve: List[float],
        trade_pnls: Optional[List[float]] = None,
        regime: str = "UNKNOWN",
    ) -> AgentRecord:
        """Compute fitness, update agent record and internal stats."""
        fc: FitnessComponents = compute_fitness(equity_curve, trade_pnls)
        agent.fitness = fc.composite
        generation_return = 0.0
        if equity_curve and equity_curve[0] != 0:
            generation_return = equity_curve[-1] / equity_curve[0] - 1.0
        agent.metrics = {
            "sharpe":        round(fc.sharpe, 4),
            "calmar":        round(fc.calmar, 4),
            "sortino":       round(fc.sortino, 4),
            "annual_return": round(fc.annual_return, 4),
            "generation_return": round(generation_return, 6),
            "max_drawdown":  round(fc.max_drawdown, 4),
            "win_rate":      round(fc.win_rate, 4),
            "profit_factor": round(fc.profit_factor, 4),
            "composite":     round(fc.composite, 4),
            "regime":        regime,
        }

        # Update rolling stats
        stats = self._agent_stats.setdefault(
            agent.agent_id,
            AgentStats(agent_id=agent.agent_id, agent_type=agent.agent_type),
        )
        stats.fitness_history.append(fc.composite)
        stats.generation_history.append(agent.generation)
        stats.current_fitness = fc.composite
        stats.peak_fitness = max(stats.peak_fitness, fc.composite)

        hist = stats.fitness_history[-self.window:]
        stats.rolling_mean = float(np.mean(hist))
        stats.rolling_std = float(np.std(hist))

        logger.debug(
            "Agent %s (%s) fitness=%.4f sharpe=%.3f calmar=%.3f drawdown=%.2f%%",
            agent.agent_id, agent.agent_type, fc.composite,
            fc.sharpe, fc.calmar, fc.max_drawdown * 100,
        )
        return agent

    # ------------------------------------------------------------------
    # Population-level snapshot after a full generation
    # ------------------------------------------------------------------

    def record_generation(
        self,
        population: List[AgentRecord],
        generation: int,
        regime: str = "UNKNOWN",
    ) -> PopulationSnapshot:
        fitnesses = [a.fitness for a in population]
        leaderboard = [
            {
                "rank":       i + 1,
                "agent_id":   a.agent_id,
                "agent_type": a.agent_type,
                "fitness":    round(a.fitness, 4),
                "metrics":    a.metrics,
            }
            for i, a in enumerate(
                sorted(population, key=lambda x: x.fitness, reverse=True)[:10]
            )
        ]
        snap = PopulationSnapshot(
            generation=generation,
            timestamp=time.time(),
            mean_fitness=float(np.mean(fitnesses)),
            std_fitness=float(np.std(fitnesses)),
            max_fitness=float(np.max(fitnesses)),
            min_fitness=float(np.min(fitnesses)),
            leaderboard=leaderboard,
            regime=regime,
        )
        self._snapshots.append(snap)
        self._log_jsonl(snap)

        if self._tb_writer:
            self._tb_writer.add_scalar("fitness/mean",  snap.mean_fitness,  generation)
            self._tb_writer.add_scalar("fitness/max",   snap.max_fitness,   generation)
            self._tb_writer.add_scalar("fitness/std",   snap.std_fitness,   generation)

        return snap

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _log_jsonl(self, snap: PopulationSnapshot) -> None:
        record = {
            "generation":    snap.generation,
            "timestamp":     snap.timestamp,
            "mean_fitness":  snap.mean_fitness,
            "std_fitness":   snap.std_fitness,
            "max_fitness":   snap.max_fitness,
            "min_fitness":   snap.min_fitness,
            "regime":        snap.regime,
            "leaderboard":   snap.leaderboard,
        }
        try:   # R-115: IOError must not crash the trading loop
            with self.log_path.open("a") as f:
                f.write(json.dumps(record) + "\n")
        except IOError as exc:
            logger.error("Evaluator log write failed (disk full?): %s", exc)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def leaderboard(self, population: List[AgentRecord], top_n: int = 10) -> List[Dict]:
        sorted_pop = sorted(population, key=lambda a: a.fitness, reverse=True)
        return [
            {
                "rank":        i + 1,
                "agent_id":    a.agent_id,
                "agent_type":  a.agent_type,
                "fitness":     round(a.fitness, 4),
                "rolling_mean": round(
                    self._agent_stats.get(a.agent_id, AgentStats(a.agent_id, a.agent_type)).rolling_mean, 4
                ),
                **{k: round(v, 4) if isinstance(v, float) else v
                   for k, v in (a.metrics or {}).items()},
            }
            for i, a in enumerate(sorted_pop[:top_n])
        ]

    def generation_history(self) -> List[Dict]:
        return [
            {
                "generation":   s.generation,
                "mean_fitness": s.mean_fitness,
                "std_fitness":  s.std_fitness,
                "max_fitness":  s.max_fitness,
                "min_fitness":  s.min_fitness,
                "regime":       s.regime,
            }
            for s in self._snapshots
        ]

    def agent_stats(self, agent_id: str) -> Optional[AgentStats]:
        return self._agent_stats.get(agent_id)
