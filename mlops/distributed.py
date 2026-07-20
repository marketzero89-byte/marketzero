"""
Distributed PBT
Multi-process population training across CPU cores.
Uses Python multiprocessing with shared memory via Manager.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from core.pbt_engine import AgentRecord, PBTEngine, GenerationResult

logger = logging.getLogger(__name__)


def _worker_evaluate(args: tuple) -> AgentRecord:
    """Top-level function for multiprocessing (must be picklable)."""
    agent_dict, evaluate_fn_name, evaluate_fn_kwargs = args
    agent = AgentRecord(**agent_dict)
    # Reconstruct evaluate function from factory
    evaluate_fn = _WORKER_FN_REGISTRY.get(evaluate_fn_name)
    if evaluate_fn is None:
        raise RuntimeError(f"Evaluate fn '{evaluate_fn_name}' not registered")
    return evaluate_fn(agent, **evaluate_fn_kwargs)


# Global registry for evaluate functions (needed because lambdas are not picklable)
_WORKER_FN_REGISTRY: Dict[str, Callable] = {}


def register_evaluate_fn(name: str, fn: Callable) -> None:
    """Register a named evaluate function for use by worker processes."""
    _WORKER_FN_REGISTRY[name] = fn


class DistributedPBTEngine(PBTEngine):
    """
    PBT Engine with multi-process evaluation.

    Parameters
    ----------
    n_workers : int   Number of worker processes (default: CPU count - 1)
    All other parameters are inherited from PBTEngine.
    """

    def __init__(
        self,
        n_workers: Optional[int] = None,
        **engine_kwargs,
    ):
        super().__init__(**engine_kwargs)
        self.n_workers = n_workers or max(1, mp.cpu_count() - 1)
        logger.info("DistributedPBTEngine: %d workers", self.n_workers)

    def run_generation(
        self,
        evaluate_fn: Callable,
        evaluate_fn_name: str = "default",
        evaluate_fn_kwargs: Optional[Dict] = None,
        **kwargs,
    ) -> GenerationResult:
        """
        Run one PBT generation using a process pool.

        Parameters
        ----------
        evaluate_fn : callable(AgentRecord) -> AgentRecord
        evaluate_fn_name : str  Key used to look up fn in worker processes
        evaluate_fn_kwargs : dict  Extra kwargs passed to workers
        """
        # Register for workers
        register_evaluate_fn(evaluate_fn_name, evaluate_fn)

        self.generation += 1
        t0 = time.monotonic()
        logger.info("=== Distributed Generation %d (%d workers) ===",
                    self.generation, self.n_workers)

        # Prepare args
        agent_dicts = [
            (
                {
                    "agent_id":    a.agent_id,
                    "agent_type":  a.agent_type,
                    "hyperparams": a.hyperparams,
                    "fitness":     a.fitness,
                    "generation":  a.generation,
                    "parent_id":   a.parent_id,
                    "lineage":     a.lineage,
                    "metrics":     a.metrics,
                    "alive":       a.alive,
                },
                evaluate_fn_name,
                evaluate_fn_kwargs or {},
            )
            for a in self.population
        ]

        # Distribute across workers
        with mp.Pool(processes=self.n_workers) as pool:
            updated = pool.map(_worker_evaluate, agent_dicts)

        self.population = updated
        for agent in self.population:
            agent.generation = self.generation

        exploit_count = self._exploit()
        explore_count = self._explore()

        fitnesses = [a.fitness for a in self.population]
        import copy
        best = max(self.population, key=lambda a: a.fitness)
        result = GenerationResult(
            generation=self.generation,
            elapsed_seconds=time.monotonic() - t0,
            population=copy.deepcopy(self.population),
            best_agent=copy.deepcopy(best),
            mean_fitness=float(np.mean(fitnesses)),
            std_fitness=float(np.std(fitnesses)),
            exploit_count=exploit_count,
            explore_count=explore_count,
        )
        self.history.append(result)
        self._save_checkpoint(result)

        for cb in self._callbacks:
            cb(result)

        logger.info(
            "Distributed Gen %d | best=%.4f mean=%.4f (%.1fs)",
            self.generation, best.fitness, result.mean_fitness, result.elapsed_seconds,
        )
        return result

    def run(
        self,
        evaluate_fn: Callable,
        n_generations: int = 50,
        evaluate_fn_name: str = "default",
        evaluate_fn_kwargs: Optional[Dict] = None,
        **kwargs,
    ):
        """Run full distributed PBT loop."""
        if not self.population:
            self.initialise_population()
        for _ in range(n_generations):
            self.run_generation(
                evaluate_fn=evaluate_fn,
                evaluate_fn_name=evaluate_fn_name,
                evaluate_fn_kwargs=evaluate_fn_kwargs,
            )
        return self.history


# ---------------------------------------------------------------------------
# GPU acceleration placeholder (v1.2 — PyTorch neural network agents)
# ---------------------------------------------------------------------------

class GPUAcceleratedTrainer:
    """
    Placeholder for GPU-accelerated neural network agent training.
    Requires: pip install torch
    """

    def __init__(self, device: str = "auto"):
        try:
            import torch
            if device == "auto":
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(device)
            logger.info("GPU trainer: device=%s", self.device)
            self._torch = torch
        except ImportError:
            logger.warning("PyTorch not installed; GPU acceleration unavailable")
            self._torch = None
            self.device = "cpu"

    def is_available(self) -> bool:
        return self._torch is not None and str(self.device) != "cpu"

    def to_tensor(self, arr) -> Any:
        if self._torch is None:
            return arr
        import numpy as np
        return self._torch.tensor(np.array(arr), dtype=self._torch.float32, device=self.device)
