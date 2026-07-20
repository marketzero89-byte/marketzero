"""
Population-Based Training (PBT) Engine
Core orchestration: population init, generation loop, exploit/explore cycle.
"""

from __future__ import annotations

import copy
import logging
import random
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AgentRecord:
    agent_id: str
    agent_type: str          # 'ppo' | 'dreamer' | 'worldmodel'
    hyperparams: Dict[str, Any]
    fitness: float = 0.0
    generation: int = 0
    parent_id: Optional[str] = None
    lineage: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    checkpoint_path: Optional[str] = None
    alive: bool = True


@dataclass
class GenerationResult:
    generation: int
    elapsed_seconds: float
    population: List[AgentRecord]
    best_agent: AgentRecord
    mean_fitness: float
    std_fitness: float
    exploit_count: int
    explore_count: int


# ---------------------------------------------------------------------------
# Hyper-parameter perturbation
# ---------------------------------------------------------------------------

# R-033: HYPERPARAM_RANGES defines the legal search bounds for Population-Based Training.
# Format per key: (lower_bound, upper_bound, kind)
#   kind = 'log'    — perturb multiplicatively (good for learning rates)
#   kind = 'linear' — perturb multiplicatively in linear space
#   kind = 'int'    — integer step perturbation (±20% delta, min ±1)
# These bounds are enforced by perturb() and respected by sample_hyperparams().
# To add a new agent-specific hyperparameter, add it here AND handle it in sample_hyperparams().
HYPERPARAM_RANGES: Dict[str, Tuple] = {
    "learning_rate":    (1e-5, 1e-2, "log"),
    "gamma":            (0.90, 0.999, "linear"),
    "entropy_coef":     (0.0, 0.05, "linear"),
    "clip_eps":         (0.1, 0.3, "linear"),
    "horizon":          (5, 50, "int"),
    "imagination_depth":(1, 15, "int"),
    "ensemble_size":    (3, 10, "int"),
    "lookback":         (10, 100, "int"),
}

PERTURB_FACTORS = [0.8, 1.2]   # multiply by one of these on explore


def perturb(value: float, spec: Tuple) -> float:
    lo, hi, kind = spec
    factor = random.choice(PERTURB_FACTORS)
    if kind == "log":
        new_val = value * factor
    elif kind == "int":
        delta = max(1, int(abs(value) * 0.2))
        new_val = value + random.choice([-delta, delta])
        new_val = int(new_val)
    else:
        new_val = value * factor
    return float(np.clip(new_val, lo, hi))


def perturb_hyperparams(hp: Dict[str, Any], mutate_prob: float = 0.8) -> Dict[str, Any]:
    """R-081: each hyperparameter perturbed with mutate_prob=0.8 probability."""
    new_hp = copy.deepcopy(hp)
    for key in list(hp.keys()):
        if random.random() > mutate_prob:   # skip with (1-mutate_prob) probability
            continue
        if key in HYPERPARAM_RANGES:
            new_hp[key] = perturb(hp[key], HYPERPARAM_RANGES[key])
    return new_hp


def sample_hyperparams(agent_type: str) -> Dict[str, Any]:
    base = {
        "learning_rate":    10 ** random.uniform(-4, -2),
        "gamma":            random.uniform(0.95, 0.999),
        "entropy_coef":     random.uniform(0.0, 0.02),
        "lookback":         random.randint(10, 60),
    }
    if agent_type == "ppo":
        base["clip_eps"] = random.uniform(0.1, 0.3)
    elif agent_type == "dreamer":
        base["imagination_depth"] = random.randint(3, 15)
    elif agent_type == "worldmodel":
        base["ensemble_size"] = random.randint(3, 8)
    return base


# ---------------------------------------------------------------------------
# PBT Engine
# ---------------------------------------------------------------------------

class PBTEngine:
    """
    Orchestrates Population-Based Training across a heterogeneous agent pool.

    Parameters
    ----------
    population_size : int
        Total number of agents in the population.
    agent_types : list[str]
        Agent family names; population is evenly distributed across them.
    exploit_fraction : float
        Top-k fraction of population whose hyperparams are copied to bottom-k.
    checkpoint_dir : str | Path
        Where per-generation snapshots are written.
    """

    def __init__(
        self,
        population_size: int = 12,
        agent_types: Optional[List[str]] = None,
        exploit_fraction: float = 0.2,
        checkpoint_dir: str | Path = "checkpoints",
        seed: int = 42,
    ):
        random.seed(seed)
        np.random.seed(seed)

        self.population_size = population_size
        self.agent_types = agent_types or ["ppo", "dreamer", "worldmodel"]
        self.exploit_fraction = exploit_fraction
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.population: List[AgentRecord] = []
        self.generation: int = 0
        self.history: List[GenerationResult] = []
        self.genealogy: Dict[str, List[str]] = {}   # agent_id -> [parent, grandparent, ...]
        self._genealogy_log: Path = Path("logs") / "pbt_genealogy.json"  # R-019
        self._genealogy_log.parent.mkdir(parents=True, exist_ok=True)

        self._callbacks: List[Any] = []

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def initialise_population(self) -> None:
        """Create a fresh heterogeneous population."""
        self.population.clear()
        types_cycle = [
            self.agent_types[i % len(self.agent_types)]
            for i in range(self.population_size)
        ]
        for atype in types_cycle:
            agent_id = str(uuid.uuid4())[:8]
            record = AgentRecord(
                agent_id=agent_id,
                agent_type=atype,
                hyperparams=sample_hyperparams(atype),
            )
            self.population.append(record)
            self.genealogy[agent_id] = []
        logger.info("Initialised population of %d agents", self.population_size)

    # ------------------------------------------------------------------
    # Exploit / Explore
    # ------------------------------------------------------------------

    def _exploit(self) -> int:
        """
        Copy hyperparams from top-k agents to bottom-k agents.
        Returns the number of agents that were exploited.
        """
        k = max(1, int(self.population_size * self.exploit_fraction))
        sorted_pop = sorted(self.population, key=lambda a: a.fitness, reverse=True)
        top_k = sorted_pop[:k]
        bottom_k = sorted_pop[-k:]

        count = 0
        for loser, winner in zip(bottom_k, top_k):
            if loser.agent_id == winner.agent_id:
                continue
            hp_before = copy.deepcopy(loser.hyperparams)
            loser.hyperparams = copy.deepcopy(winner.hyperparams)
            loser.parent_id = winner.agent_id
            # Update genealogy
            lineage = [winner.agent_id] + self.genealogy.get(winner.agent_id, [])
            self.genealogy[loser.agent_id] = lineage[:10]   # keep last 10
            # R-084/R-085: write genealogy event
            self._append_genealogy({
                "generation":        self.generation,
                "agent_id":          loser.agent_id,
                "parent_id":         winner.agent_id,
                "event":             "exploit",
                "hyperparams_before": hp_before,
                "hyperparams_after":  copy.deepcopy(loser.hyperparams),
                "timestamp":         time.time(),
            })
            count += 1
        return count

    def _explore(self) -> int:
        """
        Perturb hyperparams of agents that were exploited (or randomly chosen).
        Returns number of agents perturbed.
        """
        k = max(1, int(self.population_size * self.exploit_fraction))
        bottom_k = sorted(self.population, key=lambda a: a.fitness)[:k]
        for agent in bottom_k:
            hp_before = copy.deepcopy(agent.hyperparams)
            agent.hyperparams = perturb_hyperparams(agent.hyperparams)
            # R-084/R-085: write genealogy event
            self._append_genealogy({
                "generation":        self.generation,
                "agent_id":          agent.agent_id,
                "parent_id":         None,
                "event":             "explore",
                "hyperparams_before": hp_before,
                "hyperparams_after":  copy.deepcopy(agent.hyperparams),
                "timestamp":         time.time(),
            })
        return len(bottom_k)

    # ------------------------------------------------------------------
    # Generation loop
    # ------------------------------------------------------------------

    def run_generation(
        self,
        evaluate_fn,               # Callable[[AgentRecord], AgentRecord]
        parallel: bool = False,
    ) -> GenerationResult:
        """
        Run one PBT generation:
          1. Evaluate every agent → update fitness
          2. Exploit: copy top->bottom
          3. Explore: perturb bottom hyperparams
          4. Save checkpoint
        """
        self.generation += 1
        t0 = time.monotonic()
        logger.info("=== Generation %d ===", self.generation)

        # Reset daily P&L baseline at the start of each generation
        if hasattr(self, '_on_generation_start'):
            self._on_generation_start()

        # --- Evaluate ---
        if parallel:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor() as ex:
                updated = list(ex.map(evaluate_fn, self.population))
            self.population = updated
        else:
            self.population = [evaluate_fn(a) for a in self.population]

        for agent in self.population:
            agent.generation = self.generation

        # --- Exploit / Explore ---
        exploit_count = self._exploit()
        explore_count = self._explore()

        # --- Stats ---
        fitnesses = [a.fitness for a in self.population]
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
            "Gen %d done | best=%.4f mean=%.4f std=%.4f exploit=%d explore=%d (%.1fs)",
            self.generation, best.fitness, result.mean_fitness,
            result.std_fitness, exploit_count, explore_count,
            result.elapsed_seconds,
        )
        return result

    def run(self, evaluate_fn, n_generations: int = 50, parallel: bool = False):
        """Run full PBT loop for n_generations."""
        if not self.population:
            self.initialise_population()
        for _ in range(n_generations):
            self.run_generation(evaluate_fn, parallel=parallel)
        return self.history

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def _save_checkpoint(self, result: GenerationResult) -> None:
        import json
        gen_dir = self.checkpoint_dir / f"gen_{result.generation:04d}"
        gen_dir.mkdir(parents=True, exist_ok=True)
        # population.json
        pop_path = gen_dir / "population.json"
        data = {
            "generation": result.generation,
            "timestamp": time.time(),
            "mean_fitness": result.mean_fitness,
            "std_fitness": result.std_fitness,
            "exploit_count": result.exploit_count,
            "explore_count": result.explore_count,
            "population": [
                {
                    "agent_id": a.agent_id,
                    "agent_type": a.agent_type,
                    "fitness": a.fitness,
                    "hyperparams": a.hyperparams,
                    "parent_id": a.parent_id,
                    "metrics": a.metrics,
                }
                for a in result.population
            ],
        }
        pop_path.write_text(json.dumps(data, indent=2))
        # Also write flat gen_NNNN.json for backward-compat
        flat_path = self.checkpoint_dir / f"gen_{result.generation:04d}.json"
        flat_path.write_text(json.dumps(data, indent=2))
        # config.json (R-017)
        config_path = gen_dir / "config.json"
        config_path.write_text(json.dumps({
            "population_size":   self.population_size,
            "agent_types":       self.agent_types,
            "exploit_fraction":  self.exploit_fraction,
            "checkpoint_dir":    str(self.checkpoint_dir),
        }, indent=2))
        # R-018: prune to last 10 checkpoint directories
        self._prune_checkpoints(keep=10)

    def _prune_checkpoints(self, keep: int = 10) -> None:
        """R-018: Delete old gen_NNNN/ directories, retaining only the last `keep`."""
        dirs = sorted(self.checkpoint_dir.glob("gen_*/"), key=lambda p: p.name)
        to_delete = dirs[:-keep] if len(dirs) > keep else []
        for d in to_delete:
            import shutil
            try:
                shutil.rmtree(d)
            except OSError:
                pass
        # Also prune flat JSON files
        flat_files = sorted(self.checkpoint_dir.glob("gen_*.json"), key=lambda p: p.name)
        for f in flat_files[:-keep]:
            try:
                f.unlink()
            except OSError:
                pass

    def _append_genealogy(self, record: dict) -> None:
        """R-019: Append-only write to logs/pbt_genealogy.json."""
        import json
        try:
            with self._genealogy_log.open("a") as fh:
                fh.write(json.dumps(record) + "\n")
        except OSError as exc:
            logger.warning("Failed to write genealogy log: %s", exc)

    @classmethod
    def load_checkpoint(cls, checkpoint_path: str | Path, **engine_kwargs) -> "PBTEngine":
        import json
        data = json.loads(Path(checkpoint_path).read_text())
        engine = cls(**engine_kwargs)
        engine.generation = data["generation"]
        engine.population = [
            AgentRecord(
                agent_id=a["agent_id"],
                agent_type=a["agent_type"],
                fitness=a["fitness"],
                hyperparams=a["hyperparams"],
                parent_id=a.get("parent_id"),
                metrics=a.get("metrics", {}),
                generation=data["generation"],
            )
            for a in data["population"]
        ]
        logger.info("Resumed from checkpoint gen=%d", engine.generation)
        return engine

    # ------------------------------------------------------------------
    # Leaderboard
    # ------------------------------------------------------------------

    def leaderboard(self, top_n: int = 10) -> List[AgentRecord]:
        return sorted(self.population, key=lambda a: a.fitness, reverse=True)[:top_n]

    def add_callback(self, fn) -> None:
        self._callbacks.append(fn)
    
    def set_generation_start_hook(self, fn) -> None:
        """Register a callable invoked at the start of each generation."""
        self._on_generation_start = fn
