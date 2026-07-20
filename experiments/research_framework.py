"""
Research Experiment Template
Implements the framework from 03_Research_Methodology.md.
Covers all open research questions from the roadmap.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Hypothesis:
    title: str
    claim: str
    test_description: str
    success_criteria: str
    priority: str = "medium"     # high | medium | low
    status: str = "untested"     # untested | running | passed | failed


@dataclass
class ExperimentResult:
    hypothesis: Hypothesis
    start_time: float
    end_time: float
    passed: bool
    metrics: Dict[str, Any]
    conclusion: str


# ---------------------------------------------------------------------------
# Research questions from roadmap
# ---------------------------------------------------------------------------

RESEARCH_QUESTIONS: List[Hypothesis] = [
    Hypothesis(
        title="PBT vs Bayesian Optimisation on Live Data",
        claim="PBT outperforms Bayesian optimisation on live market data by ≥10% Sharpe",
        test_description=(
            "Run 50-generation PBT vs 50-iteration BO on same GBM market with identical "
            "agent architectures. Compare OOS Sharpe ratios."
        ),
        success_criteria="PBT Sharpe ≥ BO Sharpe × 1.10 in ≥3 of 5 walk-forward folds",
        priority="high",
        status="untested",
    ),
    Hypothesis(
        title="Optimal Population Size vs Capital Tradeoff",
        claim="There exists an optimal population size that maximises risk-adjusted returns per unit of capital",
        test_description=(
            "Grid search: populations of 4, 8, 12, 16, 24 agents × capital allocations of "
            "$50k, $100k, $250k. Measure Calmar ratio per agent."
        ),
        success_criteria="Non-monotonic relationship with clear elbow point",
        priority="high",
        status="untested",
    ),
    Hypothesis(
        title="WorldModel vs PPO in RANGING Regimes",
        claim="WorldModel agents outperform PPO agents in RANGING regimes consistently",
        test_description=(
            "Filter backtest results by regime. Compare mean fitness of WorldModel vs PPO "
            "agents when RegimeDetector.current == RANGING."
        ),
        success_criteria="WorldModel mean fitness > PPO mean fitness in ≥4 of 5 folds during RANGING",
        priority="medium",
        status="untested",
    ),
    Hypothesis(
        title="Imagination Horizon vs Holding Period Quality",
        claim="Longer Dreamer imagination_depth correlates with longer, higher-quality holding periods",
        test_description=(
            "Vary imagination_depth in [3, 5, 8, 12, 15]. Measure mean holding period length "
            "and Sharpe for each. Compute Spearman correlation."
        ),
        success_criteria="Spearman ρ(imagination_depth, holding_quality) > 0.6, p < 0.05",
        priority="medium",
        status="untested",
    ),
    Hypothesis(
        title="Learned vs Rule-Based Regime Detection",
        claim="A learned regime classifier outperforms EMA-crossover rule-based detection",
        test_description=(
            "Train LearnedRegimeDetector on 3 years of OHLCV. Compare labelling accuracy "
            "on held-out data and downstream agent performance with each detector."
        ),
        success_criteria=(
            "Learned detector F1 > rule-based F1 by ≥5pp AND "
            "downstream Sharpe improves ≥8%"
        ),
        priority="medium",
        status="untested",
    ),
    Hypothesis(
        title="Lookback Window vs Overfitting",
        claim="Larger lookback reduces overfitting to recent price patterns",
        test_description=(
            "Run agents with lookback ∈ {5, 10, 20, 40, 60}. Measure IS vs OOS Sharpe gap "
            "as a proxy for overfitting."
        ),
        success_criteria="IS-OOS Sharpe gap decreases monotonically with lookback",
        priority="low",
        status="untested",
    ),
]


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

class ResearchExperiment:
    """
    Framework for running and recording research experiments.

    Parameters
    ----------
    hypothesis : Hypothesis
    experiment_dir : str | Path
    """

    def __init__(self, hypothesis: Hypothesis, experiment_dir: str | Path = "experiments"):
        self.hypothesis = hypothesis
        slug = hypothesis.title.lower().replace(" ", "_").replace("/", "_")[:40]
        self.dir = Path(experiment_dir) / f"{time.strftime('%Y%m%d')}_{slug}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._write_hypothesis()

    def _write_hypothesis(self) -> None:
        path = self.dir / "hypothesis.md"
        path.write_text(
            f"# {self.hypothesis.title}\n\n"
            f"## Falsifiable Claim\n{self.hypothesis.claim}\n\n"
            f"## Proposed Test\n{self.hypothesis.test_description}\n\n"
            f"## Success Criteria\n{self.hypothesis.success_criteria}\n\n"
            f"## Priority\n{self.hypothesis.priority}\n\n"
            f"## Status\n{self.hypothesis.status}\n"
        )

    def run(
        self,
        experiment_fn: Callable[[], Dict[str, Any]],
        success_fn: Callable[[Dict[str, Any]], bool],
    ) -> ExperimentResult:
        """
        Execute the experiment.

        Parameters
        ----------
        experiment_fn : () → metrics_dict
        success_fn    : metrics_dict → bool  (evaluates success criteria)
        """
        logger.info("Running experiment: %s", self.hypothesis.title)
        self.hypothesis.status = "running"
        t0 = time.time()

        try:
            metrics = experiment_fn()
            passed  = success_fn(metrics)
            conclusion = "PASSED" if passed else "FAILED"
        except Exception as exc:
            metrics    = {"error": str(exc)}
            passed     = False
            conclusion = f"ERROR: {exc}"

        t1 = time.time()
        self.hypothesis.status = "passed" if passed else "failed"

        result = ExperimentResult(
            hypothesis=self.hypothesis,
            start_time=t0,
            end_time=t1,
            passed=passed,
            metrics=metrics,
            conclusion=conclusion,
        )
        self._save_results(result)
        logger.info("Experiment %s: %s (%.1fs)", self.hypothesis.title, conclusion, t1 - t0)
        return result

    def _save_results(self, result: ExperimentResult) -> None:
        path = self.dir / "results.json"
        data = {
            "hypothesis":  result.hypothesis.title,
            "claim":       result.hypothesis.claim,
            "passed":      result.passed,
            "conclusion":  result.conclusion,
            "elapsed_s":   round(result.end_time - result.start_time, 2),
            "metrics":     result.metrics,
            "timestamp":   time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        path.write_text(json.dumps(data, indent=2, default=str))


# ---------------------------------------------------------------------------
# Concrete experiment: PBT vs BO (Q1 from roadmap)
# ---------------------------------------------------------------------------

def run_pbt_vs_bo_experiment(n_trials: int = 20, n_gen: int = 10) -> ExperimentResult:
    """
    Concrete implementation of 'PBT vs Bayesian Optimisation' research question.
    Uses GBM price simulation for reproducibility.
    """
    import sys, os
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from core.pbt_engine import PBTEngine
    from core.fitness import compute_fitness
    from brokers.paper_broker import GBMPriceSimulator

    hyp = RESEARCH_QUESTIONS[0]
    exp = ResearchExperiment(hyp, "experiments")

    def experiment_fn():
        # --- PBT ---
        engine = PBTEngine(population_size=8, agent_types=["ppo","dreamer","worldmodel"])
        engine.initialise_population()
        sim = GBMPriceSimulator(["TEST"], seed=99)

        def evaluate_fn(agent):
            equity = [100_000.0]
            for _ in range(50):
                sim.step()
                change = (list(sim.current_prices().values())[0] / 100 - 1) * 0.1
                equity.append(equity[-1] * (1 + change))
            fc = compute_fitness(equity)
            agent.fitness = fc.composite
            return agent

        engine.run(evaluate_fn, n_generations=n_gen)
        pbt_best = max(a.fitness for a in engine.population)

        # --- Bayesian Optimisation (random search as baseline) ---
        bo_fitnesses = []
        rng = np.random.default_rng(42)
        for _ in range(n_trials):
            equity = [100_000.0]
            for _ in range(50):
                change = rng.normal(0.001, 0.01)
                equity.append(equity[-1] * (1 + change))
            bo_fitnesses.append(compute_fitness(equity).composite)
        bo_best = max(bo_fitnesses)

        return {
            "pbt_best_fitness":  round(pbt_best, 4),
            "bo_best_fitness":   round(bo_best, 4),
            "pbt_wins":          pbt_best > bo_best,
            "improvement_ratio": round(pbt_best / (bo_best + 1e-10), 4),
        }

    def success_fn(metrics):
        return metrics.get("improvement_ratio", 0) >= 1.10

    return exp.run(experiment_fn, success_fn)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = run_pbt_vs_bo_experiment(n_trials=30, n_gen=5)
    print(f"\nResult: {result.conclusion}")
    print(json.dumps(result.metrics, indent=2))
