# 17 — PBT Framework

---

## What is Population-Based Training?

Population-Based Training (PBT) is a hyperparameter optimisation method that runs a population of models simultaneously, using evolutionary pressure to identify better configurations without manual tuning.

Unlike grid search (evaluate all combinations) or Bayesian optimisation (sequential), PBT combines training and hyperparameter search into a single continuous loop:

```
Train all agents simultaneously
        │
At generation end:
        ▼
Evaluate fitness of all agents
        │
        ▼
Exploit: bottom 20% copy from top 20%
        │
        ▼
Explore: non-elite agents mutate hyperparameters
        │
        ▼
Continue training with new configurations
```

---

## MarketZero PBT Implementation

Source: `pbt/pbt_evolution.py`, `pbt/pbt_live_trading.py`

### Population Configuration

| Parameter | Default | Notes |
|---|---|---|
| `population_size` | 12 | Total agents |
| PPO agents | 6 | `n // 2` |
| Dreamer agents | 4 | `(n - ppo) // 2` |
| WorldModel agents | 2 | remainder |
| `generation_steps` | 500 | Steps per generation |
| `mutation_strength` | 0.20 | σ for Gaussian perturbation |

### Exploit Phase

At the end of each generation:

1. Rank all agents by fitness score
2. Identify bottom 20% (`exploit_fraction = 0.2`)
3. For each bottom agent: sample a random top-20% agent and copy its hyperparameters
4. Bottom agent weights are reset or partially inherited (configurable)
5. Elite agent (rank 1) is never touched

```python
bottom_agents = ranked[int(0.8 * n):]
top_agents    = ranked[:int(0.2 * n)]
for agent in bottom_agents:
    donor = random.choice(top_agents)
    agent.set_hyperparams(donor.get_hyperparams())
```

### Explore Phase

After exploit, all non-elite agents perturb their hyperparameters:

```python
for agent in non_elite_agents:
    hp = agent.get_hyperparams()
    for key, value in hp.items():
        if random.random() < mutate_prob:   # mutate_prob = 0.8
            hp[key] = value * (1 + random.gauss(0, mutation_strength))
            hp[key] = clamp(hp[key], bounds[key].min, bounds[key].max)
    agent.set_hyperparams(hp)
```

Gaussian perturbation is multiplicative (scales with current value), ensuring small values remain small and large values receive proportionally large mutations.

---

## Hyperparameter Bounds

| Hyperparameter | Min | Max | Scale |
|---|---|---|---|
| `learning_rate` | 1e-5 | 1e-2 | log |
| `gamma` | 0.90 | 0.999 | linear |
| `clip_epsilon` | 0.05 | 0.40 | linear |
| `confidence_threshold` | 0.50 | 0.90 | linear |
| `lookback` | 5 | 60 | integer |
| `rsi_period` | 5 | 30 | integer |
| `bb_period` | 10 | 50 | integer |
| `bb_std` | 1.5 | 3.0 | linear |
| `atr_period` | 7 | 21 | integer |

---

## Evolution Statistics

`EvolutionScheduler.get_stats()` returns:

```python
{
    "total_evolutions": 42,
    "exploits_performed": 30,
    "explorations_performed": 36,
    "best_fitness_history": [0.12, 0.18, 0.31, ...],  # per generation
    "avg_fitness_history":  [0.05, 0.09, 0.15, ...],
}
```

These are used by the dashboard fitness chart and the evolution log panel.

---

## Elite Preservation

The top-ranked agent is never mutated and never replaced. This ensures:
- The best discovered configuration is always in the population
- The fitness floor never decreases due to random mutation
- There is always a stable high-quality signal in the ensemble

---

## Genealogy Tracking

Each exploit event creates a parent-child relationship recorded in `pbt_genealogy.json`. This allows tracing the evolutionary history of any agent back to generation 0.

---

## PBT vs. Alternatives

| Method | Speed | Quality | Human effort |
|---|---|---|---|
| Grid search | Slow | Low (combinatorial) | High |
| Random search | Medium | Medium | Low |
| Bayesian optimisation | Medium | High | Medium |
| PBT (MarketZero) | Fast (parallel) | High (adaptive) | Very low |

PBT's key advantage: hyperparameters are updated while training continues, not between separate training runs. This means the population adapts to changing market regimes in real time.
