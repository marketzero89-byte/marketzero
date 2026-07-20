# 16 — MLOps

---

## Overview

MLOps in MarketZero covers the lifecycle of agent models from training through production deployment, monitoring, and retirement. The system is designed for continuous operation with automated evolution replacing the traditional retrain-redeploy cycle.

---

## Model Lifecycle

```
Offline Pre-training (optional)
        │
        ▼
Population Initialisation
        │
        ▼
Live Trading + Online Evaluation  ←─────────────┐
        │                                         │
        ▼                                         │
Evolution Cycle (Exploit + Explore)               │
        │                                         │
        ▼                                         │
Checkpoint Save                                   │
        │                                         │
        └──────────────────────────────────────────┘
              (loop indefinitely)
```

No manual retraining is required. The PBT loop handles continuous model improvement.

---

## Checkpointing

Checkpoints are saved at the end of each generation:

```
checkpoints/
└── generation_0042/
    ├── population.pkl    # All 12 agents with weights + hyperparameters
    ├── config.json       # PBTLiveConfig snapshot
    └── metrics.json      # Fitness, Sharpe, drawdown at checkpoint time
```

Only the last 10 checkpoints are retained (configurable). Older checkpoints are deleted automatically.

### Saving a Checkpoint Manually

```python
executor.save_generation()
```

Or via keyboard interrupt in `pbt-live` mode — the system saves before exiting.

### Resuming from Checkpoint

```bash
python main.py serve --resume-from checkpoints/generation_0042
```

The population is restored exactly as it was at checkpoint time, including all weights and hyperparameters.

---

## Offline Pre-training Pipeline

```bash
python main.py train --n-agents 6 --episodes 100 --symbols AAPL SPY
```

Uses `PPOTrainer` in `trading/training_ppo.py`. Produces pre-trained agent weights that can be loaded into the live population to accelerate initial fitness.

---

## Dependency Management

```bash
# Minimum required
pip install numpy

# For live trading
pip install alpaca-trade-api

# For dashboard server
pip install fastapi "uvicorn[standard]" websockets

# For TensorBoard logging
pip install torch  # or tensorflow
```

No pinned versions are required in v1.0 (planned for v1.1 with `requirements.txt` lock file).

---

## Environment Configuration

| Variable | Required | Description |
|---|---|---|
| `ALPACA_API_KEY` | Live mode only | Alpaca broker API key |
| `ALPACA_SECRET_KEY` | Live mode only | Alpaca broker secret |
| `LOG_LEVEL` | Optional | `DEBUG` / `INFO` / `WARNING` |
| `MARKETZERO_ENV` | Optional | `paper` / `live` (overrides CLI flag) |

---

## CI/CD — Test Suite

Run the full test suite before any deployment:

```bash
python tests/test_marketzero.py
# Expected: 30 tests, 0 failures
```

Tests cover: agent initialisation, signal generation, risk manager, evolution cycle, portfolio state, broker interface, fitness computation, and regime detection.

All tests are self-contained and require no external dependencies (paper mode only).

---

## Deployment Checklist

Before going live:

- [ ] All 30 tests pass
- [ ] Paper trading validated for ≥ 5 trading days
- [ ] Paper fitness ≥ 0.3, Sharpe ≥ 1.0
- [ ] `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` set
- [ ] `broker_mode = "live"` confirmed
- [ ] Daily loss limit reviewed (default -1.5%)
- [ ] Max drawdown reviewed (default -20%)
- [ ] Dashboard accessible and showing live data
- [ ] Checkpoint directory writable

---

## Model Versioning

Agent versions are implicitly tracked by generation number. The `pbt_genealogy.json` log records lineage:

```json
[
  {"generation": 5, "agent_id": "PPO_a3f2", "parent_id": "PPO_9b11", "event": "exploit"},
  {"generation": 5, "agent_id": "Dreamer_7c4a", "parent_id": null, "event": "explore"}
]
```

This allows reconstruction of any agent's lineage at any generation.
