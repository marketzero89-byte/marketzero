# 01 — Executive Summary

**MarketZero AI Quant Trading System** | Version 1.0 | January 2026

---

## Overview

MarketZero is a production-grade, fully autonomous algorithmic trading platform built on Population-Based Training (PBT). It maintains a live population of heterogeneous reinforcement learning agents — PPO, Dreamer, and WorldModel — that continuously compete, evolve, and trade across equities, forex, commodities and crypto markets simultaneously.

The system is not a single model. It is an **evolving ecosystem** of agents that self-optimise in real time, with no human intervention required between generations.

---

## Key Capabilities

- **Multi-agent PBT architecture** — 12 agents train, trade, and evolve in parallel
- **Continuous live evolution** — exploit/explore cycles run every generation (~500 steps)
- **Unified fitness function** — Sharpe, Calmar, Sortino, and drawdown composited into a single [-1, 1] score
- **Institutional-grade risk controls** — circuit breakers, position limits, daily loss caps
- **Real-time browser dashboard** — WebSocket-streamed equity curves, leaderboard, regime state
- **Paper and live broker modes** — Alpaca integration with environment-variable credentials

---

## Performance Targets

| Metric | Target |
|---|---|
| Annual Sharpe Ratio | ≥ 1.5 |
| Maximum Drawdown | ≤ 20% |
| Daily Loss Limit | ≤ 1.5% |
| Win Rate | ≥ 52% |
| Calmar Ratio | ≥ 0.8 |

---

## System At a Glance

```
Market Data → Feature Engineering → Agent Population (PPO / Dreamer / WorldModel)
                                           ↓
                               Signal Aggregation (ENSEMBLE)
                                           ↓
                               Risk Manager → Broker Execution
                                           ↓
                               Online Fitness Evaluation
                                           ↓
                               Evolution: Exploit + Explore
                                           ↓
                               Next Generation (repeat)
```

---

## Deployment

Run the full stack with a single command:

```bash
python main.py serve --port 8000 --broker-mode paper
```

Open `http://localhost:8000` to view the live dashboard.

---

## Document Map

This repository contains 22 technical documents covering every layer of the system, from investment philosophy through disaster recovery. Begin with `02_Investment_Philosophy.md` for strategic context, or jump directly to `04_System_Architecture.md` for implementation detail.
