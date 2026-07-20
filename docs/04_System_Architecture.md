# 04 — System Architecture

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        CLI (main.py)                         │
│          serve | pbt-live | train | backtest | status        │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────▼──────────────┐
        │      FastAPI Server          │
        │      (server.py)             │
        │  ┌─────────┐ ┌───────────┐  │
        │  │ REST API │ │ WebSocket │  │
        │  │ /api/... │ │   /ws     │  │
        │  └─────────┘ └───────────┘  │
        └──────────────┬──────────────┘
                       │  (broadcast thread)
        ┌──────────────▼──────────────┐
        │     PBT Live Executor        │
        │   (pbt/pbt_live_trading.py)  │
        │                              │
        │  Population[12 agents]       │
        │  ┌───────┬─────────┬──────┐  │
        │  │ 6×PPO │4×Dreamer│2×WM  │  │
        │  └───────┴─────────┴──────┘  │
        │                              │
        │  Signal Aggregator (ENSEMBLE)│
        │  Online Evaluator            │
        │  Evolution Scheduler         │
        └──┬────────────────┬──────────┘
           │                │
    ┌──────▼───┐     ┌──────▼──────┐
    │  Broker  │     │  Risk Mgr   │
    │ (Alpaca/ │     │ (circuit    │
    │  Paper)  │     │  breakers)  │
    └──────────┘     └─────────────┘
           │
    ┌──────▼──────────────────────┐
    │       Market Data            │
    │  AAPL | SPY | GLD | BTC/USD  │
    └─────────────────────────────┘
```

---

## Repository Structure

```
MarketZero/
├── main.py                          # CLI entry point
├── server.py                        # FastAPI + WebSocket server
├── dashboard.html                   # Browser dashboard (single-file)
├── pbt/
│   ├── pbt_agents.py                # PPO, Dreamer, WorldModel agent classes
│   ├── pbt_live_trading.py          # PBTLiveExecutor (core orchestrator)
│   ├── pbt_trading_config.py        # PBTLiveConfig dataclass + validation
│   ├── pbt_signal_aggregator.py     # Multi-agent signal fusion
│   ├── pbt_online_evaluator.py      # Real-time fitness + leaderboard
│   └── pbt_evolution.py             # Exploit + Explore engines
├── trading/
│   ├── live_trading.py              # Alpaca broker interface
│   ├── risk_management.py           # Risk controls & circuit breakers
│   ├── backtest_metrics.py          # Fitness metrics (Sharpe, Calmar, etc.)
│   ├── training_ppo.py              # Offline PPO pre-training
│   └── live_trading_orchestrator.py # Single-agent fallback mode
├── monitoring/
│   └── monitoring.py                # RegimeDetector, MetricsLogger, Dashboard
├── tests/
│   └── test_marketzero.py           # 30-test full system test suite
├── docs/                            # This documentation (22 files)
├── configs/                         # YAML overrides (optional)
├── logs/                            # JSON event logs + TensorBoard
└── checkpoints/                     # Population snapshots per generation
```

---

## Threading Model

The system runs two concurrent execution contexts:

| Thread | Role |
|---|---|
| `trading-engine` (daemon) | Runs `_trading_loop()` — steps agents, evaluates, evolves |
| `uvicorn` (main thread) | Serves HTTP and WebSocket connections |

State is shared via `_latest_state` dict protected by `_state_lock` (threading.Lock). The async broadcaster reads from `asyncio.Queue` (maxsize=50) and fans out to all connected WebSocket clients.

---

## Data Flow Per Step

1. Broker fetches latest price for each symbol
2. Each agent observes market state, produces `(direction, confidence, size)` signal
3. `SignalAggregator` applies ENSEMBLE weighting → single aggregated signal
4. `RiskManager` validates position size, drawdown, daily trade count
5. If approved: order submitted to broker
6. `OnlineEvaluator` updates fitness scores (Sharpe, Calmar, Sortino)
7. State dict assembled and pushed to broadcast queue
8. After `generation_steps` steps: evolution cycle runs

---

## Configuration

All parameters are controlled via `PBTLiveConfig`:

```python
PBTLiveConfig(
    population_size=12,
    agent_type_distribution={"PPO": 6, "Dreamer": 4, "WorldModel": 2},
    symbols=["AAPL", "SPY", "GLD", "BTC/USD"],
    generation_steps=500,
    total_capital=120_000,
    capital_per_agent=5_000,
    broker_mode="paper",      # or "live"
    step_sleep_ms=100,
    mutation_strength=0.20,
    log_dir="logs",
    checkpoint_dir="checkpoints",
)
```

See `17_PBT_Framework.md` for full parameter reference.
