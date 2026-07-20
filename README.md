# MarketZero

**AI Quant Trading System powered by Population-Based Training** | Version 1.0 | January 2026

MarketZero is a production-grade, fully autonomous algorithmic trading platform that maintains a live population of heterogeneous reinforcement learning agents — **PPO**, **Dreamer**, and **WorldModel** — which continuously trade, compete, and evolve across equities, forex, commodities, and crypto markets in real time.

It is not a single model. It is an **evolving ecosystem** of agents that self-optimise generation over generation, with no human intervention required between cycles.

---

## Table of Contents

- [Key Capabilities](#key-capabilities)
- [Performance Targets](#performance-targets)
- [System Architecture](#system-architecture)
- [Quick Start](#quick-start)
- [Repository Structure](#repository-structure)
- [Investment Philosophy](#investment-philosophy)
- [Research Methodology](#research-methodology)
- [Agent Types](#agent-types)
- [Population-Based Training](#population-based-training)
- [Portfolio Construction](#portfolio-construction)
- [Risk Management](#risk-management)
- [Execution Engine](#execution-engine)
- [Backtesting](#backtesting)
- [Performance Analytics](#performance-analytics)
- [Monitoring & Dashboard](#monitoring--dashboard)
- [MLOps & Deployment](#mlops--deployment)
- [Live Trading](#live-trading)
- [Model Governance](#model-governance)
- [Disaster Recovery](#disaster-recovery)
- [Research Roadmap](#research-roadmap)
- [Documentation Map](#documentation-map)

---

## Key Capabilities

- **Multi-agent PBT architecture** — 12 agents train, trade, and evolve in parallel
- **Continuous live evolution** — exploit/explore cycles run every generation (~500 steps)
- **Unified fitness function** — Sharpe, Calmar, Sortino, and drawdown composited into a single `[-1, 1]` score
- **Institutional-grade risk controls** — circuit breakers, position limits, daily loss caps
- **Real-time browser dashboard** — WebSocket-streamed equity curves, leaderboard, regime state
- **Paper and live broker modes** — Alpaca integration via environment-variable credentials

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

## System Architecture

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

The system runs two concurrent execution contexts: a `trading-engine` daemon thread (steps, evaluates, evolves agents) and the main `uvicorn` thread (serves HTTP/WebSocket). State is shared via a lock-protected dict and fanned out to clients through an `asyncio.Queue`.

---

## Quick Start

Run the dashboard and demo server with:

```bash
python main.py serve --port 8000
```

Open `http://localhost:8000` to view the dashboard.

To run the full PBT loop with paper data:

```bash
python main.py run --broker paper --port 8000 --population 12 --generations 50
```

### Installation

```bash
pip install -r requirements.txt
pip install alpaca-trade-api                        # live trading
pip install fastapi "uvicorn[standard]" websockets   # dashboard server
pip install torch                                   # optional for tensorboard/logging
```

### Core Commands

| Command | Purpose |
|---|---|
| `python main.py serve --port 8000` | Start the dashboard WebSocket server (mock/demo mode) |
| `python main.py run --broker paper --port 8000 --population 12 --generations 50` | Run the PBT engine and dashboard together with paper data |
| `python main.py pbt-live --broker alpaca --live` | Run the live PBT trading loop (requires Alpaca credentials) |
| `python main.py backtest --use-sample --n-bars 1000 --output reports/backtest.json` | Run a local backtest with the bundled sample data |
| `python main.py train --episodes 100 --lr 3e-4 --output checkpoints/pretrained_ppo.pkl` | Offline PPO pre-training |
| `python main.py validate` | Run production-readiness checks for live trading |
| `python main.py fetch-data --symbol AAPL --days 365 --offline` | Generate or download OHLCV data for backtesting |
| `python main.py run --resume-from checkpoints/gen_0007.json` | Resume the engine from a saved checkpoint |

---

## Repository Structure

```
MarketZero/
├── main.py                          # CLI entry point
├── Dockerfile                       # Container build definition
├── docker-compose.yml               # Local container orchestration
├── requirements.txt                 # Python dependencies
├── README.md                        # Project overview and usage
├── IMPLEMENTATION_PLAN.md           # Implementation roadmap
├── TASK.md                          # Current task tracker
├── MARKETZERO_COMPLIANCE_AUDIT_PROMPT.md
├── MARKETZERO_COMPLIANCE_REPORT.md
├── agents/                          # RL agent implementations
│   ├── __init__.py
│   ├── dreamer_agent.py
│   ├── ppo_agent.py
│   └── worldmodel_agent.py
├── backtest/                        # Backtesting utilities
│   ├── __init__.py
│   ├── data_loader.py
│   └── engine.py
├── brokers/                        # Broker integrations
│   ├── __init__.py
│   ├── alpaca_broker.py
│   ├── helpers.py
│   ├── paper_broker.py
│   └── replay_broker.py
├── core/                           # Core evaluation and orchestration
│   ├── __init__.py
│   ├── evaluator.py
│   ├── fitness.py
│   ├── metrics_engine.py
│   ├── pbt_engine.py
│   ├── regime.py
│   └── v2_llm_integration.py
├── dashboard/                      # FastAPI dashboard and state store
│   ├── __init__.py
│   ├── server.py
│   ├── state_store.py
│   └── static/
├── data/                           # Sample market data
│   └── AAPL_sample.csv
├── docs/                           # Technical documentation set
├── experiments/                    # Research experiments and runs
├── features/                       # Feature engineering pipeline
├── logs/                           # Runtime and validation logs
├── mlops/                          # Experiment tracking and alerts
├── monitoring/                     # Monitoring and alerting config
├── portfolio/                      # Portfolio construction logic
├── reports/                        # Generated reports
├── results/                        # Backtest and evaluation results
├── risk/                           # Risk management components
├── signals/                        # Signal generation modules
├── tests/                          # Automated test suite
└── checkpoints/                    # Saved model and population checkpoints
```

---

## Investment Philosophy

**Core thesis:** markets are partially but imperfectly efficient. Short-lived alpha exists as momentum bursts, mean-reversion windows, and volatility dislocations — no single model captures all of it. A population of diverse, competing strategies under continuous selection pressure approximates broader coverage of the opportunity surface than any individual strategy.

**Principles:**

1. **Diversity over conviction** — a heterogeneous population (PPO, Dreamer, WorldModel) produces uncorrelated signals; ensemble aggregation harvests the wisdom of the crowd.
2. **Survival of the fittest, continuously** — bottom-quartile agents are replaced every generation; stale strategies do not survive.
3. **Risk first** — position sizing, drawdown limits, and circuit breakers are hard constraints. The fitness function explicitly penalises drawdown.
4. **Regime awareness** — the system detects market regime and adapts signal weighting accordingly.
5. **No overfitting to history** — agents are evaluated on live streaming data, not replayed history, eliminating look-ahead bias.

**What MarketZero is not:** a black-box neural network (all decisions are interpretable at the hyperparameter level), a high-frequency trading system (intraday to multi-day holding periods), or a discretionary system (human override is limited to the risk kill-switch).

| Alpha Source | Mechanism | Primary Agent Type |
|---|---|---|
| Momentum | EMA crossover, trend strength | PPO |
| Mean reversion | Bollinger band extremes, RSI | WorldModel |
| Volatility premium | ATR expansion signals | Dreamer |
| Regime transitions | Regime detector + signal weight shifts | All |
| Ensemble disagreement | WorldModel uncertainty signal | WorldModel |

Capital is allocated equally across agents at initialisation (`$5,000` per agent by default). Elite agents earn more *evolutionary* influence, not more capital — preventing winner-take-all concentration.

---

## Research Methodology

```
Hypothesis → Offline Backtest → Paper Trading Validation → Live Deployment
```

No strategy reaches live capital without completing all four phases.

1. **Hypothesis formation** — a falsifiable claim about a market inefficiency, logged in `experiments/`.
2. **Offline backtesting** — `python main.py backtest --trend bull`; must clear Sharpe ≥ 0.8, MaxDrawdown ≤ 25%, Win Rate ≥ 48% to proceed.
3. **Paper trading validation** — minimum 5 trading days or 10 generations on live prices with simulated execution; graduates to live capital at fitness > 0.3 and Sharpe > 1.0.
4. **Live deployment** — identical codebase, `broker_mode = "live"`, orders routed through Alpaca.

**Avoiding research pitfalls:**

| Pitfall | Mitigation |
|---|---|
| Overfitting | Live paper validation required before live capital |
| Look-ahead bias | Online evaluation only — no historical lookahead |
| Survivorship bias | Full agent population tracked, including eliminated agents |
| Data snooping | Hypothesis must be logged before backtest is run |
| Regime mismatch | All strategies tested across BULL, BEAR, RANGING, HIGH_VOL |

---

## Agent Types

### PPO
A simplified linear stochastic policy (`action = tanh(W·obs + b + ε)`), trained with a clipped policy-gradient objective and a running-mean baseline. Best in BULL/trending regimes.

### Dreamer
Extends PPO with imagination rollouts: a learned linear transition model simulates `H` steps ahead (default `H=5`), and the agent selects the action with the highest discounted imagined value. Confidence derives from the variance of imagined returns. Best in volatile/complex regimes.

### WorldModel
An ensemble of independently-initialised linear predictors (default `N=5`). Disagreement among predictors (`std`) is converted into uncertainty; when confidence falls below a threshold, the agent abstains rather than trade. Best in RANGING regimes where false signals are costly.

| Feature | PPO | Dreamer | WorldModel |
|---|---|---|---|
| Policy type | Stochastic linear | Imagination rollout | Ensemble disagreement |
| Confidence source | Entropy of action dist. | Planning uncertainty | Predictor std deviation |
| Update method | Policy gradient | Model-based RL | Online gradient descent |
| Best regime | BULL (trend) | Volatile/complex | RANGING (uncertainty) |

**Feature engineering** is agent-local: each agent evolves its own `lookback`, `rsi_period`, `bb_period`, `bb_std`, and `atr_period`, so no two agents necessarily see the same feature view — a deliberate design choice that preserves signal diversity. All features are normalised (price relative to current price, RSI ÷ 100, Bollinger position, ATR ÷ price) and the market regime is appended as a one-hot vector.

**Emergent strategy archetypes** observed in the evolved population include the Momentum Rider (PPO, BULL), Mean Reversion Sniper (WorldModel, RANGING), Volatility Harvester (Dreamer, HIGH_VOL), Trend Follower (PPO/Dreamer), and Conservative Cash Holder (WorldModel — abstaining is treated as a positive-EV defensive stance, not a failure state).

---

## Population-Based Training

PBT combines training and hyperparameter search into one continuous loop instead of separate sequential phases:

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

- **Exploit** — bottom-quartile agents inherit a random top-quartile donor's hyperparameters.
- **Explore** — non-elite agents apply multiplicative Gaussian perturbation (`mutation_strength = 0.20` default) to each hyperparameter, clamped to its bounds.
- **Elite preservation** — the rank-1 agent is never mutated or replaced, guaranteeing the fitness floor never regresses.
- **Genealogy** — every exploit event is recorded in `logs/pbt_genealogy.json`, allowing full lineage reconstruction back to generation 0.

| Hyperparameter | Range |
|---|---|
| `learning_rate` | 1e-5 – 1e-2 |
| `gamma` | 0.90 – 0.999 |
| `clip_epsilon` | 0.05 – 0.40 |
| `confidence_threshold` | 0.50 – 0.90 |
| `lookback` | 5 – 60 |
| `rsi_period` | 5 – 30 |
| `bb_period` | 10 – 50 |
| `bb_std` | 1.5 – 3.0 |
| `atr_period` | 7 – 21 |

Default population: 12 agents (6 PPO / 4 Dreamer / 2 WorldModel), 500 steps per generation.

---

## Portfolio Construction

A **flat equal-weight model at the agent level** combined with **dynamic signal-weighted position sizing at execution**. Each agent trades its own `capital_per_agent` allocation independently; portfolio equity is the sum across agents.

Position sizing combines confidence, drawdown penalty, and ATR-based volatility scaling (targeting ~1% portfolio volatility per step). The `SignalAggregator` produces one portfolio-level signal per symbol; a trade executes only when confidence ≥ 0.55, agent agreement ≥ 55%, and `|direction| > 0.10`.

| Mode | Weight Basis | Best For |
|---|---|---|
| `EQUAL` | Uniform | Debugging / baseline |
| `FITNESS` | Rolling fitness score | Stable markets |
| `CONFIDENCE` | Per-step confidence | High-uncertainty regimes |
| `RANK` | Fitness rank (not score) | Robust to outliers |
| `ENSEMBLE` | 60% fitness + 40% confidence | **Default (production)** |

| Parameter | Default |
|---|---|
| `total_capital` | $120,000 |
| `capital_per_agent` | $5,000 |
| `max_position_size` | 5% |
| `max_portfolio_exposure` | 95% |
| `max_leverage` | 2× |

Each symbol is treated as an independent portfolio dimension in v1.0 — no cross-symbol correlation optimisation. A minimum 5% cash buffer is always retained.

---

## Risk Management

Risk management is a hard-gating layer between signal generation and order execution — no trade bypasses it, and it can halt trading system-wide.

| Control | Default | Action |
|---|---|---|
| Max position size | 5% | Reduce to limit |
| Max portfolio exposure | 95% | Reject order |
| Max leverage | 2× | Reject order |
| Stop loss per trade | -2% | Force close |
| Take profit per trade | +3% | Force close |
| Daily loss circuit breaker | -1.5% | **HALT all trading** |
| Max drawdown emergency | -20% | **HALT all trading** |
| Max trades/day | 50 | Reject order |

When a circuit breaker fires, `trading_halted = True`, all subsequent orders are rejected, the dashboard shows a red **TRADING HALTED** banner, and resuming requires an explicit `{"action": "resume_trading"}` WebSocket command or a system restart. No circuit breaker can be disabled without a code change.

---

## Execution Engine

```
SignalAggregator → aggregate signal → RiskManager.validate_order()
        → Broker.submit_order() → Fill → PortfolioState.update()
        → Broadcast to dashboard via WebSocket
```

All orders in v1.0 are **market orders** (limit/stop-limit planned for v1.1). The paper broker fills immediately with no slippage or commission; the Alpaca broker submits real market orders via the REST API, reading credentials from `ALPACA_API_KEY` / `ALPACA_SECRET_KEY`. Typical execution latency: <1ms paper (internal), 50–300ms Alpaca REST. The trading loop is throttled by `step_sleep_ms` (default 100ms).

---

## Backtesting

The lightweight synthetic backtester (`trading/backtest_metrics.py`) is built for fast hypothesis screening, not production-grade simulation — no slippage, no market impact, no order-level detail.

```bash
python main.py backtest --trend bull
python main.py backtest --trend bear
python main.py backtest --trend ranging
```

**Metrics:** Sharpe, Calmar, Sortino, Maximum Drawdown, Win Rate, Profit Factor.

**Fitness function** (composited into a single `[-1, 1]` score):

```
Fitness = 0.50 × tanh(Sharpe / 3)
        + 0.20 × tanh(Calmar / 5)
        + 0.15 × tanh(Sortino / 3)
        + 0.10 × tanh(AnnReturn × 5)
        - 0.05 × |MaxDrawdown|
```

Known limitations (no slippage, no market impact, synthetic prices only, no transaction costs) are mitigated by paper-trading on live data before any live deployment, and are tracked for resolution in v1.1.

---

## Performance Analytics

Exposed via three channels: the WebSocket dashboard, JSON logs (`logs/pbt_orchestrator.json`), and TensorBoard.

- **Population-level**: `avg_fitness`, `best_fitness`, `worst_fitness`, `fitness_std` (diversity indicator), `n_agents`.
- **Agent-level**: fitness, Sharpe, total return %, max drawdown, win rate, trade count.
- **Portfolio-level**: equity, cash, positions, total trades, daily P&L, total return %.

A healthy system shows `best_fitness` trending upward, `avg_fitness` tracking with a lag, and occasional exploration dips followed by recovery. If `avg_fitness` flatlines while `best_fitness` keeps improving, the population is converging — increase `mutation_strength` or `population_size`.

Reports are generated monthly/quarterly/annually under `reports/`, and benchmarked against SPY buy-and-hold, a 60/40 portfolio, a pure momentum factor, and a random-agent lower bound.

---

## Monitoring & Dashboard

| Layer | Tool | Location |
|---|---|---|
| Browser dashboard | WebSocket + Chart.js | `http://localhost:8000` |
| Console dashboard | ANSI terminal output | `pbt-live` mode only |
| JSON logs | JSON-lines file | `logs/pbt_orchestrator.json` |
| TensorBoard | Scalar event files | `logs/tensorboard/` |
| Genealogy log | JSON array | `logs/pbt_genealogy.json` |

The single-file `dashboard.html` auto-connects via WebSocket with exponential backoff reconnection, and includes panels for the leaderboard, live prices, risk controls, trade feed, equity curve, fitness evolution, and the evolution log. The `RegimeDetector` classifies market state (BULL / BEAR / RANGING / HIGH_VOL) from EMA crossover and annualised volatility on every step.

> v1.0 has no automated alerting (email/Slack/PagerDuty) — the dashboard halt banner is the primary alert mechanism. Planned for v1.1.

---

## MLOps & Deployment

```
Offline Pre-training (optional) → Population Initialisation → Live Trading + Online Evaluation
        ↺ Evolution Cycle (Exploit + Explore) → Checkpoint Save ↺
```

No manual retraining is required — the PBT loop handles continuous improvement. Checkpoints (`population.pkl`, `config.json`, `metrics.json`) are saved at the end of every generation; only the last 10 are retained automatically.

```bash
python tests/test_marketzero.py    # Expected: 30 tests, 0 failures
```

**Deployment checklist:**

- [ ] All 30 tests pass
- [ ] Paper trading validated for ≥ 5 trading days
- [ ] Paper fitness ≥ 0.3, Sharpe ≥ 1.0
- [ ] `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` set
- [ ] `broker_mode = "live"` confirmed
- [ ] Daily loss limit and max drawdown reviewed
- [ ] Dashboard accessible and showing live data
- [ ] Checkpoint directory writable

---

## Live Trading

```bash
export ALPACA_API_KEY=PKXXXXXXXXXXXXXXXX
export ALPACA_SECRET_KEY=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

python main.py serve \
  --broker-mode live \
  --population-size 12 \
  --symbols AAPL SPY GLD \
  --total-capital 120000 \
  --capital-per-agent 5000 \
  --generation-steps 500 \
  --port 8000
```

Alpaca live trading operates during US market hours (9:30 AM–4:00 PM ET, Mon–Fri); BTC/USD trades 24/7. There is no automatic market-hours detection yet — after-hours order rejections are logged but don't trigger circuit breakers.

**Recommended tightened limits for initial live deployment:**

```python
executor.risk_manager.max_position_size = 0.03   # 3% vs 5% default
executor.risk_manager.daily_loss_limit  = 0.01    # 1% vs 1.5% default
executor.risk_manager.max_drawdown      = 0.10    # 10% vs 20% default
executor.risk_manager.max_daily_trades  = 20      # 20 vs 50 default
```

**Gradual capital deployment schedule:**

| Week | Capital | Population | Notes |
|---|---|---|---|
| 1 | $10,000 | 4 agents | Observe, validate fills |
| 2–3 | $30,000 | 8 agents | Monitor drawdowns |
| 4+ | Full capital | 12 agents | Normal operations |

---

## Model Governance

No configuration may be deployed to live capital without passing all 30 unit tests, ≥5 days of paper validation (fitness ≥ 0.3, Sharpe ≥ 1.0), a documented risk parameter review, and a signed-off deployment checklist.

Every trade and evolution event is recorded in an append-only audit log (trade details, fitness at time of trade, regime, risk approval; hyperparameters before/after exploit events). Risk manager hard limits cannot be overridden at runtime — changes require a code modification, documented rationale, a full test run, and a restart.

MarketZero is designed for explainability at every layer: no black-box neural networks are used in v1.0 — all agent policies are linear models, and every signal, fitness score, and evolution event is decomposable and human-readable. The system collects no personal data; all state and logs remain local.

**Incident response:** halt immediately (Ctrl+C) → cancel open orders within 5 minutes → preserve logs within 1 hour → document root-cause analysis within 24 hours → resume only after all tests pass and the root cause is addressed.

---

## Disaster Recovery

Full procedures in `docs/21_Disaster_Recovery.md`. Quick reference:

### Failure Scenarios

| Scenario | Symptom | First Action |
|---|---|---|
| Process crash | Dashboard shows reconnecting overlay | `python main.py serve --resume-from checkpoints/generation_NNNN` |
| WebSocket disconnect | Grey dot in dashboard | Wait — auto-reconnects with exponential backoff; reload tab after 30s if needed |
| Circuit breaker halt | Red TRADING HALTED banner | Identify `halt_reason`; send `{"action": "resume_trading"}` only after root cause is understood |
| Alpaca API failure | `AlpacaAPIError` in logs | Check `status.alpaca.markets`; loop retries automatically (1s delay) |
| Checkpoint corruption | `pickle.UnpicklingError` on resume | Try previous checkpoint; start fresh if all are corrupt |
| Log disk full | `IOError: No space left on device` | Archive and remove `logs/pbt_orchestrator.json`; reduce `max_checkpoints` |
| Corrupted paper portfolio | Implausible equity values | Halt immediately; do not resume from checkpoint; start a fresh session |

> **Do not resume a circuit breaker blindly.** Understand why it fired before sending the resume command.

### Recovery Time Objectives

| Failure Type | RTO (paper mode) | RTO (live mode) |
|---|---|---|
| Process crash | < 2 minutes | < 5 minutes |
| Network disconnect | < 30 seconds | < 30 seconds |
| Circuit breaker | Minutes (human review) | Minutes (human review) |
| Alpaca outage | N/A (wait) | < 1 hour (after Alpaca restores) |
| Full system rebuild | < 30 minutes | < 2 hours |

### Backup Schedule

| Asset | Frequency | Destination |
|---|---|---|
| Checkpoints | Per generation (automatic) | `checkpoints/` |
| Checkpoints (offsite) | Daily | External drive / S3 |
| Logs | Weekly archive | `logs_archive/` |
| Source code | Every commit | Git remote |
| Config snapshots | Per deployment | `checkpoints/generation_N/config.json` |

### Process Supervisor (Recommended)

Run under `supervisord` for automatic restart on crash:

```ini
# /etc/supervisor/conf.d/marketzero.conf
[program:marketzero]
command=python /path/to/main.py serve --broker-mode paper
autorestart=true
startsecs=5
```

---

## Research Roadmap

Full detail in `docs/22_Research_Roadmap.md`. Summary by version:

### v1.0 — Completed (Q1–Q2 2026)

**Q1 2026 — Foundation:** core PBT architecture; PPO, Dreamer, and WorldModel agent families; paper broker; composite fitness function; signal aggregation modes; online evaluator; risk manager; regime detector; CLI; 30-test suite.

**Q2 2026 — Production Hardening:** FastAPI WebSocket server; browser dashboard (Chart.js); Alpaca live broker integration; checkpoint system; MetricsLogger with TensorBoard; genealogy tracking; offline PPO pre-training; 22-document research platform.

### v1.1 — Near-Term (Q3 2026)

**Execution:** limit order support, slippage model, transaction cost accounting, market-hours detection.

**Backtesting:** walk-forward analysis, Monte Carlo simulation (1,000-run bootstrap), historical OHLCV replay broker.

**Monitoring:** email alerts, Slack webhooks, Prometheus `/metrics` endpoint, mobile-responsive dashboard.

**MLOps:** pinned `requirements.txt`, Docker container, GitHub Actions CI.

### v1.2 — Medium-Term (Q4 2026)

**Agents:** MLP neural network policies (2 hidden layers, 32 units), Bayesian uncertainty via MC dropout, attention-based signal aggregation.

**Features:** alternative data (news NLP, options flow, short interest), cross-asset inputs (VIX, yield curve, dollar index), macro calendar awareness.

**Portfolio:** mean-variance optimisation, Kelly criterion sizing, dynamic capital allocation to higher-fitness agents.

**Infrastructure:** distributed PBT across CPU cores, GPU acceleration, MLflow / W&B experiment tracking.

### v2.0 — Long-Term (2027)

Multi-broker support (Interactive Brokers, Coinbase, Binance); options and futures trading; crypto perpetuals; LLM integration for news sentiment; regime-conditioned policies; meta-learning; cloud deployment on Kubernetes with multi-region redundancy.

### Open Research Questions

| Question | Priority |
|---|---|
| Does PBT outperform Bayesian optimisation on live market data? | High |
| What is the optimal population size vs. capital tradeoff? | High |
| Do WorldModel agents outperform PPO in RANGING regimes consistently? | Medium |
| Does imagination horizon (Dreamer) correlate with holding period quality? | Medium |
| Can regime detection be learned rather than rule-based? | Medium |
| Does larger lookback reduce overfitting to recent price patterns? | Low |

To propose a new research direction, create `experiments/YYYYMMDD_research_title/hypothesis.md` with a falsifiable claim, proposed test, and success criteria. All research must follow `docs/03_Research_Methodology.md`.

---

## Documentation Map

This repository's `docs/` directory contains the full technical documentation set, from investment philosophy through governance:

| # | Document |
|---|---|
| 01 | Executive Summary |
| 02 | Investment Philosophy |
| 03 | Research Methodology |
| 04 | System Architecture |
| 05 | Data Infrastructure |
| 06 | Feature Engineering |
| 07 | Market Simulation |
| 08 | World Model |
| 09 | Reinforcement Learning |
| 10 | Strategy Library |
| 11 | Portfolio Construction |
| 12 | Risk Management |
| 13 | Execution Engine |
| 14 | Backtesting Framework |
| 15 | Performance Analytics |
| 16 | MLOps |
| 17 | PBT Framework |
| 18 | Live Trading |
| 19 | Monitoring |
| 20 | Model Governance |
| 21 | Disaster Recovery |
| 22 | Research Roadmap |

Begin with `02_Investment_Philosophy.md` for strategic context, or jump directly to `04_System_Architecture.md` for implementation detail.