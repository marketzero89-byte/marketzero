# 22 — Research Roadmap

---

## Version 1.0 — Completed (Q1 2026)

### Q1 2026 — Foundation

- **Core PBT architecture** — population initialisation, generation loop, exploit/explore cycle
- **Three agent families** — PPO (linear policy gradient), Dreamer (imagination rollout), WorldModel (ensemble disagreement)
- **Paper broker** — GBM price simulator, portfolio state tracking, order execution simulation
- **Composite fitness function** — Sharpe, Calmar, Sortino, annual return, drawdown penalty composited to [-1, 1]
- **Signal aggregation** — EQUAL, FITNESS, CONFIDENCE, RANK, and ENSEMBLE (60/40) modes
- **Online evaluator** — rolling fitness, leaderboard, population metrics per generation
- **Risk manager** — position limits, daily loss circuit breaker, max drawdown halt, stop-loss/take-profit
- **RegimeDetector** — EMA crossover + volatility classification (BULL, BEAR, RANGING, HIGH_VOL, UNKNOWN)
- **CLI entry point** — `serve`, `pbt-live`, `train`, `backtest`, `status` commands
- **30-test suite** — full system coverage across all modules

---

## Version 1.0 — Completed (Q2 2026)

### Q2 2026 — Production Hardening

- **FastAPI WebSocket server** — real-time state broadcasting to browser dashboard
- **Browser dashboard** — Chart.js equity curve, fitness evolution chart, leaderboard, trade feed, regime pill
- **Alpaca live broker integration** — REST API order submission, position tracking, portfolio state sync
- **Checkpoint system** — per-generation population snapshots, resume-from-checkpoint support
- **MetricsLogger** — JSON-lines logging, optional TensorBoard integration
- **Genealogy tracking** — parent-child lineage per exploit event
- **Offline PPO pre-training** — `PPOTrainer` with synthetic episode generation
- **Path resolution fix** — `Path(__file__).resolve().parent` for reliable dashboard serving
- **Global exception handler** — readable 500 error pages in development
- **Documentation** — 22-document research platform, whitepaper, report templates

---

## Version 1.1 — Near-Term (Q3 2026)

### Execution Improvements
- **Limit order support** — replace market orders with limit orders at bid/ask midpoint
- **Slippage model** — add realistic slippage simulation to backtester (0.02%–0.10% per side)
- **Transaction cost accounting** — deduct commission estimates from paper P&L
- **Market hours detection** — automatic suspension of equity trading outside 9:30–4:00 ET

### Backtesting
- **Walk-forward analysis** — automated k-fold walk-forward with out-of-sample reporting
- **Monte Carlo simulation** — 1,000-run bootstrap for confidence intervals on all metrics
- **Historical replay broker** — replace GBM paper broker with OHLCV file replay

### Monitoring & Alerting
- **Email alerts** — trading halt and daily loss limit notifications
- **Slack webhook** — generation summary and circuit breaker events
- **Prometheus metrics endpoint** — `/metrics` for Grafana integration
- **Mobile-responsive dashboard** — optimise for tablet and phone viewports

### MLOps
- **`requirements.txt` with pinned versions** — reproducible environment
- **Docker container** — single-container deployment with all dependencies
- **GitHub Actions CI** — automated test run on every commit

---

## Version 1.2 — Medium-Term (Q4 2026)

### Agent Architecture
- **Neural network policies** — replace linear policies with small MLP (2 hidden layers, 32 units)
- **Heterogeneous WorldModel ensemble** — mix linear, polynomial, and MLP predictors
- **Bayesian uncertainty estimation** — MC dropout for confidence calibration
- **Attention-based signal aggregation** — replace fixed ENSEMBLE weights with learned attention

### Feature Engineering
- **Alternative data** — sentiment scores (news NLP), options flow, short interest
- **Cross-asset features** — VIX, yield curve slope, dollar index as regime inputs
- **Microstructure features** — bid-ask spread, order book imbalance (requires Level 2 data)
- **Macro calendar** — FOMC, CPI, earnings date awareness

### Portfolio Construction
- **Mean-variance optimisation** — cross-symbol correlation-aware position sizing
- **Kelly criterion sizing** — optimal f position sizing per agent
- **Dynamic capital allocation** — allocate more capital to higher-fitness agents

### Research Infrastructure
- **Distributed PBT** — multi-process population across CPU cores
- **GPU acceleration** — PyTorch-based neural network agents
- **Experiment tracking** — MLflow or Weights & Biases integration

---

## Version 2.0 — Long-Term (2027)

### Architecture
- **Multi-broker support** — Interactive Brokers, Coinbase, Binance
- **Options trading** — delta-hedged option strategies as agent action space
- **Futures trading** — ES, NQ, CL futures via CME
- **Crypto perpetuals** — funding rate arbitrage strategies

### Intelligence
- **Large language model integration** — news sentiment, earnings call analysis
- **Regime-conditioned policies** — separate policy networks per market regime
- **Meta-learning** — agents that learn how to learn faster
- **Adversarial training** — red-team agents that stress-test the population

### Infrastructure
- **Cloud deployment** — AWS/GCP Kubernetes deployment with auto-scaling
- **Multi-region redundancy** — active-active failover across two cloud regions
- **Real-time risk dashboard** — institutional-grade risk reporting
- **Regulatory reporting** — automated trade reporting for applicable jurisdictions

---

## Open Research Questions

| Question | Status | Priority |
|---|---|---|
| Does PBT outperform Bayesian optimisation on live market data? | Untested | High |
| What is the optimal population size vs. capital tradeoff? | Partially explored | High |
| Do WorldModel agents outperform PPO in RANGING regimes consistently? | Anecdotal evidence | Medium |
| Does imagination horizon (Dreamer) correlate with holding period quality? | Untested | Medium |
| Can regime detection be learned rather than rule-based? | Untested | Medium |
| Does larger lookback reduce overfitting to recent price patterns? | Untested | Low |

---

## Contributing Research

To propose a research direction:

1. Create a directory in `experiments/YYYYMMDD_research_title/`
2. Write a `hypothesis.md` with the falsifiable claim, proposed test, and success criteria
3. Run the experiment and record results in `results.json`
4. If successful, open a pull request to add to the strategy library or system architecture

All research must follow the methodology in `03_Research_Methodology.md`.
