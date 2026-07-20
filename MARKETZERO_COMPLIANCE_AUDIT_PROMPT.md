# MarketZero — Full Documentation Compliance Audit Prompt

> **How to use:** Paste this entire prompt into a new Claude session. Then either share
> your project directory (zip upload, file-by-file, or directory listing + key files) or
> give Claude direct access to the codebase. It will execute all five steps and produce a
> structured compliance report against every requirement extracted from all 22 documents.

---

## Context

You are a senior software auditor performing a full compliance review of the **MarketZero
AI Quant Trading System** (v1.0). The system is a Population-Based Training (PBT)
algorithmic trading platform with 12 heterogeneous RL agents (PPO, Dreamer, WorldModel),
a FastAPI/WebSocket server, an Alpaca broker integration, and a browser dashboard.

The project is governed by 22 technical documents (`01_Executive_Summary.md` through
`22_Research_Roadmap.md`). Your job is to cross-reference every source file against every
requirement, contract, and behavioural rule stated across those 22 documents and produce a
structured compliance report with a prioritised fix list.

---

## Step 1 — Index the Project

Scan the full directory tree. Produce a structured listing of:

**Documentation files** (already provided — all 22 `.md` docs)

**Source files to audit** — map each to its documented owner:

| File | Documented In |
|---|---|
| `main.py` | Doc 04 (CLI: serve, pbt-live, train, backtest, status) |
| `server.py` | Doc 04 (FastAPI + WebSocket), Doc 19 (dashboard serving) |
| `dashboard.html` | Doc 04, Doc 19 (panels, WebSocket, reconnect, Chart.js) |
| `pbt/pbt_agents.py` | Docs 08, 09, 06 (PPO, Dreamer, WorldModel, observation vector) |
| `pbt/pbt_live_trading.py` | Doc 04 (PBTLiveExecutor), Doc 17 (generation loop) |
| `pbt/pbt_trading_config.py` | Doc 04 (PBTLiveConfig dataclass + all defaults) |
| `pbt/pbt_signal_aggregator.py` | Docs 10, 11 (ENSEMBLE, FITNESS, RANK, etc.) |
| `pbt/pbt_online_evaluator.py` | Docs 14, 15 (fitness function, leaderboard) |
| `pbt/pbt_evolution.py` | Doc 17 (exploit/explore, elite preservation, genealogy) |
| `trading/live_trading.py` | Docs 05, 13, 18 (AlpacaBroker, PaperBroker interfaces) |
| `trading/risk_management.py` | Doc 12 (all 8 risk controls, circuit breaker logic) |
| `trading/backtest_metrics.py` | Docs 03, 14 (6 metrics, fitness formula, BacktestEvaluator) |
| `trading/training_ppo.py` | Doc 09 (PPOTrainer, offline pre-training) |
| `trading/live_trading_orchestrator.py` | Doc 13 (single-agent fallback mode) |
| `monitoring/monitoring.py` | Doc 19 (RegimeDetector, MetricsLogger, PBTLiveDashboard) |
| `tests/test_marketzero.py` | Doc 16 (30-test suite, all modules covered) |

Also note any **extra files** present in the repo that have no documentation counterpart.

---

## Step 2 — Extract All Documented Contracts

From the 22 documents, the following requirements are pre-extracted and numbered. Treat
every item below as a **binding requirement** (R-NNN). Add any additional requirements you
find during your own reading of the docs.

---

### ARCHITECTURE & CONFIGURATION (R-001 – R-030)

**R-001** Population default: 12 agents total — 6 PPO, 4 Dreamer, 2 WorldModel.
*(Doc 04, Doc 17)*

**R-002** `PBTLiveConfig` must define exactly these fields with these defaults:
`population_size=12`, `generation_steps=500`, `total_capital=120_000`,
`capital_per_agent=5_000`, `broker_mode="paper"`, `step_sleep_ms=100`,
`mutation_strength=0.20`, `log_dir="logs"`, `checkpoint_dir="checkpoints"`.
*(Doc 04)*

**R-003** CLI must expose exactly five commands: `serve`, `pbt-live`, `train`, `backtest`,
`status`. *(Doc 04)*

**R-004** `serve` command default invocation: `python main.py serve --port 8000
--broker-mode paper`. *(Doc 01)*

**R-005** Repository structure must include these top-level directories and files:
`main.py`, `server.py`, `dashboard.html`, `pbt/`, `trading/`, `monitoring/`, `tests/`,
`docs/`, `configs/`, `logs/`, `checkpoints/`. *(Doc 04)*

**R-006** Threading model: exactly two concurrent execution contexts — `trading-engine`
daemon thread running `_trading_loop()`, and `uvicorn` on the main thread. *(Doc 04)*

**R-007** Shared state must use `_latest_state` dict protected by `threading.Lock`
(`_state_lock`). Broadcaster reads from `asyncio.Queue(maxsize=50)`. *(Doc 04)*

**R-008** Data flow per step must follow this exact sequence: (1) broker fetches price,
(2) agents produce signals, (3) SignalAggregator applies ENSEMBLE, (4) RiskManager
validates, (5) order submitted if approved, (6) OnlineEvaluator updates fitness, (7) state
pushed to broadcast queue, (8) after `generation_steps` steps: evolution runs. *(Doc 04)*

**R-009** Dashboard must be served from `dashboard.html` as a single file with no build
step. Path resolution must use `Path(__file__).resolve().parent`. *(Doc 04, Doc 22)*

**R-010** WebSocket endpoint must be at `/ws`. REST endpoints under `/api/...`. *(Doc 04)*

---

### DATA INFRASTRUCTURE (R-011 – R-025)

**R-011** Default symbols: `["AAPL", "SPY", "GLD", "BTC/USD"]`. *(Docs 04, 05)*

**R-012** In paper mode, prices generated by GBM with per-symbol defaults:
AAPL μ=+0.0003 σ=0.015 S₀=185.00; SPY μ=+0.0002 σ=0.010 S₀=520.00;
GLD μ=+0.0001 σ=0.008 S₀=185.00; BTC/USD μ=+0.0005 σ=0.035 S₀=67,000.00. *(Doc 07)*

**R-013** `RegimeDetector` price deque maxlen must be 100. *(Doc 05)*

**R-014** Observation vector must contain (in order): normalised price history
(length=`lookback`), RSI/100, Bollinger position, ATR/price, regime one-hot (length 5).
Total vector length must be `lookback + 8` (varying per agent). *(Doc 06)*

**R-015** Regime one-hot encoding: UNKNOWN=[1,0,0,0,0], BULL=[0,1,0,0,0],
BEAR=[0,0,1,0,0], RANGING=[0,0,0,1,0], HIGH_VOL=[0,0,0,0,1]. *(Doc 06)*

**R-016** `pbt_orchestrator.json` log record schema must include exactly:
`step`, `timestamp`, `generation`, `avg_fitness`, `best_fitness`, `portfolio_equity`.
*(Doc 05)*

**R-017** Checkpoint directory structure: `checkpoints/generation_NNNN/` containing
`population.pkl`, `config.json`, `metrics.json`. *(Docs 05, 16)*

**R-018** Only last 10 checkpoints retained automatically; older ones deleted. *(Docs 05, 16)*

**R-019** Genealogy log at `logs/pbt_genealogy.json` must record per-exploit events with
fields: `generation`, `agent_id`, `parent_id`, `event`. *(Docs 05, 16, 17)*

**R-020** Dataset files must follow naming convention `{SYMBOL}_{YYYY-MM-DD}_{frequency}.csv`.
*(Doc 05)*

**R-021** Broker interface contract: any broker class must implement exactly:
`get_latest_price(symbol: str) -> float`,
`get_portfolio_state() -> PortfolioState`,
`submit_order(symbol, qty, side, confidence) -> dict`,
`get_positions() -> dict`. *(Doc 05)*

**R-022** `PortfolioState` dataclass must contain: `equity`, `cash`, `positions`,
`total_trades`, `daily_pnl`, `total_return_pct`. *(Doc 07)*

**R-023** Backtest command modes: `--trend bull` (μ=+0.0005, σ=0.012),
`--trend bear` (μ=−0.0003, σ=0.012), `--trend ranging` (μ=0.0000, σ=0.012),
each running 252 daily steps. *(Doc 07)*

**R-024** Log data retention: JSON logs 30 days rolling, checkpoints last 10 generations,
TensorBoard events 7 days rolling, backtest CSVs permanent. *(Doc 05)*

**R-025** Results files in `results/` must never be overwritten — each run appends a new
timestamped file. *(Doc 03)*

---

### FEATURE ENGINEERING (R-026 – R-040)

**R-026** RSI formula: RS = avg_gain/avg_loss over `rsi_period` bars; RSI = 100−(100/(1+RS)).
`rsi_period` evolved in [5, 30]. *(Doc 06)*

**R-027** Bollinger Bands: Middle = SMA(close, `bb_period`); Upper/Lower = Middle ±
`bb_std` × σ. `bb_period` ∈ [10, 50], `bb_std` ∈ [1.5, 3.0]. *(Doc 06)*

**R-028** ATR: TR = max(high−low, |high−prev_close|, |low−prev_close|); ATR = EMA(TR,
`atr_period`). `atr_period` ∈ [7, 21]. *(Doc 06)*

**R-029** EMA crossover for RegimeDetector: EMA_fast=9, EMA_slow=21;
`trend = (EMA_fast − EMA_slow) / EMA_slow`. Used for BULL/BEAR only; not exposed to
agents directly. *(Doc 06)*

**R-030** Regime classification thresholds: `annualised_vol > 0.35` → HIGH_VOL;
`trend > 0.005 and mean_return > 0` → BULL; `trend < −0.005 and mean_return < 0` → BEAR;
else → RANGING. *(Doc 19)*

**R-031** Price features normalised by dividing by current price (scale-invariant).
RSI ÷ 100 → [0,1]. Bollinger position: `(price − lower) / (upper − lower)`.
ATR ÷ current_price. *(Doc 06)*

**R-032** `lookback` hyperparameter range: [5, 60]. Observation size therefore varies
from 12 to 67. *(Doc 06)*

**R-033** New evolved hyperparameter must be registered in `PBTLiveConfig
.hyperparameter_bounds`, initialised randomly in `Agent.__init__`, and included in
`get_hyperparams()` / `set_hyperparams()`. *(Doc 06)*

---

### AGENT IMPLEMENTATIONS (R-034 – R-060)

**R-034** PPO linear policy: `logits = W × observation + b`;
`action = tanh(logits + ε)`, ε ~ N(0, σ). *(Doc 09)*

**R-035** PPO advantage update: `advantage = reward − baseline`;
`grad = advantage × (action / confidence) × observation`;
`policy_weights += learning_rate × grad`. Baseline is a running mean. *(Doc 09)*

**R-036** PPO reward shaping coefficients: α=0.3 (Sharpe), β=0.5 (drawdown penalty),
γ=0.1 (trade cost proxy). *(Doc 09)*

**R-037** Dreamer imagination rollout: H steps ahead (default H=5);
`V = Σ γ^h × r_{t+h}`; select action with highest imagined value. *(Doc 09)*

**R-038** Dreamer transition model update: `loss_T = ||s_next_pred − s_actual||²`;
`T_weights -= lr_T × ∇loss_T`. Online update every step. *(Doc 09)*

**R-039** Dreamer confidence: `exp(−var(imagined_returns) / temperature)`. *(Doc 09)*

**R-040** WorldModel ensemble: default N=5 predictors, each a linear model with
independently initialised weights. *(Doc 08)*

**R-041** WorldModel uncertainty:
`normalised_uncertainty = min(std_pred / uncertainty_scale, 1.0)`;
`confidence = 1.0 − normalised_uncertainty`.
If `confidence < confidence_threshold` → emit zero direction signal (no trade). *(Doc 08)*

**R-042** WorldModel online learning: MSE gradient descent per step:
`grad = 2 × (prediction − actual_return) × observation`;
`weights -= learning_rate × grad`. *(Doc 08)*

**R-043** WorldModel signal output: `direction = np.tanh(mean_pred)`;
`position_size = confidence × max_size`. *(Doc 08)*

**R-044** WorldModel `n_ensemble` hyperparameter range: [3, 10]. *(Doc 08)*

**R-045** `confidence_threshold` hyperparameter range for all agent types: [0.50, 0.90].
*(Docs 08, 09)*

**R-046** Full hyperparameter bounds (all agent types):

| Hyperparameter | Min | Max |
|---|---|---|
| `learning_rate` | 1e-5 | 1e-2 |
| `gamma` | 0.90 | 0.999 |
| `clip_epsilon` | 0.05 | 0.40 |
| `confidence_threshold` | 0.50 | 0.90 |
| `lookback` | 5 | 60 |
| `rsi_period` | 5 | 30 |
| `bb_period` | 10 | 50 |
| `bb_std` | 1.5 | 3.0 |
| `atr_period` | 7 | 21 |

*(Docs 09, 17)*

**R-047** Agent classes must expose `compute_signal()`, `get_hyperparams()`,
`set_hyperparams()`, and inherit from `BaseAgent`. *(Docs 06, 10)*

---

### SIGNAL AGGREGATION (R-048 – R-056)

**R-048** Five aggregation modes must be implemented: `EQUAL`, `FITNESS`, `CONFIDENCE`,
`RANK`, `ENSEMBLE`. Default production mode: `ENSEMBLE`. *(Doc 11)*

**R-049** ENSEMBLE weight formula: `w_i = 0.6 × fitness_weight_i + 0.4 × confidence_weight_i`.
*(Doc 11)*

**R-050** Aggregate direction formula:
`Σ w_i × direction_i × confidence_i / Σ w_i × confidence_i`. *(Doc 10)*

**R-051** Three conditions must ALL be true for a trade to execute:
(1) `aggregate.confidence >= 0.55`,
(2) `aggregate.agreement >= 0.55` (>55% agents agree on direction),
(3) `abs(aggregate.direction) > 0.10`. *(Doc 11)*

**R-052** Total portfolio exposure check enforced per step:
`Σ |position_i| / equity ≤ 0.95`. *(Doc 11)*

**R-053** Each symbol's signals are aggregated independently (no cross-symbol
correlation in portfolio construction v1.0). *(Doc 11)*

**R-054** Minimum cash buffer: 5% of total capital always retained. *(Doc 11)*

---

### RISK MANAGEMENT (R-055 – R-075)

**R-055** Exactly 8 risk controls with these default values:

| Control | Default |
|---|---|
| Max position size | 5% of equity |
| Max portfolio exposure | 95% |
| Max leverage | 2× |
| Stop loss per trade | −2% |
| Take profit per trade | +3% |
| Daily loss circuit breaker | −1.5% |
| Max drawdown emergency halt | −20% |
| Max trades/day | 50 |

*(Doc 12)*

**R-056** Circuit breaker sequence when fired: (1) set `trading_halted = True`, (2) all
subsequent `validate_order()` return `False`, (3) dashboard shows red TRADING HALTED
banner, (4) halt reason logged to `pbt_orchestrator.json`, (5) resume requires explicit
WebSocket command `{"action": "resume_trading"}` or system restart. *(Doc 12)*

**R-057** Stop-loss and take-profit evaluated at every step for all open positions.
Entry prices tracked per position in `PortfolioState`. *(Doc 12)*

**R-058** `validate_order()` must check in this order: (1) position size, (2) daily loss,
(3) drawdown, (4) daily trade count, (5) exposure. *(Doc 12)*

**R-059** Risk status object broadcast to dashboard must contain:
`trading_halted`, `halt_reason`, `current_drawdown_pct`, `daily_trades`,
`daily_pnl_pct`, `peak_equity`. *(Doc 12)*

**R-060** Exceeding position size: reduce qty to fit limit (do not reject outright).
Exceeding exposure or leverage: reject order. *(Doc 12)*

**R-061** Risk limits cannot be hot-patched at runtime. Changes require code modification,
test suite run, and system restart. *(Doc 20)*

**R-062** Daily trade count must auto-reset the next calendar day. *(Doc 13 — order
rejection table: `max_trades_exceeded` "auto-resets next day")*

---

### EXECUTION ENGINE (R-063 – R-078)

**R-063** All orders in v1.0 are market orders only. Time-in-force: Day (GTC in paper
mode). Fractional shares: supported. *(Doc 13)*

**R-064** `PaperBroker.submit_order()` must: (a) fill at last simulated price, (b) cap
fill at available cash (partial fill to cash limit), (c) update cash, positions, equity,
and `total_trades` atomically. Return dict with `status`, `price`, `qty`, `side`. *(Doc 13)*

**R-065** `AlpacaBroker.submit_order()` must call `api.submit_order(symbol, round(qty,6),
side, "market", "day")`. Credentials from env vars `ALPACA_API_KEY` and
`ALPACA_SECRET_KEY`. Live endpoint: `https://api.alpaca.markets`. *(Docs 13, 18)*

**R-066** Trade record broadcast to dashboard must include:
`symbol`, `side`, `qty`, `price`, `confidence`, `agent_id`, `timestamp`. *(Doc 13)*

**R-067** `step_sleep_ms = 100` throttles the trading loop to prevent order flooding.
*(Doc 04)*

**R-068** Six order rejection reasons must be handled and logged:
`position_too_large`, `daily_loss_limit`, `max_drawdown`, `max_trades_exceeded`,
`insufficient_cash`, `trading_halted`. *(Doc 13)*

**R-069** `LiveTradingOrchestrator` in `trading/live_trading_orchestrator.py` must provide
single-agent fallback mode using one PPO agent with no population dynamics. *(Doc 13)*

**R-070** Alpaca supported operations: `get_latest_trade`, `submit_order`,
`list_positions`, `get_account`, `cancel_all_orders`. *(Doc 18)*

---

### FITNESS & BACKTESTING (R-071 – R-090)

**R-071** Six backtest metrics must be implemented with these exact formulas:

| Metric | Formula |
|---|---|
| Sharpe | `mean(r) / std(r) × √252` |
| Calmar | `CAGR / abs(MaxDrawdown)` |
| Sortino | `mean(r) / downside_std(r) × √252` |
| MaxDrawdown | `max((peak − trough) / peak)` |
| Win Rate | `wins / total_trades` |
| Profit Factor | `gross_profit / abs(gross_loss)` |

*(Docs 03, 14)*

**R-072** Phase 2 → Phase 3 threshold: Sharpe ≥ 0.8, MaxDrawdown ≤ 25%, Win Rate ≥ 48%.
*(Doc 03)*

**R-073** Phase 3 → Phase 4 (live) threshold: paper fitness ≥ 0.3, Sharpe ≥ 1.0, after
minimum 5 trading days or 10 completed generations. *(Docs 03, 16)*

**R-074** Composite fitness function weights:
`0.50 × tanh(Sharpe/3) + 0.20 × tanh(Calmar/5) + 0.15 × tanh(Sortino/3)
+ 0.10 × tanh(AnnReturn×5) − 0.05 × |MaxDrawdown|`
Output range: [−1, 1]. *(Doc 14)*

**R-075** Performance targets: Annual Sharpe ≥ 1.5, MaxDrawdown ≤ 20%, Daily Loss ≤ 1.5%,
Win Rate ≥ 52%, Calmar ≥ 0.8. *(Doc 01)*

**R-076** `BacktestEvaluator` must accept a list of equity values and expose `.sharpe_ratio`
and `.to_dict()` on its result object. *(Doc 14)*

**R-077** Experiments must be logged to `experiments/YYYYMMDD_hypothesis_name/` with:
`config.yaml`, `results.json`, `equity_curve.csv`, `notes.md`. Hypothesis must be logged
before backtest runs (no data snooping). *(Doc 03)*

---

### PBT EVOLUTION (R-078 – R-095)

**R-078** Exploit fraction = 0.20 (bottom 20% copy from top 20%). *(Doc 17)*

**R-079** Exploit implementation: for each bottom agent, sample a random top-20% agent
and copy hyperparameters. *(Doc 17)*

**R-080** Elite agent (rank 1) is never mutated and never replaced. *(Docs 15, 17)*

**R-081** Explore: Gaussian multiplicative perturbation;
`hp[key] = value × (1 + gauss(0, mutation_strength))`.
`mutate_prob = 0.8` per hyperparameter. All non-elite agents perturbed. *(Doc 17)*

**R-082** Hyperparameters clamped to bounds after mutation. *(Doc 17)*

**R-083** `EvolutionScheduler.get_stats()` must return:
`total_evolutions`, `exploits_performed`, `explorations_performed`,
`best_fitness_history`, `avg_fitness_history`. *(Doc 17)*

**R-084** Genealogy log per exploit: `generation`, `agent_id`, `parent_id`, `event`.
For explore events: `parent_id = null`. *(Doc 16)*

**R-085** Evolution audit record in governance log must include:
`generation`, `agent_id`, `parent_id`, `event`,
`hyperparams_before`, `hyperparams_after`. *(Doc 20)*

---

### MONITORING & DASHBOARD (R-086 – R-105)

**R-086** Browser dashboard must render these 10 panels:
Header, Population leaderboard (12 agents ranked), Live prices (4 symbols),
Risk controls, Trade feed (last 30 trades), Metric cards, Generation progress bar,
Equity curve (600-step rolling Chart.js), Fitness evolution chart (best vs avg per
generation), Evolution log (last 25 events). *(Doc 19)*

**R-087** Dashboard WebSocket auto-connects at `ws://${location.host}/ws`. *(Doc 19)*

**R-088** Reconnect backoff: exponential 1s → 10s max. *(Doc 19)*

**R-089** Four reconnect states: Connecting (grey dot), Connected (green pulse),
Error (red dot), Halted (red banner + resume button). *(Doc 19)*

**R-090** Fitness bar normalisation in dashboard:
`(fitness + 1) × 50` → maps [−1, 1] to [0%, 100%] visual width. *(Doc 15)*

**R-091** `RegimeDetector` must expose `.regime`, `.volatility`, `.trend_strength` on
its returned state object. *(Doc 19)*

**R-092** RegimeDetector window default: 20. *(Doc 19)*

**R-093** `MetricsLogger` must support optional TensorBoard via `use_tensorboard` flag.
TensorBoard events written to `logs/tensorboard/`. *(Doc 19)*

**R-094** Console dashboard (`pbt-live` mode) must render ASCII output showing:
generation number, agent leaderboard with Sharpe/fitness/return, progress bar,
portfolio equity, cash, regime, vol, trades, drawdown, halt status. *(Doc 19)*

**R-095** v1.0 does not include automated alerting (email/Slack/PagerDuty). These are
v1.1 features — the code must NOT silently implement them as stubs that would fail at
runtime, nor claim they are active. *(Doc 19)*

---

### LIVE TRADING (R-096 – R-110)

**R-096** Alpaca live endpoint: `https://api.alpaca.markets`. Paper endpoint:
`https://paper-api.alpaca.markets`. The code must default to paper endpoint even in
`broker_mode=live` unless explicitly overridden. *(Doc 07)*

**R-097** Market hours: US equity trading 9:30 AM–4:00 PM ET Mon–Fri. After-hours order
rejections must be logged but must NOT trigger circuit breakers. Crypto (BTC/USD) trades
24/7. v1.0 has no automatic market hours detection (planned v1.1). *(Doc 18)*

**R-098** MarketZero must NOT poll for order status after submission. Fills assumed
immediate for market orders. *(Doc 18)*

**R-099** Credentials must never be hard-coded in source files. Always read from env vars
`ALPACA_API_KEY` and `ALPACA_SECRET_KEY`. *(Doc 18)*

**R-100** Recommended tightened live limits for initial deployment:
`max_position_size=0.03`, `daily_loss_limit=0.01`, `max_drawdown=0.10`,
`max_daily_trades=20`. These are guidance values — they must not be hard-coded as the
production defaults. *(Doc 18)*

**R-101** Emergency stop Option 2 note: the doc labels `{"action": "resume_trading"}`
as the WebSocket command. Verify whether a separate manual-halt WebSocket command exists
or is absent (gap if missing). *(Doc 18)*

---

### MLOPS & GOVERNANCE (R-102 – R-118)

**R-102** Deployment checklist (all must be verifiable in code or logs):
30 tests pass; paper fitness ≥ 0.3 and Sharpe ≥ 1.0; env vars set; `broker_mode="live"`
confirmed; risk limits reviewed; dashboard accessible; checkpoint dir writable. *(Doc 16)*

**R-103** Minimum pip dependencies (v1.0): `numpy`, `alpaca-trade-api`, `fastapi`,
`uvicorn[standard]`, `websockets`. TensorBoard requires `torch` or `tensorflow`
(optional). *(Doc 16)*

**R-104** `--resume-from checkpoints/generation_NNNN` flag must restore population
exactly (weights + hyperparameters). *(Doc 16)*

**R-105** `executor.save_generation()` must be callable manually and must trigger
automatically on `KeyboardInterrupt` in `pbt-live` mode. *(Doc 16)*

**R-106** Agent versions implicitly tracked by generation number in genealogy log.
No separate version file required. *(Doc 16)*

**R-107** All code must be committed to version control before live deployment.
`main` branch = production only. Feature branches for all development. *(Doc 20)*

**R-108** Trade audit records must be append-only. No log records deleted. Fields required:
`timestamp`, `generation`, `agent_id`, `agent_type`, `symbol`, `side`, `qty`, `price`,
`confidence`, `regime`, `fitness_at_time`, `risk_approved`. *(Doc 20)*

**R-109** Risk limits must not be changeable via hot-patch at runtime; require code
modification + test run + restart. *(Doc 20)*

**R-110** Explainability requirement: no black-box neural networks in v1.0. All agent
policies must be linear models. *(Doc 20)*

---

### DISASTER RECOVERY (R-111 – R-118)

**R-111** Process crash recovery: `--resume-from checkpoints/generation_NNNN` must
restore trading state within RTO < 2 minutes (paper) / < 5 minutes (live). *(Doc 21)*

**R-112** WebSocket disconnect: trading engine must continue running independently of
dashboard connection state. Dashboard auto-reconnects; no trading interruption. *(Docs 19, 21)*

**R-113** Alpaca API failure handling: trading loop must retry with 1-second delay on
`AlpacaAPIError`. *(Doc 21)*

**R-114** Checkpoint corruption fallback: system must be startable fresh (no `--resume`
flag) if all checkpoints are corrupt. *(Doc 21)*

**R-115** Log disk full: trading must continue even if log writes fail (`IOError` must not
crash the trading loop). *(Doc 21)*

**R-116** Corrupted paper portfolio: system must be restartable with a fresh portfolio
by omitting `--resume` flag. *(Doc 21)*

**R-117** Global exception handler must produce readable error pages (not raw 500 stack
traces) in development. *(Doc 22)*

**R-118** Checkpoint backup: last 3 checkpoints must be recoverable as fallback if the
latest is corrupt. (Retention of 10 checkpoints by R-018 satisfies this.) *(Doc 21)*

---

### V1.1 SCOPE BOUNDARY (R-119 – R-122)

These items are explicitly **NOT** in v1.0. Flag any code that silently implements them
as active features — they may create false expectations or runtime failures.

**R-119** No slippage model in v1.0 paper broker. *(Docs 07, 14)*

**R-120** No transaction costs / commissions in paper mode. *(Docs 07, 14)*

**R-121** Walk-forward analysis not automated in v1.0. *(Doc 14)*

**R-122** No market hours detection / auto-suspension in v1.0. *(Docs 18, 22)*

---

## Step 3 — Audit Each Source File

For every source file in the repository:

1. Read the file fully.
2. Map each function, class, endpoint, and config value to the relevant R-NNN requirements above.
3. Check for:
   - **Missing implementation** — an R-NNN has no corresponding code
   - **Value divergence** — a constant, default, threshold, or formula differs from the documented value (even by one decimal place)
   - **Logic divergence** — documented algorithm or sequence not followed
   - **Stale code** — code implements something the docs explicitly mark as v1.1 or future
   - **Interface violation** — method signatures, return shapes, or field names differ from documented contracts
   - **Naming violation** — file, class, or method names differ from documented module names
   - **Undocumented behaviour** — significant logic with no documentation counterpart

---

## Step 4 — Compliance Report

Produce the following structured report:

---

### ✅ COMPLIANT
List R-NNN IDs that are fully implemented exactly as documented. One line each.

---

### ⚠️ PARTIAL COMPLIANCE

| R-NNN | Requirement | File | Line(s) | Gap |
|---|---|---|---|---|

---

### ❌ NON-COMPLIANT / MISSING

| R-NNN | Requirement Summary | Expected | Actual / Status |
|---|---|---|---|

---

### 🔄 STALE / MISMATCHED DOCUMENTATION

| File | Function / Section | Doc Says | Code Does |
|---|---|---|---|

---

### 🔍 UNDOCUMENTED BEHAVIOUR

| File | Symbol | Behaviour | Risk Level |
|---|---|---|---|

---

### 📊 COMPLIANCE SCORECARD

```
Total requirements audited : NNN
✅ Fully compliant         : NNN (XX%)
⚠️ Partial                 : NNN
❌ Non-compliant / Missing  : NNN
🔄 Stale docs              : NNN
🔍 Undocumented behaviours : NNN
────────────────────────────────────
Overall compliance score   : XX%
```

---

## Step 5 — Prioritised Fix List

Produce a prioritised list of all non-compliant and partial items:

**CRITICAL** — Would cause a runtime crash, silent data corruption, or real capital loss
**HIGH** — Functional deviation from documented behaviour; would fail the test suite
**MEDIUM** — Wrong default value, wrong formula coefficient, or missing feature
**LOW** — Naming mismatch, missing log field, minor interface deviation

For each issue provide:
- Priority level
- R-NNN reference
- One-line description of the fix required
- File and approximate line number if known

---

## Audit Rules

1. Treat every R-NNN above as a binding contract. Documented values (defaults, thresholds,
   coefficients, field names, enum values) are exact — a value of `0.19` where the doc says
   `0.20` is a violation.
2. Treat docstrings and inline comments containing "must", "shall", "always", "never" as
   binding documentation.
3. If a requirement is ambiguous across documents, note the ambiguity rather than
   guessing intent.
4. Do not suggest new features. Only audit what is documented vs. what exists.
5. Binary files (`.pkl`, `.pyc`) — note and skip.
6. If a file is absent that the docs require, that is a MISSING IMPLEMENTATION finding.
7. The fitness formula (R-074) coefficients must match exactly: 0.50, 0.20, 0.15, 0.10,
   0.05. Any deviation is a HIGH finding.
8. Any hard-coded credential is an automatic CRITICAL finding regardless of other context.
9. The v1.1 scope boundary items (R-119 – R-122) must not be active in v1.0 code.

**Begin with Step 1. Work through all five steps sequentially.**