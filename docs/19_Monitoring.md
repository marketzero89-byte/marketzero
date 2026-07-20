# 19 — Monitoring

---

## Monitoring Stack

| Layer | Tool | Location |
|---|---|---|
| Browser dashboard | WebSocket + Chart.js | `http://localhost:8000` |
| Console dashboard | ANSI terminal output | `pbt-live` mode only |
| JSON logs | JSON-lines file | `logs/pbt_orchestrator.json` |
| TensorBoard | Scalar event files | `logs/tensorboard/` |
| Genealogy log | JSON array | `logs/pbt_genealogy.json` |

---

## Browser Dashboard

Source: `dashboard.html` (single-file, no build step)

### Panels

| Panel | Content |
|---|---|
| Header | Generation, elapsed time, agent count, regime pill, clock |
| Population leaderboard | All 12 agents ranked by fitness, with type badge and fitness bar |
| Live prices | Current prices for all 4 symbols, colour-coded by tick direction |
| Risk controls | Drawdown %, daily trades, halt status |
| Trade feed | Last 30 executed trades with side, symbol, qty, price, confidence |
| Metric cards | Portfolio equity, total P&L, best fitness, generation/step |
| Generation progress | Step progress bar + fitness normalised progress bar |
| Equity curve | Rolling 600-step equity chart (Chart.js line, WebSocket-updated) |
| Fitness evolution | Best vs avg fitness per generation (Chart.js line) |
| Evolution log | Last 25 exploit/explore events with generation and delta |

### WebSocket Connection

The dashboard auto-connects on load:
```javascript
const ws = new WebSocket(`ws://${location.host}/ws`);
```

On disconnect, it automatically reconnects with exponential backoff (1s → 10s max). A grey dot in the header indicates disconnected state; green pulsing dot indicates live feed.

### Reconnect Behaviour

| State | Indicator | Action |
|---|---|---|
| Connecting | Grey dot | Reconnect overlay shown |
| Connected | Green pulse | Dashboard live |
| Error | Red dot | Retry pending |
| Halted | Red banner | Resume button visible |

---

## Console Dashboard

The `pbt-live` command renders an ASCII dashboard after each generation:

```
╔═══════════════════════════════════════════════════════════════╗
║  MarketZero PBT – GENERATION 5                               ║
╠═══════════════════════════════════════════════════════════════╣
║  1. PPO_a3f2b1       Sharpe=1.20  Fit=+0.8471  Ret=+4.23%   ║
║  2. Dreamer_9c41     Sharpe=1.10  Fit=+0.8213  Ret=+3.87%   ║
╠═══════════════════════════════════════════════════════════════╣
║  Progress: [████████████████████░░░░░░░░░░░░░░░░░░░░] 500/500║
║  Portfolio: $125,400.00  Cash: $95,200.00                     ║
║  Regime: BULL  Vol: 0.182  Trades: 47                         ║
║  Drawdown: -1.23%  Halted: NO                                 ║
╚═══════════════════════════════════════════════════════════════╝
```

Source: `monitoring/monitoring.py` → `PBTLiveDashboard.render()`

---

## RegimeDetector

Detects market regime from price history using EMA crossover + volatility:

```python
detector = RegimeDetector(window=20)
state = detector.update("AAPL", 185.50)

state.regime      # MarketRegime.BULL
state.volatility  # 0.182 (annualised)
state.trend_strength  # 0.006 (EMA ratio)
```

### Regime Classification Logic

```
if annualised_vol > 0.35:  → HIGH_VOL
elif trend > 0.005 and mean_return > 0:  → BULL
elif trend < -0.005 and mean_return < 0: → BEAR
else: → RANGING
```

Regime updates on every step. The regime pill on the dashboard reflects the last computed value.

---

## MetricsLogger

Writes per-step records to `logs/pbt_orchestrator.json`:

```python
logger = MetricsLogger(log_dir="logs", use_tensorboard=False)
logger.log({"generation": 5, "best_fitness": 0.847}, step=5)
logger.close()
```

With TensorBoard enabled:
```python
logger = MetricsLogger(log_dir="logs", use_tensorboard=True)
# Then view:  tensorboard --logdir=logs/tensorboard
```

---

## Alerting

v1.0 does not include automated alerting (email, Slack, PagerDuty). The circuit breaker dashboard banner is the primary alert mechanism.

Planned for v1.1:
- Email alert on trading halt
- Slack webhook on daily loss limit breach
- SMS alert on emergency max drawdown halt
