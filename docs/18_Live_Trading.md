# 18 — Live Trading

---

## Prerequisites

Before running with real capital:

1. Alpaca brokerage account (live, not paper)
2. Margin agreement signed (if using leverage)
3. API key generated with trading permissions
4. All 30 tests passing
5. Paper trading validation completed (≥ 5 days, fitness ≥ 0.3)

---

## Credentials Setup

```bash
export ALPACA_API_KEY=PKXXXXXXXXXXXXXXXX
export ALPACA_SECRET_KEY=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

Never hard-code credentials in source files. Never commit credentials to version control.

---

## Starting Live Trading

```bash
python main.py serve \
  --broker-mode live \
  --population-size 12 \
  --symbols AAPL SPY GLD \
  --total-capital 120000 \
  --capital-per-agent 5000 \
  --generation-steps 500 \
  --port 8000
```

Open the dashboard at `http://localhost:8000` to monitor in real time.

---

## Alpaca Integration

The Alpaca broker client (`trading/live_trading.py`) wraps the Alpaca Trade API:

```python
import alpaca_trade_api as tradeapi

api = tradeapi.REST(
    key_id=os.environ["ALPACA_API_KEY"],
    secret_key=os.environ["ALPACA_SECRET_KEY"],
    base_url="https://api.alpaca.markets",  # live
    # base_url="https://paper-api.alpaca.markets",  # paper
)
```

### Supported Operations

| Operation | API Call |
|---|---|
| Get price | `api.get_latest_trade(symbol)` |
| Submit order | `api.submit_order(symbol, qty, side, "market", "day")` |
| Get positions | `api.list_positions()` |
| Get portfolio | `api.get_account()` |
| Cancel all orders | `api.cancel_all_orders()` |

---

## Market Hours

Alpaca live trading operates during US market hours: **9:30 AM – 4:00 PM ET, Monday–Friday**.

Outside market hours:
- Orders are rejected
- The trading loop continues to run (pricing is from last trade)
- Risk manager absorbs after-hours P&L changes silently

The system does not currently have automatic market hours detection. If running 24/7, after-hours rejections are logged but do not trigger circuit breakers.

Crypto symbols (BTC/USD) trade 24/7 without interruption.

---

## Order Monitoring

All orders submitted to Alpaca can be monitored at:
- `https://app.alpaca.markets/paper/orders` (paper account)
- `https://app.alpaca.markets/orders` (live account)

MarketZero does not currently poll for order status after submission. Fill confirmations are assumed immediate (market orders under normal liquidity conditions).

---

## Emergency Stop

To immediately halt all trading:

Option 1 — Dashboard: The red HALT button appears automatically when circuit breakers fire.

Option 2 — WebSocket command:
```json
{"action": "resume_trading"}
```
(Note: this resumes a halted system. To halt manually, use Ctrl+C.)

Option 3 — Ctrl+C in terminal: saves checkpoint and exits cleanly.

Option 4 — Alpaca dashboard: Cancel all open orders and liquidate positions manually at `app.alpaca.markets`.

---

## Risk Limits in Live Mode

In live mode, the same risk controls apply as in paper mode. Recommended tightened limits for initial live deployment:

```python
executor.risk_manager.max_position_size = 0.03      # 3% (vs 5% default)
executor.risk_manager.daily_loss_limit = 0.01        # 1% (vs 1.5% default)
executor.risk_manager.max_drawdown = 0.10            # 10% (vs 20% default)
executor.risk_manager.max_daily_trades = 20          # 20 (vs 50 default)
```

Start conservative. Expand limits only after observing stable live performance over multiple weeks.

---

## Gradual Capital Deployment

Recommended live deployment schedule:

| Week | Capital | Population | Notes |
|---|---|---|---|
| 1 | $10,000 | 4 agents | Observe, validate fills |
| 2–3 | $30,000 | 8 agents | Monitor drawdowns |
| 4+ | Full capital | 12 agents | Normal operations |
