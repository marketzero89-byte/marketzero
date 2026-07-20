# 13 — Execution Engine

---

## Overview

The execution engine is the interface between the signal pipeline and the broker. It translates agent signals into concrete buy/sell orders, handles fills, and updates portfolio state. Source: `trading/live_trading.py` (Alpaca), `pbt/pbt_live_trading.py` (orchestrator).

---

## Order Lifecycle

```
SignalAggregator → aggregate signal
        │
        ▼
RiskManager.validate_order()
        │
   Pass │   Fail → log rejection, skip
        ▼
Broker.submit_order(symbol, qty, side, confidence)
        │
        ▼
Fill (paper: immediate | live: Alpaca REST)
        │
        ▼
PortfolioState.update()
        │
        ▼
StepResult.trades.append(trade_record)
        │
        ▼
Broadcast to dashboard via WebSocket
```

---

## Order Types

In v1.0, all orders are **market orders**. Limit and stop-limit order types are planned for v1.1 (see `22_Research_Roadmap.md`).

| Field | Value |
|---|---|
| Order type | Market |
| Time in force | Day (GTC in paper mode) |
| Fractional shares | Yes (Alpaca supports fractional) |
| Short selling | Yes (if enabled in Alpaca account) |

---

## Paper Broker Execution

```python
class PaperBroker:
    def submit_order(self, symbol, qty, side, confidence):
        fill_price = self.get_latest_price(symbol)
        if side == "buy":
            cost = qty * fill_price
            if cost > self._state.cash:
                qty = self._state.cash / fill_price  # partial fill to cash limit
            self._state.cash -= qty * fill_price
            self._state.positions[symbol] = self._state.positions.get(symbol, 0) + qty
        else:  # sell
            self._state.positions[symbol] = max(0, self._state.positions.get(symbol, 0) - qty)
            self._state.cash += qty * fill_price
        self._state.total_trades += 1
        self._state.equity = self._state.cash + sum(
            qty * self.get_latest_price(s) for s, qty in self._state.positions.items()
        )
        return {"status": "filled", "price": fill_price, "qty": qty, "side": side}
```

---

## Alpaca Live Execution

```python
class AlpacaBroker:
    def submit_order(self, symbol, qty, side, confidence):
        order = self._api.submit_order(
            symbol=symbol,
            qty=round(qty, 6),
            side=side,
            type="market",
            time_in_force="day",
        )
        return {"status": order.status, "id": order.id, "filled_qty": order.filled_qty}
```

Alpaca credentials are read from environment variables:
```bash
ALPACA_API_KEY=...
ALPACA_SECRET_KEY=...
```

---

## Trade Record Format

Each executed trade is appended to `step_result.trades` and broadcast to the dashboard:

```json
{
    "symbol": "AAPL",
    "side": "buy",
    "qty": 2.5,
    "price": 185.40,
    "confidence": 0.78,
    "agent_id": "PPO_a3f2b1",
    "timestamp": 1718000000.0
}
```

---

## Execution Latency

| Mode | Typical latency |
|---|---|
| Paper (internal) | < 1ms (same-thread, synchronous) |
| Alpaca paper API | 50–200ms (REST round-trip) |
| Alpaca live API | 50–300ms (REST round-trip) |

`step_sleep_ms` (default: 100ms) throttles the trading loop to prevent order flooding.

---

## Order Rejection Reasons

| Reason | Risk control | Resolution |
|---|---|---|
| `position_too_large` | Max position size (5%) | Reduce qty or confidence |
| `daily_loss_limit` | Circuit breaker | Resume via dashboard |
| `max_drawdown` | Emergency halt | Resume via dashboard |
| `max_trades_exceeded` | Daily trade count (50) | Auto-resets next day |
| `insufficient_cash` | Cash balance | Reduce position sizes |
| `trading_halted` | Circuit breaker active | Resume via dashboard |

---

## Single-Agent Fallback

For simplified single-agent operation (no PBT, no evolution):

```python
from trading.live_trading_orchestrator import LiveTradingOrchestrator
orch = LiveTradingOrchestrator(config)
orch.run()
```

This mode is provided for debugging and comparison. It uses a single PPO agent without population dynamics.
