# 12 — Risk Management

---

## Overview

Risk management in MarketZero is implemented as a hard-gating layer between signal generation and order execution. No trade can bypass the risk manager. It operates synchronously on every order attempt and can halt all trading system-wide.

Source: `trading/risk_management.py`

---

## Risk Controls Reference

| Control | Default | Trigger | Action |
|---|---|---|---|
| Max position size | 5% | Order exceeds 5% of equity | Reduce to 5% |
| Max portfolio exposure | 95% | Total exposure > 95% equity | Reject order |
| Max leverage | 2× | Gross leverage > 2× | Reject order |
| Stop loss per trade | -2% | Position P&L < -2% | Force close |
| Take profit per trade | +3% | Position P&L > +3% | Force close |
| Daily loss circuit breaker | -1.5% | Daily P&L < -1.5% | HALT all trading |
| Max drawdown emergency | -20% | Portfolio drawdown > 20% | HALT all trading |
| Max trades/day | 50 | Trade count ≥ 50 | Reject order |

---

## Circuit Breaker Logic

```python
def validate_order(self, symbol, qty, side, portfolio):
    # 1. Position size check
    order_value = qty * price
    if order_value / portfolio.equity > self.max_position_size:
        qty = self._scale_to_limit(qty, price, portfolio.equity)

    # 2. Daily loss check
    if portfolio.daily_pnl / portfolio.equity < -self.daily_loss_limit:
        self._halt("Daily loss limit breached")
        return False

    # 3. Drawdown check
    drawdown = (portfolio.equity - self.peak_equity) / self.peak_equity
    if drawdown < -self.max_drawdown:
        self._halt("Max drawdown breached")
        return False

    # 4. Daily trade count
    if self.daily_trades >= self.max_daily_trades:
        return False

    # 5. Exposure check
    total_exposure = sum(abs(v) for v in portfolio.positions.values())
    if (total_exposure + order_value) / portfolio.equity > self.max_exposure:
        return False

    return True
```

---

## Trading Halt State

When a circuit breaker fires:

1. `risk_manager.trading_halted = True`
2. All subsequent `validate_order()` calls return `False`
3. Dashboard shows red **TRADING HALTED** banner
4. Halt reason is logged to `pbt_orchestrator.json`
5. Resume requires explicit command via WebSocket or system restart

To resume via WebSocket (browser or client):
```json
{"action": "resume_trading"}
```

To resume programmatically:
```python
executor.risk_manager.resume_trading()
```

---

## Stop-Loss and Take-Profit

Stop-loss and take-profit are evaluated at every step for all open positions:

```python
for symbol, qty in positions.items():
    position_pnl_pct = (current_price - entry_price) / entry_price
    if position_pnl_pct < -stop_loss_pct:          # -2%
        force_close(symbol, reason="stop_loss")
    elif position_pnl_pct > take_profit_pct:        # +3%
        force_close(symbol, reason="take_profit")
```

Entry prices are tracked per position in `PortfolioState`.

---

## Risk Status Object

The risk manager exposes a status dict broadcast to the dashboard:

```python
{
    "trading_halted": False,
    "halt_reason": "",
    "current_drawdown_pct": -1.23,
    "daily_trades": 12,
    "daily_pnl_pct": +0.34,
    "peak_equity": 121400.0,
}
```

---

## Adjusting Risk Parameters

Override defaults by subclassing or patching `RiskManager` before passing to executor:

```python
config = PBTLiveConfig(...)
executor = PBTLiveExecutor(config)
executor.risk_manager.max_position_size = 0.03      # tighter: 3%
executor.risk_manager.daily_loss_limit = 0.01       # tighter: 1%
executor.risk_manager.max_drawdown = 0.15           # tighter: 15%
```

For persistent configuration, add fields to `PBTLiveConfig` and wire them through in `PBTLiveExecutor.__init__`.

---

## Risk Reporting

Daily risk summary is written to `logs/pbt_orchestrator.json` at each generation end. Metrics include: peak equity, current drawdown, daily P&L, total trades, halt events, and circuit breaker trigger counts.
