"""
Risk Manager
Position limits, daily loss circuit breaker, max drawdown halt, stop-loss/take-profit.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class RiskConfig:
    # R-100 Live-mode guidance:
    # Before enabling live trading (--live flag) review and tighten these values:
    #   max_position_pct   — reduce to 0.10 for live to limit single-position concentration
    #   daily_loss_limit_pct — consider 0.02 (2%) for a conservative live circuit breaker
    #   max_drawdown_pct   — 0.10 or lower for live accounts
    #   max_trades_per_day — lower if using Alpaca's Pattern Day Trader rules apply
    # All defaults below are sized for paper-trading experimentation.
    max_position_pct:     float = 0.20    # max single position as % of equity
    max_portfolio_pct:    float = 0.80    # max total market exposure
    daily_loss_limit_pct: float = 0.20    # halt if daily loss exceeds 20%
    max_drawdown_pct:     float = 0.30    # halt if max drawdown exceeds 30%
    stop_loss_pct:        float = 0.08    # stop-loss per position (8%)
    take_profit_pct:      float = 0.15    # take-profit per position (15%)
    max_trades_per_day:   int   = 200     # excessive trading guard
    min_trade_size:       float = 1.0     # minimum qty
    max_trade_size_pct:   float = 0.10    # max single trade as % of equity


@dataclass
class RiskEvent:
    event_type: str          # 'daily_loss' | 'max_drawdown' | 'position_limit' | 'stop_loss' | 'take_profit'
    symbol: Optional[str]
    message: str
    timestamp: float = field(default_factory=time.time)
    severity: str = "warning"   # 'warning' | 'halt'


class RiskManager:
    """
    Enforces risk limits on the trading system.

    Parameters
    ----------
    config : RiskConfig
    alert_callbacks : list of callables to fire on risk events
    """

    def __init__(
        self,
        config: Optional[RiskConfig] = None,
        alert_callbacks: Optional[List[Callable]] = None,
    ):
        self.config = config or RiskConfig()
        self._callbacks = alert_callbacks or []

        self._halted: bool = False
        self._halt_reason: str = ""
        self._events: List[RiskEvent] = []

        self._session_start_equity: float = 0.0
        self._peak_equity: float = 0.0
        self._trade_count_today: int = 0
        self._session_date: str = ""
        self._daily_pnl: float = 0.0       # R-059
        self._daily_pnl_pct: float = 0.0   # R-059

        # Per-symbol entry tracking for stop-loss / take-profit
        self._entry_prices: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # State update
    # ------------------------------------------------------------------

    def update(self, equity: float, positions: Dict[str, Dict]) -> None:
        """Call every step to update risk state. Does NOT halt automatically."""
        today = time.strftime("%Y-%m-%d")
        if today != self._session_date:
            self._session_date        = today
            self._session_start_equity= equity
            self._trade_count_today   = 0
            self._daily_pnl           = 0.0
            self._daily_pnl_pct       = 0.0

        if equity > self._peak_equity:
            self._peak_equity = equity

        # Track daily PnL (R-059)
        if self._session_start_equity > 0:
            self._daily_pnl     = equity - self._session_start_equity
            self._daily_pnl_pct = self._daily_pnl / self._session_start_equity

        # Check daily loss
        if self._session_start_equity > 0:
            daily_loss = (equity - self._session_start_equity) / self._session_start_equity
            if daily_loss < -self.config.daily_loss_limit_pct:
                self._fire(RiskEvent(
                    event_type="daily_loss",
                    symbol=None,
                    message=f"Daily loss {daily_loss*100:.1f}% exceeds limit "
                            f"{self.config.daily_loss_limit_pct*100:.1f}%",
                    severity="halt",
                ))
                self._halt("Daily loss limit breached")

        # Check max drawdown
        if self._peak_equity > 0:
            dd = (equity - self._peak_equity) / self._peak_equity
            if dd < -self.config.max_drawdown_pct:
                self._fire(RiskEvent(
                    event_type="max_drawdown",
                    symbol=None,
                    message=f"Max drawdown {dd*100:.1f}% exceeds limit "
                            f"{self.config.max_drawdown_pct*100:.1f}%",
                    severity="halt",
                ))
                self._halt("Max drawdown limit breached")

    # ------------------------------------------------------------------
    # Pre-trade checks
    # ------------------------------------------------------------------

    def check_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        equity: float,
        current_positions: Dict[str, float],
    ) -> tuple[bool, str]:
        """
        Returns (allowed: bool, reason: str).
        Call before submitting any order.

        Check order R-058: (1) position size, (2) daily loss, (3) drawdown,
        (4) daily trade count, (5) exposure, (+) cash buffer (R-054).

        Parameters
        ----------
        current_positions : Dict[str, float]
            Mapping of symbol -> *market value* (not qty) of each open position.
            Example: {"AAPL": 15000.0, "MSFT": 8500.0}
        """
        if self._halted:
            return False, f"System halted: {self._halt_reason}"

        if qty < self.config.min_trade_size:
            return False, f"Order size {qty} below minimum {self.config.min_trade_size}"

        order_value = price * qty

        # (1) Position size: reduce to fit, don't reject (R-060)
        if equity > 0 and order_value / equity > self.config.max_trade_size_pct:
            allowed_qty = max(
                int(self.config.min_trade_size),
                int(equity * self.config.max_trade_size_pct / max(price, 1e-10)),
            )
            if allowed_qty < self.config.min_trade_size:
                return False, "Position size too small after reduction to fit limit"
            qty = allowed_qty
            order_value = price * qty

        # (2) Daily loss limit (R-058)
        if self._session_start_equity > 0:
            daily_loss = (equity - self._session_start_equity) / self._session_start_equity
            if daily_loss < -self.config.daily_loss_limit_pct:
                return False, (
                    f"daily_loss_limit: daily loss {daily_loss*100:.1f}% exceeds "
                    f"{self.config.daily_loss_limit_pct*100:.1f}%"
                )

        # (3) Max drawdown (R-058)
        if self._peak_equity > 0:
            dd = (equity - self._peak_equity) / self._peak_equity
            if dd < -self.config.max_drawdown_pct:
                return False, (
                    f"max_drawdown: drawdown {dd*100:.1f}% exceeds "
                    f"{self.config.max_drawdown_pct*100:.1f}%"
                )

        # (4) Daily trade count (R-058)
        if self._trade_count_today >= self.config.max_trades_per_day:
            return False, f"max_trades_exceeded: daily limit ({self.config.max_trades_per_day}) reached"

        # (5) Portfolio exposure (R-058)
        if side == "buy" and equity > 0:
            current_val = sum(current_positions.values())
            if (current_val + order_value) / equity > self.config.max_portfolio_pct:
                return False, (
                    f"exposure: portfolio exposure would exceed "
                    f"{self.config.max_portfolio_pct*100:.0f}%"
                )

        # (6) 5% minimum cash buffer (R-054)
        if side == "buy" and equity > 0:
            min_cash = equity * 0.05
            # Estimate remaining cash after this buy
            current_cash_approx = equity - sum(current_positions.values())
            if current_cash_approx - order_value < min_cash:
                return False, "cash_buffer: order would breach 5% minimum cash buffer"

        return True, "ok"

    # ------------------------------------------------------------------
    # Stop-loss / Take-profit
    # ------------------------------------------------------------------

    def register_entry(self, symbol: str, price: float) -> None:
        """Record entry price for a symbol (for SL/TP tracking)."""
        self._entry_prices[symbol] = price

    def check_stop_take(
        self, symbol: str, current_price: float
    ) -> tuple[bool, str]:
        """
        Returns (should_exit: bool, reason: str).
        """
        entry = self._entry_prices.get(symbol)
        if entry is None or entry == 0:
            return False, ""

        pct = (current_price - entry) / entry

        if pct < -self.config.stop_loss_pct:
            self._fire(RiskEvent(
                event_type="stop_loss",
                symbol=symbol,
                message=f"Stop-loss triggered on {symbol}: "
                        f"{pct*100:.2f}% < -{self.config.stop_loss_pct*100:.1f}%",
                severity="warning",
            ))
            # Clear entry price so this doesn't re-fire every subsequent step
            del self._entry_prices[symbol]
            return True, "stop_loss"

        if pct > self.config.take_profit_pct:
            self._fire(RiskEvent(
                event_type="take_profit",
                symbol=symbol,
                message=f"Take-profit triggered on {symbol}: "
                        f"{pct*100:.2f}% > {self.config.take_profit_pct*100:.1f}%",
                severity="warning",
            ))
            # Clear entry price so this doesn't re-fire every subsequent step
            del self._entry_prices[symbol]
            return True, "take_profit"

        return False, ""

    def record_trade(self) -> None:
        self._trade_count_today += 1

    # ------------------------------------------------------------------
    # Halt management
    # ------------------------------------------------------------------

    def _halt(self, reason: str) -> None:
        if not self._halted:
            self._halted = True
            self._halt_reason = reason
            logger.critical("RISK HALT: %s", reason)

    def reset_halt(self) -> None:
        self._halted = False
        self._halt_reason = ""
        logger.info("Risk halt cleared manually")

    @property
    def is_halted(self) -> bool:
        return self._halted

    @property
    def halt_reason(self) -> str:
        return self._halt_reason

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def _fire(self, event: RiskEvent) -> None:
        self._events.append(event)
        for cb in self._callbacks:
            try:
                cb(event)
            except Exception as exc:
                logger.error("Risk callback error: %s", exc)
        logger.warning("[%s] %s", event.severity.upper(), event.message)

    def recent_events(self, n: int = 20) -> List[Dict]:
        return [
            {
                "event_type": e.event_type,
                "symbol":     e.symbol,
                "message":    e.message,
                "severity":   e.severity,
                "timestamp":  e.timestamp,
            }
            for e in self._events[-n:]
        ]

    def status(self) -> Dict:
        return {
            "halted":             self._halted,
            "halt_reason":        self._halt_reason,
            "trades_today":       self._trade_count_today,
            "peak_equity":        round(self._peak_equity, 2),
            "session_start":      round(self._session_start_equity, 2),
            "daily_pnl":          round(self._daily_pnl, 2),
            "daily_pnl_pct":      round(self._daily_pnl_pct, 4),   # R-059
            "recent_event_count": len(self._events),
        }
