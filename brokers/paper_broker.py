"""
Paper Broker — GBM Price Simulator
Geometric Brownian Motion price simulator with portfolio state tracking and order execution.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Order:
    order_id: str
    symbol: str
    side: str           # 'buy' | 'sell'
    qty: float
    order_type: str     # 'market' | 'limit'
    limit_price: Optional[float] = None
    filled_qty: float = 0.0
    filled_price: Optional[float] = None
    status: str = "pending"   # pending | filled | cancelled
    timestamp: float = field(default_factory=time.time)
    commission: float = 0.0


@dataclass
class Position:
    symbol: str
    qty: float = 0.0
    avg_cost: float = 0.0
    unrealised_pnl: float = 0.0
    realised_pnl: float = 0.0


@dataclass
class Trade:
    trade_id: str
    symbol: str
    side: str
    qty: float
    price: float
    pnl: float
    timestamp: float
    commission: float = 0.0


# ---------------------------------------------------------------------------
# GBM Price Simulator
# ---------------------------------------------------------------------------

class GBMPriceSimulator:
    """
    Multi-symbol Geometric Brownian Motion price simulator.

    Parameters
    ----------
    symbols : list[str]
    initial_prices : dict[str, float] | None   defaults to $100 per symbol
    mu : float     annualised drift
    sigma : float  annualised volatility
    dt : float     time step in years (1/252 = 1 trading day)
    """

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        initial_prices: Optional[Dict[str, float]] = None,
        mu: float = 0.08,
        sigma: float = 0.015,
        dt: float = 1 / 252,
        seed: int = 0,
    ):
        self.symbols = symbols or ["AAPL"]
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self._rng = np.random.default_rng(seed)
        self._initial_prices: Dict[str, float] = (initial_prices or {}).copy()
        for sym in self.symbols:
            self._initial_prices.setdefault(sym, 100.0)
        self._prices: Dict[str, float] = dict(self._initial_prices)
        self._history: Dict[str, List[float]] = {s: [self._prices[s]] for s in self.symbols}

    def step(self) -> Dict[str, float]:
        """Advance one time step; return new prices."""
        new_prices: Dict[str, float] = {}
        for sym in self.symbols:
            p = self._prices[sym]
            z = self._rng.standard_normal()
            # GBM: dS = S*(mu*dt + sigma*sqrt(dt)*z)
            change = p * (self.mu * self.dt + self.sigma * np.sqrt(self.dt) * z)
            new_p = max(p + change, 0.01)
            self._prices[sym] = new_p
            new_prices[sym] = new_p
            self._history[sym].append(new_p)
        return new_prices

    def current_prices(self) -> Dict[str, float]:
        return dict(self._prices)

    def reset(self) -> None:
        """Reset prices to initial values. Call between generations to prevent
        unbounded GBM drift that triggers spurious stop/take-profit alerts."""
        self._prices = dict(self._initial_prices)
        self._history = {s: [self._prices[s]] for s in self.symbols}
        logger.debug("GBM price simulator reset to initial prices")

    def price_history(self, symbol: str) -> List[float]:
        return list(self._history.get(symbol, []))

    def ohlcv(self, symbol: str, n_bars: int = 1) -> List[Dict]:
        """Return last n_bars of simulated OHLCV (simplified: open=prev_close, high/low random)."""
        hist = self._history.get(symbol, [])
        bars = []
        for i in range(max(1, len(hist) - n_bars), len(hist)):
            close = hist[i]
            prev  = hist[i - 1] if i > 0 else close
            noise = abs(close * self.sigma * np.sqrt(self.dt))
            bars.append({
                "symbol": symbol,
                "open":   prev,
                "high":   max(prev, close) + noise * self._rng.random(),
                "low":    min(prev, close) - noise * self._rng.random(),
                "close":  close,
                "volume": int(self._rng.integers(100_000, 5_000_000)),
            })
        return bars


# ---------------------------------------------------------------------------
# Paper Broker
# ---------------------------------------------------------------------------

class PaperBroker:
    """
    Simulated paper trading broker.

    Parameters
    ----------
    initial_cash : float
    symbols : list[str]
    commission_pct : float   e.g. 0.001 = 0.1%
    slippage_pct : float     e.g. 0.0005 = 0.05%
    """

    def __init__(
        self,
        initial_cash: float = 100_000.0,
        symbols: Optional[List[str]] = None,
        commission_pct: float = 0.001,
        slippage_pct: float = 0.0005,
        simulator: Optional[GBMPriceSimulator] = None,
        seed: int = 0,
    ):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        self.symbols = symbols or ["AAPL"]
        self.simulator = simulator or GBMPriceSimulator(self.symbols, seed=seed)
        self.positions: Dict[str, Position] = {s: Position(s) for s in self.symbols}
        self.trades: List[Trade] = []
        self.orders: List[Order] = []
        self.equity_curve: List[float] = [initial_cash]
        self._session_start_equity = initial_cash
        self._trades_at_generation_start = 0
        self._current_prices: Dict[str, float] = self.simulator.current_prices()

    # ------------------------------------------------------------------
    # Simulation step
    # ------------------------------------------------------------------

    def step(self) -> Dict[str, float]:
        """Advance the price simulator and update P&L."""
        self._current_prices = self.simulator.step()
        self._update_unrealised_pnl()
        self._process_limit_orders()
        equity = self._compute_equity()
        self.equity_curve.append(equity)
        return self._current_prices

    def _compute_equity(self) -> float:
        market_value = sum(
            pos.qty * self._current_prices.get(sym, 0)
            for sym, pos in self.positions.items()
        )
        return self.cash + market_value

    def _update_unrealised_pnl(self) -> None:
        for sym, pos in self.positions.items():
            if pos.qty != 0:
                mkt = self._current_prices.get(sym, pos.avg_cost)
                pos.unrealised_pnl = (mkt - pos.avg_cost) * pos.qty

    def _process_limit_orders(self) -> None:
        for order in self.orders:
            if order.status != "pending" or order.order_type != "limit":
                continue
            price = self._current_prices.get(order.symbol, 0)
            if order.side == "buy" and price <= order.limit_price:
                self._execute_order(order, price)
            elif order.side == "sell" and price >= order.limit_price:
                self._execute_order(order, price)

    # ------------------------------------------------------------------
    # Order submission
    # ------------------------------------------------------------------

    def submit_market_order(self, symbol: str, side: str, qty: float) -> Order:
        price = self._current_prices.get(symbol)
        if price is None:
            raise ValueError(f"Unknown symbol: {symbol}")
        order = Order(
            order_id=str(uuid.uuid4())[:8],
            symbol=symbol,
            side=side,
            qty=qty,
            order_type="market",
        )
        self.orders.append(order)
        self._execute_order(order, price)
        return order

    def submit_limit_order(
        self, symbol: str, side: str, qty: float, limit_price: float
    ) -> Order:
        order = Order(
            order_id=str(uuid.uuid4())[:8],
            symbol=symbol,
            side=side,
            qty=qty,
            order_type="limit",
            limit_price=limit_price,
        )
        self.orders.append(order)
        logger.debug("Limit order %s %s %s@%.2f submitted", order.order_id, side, symbol, limit_price)
        return order

    def _execute_order(self, order: Order, market_price: float) -> None:
        """Fill an order at market_price ± slippage."""
        slippage = market_price * self.slippage_pct
        if order.side == "buy":
            fill_price = market_price + slippage
        else:
            fill_price = market_price - slippage

        commission = fill_price * order.qty * self.commission_pct
        cost = fill_price * order.qty + (commission if order.side == "buy" else -commission)

        # Cash & position update
        if order.side == "buy":
            if self.cash < cost:
                # R-064: partial fill to available cash rather than full cancel
                max_qty = int((self.cash / (fill_price * (1 + self.commission_pct))) )
                if max_qty <= 0:
                    order.status = "cancelled"
                    logger.warning("Insufficient cash for any fill on order %s", order.order_id)
                    return
                # Recompute with reduced qty
                order.qty     = float(max_qty)
                commission    = fill_price * order.qty * self.commission_pct
                cost          = fill_price * order.qty + commission
                logger.info(
                    "Partial fill %s: cash limited to %d shares (%.2f)",
                    order.order_id, max_qty, cost,
                )
            self.cash -= cost
            pos = self.positions[order.symbol]
            total_qty   = pos.qty + order.qty
            pos.avg_cost = (pos.avg_cost * pos.qty + fill_price * order.qty) / total_qty
            pos.qty = total_qty
            pnl = 0.0
        else:  # sell
            pos = self.positions[order.symbol]
            actual_qty = min(order.qty, pos.qty)
            if actual_qty <= 0:
                order.status = "cancelled"
                return
            pnl = (fill_price - pos.avg_cost) * actual_qty - commission
            self.cash += fill_price * actual_qty - commission
            pos.qty -= actual_qty
            pos.realised_pnl += pnl
            if pos.qty == 0:
                pos.avg_cost = 0.0

        order.filled_qty   = order.qty
        order.filled_price = fill_price
        order.commission   = commission
        order.status       = "filled"

        trade = Trade(
            trade_id=str(uuid.uuid4())[:8],
            symbol=order.symbol,
            side=order.side,
            qty=order.qty,
            price=fill_price,
            pnl=pnl,
            timestamp=time.time(),
            commission=commission,
        )
        self.trades.append(trade)
        logger.debug(
            "Filled %s %s %.0f@%.4f  pnl=%.2f  cash=%.2f",
            order.side, order.symbol, order.qty, fill_price, pnl, self.cash,
        )

    # ------------------------------------------------------------------
    # Portfolio state
    # ------------------------------------------------------------------

    def portfolio_state(self) -> Dict:
        equity = self._compute_equity()
        daily_pnl = equity - self.initial_cash   # simple proxy: total since init
        return {
            "cash":            round(self.cash, 2),
            "equity":          round(equity, 2),
            "daily_pnl":       round(equity - self._session_start_equity, 2),             # R-022
            "total_return_pct": round((equity / self.initial_cash - 1) * 100, 2),
            "positions": {
                sym: {
                    "qty":              pos.qty,
                    "avg_cost":         round(pos.avg_cost, 4),
                    "current_price":    round(self._current_prices.get(sym, 0), 4),
                    "unrealised_pnl":   round(pos.unrealised_pnl, 2),
                    "realised_pnl":     round(pos.realised_pnl, 2),
                }
                for sym, pos in self.positions.items()
                if pos.qty != 0 or pos.realised_pnl != 0
            },
            "n_trades":  len(self.trades),
            "prices":    {k: round(v, 4) for k, v in self._current_prices.items()},
        }

    def reset_daily_pnl(self) -> None:
        """Call at the start of each generation to reset the daily P&L baseline."""
        self._session_start_equity = self._compute_equity()

    def reset_equity_curve(self) -> None:
        """Reset equity curve to current equity at generation boundary."""
        current = self._compute_equity()
        self.equity_curve = [current]
        self.initial_cash = current  # rebases return % each generation

    def mark_generation_start(self) -> None:
        """Record the trade count at generation start so per-generation counts can be computed."""
        self._trades_at_generation_start = len(self.trades)

    def reset_prices(self) -> None:
        """Reset the GBM price simulator to initial prices and clear open positions.
        Call at the start of each generation to prevent unbounded price drift
        that causes phantom 100%+ take-profit alerts."""
        self.simulator.reset()
        self._current_prices = self.simulator.current_prices()
        # Clear all open positions so agents start fresh each generation
        for sym in self.positions:
            pos = self.positions[sym]
            pos.qty = 0.0
            pos.avg_cost = 0.0
            pos.unrealised_pnl = 0.0
        # Clear pending orders (filled/realised trades are kept for history)
        self.orders = [o for o in self.orders if o.status == "filled"]
        logger.info("PaperBroker: prices and positions reset for new generation")

    def per_generation_trades(self) -> int:
        return max(0, len(self.trades) - self._trades_at_generation_start)

    def trade_pnls(self) -> List[float]:
        return [t.pnl for t in self.trades]

    def recent_trades(self, n: int = 20) -> List[Dict]:
        return [
            {
                "trade_id": str(t.trade_id),
                "symbol":   t.symbol,
                "side":     t.side,
                "qty":      int(t.qty) if float(t.qty).is_integer() else float(t.qty),
                "price":    float(round(t.price, 4)),
                "pnl":      float(round(t.pnl, 2)),
                "commission": float(round(t.commission, 4)),
                "timestamp": float(t.timestamp),
            }
            for t in self.trades[-n:]
        ]
