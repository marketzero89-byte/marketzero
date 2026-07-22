"""
Alpaca Live Broker Integration
REST API order submission, position tracking, portfolio state sync.
Requires: pip install alpaca-py
"""

from __future__ import annotations

import logging
import os
import socket
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Network reliability ────────────────────────────────────────────────────
# Alpaca-py sets no connect timeout by default, causing the evaluation loop
# to stall when data.alpaca.markets is slow or resets the connection.
# A module-level socket timeout (10 s connect + 15 s read) bounds every
# underlying urllib3 / httpx call made by the SDK without patching its internals.
_CONNECT_TIMEOUT = 10   # seconds — max time to establish a TCP connection
_READ_TIMEOUT    = 15   # seconds — max time to wait for the first byte
try:
    socket.setdefaulttimeout(_CONNECT_TIMEOUT + _READ_TIMEOUT)
except Exception:
    pass  # non-critical; best-effort


class BrokerDisconnectedError(RuntimeError):
    """Raised when live trading is attempted without a broker connection."""


@dataclass
class AlpacaPosition:
    symbol: str
    qty: float = 0.0
    avg_cost: float = 0.0


class AlpacaBroker:
    """
    Alpaca Markets broker integration (paper + live).

    Parameters
    ----------
    api_key : str       APCA_API_KEY_ID env var or explicit
    api_secret : str    APCA_API_SECRET_KEY env var or explicit
    paper : bool        True = paper trading endpoint
    symbols : list[str] Symbols to track
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        paper: bool = True,
        symbols: Optional[List[str]] = None,
    ):
        self.api_key    = api_key    or os.getenv("APCA_API_KEY_ID", "")
        self.api_secret = api_secret or os.getenv("APCA_API_SECRET_KEY", "")
        self.paper      = paper
        self.symbols    = symbols or ["AAPL"]
        self._client    = None
        self._trading   = None
        self._data_client = None
        self.equity_curve: List[float] = []
        self.trades: List[Dict] = []
        self.cash: float = 0.0
        self.positions: Dict[str, AlpacaPosition] = {
            s: AlpacaPosition(symbol=s) for s in self.symbols
        }
        self._current_prices: Dict[str, float] = {}
        self._price_history: Dict[str, List[float]] = {s: [] for s in self.symbols}
        self._trades_at_generation_start: int = 0  # for per-gen trade counting
        self._generation_start_time: float = time.time()  # timestamp of current gen start
        self._connect()

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def _connect(self) -> None:
        try:
            from alpaca.trading.client import TradingClient
            from alpaca.data.historical import StockHistoricalDataClient
            if not self.api_key or not self.api_secret:
                raise ValueError("Alpaca API credentials not configured")
            self._trading = TradingClient(
                self.api_key, self.api_secret, paper=self.paper
            )
            self._data_client = StockHistoricalDataClient(self.api_key, self.api_secret)
            account = self._trading.get_account()
            equity = float(account.equity)
            self.cash = float(account.cash)
            self.equity_curve.append(equity)
            mode = "PAPER" if self.paper else "LIVE"
            logger.info("Connected to Alpaca %s | equity=%.2f", mode, equity)
            self._sync_positions()
        except ImportError:
            logger.warning(
                "alpaca-py not installed. Run: pip install alpaca-py  "
                "AlpacaBroker will operate in stub mode (paper only)."
            )
        except Exception as exc:
            logger.error("Alpaca connection failed: %s", exc)

    def is_connected(self) -> bool:
        return self._trading is not None

    def _require_connection(self) -> None:
        if not self.is_connected():
            if self.paper:
                return
            raise BrokerDisconnectedError(
                "Alpaca broker is not connected — live trading halted"
            )

    # ------------------------------------------------------------------
    # Simulation step (PBT loop interface)
    # ------------------------------------------------------------------

    def step(self) -> Dict[str, float]:
        """Sync prices and portfolio state; append to equity curve."""
        state = self.portfolio_state()
        if state.get("status") in ("disconnected", "error"):
            if not self.paper:
                raise BrokerDisconnectedError(state.get("msg", "Alpaca disconnected"))
            return dict(self._current_prices)

        for sym in self.symbols:
            bar = self.get_latest_bar(sym)
            if bar:
                price = float(bar["close"])
                self._current_prices[sym] = price
                hist = self._price_history.setdefault(sym, [])
                hist.append(price)
                if len(hist) > 500:
                    self._price_history[sym] = hist[-500:]

        equity = float(state.get("equity", 0))
        if equity > 0:
            self.equity_curve.append(equity)
        return dict(self._current_prices)

    def price_history(self, symbol: str) -> List[float]:
        return list(self._price_history.get(symbol, []))

    def _sync_positions(self) -> None:
        if not self.is_connected():
            return
        try:
            for p in self._trading.get_all_positions():
                sym = p.symbol
                if sym not in self.positions:
                    self.positions[sym] = AlpacaPosition(symbol=sym)
                self.positions[sym].qty = float(p.qty)
                self.positions[sym].avg_cost = float(p.avg_entry_price)
        except Exception as exc:
            logger.error("Position sync failed: %s", exc)

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """Fetch open orders, optionally filtered by symbol."""
        if not self.is_connected():
            return []
        try:
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus
            req = GetOrdersRequest(
                status=QueryOrderStatus.OPEN,
                symbols=[symbol] if symbol else None,
            )
            orders = self._trading.get_orders(req)
            return [
                {
                    "order_id": str(o.id),
                    "symbol":   o.symbol,
                    "side":     str(o.side).split(".")[-1].lower(),
                    "qty":      float(o.qty) if o.qty else None,
                    "type":     str(o.order_type),
                    "status":   str(o.status),
                }
                for o in orders
            ]
        except Exception as exc:
            logger.error("Failed to fetch open orders for %s: %s", symbol, exc)
            return []

    def _guard_wash_trade(
        self, symbol: str, side: str, auto_cancel_opposite: bool
    ) -> Optional[Dict]:
        """
        R-071: Check for existing open orders on the opposite side of `symbol`
        before submitting, to avoid Alpaca's wash-trade rejection (40310000).

        Returns an error dict if the caller should abort submission, or None
        if it's safe to proceed (either no conflict, or conflict was cleared).
        """
        opposite = [
            o for o in self.get_open_orders(symbol) if o["side"] != side
        ]
        if not opposite:
            return None

        if not auto_cancel_opposite:
            o = opposite[0]
            logger.warning(
                "Wash-trade guard: existing %s order %s open on %s; "
                "skipping new %s order",
                o["side"], o["order_id"], symbol, side,
            )
            return {
                "status": "rejected",
                "msg": (
                    f"Wash-trade guard: existing {o['side']} order {o['order_id']} "
                    f"open on {symbol}; new {side} order not submitted"
                ),
                "conflicting_order_id": o["order_id"],
            }

        for o in opposite:
            try:
                self._trading.cancel_order_by_id(o["order_id"])
                logger.info(
                    "Wash-trade guard: cancelled conflicting %s order %s on %s "
                    "before submitting %s order",
                    o["side"], o["order_id"], symbol, side,
                )
            except Exception as exc:
                logger.error(
                    "Wash-trade guard: failed to cancel conflicting order %s: %s",
                    o["order_id"], exc,
                )
                return {
                    "status": "error",
                    "msg": (
                        f"Wash-trade guard: could not cancel conflicting order "
                        f"{o['order_id']} on {symbol}: {exc}"
                    ),
                }
        time.sleep(0.3)  # let the cancel propagate before resubmitting
        return None

    def submit_market_order(
        self, symbol: str, side: str, qty: float, auto_cancel_opposite: bool = True
    ) -> Dict:
        self._require_connection()
        qty = round(qty, 6)   # R-065: Alpaca requires max 6 decimal places
        if not self.is_connected():
            return self._stub_order(symbol, side, qty, "market")
        guard_result = self._guard_wash_trade(symbol, side, auto_cancel_opposite)
        if guard_result is not None:
            return guard_result
        try:
            from alpaca.trading.requests import MarketOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce
            req = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
            )
            order = self._trading.submit_order(req)
            result = {
                "order_id":  str(order.id),
                "symbol":    symbol,
                "side":      side,
                "qty":       qty,
                "status":    str(order.status),
                "timestamp": time.time(),
            }
            self.trades.append(result)
            self._sync_positions()
            logger.info("Market order submitted: %s %s %s", side, qty, symbol)
            return result
        except Exception as exc:
            logger.error("Order submission failed: %s", exc)
            return {"status": "error", "msg": str(exc)}

    def submit_limit_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        limit_price: float,
        auto_cancel_opposite: bool = True,
    ) -> Dict:
        self._require_connection()
        if not self.is_connected():
            return self._stub_order(symbol, side, qty, "limit", limit_price)
        guard_result = self._guard_wash_trade(symbol, side, auto_cancel_opposite)
        if guard_result is not None:
            return guard_result
        try:
            from alpaca.trading.requests import LimitOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce
            req = LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
                limit_price=limit_price,
            )
            order = self._trading.submit_order(req)
            result = {
                "order_id":    str(order.id),
                "symbol":      symbol,
                "side":        side,
                "qty":         qty,
                "limit_price": limit_price,
                "status":      str(order.status),
                "timestamp":   time.time(),
            }
            self.trades.append(result)  # count limit orders same as market orders
            self._sync_positions()
            logger.info("Limit order submitted: %s %s %s @ %.4f", side, qty, symbol, limit_price)
            return result
        except Exception as exc:
            logger.error("Limit order failed: %s", exc)
            return {"status": "error", "msg": str(exc)}

    def cancel_all_orders(self) -> None:
        if self.is_connected():
            try:
                self._trading.cancel_orders()
                logger.info("All orders cancelled")
            except Exception as exc:
                logger.error("Cancel orders failed: %s", exc)

    def close_all_positions(self) -> None:
        if self.is_connected():
            try:
                self._trading.close_all_positions(cancel_orders=True)
                logger.info("All positions closed")
                self._sync_positions()
            except Exception as exc:
                logger.error("Close positions failed: %s", exc)

    # ------------------------------------------------------------------
    # State sync
    # ------------------------------------------------------------------

    def portfolio_state(self) -> Dict:
        if not self.is_connected():
            return {"status": "disconnected"}
        try:
            account = self._trading.get_account()
            positions = self._trading.get_all_positions()
            equity = float(account.equity)
            last_equity = float(account.last_equity)
            self.cash = float(account.cash)
            pos_map = {}
            for p in positions:
                sym = p.symbol
                price = float(p.current_price)
                self._current_prices[sym] = price
                if sym not in self.positions:
                    self.positions[sym] = AlpacaPosition(symbol=sym)
                self.positions[sym].qty = float(p.qty)
                self.positions[sym].avg_cost = float(p.avg_entry_price)
                pos_map[sym] = {
                    "qty":            float(p.qty),
                    "avg_cost":       round(float(p.avg_entry_price), 4),
                    "current_price":  round(price, 4),
                    "unrealised_pnl": round(float(p.unrealized_pl), 2),
                    "realised_pnl":   round(float(getattr(p, 'realized_pl', 0) or 0), 2),
                }
            return {
                "cash":             round(self.cash, 2),
                "equity":           round(equity, 2),
                "buying_power":     round(float(account.buying_power), 2),
                "total_return_pct": round((equity / last_equity - 1) * 100, 4) if last_equity else 0.0,
                "daily_pnl":        round(equity - last_equity, 2),
                "positions":        pos_map,
                "prices":           {k: round(v, 4) for k, v in self._current_prices.items()},
                "n_trades":         len(self.trades),
                "paper":            self.paper,
            }
        except Exception as exc:
            logger.error("Portfolio state sync failed: %s", exc)
            return {"status": "error", "msg": str(exc)}

    def _fetch_latest_bar_once(self, symbol: str) -> Optional[Dict]:
        """Single attempt to fetch the most recent bar (no retry)."""
        from alpaca.data.requests import StockLatestBarRequest
        req = StockLatestBarRequest(symbol_or_symbols=symbol)
        bar = self._data_client.get_stock_latest_bar(req)[symbol]
        return {
            "symbol":    symbol,
            "open":      float(bar.open),
            "high":      float(bar.high),
            "low":       float(bar.low),
            "close":     float(bar.close),
            "volume":    float(bar.volume),
            "timestamp": str(bar.timestamp),
        }

    def get_latest_bar(
        self,
        symbol: str,
        retries: int = 3,
        backoff: float = 2.0,
    ) -> Optional[Dict]:
        """
        Fetch the most recent bar for *symbol* with retry + exponential backoff.

        On persistent failure the last cached price is returned as a synthetic
        bar so the evaluation loop never stalls or skips a step silently.

        Parameters
        ----------
        retries : int
            Maximum number of attempts before giving up (default 3).
        backoff : float
            Base seconds to wait between attempts; doubles each retry.
        """
        if not self.is_connected():
            return None

        last_exc: Optional[Exception] = None
        delay = backoff
        for attempt in range(1, retries + 1):
            try:
                return self._fetch_latest_bar_once(symbol)
            except Exception as exc:
                last_exc = exc
                if attempt < retries:
                    logger.warning(
                        "get_latest_bar %s attempt %d/%d failed (%s) — retrying in %.1fs",
                        symbol, attempt, retries, type(exc).__name__, delay,
                    )
                    time.sleep(delay)
                    delay *= 2.0   # exponential backoff

        # All retries exhausted — fall back to last known cached price
        cached = self._current_prices.get(symbol)
        if cached is not None:
            logger.warning(
                "get_latest_bar %s: all %d retries failed (%s) — using cached price %.4f",
                symbol, retries, type(last_exc).__name__, cached,
            )
            return {
                "symbol":    symbol,
                "open":      cached,
                "high":      cached,
                "low":       cached,
                "close":     cached,
                "volume":    0.0,
                "timestamp": str(int(time.time())),
                "stale":     True,
            }

        logger.error("Failed to fetch bar for %s: %s", symbol, last_exc)
        return None

    def is_market_open(self) -> bool:
        """Check if the US equity market is currently open."""
        if not self.is_connected():
            return False
        try:
            clock = self._trading.get_clock()
            return bool(clock.is_open)
        except Exception:
            return False

    def mark_generation_start(self) -> None:
        """Record generation boundary for both local and API-based trade counting."""
        self._trades_at_generation_start = len(self.trades)
        self._generation_start_time = time.time()

    def per_generation_trades(self) -> int:
        """Return the number of confirmed filled orders this generation.

        Uses the Alpaca API (filtered by generation start timestamp) to count
        actual fills rather than local submissions.  Falls back to local count
        if the API is unavailable.
        """
        if self.is_connected():
            try:
                import datetime as _dt
                from alpaca.trading.requests import GetOrdersRequest
                from alpaca.trading.enums import QueryOrderStatus
                after_dt = _dt.datetime.fromtimestamp(
                    self._generation_start_time, tz=_dt.timezone.utc
                )
                req = GetOrdersRequest(
                    status=QueryOrderStatus.CLOSED,
                    after=after_dt,
                    limit=500,
                )
                orders = self._trading.get_orders(req)
                return sum(
                    1 for o in orders
                    if o.filled_qty and float(o.filled_qty) > 0
                )
            except Exception as exc:
                logger.warning(
                    "per_generation_trades: API fetch failed, using local count: %s", exc
                )
        return max(0, len(self.trades) - self._trades_at_generation_start)

    def total_filled_orders(self) -> int:
        """Return the all-time total of filled orders from the Alpaca API.

        This is cross-session accurate — it queries Alpaca directly rather than
        relying on the in-memory ``self.trades`` list which resets each session.
        Falls back to ``len(self.trades)`` if the API is unavailable.
        """
        if self.is_connected():
            try:
                from alpaca.trading.requests import GetOrdersRequest
                from alpaca.trading.enums import QueryOrderStatus
                req = GetOrdersRequest(
                    status=QueryOrderStatus.CLOSED,
                    limit=500,
                )
                orders = self._trading.get_orders(req)
                return sum(1 for o in orders if o.filled_qty and float(o.filled_qty) > 0)
            except Exception as exc:
                logger.warning("total_filled_orders: API fetch failed, using local count: %s", exc)
        return len(self.trades)

    def trade_pnls(self) -> List[float]:
        return [t.get("pnl", 0.0) for t in self.trades if "pnl" in t]

    def recent_trades(self, n: int = 20) -> List[Dict]:
        """Return the n most recent filled orders.

        Attempts to fetch closed orders from the Alpaca API first (so fills that
        happened before this Python session are included).  Falls back to the
        local in-session ``self.trades`` list if the API call fails.
        """
        if self.is_connected():
            try:
                from alpaca.trading.requests import GetOrdersRequest
                from alpaca.trading.enums import QueryOrderStatus
                req = GetOrdersRequest(
                    status=QueryOrderStatus.CLOSED,
                    limit=n,
                )
                orders = self._trading.get_orders(req)
                result = []
                for o in orders:
                    side = str(o.side).split(".")[-1].lower()
                    qty = float(o.filled_qty) if o.filled_qty else (float(o.qty) if o.qty else 0.0)
                    price = float(o.filled_avg_price) if o.filled_avg_price else 0.0
                    submitted = o.submitted_at
                    filled = o.filled_at or submitted
                    holding_secs = 0.0
                    if submitted and filled:
                        try:
                            delta = filled - submitted
                            holding_secs = round(delta.total_seconds() / 3600, 4)
                        except Exception:
                            pass
                    result.append({
                        "trade_id":    str(o.id),
                        "symbol":      str(o.symbol),
                        "side":        side,
                        "qty":         qty,
                        "price":       round(price, 4),
                        "pnl":         0.0,  # Alpaca REST doesn't return realized PnL per order
                        "holding_time": holding_secs,
                        "timestamp":   filled.timestamp() if filled else time.time(),
                    })
                return result
            except Exception as exc:
                logger.warning("recent_trades: Alpaca API fetch failed, using local cache: %s", exc)
        # Fallback: in-session trades only
        return list(self.trades[-n:])

    # ------------------------------------------------------------------
    # Stub (paper dev mode only — never used for live)
    # ------------------------------------------------------------------

    def _stub_order(self, symbol, side, qty, order_type, limit_price=None) -> Dict:
        if not self.paper:
            raise BrokerDisconnectedError(
                f"Cannot submit {order_type} order in live mode without Alpaca connection"
            )
        logger.warning("AlpacaBroker stub: %s %s %s %s", order_type, side, qty, symbol)
        return {
            "order_id":  "stub",
            "symbol":    symbol,
            "side":      side,
            "qty":       qty,
            "order_type": order_type,
            "limit_price": limit_price,
            "status":    "stub_filled",
            "timestamp": time.time(),
        }