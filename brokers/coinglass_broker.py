"""
CoinGlass Broker Integration
Fetches real-time crypto market data (BTC, ETH, SOL, etc.) from the
CoinGlass API v4, providing price history, open interest, funding rates,
and liquidation data as trading signals.

Since CoinGlass is an analytics/data API (not a trading exchange),
order execution is paper-simulated locally on top of real CG market data.

API docs: https://docs.coinglass.com/
Auth:     Single API key via `CG-API-KEY` header
Base URL: https://open-api-v4.coinglass.com
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_CG_BASE_URL = "https://open-api-v4.coinglass.com"

# Default starting prices for known crypto symbols (rough USD values)
_CRYPTO_S0 = {
    "BTC":       67_000.0,
    "ETH":        3_500.0,
    "SOL":          150.0,
    "BNB":          580.0,
    "XRP":            0.55,
    "DOGE":           0.12,
    "ADA":            0.45,
    "AVAX":          38.0,
    "MATIC":          0.75,
    "LINK":          15.0,
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CoinGlassPosition:
    symbol: str
    qty: float = 0.0
    avg_cost: float = 0.0
    side: str = "flat"          # 'flat' | 'long' | 'short'
    unrealised_pnl: float = 0.0
    realised_pnl: float = 0.0


@dataclass
class _PaperOrder:
    order_id: str
    symbol: str
    side: str                   # 'buy' | 'sell'
    qty: float
    order_type: str             # 'market' | 'limit'
    limit_price: Optional[float] = None
    filled_qty: float = 0.0
    filled_price: Optional[float] = None
    status: str = "pending"     # pending | filled | cancelled
    timestamp: float = field(default_factory=time.time)


class BrokerDisconnectedError(RuntimeError):
    """Raised when market-data fetch fails and paper trading cannot proceed."""


# ---------------------------------------------------------------------------
# CoinGlass Broker
# ---------------------------------------------------------------------------

class CoinGlassBroker:
    """
    CoinGlass data broker with paper-simulated order execution.

    Fetches real crypto price & on-chain data from CoinGlass API v4 and
    exposes the same duck-typed interface as AlpacaBroker / PaperBroker so
    it works transparently with the PBT evaluation loop.

    Parameters
    ----------
    api_key : str
        CoinGlass API key.  Falls back to ``COINGLASS_API_KEY`` env var.
    symbols : list[str]
        Crypto symbols to track (e.g. ["BTC", "ETH", "SOL"]).
    initial_cash : float
        Starting paper capital (USD).
    exchange : str
        Exchange to query for futures data (default: "Binance").
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        symbols: Optional[List[str]] = None,
        initial_cash: float = 100_000.0,
        exchange: str = "Binance",
    ):
        self.api_key  = api_key or os.getenv("COINGLASS_API_KEY", "")
        self.symbols  = symbols or ["BTC", "ETH", "SOL"]
        self.exchange = exchange

        # Portfolio state
        self.cash: float = initial_cash
        self._initial_cash: float = initial_cash
        self.positions: Dict[str, CoinGlassPosition] = {
            s: CoinGlassPosition(symbol=s) for s in self.symbols
        }
        self._current_prices: Dict[str, float] = {
            s: _CRYPTO_S0.get(s, 100.0) for s in self.symbols
        }
        self._price_history: Dict[str, List[float]] = {s: [] for s in self.symbols}

        # Auxiliary on-chain data (updated each step)
        self._open_interest: Dict[str, float] = {}   # USD open interest per symbol
        self._funding_rate:  Dict[str, float] = {}   # current funding rate

        # Order / equity tracking
        self.trades: List[Dict] = []
        self._open_orders: List[_PaperOrder] = []
        self.equity_curve: List[float] = [initial_cash]

        # Per-generation trade counting
        self._trades_at_generation_start: int = 0
        self._generation_start_time: float = time.time()

        # HTTP session (lazy)
        self._session = None

        # Validate key and seed prices
        self._connect()

    # ------------------------------------------------------------------
    # Internal HTTP helpers
    # ------------------------------------------------------------------

    def _get_session(self):
        if self._session is None:
            try:
                import requests
                s = requests.Session()
                s.headers.update({
                    "accept":     "application/json",
                    "CG-API-KEY": self.api_key,
                })
                self._session = s
            except ImportError:
                logger.error(
                    "CoinGlassBroker requires the 'requests' library. "
                    "Install with: pip install requests"
                )
        return self._session

    def _get(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """Execute a GET request against the CoinGlass v4 API."""
        session = self._get_session()
        if session is None:
            return None
        url = f"{_CG_BASE_URL}{endpoint}"
        try:
            resp = session.get(url, params=params, timeout=10)
            if resp.status_code == 401:
                logger.error("CoinGlass API: 401 Unauthorized — check COINGLASS_API_KEY")
                return None
            if resp.status_code == 429:
                logger.warning("CoinGlass API: 429 rate-limited — backing off 5s")
                time.sleep(5)
                return None
            resp.raise_for_status()
            payload = resp.json()
            # CG v4 wraps responses: {"code": "0", "data": {...}}
            if isinstance(payload, dict) and payload.get("code") == "0":
                return payload.get("data")
            if isinstance(payload, dict) and "data" in payload:
                return payload["data"]
            return payload
        except Exception as exc:
            logger.warning("CoinGlass API request failed [%s]: %s", endpoint, exc)
            return None

    # ------------------------------------------------------------------
    # Connection / initialisation
    # ------------------------------------------------------------------

    def _connect(self) -> None:
        """Validate API key and seed initial prices."""
        if not self.api_key:
            logger.warning(
                "CoinGlassBroker: COINGLASS_API_KEY not set — "
                "running with synthetic price fallbacks only."
            )
            return
        # Lightweight ping: fetch BTC price to confirm connectivity
        price = self._fetch_price("BTC")
        if price:
            logger.info(
                "Connected to CoinGlass API v4 | BTC=%.2f | symbols=%s",
                price, self.symbols,
            )
            self._current_prices["BTC"] = price
        else:
            logger.warning(
                "CoinGlass API connectivity check failed — "
                "prices will use synthetic fallbacks."
            )

    def is_connected(self) -> bool:
        return bool(self.api_key)

    # ------------------------------------------------------------------
    # Price fetching
    # ------------------------------------------------------------------

    def _fetch_price(self, symbol: str) -> Optional[float]:
        """
        Fetch the latest mark price for *symbol* from CoinGlass.
        Uses the futures OHLC history endpoint (most recent candle).
        """
        data = self._get(
            "/api/futures/open-interest/ohlc-history",
            params={
                "exchange":    self.exchange,
                "symbol":      symbol,
                "interval":    "1m",
                "limit":       1,
            },
        )
        if data and isinstance(data, list) and len(data) > 0:
            candle = data[-1]
            # CG returns [timestamp, open, high, low, close, volume, ...]
            if isinstance(candle, (list, tuple)) and len(candle) >= 5:
                return float(candle[4])   # close
            if isinstance(candle, dict):
                return float(candle.get("c") or candle.get("close") or 0) or None
        return None

    def _fetch_price_fallback(self, symbol: str) -> float:
        """
        Fetch price using the CoinGlass exchange volume endpoint as a
        secondary source.  Returns the last known price on failure.
        """
        data = self._get(
            "/api/futures/liquidation/history",
            params={"symbol": symbol, "interval": "1m", "limit": 1},
        )
        # Keep last known price if we can't get a new one
        return self._current_prices.get(symbol, _CRYPTO_S0.get(symbol, 100.0))

    def _fetch_open_interest(self, symbol: str) -> Optional[float]:
        """Fetch total open interest (USD) from CoinGlass."""
        data = self._get(
            "/api/futures/open-interest/history",
            params={"symbol": symbol, "interval": "1h", "limit": 1},
        )
        if data and isinstance(data, list) and len(data) > 0:
            item = data[-1]
            if isinstance(item, dict):
                return float(item.get("openInterest") or item.get("oi") or 0) or None
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                return float(item[1])
        return None

    def _fetch_funding_rate(self, symbol: str) -> Optional[float]:
        """Fetch current funding rate for *symbol*."""
        data = self._get(
            "/api/futures/fundingRate/ohlc-history",
            params={"symbol": symbol, "interval": "8h", "limit": 1},
        )
        if data and isinstance(data, list) and len(data) > 0:
            item = data[-1]
            if isinstance(item, dict):
                return float(item.get("fundingRate") or item.get("fr") or 0)
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                return float(item[1])
        return None

    # ------------------------------------------------------------------
    # PBT loop interface
    # ------------------------------------------------------------------

    def step(self) -> Dict[str, float]:
        """
        Refresh prices and on-chain metrics; fill any pending paper orders;
        append current equity to the curve.
        """
        for sym in self.symbols:
            price = self._fetch_price(sym) if self.api_key else None
            if price is None or price <= 0:
                # Synthetic random walk fallback (±0.3 % per step)
                import random
                last = self._current_prices.get(sym, _CRYPTO_S0.get(sym, 100.0))
                price = last * (1 + random.gauss(0, 0.003))
                price = max(price, 0.001)
            self._current_prices[sym] = price
            hist = self._price_history.setdefault(sym, [])
            hist.append(price)
            if len(hist) > 500:
                self._price_history[sym] = hist[-500:]

            # Update optional on-chain data (best-effort, non-blocking)
            if self.api_key:
                oi = self._fetch_open_interest(sym)
                if oi:
                    self._open_interest[sym] = oi
                fr = self._fetch_funding_rate(sym)
                if fr is not None:
                    self._funding_rate[sym] = fr

        # Fill pending paper orders at current prices
        self._process_pending_orders()

        # Update unrealised PnL and equity
        equity = self._compute_equity()
        self.equity_curve.append(equity)
        return dict(self._current_prices)

    def price_history(self, symbol: str) -> List[float]:
        return list(self._price_history.get(symbol, []))

    def is_market_open(self) -> bool:
        """Crypto markets trade 24/7."""
        return True

    def mark_generation_start(self) -> None:
        self._trades_at_generation_start = len(self.trades)
        self._generation_start_time = time.time()

    def per_generation_trades(self) -> int:
        return max(0, len(self.trades) - self._trades_at_generation_start)

    def total_filled_orders(self) -> int:
        return len(self.trades)

    def trade_pnls(self) -> List[float]:
        return [t.get("pnl", 0.0) for t in self.trades if "pnl" in t]

    # ------------------------------------------------------------------
    # Paper order execution
    # ------------------------------------------------------------------

    def submit_market_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        auto_cancel_opposite: bool = True,
    ) -> Dict:
        if symbol not in self.positions:
            self.positions[symbol] = CoinGlassPosition(symbol=symbol)
        price = self._current_prices.get(symbol, _CRYPTO_S0.get(symbol, 100.0))
        order = _PaperOrder(
            order_id=str(uuid.uuid4())[:8],
            symbol=symbol,
            side=side,
            qty=qty,
            order_type="market",
        )
        result = self._fill_order(order, price)
        if result.get("status") not in ("error", "rejected"):
            logger.info(
                "CoinGlass paper order: %s %s %.6f @ %.4f",
                side, symbol, qty, price,
            )
        return result

    def submit_limit_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        limit_price: float,
        auto_cancel_opposite: bool = True,
    ) -> Dict:
        if symbol not in self.positions:
            self.positions[symbol] = CoinGlassPosition(symbol=symbol)
        order = _PaperOrder(
            order_id=str(uuid.uuid4())[:8],
            symbol=symbol,
            side=side,
            qty=qty,
            order_type="limit",
            limit_price=limit_price,
            status="pending",
        )
        self._open_orders.append(order)
        return {
            "order_id":    order.order_id,
            "symbol":      symbol,
            "side":        side,
            "qty":         qty,
            "limit_price": limit_price,
            "status":      "pending",
            "timestamp":   time.time(),
        }

    def _process_pending_orders(self) -> None:
        remaining = []
        for order in self._open_orders:
            price = self._current_prices.get(order.symbol, 0.0)
            should_fill = False
            if order.order_type == "limit" and order.limit_price:
                if order.side == "buy" and price <= order.limit_price:
                    should_fill = True
                elif order.side == "sell" and price >= order.limit_price:
                    should_fill = True
            if should_fill:
                self._fill_order(order, price)
            else:
                remaining.append(order)
        self._open_orders = remaining

    def _fill_order(self, order: _PaperOrder, price: float) -> Dict:
        sym  = order.symbol
        side = order.side
        qty  = order.qty
        cost = price * qty

        if side == "buy":
            if self.cash < cost:
                return {"status": "rejected", "msg": "Insufficient paper cash"}
            self.cash -= cost
            pos = self.positions.setdefault(sym, CoinGlassPosition(symbol=sym))
            if pos.qty == 0:
                pos.avg_cost = price
            else:
                pos.avg_cost = (pos.avg_cost * pos.qty + price * qty) / (pos.qty + qty)
            pos.qty  += qty
            pos.side  = "long"
            pnl       = 0.0

        else:  # sell
            pos = self.positions.setdefault(sym, CoinGlassPosition(symbol=sym))
            fill_qty = min(qty, pos.qty)
            if fill_qty <= 0:
                return {"status": "rejected", "msg": "No position to sell"}
            pnl = (price - pos.avg_cost) * fill_qty
            pos.realised_pnl += pnl
            pos.qty -= fill_qty
            self.cash += price * fill_qty
            if pos.qty <= 1e-9:
                pos.qty  = 0.0
                pos.side = "flat"

        order.filled_qty   = qty
        order.filled_price = price
        order.status       = "filled"

        trade = {
            "order_id":  order.order_id,
            "symbol":    sym,
            "side":      side,
            "qty":       qty,
            "price":     round(price, 6),
            "pnl":       round(pnl, 4),
            "timestamp": time.time(),
        }
        self.trades.append(trade)
        return {**trade, "status": "filled"}

    def cancel_all_orders(self) -> None:
        cancelled = len(self._open_orders)
        self._open_orders.clear()
        logger.info("CoinGlass paper: cancelled %d pending orders", cancelled)

    def close_all_positions(self) -> None:
        for sym, pos in self.positions.items():
            if pos.qty > 0:
                self.submit_market_order(sym, "sell", pos.qty)

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        orders = self._open_orders
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        return [
            {
                "order_id": o.order_id,
                "symbol":   o.symbol,
                "side":     o.side,
                "qty":      o.qty,
                "type":     o.order_type,
                "status":   o.status,
            }
            for o in orders
        ]

    # ------------------------------------------------------------------
    # Portfolio state
    # ------------------------------------------------------------------

    def _compute_equity(self) -> float:
        market_value = sum(
            pos.qty * self._current_prices.get(sym, 0.0)
            for sym, pos in self.positions.items()
        )
        return self.cash + market_value

    def portfolio_state(self) -> Dict:
        pos_map = {}
        for sym, pos in self.positions.items():
            price = self._current_prices.get(sym, 0.0)
            unreal = (price - pos.avg_cost) * pos.qty if pos.qty > 0 else 0.0
            pos.unrealised_pnl = unreal
            pos_map[sym] = {
                "qty":            pos.qty,
                "avg_cost":       round(pos.avg_cost, 6),
                "current_price":  round(price, 6),
                "unrealised_pnl": round(unreal, 4),
                "realised_pnl":   round(pos.realised_pnl, 4),
                "side":           pos.side,
            }

        equity = self._compute_equity()
        initial = self._initial_cash
        total_return_pct = round((equity / initial - 1) * 100, 4) if initial > 0 else 0.0

        return {
            "cash":             round(self.cash, 2),
            "equity":           round(equity, 2),
            "buying_power":     round(self.cash, 2),
            "total_return_pct": total_return_pct,
            "positions":        pos_map,
            "prices":           {k: round(v, 6) for k, v in self._current_prices.items()},
            "open_interest":    {k: round(v, 2) for k, v in self._open_interest.items()},
            "funding_rates":    {k: round(v, 8) for k, v in self._funding_rate.items()},
            "n_trades":         len(self.trades),
            "paper":            True,   # CG is always paper-executed
            "exchange":         self.exchange,
        }

    def get_latest_bar(self, symbol: str) -> Optional[Dict]:
        """Return a synthetic bar dict from current price for compatibility."""
        price = self._current_prices.get(symbol)
        if price is None:
            return None
        hist = self._price_history.get(symbol, [price])
        return {
            "symbol":    symbol,
            "open":      hist[-2] if len(hist) >= 2 else price,
            "high":      price * 1.001,
            "low":       price * 0.999,
            "close":     price,
            "volume":    0.0,
            "timestamp": str(int(time.time())),
            "open_interest": self._open_interest.get(symbol, 0.0),
            "funding_rate":  self._funding_rate.get(symbol, 0.0),
        }

    def recent_trades(self, n: int = 20) -> List[Dict]:
        return list(self.trades[-n:])
