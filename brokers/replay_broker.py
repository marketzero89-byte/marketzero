"""
Historical Replay Broker
Replays OHLCV data from a CSV/JSON file for walk-forward backtesting.
"""

from __future__ import annotations

import csv
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Bar:
    timestamp: str
    symbol: str
    open:   float
    high:   float
    low:    float
    close:  float
    volume: float


class HistoricalReplayBroker:
    """
    Backtesting broker that replays OHLCV bars from a file.

    CSV format expected: timestamp,symbol,open,high,low,close,volume
    JSON format expected: list of {timestamp, symbol, open, high, low, close, volume}

    Parameters
    ----------
    data_path : str | Path
    initial_cash : float
    commission_pct : float
    slippage_pct : float
    """

    def __init__(
        self,
        data_path: str | Path,
        initial_cash: float = 100_000.0,
        commission_pct: float = 0.001,
        slippage_pct: float = 0.0005,
    ):
        self.data_path = Path(data_path)
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct

        self._bars: List[Bar] = self._load_bars()
        self._idx: int = 0
        self._current_bar: Optional[Bar] = None
        self.positions: Dict[str, float] = {}     # symbol -> qty
        self.avg_costs: Dict[str, float] = {}
        self.realised_pnl: Dict[str, float] = {}
        self.equity_curve: List[float] = [initial_cash]
        self.trades: List[Dict] = []

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_bars(self) -> List[Bar]:
        suffix = self.data_path.suffix.lower()
        bars: List[Bar] = []

        if suffix == ".csv":
            with self.data_path.open() as f:
                reader = csv.DictReader(f)
                for row in reader:
                    bars.append(Bar(
                        timestamp=row.get("timestamp", ""),
                        symbol=row.get("symbol", "UNKNOWN"),
                        open=float(row["open"]),
                        high=float(row["high"]),
                        low=float(row["low"]),
                        close=float(row["close"]),
                        volume=float(row.get("volume", 0)),
                    ))
        elif suffix in (".json", ".jsonl"):
            with self.data_path.open() as f:
                if suffix == ".jsonl":
                    raw = [json.loads(line) for line in f if line.strip()]
                else:
                    raw = json.load(f)
            for row in raw:
                bars.append(Bar(
                    timestamp=str(row.get("timestamp", "")),
                    symbol=str(row.get("symbol", "UNKNOWN")),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row.get("volume", 0)),
                ))
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

        logger.info("Loaded %d bars from %s", len(bars), self.data_path)
        return bars

    # ------------------------------------------------------------------
    # Replay interface
    # ------------------------------------------------------------------

    def has_next(self) -> bool:
        return self._idx < len(self._bars)

    def step(self) -> Optional[Bar]:
        if not self.has_next():
            return None
        self._current_bar = self._bars[self._idx]
        self._idx += 1
        self._update_equity()
        return self._current_bar

    def reset(self) -> None:
        self._idx = 0
        self.cash = self.initial_cash
        self.positions.clear()
        self.avg_costs.clear()
        self.realised_pnl.clear()
        self.equity_curve = [self.initial_cash]
        self.trades.clear()

    def bars_remaining(self) -> int:
        return len(self._bars) - self._idx

    def progress(self) -> float:
        return self._idx / max(len(self._bars), 1)

    def current_price(self) -> float:
        return self._current_bar.close if self._current_bar else 0.0

    def current_bar(self) -> Optional[Bar]:
        return self._current_bar

    def price_history(self, symbol: str, n: int = 100) -> List[float]:
        start = max(0, self._idx - n)
        return [b.close for b in self._bars[start:self._idx] if b.symbol == symbol]

    # ------------------------------------------------------------------
    # Order execution (on next bar's open to avoid look-ahead)
    # ------------------------------------------------------------------

    def _apply_slippage(self, price: float, side: str) -> float:
        slip = price * self.slippage_pct
        return price + slip if side == "buy" else price - slip

    def submit_market_order(self, symbol: str, side: str, qty: float) -> Dict:
        if self._current_bar is None:
            return {"status": "error", "msg": "no bar loaded"}

        # Simulate fill at current bar close (or next open in a stricter model)
        price = self._apply_slippage(self._current_bar.close, side)
        commission = price * qty * self.commission_pct

        if side == "buy":
            cost = price * qty + commission
            if self.cash < cost:
                qty = int(self.cash / (price * (1 + self.commission_pct)))
                if qty <= 0:
                    return {"status": "rejected", "msg": "insufficient cash"}
                cost = price * qty + commission
            self.cash -= cost
            prev_qty  = self.positions.get(symbol, 0)
            prev_cost = self.avg_costs.get(symbol, 0)
            new_qty   = prev_qty + qty
            self.avg_costs[symbol] = (prev_cost * prev_qty + price * qty) / new_qty
            self.positions[symbol] = new_qty
            pnl = 0.0
        else:
            avail = self.positions.get(symbol, 0)
            qty   = min(qty, avail)
            if qty <= 0:
                return {"status": "rejected", "msg": "no position to sell"}
            avg  = self.avg_costs.get(symbol, price)
            pnl  = (price - avg) * qty - commission
            self.cash += price * qty - commission
            self.positions[symbol] = avail - qty
            self.realised_pnl[symbol] = self.realised_pnl.get(symbol, 0) + pnl

        trade = {
            "trade_id":  str(uuid.uuid4())[:8],
            "symbol":    symbol,
            "side":      side,
            "qty":       qty,
            "price":     round(price, 4),
            "pnl":       round(pnl, 2),
            "commission": round(commission, 4),
            "timestamp": self._current_bar.timestamp,
        }
        self.trades.append(trade)
        return {**trade, "status": "filled"}

    def _update_equity(self) -> None:
        if self._current_bar:
            mkt_val = sum(
                qty * (self._current_bar.close if self._current_bar.symbol == sym
                       else self.avg_costs.get(sym, 0))
                for sym, qty in self.positions.items()
            )
            self.equity_curve.append(self.cash + mkt_val)

    def portfolio_state(self) -> Dict:
        equity = self.equity_curve[-1] if self.equity_curve else self.initial_cash
        return {
            "cash":              round(self.cash, 2),
            "equity":            round(equity, 2),
            "total_return_pct":  round((equity / self.initial_cash - 1) * 100, 2),
            "positions":         dict(self.positions),
            "n_trades":          len(self.trades),
            "progress":          round(self.progress() * 100, 1),
        }

    def trade_pnls(self) -> List[float]:
        return [t["pnl"] for t in self.trades]
