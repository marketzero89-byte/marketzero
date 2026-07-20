"""
Feature Engineering
Technical indicators, alternative data (sentiment/options/short interest),
cross-asset features (VIX, yield curve, DXY), microstructure, macro calendar.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Technical features
# ---------------------------------------------------------------------------

def returns(prices: List[float], n: int = 1) -> List[float]:
    arr = np.array(prices)
    r = np.diff(arr) / np.where(arr[:-1] != 0, arr[:-1], 1e-10)
    return list(r[-n:]) if n > 0 else list(r)


def ema(prices: List[float], span: int) -> float:
    arr = np.array(prices, dtype=float)
    alpha = 2.0 / (span + 1)
    e = arr[0]
    for p in arr[1:]:
        e = alpha * p + (1 - alpha) * e
    return float(e)


def rsi(prices: List[float], period: int = 14) -> float:
    r = np.diff(np.array(prices, dtype=float))
    if len(r) < period:
        return 50.0
    gains = np.where(r > 0, r, 0)
    losses = np.where(r < 0, -r, 0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    rs = avg_gain / (avg_loss + 1e-10)
    return float(100 - 100 / (1 + rs))


def bollinger_bands(prices: List[float], period: int = 20, n_std: float = 2.0):
    arr = np.array(prices[-period:], dtype=float)
    mid = float(np.mean(arr))
    std = float(np.std(arr))
    return mid - n_std * std, mid, mid + n_std * std


def atr(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> float:
    trs = []
    for i in range(1, min(len(highs), len(lows), len(closes))):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        trs.append(tr)
    if not trs:
        return 0.0
    return float(np.mean(trs[-period:]))


def macd(prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
    e_fast   = ema(prices, fast)
    e_slow   = ema(prices, slow)
    macd_val = e_fast - e_slow
    signal_val = ema(prices[-signal:] if len(prices) >= signal else prices, signal)
    return float(macd_val), float(signal_val), float(macd_val - signal_val)


def realised_vol(prices: List[float], window: int = 20) -> float:
    r = returns(prices)
    tail = r[-window:] if len(r) >= window else r
    return float(np.std(tail) * np.sqrt(252)) if tail else 0.0


# ---------------------------------------------------------------------------
# State vector builder
# ---------------------------------------------------------------------------

class FeatureBuilder:
    """
    Converts raw OHLCV data and optional alt-data into a normalised state vector
    ready for agent consumption.

    v1.0 spec (R-014 / R-015):
        state = [price_history / current_price (length=lookback),
                 RSI/100, BB_pos, ATR/price,
                 regime_one_hot (5 elements)]
        state_dim = lookback + 8

    Parameters
    ----------
    lookback : int   Number of past bars to include in price history
    include_alt_data : bool  (used by build_full only)
    include_macro : bool     (used by build_full only)
    """

    # Regime one-hot order (R-015)
    _REGIMES = ["BULL", "BEAR", "RANGING", "HIGH_VOL", "UNKNOWN"]

    def __init__(
        self,
        lookback: int = 20,
        include_alt_data: bool = False,
        include_macro: bool = False,
    ):
        self.lookback        = lookback
        self.include_alt_data= include_alt_data
        self.include_macro   = include_macro

    # ------------------------------------------------------------------
    # v1.0 compliant build (R-014 / R-015)
    # ------------------------------------------------------------------

    def build(
        self,
        closes:   List[float],
        highs:    Optional[List[float]] = None,
        lows:     Optional[List[float]]  = None,
        volumes:  Optional[List[float]]  = None,
        position: float = 0.0,
        cash_pct: float = 1.0,
        regime:   Optional[str] = None,
        alt_data: Optional[Dict] = None,
        macro:    Optional[Dict] = None,
    ) -> np.ndarray:
        """
        Build the spec-compliant lookback+8 feature vector (R-014/R-015).

        Returns
        -------
        np.ndarray  shape (lookback+8,)  i.e. (state_dim,)
        """
        if len(closes) < 5:
            return np.zeros(self.state_dim, dtype=np.float32)

        highs   = highs or closes
        lows    = lows  or closes

        price = closes[-1]

        # 1. Normalised price history (length = lookback), each / current_price (R-014)
        hist = closes[-self.lookback:] if len(closes) >= self.lookback else closes
        # Pad left with first value if shorter than lookback
        pad = [hist[0]] * (self.lookback - len(hist))
        hist = pad + list(hist)
        price_hist = np.array(hist, dtype=np.float32) / (price + 1e-10)   # normalise by current

        # 2. RSI / 100  (R-014)
        rsi_val = rsi(closes, 14) / 100.0

        # 3. Bollinger position  (R-014)
        bb_lo, bb_mid, bb_hi = bollinger_bands(closes, 20)
        bb_pos = float((price - bb_mid) / (bb_hi - bb_lo + 1e-10))

        # 4. ATR / price  (R-014, R-031)
        atr_val = atr(highs, lows, closes, 14) / (price + 1e-10)

        # 5. Regime one-hot (5 elements)  (R-015)
        regime_oh = np.zeros(5, dtype=np.float32)
        if regime and regime.upper() in self._REGIMES:
            regime_oh[self._REGIMES.index(regime.upper())] = 1.0
        else:
            regime_oh[4] = 1.0   # UNKNOWN

        features = np.concatenate([
            price_hist,
            [rsi_val, bb_pos, atr_val],
            regime_oh,
        ]).astype(np.float32)

        return np.clip(features, -5.0, 5.0)

    # ------------------------------------------------------------------
    # Extended build (v1.2 richer feature set — not used in v1.0 run loop)
    # ------------------------------------------------------------------

    def build_full(
        self,
        closes:   List[float],
        highs:    Optional[List[float]] = None,
        lows:     Optional[List[float]]  = None,
        volumes:  Optional[List[float]]  = None,
        alt_data: Optional[Dict]         = None,
        macro:    Optional[Dict]         = None,
        position: float = 0.0,
        cash_pct: float = 1.0,
    ) -> np.ndarray:
        """Extended 12-element vector (backward-compat; used for dashboard/logging)."""
        if len(closes) < 5:
            return np.zeros(self.state_dim_full, dtype=np.float32)

        highs   = highs   or closes
        lows    = lows    or closes
        volumes = volumes or [1.0] * len(closes)

        price = closes[-1]

        ret_1  = (closes[-1] / closes[-2] - 1) if len(closes) >= 2 else 0.0
        ret_5  = (closes[-1] / closes[-6] - 1) if len(closes) >= 6 else 0.0
        ret_20 = (closes[-1] / closes[-21] - 1) if len(closes) >= 21 else 0.0

        rsi_val = rsi(closes, 14) / 100.0 - 0.5
        vol     = realised_vol(closes, 20)
        macd_v, macd_sig, macd_hist = macd(closes)

        ema_fast = ema(closes, 12)
        ema_slow = ema(closes, 26)
        ema_ratio = (ema_fast - ema_slow) / (ema_slow + 1e-10)

        bb_lo, bb_mid, bb_hi = bollinger_bands(closes, 20)
        bb_pos = (price - bb_mid) / (bb_hi - bb_lo + 1e-10)

        vol_ratio = (volumes[-1] / np.mean(volumes[-20:]) - 1) if len(volumes) >= 2 else 0.0
        atr_val = atr(highs, lows, closes, 14) / (price + 1e-10)

        features = [
            ret_1, ret_5, ret_20,
            rsi_val, vol,
            macd_hist / (price + 1e-10),
            ema_ratio, bb_pos, vol_ratio, atr_val,
            position, cash_pct,
        ]

        if self.include_alt_data and alt_data:
            features.extend([
                float(alt_data.get("sentiment", 0.0)),
                float(alt_data.get("options_flow", 0.0)),
                float(alt_data.get("short_interest", 0.0)),
            ])
        if self.include_macro and macro:
            features.extend([
                float(macro.get("vix", 20.0)) / 100.0,
                float(macro.get("yield_slope", 0.01)),
                float(macro.get("dxy", 0.0)),
            ])

        arr = np.array(features, dtype=np.float32)
        return np.clip(arr, -5.0, 5.0)

    # ------------------------------------------------------------------
    # Dimensions
    # ------------------------------------------------------------------

    @property
    def state_dim(self) -> int:
        """v1.0 spec dimension: lookback + RSI + BB_pos + ATR/price + regime_onehot(5) = lookback + 8"""
        return self.lookback + 8

    @property
    def state_dim_full(self) -> int:
        """Extended feature set dimension."""
        dim = 12
        if self.include_alt_data:
            dim += 3
        if self.include_macro:
            dim += 3
        return dim



# ---------------------------------------------------------------------------
# Macro calendar (event-awareness)
# ---------------------------------------------------------------------------

FOMC_DATES_2026 = [
    date(2026, 1, 28), date(2026, 3, 18), date(2026, 5, 6),
    date(2026, 6, 17), date(2026, 7, 29), date(2026, 9, 16),
    date(2026, 11, 4), date(2026, 12, 16),
]

CPI_DATES_2026 = [
    date(2026, 1, 14), date(2026, 2, 11), date(2026, 3, 11),
    date(2026, 4, 10), date(2026, 5, 12), date(2026, 6, 11),
    date(2026, 7, 14), date(2026, 8, 12), date(2026, 9, 11),
    date(2026, 10, 13), date(2026, 11, 12), date(2026, 12, 11),
]


def days_to_next_event(event_dates: List[date], today: Optional[date] = None) -> int:
    today = today or date.today()
    future = [d for d in event_dates if d >= today]
    if not future:
        return 999
    return (min(future) - today).days


def macro_features(today: Optional[date] = None) -> Dict[str, float]:
    today = today or date.today()
    return {
        "days_to_fomc":    days_to_next_event(FOMC_DATES_2026, today),
        "days_to_cpi":     days_to_next_event(CPI_DATES_2026,  today),
        "day_of_week":     today.weekday() / 4.0,   # 0=Mon, 1=Fri
        "month":           today.month / 12.0,
    }


# ---------------------------------------------------------------------------
# Microstructure features (requires Level 2 data)
# ---------------------------------------------------------------------------

def bid_ask_spread(bid: float, ask: float) -> float:
    mid = (bid + ask) / 2.0
    return (ask - bid) / (mid + 1e-10)


def order_imbalance(bid_size: float, ask_size: float) -> float:
    total = bid_size + ask_size
    return (bid_size - ask_size) / (total + 1e-10)
