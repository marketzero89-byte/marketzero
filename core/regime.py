"""
Regime Detector
EMA crossover + volatility classification → BULL, BEAR, RANGING, HIGH_VOL, UNKNOWN.
Supports both rule-based (v1.x) and learned (v1.2+) approaches.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class Regime(str, Enum):
    BULL     = "BULL"
    BEAR     = "BEAR"
    RANGING  = "RANGING"
    HIGH_VOL = "HIGH_VOL"
    UNKNOWN  = "UNKNOWN"


def _ema(prices: np.ndarray, span: int) -> np.ndarray:
    """Exponential moving average."""
    alpha = 2.0 / (span + 1)
    result = np.zeros_like(prices, dtype=float)
    result[0] = prices[0]
    for i in range(1, len(prices)):
        result[i] = alpha * prices[i] + (1 - alpha) * result[i - 1]
    return result


def _realised_vol(returns: np.ndarray, window: int) -> float:
    """Annualised rolling realised volatility (last `window` returns)."""
    tail = returns[-window:] if len(returns) >= window else returns
    return float(np.std(tail) * np.sqrt(252))


class RegimeDetector:
    """
    Rule-based regime classifier.

    Parameters
    ----------
    fast_span : int   EMA fast period (default 20)
    slow_span : int   EMA slow period (default 50)
    vol_window : int  Lookback for volatility estimation
    high_vol_threshold : float  Annualised vol above which HIGH_VOL fires
    trend_threshold : float     Minimum EMA spread (% of price) for trend
    """

    def __init__(
        self,
        fast_span: int = 20,
        slow_span: int = 50,
        vol_window: int = 20,
        high_vol_threshold: float = 0.35,
        trend_threshold: float = 0.005,
    ):
        self.fast_span = fast_span
        self.slow_span = slow_span
        self.vol_window = vol_window
        self.high_vol_threshold = high_vol_threshold
        self.trend_threshold = trend_threshold

        self._current: Regime = Regime.UNKNOWN
        self._history: List[Regime] = []

    # ------------------------------------------------------------------

    def detect(self, prices: List[float]) -> Regime:
        """
        Classify the current market regime from a price series.

        Parameters
        ----------
        prices : list[float]   Closing prices (oldest → newest)

        Returns
        -------
        Regime
        """
        arr = np.array(prices, dtype=float)
        if len(arr) < self.slow_span + 2:
            self._current = Regime.UNKNOWN
            return self._current

        fast_ema = _ema(arr, self.fast_span)
        slow_ema = _ema(arr, self.slow_span)
        rets = np.diff(arr) / np.where(arr[:-1] != 0, arr[:-1], 1e-10)
        vol = _realised_vol(rets, self.vol_window)

        # HIGH_VOL takes priority
        if vol >= self.high_vol_threshold:
            regime = Regime.HIGH_VOL
        else:
            spread = (fast_ema[-1] - slow_ema[-1]) / (slow_ema[-1] or 1)
            if spread > self.trend_threshold:
                regime = Regime.BULL
            elif spread < -self.trend_threshold:
                regime = Regime.BEAR
            else:
                regime = Regime.RANGING

        self._current = regime
        self._history.append(regime)
        logger.debug(
            "Regime: %s  vol=%.2f%%  ema_spread=%.4f",
            regime.value, vol * 100,
            (fast_ema[-1] - slow_ema[-1]) / (slow_ema[-1] or 1),
        )
        return regime

    @property
    def current(self) -> Regime:
        return self._current

    @property
    def history(self) -> List[Regime]:
        return list(self._history)

    def regime_duration(self) -> int:
        """How many consecutive steps has the current regime persisted?"""
        if not self._history:
            return 0
        count = 0
        for r in reversed(self._history):
            if r == self._current:
                count += 1
            else:
                break
        return count

    def to_dict(self) -> dict:
        return {
            "current": self._current.value,
            "duration_bars": self.regime_duration(),
            "history_tail": [r.value for r in self._history[-10:]],
        }


# ---------------------------------------------------------------------------
# Learned regime classifier (v1.2+ placeholder)
# ---------------------------------------------------------------------------

class LearnedRegimeDetector:
    """
    Placeholder for a trainable regime classifier (v1.2).
    Falls back to rule-based until a model is fitted.
    """

    def __init__(self, fallback: Optional[RegimeDetector] = None):
        self._fallback = fallback or RegimeDetector()
        self._model = None   # sklearn / PyTorch model slot

    def fit(self, price_matrix: np.ndarray, labels: np.ndarray) -> None:
        """Train a simple logistic classifier on features."""
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            features = self._extract_features(price_matrix)
            self._scaler = StandardScaler().fit(features)
            self._model = LogisticRegression(max_iter=500)
            self._model.fit(self._scaler.transform(features), labels)
            logger.info("LearnedRegimeDetector fitted on %d samples", len(labels))
        except ImportError:
            logger.warning("scikit-learn not installed; LearnedRegimeDetector unavailable")

    def _extract_features(self, prices: np.ndarray) -> np.ndarray:
        """Return a feature matrix from price windows."""
        rets = np.diff(prices, axis=-1) / (prices[..., :-1] + 1e-10)
        return np.column_stack([
            np.mean(rets, axis=-1),
            np.std(rets, axis=-1),
            np.min(rets, axis=-1),
            np.max(rets, axis=-1),
        ])

    def detect(self, prices: List[float]) -> Regime:
        arr = np.array(prices, dtype=float)
        if self._model is None or len(arr) < 50:
            return self._fallback.detect(prices)
        feats = self._extract_features(arr.reshape(1, -1))
        label = self._model.predict(self._scaler.transform(feats))[0]
        return Regime(label)
