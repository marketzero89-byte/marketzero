"""
OHLCV data loading and optional download for backtesting.

R-020 Dataset Naming Convention
---------------------------------
Files dropped into the data/ directory should follow:

    {SYMBOL}_{PERIOD}.csv          e.g.  AAPL_1y.csv  /  SPY_504d.csv
    {SYMBOL}_{YYYYMMDD}_{YYYYMMDD}.csv   e.g.  AAPL_20230101_20241231.csv

Required CSV columns (case-insensitive):
    timestamp | date  — ISO-8601 date string  (YYYY-MM-DD)
    symbol             — ticker, e.g. "AAPL"
    open, high, low, close — float, USD
    volume             — integer

The bundled sample follows this convention: data/AAPL_sample.csv
"""

from __future__ import annotations

import csv
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def load_closes_from_file(
    data_path: str | Path,
    symbol: Optional[str] = None,
) -> tuple[List[float], str]:
    """
    Load closing prices from CSV/JSON OHLCV file.
    Returns (closes, resolved_symbol).
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    suffix = path.suffix.lower()
    rows: list[dict] = []

    if suffix == ".csv":
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                rows.append(row)
    elif suffix in (".json", ".jsonl"):
        import json
        with path.open() as f:
            if suffix == ".jsonl":
                raw = [json.loads(line) for line in f if line.strip()]
            else:
                raw = json.load(f)
        rows = [dict(r) for r in raw]
    else:
        raise ValueError(f"Unsupported format: {suffix}")

    if symbol:
        rows = [r for r in rows if r.get("symbol", "").upper() == symbol.upper()]
    elif rows:
        sym = rows[0].get("symbol", "UNKNOWN")
        rows = [r for r in rows if r.get("symbol", sym) == sym]
        symbol = sym

    closes = [float(r["close"]) for r in rows]
    if len(closes) < 50:
        raise ValueError(f"Insufficient bars ({len(closes)}) — need at least 50")

    logger.info("Loaded %d closes for %s from %s", len(closes), symbol, path.name)
    return closes, str(symbol or "UNKNOWN")


def generate_sample_ohlcv(
    output_path: str | Path,
    symbol: str = "AAPL",
    n_bars: int = 504,
    seed: int = 42,
) -> Path:
    """
    Generate a realistic sample OHLCV CSV for backtesting (2 years daily bars).
    Uses GBM with AAPL-like starting price.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    price = 150.0
    mu, sigma, dt = 0.10, 0.25, 1 / 252
    start = datetime(2024, 1, 2)

    with out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "symbol", "open", "high", "low", "close", "volume"])
        for i in range(n_bars):
            z = rng.standard_normal()
            ret = mu * dt + sigma * np.sqrt(dt) * z
            open_p = price
            close_p = max(price * (1 + ret), 1.0)
            wick = abs(close_p - open_p) * (0.5 + rng.random())
            high_p = max(open_p, close_p) + wick
            low_p = min(open_p, close_p) - wick * 0.8
            vol = int(rng.integers(20_000_000, 80_000_000))
            ts = (start + timedelta(days=i)).strftime("%Y-%m-%d")
            writer.writerow([
                ts, symbol,
                round(open_p, 2), round(high_p, 2),
                round(low_p, 2), round(close_p, 2), vol,
            ])
            price = close_p

    logger.info("Generated %d bars → %s", n_bars, out)
    return out


def fetch_ohlcv_to_csv(
    symbol: str,
    days: int = 365,
    output_path: str | Path | None = None,
) -> Path:
    """
    Download daily OHLCV via yfinance and save as CSV.
    Requires: pip install yfinance
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError(
            "yfinance required for download. Install: pip install yfinance"
        ) from exc

    out = Path(output_path or f"data/{symbol.upper()}_{days}d.csv")
    out.parent.mkdir(parents=True, exist_ok=True)

    ticker = yf.Ticker(symbol)
    hist = ticker.history(period=f"{days}d", interval="1d")
    if hist.empty:
        raise ValueError(f"No data returned for {symbol}")

    with out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "symbol", "open", "high", "low", "close", "volume"])
        for ts, row in hist.iterrows():
            writer.writerow([
                ts.strftime("%Y-%m-%d"),
                symbol.upper(),
                round(float(row["Open"]), 2),
                round(float(row["High"]), 2),
                round(float(row["Low"]), 2),
                round(float(row["Close"]), 2),
                int(row["Volume"]),
            ])

    logger.info("Downloaded %d bars for %s → %s", len(hist), symbol, out)
    return out


def default_sample_path(data_dir: str | Path = "data") -> Path:
    return Path(data_dir) / "AAPL_sample.csv"


def ensure_sample_data(data_dir: str | Path = "data") -> Path:
    """Create sample OHLCV if missing."""
    path = default_sample_path(data_dir)
    if not path.exists():
        generate_sample_ohlcv(path)
    return path
