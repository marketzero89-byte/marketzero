# 07 — Market Simulation

---

## Paper Trading Mode

When `broker_mode = "paper"`, MarketZero uses an internal `PaperBroker` that simulates market prices and order execution without interacting with any real exchange. This mode is the default and is safe to run at any time without capital risk.

---

## Price Simulation

The paper broker generates synthetic prices using Geometric Brownian Motion (GBM):

```
S(t+1) = S(t) × exp((μ - 0.5σ²)Δt + σ√Δt × Z)

where:
  μ = drift parameter (per-symbol, configurable)
  σ = volatility parameter (per-symbol, configurable)
  Z ~ N(0, 1)
  Δt = time step size
```

Default parameters per symbol:

| Symbol | Drift (μ) | Volatility (σ) | Starting Price |
|---|---|---|---|
| AAPL | +0.0003 | 0.015 | 185.00 |
| SPY | +0.0002 | 0.010 | 520.00 |
| GLD | +0.0001 | 0.008 | 185.00 |
| BTC/USD | +0.0005 | 0.035 | 67,000.00 |

---

## Order Execution Simulation

The paper broker simulates realistic execution with:

- **Fill price**: last simulated price (no slippage model in v1.0)
- **Fill latency**: immediate (same step)
- **Partial fills**: not simulated (full fill assumed)
- **Commission**: zero (paper mode)

Order flow:

```
submit_order(symbol, qty, side, confidence)
        │
        ▼
RiskManager.validate()   ← position size, drawdown, daily trade count
        │
    Pass │  Fail
        │       └→ Order rejected, reason logged
        ▼
PaperBroker.fill()
        │
        ▼
PortfolioState updated (cash, positions, equity)
```

---

## Portfolio State

The `PortfolioState` object tracks:

```python
@dataclass
class PortfolioState:
    equity: float           # total market value of portfolio
    cash: float             # undeployed cash
    positions: dict         # {symbol: qty}
    total_trades: int       # cumulative trade count
    daily_pnl: float        # P&L since midnight
    total_return_pct: float # return since inception
```

---

## Backtesting Mode

The `backtest` command runs a synthetic equity curve evaluation:

```bash
python main.py backtest --trend bull    # μ = +0.0005, σ = 0.012
python main.py backtest --trend bear    # μ = -0.0003, σ = 0.012
python main.py backtest --trend ranging # μ = 0.0000,  σ = 0.012
```

252 daily steps are simulated and evaluated against all six fitness metrics. Results are printed to stdout. For programmatic use, see `BacktestEvaluator` in `trading/backtest_metrics.py`.

---

## Regime Simulation

The `RegimeDetector` can be seeded with a specific regime for testing:

```python
detector = RegimeDetector(window=20)
# Simulate bull market: feed rising prices
for i in range(30):
    state = detector.update("AAPL", 180.0 + i * 0.5)
assert state.regime == MarketRegime.BULL
```

---

## Live Mode

When `broker_mode = "live"`, the `PaperBroker` is replaced with the Alpaca broker client. The interface is identical — the executor does not know which broker is active. All risk controls remain in effect.

Requirements for live mode:
```bash
export ALPACA_API_KEY=your_key
export ALPACA_SECRET_KEY=your_secret
```

The Alpaca client targets the paper trading endpoint by default even in `broker_mode=live` unless the base URL is overridden to the live endpoint. Verify your Alpaca account type before deploying real capital.

---

## Extending the Simulator

To implement a custom price simulator (e.g., replay of historical data):

```python
class ReplayBroker:
    def __init__(self, ohlcv_df):
        self._df = ohlcv_df
        self._idx = 0

    def get_latest_price(self, symbol: str) -> float:
        price = float(self._df.iloc[self._idx]['close'])
        self._idx = (self._idx + 1) % len(self._df)
        return price

    # implement remaining interface methods ...
```
