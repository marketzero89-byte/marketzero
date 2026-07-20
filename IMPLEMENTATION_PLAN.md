# MarketZero Compliance Report — Implementation Plan

This plan implements fixes from `MARKETZERO_COMPLIANCE_REPORT.md`, working through all **CRITICAL → HIGH → MEDIUM → LOW** items.
The codebase has already been substantially refactored from the original MarketZero into the `pbt_trading` package layout (new module paths, renamed files), so each fix is mapped to the **current** file that owns the behaviour.

---

## User Review Required

> [!CAUTION]
> A live `python main.py run --broker paper --generations 50` process is running. The fixes below do NOT touch the PBT engine hot-path in a way that would corrupt an in-flight session (no checkpoint format changes that break reading). However the process must be restarted afterward to pick up any agent/feature changes.

> [!WARNING]
> **R-014 / R-015 — observation vector reshape** is the highest-impact change. All agent weight matrices are currently sized for a 5-element feature vector. Rebuilding `_build_features()` to return `lookback+8` elements requires agents created with the new `FeatureBuilder` to be re-initialised. Old checkpoint hyperparams (lookback value) will still load correctly; only the in-memory weight tensors change on first instantiation.

> [!IMPORTANT]
> The report is based on the original `pbt/` flat-file layout. This codebase uses a refactored package structure (`agents/`, `core/`, `risk/`, `signals/`, `brokers/`). Every fix is mapped to the new location.

---

## Open Questions

None — all items have clear deterministic fixes derived directly from the compliance report.

---

## Proposed Changes

### CRITICAL fixes

---

#### [MODIFY] [main.py](file:///c:/Users/user/pbt_trading/main.py) — `_build_broker` GBM initial prices (R-012)

Set per-symbol GBM S₀ to documented values: AAPL=185, SPY=520, GLD=185, BTC=67000.

---

### Risk module

#### [MODIFY] [risk/risk_manager.py](file:///c:/Users/user/pbt_trading/risk/risk_manager.py)

- **R-054**: Add 5% minimum cash buffer check in `check_order()`
- **R-058**: Reorder checks: (1) position-size, (2) daily loss, (3) drawdown, (4) trade count, (5) exposure — move daily-loss/drawdown/trade-count from `update()` into `check_order()` and call `_check_circuit_breakers()` from `check_order()` as well
- **R-059**: Add `daily_pnl_pct` to `status()` return dict
- **R-060**: Position size exceeded → reduce qty instead of rejecting
- **R-062**: `update()` already resets daily counters on date change ✅ — verify it's called from the run loop (it is, via `risk_manager.update()` in `main.py L261`)

---

### Signals module

#### [MODIFY] [signals/aggregator.py](file:///c:/Users/user/pbt_trading/signals/aggregator.py)

- **R-050**: Fix direction aggregation formula: numerator must include per-agent confidence weight (`Σ w_i × dir_i × conf_i / Σ w_i × conf_i`)
- **R-101**: (No change needed here — this is a dashboard WebSocket concern)

---

### Brokers module

#### [MODIFY] [brokers/paper_broker.py](file:///c:/Users/user/pbt_trading/brokers/paper_broker.py)

- **R-012**: `GBMPriceSimulator` already uses correct GBM formula ✅. The S₀ defaults just need the right per-symbol initial prices (fix is in `main.py _build_broker`).
- **R-064**: BUY with insufficient cash → partial fill instead of full cancel

#### [MODIFY] [brokers/alpaca_broker.py](file:///c:/Users/user/pbt_trading/brokers/alpaca_broker.py)

- **R-065**: `submit_market_order` already uses `TimeInForce.DAY` ✅ and needs `round(qty, 6)` — add qty rounding
- **R-070**: `cancel_all_orders()` already exists ✅

---

### Agents module

#### [MODIFY] [agents/ppo_agent.py](file:///c:/Users/user/pbt_trading/agents/ppo_agent.py)

- **R-034**: Add Gaussian noise ε~N(0,σ) to action selection in `act()` (σ from hyperparams or default 0.1)
- **R-035 / R-036**: Fix `update()` advantage formula (already correct in this file ✅ — `advantage = returns - baselines`). Add PPO reward shaping coefficients α=0.3/β=0.5/γ=0.1 as a reward transform wrapper in main.py's `evaluate_fn`

#### [MODIFY] [agents/dreamer_agent.py](file:///c:/Users/user/pbt_trading/agents/dreamer_agent.py)

- **R-037**: Change `imagination_depth` default from 8 → 5 (the `DreamerAgent` here uses `imagination_depth=8`; compliance report says 10 in the old codebase, spec says 5)
- **R-038**: Add `learn(observation, actual_return)` method performing per-step online MSE gradient descent on transition model weights

#### [MODIFY] [agents/worldmodel_agent.py](file:///c:/Users/user/pbt_trading/agents/worldmodel_agent.py)

- **R-041**: Fix confidence formula: `1.0 − min(std/uncertainty_scale, 1.0)` with `uncertainty_scale` hyperparameter; add zero-signal path when `confidence < confidence_threshold`
- **R-042**: Add `learn(observation, actual_return)` method for per-step MSE gradient descent on ensemble
- **R-043**: Wrap direction signal in outer `np.tanh()`: `direction = np.tanh(mean_pred)`
- **R-044**: Move `ensemble_size` (n_ensemble) from class constant into hyperparams with range [3, 10]
- **R-081**: Add `mutate_prob=0.8` probability in `PBTEngine.perturb_hyperparams()` so each key is only perturbed with 80% probability

---

### Core module

#### [MODIFY] [core/pbt_engine.py](file:///c:/Users/user/pbt_trading/core/pbt_engine.py)

- **R-017**: `_save_checkpoint()` already saves per-gen JSON ✅. Also write `config.json` alongside checkpoint with engine hyperparams
- **R-018**: After saving, prune to keep only last 10 checkpoints
- **R-019**: Write genealogy events to `logs/pbt_genealogy.json` on each exploit/explore
- **R-081**: Apply `mutate_prob=0.8` in `perturb_hyperparams()`
- **R-084 / R-085**: Restructure genealogy log records with correct fields: `generation`, `agent_id`, `parent_id`, `event`, `hyperparams_before`, `hyperparams_after`

#### [MODIFY] [core/evaluator.py](file:///c:/Users/user/pbt_trading/core/evaluator.py)

- **R-115**: Wrap `_log_jsonl()` file write in `try/except IOError` so disk-full does not crash the loop

---

### Feature builder (NEW)

#### [NEW] [features/feature_builder.py](file:///c:/Users/user/pbt_trading/features/feature_builder.py)

The compliance report (R-014/R-015) requires the observation vector to be:
`[price_history_normalised (lookback), RSI/100, BB_pos, ATR/price, regime_one_hot (5)] = lookback+8 elements`

Currently `main.py` uses an inline `FeatureBuilder` class. A new `features/feature_builder.py` will implement the full compliant feature vector. The `main.py` import will switch to this.

- Normalised price history of length `lookback` (divide each price by current price)
- RSI/100
- Bollinger position
- ATR/price
- 5-element regime one-hot (`[BULL, BEAR, RANGING, HIGH_VOL, UNKNOWN]`)
- `state_dim = lookback + 8`

---

### Main CLI

#### [MODIFY] [main.py](file:///c:/Users/user/pbt_trading/main.py)

- **R-012**: Pass correct per-symbol S₀ to `GBMPriceSimulator` (AAPL=185, SPY=520, GLD=185, BTC=67000)
- **R-023**: Fix `--trend ranging` → `mu = 0.0` (currently it falls through to bear mu=-0.0003)
- **R-025**: After backtest completes, write timestamped results to `results/{symbol}_{timestamp}.json`
- **R-016**: Ensure `step` field is a top-level key in the metrics log record
- **R-077**: On backtest, create `experiments/{timestamp}/` with `config.yaml`, `results.json`, `equity_curve.csv`, `notes.md`

---

### MLOps

#### [MODIFY] [mlops/validation.py](file:///c:/Users/user/pbt_trading/mlops/validation.py)

- **R-016** (serve mode log): No change needed — `OnlineEvaluator` already writes `logs/evaluator.jsonl`; orchestrator log is outside this file's scope.

---

### Monitoring

#### [MODIFY] [monitoring/__init__.py](file:///c:/Users/user/pbt_trading/monitoring/__init__.py)

- **R-095**: Gate `alerting.py` so it is not importable in v1.0. Add a module-level `__all__ = []` and a `DeprecationWarning` that it is a v1.1 stub only.

---

### Misc / Structure

- **R-005**: Create `configs/.gitkeep`
- **R-033**: Add `hyperparameter_bounds` dict comment block to `core/pbt_engine.py` `HYPERPARAM_RANGES`
- **R-100**: Add live-mode guidance comment to `PBTEngine.__init__` or `RiskConfig`
- **R-020**: Add dataset naming convention comment to `backtest/data_loader.py`
- **R-024**: Add log-rotation helper in `core/evaluator.py` `_log_jsonl` (trim to last 30 days)

---

### Tests

#### [MODIFY] [tests/test_suite.py](file:///c:/Users/user/pbt_trading/tests/test_suite.py)

- **R-102**: Add 4+ new tests to reach ≥30 total (currently 26). New tests:
  1. `test_risk_cash_buffer` — verify 5% cash buffer enforcement
  2. `test_partial_fill_paper_broker` — buy with insufficient cash → partial fill not cancel
  3. `test_feature_builder_state_dim` — assert `state_dim == lookback + 8`
  4. `test_checkpoint_pruning` — verify only 10 checkpoints retained after 12 saves

---

## Verification Plan

### Automated Tests
```
pwsh -c .venv\Scripts\python.exe -m pytest tests/test_suite.py -v
```
Target: ≥30 tests, all passing.

### Syntax check (all modified files)
```
pwsh -c .venv\Scripts\python.exe -m py_compile agents/ppo_agent.py agents/dreamer_agent.py agents/worldmodel_agent.py core/pbt_engine.py risk/risk_manager.py signals/aggregator.py features/feature_builder.py main.py
```

### Manual Verification
- Verify `python main.py validate` still runs and produces `reports/validation.json`
- Verify `python main.py backtest --use-sample` outputs a timestamped file under `results/` and an experiment under `experiments/`
- Verify `configs/` directory exists in repo root
