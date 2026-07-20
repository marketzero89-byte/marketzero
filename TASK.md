# MarketZero Compliance — Task List

## CRITICAL
- [ ] R-095: Gate `monitoring/alerting.py` as v1.1 stub (not importable in v1.0)
- [ ] R-012: Fix GBM per-symbol S₀ (AAPL=185, SPY=520, GLD=185, BTC=67000) in `main.py _build_broker`

## HIGH
- [ ] R-014/R-015: New `features/feature_builder.py` — `lookback+8` observation vector with regime one-hot
- [ ] R-058: Reorder `check_order()` checks: pos-size → daily-loss → drawdown → trade-count → exposure
- [ ] R-054: Add 5% minimum cash buffer check in `check_order()`
- [ ] R-059: Add `daily_pnl_pct` to `risk_manager.status()`
- [ ] R-060: Position size exceeded → reduce qty (not reject)
- [ ] R-038: Add `learn()` to `DreamerAgent` (per-step transition model MSE update)
- [ ] R-042: Add `learn()` to `WorldModelAgent` (per-step ensemble MSE update)
- [ ] R-064: Paper broker BUY with insufficient cash → partial fill not full cancel
- [ ] R-018: Checkpoint pruning — keep only last 10
- [ ] R-019: Write genealogy events to `logs/pbt_genealogy.json`
- [ ] R-084/R-085: Fix genealogy record fields (generation, agent_id, parent_id, event, hyperparams_before/after)
- [ ] R-115: Wrap `_log_jsonl()` in `try/except IOError`
- [ ] R-108: `evaluate_fn` stop-loss/take-profit is already wired ✅ — verify main.py
- [ ] R-115: IOError guard on evaluator log write

## MEDIUM
- [ ] R-037: Change `imagination_depth` default → 5 in `DreamerAgent`
- [ ] R-041: Fix WorldModel confidence formula + zero-signal path
- [ ] R-043: Wrap WorldModel direction in outer `np.tanh()`
- [ ] R-044: Move `ensemble_size` to hyperparams [3,10]
- [ ] R-081: Add `mutate_prob=0.8` in `perturb_hyperparams()`
- [ ] R-034: Add Gaussian noise ε to PPO `act()`
- [ ] R-050: Fix aggregator direction formula (Σ w_i × dir_i × conf_i / Σ w_i × conf_i)
- [ ] R-022: Add `daily_pnl` and `total_return_pct` to `PaperBroker.portfolio_state()`  (already present ✅ — verify)
- [ ] R-023: Fix backtest ranging mu = 0.0
- [ ] R-025: Write timestamped results to `results/` after backtest
- [ ] R-065: Round Alpaca order qty to 6 decimal places
- [ ] R-017: Write `config.json` alongside checkpoint
- [ ] R-024: Log rotation hint in evaluator
- [ ] R-016: Ensure `step` is top-level key in metrics log record
- [ ] R-033: Add `hyperparameter_bounds` comment to `HYPERPARAM_RANGES`
- [ ] R-101: Add `halt_trading` WebSocket handler — *not in scope (server.py not in this package)*
- [ ] R-077: Create `experiments/{timestamp}/` dir on backtest
- [ ] R-100: Add live-mode guidance comment to `RiskConfig`

## LOW
- [ ] R-005: Create `configs/.gitkeep`
- [ ] R-020: Add dataset naming convention comment to `backtest/data_loader.py`
- [ ] R-070: `cancel_all_orders()` already exists in alpaca_broker.py ✅
- [ ] R-094: Console leaderboard shows all agents (not just top 5)
- [ ] R-113: AlpacaAPIError retry with 1s delay — *server.py not in this package, skip*
- [ ] R-033: `hyperparameter_bounds` in `PBTLiveConfig` — *pbt_trading_config not in this package*

## TESTS
- [ ] R-102: Add 4 new tests to reach ≥ 30 total in `tests/test_suite.py`

## VERIFY
- [ ] Syntax check all modified files
- [ ] Run full test suite: `pytest tests/test_suite.py -v`
