# 20 — Model Governance

---

## Overview

Model governance covers the policies, controls, and audit trails that ensure MarketZero operates safely, fairly, and within defined risk boundaries. This is particularly important in live trading where model failures have direct financial consequences.

---

## Model Approval Policy

No agent type or hyperparameter configuration may be deployed to live capital without:

1. **Unit tests passing** — all 30 tests in `test_marketzero.py`
2. **Paper trading validation** — minimum 5 days, fitness ≥ 0.3, Sharpe ≥ 1.0
3. **Risk parameter review** — all circuit breaker thresholds reviewed and documented
4. **Deployment checklist signed off** — see `18_Live_Trading.md`

---

## Audit Trail

All model decisions are logged with sufficient detail to reconstruct any trade:

### Trade Audit Record
```json
{
  "timestamp": 1718000000.0,
  "generation": 42,
  "agent_id": "PPO_a3f2b1",
  "agent_type": "PPO",
  "symbol": "AAPL",
  "side": "buy",
  "qty": 2.5,
  "price": 185.40,
  "confidence": 0.78,
  "regime": "BULL",
  "fitness_at_time": 0.847,
  "risk_approved": true
}
```

Logs are append-only. No log records are deleted.

### Evolution Audit Record
```json
{
  "generation": 5,
  "agent_id": "PPO_7a22",
  "parent_id": "PPO_a3f2",
  "event": "exploit",
  "hyperparams_before": {"learning_rate": 0.001, "gamma": 0.95},
  "hyperparams_after": {"learning_rate": 0.0008, "gamma": 0.97}
}
```

---

## Risk Override Policy

The risk manager's hard limits cannot be overridden at runtime without code modification. This is intentional. To change a limit:

1. Modify the value in `RiskManager.__init__` or via the executor before `initialize_population()`
2. Document the change with rationale in `experiments/`
3. Run the test suite
4. Restart the system (no hot-patching of risk limits)

---

## Circuit Breaker Authority

Circuit breakers fire automatically without human intervention. Resuming trading after a halt requires:

1. **Automatic halt** (daily loss limit, drawdown): Resume via dashboard command or system restart. The halt reason is logged.
2. **Manual halt** (Ctrl+C): Resume by restarting with `--resume-from` flag pointing to the last checkpoint.

No circuit breaker can be disabled without code modification.

---

## Explainability

MarketZero is designed for explainability at every layer:

| Layer | Explanation Available |
|---|---|
| Signal direction | `direction ∈ (-1, 1)`: negative = bearish, positive = bullish |
| Signal confidence | `confidence ∈ (0, 1)`: calibrated probability-like score |
| Agent type | PPO (policy gradient), Dreamer (imagination), WorldModel (ensemble) |
| Fitness score | Decomposable into Sharpe, Calmar, Sortino components |
| Evolution event | Explicit exploit/explore log per agent per generation |
| Hyperparameters | All hyperparameters human-readable and bounded |

No black-box neural networks are used in v1.0. All agent policies are linear models.

---

## Data Privacy

MarketZero does not collect, transmit, or store any personal data. All data is local:
- Price data from Alpaca (market data)
- Portfolio state in memory (no cloud storage)
- Logs written to local disk only

---

## Version Control Policy

- All code is committed to version control before any live deployment
- `main` branch = production-ready code only
- Feature branches for all development
- Semantic versioning: `MAJOR.MINOR.PATCH`
- Breaking changes to config schema or agent interface require a MAJOR version bump

---

## Incident Response

If a live trading incident occurs (unexpected loss, system crash, data corruption):

1. **Immediately**: Ctrl+C to halt the system
2. **Within 5 minutes**: Cancel all open orders via Alpaca dashboard
3. **Within 1 hour**: Collect and preserve all logs from `logs/`
4. **Within 24 hours**: Root cause analysis documented in `experiments/incidents/`
5. **Before resuming**: All 30 tests pass, root cause addressed
