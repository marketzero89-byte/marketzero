# 21 — Disaster Recovery

---

## Failure Scenarios and Recovery Procedures

---

## Scenario 1 — Process Crash (System Exit / OOM / Segfault)

**Symptom**: Trading engine stops, dashboard shows reconnecting overlay, terminal shows error.

**Recovery**:
```bash
# 1. Identify last checkpoint
ls -lt checkpoints/ | head -5

# 2. Restart from checkpoint
python main.py serve --resume-from checkpoints/generation_0042

# 3. Verify dashboard reconnects and trading resumes
```

**Prevention**: Run inside `systemd` or `supervisord` with automatic restart:
```ini
# /etc/supervisor/conf.d/marketzero.conf
[program:marketzero]
command=python /path/to/main.py serve --broker paper
autorestart=true
startsecs=5
```

---

## Scenario 2 — WebSocket Disconnect (Network Interruption)

**Symptom**: Dashboard shows grey dot and reconnect overlay. Trading engine continues unaffected.

**Recovery**: The dashboard auto-reconnects with exponential backoff. No action required. If reconnection fails after 30 seconds, reload the browser tab.

**Note**: The trading engine thread is independent of the WebSocket server. A dashboard disconnect does not affect trading.

---

## Scenario 3 — Circuit Breaker Halt

**Symptom**: Dashboard shows red TRADING HALTED banner. All order submissions blocked.

**Recovery**:
1. Identify halt reason in dashboard (`halt_reason` field)
2. Assess whether underlying condition has resolved
3. Resume via WebSocket command if appropriate:
   ```json
   {"action": "resume_trading"}
   ```
4. Or restart the system:
   ```bash
   python main.py serve --resume-from checkpoints/latest
   ```

**Do not resume blindly.** Understand why the circuit breaker fired before resuming.

---

## Scenario 4 — Alpaca API Failure (Live Mode)

**Symptom**: Orders failing, log shows `AlpacaAPIError`, equity not updating.

**Recovery**:
```bash
# 1. Check Alpaca status
curl https://status.alpaca.markets/api/v2/status.json

# 2. If Alpaca is down: trading loop will retry automatically (1s delay on error)
# 3. If credentials expired: rotate keys in Alpaca dashboard
export ALPACA_API_KEY=new_key
export ALPACA_SECRET_KEY=new_secret
# Restart the system

# 4. Check for open orders that may not have filled:
# → Alpaca dashboard: app.alpaca.markets/orders
```

---

## Scenario 5 — Checkpoint Corruption

**Symptom**: `--resume-from` fails with `pickle.UnpicklingError` or similar.

**Recovery**:
```bash
# 1. Try the previous checkpoint
python main.py serve --resume-from checkpoints/generation_0041

# 2. If all checkpoints are corrupt, start fresh
python main.py serve  # no resume flag

# 3. Pre-train first if available
python main.py train --n-agents 6 --episodes 50
python main.py serve --resume-from <pre-trained checkpoint>
```

**Prevention**: Keep a backup of the last 3 checkpoints on a separate drive or cloud storage.

---

## Scenario 6 — Log Disk Full

**Symptom**: `IOError: No space left on device` in log output. Trading continues but logs are lost.

**Recovery**:
```bash
# 1. Check disk usage
df -h logs/

# 2. Archive old logs
tar -czf logs_archive_$(date +%Y%m%d).tar.gz logs/
rm logs/pbt_orchestrator.json
touch logs/pbt_orchestrator.json

# 3. Reduce checkpoint retention (edit PBTLiveExecutor)
# max_checkpoints = 5  (default 10)
```

---

## Scenario 7 — Corrupted Portfolio State (Paper Mode)

**Symptom**: Equity shows implausible values (negative, astronomical). Agents trading against phantom positions.

**Recovery**:
```bash
# 1. Halt immediately (Ctrl+C)
# 2. Do NOT resume from this checkpoint
# 3. Start fresh with a new paper portfolio
python main.py serve  # new session, fresh portfolio

# 4. Report: document what caused the corruption in experiments/incidents/
```

---

## Backup Schedule

| Asset | Frequency | Destination |
|---|---|---|
| Checkpoints | Per generation (automatic) | `checkpoints/` |
| Checkpoints (offsite) | Daily | External drive / S3 |
| Logs | Weekly archive | `logs_archive/` |
| Source code | Every commit | Git remote |
| Config snapshots | Per deployment | `checkpoints/generation_N/config.json` |

---

## Recovery Time Objectives

| Failure Type | RTO (paper mode) | RTO (live mode) |
|---|---|---|
| Process crash | < 2 minutes | < 5 minutes |
| Network disconnect | < 30 seconds | < 30 seconds |
| Circuit breaker | Minutes (human review) | Minutes (human review) |
| Alpaca outage | N/A (wait) | < 1 hour (after Alpaca restores) |
| Full system rebuild | < 30 minutes | < 2 hours |
