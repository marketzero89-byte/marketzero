# Recommended workflow for training agents and passing the go-live checklist

This workflow is the shortest path to get the system from a local paper run to a production-ready live-trading gate.

## 1. Prepare the environment

1. Activate the project virtual environment.
2. Install dependencies if needed:

```bash
pip install -r requirements.txt
```

3. Create a local environment file with the required secrets:

```env
PBT_API_KEY=change-me
APCA_API_KEY_ID=your_alpaca_key
APCA_API_SECRET_KEY=your_alpaca_secret
ALERT_SLACK_WEBHOOK=https://hooks.slack.com/services/...
# Optional email alerts
ALERT_EMAIL_TO=alerts@example.com
ALERT_SMTP_PASS=your-smtp-password
```

4. Make sure the sample dataset exists:

```bash
python main.py fetch-data --offline --symbol AAPL --days 365 --output data/AAPL_sample.csv
```

## 2. Train agents in paper mode

Use the paper broker first and let the population learn for at least 5 calendar days of logged paper sessions.

### Recommended starting command

```bash
python main.py run \
  --broker paper \
  --generations 50 \
  --population 12 \
  --steps-per-gen 200 \
  --lookback 50 \
  --exploit-fraction 0.2 \
  --daily-loss-limit 0.05 \
  --max-drawdown 0.15 \
  --stop-loss 0.03 \
  --take-profit 0.08
```

### What to watch while it runs

- Keep the dashboard open to monitor equity, fitness, and trade activity.
- Let the run continue long enough to log at least 5 distinct paper-trading days.
- The validator target is a peak fitness of at least 0.3.
- If fitness stays below 0.3, continue training with larger horizon or more generations.

### If fitness is still weak

Try one of these upgrades:

```bash
python main.py run --broker paper --generations 100 --population 12 --steps-per-gen 200 --lookback 50
```

or

```bash
python main.py run --broker paper --generations 100 --population 16 --steps-per-gen 250 --lookback 50
```

or

```bash
python main.py run --broker paper --generations 100 --population 16 --steps-per-gen 200 --lookback 30 --exploit-fraction 0.15 --daily-loss-limit 0.06 --max-drawdown 0.12 --stop-loss 0.03 --take-profit 0.06 --heterogeneous --use-mlp --parallel
```

The main levers are:
- more generations
- more population diversity
- longer lookback for richer features
- more realistic paper training time

## 3. Backtest on real or sample OHLCV data

Before you can pass the backtest gate, run a backtest report.

### Sample OHLCV backtest

```bash
python main.py backtest --use-sample --output reports/backtest_ohlcv.json
```

### Optional real-data backtest

```bash
python main.py backtest --data data/AAPL_sample.csv --symbol AAPL --output reports/backtest_ohlcv.json
```

The validator checks that a real-data-style OHLCV report exists and that the out-of-sample Sharpe is non-negative.

## 4. Validate readiness

Run the built-in validator after paper training and backtesting:

```bash
python main.py validate --fix
```

You need all required checks to pass:
- checkpoints present
- paper-trading duration >= 5 calendar days
- peak fitness >= 0.3
- real-data OHLCV backtest report present
- Alpaca credentials configured
- alerting configured (recommended)
- dashboard API key configured if the dashboard is exposed publicly

## 5. Move to Alpaca paper trading

Only after the validator passes should you start paper trading through Alpaca:

```bash
python main.py run \
  --broker alpaca \
  --generations 100 \
  --population 12 \
  --steps-per-gen 200 \
  --lookback 50
```

## 6. Go live only after paper validation passes

When the Alpaca paper run is stable and the validation report is still green, switch to live:

```bash
python main.py run --broker alpaca --live
```

Do not use `--force-live` unless you are intentionally bypassing the normal validation gate.

## 7. Definition of done

You are ready to go live when:
- paper training has run long enough to log 5+ days
- the best fitness in the checkpoints is at least 0.3
- an OHLCV backtest report exists
- the validator reports all required checks as passed
- Alpaca credentials and alerts are configured

## 8. Local shortcut for development/testing

If you only need to test the validation flow locally, you can bootstrap fake paper sessions:

```bash
python main.py validate --bootstrap --bootstrap-days 5 --bootstrap-fitness 0.35 --fix
```

That is useful for testing the workflow, but it does not replace the real paper-trading requirement for production readiness.

