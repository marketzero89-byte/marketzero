#!/usr/bin/env python3
"""
PBT Trading Platform — CLI Entry Point
Commands: serve, run, pbt-live, train, backtest, validate, fetch-data, status
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# Load .env before any module reads environment variables
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(dotenv_path=_env_path, override=False)
except ImportError:
    pass  # python-dotenv optional; use OS env vars directly

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pbt.cli")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _build_engine(args):
    from core.pbt_engine import PBTEngine
    return PBTEngine(
        population_size=getattr(args, "population", 12),
        agent_types=["ppo", "dreamer", "worldmodel"],
        exploit_fraction=getattr(args, "exploit_fraction", 0.2),
        checkpoint_dir=getattr(args, "checkpoint_dir", "checkpoints"),
    )


def _build_broker(args):
    broker_type = getattr(args, "broker", "paper")
    symbols = [s.strip() for s in getattr(args, "symbols", "AAPL,MSFT,GOOGL").split(",") if s.strip()]
    if broker_type == "alpaca":
        from brokers.alpaca_broker import AlpacaBroker
        return AlpacaBroker(
            paper=not getattr(args, "live", False),
            symbols=symbols or ["AAPL"],
        )
    if broker_type == "coinglass":
        from brokers.coinglass_broker import CoinGlassBroker
        # Default to BTC/ETH/SOL if the user hasn't overridden --symbols
        default_crypto = ["BTC", "ETH", "SOL"]
        crypto_symbols = symbols if symbols != ["AAPL", "MSFT", "GOOGL"] else default_crypto
        return CoinGlassBroker(
            symbols=crypto_symbols,
            initial_cash=getattr(args, "capital", 100_000),
        )
    from brokers.paper_broker import PaperBroker, GBMPriceSimulator
    # R-012: documented per-symbol starting prices (spec S₀ values)
    _S0_DEFAULTS = {
        "AAPL": 185.00,
        "SPY":  520.00,
        "GLD":  185.00,
        "BTC":  67_000.00,
        "BTC/USD": 67_000.00,
    }
    initial_prices = {s: _S0_DEFAULTS.get(s, 100.0) for s in symbols}
    sim = GBMPriceSimulator(symbols=symbols, initial_prices=initial_prices, mu=0.08, sigma=0.015)
    return PaperBroker(
        initial_cash=getattr(args, "capital", 100_000),
        symbols=symbols,
        simulator=sim,
    )


def _build_engine_from_args(args):
    """Build PBTEngine, optionally resuming from a checkpoint."""
    resume = getattr(args, "resume_from", "") or ""
    if resume:
        from core.pbt_engine import PBTEngine
        path = Path(resume)
        if not path.exists():
            logger.error("Checkpoint not found: %s", resume)
            sys.exit(1)
        engine = PBTEngine.load_checkpoint(
            path,
            population_size=getattr(args, "population", 12),
            agent_types=["ppo", "dreamer", "worldmodel"],
            exploit_fraction=getattr(args, "exploit_fraction", 0.2),
            checkpoint_dir=getattr(args, "checkpoint_dir", "checkpoints"),
        )
        logger.info("Resumed from checkpoint %s (gen %d)", path.name, engine.generation)
        return engine
    return _build_engine(args)


def _enforce_live_gate(args) -> None:
    """Block --live unless production validation passes."""
    if not getattr(args, "live", False):
        return
    if getattr(args, "force_live", False):
        logger.warning("Live trading gate bypassed with --force-live")
        return
    from mlops.validation import ProductionValidator
    report = ProductionValidator().run_all()
    if report.ready_for_live:
        logger.info("Live trading gate: all required checks passed")
        return
    for check in report.checks:
        if not check.passed and check.required:
            logger.error("  [%s] %s", check.name, check.message)
    logger.error(
        "Live trading blocked. Fix issues above, or run: python main.py validate"
    )
    sys.exit(1)


def _record_paper_session(args, result, equity, drawdown, broker) -> None:
    if getattr(args, "live", False):
        return
    from mlops.validation import ProductionValidator
    n_trades = len(broker.trades) if hasattr(broker, "trades") else 0
    ProductionValidator().record_session(
        broker=getattr(args, "broker", "paper"),
        live=False,
        generation=result.generation,
        best_fitness=result.best_agent.fitness,
        mean_fitness=result.mean_fitness,
        equity=equity,
        drawdown=drawdown,
        n_trades=n_trades,
    )


def _build_feature_builder(args):
    from features.engineer import FeatureBuilder
    return FeatureBuilder(
        lookback=getattr(args, "lookback", 50),
        include_alt_data=getattr(args, "alt_data", False),
        include_macro=getattr(args, "macro", False),
    )


def _build_risk(args):
    from risk.risk_manager import RiskManager, RiskConfig
    from mlops.alerts import AlertManager
    alerts = AlertManager()
    cfg = RiskConfig(
        daily_loss_limit_pct=getattr(args, "daily_loss_limit", 0.20),
        max_drawdown_pct=getattr(args, "max_drawdown", 0.30),
        stop_loss_pct=getattr(args, "stop_loss", 0.08),
        take_profit_pct=getattr(args, "take_profit", 0.15),
    )

    def _on_risk(event):
        if event.severity == "halt":
            eq    = rm._session_start_equity or 0.0
            peak  = rm._peak_equity or eq
            dd    = (eq - peak) / peak if peak > 0 else 0.0
            alerts.halt(event.message, equity=eq, drawdown=dd)
        else:
            alerts.circuit_breaker(event.event_type, event.message)

    rm = RiskManager(config=cfg, alert_callbacks=[_on_risk])
    return rm


# ---------------------------------------------------------------------------
# evaluate_agent: wraps broker + agents into PBT evaluate_fn
# ---------------------------------------------------------------------------

def _make_evaluate_fn(broker, feature_builder, risk_manager, aggregator, n_steps=200, use_mlp=False, heterogeneous=False):
    """Factory that returns an evaluate_fn(AgentRecord) -> AgentRecord."""

    from agents import build_agent
    from brokers.helpers import (
        broker_cash,
        broker_current_price,
        broker_equity,
        broker_is_market_open,
        broker_position_qty,
        broker_positions_market_value,
        broker_price_history,
        broker_step,
        broker_symbols,
        order_succeeded,
    )
    from brokers.alpaca_broker import BrokerDisconnectedError
    from core.evaluator import OnlineEvaluator
    from core.regime import RegimeDetector

    evaluator = OnlineEvaluator(log_path="logs/evaluator.jsonl")
    regime_detector = RegimeDetector()
    _agent_instances = {}

    def _submit_order(side, symbol, qty, price, equity):
        positions_mv = broker_positions_market_value(broker)
        allowed, reason = risk_manager.check_order(
            symbol, side, qty, price, equity, positions_mv,
        )
        if not allowed:
            logger.debug("Order rejected: %s %s %s — %s", side, qty, symbol, reason)
            return False
        try:
            result = broker.submit_market_order(symbol, side, qty)
        except BrokerDisconnectedError:
            logger.critical("Broker disconnected — halting evaluation")
            risk_manager._halt("Broker disconnected")
            return False
        except Exception as exc:
            logger.error("Order failed %s %s %s: %s", side, qty, symbol, exc)
            return False
        if not order_succeeded(result):
            logger.error("Order not filled: %s %s %s → %s", side, qty, symbol, result)
            return False
        if side == "buy":
            risk_manager.register_entry(symbol, price)
        risk_manager.record_trade()
        return True

    def evaluate_fn(agent_record):
        aid = agent_record.agent_id
        if aid not in _agent_instances:
            _agent_instances[aid] = build_agent(
                agent_type=agent_record.agent_type,
                state_dim=feature_builder.state_dim,
                hyperparams=agent_record.hyperparams,
                use_mlp=use_mlp,
                heterogeneous=heterogeneous,
                seed=hash(aid) % 2**31,
            )
        agent_inst = _agent_instances[aid]

        equity_start = broker_equity(broker)
        local_equity = [equity_start]
        symbols = broker_symbols(broker)
        symbol = symbols[0]
        _current_regime = "UNKNOWN"  # updated each step for state builder

        for _ in range(n_steps):
            if not broker_is_market_open(broker):
                time.sleep(1.0)
                continue

            try:
                broker_step(broker)
            except BrokerDisconnectedError as exc:
                logger.critical("Broker step failed: %s", exc)
                risk_manager._halt(str(exc))
                break

            prices = broker_price_history(broker, symbol)
            if len(prices) < 5:
                continue

            pos_qty = broker_position_qty(broker, symbol)
            equity = broker_equity(broker, equity_start)
            cash = broker_cash(broker)
            price = broker_current_price(broker, symbol, prices[-1])

            # Detect regime before building state so one-hot is populated (R-014/R-015)
            _current_regime = regime_detector.detect(prices).value
            state = feature_builder.build(
                closes=prices,
                position=pos_qty / 100.0,
                cash_pct=cash / max(equity, 1),
                regime=_current_regime,
            )

            action, log_prob = agent_inst.act(state)
            risk_manager.update(equity, broker_positions_market_value(broker))

            if risk_manager.is_halted:
                break

            # Stop-loss / take-profit — only check when we actually hold a position
            if pos_qty > 0:
                should_exit, exit_reason = risk_manager.check_stop_take(symbol, price)
                if should_exit:
                    sold = _submit_order("sell", symbol, int(pos_qty), price, equity)
                    if sold:
                        risk_manager.deregister_entry(symbol)
            else:
                # No position held — ensure stale entry price is cleared
                risk_manager.deregister_entry(symbol)

            qty = max(1, int(equity * 0.05 / max(price, 1)))
            if action == 1:
                _submit_order("buy", symbol, qty, price, equity)
            elif action == 2 and pos_qty > 0:
                sold = _submit_order("sell", symbol, min(qty, int(pos_qty)), price, equity)
                if sold:
                    risk_manager.deregister_entry(symbol)

            new_equity = broker_equity(broker, equity_start)
            prev_equity = local_equity[-1]
            raw_return = (new_equity - prev_equity) / max(prev_equity, 1)
            drawdown = 0.0
            if prev_equity > 0:
                peak_equity = max(local_equity)
                drawdown = max(0.0, (peak_equity - new_equity) / max(peak_equity, 1))
            reward = raw_return - 0.35 * drawdown
            if action == 1 and pos_qty > 0:
                reward -= 0.001
            if action == 2 and pos_qty == 0:
                reward -= 0.0005
            local_equity.append(new_equity)

            next_prices = broker_price_history(broker, symbol)
            if len(next_prices) >= 5 and hasattr(agent_inst, "store"):
                next_pos = broker_position_qty(broker, symbol)
                next_cash = broker_cash(broker)
                next_eq = broker_equity(broker, equity_start)
                next_regime = regime_detector.detect(next_prices).value
                next_state = feature_builder.build(
                    closes=next_prices,
                    position=next_pos / 100.0,
                    cash_pct=next_cash / max(next_eq, 1),
                    regime=next_regime,
                )
                if type(agent_inst).__name__ == "PPOAgent":
                    from agents.ppo_agent import Transition
                    agent_inst.store(Transition(
                        state=state, action=action, reward=reward,
                        next_state=next_state, done=False, log_prob=log_prob,
                    ))
                else:
                    agent_inst.store(state, action, reward, next_state)
                # Per-step online learn for Dreamer / WorldModel (R-038, R-042)
                if hasattr(agent_inst, "learn"):
                    agent_inst.learn(state, reward)

        if hasattr(agent_inst, "update"):
            agent_inst.update()

        regime_prices = broker_price_history(broker, symbol) or [100.0]
        regime = regime_detector.detect(regime_prices)

        return evaluator.evaluate_agent(
            agent_record,
            equity_curve=local_equity,
            trade_pnls=broker.trade_pnls(),
            regime=regime.value,
        )

    return evaluate_fn


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_serve(args):
    """Start the FastAPI WebSocket dashboard server (mock data only)."""
    logger.info("Starting dashboard server on %s:%d", args.host, args.port)
    from dashboard.server import serve
    serve(host=args.host, port=args.port)


def cmd_run(args):
    """
    Unified command: run PBT engine + dashboard server together.

    The PBT generation loop runs in a daemon background thread.
    The FastAPI/uvicorn server runs in the main thread.
    A shared LiveStateStore bridges the two so the dashboard shows
    real live data instead of mock data.
    """
    _enforce_live_gate(args)
    import threading
    from dashboard.state_store import LiveStateStore
    from dashboard.server import create_app
    import uvicorn

    logger.info("=== PBT RUN (engine + dashboard) ===")
    logger.info("Population: %d | Generations: %d | Broker: %s",
                args.population, args.generations, args.broker)
    logger.info("Dashboard: http://%s:%d", args.host if args.host != '0.0.0.0' else 'localhost', args.port)

    # --- Shared live state ---
    store = LiveStateStore()

    # --- Build PBT components ---
    engine  = _build_engine_from_args(args)
    broker  = _build_broker(args)
    feat    = _build_feature_builder(args)
    risk    = _build_risk(args)

    from signals.aggregator import SignalAggregator, AggMode
    agg = SignalAggregator(mode=AggMode(args.agg_mode))

    from core.metrics_engine import build_dashboard_state
    from mlops.metrics_logger import MetricsLogger
    from mlops.alerts import AlertManager
    metrics_logger = MetricsLogger(log_dir="logs", tensorboard=args.tensorboard)
    alerts = AlertManager()

    if not getattr(args, "resume_from", ""):
        engine.initialise_population()

    evaluate_fn = _make_evaluate_fn(
        broker=broker,
        feature_builder=feat,
        risk_manager=risk,
        aggregator=agg,
        n_steps=args.steps_per_gen,
        use_mlp=args.use_mlp,
        heterogeneous=args.heterogeneous,
    )

    def on_generation(result):
        best = result.best_agent

        portfolio = broker.portfolio_state() if hasattr(broker, 'portfolio_state') else {}
        equity = portfolio.get('equity', broker.equity_curve[-1] if broker.equity_curve else 0)
        eq_curve = broker.equity_curve if hasattr(broker, 'equity_curve') else [equity]
        peak = max(eq_curve) if eq_curve else equity
        drawdown = (peak - equity) / peak if peak > 0 else 0.0
        recent = broker.recent_trades(12) if hasattr(broker, 'recent_trades') else []
        prices = portfolio.get('prices', {})
        positions = portfolio.get('positions', {})

        state = build_dashboard_state(
            result=result,
            broker_state={
                **portfolio,
                'drawdown': drawdown,
                'halted': risk.is_halted if hasattr(risk, 'is_halted') else False,
                'n_trades': broker.per_generation_trades() if hasattr(broker, 'per_generation_trades') else len(broker.trades) if hasattr(broker, 'trades') else 0,
            },
            broker=broker,
            equity_curve=eq_curve,
            trade_pnls=broker.trade_pnls() if hasattr(broker, 'trade_pnls') else [],
            recent_trades=recent,
            prices=prices,
            positions=positions,
            regime=best.metrics.get('regime', 'UNKNOWN'),
        )

        # For brokers that can provide an authoritative all-time fill count
        # (e.g. AlpacaBroker queries the API), inject it directly so that
        # state_store skips its in-memory delta accumulation and uses the
        # cross-session-accurate value instead.
        if hasattr(broker, 'total_filled_orders'):
            state['cumulative_trades'] = broker.total_filled_orders()

        store.update(state)

        metrics_logger.log_generation(result.generation, {
            'mean_fitness':  result.mean_fitness,
            'std_fitness':   result.std_fitness,
            'best_fitness':  best.fitness,
            'exploit_count': result.exploit_count,
            'explore_count': result.explore_count,
            'elapsed':       result.elapsed_seconds,
            **best.metrics,
        })
        alerts.generation_summary(result.generation, {
            'max_fitness':  best.fitness,
            'mean_fitness': result.mean_fitness,
            **best.metrics,
        })
        _record_paper_session(args, result, equity, drawdown, broker)
        logger.info(
            "Gen %d | best=%.4f mean=%.4f | equity=%.2f dd=%.2f%% | agent=%s (%s)",
            result.generation, best.fitness, result.mean_fitness,
            equity, drawdown * 100, best.agent_id, best.agent_type,
        )

    engine.add_callback(on_generation)

    # Reset daily P&L, equity curve, and GBM prices at the start of each generation
    def _generation_start_hook():
        broker.reset_daily_pnl()
        broker.reset_equity_curve()
        if hasattr(broker, 'mark_generation_start'):
            broker.mark_generation_start()
        if hasattr(broker, 'reset_prices'):
            broker.reset_prices()   # Resets GBM to initial prices — kills phantom take-profit cascade

    if hasattr(broker, 'reset_daily_pnl'):
        engine.set_generation_start_hook(_generation_start_hook)

    def _pbt_thread():
        try:
            engine.run(evaluate_fn, n_generations=args.generations, parallel=args.parallel)
            logger.info("PBT run complete after %d generations.", args.generations)
        except Exception as exc:
            logger.error("PBT engine error: %s", exc, exc_info=True)
            store.set_halted(True, str(exc))

    thread = threading.Thread(target=_pbt_thread, name="pbt-engine", daemon=True)
    thread.start()
    logger.info("PBT engine started in background thread.")

    # --- Start dashboard server (blocks in main thread) ---
    app = create_app(state_provider=store.as_provider())
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


def cmd_pbt_live(args):
    """Run PBT live trading loop."""
    _enforce_live_gate(args)
    logger.info("=== PBT LIVE MODE ===")
    logger.info("Population: %d | Generations: %d | Broker: %s",
                args.population, args.generations, args.broker)

    engine  = _build_engine_from_args(args)
    broker  = _build_broker(args)
    feat    = _build_feature_builder(args)
    risk    = _build_risk(args)

    from signals.aggregator import SignalAggregator, AggMode
    agg = SignalAggregator(mode=AggMode(args.agg_mode))

    from mlops.metrics_logger import MetricsLogger
    from mlops.alerts import AlertManager
    metrics_logger = MetricsLogger(log_dir="logs", tensorboard=args.tensorboard)
    alerts = AlertManager()

    if not getattr(args, "resume_from", ""):
        engine.initialise_population()

    evaluate_fn = _make_evaluate_fn(
        broker=broker,
        feature_builder=feat,
        risk_manager=risk,
        aggregator=agg,
        n_steps=args.steps_per_gen,
        use_mlp=args.use_mlp,
        heterogeneous=args.heterogeneous,
    )

    def on_generation(result):
        best = result.best_agent
        metrics_logger.log_generation(result.generation, {
            "mean_fitness":  result.mean_fitness,
            "std_fitness":   result.std_fitness,
            "best_fitness":  best.fitness,
            "exploit_count": result.exploit_count,
            "explore_count": result.explore_count,
            "elapsed":       result.elapsed_seconds,
            **best.metrics,
        })
        alerts.generation_summary(result.generation, {
            "max_fitness":  best.fitness,
            "mean_fitness": result.mean_fitness,
            **best.metrics,
        })
        portfolio = broker.portfolio_state() if hasattr(broker, "portfolio_state") else {}
        equity = portfolio.get("equity", broker.equity_curve[-1] if broker.equity_curve else 0)
        eq_curve = broker.equity_curve if hasattr(broker, "equity_curve") else [equity]
        peak = max(eq_curve) if eq_curve else equity
        drawdown = (peak - equity) / peak if peak > 0 else 0.0
        _record_paper_session(args, result, equity, drawdown, broker)
        logger.info(
            "Gen %d | best=%.4f mean=%.4f | agent=%s (%s)",
            result.generation, best.fitness, result.mean_fitness,
            best.agent_id, best.agent_type,
        )

    engine.add_callback(on_generation)
    engine.run(evaluate_fn, n_generations=args.generations, parallel=args.parallel)
    logger.info("PBT live run complete.")


def cmd_train(args):
    """Offline pre-training of PPO agents."""
    logger.info("=== OFFLINE TRAINING ===")
    from agents.ppo_agent import PPOAgent
    state_dim = 12 + (3 if args.alt_data else 0) + (3 if args.macro else 0)
    agent = PPOAgent(
        state_dim=state_dim,
        hyperparams={"learning_rate": args.lr, "gamma": args.gamma},
        use_mlp=args.use_mlp,
    )
    losses = agent.pretrain(n_episodes=args.episodes, episode_len=args.episode_len)
    avg_loss = sum(losses) / len(losses) if losses else 0
    logger.info("Training complete. Avg loss: %.6f", avg_loss)

    # Save weights
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    import pickle
    with out.open("wb") as f:
        pickle.dump(agent, f)
    logger.info("Agent saved to %s", out)


def cmd_backtest(args):
    """Run walk-forward + Monte Carlo backtest."""
    logger.info("=== BACKTEST ===")

    from backtest.engine import WalkForwardAnalyser, MonteCarloSimulator, save_backtest_report
    from backtest.data_loader import ensure_sample_data, load_closes_from_file
    from brokers.paper_broker import GBMPriceSimulator

    data_source = "synthetic"
    data_file = ""
    symbol = "SIM"

    if args.data and Path(args.data).exists():
        prices, symbol = load_closes_from_file(args.data, symbol=args.symbol or None)
        data_source = "ohlcv"
        data_file = str(args.data)
        logger.info("Loaded %d bars from %s (%s)", len(prices), args.data, symbol)
    elif args.use_sample:
        sample = ensure_sample_data()
        prices, symbol = load_closes_from_file(sample)
        data_source = "ohlcv"
        data_file = str(sample)
        logger.info("Using sample OHLCV: %s (%d bars)", sample, len(prices))
    else:
        logger.info("No data file specified; generating GBM prices (%d bars)", args.n_bars)
        logger.info("Tip: use --use-sample or --data data/AAPL_sample.csv for real OHLCV backtest")
        # R-023: mu=0.0 gives a flat/ranging synthetic baseline (no upward drift bias)
        sim = GBMPriceSimulator(symbols=["SIM"], mu=0.0, sigma=0.20, seed=42)
        prices = [100.0]
        for _ in range(args.n_bars):
            p = sim.step()["SIM"]
            prices.append(p)

    # Strategy function: simple momentum crossover for baseline
    def strategy_fn(train_prices, test_prices):
        equity = [args.capital]
        pnls   = []
        cash   = args.capital
        pos    = 0
        for i in range(20, len(test_prices)):
            fast = sum(test_prices[i-5:i])  / 5
            slow = sum(test_prices[i-20:i]) / 20
            price = test_prices[i]
            qty = max(1, int(cash * 0.05 / max(price, 1)))
            if fast > slow and pos == 0:
                cost = price * qty * 1.001
                if cash >= cost:
                    cash -= cost
                    pos   = qty
                    entry = price
            elif fast < slow and pos > 0:
                rev  = price * pos * 0.999
                cash += rev
                pnls.append(rev - entry * pos)
                pos = 0
            equity.append(cash + pos * price)
        return equity, pnls

    wf = WalkForwardAnalyser(n_folds=args.folds, train_pct=args.train_pct)
    wf_report = wf.run(prices, strategy_fn)

    mc = MonteCarloSimulator(n_runs=args.mc_runs, confidence=0.95)
    all_pnls = [p for r in wf_report.results for p in r.trade_pnls]
    mc_report = mc.run(all_pnls, initial_equity=args.capital)

    save_backtest_report(
        wf_report, mc_report, args.output,
        data_source=data_source, data_file=data_file, symbol=symbol,
    )

    # R-025: Write timestamped results to results/ directory
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%S")
    results_path = results_dir / f"backtest_{ts}.json"
    import shutil
    shutil.copy2(args.output, str(results_path))
    logger.info("Timestamped results written to %s", results_path)

    # R-077: Create experiments/{timestamp}/ directory for this run
    exp_dir = Path("experiments") / ts
    exp_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.output, str(exp_dir / "backtest_report.json"))
    logger.info("Experiment snapshot saved to %s", exp_dir)

    logger.info("Walk-Forward Results:")
    logger.info("  OOS Sharpe:      %.4f", wf_report.oos_sharpe)
    logger.info("  OOS Calmar:      %.4f", wf_report.oos_calmar)
    logger.info("  OOS Annual Ret:  %.2f%%", wf_report.oos_annual_ret * 100)
    logger.info("  OOS Max DD:      %.2f%%", wf_report.oos_max_dd * 100)
    logger.info("  Consistency:     %.1f%%", wf_report.consistency * 100)
    logger.info("Monte Carlo (95%% CI):")
    logger.info("  Sharpe:      [%.3f, %.3f]", *mc_report.sharpe_ci)
    logger.info("  Annual Ret:  [%.2f%%, %.2f%%]",
                mc_report.annual_ret_ci[0]*100, mc_report.annual_ret_ci[1]*100)
    logger.info("  %% Positive:  %.1f%%", mc_report.pct_positive * 100)
    logger.info("Report saved to %s", args.output)


def cmd_validate(args):
    """Run production readiness checks."""
    from mlops.validation import ProductionValidator, CheckResult
    from backtest.data_loader import ensure_sample_data

    validator = ProductionValidator()

    if args.bootstrap:
        ensure_sample_data()
        today = time.strftime("%Y-%m-%d")
        validator.validation_log.parent.mkdir(parents=True, exist_ok=True)
        for i in range(args.bootstrap_days):
            import datetime
            d = (datetime.date.today() - datetime.timedelta(days=args.bootstrap_days - 1 - i)).isoformat()
            rec = {
                "timestamp": time.time(),
                "date": d,
                "broker": "paper",
                "live": False,
                "generation": i + 1,
                "best_fitness": args.bootstrap_fitness,
                "mean_fitness": args.bootstrap_fitness,
                "equity": 100_000,
                "drawdown": 0.01,
                "n_trades": 10,
                "bootstrapped": True,
            }
            with validator.validation_log.open("a") as f:
                f.write(json.dumps(rec) + "\n")
        logger.info("Bootstrapped %d paper-trading days (dev only)", args.bootstrap_days)

    if args.run_backtest:
        sample = ensure_sample_data()
        logger.info("Running OHLCV backtest on %s ...", sample)
        import argparse
        bt_args = argparse.Namespace(
            data=str(sample), symbol="AAPL", use_sample=False,
            n_bars=500, capital=100_000, folds=3, train_pct=0.70,
            mc_runs=100, output="reports/backtest_ohlcv.json",
        )
        cmd_backtest(bt_args)

    report = validator.run_all()
    for line in report.summary_lines():
        print(line)

    out = validator.save_report(report)
    logger.info("Validation report saved to %s", out)

    if not report.ready_for_live and args.fix:
        print("Suggested next steps:")
        by_name = {c.name: c for c in report.checks}
        if not by_name.get("Paper trading duration", CheckResult("", True, "")).passed:
            print("  - Run paper trading daily: python main.py run --broker paper")
        if not by_name.get("Agent fitness", CheckResult("", True, "")).passed:
            print("  - Continue training until fitness >= 0.3")
        if not by_name.get("Real-data backtest", CheckResult("", True, "")).passed:
            print("  - python main.py backtest --use-sample --output reports/backtest_ohlcv.json")
        if not by_name.get("Alpaca credentials", CheckResult("", True, "")).passed:
            print("  - Set APCA_API_KEY_ID and APCA_API_SECRET_KEY in .env")
        if not by_name.get("Alert notifications", CheckResult("", True, "")).passed:
            print("  - Set ALERT_SLACK_WEBHOOK or email alerts in .env")
        if not by_name.get("Dashboard API key", CheckResult("", True, "")).passed:
            print("  - Set PBT_API_KEY in .env before public deployment")

    sys.exit(0 if report.ready_for_live else 1)


def cmd_fetch_data(args):
    """Download OHLCV data for backtesting."""
    from backtest.data_loader import fetch_ohlcv_to_csv, generate_sample_ohlcv

    if args.offline:
        path = generate_sample_ohlcv(
            args.output or f"data/{args.symbol.upper()}_sample.csv",
            symbol=args.symbol.upper(),
            n_bars=args.days,
        )
        logger.info("Generated offline sample → %s", path)
        return

    out = args.output or f"data/{args.symbol.upper()}_{args.days}d.csv"
    try:
        path = fetch_ohlcv_to_csv(args.symbol, days=args.days, output_path=out)
        logger.info("Downloaded → %s", path)
    except ImportError:
        logger.warning("yfinance not installed; generating offline sample instead")
        path = generate_sample_ohlcv(out, symbol=args.symbol.upper(), n_bars=args.days)
        logger.info("Generated offline sample → %s", path)


def cmd_status(args):
    """Show current system status from the latest checkpoint."""
    import glob
    ckpts = sorted(glob.glob("checkpoints/gen_*.json"))
    if not ckpts:
        logger.info("No checkpoints found in ./checkpoints/")
        return

    latest = Path(ckpts[-1])
    data   = json.loads(latest.read_text())
    gen    = data["generation"]
    pop    = data["population"]
    best   = max(pop, key=lambda a: a["fitness"])

    print(f"\n{'='*60}")
    print(f"  PBT Status — Generation {gen}")
    print(f"{'='*60}")
    print(f"  Checkpoint : {latest.name}")
    print(f"  Population : {len(pop)} agents")
    print(f"  Mean fitness: {data['mean_fitness']:.4f}  ±{data['std_fitness']:.4f}")
    print(f"\n  Best Agent:")
    print(f"    ID       : {best['agent_id']}")
    print(f"    Type     : {best['agent_type']}")
    print(f"    Fitness  : {best['fitness']:.4f}")
    if best.get("metrics"):
        for k, v in best["metrics"].items():
            print(f"    {k:<16}: {v}")
    # R-094: show ALL agents, not just top 5
    print(f"\n  Full Leaderboard ({len(pop)} agents):")
    for i, a in enumerate(sorted(pop, key=lambda x: x["fitness"], reverse=True), 1):
        print(f"    {i:>2}. {a['agent_id']} ({a['agent_type']}) fitness={a['fitness']:.4f}")
    print()

    if args.log_tail:
        log_path = Path("logs/metrics.jsonl")
        if log_path.exists():
            lines = log_path.read_text().strip().split("\n")
            print(f"  Last {args.log_tail} metric entries:")
            for line in lines[-args.log_tail:]:
                try:
                    rec = json.loads(line)
                    print(f"    gen={rec.get('step')} best={rec.get('best_fitness','?'):.4f} "
                          f"mean={rec.get('mean_fitness','?'):.4f}")
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pbt",
        description="PBT Trading Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- serve ----
    p_serve = sub.add_parser("serve", help="Start the dashboard WebSocket server (mock data)")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8000)

    # ---- run (engine + dashboard together, live data) ----
    p_run = sub.add_parser("run", help="Run PBT engine + dashboard server together (live data)")
    p_run.add_argument("--host",            default="0.0.0.0")
    p_run.add_argument("--port",            type=int,   default=8000)
    p_run.add_argument("--population",      type=int,   default=12)
    p_run.add_argument("--generations",     type=int,   default=50)
    p_run.add_argument("--steps-per-gen",   type=int,   default=200, dest="steps_per_gen")
    p_run.add_argument("--broker",          default="paper", choices=["paper", "alpaca", "coinglass"])
    p_run.add_argument("--live",            action="store_true", help="Use Alpaca live (not paper)")
    p_run.add_argument("--force-live",      action="store_true", dest="force_live",
                        help="Bypass live-trading validation gate (use with caution)")
    p_run.add_argument("--symbols",         default="AAPL,MSFT,GOOGL")
    p_run.add_argument("--capital",         type=float, default=100_000)
    p_run.add_argument("--agg-mode",        default="ENSEMBLE", dest="agg_mode",
                        choices=["EQUAL","FITNESS","CONFIDENCE","RANK","ENSEMBLE","ATTENTION"])
    p_run.add_argument("--exploit-fraction",type=float, default=0.2, dest="exploit_fraction")
    p_run.add_argument("--lookback",        type=int,   default=50)
    p_run.add_argument("--alt-data",        action="store_true", dest="alt_data")
    p_run.add_argument("--macro",           action="store_true")
    p_run.add_argument("--use-mlp",         action="store_true", dest="use_mlp")
    p_run.add_argument("--heterogeneous",   action="store_true")
    p_run.add_argument("--parallel",        action="store_true")
    p_run.add_argument("--tensorboard",     action="store_true")
    p_run.add_argument("--checkpoint-dir",  default="checkpoints", dest="checkpoint_dir")
    p_run.add_argument("--daily-loss-limit",type=float, default=0.05, dest="daily_loss_limit")
    p_run.add_argument("--max-drawdown",    type=float, default=0.15, dest="max_drawdown")
    p_run.add_argument("--stop-loss",       type=float, default=0.03, dest="stop_loss")
    p_run.add_argument("--take-profit",     type=float, default=0.08, dest="take_profit")
    p_run.add_argument("--resume-from",     default="", dest="resume_from",
                        help="Resume from a checkpoint JSON (e.g. checkpoints/gen_0007.json)")

    # ---- pbt-live ----
    p_live = sub.add_parser("pbt-live", help="Run PBT live trading loop")
    p_live.add_argument("--population",     type=int,   default=12)
    p_live.add_argument("--generations",    type=int,   default=50)
    p_live.add_argument("--steps-per-gen",  type=int,   default=200, dest="steps_per_gen")
    p_live.add_argument("--broker",         default="paper", choices=["paper", "alpaca", "coinglass"])
    p_live.add_argument("--live",           action="store_true", help="Use Alpaca live (not paper)")
    p_live.add_argument("--force-live",     action="store_true", dest="force_live",
                        help="Bypass live-trading validation gate (use with caution)")
    p_live.add_argument("--symbols",        default="AAPL,MSFT,GOOGL")
    p_live.add_argument("--capital",        type=float, default=100_000)
    p_live.add_argument("--agg-mode",       default="ENSEMBLE", dest="agg_mode",
                        choices=["EQUAL","FITNESS","CONFIDENCE","RANK","ENSEMBLE","ATTENTION"])
    p_live.add_argument("--exploit-fraction", type=float, default=0.2, dest="exploit_fraction")
    p_live.add_argument("--lookback",       type=int,   default=50)
    p_live.add_argument("--alt-data",       action="store_true", dest="alt_data")
    p_live.add_argument("--macro",          action="store_true")
    p_live.add_argument("--use-mlp",        action="store_true", dest="use_mlp")
    p_live.add_argument("--heterogeneous",  action="store_true")
    p_live.add_argument("--parallel",       action="store_true")
    p_live.add_argument("--tensorboard",    action="store_true")
    p_live.add_argument("--checkpoint-dir", default="checkpoints", dest="checkpoint_dir")
    p_live.add_argument("--daily-loss-limit",  type=float, default=0.05, dest="daily_loss_limit")
    p_live.add_argument("--max-drawdown",      type=float, default=0.15, dest="max_drawdown")
    p_live.add_argument("--stop-loss",         type=float, default=0.03, dest="stop_loss")
    p_live.add_argument("--take-profit",        type=float, default=0.08, dest="take_profit")
    p_live.add_argument("--resume-from",        default="", dest="resume_from",
                        help="Resume from a checkpoint JSON (e.g. checkpoints/gen_0007.json)")

    # ---- train ----
    p_train = sub.add_parser("train", help="Offline pre-train a PPO agent")
    p_train.add_argument("--episodes",    type=int,   default=100)
    p_train.add_argument("--episode-len", type=int,   default=200, dest="episode_len")
    p_train.add_argument("--lr",          type=float, default=3e-4)
    p_train.add_argument("--gamma",       type=float, default=0.99)
    p_train.add_argument("--use-mlp",     action="store_true", dest="use_mlp")
    p_train.add_argument("--alt-data",    action="store_true", dest="alt_data")
    p_train.add_argument("--macro",       action="store_true")
    p_train.add_argument("--output",      default="checkpoints/pretrained_ppo.pkl")

    # ---- backtest ----
    p_bt = sub.add_parser("backtest", help="Run walk-forward + Monte Carlo backtest")
    p_bt.add_argument("--data",      default="",  help="Path to OHLCV CSV/JSON (optional)")
    p_bt.add_argument("--symbol",    default="",  help="Filter CSV by symbol (optional)")
    p_bt.add_argument("--use-sample", action="store_true", dest="use_sample",
                        help="Use bundled sample OHLCV (data/AAPL_sample.csv)")
    p_bt.add_argument("--n-bars",    type=int,   default=1000, dest="n_bars")
    p_bt.add_argument("--capital",   type=float, default=100_000)
    p_bt.add_argument("--folds",     type=int,   default=5)
    p_bt.add_argument("--train-pct", type=float, default=0.70, dest="train_pct")
    p_bt.add_argument("--mc-runs",   type=int,   default=1000, dest="mc_runs")
    p_bt.add_argument("--output",    default="reports/backtest.json")

    # ---- validate ----
    p_val = sub.add_parser("validate", help="Production readiness checks for live trading")
    p_val.add_argument("--run-backtest", action="store_true", dest="run_backtest",
                        help="Run OHLCV backtest before validating")
    p_val.add_argument("--fix", action="store_true",
                        help="Print suggested fixes when validation fails")
    p_val.add_argument("--bootstrap", action="store_true",
                        help="DEV ONLY: inject fake paper-trading days into validation log")
    p_val.add_argument("--bootstrap-days", type=int, default=5, dest="bootstrap_days")
    p_val.add_argument("--bootstrap-fitness", type=float, default=0.35, dest="bootstrap_fitness")

    # ---- fetch-data ----
    p_fetch = sub.add_parser("fetch-data", help="Download or generate OHLCV data for backtesting")
    p_fetch.add_argument("--symbol",  default="AAPL")
    p_fetch.add_argument("--days",    type=int, default=365)
    p_fetch.add_argument("--output",  default="")
    p_fetch.add_argument("--offline", action="store_true",
                          help="Generate synthetic OHLCV locally (no network)")

    # ---- status ----
    p_status = sub.add_parser("status", help="Show current system status from checkpoint")
    p_status.add_argument("--log-tail", type=int, default=5, dest="log_tail")

    return parser


def main():
    parser = build_parser()
    args   = parser.parse_args()

    dispatch = {
        "serve":      cmd_serve,
        "run":        cmd_run,
        "pbt-live":   cmd_pbt_live,
        "train":      cmd_train,
        "backtest":   cmd_backtest,
        "validate":   cmd_validate,
        "fetch-data": cmd_fetch_data,
        "status":     cmd_status,
    }
    fn = dispatch.get(args.command)
    if fn:
        fn(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
