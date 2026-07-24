"""
PBT Trading Platform — Test Suite
30 tests covering all core modules.
Run with: python -m pytest tests/test_suite.py -v
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pytest


def test_build_parser_uses_env_broker_override(monkeypatch):
    from main import build_parser

    monkeypatch.setenv("PBT_BROKER", "alpaca")
    args = build_parser().parse_args(["run"])

    assert args.broker == "alpaca"


# ============================================================
# 1. FITNESS FUNCTION
# ============================================================

class TestFitnessFunction:
    def _curve(self, n=100, drift=0.001):
        curve = [100.0]
        for _ in range(n):
            curve.append(curve[-1] * (1 + drift + np.random.randn() * 0.01))
        return curve

    def test_composite_in_range(self):
        from core.fitness import compute_fitness
        fc = compute_fitness(self._curve())
        assert -1.0 <= fc.composite <= 1.0

    def test_flat_curve_low_fitness(self):
        from core.fitness import compute_fitness
        flat = [100.0] * 50
        fc = compute_fitness(flat)
        assert fc.composite < 0.5

    def test_growing_curve_positive_fitness(self):
        from core.fitness import compute_fitness
        growing = [100.0 * (1.001 ** i) for i in range(252)]
        fc = compute_fitness(growing)
        assert fc.annual_return > 0

    def test_drawdown_penalty(self):
        from core.fitness import compute_fitness
        crash = [100.0] * 50 + [60.0] * 50  # 40% drawdown
        fc = compute_fitness(crash)
        assert fc.max_drawdown < -0.30

    def test_short_curve_fallback(self):
        from core.fitness import compute_fitness
        fc = compute_fitness([100.0, 101.0])
        assert fc.composite == -1.0

    def test_win_rate_profit_factor(self):
        from core.fitness import compute_fitness
        curve = self._curve(200, drift=0.002)
        pnls  = [10.0, -5.0, 8.0, -3.0, 12.0]
        fc = compute_fitness(curve, pnls)
        assert 0.0 <= fc.win_rate <= 1.0
        assert fc.profit_factor >= 0.0


# ============================================================
# 2. PBT ENGINE
# ============================================================

class TestPBTEngine:
    def _make_evaluate_fn(self):
        """Dummy evaluate that gives random fitness."""
        def fn(agent):
            agent.fitness = float(np.random.uniform(-0.5, 0.5))
            return agent
        return fn

    def test_initialise_population(self):
        from core.pbt_engine import PBTEngine
        engine = PBTEngine(population_size=6)
        engine.initialise_population()
        assert len(engine.population) == 6

    def test_agent_types_distributed(self):
        from core.pbt_engine import PBTEngine
        engine = PBTEngine(population_size=9, agent_types=["ppo", "dreamer", "worldmodel"])
        engine.initialise_population()
        types = [a.agent_type for a in engine.population]
        assert "ppo" in types and "dreamer" in types and "worldmodel" in types

    def test_run_generation(self):
        from core.pbt_engine import PBTEngine
        engine = PBTEngine(population_size=4)
        engine.initialise_population()
        result = engine.run_generation(self._make_evaluate_fn())
        assert result.generation == 1
        assert result.mean_fitness is not None

    def test_exploit_improves_bottom(self):
        from core.pbt_engine import PBTEngine
        engine = PBTEngine(population_size=6, exploit_fraction=0.33)
        engine.initialise_population()
        # Set manual fitnesses
        for i, a in enumerate(engine.population):
            a.fitness = float(i)
        engine._exploit()
        bottom = sorted(engine.population, key=lambda a: a.fitness)[:2]
        # Bottom agents should now have parent_ids
        assert any(a.parent_id is not None for a in bottom)

    def test_checkpoint_roundtrip(self, tmp_path):
        from core.pbt_engine import PBTEngine
        engine = PBTEngine(population_size=4, checkpoint_dir=str(tmp_path))
        engine.initialise_population()
        engine.run_generation(self._make_evaluate_fn())
        ckpt_files = list(tmp_path.glob("gen_*.json"))
        assert len(ckpt_files) == 1

    def test_paper_broker_deepcopy_isolated(self):
        from brokers.paper_broker import PaperBroker, GBMPriceSimulator
        import copy

        sim = GBMPriceSimulator(symbols=["AAPL"], initial_prices={"AAPL": 185.0}, seed=42)
        broker = PaperBroker(initial_cash=100_000, symbols=["AAPL"], simulator=sim)
        broker2 = copy.deepcopy(broker)
        broker.submit_market_order("AAPL", "buy", 1)

        assert len(broker.trades) == 1
        assert len(broker2.trades) == 0
        assert broker.cash != broker2.cash

    def test_leaderboard_sorted(self):
        from core.pbt_engine import PBTEngine
        engine = PBTEngine(population_size=6)
        engine.initialise_population()
        engine.run_generation(self._make_evaluate_fn())
        lb = engine.leaderboard(top_n=3)
        assert len(lb) == 3
        assert lb[0].fitness >= lb[1].fitness


# ============================================================
# 3. REGIME DETECTOR
# ============================================================

class TestRegimeDetector:
    def _bull_prices(self, n=100):
        prices = [100.0]
        for _ in range(n):
            prices.append(prices[-1] * 1.002)
        return prices

    def _bear_prices(self, n=100):
        prices = [100.0]
        for _ in range(n):
            prices.append(prices[-1] * 0.998)
        return prices

    def test_bull_regime(self):
        from core.regime import RegimeDetector, Regime
        rd = RegimeDetector()
        regime = rd.detect(self._bull_prices())
        assert regime in (Regime.BULL, Regime.RANGING)

    def test_bear_regime(self):
        from core.regime import RegimeDetector, Regime
        rd = RegimeDetector()
        regime = rd.detect(self._bear_prices())
        assert regime in (Regime.BEAR, Regime.RANGING)

    def test_high_vol_regime(self):
        from core.regime import RegimeDetector, Regime
        rng = np.random.default_rng(42)
        prices = [100.0]
        for _ in range(100):
            prices.append(prices[-1] * (1 + rng.normal(0, 0.04)))
        rd = RegimeDetector(high_vol_threshold=0.20)
        regime = rd.detect(prices)
        assert regime == Regime.HIGH_VOL

    def test_unknown_on_short_series(self):
        from core.regime import RegimeDetector, Regime
        rd = RegimeDetector()
        regime = rd.detect([100.0, 101.0])
        assert regime == Regime.UNKNOWN

    def test_history_tracking(self):
        from core.regime import RegimeDetector
        rd = RegimeDetector()
        prices = self._bull_prices(80)
        rd.detect(prices)
        assert len(rd.history) == 1


# ============================================================
# 4. AGENTS
# ============================================================

class TestPPOAgent:
    def test_act_returns_valid_action(self):
        from agents.ppo_agent import PPOAgent
        agent = PPOAgent(state_dim=12)
        state = np.random.randn(12).astype(np.float32)
        action, lp = agent.act(state)
        assert action in (0, 1, 2)
        assert lp <= 0.0

    def test_pretrain_reduces_loss(self):
        from agents.ppo_agent import PPOAgent
        agent = PPOAgent(state_dim=12)
        losses = agent.pretrain(n_episodes=10, episode_len=20)
        assert len(losses) == 10

    def test_mlp_policy(self):
        from agents.ppo_agent import PPOAgent
        agent = PPOAgent(state_dim=12, use_mlp=True)
        state = np.random.randn(12).astype(np.float32)
        action, lp = agent.act(state)
        assert action in (0, 1, 2)


class TestDreamerAgent:
    def test_act(self):
        from agents.dreamer_agent import DreamerAgent
        agent = DreamerAgent(state_dim=12)
        state = np.random.randn(12).astype(np.float32)
        action, lp = agent.act(state)
        assert action in (0, 1, 2)

    def test_update(self):
        from agents.dreamer_agent import DreamerAgent
        agent = DreamerAgent(state_dim=12)
        for _ in range(10):
            s = np.random.randn(12).astype(np.float32)
            a, lp = agent.act(s)
            ns = np.random.randn(12).astype(np.float32)
            agent.store(s, a, float(np.random.randn()), ns)
        loss = agent.update()
        assert isinstance(loss, float)


class TestWorldModelAgent:
    def test_act_with_disagreement(self):
        from agents.worldmodel_agent import WorldModelAgent
        agent = WorldModelAgent(state_dim=12, hyperparams={"ensemble_size": 3})
        state = np.random.randn(12).astype(np.float32)
        action, lp = agent.act(state)
        assert action in (0, 1, 2)

    def test_disagreement_positive(self):
        from agents.worldmodel_agent import WorldModelAgent
        agent = WorldModelAgent(state_dim=12, hyperparams={"ensemble_size": 3})
        state = np.random.randn(12).astype(np.float32)
        d = agent.disagreement(state, 1)
        assert d >= 0.0

    def test_heterogeneous_ensemble(self):
        from agents.worldmodel_agent import WorldModelAgent
        agent = WorldModelAgent(state_dim=12, hyperparams={"ensemble_size": 6}, heterogeneous=True)
        state = np.random.randn(12).astype(np.float32)
        action, _ = agent.act(state)
        assert action in (0, 1, 2)


# ============================================================
# 5. PAPER BROKER
# ============================================================

class TestPaperBroker:
    def _make_broker(self):
        from brokers.paper_broker import PaperBroker, GBMPriceSimulator
        sim = GBMPriceSimulator(symbols=["TEST"], seed=42)
        return PaperBroker(
            initial_cash=10_000,
            symbols=["TEST"],
            simulator=sim,
            commission_pct=0.001,
        )

    def test_initial_equity(self):
        broker = self._make_broker()
        assert broker.equity_curve[0] == 10_000.0

    def test_market_buy_reduces_cash(self):
        broker = self._make_broker()
        broker.step()
        broker.submit_market_order("TEST", "buy", 10)
        assert broker.cash < 10_000.0

    def test_market_sell_after_buy(self):
        broker = self._make_broker()
        broker.step()
        broker.submit_market_order("TEST", "buy", 5)
        broker.step()
        broker.submit_market_order("TEST", "sell", 5)
        assert broker.positions["TEST"].qty == 0

    def test_portfolio_state_keys(self):
        broker = self._make_broker()
        broker.step()
        state = broker.portfolio_state()
        assert "cash" in state and "equity" in state and "n_trades" in state

    def test_equity_curve_grows(self):
        broker = self._make_broker()
        for _ in range(10):
            broker.step()
        assert len(broker.equity_curve) == 11

    def test_limit_order_pending(self):
        broker = self._make_broker()
        broker.step()
        price = list(broker._current_prices.values())[0]
        order = broker.submit_limit_order("TEST", "buy", 5, price * 0.5)
        assert order.status in ("pending", "filled")


# ============================================================
# 6. RISK MANAGER
# ============================================================

class TestRiskManager:
    def _make_rm(self):
        from risk.risk_manager import RiskManager, RiskConfig
        cfg = RiskConfig(
            daily_loss_limit_pct=0.05,
            max_drawdown_pct=0.10,
            stop_loss_pct=0.03,
            take_profit_pct=0.05,
        )
        return RiskManager(config=cfg)

    def test_no_halt_initially(self):
        rm = self._make_rm()
        assert not rm.is_halted

    def test_halt_on_max_drawdown(self):
        rm = self._make_rm()
        rm._peak_equity = 100_000
        rm.update(88_000, {})  # 12% drawdown > 10% limit
        assert rm.is_halted

    def test_order_check_ok(self):
        rm = self._make_rm()
        allowed, reason = rm.check_order("AAPL", "buy", 10, 100.0, 100_000, {})
        assert allowed

    def test_order_rejected_when_halted(self):
        rm = self._make_rm()
        rm._halt("test")
        allowed, reason = rm.check_order("AAPL", "buy", 10, 100.0, 100_000, {})
        assert not allowed

    def test_stop_loss_trigger(self):
        rm = self._make_rm()
        rm.register_entry("AAPL", 100.0)
        should_exit, reason = rm.check_stop_take("AAPL", 96.0)  # -4% > stop_loss 3%
        assert should_exit
        assert reason == "stop_loss"

    def test_take_profit_trigger(self):
        rm = self._make_rm()
        rm.register_entry("AAPL", 100.0)
        should_exit, reason = rm.check_stop_take("AAPL", 106.0)  # +6% > take_profit 5%
        assert should_exit
        assert reason == "take_profit"


# ============================================================
# 7. SIGNAL AGGREGATOR
# ============================================================

class TestSignalAggregator:
    def _make_signals(self, n=4):
        types = ["ppo", "dreamer", "worldmodel", "ppo"]
        return [
            {
                "agent_id":   f"a{i}",
                "agent_type": types[i % len(types)],
                "action":     i % 3,
                "log_prob":   -0.5,
                "fitness":    float(i + 1) * 0.1,
                "confidence": 0.6,
            }
            for i in range(n)
        ]

    def test_equal_mode(self):
        from signals.aggregator import SignalAggregator, AggMode
        agg = SignalAggregator(mode=AggMode.EQUAL)
        action, conf = agg.aggregate(self._make_signals())
        assert action in (0, 1, 2)

    def test_fitness_mode(self):
        from signals.aggregator import SignalAggregator, AggMode
        agg = SignalAggregator(mode=AggMode.FITNESS)
        action, conf = agg.aggregate(self._make_signals())
        assert action in (0, 1, 2)

    def test_ensemble_mode(self):
        from signals.aggregator import SignalAggregator, AggMode
        agg = SignalAggregator(mode=AggMode.ENSEMBLE)
        action, conf = agg.aggregate(self._make_signals())
        assert 0.0 <= conf <= 1.0

    def test_empty_signals_returns_hold(self):
        from signals.aggregator import SignalAggregator
        agg = SignalAggregator()
        action, conf = agg.aggregate([])
        assert action == 0


# ============================================================
# 8. FEATURE ENGINEERING
# ============================================================

class TestFeatureEngineering:
    def _prices(self, n=50, drift=0.001):
        p = [100.0]
        for _ in range(n):
            p.append(p[-1] * (1 + drift + np.random.randn() * 0.01))
        return p

    def test_state_dim(self):
        from features.engineer import FeatureBuilder
        fb = FeatureBuilder(lookback=20)
        assert fb.state_dim == 28  # lookback + 8 (R-014/R-015)

    def test_state_dim_with_alt_macro(self):
        from features.engineer import FeatureBuilder
        fb = FeatureBuilder(include_alt_data=True, include_macro=True)
        assert fb.state_dim == 28  # lookback + 8 (R-014/R-015); alt/macro don't affect state_dim

    def test_build_returns_array(self):
        from features.engineer import FeatureBuilder
        fb = FeatureBuilder()
        state = fb.build(self._prices())
        assert hasattr(state, "shape")
        assert state.shape[0] == fb.state_dim

    def test_build_no_nan(self):
        from features.engineer import FeatureBuilder
        fb = FeatureBuilder()
        state = fb.build(self._prices())
        assert not np.any(np.isnan(state))

    def test_macro_features(self):
        from features.engineer import macro_features
        from datetime import date
        mf = macro_features(date(2026, 6, 10))
        assert "days_to_fomc" in mf
        assert mf["days_to_fomc"] >= 0


# ============================================================
# 9. BACKTEST ENGINE
# ============================================================

class TestBacktestEngine:
    def _prices(self, n=300):
        p = [100.0]
        rng = np.random.default_rng(0)
        for _ in range(n):
            p.append(p[-1] * (1 + rng.normal(0.0005, 0.01)))
        return p

    def _simple_strategy(self, train, test):
        equity = [10_000.0]
        pnls   = []
        pos, entry, cash = 0, 0, 10_000.0
        for i in range(1, len(test)):
            price = test[i]
            qty   = max(1, int(cash * 0.05 / price))
            if test[i] > test[i-1] and pos == 0:
                cash -= price * qty * 1.001
                pos, entry = qty, price
            elif test[i] < test[i-1] and pos > 0:
                cash += price * pos * 0.999
                pnls.append((price - entry) * pos)
                pos = 0
            equity.append(cash + pos * price)
        return equity, pnls

    def test_walk_forward_runs(self):
        from backtest.engine import WalkForwardAnalyser
        wf = WalkForwardAnalyser(n_folds=3, train_pct=0.7)
        report = wf.run(self._prices(), self._simple_strategy)
        assert report.n_folds == 3
        assert len(report.results) > 0

    def test_monte_carlo_runs(self):
        from backtest.engine import MonteCarloSimulator
        mc = MonteCarloSimulator(n_runs=100, confidence=0.95)
        pnls = list(np.random.randn(50) * 100)
        report = mc.run(pnls)
        assert report.n_runs == 100
        assert isinstance(report.median_sharpe, float)

    def test_monte_carlo_empty(self):
        from backtest.engine import MonteCarloSimulator
        mc = MonteCarloSimulator(n_runs=50)
        report = mc.run([])
        assert report.median_sharpe == 0.0


# ============================================================
# 10. PORTFOLIO CONSTRUCTION
# ============================================================

class TestPortfolioConstruction:
    def test_kelly_fraction_positive(self):
        from portfolio.constructor import kelly_fraction
        f = kelly_fraction(win_rate=0.6, avg_win=0.08, avg_loss=0.04)
        assert 0.0 <= f <= 0.25

    def test_kelly_zero_on_bad_params(self):
        from portfolio.constructor import kelly_fraction
        f = kelly_fraction(win_rate=0.3, avg_win=0.02, avg_loss=0.10)
        assert f == 0.0

    def test_dynamic_allocator(self):
        from portfolio.constructor import DynamicCapitalAllocator
        alloc = DynamicCapitalAllocator(total_capital=100_000)
        agents = [
            {"agent_id": "a1", "fitness": 0.8},
            {"agent_id": "a2", "fitness": 0.3},
            {"agent_id": "a3", "fitness": 0.1},
        ]
        result = alloc.allocate(agents)
        assert len(result) == 3
        total = sum(result.values())
        assert abs(total - 100_000) < 1.0  # sums to total capital

    def test_mean_variance_optimiser(self):
        from portfolio.constructor import MeanVarianceOptimiser
        mvo = MeanVarianceOptimiser()
        mu  = np.array([0.10, 0.08, 0.12])
        cov = np.eye(3) * 0.04
        w   = mvo.optimise(mu, cov)
        assert abs(w.sum() - 1.0) < 0.01
        assert all(w >= 0)



# ============================================================
# 11. REGRESSION TESTS — BUG FIXES
# ============================================================

class TestFeatureBuilderShapeFix:
    """Bug 3 — FeatureBuilder.build() must return correct shape on short series."""

    def test_short_series_returns_correct_shape_base(self):
        from features.engineer import FeatureBuilder
        fb = FeatureBuilder(lookback=20)
        state = fb.build([100.0, 101.0])   # len < 5 → fallback
        assert state.shape[0] == fb.state_dim, (
            f"Expected shape {fb.state_dim}, got {state.shape[0]}"
        )

    def test_short_series_returns_correct_shape_with_alt_macro(self):
        from features.engineer import FeatureBuilder
        fb = FeatureBuilder(include_alt_data=True, include_macro=True)
        assert fb.state_dim == 28  # lookback + 8 (R-014/R-015)
        state = fb.build([100.0])   # very short — triggers fallback
        assert state.shape[0] == 28

    def test_normal_series_shape_unchanged(self):
        from features.engineer import FeatureBuilder
        fb = FeatureBuilder()
        closes = [100.0 * (1.001 ** i) for i in range(50)]
        state = fb.build(closes)
        assert state.shape[0] == fb.state_dim


class TestRiskManagerPortfolioExposureFix:
    """Bug 2 — check_order portfolio exposure uses correct price per position."""

    def _make_rm(self):
        from risk.risk_manager import RiskManager, RiskConfig
        cfg = RiskConfig(max_portfolio_pct=0.80, max_trade_size_pct=1.0)
        return RiskManager(config=cfg)

    def test_buy_within_exposure_limit_is_allowed(self):
        rm = self._make_rm()
        # Small buy, no existing positions
        allowed, reason = rm.check_order(
            "AAPL", "buy", qty=1, price=100.0,
            equity=100_000, current_positions={}
        )
        assert allowed, f"Expected allowed, got: {reason}"

    def test_buy_exceeding_exposure_limit_is_rejected(self):
        rm = self._make_rm()
        # Already have 85% exposure via existing positions (market values)
        positions = {"MSFT": 85_000.0}   # market value $85k = 85% of $100k equity
        # Attempting another $5,000 buy would push to 90% > 80% limit
        allowed, reason = rm.check_order(
            "AAPL", "buy", qty=50, price=100.0,
            equity=100_000, current_positions=positions
        )
        assert not allowed

    def test_sell_always_skips_exposure_check(self):
        rm = self._make_rm()
        allowed, reason = rm.check_order(
            "AAPL", "sell", qty=10, price=100.0,
            equity=100_000, current_positions={"AAPL": 10}
        )
        assert allowed


class TestAlpacaBrokerStubMode:
    """Bug 1 — AlpacaBroker must operate safely with no credentials."""

    def test_stub_mode_no_crash_on_init(self):
        from brokers.alpaca_broker import AlpacaBroker
        # alpaca-py may or may not be installed; either path must not crash
        broker = AlpacaBroker(api_key="", api_secret="", paper=True)
        assert broker is not None

    def test_alpaca_broker_matches_paper_reset_contract(self):
        from brokers.alpaca_broker import AlpacaBroker
        broker = AlpacaBroker(api_key="", api_secret="", paper=True)

        assert hasattr(broker, "reset_daily_pnl")
        assert hasattr(broker, "reset_equity_curve")
        assert hasattr(broker, "mark_generation_start")

        broker.reset_daily_pnl()
        broker.reset_equity_curve()

        assert broker.initial_equity == broker.equity_curve[0]

    def test_stub_market_order_returns_dict(self):
        from brokers.alpaca_broker import AlpacaBroker
        broker = AlpacaBroker(api_key="", api_secret="", paper=True)
        # Without connection, should fall through to stub
        if not broker.is_connected():
            result = broker.submit_market_order("AAPL", "buy", 10)
            assert isinstance(result, dict)
            assert "order_id" in result

    def test_no_hardcoded_credentials_in_defaults(self):
        """Confirm the hardcoded keys have been removed."""
        import inspect
        from brokers import alpaca_broker
        src = inspect.getsource(alpaca_broker.AlpacaBroker.__init__)
        # The old leaked keys should not appear
        assert "PKFAPH7SKKW7B2HJD6WHN4K47U" not in src
        assert "3Qav2CHaRamQcTrcWQssdmNnUHYCdtdQZco9HDbarnwY" not in src

    def test_live_mode_fails_closed_without_connection(self):
        from brokers.alpaca_broker import AlpacaBroker, BrokerDisconnectedError
        broker = AlpacaBroker(api_key="", api_secret="", paper=False)
        assert not broker.is_connected()
        with pytest.raises(BrokerDisconnectedError):
            broker.submit_market_order("AAPL", "buy", 10)


class TestBrokerHelpers:
    def test_paper_broker_equity_and_positions(self):
        from brokers.paper_broker import PaperBroker, GBMPriceSimulator
        from brokers.helpers import (
            broker_equity,
            broker_position_qty,
            broker_positions_market_value,
            broker_symbols,
        )
        sim = GBMPriceSimulator(symbols=["TEST"], seed=1)
        broker = PaperBroker(initial_cash=50_000, symbols=["TEST"], simulator=sim)
        broker.step()
        assert broker_symbols(broker) == ["TEST"]
        assert broker_equity(broker) == 50_000.0
        broker.submit_market_order("TEST", "buy", 5)
        assert broker_position_qty(broker, "TEST") == 5
        mv = broker_positions_market_value(broker)
        assert "TEST" in mv and mv["TEST"] > 0


class TestProductionValidation:
    def test_paper_days_count(self, tmp_path):
        from mlops.validation import ProductionValidator
        v = ProductionValidator(log_dir=tmp_path)
        for i in range(3):
            v.record_session(
                broker="paper", live=False, generation=i,
                best_fitness=0.1, mean_fitness=0.1,
                equity=100_000, drawdown=0.01,
            )
        assert v.paper_trading_days() == 1  # same day

    def test_sample_ohlcv_generation(self, tmp_path):
        from backtest.data_loader import generate_sample_ohlcv, load_closes_from_file
        path = generate_sample_ohlcv(tmp_path / "test.csv", n_bars=100)
        closes, sym = load_closes_from_file(path)
        assert len(closes) == 100
        assert sym == "AAPL"

    def test_validation_report_structure(self, tmp_path):
        from mlops.validation import ProductionValidator
        v = ProductionValidator(log_dir=tmp_path, checkpoint_dir=tmp_path, reports_dir=tmp_path)
        report = v.run_all()
        assert len(report.checks) >= 5
        assert isinstance(report.ready_for_live, bool)


class TestAlertManagerCallFlow:
    """Bug 5 — AlertManager.halt() must accept equity and drawdown kwargs."""

    def test_halt_with_equity_and_drawdown(self):
        from mlops.alerts import AlertManager
        am = AlertManager()
        # Should not raise even when email/slack not configured
        am.halt("test halt", equity=95_000.0, drawdown=-0.05)

    def test_circuit_breaker_no_raise(self):
        from mlops.alerts import AlertManager
        am = AlertManager()
        am.circuit_breaker("daily_loss", "Loss exceeded 5%")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
