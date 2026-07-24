import time

from core.metrics_engine import (
    build_dashboard_state,
    build_trade_analytics,
    compute_performance_metrics,
    compute_risk_metrics,
    compute_trade_metrics,
    normalize_portfolio_state,
)
from core.pbt_engine import AgentRecord, GenerationResult
from dashboard.state_store import LiveStateStore


def test_normalize_portfolio_state_with_positions():
    portfolio = {
        "cash": 10000.0,
        "equity": 10500.0,
        "daily_pnl": 500.0,
        "total_return_pct": 5.0,
        "positions": {
            "AAPL": {
                "qty": 10.0,
                "avg_cost": 150.0,
                "current_price": 155.0,
                "unrealised_pnl": 50.0,
                "realised_pnl": 0.0,
            }
        },
    }

    state = normalize_portfolio_state(portfolio)

    assert state["cash"] == 10000.0
    assert state["equity"] == 10500.0
    assert state["total_return_pct"] == 5.0
    assert state["unrealized_pnl"] == 50.0
    assert state["realized_pnl"] == 0.0
    assert state["gross_exposure"] == 1550.0
    assert state["net_exposure"] == 1550.0
    assert state["leverage"] == round(1550.0 / 10500.0, 4)
    assert state["positions"][0]["symbol"] == "AAPL"


def test_compute_performance_metrics_basic():
    equity_curve = [100.0, 110.0, 105.0, 115.0]
    metrics = compute_performance_metrics(equity_curve, [10.0, -5.0, 10.0])

    assert metrics["total_return_pct"] == 15.0
    assert metrics["annual_return_pct"] != 0.0
    assert metrics["sharpe"] != 0.0
    assert 0.0 <= metrics["win_rate"] <= 1.0


def test_compute_trade_metrics():
    trades = [
        {"pnl": 10.0, "timestamp": time.time() - 3600},
        {"pnl": -5.0, "timestamp": time.time()},
    ]
    stats = compute_trade_metrics(trades)

    assert stats["total_trades"] == 2
    assert stats["winning_trades"] == 1
    assert stats["losing_trades"] == 1
    assert stats["largest_win"] == 10.0
    assert stats["largest_loss"] == -5.0
    assert stats["win_rate"] == 0.5


def test_compute_risk_metrics_returns_expected_shape():
    equity_curve = [100.0, 110.0, 108.0, 120.0]
    metrics = compute_risk_metrics(equity_curve, [10.0, -3.0, 8.0], current_drawdown=0.05)

    assert metrics["current_drawdown_pct"] == 5.0
    assert metrics["max_drawdown_pct"] >= 0.0
    assert metrics["sharpe"] >= 0.0
    assert 0.0 <= metrics["risk_utilization"] <= 1.0


def test_build_trade_analytics_groups_recent_trades_by_symbol():
    trade_stats = {
        "total_trades": 2,
        "winning_trades": 1,
        "losing_trades": 1,
        "win_rate": 0.5,
        "largest_win": 10.0,
        "largest_loss": -5.0,
        "trade_frequency_per_day": 1.0,
    }
    recent_trades = [
        {"symbol": "AAPL", "pnl": 10.0, "holding_time": 3.0},
        {"symbol": "MSFT", "pnl": -5.0, "holding_time": 2.0},
        {"symbol": "AAPL", "pnl": 2.0, "holding_time": 1.0},
    ]

    analytics = build_trade_analytics(trade_stats, recent_trades, {"total_return_pct": 12.0})

    assert analytics["trade_distribution"]["total_trades"] == 2
    assert analytics["best_performing_symbols"][0]["symbol"] == "AAPL"
    assert analytics["worst_performing_symbols"][0]["symbol"] == "MSFT"
    assert analytics["strategy_performance"][0]["return_pct"] == 12.0


def test_build_dashboard_state_returns_expected_keys():
    agent = AgentRecord(
        agent_id="agent1",
        agent_type="ppo",
        hyperparams={"learning_rate": 0.001},
        fitness=0.5,
        generation=1,
        metrics={"sharpe": 1.0, "win_rate": 0.5},
    )
    result = GenerationResult(
        generation=1,
        elapsed_seconds=1.0,
        population=[agent],
        best_agent=agent,
        mean_fitness=0.5,
        std_fitness=0.0,
        exploit_count=1,
        explore_count=1,
    )

    broker_state = {
        "cash": 10000.0,
        "equity": 10050.0,
        "daily_pnl": 50.0,
        "total_return_pct": 0.5,
        "positions": {},
        "prices": {"AAPL": 150.0},
        "n_trades": 0,
    }
    state = build_dashboard_state(
        result=result,
        broker_state=broker_state,
        broker=type("B", (), {"orders": [], "trade_pnls": lambda self=None: [], "is_connected": lambda self=None: True})(),
        equity_curve=[10000.0, 10050.0],
        trade_pnls=[],
        recent_trades=[],
        prices={"AAPL": 150.0},
        positions={},
        regime="BULL",
    )

    assert state["generation"] == 1
    assert state["equity"] == 10050.0
    assert state["performance"]["total_return_pct"] == 0.5
    assert state["risk"]["max_drawdown_pct"] == 0.0
    assert state["market_intelligence"]["current_market_regime"] == "BULL"
    assert state["agent_leaderboard"][0]["agent_id"] == "agent1"
    assert state["n_trades"] == 0


def test_live_state_store_provider_uses_equity_history_for_metrics():
    store = LiveStateStore()
    store.update({
        "equity": 10500.0,
        "equity_history": [10000.0, 10250.0, 10500.0],
        "trade_pnls": [10.0, -2.0, 5.0],
        "initial_equity": 10000.0,
    })

    provider = store.as_provider()
    state = provider()

    assert "performance" in state
    assert "risk" in state
    assert state["performance"]["total_return_pct"] == 5.0
    assert state["risk"]["max_drawdown_pct"] >= 0.0


def test_build_dashboard_state_exposes_legacy_leaderboard_alias():
    agent = AgentRecord(
        agent_id="agent1",
        agent_type="ppo",
        hyperparams={"learning_rate": 0.001},
        fitness=0.5,
        generation=1,
        metrics={"sharpe": 1.0, "win_rate": 0.5},
    )
    result = GenerationResult(
        generation=1,
        elapsed_seconds=1.0,
        population=[agent],
        best_agent=agent,
        mean_fitness=0.5,
        std_fitness=0.0,
        exploit_count=1,
        explore_count=1,
    )

    state = build_dashboard_state(
        result=result,
        broker_state={
            "cash": 10000.0,
            "equity": 10050.0,
            "daily_pnl": 50.0,
            "total_return_pct": 0.5,
            "positions": {},
            "prices": {"AAPL": 150.0},
            "n_trades": 0,
        },
        broker=type("B", (), {"orders": [], "trade_pnls": lambda self=None: [], "is_connected": lambda self=None: True})(),
        equity_curve=[10000.0, 10050.0],
        trade_pnls=[],
        recent_trades=[],
        prices={"AAPL": 150.0},
        positions={},
        regime="BULL",
    )

    assert "leaderboard" in state
    assert state["leaderboard"] == state["agent_leaderboard"]
