"""
Unified metrics engine for MarketZero dashboard state.

This module normalizes broker portfolio data, computes performance and risk
analytics, and builds a consistent dashboard payload for both retail and
institutional views.
"""

from __future__ import annotations

import math
import statistics
import time
from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np

from core.fitness import (
    annual_return,
    calmar_ratio,
    compute_returns,
    compute_fitness,
    max_drawdown,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
    win_rate,
)
from core.pbt_engine import GenerationResult


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _order_get(order: Any, key: str, default: Any = None) -> Any:
    if isinstance(order, dict):
        return order.get(key, default)
    return getattr(order, key, default)


def _percent(value: float) -> float:
    return round(value * 100.0, 2)


def _position_field(pos: Any, names: List[str], default: Any = 0.0) -> Any:
    if isinstance(pos, dict):
        for name in names:
            if name in pos:
                return pos[name]
        return default
    for name in names:
        if hasattr(pos, name):
            return getattr(pos, name)
    return default


def normalize_portfolio_state(portfolio: Dict[str, Any]) -> Dict[str, Any]:
    cash = _safe_float(portfolio.get("cash"))
    equity = _safe_float(portfolio.get("equity"))
    buying_power = portfolio.get("buying_power")
    buying_power = _safe_float(buying_power, cash)
    total_return = _safe_float(portfolio.get("total_return_pct"))
    daily_pnl = _safe_float(portfolio.get("daily_pnl"))
    unrealized = _safe_float(portfolio.get("unrealized_pnl"))
    realized = _safe_float(portfolio.get("realized_pnl"))
    position_value = 0.0
    positions = []

    raw_positions = portfolio.get("positions") or {}
    if isinstance(raw_positions, list):
        raw_positions = [(None, pos) for pos in raw_positions]
    elif isinstance(raw_positions, dict):
        raw_positions = [(symbol, pos) for symbol, pos in raw_positions.items()]
    else:
        raw_positions = []

    for symbol, pos in raw_positions:
        qty = _safe_float(_position_field(pos, ["qty", "quantity", "size"]))
        current_price = _safe_float(
            _position_field(pos, ["current_price", "price", "mark_price", "last_price"])
        )
        entry_price = _safe_float(
            _position_field(pos, ["entry_price", "avg_cost", "avg_price", "avg_entry_price", "cost_basis"])
        )
        unreal = _safe_float(
            _position_field(pos, ["unrealized_pnl", "unrealised_pnl", "unrealized_pl", "unrealised_pl"])
        )
        real = _safe_float(
            _position_field(pos, ["realized_pnl", "realised_pnl", "realized_pl", "realised_pl"])
        )
        raw_pos_value = _position_field(pos, ["position_value", "market_value", "value"], None)
        value = _safe_float(raw_pos_value, qty * current_price)

        unrealized += unreal
        realized += real
        position_value += abs(value)

        positions.append(
            {
                "symbol": symbol or _position_field(pos, ["symbol", "ticker", "instrument"]),
                "qty": qty,
                "entry_price": entry_price,
                "current_price": current_price,
                "unrealized_pnl": unreal,
                "realized_pnl": real,
                "position_value": round(value, 2),
            }
        )

    net_exposure = 0.0
    gross_exposure = 0.0
    leverage = 0.0

    for pos in positions:
        gross_exposure += abs(pos["position_value"])
        net_exposure += pos["position_value"]

    if equity > 0:
        leverage = gross_exposure / max(equity, 1.0)

    return {
        "cash": round(cash, 2),
        "equity": round(equity, 2),
        "buying_power": round(buying_power, 2),
        "total_return_pct": round(total_return, 2),
        "daily_pnl": round(daily_pnl, 2),
        "unrealized_pnl": round(unrealized, 2),
        "realized_pnl": round(realized, 2),
        "position_value": round(position_value, 2),
        "gross_exposure": round(gross_exposure, 2),
        "net_exposure": round(net_exposure, 2),
        "leverage": round(leverage, 4),
        "positions": sorted(positions, key=lambda p: abs(p["position_value"]), reverse=True),
        "n_trades": _safe_int(portfolio.get("n_trades", 0)),
    }


def compute_performance_metrics(
    equity_curve: List[float],
    trade_pnls: Optional[List[float]] = None,
) -> Dict[str, Any]:
    if not equity_curve:
        return {
            "total_return_pct": 0.0,
            "annual_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "volatility": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "calmar": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
        }

    returns = compute_returns(equity_curve)
    ann = annual_return(equity_curve)
    max_dd = max_drawdown(equity_curve)
    sharpe = sharpe_ratio(returns)
    sortino = sortino_ratio(returns)
    calmar = calmar_ratio(ann, max_dd)
    wr = win_rate(trade_pnls)
    pf = profit_factor(trade_pnls)
    volatility = float(np.std(returns) * math.sqrt(252)) if len(returns) > 0 else 0.0

    total_return_pct = round((equity_curve[-1] / equity_curve[0] - 1.0) * 100.0, 2) if equity_curve[0] != 0 else 0.0
    max_drawdown_pct = round(abs(max_dd) * 100.0, 2)

    return {
        "total_return_pct": total_return_pct,
        "annual_return_pct": round(ann * 100.0, 2),
        "max_drawdown_pct": max_drawdown_pct,
        "volatility": round(volatility, 4),
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
        "calmar": round(calmar, 4),
        "win_rate": round(wr, 4),
        "profit_factor": round(pf, 4),
        "daily_return_pct": None,
        "weekly_return_pct": None,
        "monthly_return_pct": None,
        "rolling_returns": [],
    }


def compute_periodic_returns(equity_curve: List[float]) -> Dict[str, Optional[float]]:
    """Derive simple period returns from an equity curve (assumes daily sampling).

    Returns daily (1-step), weekly (~5-step), monthly (~21-step) simple returns in percent.
    """
    if not equity_curve or len(equity_curve) < 2:
        return {"daily": None, "weekly": None, "monthly": None}
    def pct_return(n: int):
        if len(equity_curve) <= n:
            return None
        start = equity_curve[-(n + 1)]
        end = equity_curve[-1]
        if not start:
            return None
        return round((end / start - 1.0) * 100.0, 4)

    return {
        "daily": pct_return(1),
        "weekly": pct_return(5),
        "monthly": pct_return(21),
    }


def compute_rolling_returns(equity_curve: List[float], window: int = 21) -> List[float]:
    if not equity_curve or len(equity_curve) < 2:
        return []
    rolls = []
    for i in range(window, len(equity_curve)):
        start = equity_curve[i - window]
        end = equity_curve[i]
        if not start:
            rolls.append(0.0)
        else:
            rolls.append(round((end / start - 1.0) * 100.0, 4))
    return rolls


def compute_risk_metrics(
    equity_curve: List[float],
    trade_pnls: Optional[List[float]] = None,
    current_drawdown: float = 0.0,
) -> Dict[str, Any]:
    performance = compute_performance_metrics(equity_curve, trade_pnls)
    max_dd = performance["max_drawdown_pct"] / 100.0
    downside = performance["sortino"]
    recovery = round(_safe_float(performance["annual_return_pct"]) / max(max_dd, 1e-6), 4) if max_dd > 0 else 0.0
    returns = compute_returns(equity_curve)
    var = float(np.percentile(returns, 5)) if len(returns) > 0 else 0.0
    cvar_candidates = [r for r in returns if r <= var]
    cvar = float(np.mean(cvar_candidates)) if cvar_candidates else 0.0

    return {
        "sharpe": performance["sharpe"],
        "sortino": downside,
        "calmar": performance["calmar"],
        "profit_factor": performance["profit_factor"],
        "expectancy": round((performance["win_rate"] * 1.0) - ((1.0 - performance["win_rate"]) * 1.0), 4),
        "max_drawdown_pct": performance["max_drawdown_pct"],
        "current_drawdown_pct": round(current_drawdown * 100.0, 2),
        "recovery_factor": round(recovery, 3),
        "var": round(var, 4),
        "cvar": round(cvar, 4),
        "risk_utilization": round(min(1.0, abs(performance["volatility"] / 0.15)), 4),
        "portfolio_volatility": performance["volatility"],
    }


def compute_trade_metrics(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    pnls = [_safe_float(t.get("pnl")) for t in trades]
    total_trades = len(pnls)
    wins = sum(1 for pnl in pnls if pnl > 0)
    losses = sum(1 for pnl in pnls if pnl < 0)
    largest_win = round(max((p for p in pnls if p > 0), default=0.0), 2)
    largest_loss = round(min((p for p in pnls if p < 0), default=0.0), 2)
    win_rate_value = round(wins / total_trades, 4) if total_trades else 0.0
    avg_win = round(sum((p for p in pnls if p > 0), 0.0) / max(wins, 1), 2) if wins else 0.0
    avg_loss = round(abs(sum((p for p in pnls if p < 0), 0.0)) / max(losses, 1), 2) if losses else 0.0
    expectancy = round((avg_win * win_rate_value) - (avg_loss * (1.0 - win_rate_value)), 4)
    trade_times = [t.get("timestamp") for t in trades if isinstance(t.get("timestamp"), (int, float))]
    trade_frequency = 0.0
    if len(trade_times) >= 2:
        duration_days = max((max(trade_times) - min(trade_times)) / 86400.0, 1.0)
        trade_frequency = round(total_trades / duration_days, 4)

    return {
        "total_trades": total_trades,
        "winning_trades": wins,
        "losing_trades": losses,
        "win_rate": win_rate_value,
        "largest_win": largest_win,
        "largest_loss": largest_loss,
        "expectancy": expectancy,
        "trade_frequency_per_day": trade_frequency,
    }


def compute_pnl_stats(trade_pnls: List[float]) -> Dict[str, Any]:
    """Compute aggregate trade statistics from a complete all-session PnL list.

    Unlike ``compute_trade_metrics`` (which works from recent_trades dicts and
    is limited to the last N entries), this function uses the full broker PnL
    history so that win rate, expectancy, and largest win/loss reflect *all*
    trades taken in the session — not just the most recent window.
    """
    if not trade_pnls:
        return {}
    total = len(trade_pnls)
    wins = sum(1 for p in trade_pnls if p > 0)
    losses = sum(1 for p in trade_pnls if p < 0)
    largest_win = round(max((p for p in trade_pnls if p > 0), default=0.0), 2)
    largest_loss = round(min((p for p in trade_pnls if p < 0), default=0.0), 2)
    wr = round(wins / total, 4) if total else 0.0
    avg_win = round(sum(p for p in trade_pnls if p > 0) / max(wins, 1), 2) if wins else 0.0
    avg_loss = round(abs(sum(p for p in trade_pnls if p < 0)) / max(losses, 1), 2) if losses else 0.0
    expectancy = round((avg_win * wr) - (avg_loss * (1.0 - wr)), 4)
    return {
        "total_trades": total,
        "winning_trades": wins,
        "losing_trades": losses,
        "win_rate": wr,
        "largest_win": largest_win,
        "largest_loss": largest_loss,
        "expectancy": expectancy,
    }


def build_agent_leaderboard(population: List[Any], top_n: int = 12) -> List[Dict[str, Any]]:
    sorted_pop = sorted(population, key=lambda a: a.fitness, reverse=True)
    leaderboard = []
    for i, agent in enumerate(sorted_pop[:top_n]):
        metrics = getattr(agent, "metrics", {}) or {}
        leaderboard.append(
            {
                "rank": i + 1,
                "agent_id": agent.agent_id,
                "agent_type": agent.agent_type,
                "fitness": round(agent.fitness, 4),
                "sharpe": round(metrics.get("sharpe", 0.0), 4),
                "win_rate": round(metrics.get("win_rate", 0.0), 4),
                "trade_count": _safe_int(metrics.get("trade_count", 0)),
                "deployment_target": metrics.get("deployment_target", "Training"),
                "current_status": "Active" if getattr(agent, "alive", True) else "Inactive",
            }
        )
    return leaderboard


def collect_execution_metrics(broker: Any, recent_trades: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    orders = getattr(broker, "orders", []) or []
    total = len(orders)
    rejected = sum(
        1
        for order in orders
        if str(_order_get(order, "status", "")).lower() in {"cancelled", "rejected", "error"}
    )
    partial = sum(
        1
        for order in orders
        if _safe_float(_order_get(order, "filled_qty")) < _safe_float(_order_get(order, "qty"))
        and _order_get(order, "status") == "filled"
    )
    filled = sum(
        1
        for order in orders
        if str(_order_get(order, "status", "")).lower() == "filled"
    )
    avg_slippage = 0.0
    slippages = [
        abs(
            _safe_float(_order_get(order, "filled_price"))
            - _safe_float(_order_get(order, "limit_price", _order_get(order, "price")))
        )
        for order in orders
        if _safe_float(_order_get(order, "filled_price")) and _safe_float(_order_get(order, "qty"))
    ]
    if slippages:
        avg_slippage = round(sum(slippages) / len(slippages), 4)

    # Compute avg fill latency (ms) from recent_trades holding_time field.
    # AlpacaBroker populates holding_time = (filled_at - submitted_at) in hours.
    avg_latency_ms = 0.0
    if recent_trades:
        latencies = [
            t["holding_time"] * 3_600_000  # hours → ms
            for t in recent_trades
            if isinstance(t.get("holding_time"), (int, float)) and t["holding_time"] >= 0
        ]
        if latencies:
            avg_latency_ms = round(sum(latencies) / len(latencies), 1)

    return {
        "average_fill_time_ms": avg_latency_ms,
        "average_slippage": avg_slippage,
        "order_latency_ms": avg_latency_ms,
        "execution_quality": round(1.0 - min(1.0, avg_slippage / 100.0), 4),
        "rejected_orders": rejected,
        "partial_fills": partial,
        "fill_percentage": round((filled / total) * 100.0, 2) if total else 0.0,
    }


def build_infrastructure_metrics(broker: Any) -> Dict[str, Any]:
    status = "connected" if getattr(broker, "positions", None) is not None else "disconnected"
    if hasattr(broker, "is_connected"):
        try:
            connected = broker.is_connected()
            status = "connected" if connected else "disconnected"
        except Exception:
            status = status

    cpu_pct = 0.0
    ram_pct = 0.0
    try:
        import psutil
        cpu_pct = round(psutil.cpu_percent(interval=None), 1)
        ram_pct = round(psutil.virtual_memory().percent, 1)
    except ImportError:
        pass

    return {
        "broker_status": status,
        "market_data_feed_status": "connected",
        "database_status": "ok",
        "api_status": "ok",
        "gpu_usage_pct": 0.0,
        "cpu_usage_pct": cpu_pct,
        "ram_usage_pct": ram_pct,
        "queue_length": 0,
        "inference_latency_ms": 0,
        "prediction_frequency_hz": 0.0,
        "synchronization_status": "synced",
    }


_PRICE_HISTORY_WINDOW = 20
_price_history: Dict[str, deque] = {}
_return_history: deque = deque(maxlen=_PRICE_HISTORY_WINDOW)


def _get_price_history(symbol: str) -> deque:
    hist = _price_history.get(symbol)
    if hist is None:
        hist = deque(maxlen=_PRICE_HISTORY_WINDOW)
        _price_history[symbol] = hist
    return hist


def build_market_intelligence(
    regime: str,
    prices: Dict[str, float],
    positions: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    volatility = 0.0
    trend_direction = "Neutral"
    trend_strength = "weak"

    if prices:
        symbol_returns = []
        for symbol, price in prices.items():
            hist = _get_price_history(symbol)
            hist.append(_safe_float(price))
            if len(hist) >= 2 and hist[0]:
                symbol_returns.append(hist[-1] / hist[0] - 1.0)

        if symbol_returns:
            pct_move = float(np.mean(symbol_returns))
            _return_history.append(pct_move)
            if len(_return_history) >= 2:
                volatility = float(np.std(list(_return_history)) * math.sqrt(252))

            trend_direction = "Bullish" if pct_move > 0 else ("Bearish" if pct_move < 0 else "Neutral")
            magnitude = abs(pct_move)
            if magnitude > 0.05:
                trend_strength = "strong"
            elif magnitude > 0.02:
                trend_strength = "moderate"
            else:
                trend_strength = "weak"

    regime_confidence = 0.75 if regime != "UNKNOWN" else 0.0
    dominant_features = ["momentum", "volatility", "liquidity"]
    if positions:
        dominant_features.insert(0, positions[0].get("symbol", "n/a"))

    return {
        "current_market_regime": regime,
        "regime_confidence": round(regime_confidence, 4),
        "volatility_state": "high" if volatility > 0.05 else "low",
        "trend_direction": trend_direction,
        "trend_strength": trend_strength,
        "liquidity_state": "stable",
        "time_since_regime_change_sec": None,
        "dominant_market_features": dominant_features,
    }


def build_portfolio_analytics(portfolio: Dict[str, Any], performance: Dict[str, Any]) -> Dict[str, Any]:
    positions = portfolio.get("positions", [])
    allocation_by_strategy = [{"strategy": "Default", "allocation_pct": 100.0}] if not positions else []
    allocation_by_agent = [{"agent": "All", "allocation_pct": 100.0}] if not positions else []
    return {
        "aum": portfolio.get("equity", 0.0),
        "equity": portfolio.get("equity", 0.0),
        "cash": portfolio.get("cash", 0.0),
        "margin_usage": round(min(1.0, abs(portfolio.get("gross_exposure", 0.0)) / max(portfolio.get("equity", 1.0), 1.0)), 4),
        "buying_power": portfolio.get("buying_power", 0.0),
        "exposure": round(portfolio.get("net_exposure", 0.0), 2),
        "leverage": portfolio.get("leverage", 0.0),
        "gross_exposure": portfolio.get("gross_exposure", 0.0),
        "net_exposure": portfolio.get("net_exposure", 0.0),
        "capital_allocation_by_strategy": allocation_by_strategy,
        "capital_allocation_by_agent": allocation_by_agent,
    }


def build_trade_analytics(trade_stats: Dict[str, Any], recent_trades: List[Dict[str, Any]], performance: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    performance = performance or {}
    avg_holding = round(sum((t.get("holding_time", 0.0) for t in recent_trades), 0.0) / max(len(recent_trades), 1), 2)
    symbol_performance = {}
    for t in recent_trades:
        symbol = t.get("symbol", "n/a")
        symbol_performance.setdefault(symbol, {"pnl": 0.0, "count": 0})
        symbol_performance[symbol]["pnl"] += _safe_float(t.get("pnl"))
        symbol_performance[symbol]["count"] += 1
    best_symbols = sorted(symbol_performance.items(), key=lambda x: x[1]["pnl"], reverse=True)[:3]
    worst_symbols = sorted(symbol_performance.items(), key=lambda x: x[1]["pnl"])[:3]
    return {
        "trade_distribution": {
            "total_trades": trade_stats.get("total_trades", 0),
            "winning_trades": trade_stats.get("winning_trades", 0),
            "losing_trades": trade_stats.get("losing_trades", 0),
        },
        "win_loss_distribution": {
            "win_rate": trade_stats.get("win_rate", 0.0),
            "largest_win": trade_stats.get("largest_win", 0.0),
            "largest_loss": trade_stats.get("largest_loss", 0.0),
        },
        "average_holding_time": avg_holding,
        "trade_frequency": trade_stats.get("trade_frequency_per_day", 0.0),
        "best_performing_symbols": [{"symbol": k, "pnl": round(v["pnl"], 2)} for k, v in best_symbols],
        "worst_performing_symbols": [{"symbol": k, "pnl": round(v["pnl"], 2)} for k, v in worst_symbols],
        "strategy_performance": [{"strategy": "Default", "return_pct": performance.get("total_return_pct", 0.0)}],
    }


def build_ai_monitoring(result: GenerationResult, broker: Any) -> Dict[str, Any]:
    """Build AI monitoring metrics from the live PBT generation result.

    Previously this returned mostly hardcoded defaults because the broker
    doesn't expose ML training attributes.  Now we derive meaningful values
    directly from the ``GenerationResult`` so the panel reflects real state.
    """
    best_agent = result.best_agent
    # Determine the current phase based on exploit vs. explore counts
    policy = "exploit" if result.exploit_count >= result.explore_count else "explore"
    checkpoint_label = f"gen-{result.generation}"
    return {
        "active_model": getattr(broker, "model_version",
                               getattr(best_agent, "agent_type", "pbt-agent")),
        "current_policy": getattr(broker, "current_policy", policy),
        "model_version": getattr(broker, "model_version", checkpoint_label),
        "current_checkpoint": getattr(broker, "current_checkpoint", checkpoint_label),
        "replay_buffer_size": getattr(broker, "replay_buffer_size", len(result.population)),
        "episodes_completed": getattr(broker, "episodes_completed", result.generation),
        "training_epoch": getattr(broker, "training_epoch", result.generation),
        "last_validation_score": round(
            getattr(broker, "last_validation_score", result.best_agent.fitness), 4
        ),
        "best_validation_score": round(
            getattr(broker, "best_validation_score", result.best_agent.fitness), 4
        ),
        "model_drift_detection": getattr(broker, "model_drift_detection", "stable"),
    }


def build_live_decision_stream(broker: Any) -> List[Dict[str, Any]]:
    return getattr(broker, "live_decisions", []) or []


def build_alerts_events(notifications: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return notifications or []


def build_ai_evolution_metrics(result: GenerationResult) -> Dict[str, Any]:
    fitnesses = [agent.fitness for agent in result.population]
    return {
        "current_generation": result.generation,
        "population_size": len(result.population),
        "active_agents": len([a for a in result.population if getattr(a, "alive", True)]),
        "elite_agents": len([a for a in result.population if a.fitness >= 0.75]),
        "mutated_agents": result.explore_count,
        "crossover_count": result.exploit_count,
        "current_evolution_phase": "exploit/explore",
        "evolution_speed": round(1.0 / max(result.elapsed_seconds, 1e-6), 4),
        "best_fitness": round(result.best_agent.fitness, 4),
        "average_fitness": round(result.mean_fitness, 4),
        "median_fitness": round(statistics.median(fitnesses) if fitnesses else 0.0, 4),
        "worst_fitness": round(min(fitnesses) if fitnesses else 0.0, 4),
        "fitness_stability": round(statistics.stdev(fitnesses) if len(fitnesses) > 1 else 0.0, 4),
    }


def build_notifications(
    result: GenerationResult,
    broker_state: Dict[str, Any],
    risk_halted: bool,
    broker: Any = None,
) -> List[Dict[str, Any]]:
    notifications = [
        {
            "type": "generation",
            "title": "Generation Complete",
            "message": f"Gen {result.generation} complete. Best fitness {round(result.best_agent.fitness,4)}.",
            "timestamp": time.time(),
        }
    ]
    if risk_halted:
        notifications.append(
            {
                "type": "risk",
                "title": "Trading Halted",
                "message": "Risk manager has halted the system.",
                "timestamp": time.time(),
            }
        )
    if broker is not None and hasattr(broker, "is_connected") and not broker.is_connected():
        notifications.append(
            {
                "type": "alert",
                "title": "Broker Disconnected",
                "message": "Broker is not connected — portfolio, positions, and trade data may be stale or unavailable.",
                "timestamp": time.time(),
            }
        )
    if broker_state.get("buying_power", 0) <= 0:
        notifications.append(
            {
                "type": "alert",
                "title": "No Buying Power",
                "message": "Buying power is depleted.",
                "timestamp": time.time(),
            }
        )
    return notifications


def build_dashboard_state(
    result: GenerationResult,
    broker_state: Dict[str, Any],
    broker: Any,
    equity_curve: List[float],
    trade_pnls: List[float],
    recent_trades: List[Dict[str, Any]],
    prices: Dict[str, float],
    positions: Dict[str, Any],
    regime: str = "UNKNOWN",
) -> Dict[str, Any]:
    portfolio = normalize_portfolio_state(broker_state)
    performance = compute_performance_metrics(equity_curve, trade_pnls)
    risk = compute_risk_metrics(equity_curve, trade_pnls, current_drawdown=broker_state.get("drawdown", 0.0))
    trade_stats = compute_trade_metrics(recent_trades if recent_trades else [])
    # Merge all-session PnL stats so that win rate, expectancy, and largest
    # win/loss reflect every trade taken this session — not just the last N.
    if trade_pnls:
        trade_stats.update(compute_pnl_stats(trade_pnls))
    portfolio_analytics = build_portfolio_analytics(portfolio, performance)
    trade_analytics = build_trade_analytics(trade_stats, recent_trades, performance)
    ai_evolution = build_ai_evolution_metrics(result)
    ai_monitoring = build_ai_monitoring(result, broker)
    live_decision_stream = build_live_decision_stream(broker)
    notifications = build_notifications(result, portfolio, risk_halted=broker_state.get("halted", False), broker=broker)
    alerts_events = build_alerts_events(notifications)
    execution = collect_execution_metrics(broker, recent_trades)
    infrastructure = build_infrastructure_metrics(broker)
    market_intelligence = build_market_intelligence(regime, prices, portfolio.get("positions"))
    agent_leaderboard = build_agent_leaderboard(result.population, top_n=min(12, len(result.population)))

    # augment performance with periodic and rolling returns
    periodic = compute_periodic_returns(equity_curve)
    performance["daily_return_pct"] = periodic.get("daily")
    performance["weekly_return_pct"] = periodic.get("weekly")
    performance["monthly_return_pct"] = periodic.get("monthly")
    performance["rolling_returns"] = compute_rolling_returns(equity_curve, window=21)

    # fitness distribution / history
    fitness_history_vals = getattr(result, 'fitness_history', None) or []
    if not fitness_history_vals and result.population:
        fitness_history_vals = [a.fitness for a in result.population]

    fitness_distribution = {
        "best": round(result.best_agent.fitness, 4),
        "average": round(result.mean_fitness, 4),
        "median": round(statistics.median(fitness_history_vals) if fitness_history_vals else result.mean_fitness, 4),
        "worst": round(min(fitness_history_vals) if fitness_history_vals else result.best_agent.fitness, 4),
        "std": round(statistics.stdev(fitness_history_vals) if len(fitness_history_vals) > 1 else 0.0, 4),
        "improvement": round((fitness_history_vals[-1] - fitness_history_vals[0]) if len(fitness_history_vals) > 1 else 0.0, 4),
        "history": fitness_history_vals,
    }

    return {
        "leaderboard": agent_leaderboard,
        "portfolio_analytics": portfolio_analytics,
        "trade_analytics": trade_analytics,
        "ai_monitoring": ai_monitoring,
        "live_decision_stream": live_decision_stream,
        "alerts_events": alerts_events,
        "notifications": notifications,
        "performance": performance,
        "risk": risk,
        "execution": execution,
        "infrastructure": infrastructure,
        "market_intelligence": market_intelligence,
        "ai_evolution": ai_evolution,
        "agent_leaderboard": agent_leaderboard,
        "generation": result.generation,
        "best_fitness": round(result.best_agent.fitness, 4),
        "mean_fitness": round(result.mean_fitness, 4),
        "std_fitness": round(result.std_fitness, 4),
        "equity": portfolio["equity"],
        "cash": portfolio["cash"],
        "buying_power": portfolio["buying_power"],
        "total_return_pct": portfolio["total_return_pct"],
        "daily_pnl": portfolio["daily_pnl"],
        "unrealized_pnl": portfolio["unrealized_pnl"],
        "realized_pnl": portfolio["realized_pnl"],
        "drawdown": round(broker_state.get("drawdown", 0.0), 4),
        "regime": regime,
        "halted": bool(broker_state.get("halted", False)),
        "n_trades": broker_state.get("n_trades", portfolio.get("n_trades", 0)),
        "leaderboard": agent_leaderboard,
        "recent_trades": recent_trades,
        "equity_history": equity_curve[:],
        "fitness_history": getattr(result, 'fitness_history', [round(result.best_agent.fitness, 4)]),
        "mean_history": getattr(result, 'mean_history', [round(result.mean_fitness, 4)]),
        "prices": {k: round(v, 4) for k, v in (prices or {}).items()},
        "positions": portfolio.get("positions", []),
        "exploit_count": result.exploit_count,
        "explore_count": result.explore_count,
        "elapsed": round(result.elapsed_seconds, 2),
        "portfolio": portfolio,
        "trade_stats": trade_stats,
        "fitness_distribution": fitness_distribution,
    }