"""
Broker-agnostic helpers for the PBT evaluation loop.
Works with PaperBroker and AlpacaBroker via duck typing.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def broker_symbols(broker: Any) -> List[str]:
    return list(getattr(broker, "symbols", ["AAPL"]))


def broker_step(broker: Any) -> Dict[str, float]:
    if hasattr(broker, "step"):
        return broker.step()
    return {}


def broker_price_history(broker: Any, symbol: str) -> List[float]:
    if hasattr(broker, "simulator"):
        return broker.simulator.price_history(symbol)
    if hasattr(broker, "price_history"):
        return broker.price_history(symbol)
    return []


def broker_equity(broker: Any, fallback: float = 100_000.0) -> float:
    curve = getattr(broker, "equity_curve", None)
    if curve:
        return float(curve[-1])
    state = broker.portfolio_state()
    return float(state.get("equity", fallback))


def broker_cash(broker: Any) -> float:
    if hasattr(broker, "cash"):
        return float(broker.cash)
    state = broker.portfolio_state()
    return float(state.get("cash", 0))


def broker_position_qty(broker: Any, symbol: str) -> float:
    positions = getattr(broker, "positions", None)
    if positions and symbol in positions:
        pos = positions[symbol]
        return float(getattr(pos, "qty", pos.get("qty", 0) if isinstance(pos, dict) else 0))
    state = broker.portfolio_state()
    pos_data = state.get("positions", {}).get(symbol, {})
    return float(pos_data.get("qty", 0))


def broker_current_price(broker: Any, symbol: str, fallback: float = 100.0) -> float:
    current = getattr(broker, "_current_prices", None)
    if current and symbol in current:
        return float(current[symbol])
    state = broker.portfolio_state()
    prices = state.get("prices", {})
    if symbol in prices:
        return float(prices[symbol])
    hist = broker_price_history(broker, symbol)
    return float(hist[-1]) if hist else fallback


def broker_positions_market_value(broker: Any) -> Dict[str, float]:
    """Symbol -> market value mapping for risk exposure checks."""
    result: Dict[str, float] = {}
    positions = getattr(broker, "positions", None)
    current = getattr(broker, "_current_prices", None)
    if positions and current:
        for sym, pos in positions.items():
            qty = float(getattr(pos, "qty", 0))
            price = float(current.get(sym, 0))
            if qty > 0:
                result[sym] = qty * price
        return result

    state = broker.portfolio_state()
    for sym, pdata in state.get("positions", {}).items():
        qty = float(pdata.get("qty", 0))
        price = float(pdata.get("current_price", 0))
        if qty > 0:
            result[sym] = qty * price
    return result


def broker_is_market_open(broker: Any) -> bool:
    if hasattr(broker, "is_market_open"):
        return bool(broker.is_market_open())
    return True


def order_succeeded(result: Any) -> bool:
    if result is None:
        return False
    if hasattr(result, "status"):
        return result.status == "filled"
    if isinstance(result, dict):
        status = result.get("status", "")
        return status not in ("error", "rejected", "stub_filled") and "order_id" in result
    return False
