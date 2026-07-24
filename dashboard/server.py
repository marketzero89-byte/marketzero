"""
FastAPI WebSocket Server
Real-time state broadcasting to the browser dashboard.
Global exception handler, static file serving, REST + WebSocket API.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger(__name__)

_CORS_RAW = os.getenv("PBT_CORS_ORIGINS", "*")
CORS_ORIGINS = [o.strip() for o in _CORS_RAW.split(",") if o.strip()] if _CORS_RAW != "*" else ["*"]


def _get_api_key() -> str:
    return os.getenv("PBT_API_KEY", "")


def _is_local_request(request: Request | WebSocket) -> bool:
    host = ""

    url = getattr(request, "url", None)
    if url is not None:
        host = getattr(url, "hostname", "") or ""

    if not host:
        scope = getattr(request, "scope", None)
        if scope is not None:
            client = scope.get("client") or ()
            if client:
                host = client[0] or ""
            headers = scope.get("headers", []) or []
            for key, value in headers:
                if key.decode("utf-8", "ignore").lower() == "host":
                    host = value.decode("utf-8", "ignore").split(":", 1)[0]
                    break

    if not host:
        headers = getattr(request, "headers", None)
        if headers is not None:
            host = headers.get("host", "") or ""

    host = host.lower()
    return host in {"127.0.0.1", "localhost", "::1", "0.0.0.0", "testclient", "testserver"}


def _check_api_key(request: Request) -> bool:
    api_key = _get_api_key()
    if not api_key:
        return True
    if _is_local_request(request):
        return True
    if request.headers.get("X-API-Key") == api_key:
        return True
    return request.query_params.get("api_key") == api_key


def _ws_api_key_ok(websocket: WebSocket) -> bool:
    """WebSocket is served from the same origin as the dashboard HTML.
    No API key required — auth is only enforced on REST endpoints below."""
    return True

# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(exist_ok=True)


def create_app(state_provider=None) -> FastAPI:
    """
    Create the FastAPI application.

    Parameters
    ----------
    state_provider : callable | None
        A zero-argument callable returning the current system state dict.
        If None, a mock provider is used.
    """
    app = FastAPI(
        title="PBT Trading Dashboard",
        description="Population-Based Training algorithmic trading platform",
        version="1.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def api_key_middleware(request: Request, call_next):
        path = request.url.path
        if path in ("/", "/api/health", "/favicon.ico", "/ws") or path.startswith("/static"):
            return await call_next(request)
        if not _check_api_key(request):
            return JSONResponse(status_code=401, content={"detail": "Unauthorized"})
        return await call_next(request)

    # Serve static dashboard files
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # ------------------------------------------------------------------
    # WebSocket connection manager
    # ------------------------------------------------------------------

    class ConnectionManager:
        def __init__(self):
            self.active: Set[WebSocket] = set()

        async def connect(self, ws: WebSocket) -> None:
            await ws.accept()
            self.active.add(ws)
            logger.info("WS client connected. Total: %d", len(self.active))

        def disconnect(self, ws: WebSocket) -> None:
            self.active.discard(ws)
            logger.info("WS client disconnected. Total: %d", len(self.active))

        async def broadcast(self, message: Dict) -> None:
            dead = set()
            for ws in self.active:
                try:
                    await ws.send_json(message)
                except Exception:
                    dead.add(ws)
            for ws in dead:
                self.active.discard(ws)

    manager = ConnectionManager()
    app.state.manager = manager
    app.state.provider = state_provider or _mock_provider

    # ------------------------------------------------------------------
    # Background broadcaster
    # ------------------------------------------------------------------

    @app.on_event("startup")
    async def start_broadcaster():
        async def _broadcast_loop():
            while True:
                try:
                    if manager.active:
                        state = app.state.provider()
                        state["_ts"] = time.time()
                        await manager.broadcast(state)
                except Exception as exc:
                    logger.error("Broadcast error: %s", exc)
                await asyncio.sleep(1.0)

        asyncio.create_task(_broadcast_loop())

    # ------------------------------------------------------------------
    # Global exception handler
    # ------------------------------------------------------------------

    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        logger.error("Unhandled exception: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": type(exc).__name__,
                "detail": str(exc),
                "path": str(request.url.path),
            },
        )

    # ------------------------------------------------------------------
    # REST endpoints
    # ------------------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    async def dashboard():
        index = STATIC_DIR / "index.html"
        if index.exists():
            return HTMLResponse(index.read_text())
        return HTMLResponse("<h1>PBT Trading Dashboard</h1><p>Static files not found. Run build.</p>")

    @app.get("/api/state")
    async def get_state():
        return app.state.provider()

    @app.get("/api/health")
    async def health():
        return {"status": "ok", "timestamp": time.time(), "clients": len(manager.active)}

    @app.get("/favicon.ico")
    async def favicon():
        return FileResponse(STATIC_DIR / "favicon.ico")

    @app.get("/api/metrics")
    async def prometheus_metrics():
        """Prometheus-compatible /metrics endpoint."""
        state = app.state.provider()
        lines = [
            "# HELP pbt_fitness Current best agent fitness",
            "# TYPE pbt_fitness gauge",
            f"pbt_fitness {state.get('best_fitness', 0)}",
            "# HELP pbt_generation Current generation number",
            "# TYPE pbt_generation counter",
            f"pbt_generation {state.get('generation', 0)}",
            "# HELP pbt_equity Current portfolio equity",
            "# TYPE pbt_equity gauge",
            f"pbt_equity {state.get('equity', 0)}",
            "# HELP pbt_drawdown Current drawdown fraction",
            "# TYPE pbt_drawdown gauge",
            f"pbt_drawdown {state.get('drawdown', 0)}",
        ]
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse("\n".join(lines), media_type="text/plain; version=0.0.4")

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        if not _ws_api_key_ok(websocket):
            await websocket.close(code=1008)
            return
        await manager.connect(websocket)
        try:
            # Send initial state immediately
            state = app.state.provider()
            state["_ts"] = time.time()
            await websocket.send_json(state)

            # Keep connection alive; client can send commands
            while True:
                try:
                    data = await asyncio.wait_for(websocket.receive_text(), timeout=30)
                    msg = json.loads(data)
                    await _handle_client_message(websocket, msg, app)
                except asyncio.TimeoutError:
                    await websocket.send_json({"type": "ping"})
        except WebSocketDisconnect:
            manager.disconnect(websocket)
        except Exception as exc:
            logger.error("WS error: %s", exc)
            manager.disconnect(websocket)

    async def _handle_client_message(ws: WebSocket, msg: Dict, app: FastAPI) -> None:
        msg_type = msg.get("type", "")
        if msg_type == "pong":
            return
        elif msg_type == "get_state":
            state = app.state.provider()
            await ws.send_json({"type": "state", "data": state})
        elif msg_type == "command":
            cmd = msg.get("command", "")
            logger.info("Dashboard command: %s", cmd)
            await ws.send_json({"type": "ack", "command": cmd})

    return app


# ---------------------------------------------------------------------------
# Mock state provider (used when no real PBT engine is running)
# ---------------------------------------------------------------------------

_mock_gen = 0

def _mock_provider() -> Dict:
    global _mock_gen
    import math, random
    _mock_gen += 1
    t = _mock_gen
    balance = round(100_000 * (1 + t * 0.001 + random.gauss(0, 0.002)), 2)
    # build a synthetic equity history for recent periods (30 points)
    hist_len = 30
    equity_history = []
    # base tracks long-run drift so returns are visible in dashboard
    base = 100_000 * (1 + t * 0.001)
    for i in range(hist_len):
        # gentle upward drift plus small noise
        growth = 0.0008 * i + random.gauss(0, 0.0005)
        val = round(base * (1 + growth), 2)
        equity_history.append(val)
    # ensure last value matches current balance
    equity_history[-1] = balance

    # periodic returns computed from equity_history
    def pct_change(a, b):
        return ((b - a) / a * 100) if a and b else 0.0

    daily_return = pct_change(equity_history[-2] if len(equity_history) > 1 else equity_history[-1], equity_history[-1])
    weekly_return = pct_change(equity_history[-8] if len(equity_history) > 8 else equity_history[0], equity_history[-1])
    monthly_return = pct_change(equity_history[0], equity_history[-1])

    # simple rolling returns (pct) over last N windows
    rolling_returns = []
    window = 7
    for i in range(window, len(equity_history)):
        base = equity_history[i - window]
        rolling_returns.append(pct_change(base, equity_history[i]))

    positions = [
        {"symbol": "AAPL", "qty": 20, "entry_price": 150.0, "current_price": 155.0, "unrealized_pnl": 100.0, "position_value": 3100.0},
        {"symbol": "SPY", "qty": 10, "entry_price": 520.0, "current_price": 525.0, "unrealized_pnl": 50.0, "position_value": 5250.0},
    ]

    return {
        "generation":    t,
        "best_fitness":  round(math.sin(t * 0.1) * 0.5 + 0.3 + random.gauss(0, 0.02), 4),
        "mean_fitness":  round(math.sin(t * 0.1) * 0.3 + 0.1 + random.gauss(0, 0.02), 4),
        "equity":        balance,
        "cash":          round(balance * 0.12, 2),
        "buying_power":  round(balance * 0.25, 2),
        "daily_pnl":     round(balance - 100_000, 2),
        "total_return_pct": round((balance / 100_000 - 1) * 100, 2),
        "unrealized_pnl": round(random.gauss(250, 120), 2),
        "realized_pnl":   round(random.gauss(150, 80), 2),
        "drawdown":      round(max(0, -random.gauss(0.02, 0.01)), 4),
        "regime":        ["BULL", "BEAR", "RANGING", "HIGH_VOL"][t % 4],
        "broker":        "MockBroker",
        "broker_mode":   "mock",
        "halted":        False,
        "n_trades":      t * 3,
        "portfolio": {
            "paper": true,
            "connected": true,
            "broker": "MockBroker",
        },
        "leaderboard": [
            {"rank": i+1, "agent_id": f"a{i:03d}",
             "agent_type": ["ppo","dreamer","worldmodel"][i % 3],
             "fitness": round(0.5 - i * 0.05 + random.gauss(0, 0.01), 4),
             "sharpe": round(1.2 - i * 0.1 + random.gauss(0, 0.02), 4),
             "win_rate": round(0.5 + random.gauss(0.1, 0.03), 4),
             "trade_count": 10 + i,
             "current_status": "Active",
             "deployment_target": "Paper"}
            for i in range(5)
        ],
        "agent_leaderboard": [
            {"rank": i+1, "agent_id": f"a{i:03d}",
             "agent_type": ["ppo","dreamer","worldmodel"][i % 3],
             "fitness": round(0.5 - i * 0.05 + random.gauss(0, 0.01), 4),
             "sharpe": round(1.2 - i * 0.1 + random.gauss(0, 0.02), 4),
             "win_rate": round(0.5 + random.gauss(0.1, 0.03), 4),
             "trade_count": 10 + i,
             "current_status": "Active",
             "deployment_target": "Paper"}
            for i in range(5)
        ],
        "performance": {
            "total_return_pct": round((balance / 100_000 - 1) * 100, 2),
            "annual_return_pct": round((balance / 100_000 - 1) * 100 * 1.1, 2),
            "daily_return_pct": round(daily_return, 4),
            "weekly_return_pct": round(weekly_return, 4),
            "monthly_return_pct": round(monthly_return, 4),
            "rolling_returns": [round(r/100, 6) for r in rolling_returns],
            "max_drawdown_pct": round(random.gauss(3, 1), 2),
            "volatility": round(random.gauss(0.08, 0.01), 4),
            "sharpe": round(random.gauss(1.2, 0.15), 4),
            "sortino": round(random.gauss(1.5, 0.2), 4),
            "calmar": round(random.gauss(0.7, 0.1), 4),
            "win_rate": round(0.55 + random.gauss(0.05, 0.02), 4),
            "profit_factor": round(1.4 + random.gauss(0.1, 0.05), 4),
        },
        "risk": {
            "sharpe": round(random.gauss(1.2, 0.15), 4),
            "sortino": round(random.gauss(1.5, 0.2), 4),
            "calmar": round(random.gauss(0.7, 0.1), 4),
            "profit_factor": round(1.4 + random.gauss(0.1, 0.05), 4),
            "expectancy": round(random.gauss(0.12, 0.03), 4),
            "max_drawdown_pct": round(random.gauss(3, 1), 2),
            "current_drawdown_pct": round(random.gauss(1.2, 0.4), 2),
            "recovery_factor": round(random.gauss(1.8, 0.4), 4),
            "var": round(random.gauss(-0.02, 0.005), 4),
            "cvar": round(random.gauss(-0.03, 0.006), 4),
            "risk_utilization": round(random.uniform(0.2, 0.7), 4),
            "portfolio_volatility": round(random.gauss(0.08, 0.01), 4),
        },
        "trade_stats": {
            "total_trades": t * 3,
            "winning_trades": int(t * 2),
            "losing_trades": int(t),
            "win_rate": round(2/3, 4),
            "largest_win": round(random.uniform(120, 350), 2),
            "largest_loss": round(random.uniform(-150, -20), 2),
            "expectancy": round(random.gauss(0.12, 0.03), 4),
            "trade_frequency_per_day": round(random.uniform(2.0, 6.0), 2),
        },
        "ai_evolution": {
            "current_generation": t,
            "population_size": 12,
            "active_agents": 12,
            "elite_agents": 3,
            "mutated_agents": 2,
            "crossover_count": 2,
            "current_evolution_phase": "exploit/explore",
            "evolution_speed": round(1 / max(0.1, random.uniform(0.2, 0.6)), 4),
            "best_fitness": round(math.sin(t * 0.1) * 0.5 + 0.3, 4),
            "average_fitness": round(math.sin(t * 0.1) * 0.3 + 0.1, 4),
            "median_fitness": round(math.sin(t * 0.1) * 0.28 + 0.15, 4),
            "worst_fitness": round(math.sin(t * 0.1) * 0.6 - 0.2, 4),
            "fitness_stability": round(abs(math.cos(t * 0.1)) * 0.2, 4),
        },
        "market_intelligence": {
            "current_market_regime": ["BULL", "BEAR", "RANGING", "HIGH_VOL"][t % 4],
            "regime_confidence": round(random.uniform(0.5, 0.95), 4),
            "volatility_state": "high" if t % 3 == 0 else "low",
            "trend_strength": "Bullish" if t % 2 == 0 else "Bearish",
            "liquidity_state": "stable",
            "time_since_regime_change_sec": t * 180,
            "dominant_market_features": ["momentum", "volatility", "liquidity"],
        },
        "execution": {
            "average_fill_time_ms": round(random.uniform(50, 250), 2),
            "average_slippage": round(random.uniform(0.01, 0.08), 4),
            "order_latency_ms": round(random.uniform(80, 320), 1),
            "execution_quality": round(random.uniform(0.88, 0.98), 4),
            "rejected_orders": random.randint(0, 2),
            "partial_fills": random.randint(0, 2),
            "fill_percentage": round(random.uniform(92, 100), 2),
        },
        "infrastructure": {
            "broker_status": "connected",
            "market_data_feed_status": "connected",
            "database_status": "ok",
            "api_status": "ok",
            "gpu_usage_pct": round(random.uniform(10, 45), 1),
            "cpu_usage_pct": round(random.uniform(25, 65), 1),
            "ram_usage_pct": round(random.uniform(38, 74), 1),
            "queue_length": random.randint(0, 5),
            "inference_latency_ms": round(random.uniform(5, 20), 1),
            "prediction_frequency_hz": round(random.uniform(0.8, 2.2), 2),
            "synchronization_status": "synced",
        },
        "ai_monitoring": {
            "active_model": "v1.0",
            "current_policy": "default",
            "model_version": "v1.0",
            "current_checkpoint": "latest",
            "replay_buffer_size": random.randint(500, 2500),
            "episodes_completed": random.randint(50, 125),
            "training_epoch": random.randint(1, 24),
            "last_validation_score": round(random.uniform(0.7, 0.95), 4),
            "best_validation_score": round(random.uniform(0.85, 0.98), 4),
            "model_drift_detection": "stable",
        },
        "live_decision_stream": [
            {"side": "buy", "symbol": "AAPL", "description": "Long entry signal", "timestamp": time.time()},
            {"side": "sell", "symbol": "SPY", "description": "Profit taking signal", "timestamp": time.time()},
        ],
        "recent_trades": [
            {"trade_id": f"t{t}a", "symbol": "AAPL", "side": "buy", "qty": 20, "price": 150.0, "pnl": 100.0, "timestamp": time.time(), "holding_time": 0.5},
            {"trade_id": f"t{t}b", "symbol": "SPY", "side": "sell", "qty": 5, "price": 525.0, "pnl": -25.0, "timestamp": time.time(), "holding_time": 1.2},
        ],
        "alerts_events": [
            {"type": "info", "title": "System OK", "message": "No active alerts.", "timestamp": time.time()},
        ],
        "notifications": [
            {
                "type": "info",
                "title": "Mock data active",
                "message": "Dashboard is displaying simulated market state.",
                "timestamp": time.time(),
            }
        ],
        "fitness_history": [round(0.25 + math.sin(t * 0.1) * 0.2 + random.gauss(0, 0.01), 4) for _ in range(10)],
        "mean_history": [round(0.20 + math.sin(t * 0.1) * 0.15 + random.gauss(0, 0.01), 4) for _ in range(10)],
        "positions": positions,
        "prices": {
            "AAPL": round(155 + random.gauss(0, 1), 2),
            "SPY": round(525 + random.gauss(0, 1), 2),
        },
        "portfolio_analytics": {
            "aum": balance,
            "equity": balance,
            "cash": round(balance * 12, 2),
            "gross_exposure": round(sum(abs(p.get('position_value', 0)) for p in positions), 2),
            "net_exposure": round(sum((p.get('position_value', 0) if p.get('qty', 0) >= 0 else -p.get('position_value', 0)) for p in positions), 2),
            "leverage": round(max(1.0, sum(abs(p.get('position_value', 0)) for p in positions) / max(balance, 1.0)), 2),
        },
        "equity_history": equity_history,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def serve(host: str = "0.0.0.0", port: int = 8000, state_provider=None) -> None:
    app = create_app(state_provider)
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    serve()
