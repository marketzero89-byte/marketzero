"""
MetricsLogger
JSON-lines structured logging with optional TensorBoard and Prometheus integration.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class MetricsLogger:
    """
    Structured metrics logger.

    Outputs:
    - JSON-lines to disk (always)
    - TensorBoard (optional)
    - Prometheus (optional via push gateway)

    Parameters
    ----------
    log_dir : str | Path
    tensorboard : bool
    prometheus_gateway : str | None  e.g. 'localhost:9091'
    """

    def __init__(
        self,
        log_dir: str | Path = "logs",
        tensorboard: bool = False,
        prometheus_gateway: Optional[str] = None,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._jsonl_path = self.log_dir / "metrics.jsonl"
        self._tb = None
        self._prom = None
        self._step = 0

        if tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                tb_dir = self.log_dir / "tensorboard"
                self._tb = SummaryWriter(str(tb_dir))
                logger.info("TensorBoard writer at %s", tb_dir)
            except ImportError:
                logger.warning("TensorBoard unavailable (install torch)")

        if prometheus_gateway:
            try:
                from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
                self._prom_registry = CollectorRegistry()
                self._prom_gateway  = prometheus_gateway
                self._prom_push     = push_to_gateway
                self._prom_gauge: Dict[str, Any] = {}
                logger.info("Prometheus push gateway: %s", prometheus_gateway)
            except ImportError:
                logger.warning("prometheus_client unavailable")

    # ------------------------------------------------------------------
    # Core log method
    # ------------------------------------------------------------------

    def log(self, tag: str, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """
        Log a dict of metrics.

        Parameters
        ----------
        tag : str   category (e.g. 'generation', 'trade', 'risk')
        metrics : dict
        step : int | None   if None, auto-increments
        """
        self._step = step if step is not None else self._step + 1
        record = {
            "tag":       tag,
            "step":      self._step,
            "timestamp": time.time(),
            **metrics,
        }
        self._write_jsonl(record)

        if self._tb:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self._tb.add_scalar(f"{tag}/{k}", v, self._step)

    def log_generation(self, generation: int, metrics: Dict[str, Any]) -> None:
        self.log("generation", metrics, step=generation)

    def log_trade(self, trade: Dict[str, Any]) -> None:
        self.log("trade", trade)

    def log_risk_event(self, event: Dict[str, Any]) -> None:
        self.log("risk", event)

    def log_agent(self, agent_id: str, metrics: Dict[str, Any], step: int) -> None:
        self.log(f"agent/{agent_id}", metrics, step=step)

    # ------------------------------------------------------------------
    # Prometheus metrics endpoint
    # ------------------------------------------------------------------

    def push_prometheus(self, metrics: Dict[str, float], job: str = "pbt_trading") -> None:
        if not hasattr(self, "_prom_gateway"):
            return
        try:
            from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
            registry = CollectorRegistry()
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    safe_k = k.replace("/", "_").replace("-", "_").replace(".", "_")
                    g = Gauge(safe_k, safe_k, registry=registry)
                    g.set(v)
            push_to_gateway(self._prom_gateway, job=job, registry=registry)
        except Exception as exc:
            logger.error("Prometheus push failed: %s", exc)

    # ------------------------------------------------------------------
    # File I/O
    # ------------------------------------------------------------------

    def _write_jsonl(self, record: Dict) -> None:
        try:
            with self._jsonl_path.open("a") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception as exc:
            logger.error("MetricsLogger write error: %s", exc)

    def tail(self, n: int = 50) -> list:
        """Return last n log lines."""
        if not self._jsonl_path.exists():
            return []
        lines = self._jsonl_path.read_text().strip().split("\n")
        return [json.loads(l) for l in lines[-n:] if l]

    def flush(self) -> None:
        if self._tb:
            self._tb.flush()

    def close(self) -> None:
        if self._tb:
            self._tb.close()
        logger.info("MetricsLogger closed")
