"""
Production readiness validation and paper-trading session tracking.
"""

from __future__ import annotations

import glob
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class CheckResult:
    name: str
    passed: bool
    message: str
    required: bool = True


@dataclass
class ValidationReport:
    checks: List[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks if c.required)

    @property
    def ready_for_live(self) -> bool:
        return self.passed

    def summary_lines(self) -> List[str]:
        lines = ["", "=" * 60, "  Production Readiness Validation", "=" * 60]
        for c in self.checks:
            icon = "PASS" if c.passed else "FAIL"
            req = "" if c.required else " (optional)"
            lines.append(f"  [{icon}] {c.name}{req}")
            lines.append(f"         {c.message}")
        lines.append("")
        status = "READY FOR LIVE TRADING" if self.ready_for_live else "NOT READY FOR LIVE TRADING"
        lines.append(f"  Result: {status}")
        lines.append("=" * 60)
        lines.append("")
        return lines


class ProductionValidator:
    """
    Tracks paper-trading sessions and validates go-live criteria.

    Criteria (from docs/18_Live_Trading.md):
      - ≥ 5 calendar days of paper trading logged
      - best fitness ≥ 0.3
      - backtest on real OHLCV data completed
      - Alpaca credentials configured
      - alerts configured (email or Slack)
    """

    MIN_PAPER_DAYS = 5
    MIN_FITNESS = 0.3
    MIN_BACKTEST_SHARPE = 0.0

    def __init__(
        self,
        log_dir: str | Path = "logs",
        checkpoint_dir: str | Path = "checkpoints",
        reports_dir: str | Path = "reports",
        data_dir: str | Path = "data",
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.validation_log = self.log_dir / "validation.jsonl"
        self.checkpoint_dir = Path(checkpoint_dir)
        self.reports_dir = Path(reports_dir)
        self.data_dir = Path(data_dir)

    # ------------------------------------------------------------------
    # Session recording (called each PBT generation in paper mode)
    # ------------------------------------------------------------------

    def record_session(
        self,
        *,
        broker: str,
        live: bool,
        generation: int,
        best_fitness: float,
        mean_fitness: float,
        equity: float,
        drawdown: float,
        n_trades: int = 0,
    ) -> None:
        if live:
            return
        today = time.strftime("%Y-%m-%d")
        rec = {
            "timestamp": time.time(),
            "date": today,
            "broker": broker,
            "live": False,
            "generation": generation,
            "best_fitness": round(best_fitness, 4),
            "mean_fitness": round(mean_fitness, 4),
            "equity": round(equity, 2),
            "drawdown": round(drawdown, 4),
            "n_trades": n_trades,
        }
        with self.validation_log.open("a") as f:
            f.write(json.dumps(rec) + "\n")

    def _load_sessions(self) -> List[dict]:
        if not self.validation_log.exists():
            return []
        sessions = []
        for line in self.validation_log.read_text().strip().split("\n"):
            if not line.strip():
                continue
            try:
                sessions.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return sessions

    def paper_trading_days(self) -> int:
        dates = {s["date"] for s in self._load_sessions() if not s.get("live", False)}
        return len(dates)

    def peak_fitness(self) -> float:
        sessions = self._load_sessions()
        checkpoint_fitness = self._latest_checkpoint_fitness()
        session_peak = max((s.get("best_fitness", -1.0) for s in sessions), default=-1.0)
        return max(session_peak, checkpoint_fitness)

    def _latest_checkpoint_fitness(self) -> float:
        ckpts = sorted(glob.glob(str(self.checkpoint_dir / "gen_*.json")))
        if not ckpts:
            return -1.0
        data = json.loads(Path(ckpts[-1]).read_text())
        pop = data.get("population", [])
        if not pop:
            return -1.0
        return max(a.get("fitness", -1.0) for a in pop)

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def check_paper_days(self) -> CheckResult:
        days = self.paper_trading_days()
        ok = days >= self.MIN_PAPER_DAYS
        return CheckResult(
            name="Paper trading duration",
            passed=ok,
            message=f"{days}/{self.MIN_PAPER_DAYS} calendar days logged "
                    f"(logs/validation.jsonl)",
        )

    def check_fitness(self) -> CheckResult:
        fitness = self.peak_fitness()
        ok = fitness >= self.MIN_FITNESS
        return CheckResult(
            name="Agent fitness",
            passed=ok,
            message=f"Peak fitness {fitness:.4f} (required >= {self.MIN_FITNESS})",
        )

    def check_backtest_on_real_data(self) -> CheckResult:
        reports = list(self.reports_dir.glob("*.json"))
        real_report = None
        for path in sorted(reports, reverse=True):
            try:
                data = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            if data.get("data_source") == "ohlcv":
                real_report = data
                break

        if real_report is None:
            sample = self.data_dir / "AAPL_sample.csv"
            hint = f"Run: python main.py backtest --data {sample}"
            return CheckResult(
                name="Real-data backtest",
                passed=False,
                message=f"No OHLCV backtest report found. {hint}",
            )

        sharpe = real_report.get("walk_forward", {}).get("oos_sharpe", -999)
        ok = sharpe >= self.MIN_BACKTEST_SHARPE
        return CheckResult(
            name="Real-data backtest",
            passed=ok,
            message=f"OOS Sharpe {sharpe:.4f} from {real_report.get('data_file', '?')} "
                    f"(required >= {self.MIN_BACKTEST_SHARPE})",
        )

    def check_alpaca_credentials(self) -> CheckResult:
        key = os.getenv("APCA_API_KEY_ID", "")
        secret = os.getenv("APCA_API_SECRET_KEY", "")
        ok = bool(key and secret and "your_alpaca" not in key)
        return CheckResult(
            name="Alpaca credentials",
            passed=ok,
            message="APCA_API_KEY_ID and APCA_API_SECRET_KEY configured in .env"
                    if ok else "Set Alpaca API keys in .env before live trading",
        )

    def check_alerts(self) -> CheckResult:
        slack = os.getenv("ALERT_SLACK_WEBHOOK", "")
        email = os.getenv("ALERT_EMAIL_TO", "") and os.getenv("ALERT_SMTP_PASS", "")
        ok = bool(slack or email)
        return CheckResult(
            name="Alert notifications",
            passed=ok,
            required=False,
            message="Email or Slack alerts configured"
                    if ok else "No alerts configured (recommended: ALERT_SLACK_WEBHOOK)",
        )

    def check_dashboard_auth(self) -> CheckResult:
        key = os.getenv("PBT_API_KEY", "")
        ok = bool(key)
        return CheckResult(
            name="Dashboard API key",
            passed=ok,
            required=False,
            message="PBT_API_KEY set" if ok else "Set PBT_API_KEY before exposing dashboard publicly",
        )

    def check_checkpoints(self) -> CheckResult:
        ckpts = glob.glob(str(self.checkpoint_dir / "gen_*.json"))
        ok = len(ckpts) > 0
        return CheckResult(
            name="Checkpoints",
            passed=ok,
            message=f"{len(ckpts)} checkpoint(s) in {self.checkpoint_dir}/"
                    if ok else "No checkpoints — run paper trading first",
        )

    # ------------------------------------------------------------------
    # Full report
    # ------------------------------------------------------------------

    def run_all(self) -> ValidationReport:
        report = ValidationReport(checks=[
            self.check_checkpoints(),
            self.check_paper_days(),
            self.check_fitness(),
            self.check_backtest_on_real_data(),
            self.check_alpaca_credentials(),
            self.check_alerts(),
            self.check_dashboard_auth(),
        ])
        return report

    def save_report(self, report: ValidationReport, path: str | Path | None = None) -> Path:
        out = Path(path or self.reports_dir / "validation.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp": time.time(),
            "ready_for_live": report.ready_for_live,
            "paper_days": self.paper_trading_days(),
            "peak_fitness": self.peak_fitness(),
            "checks": [
                {"name": c.name, "passed": c.passed, "message": c.message, "required": c.required}
                for c in report.checks
            ],
        }
        out.write_text(json.dumps(payload, indent=2))
        return out
