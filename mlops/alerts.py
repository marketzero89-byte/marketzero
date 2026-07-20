"""
Alert System
Email alerts and Slack webhook for trading halts, circuit breakers, daily summaries.
"""

from __future__ import annotations

import json
import logging
import os
import smtplib
import urllib.request
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class EmailAlerter:
    """
    SMTP email alerter.

    Configure via env vars or explicit parameters:
      ALERT_EMAIL_FROM, ALERT_EMAIL_TO, ALERT_SMTP_HOST,
      ALERT_SMTP_PORT, ALERT_SMTP_USER, ALERT_SMTP_PASS
    """

    def __init__(
        self,
        from_addr:  Optional[str] = None,
        to_addrs:   Optional[List[str]] = None,
        smtp_host:  Optional[str] = None,
        smtp_port:  int = 587,
        smtp_user:  Optional[str] = None,
        smtp_pass:  Optional[str] = None,
    ):
        self.from_addr = from_addr  or os.getenv("ALERT_EMAIL_FROM", "")
        self.to_addrs  = to_addrs   or [a.strip() for a in os.getenv("ALERT_EMAIL_TO", "").split(",") if a.strip()]
        self.smtp_host = smtp_host  or os.getenv("ALERT_SMTP_HOST", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("ALERT_SMTP_PORT", str(smtp_port)))
        self.smtp_user = smtp_user  or os.getenv("ALERT_SMTP_USER", "")
        self.smtp_pass = smtp_pass  or os.getenv("ALERT_SMTP_PASS", "")

    def _configured(self) -> bool:
        return bool(self.from_addr and self.to_addrs and self.smtp_host and self.smtp_pass)

    def send(self, subject: str, body: str, html: bool = False) -> bool:
        if not self._configured():
            logger.warning("EmailAlerter not configured; skipping send.")
            return False
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"]    = self.from_addr
            msg["To"]      = ", ".join(self.to_addrs)
            part = MIMEText(body, "html" if html else "plain")
            msg.attach(part)
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_pass)
                server.sendmail(self.from_addr, self.to_addrs, msg.as_string())
            logger.info("Email sent: %s → %s", subject, self.to_addrs)
            return True
        except Exception as exc:
            logger.error("Email send failed: %s", exc)
            return False

    def send_halt_alert(self, reason: str, equity: float, drawdown: float) -> bool:
        subject = f"🚨 PBT Trading HALT: {reason}"
        body = (
            f"<h2>Trading Halt Triggered</h2>"
            f"<p><b>Reason:</b> {reason}</p>"
            f"<p><b>Current Equity:</b> ${equity:,.2f}</p>"
            f"<p><b>Drawdown:</b> {drawdown*100:.2f}%</p>"
            f"<p>System has been halted. Manual review required.</p>"
        )
        return self.send(subject, body, html=True)

    def send_daily_summary(self, metrics: Dict) -> bool:
        subject = f"📊 PBT Daily Summary — P&L: {metrics.get('daily_pnl_pct', 0):.2f}%"
        rows = "".join(
            f"<tr><td>{k}</td><td>{v}</td></tr>"
            for k, v in metrics.items()
        )
        body = (
            f"<h2>Daily Trading Summary</h2>"
            f"<table border='1' cellpadding='4'>{rows}</table>"
        )
        return self.send(subject, body, html=True)


class SlackAlerter:
    """
    Slack Incoming Webhook alerter.

    Configure via ALERT_SLACK_WEBHOOK env var or explicit webhook_url.
    """

    def __init__(self, webhook_url: Optional[str] = None):
        self.webhook_url = webhook_url or os.getenv("ALERT_SLACK_WEBHOOK", "")

    def _configured(self) -> bool:
        return bool(self.webhook_url)

    def send(self, text: str, blocks: Optional[List[Dict]] = None) -> bool:
        if not self._configured():
            logger.warning("SlackAlerter not configured; skipping send.")
            return False
        try:
            payload = {"text": text}
            if blocks:
                payload["blocks"] = blocks
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self.webhook_url,
                data=data,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                ok = resp.status == 200
            if ok:
                logger.info("Slack alert sent: %s", text[:80])
            return ok
        except Exception as exc:
            logger.error("Slack send failed: %s", exc)
            return False

    def send_generation_summary(self, gen: int, metrics: Dict) -> bool:
        text = (
            f"*Generation {gen} complete* | "
            f"best={metrics.get('max_fitness', 0):.4f} "
            f"mean={metrics.get('mean_fitness', 0):.4f} "
            f"regime={metrics.get('regime', 'UNKNOWN')}"
        )
        blocks = [
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": text},
                "fields": [
                    {"type": "mrkdwn", "text": f"*Sharpe*\n{metrics.get('sharpe', 0):.3f}"},
                    {"type": "mrkdwn", "text": f"*Drawdown*\n{metrics.get('max_drawdown', 0)*100:.2f}%"},
                    {"type": "mrkdwn", "text": f"*Win Rate*\n{metrics.get('win_rate', 0)*100:.1f}%"},
                    {"type": "mrkdwn", "text": f"*Trades*\n{metrics.get('n_trades', 0)}"},
                ],
            }
        ]
        return self.send(text, blocks)

    def send_circuit_breaker(self, event_type: str, message: str) -> bool:
        text = f":rotating_light: *Circuit Breaker* [{event_type}]\n{message}"
        return self.send(text)

    def send_halt(self, reason: str) -> bool:
        text = f":red_circle: *TRADING HALT*\n{reason}"
        return self.send(text)


class AlertManager:
    """Combines email + Slack into a unified alert interface."""

    def __init__(
        self,
        email:   Optional[EmailAlerter] = None,
        slack:   Optional[SlackAlerter] = None,
    ):
        self.email = email or EmailAlerter()
        self.slack = slack or SlackAlerter()

    def halt(self, reason: str, equity: float = 0.0, drawdown: float = 0.0) -> None:
        self.slack.send_halt(reason)
        self.email.send_halt_alert(reason, equity, drawdown)

    def circuit_breaker(self, event_type: str, message: str) -> None:
        self.slack.send_circuit_breaker(event_type, message)
        self.email.send(f"Circuit Breaker: {event_type}", message)

    def generation_summary(self, gen: int, metrics: Dict) -> None:
        self.slack.send_generation_summary(gen, metrics)

    def daily_summary(self, metrics: Dict) -> None:
        self.email.send_daily_summary(metrics)
