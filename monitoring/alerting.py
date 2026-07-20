"""
Alert Manager (v1.1 stub)

RESERVED FOR FUTURE USE — NOT IMPORTABLE IN v1.0
This module is intentionally blocked from import in v1.0 to prevent
accidental dependency on unreleased features. Full WebSocket alerting,
distributed halt signals, and cloud notification routing are scheduled
for v1.1 release.

In v1.0: Use risk_manager.status() and mlops.alerts for monitoring.
"""

__all__ = []

# Intentionally empty in v1.0 — raise on any import attempt
raise ImportError(
    "alerting module is v1.1+ only. "
    "In v1.0, use risk_manager.status() and mlops.alerts instead."
)
