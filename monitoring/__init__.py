"""
monitoring package — v1.0

v1.1 alert sinks (Email / Slack / Prometheus) are NOT available in v1.0.
Any attempt to import `monitoring.alerting` will raise ImportError to enforce
the scope boundary defined in R-095.
"""

__all__: list = []


def __getattr__(name: str):
    if name == "alerting":
        raise ImportError(
            "monitoring.alerting is a v1.1 feature and must not be used in v1.0. "
            "Remove this import or gate it behind a version flag."
        )
    raise AttributeError(f"module 'monitoring' has no attribute {name!r}")
