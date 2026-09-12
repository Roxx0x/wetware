"""Demo tasks. Each exposes demo(W, **kw) -> dict of metrics."""

from . import digits, timeseries

TASKS = {"timeseries": timeseries.demo, "digits": digits.demo}

__all__ = ["TASKS", "timeseries", "digits"]
