"""Shared metric registry for the test suite (single module instance).

Lives outside conftest.py on purpose: pytest imports conftest under its own
top-level module name, so keeping the registry here guarantees that test
modules and the terminal-summary hook see the same `_RECORDED` dict.
"""

from __future__ import annotations

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [TESTS] %(levelname)s %(message)s")
metrics_logger = logging.getLogger("tests.metrics")

_RECORDED: dict[str, dict] = {}


def record_metric(name: str, value: float, unit: str = "") -> None:
    """Log a measured metric clearly and collect it for the end-of-run report."""
    _RECORDED[name] = {"value": round(float(value), 6), "unit": unit}
    metrics_logger.info("METRIC %-45s = %s %s", name, round(float(value), 6), unit)
