"""Shared fixtures, validation-dataset sampling criteria and metric logging.

Author: Group 105

Validation dataset — basis and criteria (QA feedback)
-----------------------------------------------------
All test fixtures draw from the shipped fleet telemetry export
(`data/hydraulic_fleet_telemetry.csv`, 10,000 cycles across 5 machine types),
under the following explicit criteria:

* **Reproducibility** — every sample uses a fixed `random_state`, so each test
  sees byte-identical data on every run and on every machine (local and CI).
* **Representativeness** — samples of 2,000–3,000 rows are large enough to
  contain every class of every target (verified by `test_condition_model_
  output_shape_and_classes`) while keeping the whole suite under ~1 minute.
* **Stratification** — wherever a train/validation split matters, we stratify
  on `stability_flag` (the rarest binary label) so both splits keep the true
  class balance; the trainer itself stratifies the same way.
* **Independence** — drift tests compare *disjoint* halves of the fleet
  history (first 5,000 vs last 5,000 cycles), never overlapping samples.

Metric logging (QA feedback)
----------------------------
Tests call ``record_metric(name, value, unit)``. Each metric is logged at INFO
level immediately, and a consolidated table is printed at the end of the
pytest run and written to ``tests/metrics_report.json`` for evidence.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from tests._metrics import _RECORDED, record_metric  # noqa: F401  (re-export)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "hydraulic_fleet_telemetry.csv"
METRICS_OUT = Path(__file__).resolve().parent / "metrics_report.json"


@pytest.fixture(scope="session")
def telemetry() -> pd.DataFrame:
    """The full 10,000-cycle fleet telemetry dataset (read once per run)."""
    return pd.read_csv(DATA)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if not _RECORDED:
        return
    terminalreporter.write_sep("-", "measured metrics")
    for name, entry in sorted(_RECORDED.items()):
        terminalreporter.write_line(f"{name:<48} {entry['value']} {entry['unit']}")
    METRICS_OUT.write_text(json.dumps(_RECORDED, indent=2), encoding="utf-8")
    terminalreporter.write_line(f"metrics written to {METRICS_OUT}")
