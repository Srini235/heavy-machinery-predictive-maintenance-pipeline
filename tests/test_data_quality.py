"""Data validation tests — Assignment II, Objective 2, requirements 6 and 8b.

Author: Group 105

Exercises the data-quality module (src/quality) against the real fleet
telemetry dataset and against deliberately corrupted copies:

  * schema validation passes on the shipped dataset and catches corruption;
  * missing-value metric flags columns above the budget and rejects empty ones;
  * PSI drift metric is ~0 for identically distributed splits and fires on a
    genuinely shifted feature.

Run from the repo root:   pytest tests/test_data_quality.py -q
"""

from pathlib import Path

import numpy as np
import pytest

from src.quality import (
    DataQualityError,
    check_missing_values,
    compute_psi,
    detect_drift,
    validate_schema,
)
from src.quality.data_quality import run_all_checks
from tests._metrics import record_metric

pytestmark = pytest.mark.data_quality

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "hydraulic_fleet_telemetry.csv"

NUMERIC = ["pressure_mean_bar", "flow_mean_lpm", "oil_temp_mean_c", "vibration_rms_mms"]


# --------------------------------------------------------------------------- #
# Metric 1 — schema validation
# --------------------------------------------------------------------------- #
def test_shipped_dataset_passes_schema(telemetry):
    assert validate_schema(telemetry) == []


def test_schema_catches_missing_column(telemetry):
    broken = telemetry.drop(columns=["oil_temp_mean_c"])
    with pytest.raises(DataQualityError, match="oil_temp_mean_c"):
        validate_schema(broken)


def test_schema_catches_out_of_range_values(telemetry):
    corrupted = telemetry.copy()
    corrupted.loc[corrupted.index[:5], "oil_temp_mean_c"] = 999.0  # physically impossible
    with pytest.raises(DataQualityError, match="oil_temp_mean_c"):
        validate_schema(corrupted)


def test_schema_catches_invalid_label(telemetry):
    corrupted = telemetry.copy()
    corrupted.loc[corrupted.index[0], "pump_leakage"] = 7  # not in {0, 1, 2}
    with pytest.raises(DataQualityError, match="pump_leakage"):
        validate_schema(corrupted)


# --------------------------------------------------------------------------- #
# Metric 2 — missing values
# --------------------------------------------------------------------------- #
def test_missing_rates_measured_on_real_data(telemetry):
    rates = check_missing_values(telemetry)
    assert set(rates) == set(telemetry.columns)
    assert all(0.0 <= r < 1.0 for r in rates.values())


def test_missing_check_rejects_empty_column(telemetry):
    broken = telemetry.copy()
    broken["flow_mean_lpm"] = np.nan
    with pytest.raises(DataQualityError, match="flow_mean_lpm"):
        check_missing_values(broken)


# --------------------------------------------------------------------------- #
# Metric 3 — drift (PSI)
# --------------------------------------------------------------------------- #
def test_psi_near_zero_for_same_distribution(telemetry):
    half = len(telemetry) // 2
    reference, current = telemetry.iloc[:half], telemetry.iloc[half:]
    psi = compute_psi(
        reference["pressure_mean_bar"].to_numpy(), current["pressure_mean_bar"].to_numpy()
    )
    record_metric("data_quality.psi_same_distribution", psi)
    assert psi < 0.10, f"identical distributions reported PSI={psi:.3f}"


def test_psi_detects_injected_drift(telemetry):
    reference = telemetry.sample(n=3000, random_state=1)
    shifted = reference.copy()
    shifted["oil_temp_mean_c"] = shifted["oil_temp_mean_c"] + 25.0  # cooling failure fleet-wide

    psi = detect_drift(reference, shifted, NUMERIC)
    record_metric("data_quality.psi_injected_drift", psi["oil_temp_mean_c"])
    assert psi["oil_temp_mean_c"] > 0.25, "large temperature shift not flagged as drift"
    assert psi["pressure_mean_bar"] < 0.10, "untouched feature wrongly flagged"


def test_run_all_checks_produces_full_report(telemetry):
    reference = telemetry.sample(n=2000, random_state=2)
    current = telemetry.sample(n=2000, random_state=3)
    report = run_all_checks(current, reference_df=reference, numeric_features=NUMERIC)
    assert report.schema_ok
    assert report.worst_missing_rate < 0.05
    assert set(report.psi) == set(NUMERIC)
    assert report.drifted_features == []
