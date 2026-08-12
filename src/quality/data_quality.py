"""Data quality metrics for the hydraulic fleet telemetry pipeline.

Author: Group 105

Implements the three data-quality controls required before training or
serving the predictive-maintenance models:

1. Schema validation   — column presence, dtype family, and physical value ranges.
2. Missing-value check — per-column missing rate against a configurable budget.
3. Drift detection     — Population Stability Index (PSI) between a reference
                         (training) sample and a new (production) sample.

Design notes:
- Pure functions + one report dataclass: easy to unit-test and reuse from the
  training script, the API, or a scheduled monitoring job.
- Errors raise DataQualityError (fail-fast for pipelines); callers that only
  want measurement can inspect the returned DataQualityReport instead.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [DATA-QUALITY] %(levelname)s %(message)s",
)
logger = logging.getLogger("hydraulics.data_quality")


class DataQualityError(Exception):
    """Raised when a dataset fails a hard data-quality gate."""


# Expected schema for the fleet telemetry dataset: dtype family + physical range.
# Ranges are deliberately generous (sensor spec limits, not statistical limits).
TELEMETRY_SCHEMA: dict[str, dict[str, Any]] = {
    "machine_id": {"kind": "numeric", "min": 1, "max": 10_000},
    "machine_type": {"kind": "string"},
    "cycle_id": {"kind": "numeric", "min": 0, "max": 1_000_000},
    "operating_hours": {"kind": "numeric", "min": 0, "max": 200_000},
    "pressure_mean_bar": {"kind": "numeric", "min": 0, "max": 400},
    "pressure_std_bar": {"kind": "numeric", "min": 0, "max": 100},
    "flow_mean_lpm": {"kind": "numeric", "min": 0, "max": 100},
    "oil_temp_mean_c": {"kind": "numeric", "min": -20, "max": 130},
    "vibration_rms_mms": {"kind": "numeric", "min": 0, "max": 50},
    "motor_power_kw": {"kind": "numeric", "min": 0, "max": 200},
    "pump_speed_mean_rpm": {"kind": "numeric", "min": 0, "max": 5_000},
    "cooling_efficiency_pct": {"kind": "numeric", "min": 0, "max": 100},
    "cooler_condition": {"kind": "numeric", "allowed": [3, 20, 100]},
    "valve_condition": {"kind": "numeric", "allowed": [73, 80, 90, 100]},
    "pump_leakage": {"kind": "numeric", "allowed": [0, 1, 2]},
    "accumulator_pressure": {"kind": "numeric", "allowed": [90, 100, 115, 130]},
    "stability_flag": {"kind": "numeric", "allowed": [0, 1]},
}


@dataclass
class DataQualityReport:
    """Outcome of the data-quality checks, suitable for logging or dashboards."""

    schema_ok: bool = True
    schema_errors: list[str] = field(default_factory=list)
    missing_rates: dict[str, float] = field(default_factory=dict)
    worst_missing_rate: float = 0.0
    psi: dict[str, float] = field(default_factory=dict)
    drifted_features: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_ok": self.schema_ok,
            "schema_errors": self.schema_errors,
            "missing_rates": self.missing_rates,
            "worst_missing_rate": self.worst_missing_rate,
            "psi": self.psi,
            "drifted_features": self.drifted_features,
        }


def validate_schema(
    df: pd.DataFrame,
    schema: dict[str, dict[str, Any]] | None = None,
    strict: bool = True,
) -> list[str]:
    """Metric 1 — schema validation.

    Checks column presence, dtype family and physical value ranges.
    Returns the list of violations; raises DataQualityError when strict.
    """
    schema = schema or TELEMETRY_SCHEMA
    errors: list[str] = []

    for column, spec in schema.items():
        if column not in df.columns:
            errors.append(f"missing column: {column}")
            continue
        series = df[column]
        if spec["kind"] == "numeric":
            if not pd.api.types.is_numeric_dtype(series):
                errors.append(f"{column}: expected numeric dtype, got {series.dtype}")
                continue
            values = series.dropna()
            if "allowed" in spec:
                bad = set(values.unique()) - set(spec["allowed"])
                if bad:
                    errors.append(f"{column}: unexpected values {sorted(bad)[:5]}")
            else:
                if "min" in spec and (values < spec["min"]).any():
                    errors.append(f"{column}: values below {spec['min']}")
                if "max" in spec and (values > spec["max"]).any():
                    errors.append(f"{column}: values above {spec['max']}")
        elif spec["kind"] == "string":
            if not (
                pd.api.types.is_object_dtype(series)
                or isinstance(series.dtype, pd.CategoricalDtype)
            ):
                errors.append(f"{column}: expected string dtype, got {series.dtype}")

    if errors:
        logger.error("Schema validation failed with %d violation(s): %s", len(errors), errors[:3])
        if strict:
            raise DataQualityError(f"schema validation failed: {errors}")
    else:
        logger.info("Schema validation passed for %d columns, %d rows", len(schema), len(df))
    return errors


def check_missing_values(
    df: pd.DataFrame,
    max_missing_rate: float = 0.05,
    columns: list[str] | None = None,
) -> dict[str, float]:
    """Metric 2 — per-column missing-value rate against a budget.

    Returns {column: missing_rate}. Logs a WARNING for any column above the
    budget and raises DataQualityError when a column is entirely empty.
    """
    columns = columns or list(df.columns)
    rates = {c: float(df[c].isna().mean()) for c in columns if c in df.columns}

    for column, rate in rates.items():
        if rate >= 1.0:
            logger.error("Column %s is 100%% missing", column)
            raise DataQualityError(f"column {column} is entirely missing")
        if rate > max_missing_rate:
            logger.warning(
                "Column %s missing rate %.2f%% exceeds budget %.2f%%",
                column,
                rate * 100,
                max_missing_rate * 100,
            )
    logger.info(
        "Missing-value check done: worst rate %.3f%% over %d columns",
        max(rates.values(), default=0.0) * 100,
        len(rates),
    )
    return rates


def compute_psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between two 1-D numeric samples.

    Rule of thumb: PSI < 0.10 stable, 0.10–0.25 moderate shift, > 0.25 drifted.
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    reference = reference[~np.isnan(reference)]
    current = current[~np.isnan(current)]
    if reference.size == 0 or current.size == 0:
        raise DataQualityError("PSI requires non-empty samples")

    # Bin edges from the reference distribution (quantiles -> robust to outliers)
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if edges.size < 3:  # near-constant feature; nothing meaningful to compare
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    ref_pct = np.histogram(reference, bins=edges)[0] / reference.size
    cur_pct = np.histogram(current, bins=edges)[0] / current.size
    # avoid log(0) — standard epsilon substitution
    ref_pct = np.where(ref_pct == 0, 1e-4, ref_pct)
    cur_pct = np.where(cur_pct == 0, 1e-4, cur_pct)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def detect_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    features: list[str],
    threshold: float = 0.25,
) -> dict[str, float]:
    """Metric 3 — drift detection across numeric features via PSI.

    Returns {feature: psi}. Logs a WARNING for each drifted feature.
    """
    psi_values: dict[str, float] = {}
    for feature in features:
        if feature not in reference_df.columns or feature not in current_df.columns:
            logger.warning("Drift check skipped for missing feature %s", feature)
            continue
        psi_values[feature] = compute_psi(
            reference_df[feature].to_numpy(), current_df[feature].to_numpy()
        )
        if psi_values[feature] > threshold:
            logger.warning(
                "Drift detected on %s: PSI=%.3f (threshold %.2f)",
                feature,
                psi_values[feature],
                threshold,
            )
    stable = {k: v for k, v in psi_values.items() if v <= threshold}
    logger.info("Drift check done: %d/%d features stable", len(stable), len(psi_values))
    return psi_values


def run_all_checks(
    df: pd.DataFrame,
    reference_df: pd.DataFrame | None = None,
    numeric_features: list[str] | None = None,
    max_missing_rate: float = 0.05,
    psi_threshold: float = 0.25,
) -> DataQualityReport:
    """Run schema + missing-value (+ optional drift) checks and return a report."""
    report = DataQualityReport()
    report.schema_errors = validate_schema(df, strict=False)
    report.schema_ok = not report.schema_errors
    report.missing_rates = check_missing_values(df, max_missing_rate=max_missing_rate)
    report.worst_missing_rate = max(report.missing_rates.values(), default=0.0)
    if reference_df is not None and numeric_features:
        report.psi = detect_drift(reference_df, df, numeric_features, threshold=psi_threshold)
        report.drifted_features = [f for f, v in report.psi.items() if v > psi_threshold]
    return report
