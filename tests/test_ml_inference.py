"""ML inference tests — Assignment II, Objective 2, requirement 7b.

Author: Group 105

Verifies the *inference behaviour* of the trained pipelines:

  * output shape and class-range checks for both models;
  * probability outputs lie in [0, 1] and rows sum to 1;
  * invariance — identical inputs give identical predictions, and an unseen
    machine_type is handled gracefully (OneHotEncoder handle_unknown="ignore");
  * directional expectation — degrading the sensor readings (hotter oil, more
    vibration, worse cooling) must not lower the predicted instability risk.

Run from the repo root:   pytest tests/test_ml_inference.py -q
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "hydraulic_fleet_telemetry.csv"

NUMERIC = [
    "operating_hours",
    "pressure_mean_bar",
    "pressure_std_bar",
    "flow_mean_lpm",
    "oil_temp_mean_c",
    "vibration_rms_mms",
    "motor_power_kw",
    "pump_speed_mean_rpm",
    "cooling_efficiency_pct",
]
CATEGORICAL = ["machine_type"]
TARGETS = ["cooler_condition", "valve_condition", "pump_leakage", "accumulator_pressure"]
ALLOWED_CLASSES = {
    "cooler_condition": {3, 20, 100},
    "valve_condition": {73, 80, 90, 100},
    "pump_leakage": {0, 1, 2},
    "accumulator_pressure": {90, 100, 115, 130},
}


def _make_preprocessor():
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )


@pytest.fixture(scope="module")
def data():
    df = pd.read_csv(DATA)
    return df.sample(n=2500, random_state=7)


@pytest.fixture(scope="module")
def condition_model(data):
    model = Pipeline(
        [
            ("prep", _make_preprocessor()),
            (
                "clf",
                MultiOutputClassifier(
                    RandomForestClassifier(n_estimators=60, max_depth=10, random_state=42)
                ),
            ),
        ]
    )
    model.fit(data[NUMERIC + CATEGORICAL], data[TARGETS])
    return model


@pytest.fixture(scope="module")
def stability_model(data):
    model = Pipeline(
        [
            ("prep", _make_preprocessor()),
            ("clf", RandomForestClassifier(n_estimators=60, max_depth=6, random_state=42)),
        ]
    )
    model.fit(data[NUMERIC + CATEGORICAL], data["stability_flag"])
    return model


@pytest.fixture(scope="module")
def healthy_reading():
    return pd.DataFrame(
        [
            {
                "operating_hours": 100.0,
                "pressure_mean_bar": 215.0,
                "pressure_std_bar": 3.0,
                "flow_mean_lpm": 8.4,
                "oil_temp_mean_c": 45.0,
                "vibration_rms_mms": 1.0,
                "motor_power_kw": 15.0,
                "pump_speed_mean_rpm": 1445.0,
                "cooling_efficiency_pct": 99.0,
                "machine_type": "Telehandler",
            }
        ]
    )


def test_condition_model_output_shape_and_classes(condition_model, data):
    """Output must be (n_samples, 4) and every class must be a valid label."""
    X = data[NUMERIC + CATEGORICAL].head(50)
    pred = np.asarray(condition_model.predict(X))
    assert pred.shape == (50, len(TARGETS))
    for i, target in enumerate(TARGETS):
        assert set(np.unique(pred[:, i])).issubset(
            ALLOWED_CLASSES[target]
        ), f"{target} produced classes outside the label set"


def test_stability_probabilities_are_valid(stability_model, data):
    """predict_proba must produce probabilities in [0, 1] that sum to 1."""
    X = data[NUMERIC + CATEGORICAL].head(50)
    proba = stability_model.predict_proba(X)
    assert proba.shape == (50, 2)
    assert np.all(proba >= 0.0) and np.all(proba <= 1.0)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)


def test_inference_is_deterministic(stability_model, healthy_reading):
    """Invariance: the same input must always yield the same prediction."""
    first = stability_model.predict_proba(healthy_reading)
    second = stability_model.predict_proba(pd.concat([healthy_reading] * 3, ignore_index=True))
    np.testing.assert_allclose(second, np.repeat(first, 3, axis=0))


def test_unseen_machine_type_is_handled(stability_model, healthy_reading):
    """Invariance: an unknown category must not crash inference (encoder ignores it)."""
    exotic = healthy_reading.copy()
    exotic["machine_type"] = "MoonCrawler-9000"
    proba = stability_model.predict_proba(exotic)
    assert proba.shape == (1, 2)
    assert np.all(proba >= 0.0) and np.all(proba <= 1.0)


def test_degraded_sensors_do_not_lower_risk(stability_model, healthy_reading):
    """Directional test: severe degradation must not reduce P(unstable)."""
    degraded = healthy_reading.copy()
    degraded["oil_temp_mean_c"] = 95.0
    degraded["vibration_rms_mms"] = 12.0
    degraded["cooling_efficiency_pct"] = 25.0
    degraded["pressure_std_bar"] = 25.0

    unstable_idx = list(stability_model.classes_).index(1)
    p_healthy = stability_model.predict_proba(healthy_reading)[0, unstable_idx]
    p_degraded = stability_model.predict_proba(degraded)[0, unstable_idx]
    assert (
        p_degraded >= p_healthy
    ), f"risk fell after degradation: healthy={p_healthy:.3f} degraded={p_degraded:.3f}"
