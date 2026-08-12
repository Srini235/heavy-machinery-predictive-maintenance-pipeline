"""ML training tests — Assignment II, Objective 2, requirement 7a.

Author: Group 105

Verifies the *training procedure* itself (not just the final metrics):

  * a model with enough capacity can overfit a tiny batch (sanity check that
    the features carry signal and the training loop works end-to-end);
  * training loss decreases as boosting iterations are added;
  * the production preprocessing + model pipeline trained on a small sample of
    the real telemetry beats a majority-class baseline.

Run from the repo root:   pytest tests/test_ml_training.py -q
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
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


@pytest.fixture(scope="module")
def telemetry_sample():
    df = pd.read_csv(DATA)
    return df.sample(n=2000, random_state=42)


def test_model_can_overfit_small_batch(telemetry_sample):
    """A high-capacity model must reach ~100% accuracy on a tiny batch.

    If it cannot, the features/labels are broken or training is mis-wired.
    """
    batch = telemetry_sample.sample(n=64, random_state=0)
    X = batch[NUMERIC]
    y = batch["stability_flag"]
    model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=0)
    model.fit(X, y)
    train_acc = accuracy_score(y, model.predict(X))
    assert train_acc >= 0.99, f"could not overfit a 64-row batch (acc={train_acc:.3f})"


def test_training_loss_decreases_with_iterations(telemetry_sample):
    """Log-loss on the training set must fall as boosting stages are added."""
    X = telemetry_sample[NUMERIC]
    y = telemetry_sample["stability_flag"]
    model = GradientBoostingClassifier(n_estimators=50, random_state=0)
    model.fit(X, y)

    losses = [log_loss(y, proba) for proba in model.staged_predict_proba(X)]
    assert losses[-1] < losses[0], "training loss did not decrease"
    # loss should be monotonically non-increasing for a gradient booster on train data
    assert all(b <= a + 1e-9 for a, b in zip(losses, losses[1:])), "loss increased mid-training"


def test_production_pipeline_beats_majority_baseline(telemetry_sample):
    """The production preprocessing+model pipeline must beat predicting the
    majority class, proving the pipeline learns real signal end-to-end."""
    X = telemetry_sample[NUMERIC + CATEGORICAL]
    y = telemetry_sample["stability_flag"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    pipeline = Pipeline(
        [
            (
                "prep",
                ColumnTransformer(
                    [
                        ("num", StandardScaler(), NUMERIC),
                        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
                    ]
                ),
            ),
            ("clf", RandomForestClassifier(n_estimators=60, max_depth=6, random_state=42)),
        ]
    )
    pipeline.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, pipeline.predict(X_te))
    majority = float(np.mean(y_te == y_te.mode()[0]))
    assert acc > majority, f"pipeline acc {acc:.3f} did not beat majority baseline {majority:.3f}"
