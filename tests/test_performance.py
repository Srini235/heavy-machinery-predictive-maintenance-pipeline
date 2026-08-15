"""Time & space complexity tests (colleague feedback, QA).

Author: Group 105

Measures and gates the runtime characteristics of the *production* artifacts
in `model_registry/` (train once with `python train_and_save.py` first):

  * single-reading inference latency vs the 100 ms quality requirement;
  * batch (vectorized) per-reading latency, demonstrating the scalability
    win over per-request calls;
  * peak memory of a prediction (tracemalloc);
  * on-disk artifact size.

All measurements are logged via `record_metric` and written to
tests/metrics_report.json at the end of the run.

Run from the repo root:   pytest -m performance -q
"""

import time
import tracemalloc
from pathlib import Path

import joblib
import pandas as pd
import pytest

from tests._metrics import record_metric

pytestmark = pytest.mark.performance

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "model_registry"

READING = {
    "operating_hours": 1000.0,
    "pressure_mean_bar": 150.0,
    "pressure_std_bar": 5.0,
    "flow_mean_lpm": 8.0,
    "oil_temp_mean_c": 75.0,
    "vibration_rms_mms": 5.0,
    "motor_power_kw": 20.0,
    "pump_speed_mean_rpm": 1200.0,
    "cooling_efficiency_pct": 60.0,
    "machine_type": "Excavator",
}


@pytest.fixture(scope="module")
def stability_model():
    return joblib.load(REGISTRY / "stability_model.joblib")


@pytest.fixture(scope="module")
def single_row():
    return pd.DataFrame([READING])


def _median_latency_ms(fn, repeats: int = 30) -> float:
    fn()  # warm-up: exclude one-time lazy initialisation from the measurement
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000)
    samples.sort()
    return samples[len(samples) // 2]


def test_single_inference_latency_under_100ms(stability_model, single_row):
    """Quality Requirement #2 — real-time stability inference under 100 ms."""
    latency = _median_latency_ms(lambda: stability_model.predict(single_row))
    record_metric("performance.single_inference_median_latency", latency, "ms")
    assert latency < 100, f"median latency {latency:.1f}ms breaches the 100ms budget"


def test_batch_inference_amortizes_per_reading_cost(stability_model, single_row):
    """Time complexity: vectorized batch must be far cheaper per reading."""
    batch = pd.concat([single_row] * 200, ignore_index=True)
    single = _median_latency_ms(lambda: stability_model.predict(single_row), repeats=10)
    batch_total = _median_latency_ms(lambda: stability_model.predict(batch), repeats=10)
    per_reading = batch_total / len(batch)
    record_metric("performance.batch_per_reading_latency", per_reading, "ms")
    # a 200-row batch must NOT cost 200x a single call (vectorization pays off)
    assert batch_total < single * 20, "batch inference is not amortizing per-reading cost"


def test_prediction_memory_footprint(stability_model, single_row):
    """Space complexity: a prediction must not allocate excessive memory."""
    tracemalloc.start()
    stability_model.predict(single_row)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak / 1e6
    record_metric("performance.prediction_peak_memory", peak_mb, "MB")
    assert peak_mb < 50, f"single prediction allocated {peak_mb:.1f} MB"


def test_artifact_sizes_are_deployable():
    """Space complexity: artifacts must stay small enough for easy deployment."""
    sizes = {p.name: p.stat().st_size / 1e6 for p in REGISTRY.glob("*.joblib")}
    for name, mb in sizes.items():
        record_metric(f"performance.artifact_size.{name}", mb, "MB")
    assert sum(sizes.values()) < 200, f"artifacts too large to ship: {sizes}"
