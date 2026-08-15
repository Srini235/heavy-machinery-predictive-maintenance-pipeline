"""API integration tests — the real FastAPI app with real trained artifacts.

Author: Group 105

Exercises the service end-to-end through `TestClient`: liveness, security
rejection at the boundary, and the vectorized batch endpoint.

Requires `model_registry/` (run `python train_and_save.py` once first).

Run from the repo root:   pytest -m integration -q
"""

import pytest
from fastapi.testclient import TestClient

from src.api.server import app
from tests._metrics import record_metric

pytestmark = pytest.mark.integration

READING = {
    "operating_hours": 1000,
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
def client():
    return TestClient(app)


def test_health_endpoint_is_alive(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert "targets" in body
    assert isinstance(body["machine_types"], list)


def test_predict_endpoint_rejects_bad_api_key(client):
    response = client.post("/predict", json=READING, headers={"x-api-key": "bad-key"})
    assert response.status_code == 400
    assert response.json()["detail"].strip().lower() == "invalid api key"


def test_predict_endpoint_returns_full_response(client):
    response = client.post("/predict", json=READING)
    assert response.status_code == 200
    body = response.json()
    assert body["stability"] in {"stable", "unstable"}
    assert len(body["components"]) == 4
    record_metric("integration.single_predict_latency", body["latency_ms"], "ms")


def test_batch_endpoint_vectorizes_and_scales(client):
    n = 20
    response = client.post("/predict/batch", json={"readings": [READING] * n})
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == n
    assert len(body["results"]) == n
    # identical inputs must produce identical outputs across the batch
    assert all(r == body["results"][0] for r in body["results"])
    record_metric("integration.batch_latency_per_reading", body["latency_per_reading_ms"], "ms")


def test_batch_endpoint_rejects_corrupt_reading(client):
    bad = {**READING, "pressure_mean_bar": 99999}
    response = client.post("/predict/batch", json={"readings": [READING, bad]})
    assert response.status_code == 400
    assert "reading[1]" in response.json()["detail"]


def test_batch_endpoint_enforces_size_limit(client):
    response = client.post("/predict/batch", json={"readings": []})
    assert response.status_code == 422  # pydantic min_length=1
