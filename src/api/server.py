"""
api_server.py — FastAPI backend for the Predictive-Maintenance web app.

Author: Aman Kushwah (2024AC05064) — Group 105

Serves the multi-output condition model + real-time stability model to the React
frontend, guarded by the application-wide security layer, and enriched with RAG
repair guidance (with optional OpenRouter LLM generation).

Run:  uvicorn api_server:app --reload --port 8000
Docs: http://localhost:8000/docs
"""

import json
import logging
import os
from pathlib import Path
from time import perf_counter

import joblib
import pandas as pd
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.maintenance_advisor import MaintenanceAdvisor
from src.model_registry import ModelRegistry
from src.security_layer import (
    HYDRAULIC_BOUNDS,
    ApiKeyAuthenticator,
    AuditTrail,
    RateLimiter,
    SecureInferenceGateway,
    SecurityError,
    compute_file_sha256,
    validate_sensor_payload,
)

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "model_registry"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [API] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)
COND_PATH = REGISTRY / "condition_model.joblib"
STAB_PATH = REGISTRY / "stability_model.joblib"
SCHEMA_PATH = REGISTRY / "schema.json"
API_KEY = os.getenv("HYDRAULICS_API_KEY", "fleet-ops-secret-key")

registry = ModelRegistry(REGISTRY)
logger.info("Loading model registry from %s", REGISTRY)
if not (COND_PATH.exists() and STAB_PATH.exists() and SCHEMA_PATH.exists()):
    logger.error("Missing model artifacts in %s", REGISTRY)
    raise RuntimeError("Models not found. Run: python train_and_save.py from the repository root")

try:
    _condition_model = joblib.load(COND_PATH)
    _stability_model = joblib.load(STAB_PATH)
except Exception as exc:
    logger.exception("Failed to load model artifacts")
    raise RuntimeError("Unable to load model artifacts") from exc

with open(SCHEMA_PATH, encoding="utf-8") as f:
    SCHEMA = json.load(f)
NUMERIC = SCHEMA["numeric_features"]
FEATURES = NUMERIC + SCHEMA["categorical_features"]
TARGETS = SCHEMA["targets"]
HEALTHY = SCHEMA["healthy_class"]
_model_sha = compute_file_sha256(COND_PATH)

for artifact_name, path in [
    ("condition_model.joblib", COND_PATH),
    ("stability_model.joblib", STAB_PATH),
    ("schema.json", SCHEMA_PATH),
]:
    entry = registry.get(artifact_name)
    if entry is None:
        logger.error("Model registry missing %s entry", artifact_name)
        raise RuntimeError(f"Model registry missing {artifact_name} entry")
    if entry.sha256 != compute_file_sha256(path):
        logger.error("Registry SHA mismatch for %s", artifact_name)
        raise RuntimeError(f"Registry SHA mismatch for {artifact_name}: artifact may be tampered")
    logger.info("Verified %s integrity", artifact_name)

_auth = ApiKeyAuthenticator([API_KEY])
_rate = RateLimiter(max_calls=30, window_seconds=1.0)
_audit = AuditTrail()
_gateway = SecureInferenceGateway(_auth, _rate, _audit)
_advisor = MaintenanceAdvisor()
logger.info(
    "API initialized with %s machine types and %d targets",
    len(SCHEMA["machine_types"]),
    len(TARGETS),
)

# human-readable severity ranking so we can pick the worst component to advise on
SEVERITY = {
    "cooler_condition": {100: 0, 20: 2, 3: 3},
    "valve_condition": {100: 0, 90: 1, 80: 2, 73: 3},
    "pump_leakage": {0: 0, 1: 2, 2: 3},
    "accumulator_pressure": {130: 0, 115: 1, 100: 2, 90: 3},
}

app = FastAPI(
    title="Predictive Maintenance — Mobile Hydraulics API",
    description="Group 105 · condition monitoring + stability + RAG repair advisor",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Warm up both models at startup so the first real request hits steady-state latency
_warm = pd.DataFrame([[0] * len(NUMERIC) + [SCHEMA["machine_types"][0]]], columns=FEATURES)
try:
    _condition_model.predict(_warm)
    _stability_model.predict(_warm)
except Exception:
    pass


class SensorReading(BaseModel):
    operating_hours: float = Field(..., example=900)
    pressure_mean_bar: float = Field(..., example=165)
    pressure_std_bar: float = Field(..., example=6.5)
    flow_mean_lpm: float = Field(..., example=7.2)
    oil_temp_mean_c: float = Field(..., example=68)
    vibration_rms_mms: float = Field(..., example=3.5)
    motor_power_kw: float = Field(..., example=18)
    pump_speed_mean_rpm: float = Field(..., example=1400)
    cooling_efficiency_pct: float = Field(..., example=78)
    machine_type: str = Field(..., example="Excavator")


class ComponentResult(BaseModel):
    component: str
    predicted_class: int
    status: str  # "healthy" or "attention"


class PredictionResponse(BaseModel):
    components: list[ComponentResult]
    stability: str  # "stable" / "unstable"
    flagged_component: str | None
    repair_procedure: str | None
    repair_guidance: str | None
    llm_recommendation: str | None
    latency_ms: float


class BatchPredictionRequest(BaseModel):
    readings: list[SensorReading] = Field(..., min_length=1, max_length=1000)


class BatchItemResult(BaseModel):
    components: list[ComponentResult]
    stability: str
    flagged_component: str | None


class BatchPredictionResponse(BaseModel):
    results: list[BatchItemResult]
    count: int
    latency_ms: float
    latency_per_reading_ms: float


def _resolve_components(cond_pred) -> tuple[list[ComponentResult], str | None]:
    """Map raw condition-model classes to per-component results + worst offender."""
    components, worst = [], None
    worst_sev = 0
    for i, target in enumerate(TARGETS):
        cls = int(cond_pred[i])
        healthy = cls == HEALTHY[target]
        components.append(
            ComponentResult(
                component=target, predicted_class=cls, status="healthy" if healthy else "attention"
            )
        )
        sev = SEVERITY.get(target, {}).get(cls, 0)
        if not healthy and sev > worst_sev:
            worst_sev, worst = sev, target
    return components, worst


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_sha256": _model_sha[:12],
        "targets": TARGETS,
        "machine_types": SCHEMA["machine_types"],
        "llm_enabled": bool(os.getenv("OPENROUTER_API_KEY")),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(reading: SensorReading, x_api_key: str = Header(default=API_KEY)):
    t0 = perf_counter()
    payload = reading.model_dump()
    machine_type = payload.pop("machine_type")
    logger.info("Received prediction request for machine_type=%s", reading.machine_type)

    # Filter 1 & 2: Security, Rate-Limiting, and Hydraulic Outlier Filtering
    try:

        def predict_fn(clean_payload):
            return clean_payload

        clean = _gateway.guarded_predict(
            client_id="frontend",
            api_key=x_api_key,
            payload=payload,
            predict_fn=predict_fn,
            bounds=HYDRAULIC_BOUNDS,
        )
    except SecurityError as e:
        logger.warning("Prediction blocked: %s", str(e))
        raise HTTPException(status_code=400, detail=str(e))

    clean["machine_type"] = machine_type
    X = pd.DataFrame([[clean[c] for c in FEATURES]], columns=FEATURES)

    # Filter 3 & 4: Model Registry Validation & Machine Learning Inference
    cond_pred = _condition_model.predict(X)[0]
    stability = "unstable" if int(_stability_model.predict(X)[0]) == 1 else "stable"
    latency = round((perf_counter() - t0) * 1000, 2)

    # Filter 5: Maintenance Advisor Severity Resolution & RAG Enrichment
    components, worst = _resolve_components(cond_pred)

    procedure = guidance = llm = None
    if worst:
        advice = _advisor.advise(worst, k=1)[0]
        procedure, guidance = advice["procedure"], advice["guidance"]
        llm = advice.get("llm_recommendation")

    _audit.record("frontend", "predict", f"stability={stability} worst={worst}")
    logger.info(
        "Prediction completed: stability=%s worst=%s latency=%.2fms", stability, worst, latency
    )

    # Filter 6: Structured Prediction Response Compilation
    return PredictionResponse(
        components=components,
        stability=stability,
        flagged_component=worst,
        repair_procedure=procedure,
        repair_guidance=guidance,
        llm_recommendation=llm,
        latency_ms=latency,
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictionRequest, x_api_key: str = Header(default=API_KEY)):
    """Vectorized fleet-scale inference: N readings, ONE model call per model.

    Scalability path: per-row Python overhead (validation aside) is constant —
    the heavy lifting is a single vectorized sklearn predict over the whole
    frame, so throughput grows with batch size instead of request count. The
    service itself is stateless (models are read-only after startup), so it
    also scales horizontally behind a load balancer.
    """
    t0 = perf_counter()
    n = len(request.readings)
    logger.info("Received batch prediction request with %d readings", n)

    try:
        _auth.authenticate(x_api_key)
        _rate.check("frontend-batch")
        rows = []
        for i, reading in enumerate(request.readings):
            payload = reading.model_dump()
            machine_type = payload.pop("machine_type")
            try:
                clean = validate_sensor_payload(payload, bounds=HYDRAULIC_BOUNDS)
            except SecurityError as e:
                raise SecurityError(f"reading[{i}]: {e}") from e
            clean["machine_type"] = machine_type
            rows.append([clean[c] for c in FEATURES])
    except SecurityError as e:
        logger.warning("Batch prediction blocked: %s", str(e))
        raise HTTPException(status_code=400, detail=str(e))

    X = pd.DataFrame(rows, columns=FEATURES)
    cond_preds = _condition_model.predict(X)  # one vectorized call for all N rows
    stab_preds = _stability_model.predict(X)

    results = []
    for cond_pred, stab in zip(cond_preds, stab_preds):
        components, worst = _resolve_components(cond_pred)
        results.append(
            BatchItemResult(
                components=components,
                stability="unstable" if int(stab) == 1 else "stable",
                flagged_component=worst,
            )
        )

    latency = round((perf_counter() - t0) * 1000, 2)
    _audit.record("frontend-batch", "predict_batch", f"n={n}")
    logger.info("Batch prediction completed: n=%d latency=%.2fms", n, latency)
    return BatchPredictionResponse(
        results=results,
        count=n,
        latency_ms=latency,
        latency_per_reading_ms=round(latency / n, 3),
    )
