"""
Tests for the Predictive-Maintenance-for-Mobile-Hydraulics solution.

Author: Aman Kushwah (2024AC05064) — Group 105

Covers the model pipeline and the application-wide security layer.
Run from the repo root:   pytest -q
"""
import os
import math
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from src.security_layer import (
    validate_sensor_payload, ApiKeyAuthenticator, RateLimiter, AuditTrail,
    compute_file_sha256, verify_model_integrity, SecurityError,
)
from src.maintenance_advisor import (
    MaintenanceAdvisor, load_knowledge_base, TfidfRetriever, Document,
)


# --------------------------------------------------------------------------- #
# Fixtures — a small synthetic hydraulic dataset + trained model
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def dataset():
    rng = np.random.default_rng(42)
    n = 1500
    duty = rng.uniform(0, 1, n)
    df = pd.DataFrame({
        "pressure":    150 + 40*duty + rng.normal(0, 6, n),
        "temperature": 55 + 35*duty + rng.normal(0, 4, n),
        "vibration":   2.0 + 6.0*duty + rng.normal(0, 0.6, n),
        "flow_rate":   90 - 15*duty + rng.normal(0, 3, n),
        "oil_debris":  20 + 120*duty + rng.normal(0, 12, n),
    })
    risk = (0.35*(df.temperature-55)/35 + 0.35*(df.vibration-2)/6
            + 0.30*(df.oil_debris-20)/120)
    prob = 1/(1+np.exp(-10*(risk-0.5)))
    y = (rng.uniform(0, 1, n) < prob).astype(int)
    return df, y


@pytest.fixture(scope="module")
def trained(dataset):
    X, y = dataset
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                          random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=120, max_depth=10,
                                   random_state=42, n_jobs=-1)
    model.fit(Xtr, ytr)
    return model, Xte, yte


# --------------------------------------------------------------------------- #
# Model / pipeline tests
# --------------------------------------------------------------------------- #
def test_model_accuracy_is_reasonable(trained):
    model, Xte, yte = trained
    acc = accuracy_score(yte, model.predict(Xte))
    assert acc > 0.80, f"accuracy too low: {acc:.3f}"


def test_robustness_to_corrupted_sensors(trained):
    """Quality Requirement #1 — accuracy retained under 15% corruption."""
    model, Xte, yte = trained
    clean = accuracy_score(yte, model.predict(Xte))
    rng = np.random.default_rng(7)
    corrupt = Xte.copy()
    mask = rng.uniform(0, 1, corrupt.shape) < 0.15
    corrupt = corrupt.mask(mask, corrupt + rng.normal(0, corrupt.std().values, corrupt.shape))
    corrupt = corrupt.fillna(corrupt.median())
    degraded = accuracy_score(yte, model.predict(corrupt))
    assert degraded / clean >= 0.90, "model not robust enough to corruption"


def test_explainability_available(trained):
    """Quality Requirement #3 — model exposes feature importances."""
    model, Xte, _ = trained
    importances = model.feature_importances_
    assert len(importances) == Xte.shape[1]
    assert math.isclose(importances.sum(), 1.0, abs_tol=1e-6)


# --------------------------------------------------------------------------- #
# Security tests — one per control
# --------------------------------------------------------------------------- #
GOOD = {"pressure": 180, "temperature": 70, "vibration": 4,
        "flow_rate": 85, "oil_debris": 60}


def test_input_validation_accepts_valid():
    assert validate_sensor_payload(GOOD) == {k: float(v) for k, v in GOOD.items()}


@pytest.mark.parametrize("bad", [
    {**GOOD, "pressure": 99999},             # out of range
    {**GOOD, "vibration": -5},               # out of range
    {**GOOD, "temperature": float("nan")},   # NaN
    {k: v for k, v in GOOD.items() if k != "flow_rate"},  # missing field
])
def test_input_validation_rejects_bad(bad):
    with pytest.raises(SecurityError):
        validate_sensor_payload(bad)


def test_api_key_authentication():
    auth = ApiKeyAuthenticator(["good-key"])
    assert auth.authenticate("good-key") is True
    with pytest.raises(SecurityError):
        auth.authenticate("bad-key")
    with pytest.raises(SecurityError):
        auth.authenticate(None)


def test_model_integrity(tmp_path):
    p = tmp_path / "model.bin"
    p.write_bytes(b"trained-model-weights")
    checksum = compute_file_sha256(str(p))
    assert verify_model_integrity(str(p), checksum) is True
    p.write_bytes(b"trained-model-weights-TAMPERED")   # modify the artifact
    with pytest.raises(SecurityError):
        verify_model_integrity(str(p), checksum)


def test_audit_trail_is_tamper_evident():
    audit = AuditTrail()
    audit.record("tech-1", "predict", "EX-1")
    audit.record("tech-1", "predict", "EX-2")
    assert audit.verify_chain() is True
    audit._entries[0]["detail"] = "EX-HACKED"
    assert audit.verify_chain() is False


def test_rate_limiter_blocks_excess():
    rl = RateLimiter(max_calls=3, window_seconds=1.0)
    allowed = 0
    for i in range(6):
        try:
            rl.check("client-x", now=100.0 + i * 0.1)
            allowed += 1
        except SecurityError:
            pass
    assert allowed == 3


def test_rate_limiter_recovers_after_window():
    rl = RateLimiter(max_calls=2, window_seconds=1.0)
    rl.check("c", now=0.0)
    rl.check("c", now=0.1)
    with pytest.raises(SecurityError):
        rl.check("c", now=0.2)
    # after the window slides past, calls are allowed again
    assert rl.check("c", now=2.0) is True


# --------------------------------------------------------------------------- #
# RAG Maintenance Advisor tests
# --------------------------------------------------------------------------- #
def test_knowledge_base_loads():
    kb = load_knowledge_base()
    assert len(kb) >= 5
    assert all(isinstance(d, Document) and d.text for d in kb)


def test_retriever_ranks_relevant_doc_first():
    retriever = TfidfRetriever().add([
        Document("Pump", "internal pump leakage falling flow rate volumetric efficiency"),
        Document("Cooler", "cooler efficiency drop high oil temperature thermal load"),
    ])
    hits = retriever.retrieve("pump leakage low flow", k=1)
    assert hits[0][0].title == "Pump"
    assert hits[0][1] > 0


@pytest.mark.parametrize("component,expected_keyword", [
    ("pump_leakage", "pump"),
    ("cooler_condition", "cooler"),
    ("accumulator_pressure", "accumulator"),
])
def test_advisor_retrieves_matching_procedure(component, expected_keyword):
    advisor = MaintenanceAdvisor()
    result = advisor.advise(component, k=1)[0]
    assert expected_keyword in result["procedure"].lower()
    assert 0.0 <= result["relevance"] <= 1.0
