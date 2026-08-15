"""Idempotency & reproducibility tests (colleague feedback, QA).

Author: Group 105

Verifies that the system is deterministic end-to-end:

  * retraining with the same seed on the same data yields byte-identical
    predictions and probabilities;
  * the model registry is idempotent — re-registering an artifact updates the
    single existing entry instead of duplicating it;
  * the training script's up-to-date guard correctly detects current,
    missing and tampered artifacts (this is what makes
    `python train_and_save.py` safe to run repeatedly).

Run from the repo root:   pytest -m idempotency -q
"""

import numpy as np
import pytest

import src.ml.train as train_mod
from src.model_registry import ModelArtifact, ModelRegistry
from src.security_layer import compute_file_sha256
from tests._metrics import record_metric

pytestmark = pytest.mark.idempotency

FEATURES = train_mod.NUMERIC_FEATURES + train_mod.CATEGORICAL_FEATURES


@pytest.fixture(scope="module")
def sample(telemetry):
    # Sampling criteria documented in tests/conftest.py (fixed seed, 2000 rows)
    return telemetry.sample(n=2000, random_state=11)


def test_retraining_is_reproducible(sample):
    """Two training runs with identical seed/data must agree on every output.

    Class predictions must match exactly; probabilities are compared within
    1e-12 because parallel tree averaging (n_jobs=-1) can reorder float sums,
    which shifts results only at machine epsilon (~1e-16).
    """
    holdout = sample.iloc[:200]
    fit_data = sample.iloc[200:]

    probas, preds = [], []
    for _ in range(2):
        model = train_mod.build_stability_model()
        model.fit(fit_data[FEATURES], fit_data[train_mod.RT_TARGET])
        probas.append(model.predict_proba(holdout[FEATURES]))
        preds.append(model.predict(holdout[FEATURES]))

    max_diff = float(np.max(np.abs(probas[0] - probas[1])))
    record_metric("idempotency.max_probability_diff_across_retrains", max_diff)
    np.testing.assert_array_equal(preds[0], preds[1])
    np.testing.assert_allclose(probas[0], probas[1], atol=1e-12)


def test_condition_model_builder_is_reproducible(sample):
    """The multi-output condition model must also retrain deterministically."""
    holdout = sample.iloc[:200]
    fit_data = sample.iloc[200:]

    preds = []
    for _ in range(2):
        model = train_mod.build_condition_model()
        model.fit(fit_data[FEATURES], fit_data[train_mod.TARGETS])
        preds.append(np.asarray(model.predict(holdout[FEATURES])))

    np.testing.assert_array_equal(preds[0], preds[1])


def test_registry_registration_is_idempotent(tmp_path):
    """Registering the same artifact N times keeps exactly one index entry."""
    registry = ModelRegistry(tmp_path)
    artifact_file = tmp_path / "model.joblib"
    artifact_file.write_bytes(b"weights-v1")
    sha = compute_file_sha256(artifact_file)

    for _ in range(3):
        registry.register(ModelArtifact(path="model.joblib", sha256=sha, metadata={"type": "m"}))

    assert list(registry.list().keys()) == ["model.joblib"]
    assert registry.get("model.joblib").sha256 == sha


def test_up_to_date_guard_detects_current_missing_and_tampered(tmp_path, monkeypatch):
    """The guard behind `train_and_save.py` skip-logic must be trustworthy."""
    monkeypatch.setattr(train_mod, "REGISTRY", tmp_path)
    registry = ModelRegistry(tmp_path)

    # empty registry directory -> not up to date
    assert train_mod.artifacts_up_to_date(registry) is False

    # all artifacts present + checksums registered -> up to date
    for name in train_mod.ARTIFACT_NAMES:
        path = tmp_path / name
        path.write_bytes(f"content-of-{name}".encode())
        registry.register(
            ModelArtifact(path=name, sha256=compute_file_sha256(path), metadata={"type": "t"})
        )
    assert train_mod.artifacts_up_to_date(registry) is True

    # tampering with one artifact -> guard must notice
    (tmp_path / train_mod.ARTIFACT_NAMES[0]).write_bytes(b"TAMPERED")
    assert train_mod.artifacts_up_to_date(registry) is False
