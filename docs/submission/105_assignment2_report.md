---
title: "AIMLCZG546 — Software Engineering for Machine Learning"
subtitle: "Assignment II — Predictive Maintenance in Mobile Hydraulic Systems"
---

# BITS Pilani — Work Integrated Learning Programmes

## M.Tech. Artificial Intelligence & Machine Learning

**Course:** AIMLCZG546 — Software Engineering for Machine Learning

# Assignment II

**Work Title:** *Predictive Maintenance in Mobile Hydraulic Systems*

**GitHub repository:** heavy-machinery-predictive-maintenance-pipeline

## Group No. 105

### Group Member Names with Contribution

| Sl. No | BITS ID | Name | Contribution of team member (Qualitative) | Percentage Contribution (Out of 100) |
|---|---|---|---|---|
| 1 | 2024AC05744 | Srinivasan R | Code refactoring into OOP modules (`src/core`, `src/api`), Model Registry design, architectural patterns, code review and system integration. | 100% |
| 2 | 2024AC05100 | Vineet Kumar | Docker-based deployment of API + frontend, CI support, linting/formatting toolchain (black, isort, flake8) and lint reports. | 100% |
| 3 | 2024AD05482 | Vibhav Sharma | Data-quality metrics module (schema validation, missing-value checks, PSI drift detection), production testing strategy write-up, report preparation. | 100% |
| 4 | 2024AC05064 | Aman Kushwah | Model training/evaluation, FastAPI REST API, security layer, pytest test suite (unit, integration, ML training/inference and data-validation tests), documentation. | 100% |

\newpage

# 1. Introduction

This assignment builds on our Assignment I system: an IoT-based predictive-maintenance
application for mobile hydraulic systems (excavators, telehandlers, backhoe loaders).
The system ingests fleet telemetry (pressure, flow, oil temperature, vibration, motor
power, pump speed, cooling efficiency), predicts the health of four hydraulic
components (cooler, valve, pump, accumulator) plus an overall stability flag, and
serves the results through a FastAPI backend with a React frontend and a RAG-based
maintenance advisor.

Assignment II focuses on **implementation quality and testing/QA**. This report walks
through each requirement of the two objectives with evidence from the repository.

**Repository layout (production package):**

```
├── api_server.py            # thin uvicorn entry point -> src/api/server.py
├── train_and_save.py        # production training script -> model_registry/
├── src/
│   ├── api/server.py        # FastAPI REST API (endpoints, schemas, status codes)
│   ├── core/model_registry.py  # on-disk model registry w/ SHA-256 integrity
│   ├── core/pipeline.py     # pipe-and-filter composition helpers
│   ├── ml/train.py          # modular training pipeline
│   ├── quality/data_quality.py # schema / missing-value / drift metrics  (NEW)
│   ├── rag/maintenance_advisor.py # TF-IDF retriever + advisor
│   └── security/security_layer.py # auth, validation, rate limit, audit
├── tests/
│   ├── test_predictive_maintenance.py  # unit + integration tests
│   ├── test_ml_training.py             # ML training tests            (NEW)
│   ├── test_ml_inference.py            # ML inference tests           (NEW)
│   └── test_data_quality.py            # data-validation tests        (NEW)
├── notebook/105.ipynb       # research notebook (Assignment I prototype)
├── data/hydraulic_fleet_telemetry.csv
├── docs/lint/               # before/after lint + pytest reports
└── docker-compose.yml, Dockerfile, frontend/
```

\newpage

# 2. Objective 1 — Implementation and Code Sharing

## 2.1 Task 1 — Refactoring with OOP / functional principles

The application is decomposed into single-responsibility modules under `src/`,
combining OOP (classes for stateful components) with functional style (pure
functions for stateless computations):

| Module | Responsibility | Principle / Pattern |
|---|---|---|
| `src/ml/train.py` | Data ingestion, preprocessing, model training | Pipeline pattern (`sklearn.Pipeline` + `ColumnTransformer`), pure `main()` |
| `src/core/model_registry.py` | Artifact registration + SHA-256 integrity index | **Model Registry** pattern, `@dataclass ModelArtifact`, single responsibility |
| `src/api/server.py` | REST inference service | Layered architecture; Pydantic schemas as DTOs |
| `src/security/security_layer.py` | Auth, input validation, rate limiting, audit | **Facade** (`SecureInferenceGateway`) composing `ApiKeyAuthenticator`, `RateLimiter`, `AuditTrail` |
| `src/rag/maintenance_advisor.py` | Knowledge-base retrieval for repair guidance | Strategy-style retriever (`TfidfRetriever`), immutable `Document` dataclass |
| `src/quality/data_quality.py` | Schema, missing-value and drift metrics | Pure functions + `DataQualityReport` dataclass |

Top-level shims (`src/model_registry.py`, `src/security_layer.py`,
`src/maintenance_advisor.py`) are thin **facades** that re-export the concrete
implementations, so callers depend on stable import paths while implementations
can evolve independently.

Example — the registry artifact as an immutable value object plus a
single-responsibility registry class (`src/core/model_registry.py`):

```python
@dataclass
class ModelArtifact:
    path: str
    sha256: str
    metadata: dict[str, Any]

class ModelRegistry:
    """Minimal on-disk model registry for the hydraulic predictive-maintenance system."""
    def register(self, artifact: ModelArtifact) -> None: ...
    def get(self, path: str) -> ModelArtifact | None: ...
    def list(self) -> dict[str, Any]: ...
```

## 2.2 Task 2 — Research code vs production code

We demonstrate the contrast on the **model-training component**, which exists in
both forms in the repository:

| Aspect | Research code — `notebook/105.ipynb` | Production code — `train_and_save.py` + `src/` |
|---|---|---|
| Purpose | Exploration: EDA, feature experiments, model comparison | Repeatable artifact production for the API |
| Structure | Linear cells, global state, inline plots | Functions/classes, `main()` entry point, importable modules |
| Reproducibility | Depends on cell execution order | Deterministic script: `python train_and_save.py`; DVC stage (`dvc.yaml`) pins data + code deps |
| Error handling | None — exceptions surface in the cell | Explicit checks (dataset presence, artifact load failures) with logging and meaningful exceptions |
| Outputs | Transient (in-memory, plots) | Versioned artifacts in `model_registry/` with SHA-256 checksums + `schema.json` contract |
| Testing | Manual visual inspection | `pytest` suite (37 tests), lint-clean under flake8/black/isort |
| Interface | A human reading the notebook | The FastAPI service and CI consume the artifacts |

The notebook remains the *system of record for experiments*; the `src/` package is
the *system of record for behaviour*. The training logic promoted to production was
rewritten with schema contracts (`model_registry/schema.json`) so the API never
hard-codes feature lists.

## 2.3 Task 3 — Error handling and logging

Python's `logging` module is configured in four critical modules, each with a
distinct tag and meaningful levels:

| Module | Logger | INFO | WARNING | ERROR |
|---|---|---|---|---|
| `src/api/server.py` | `[API]` | request received / completed, artifact verification | prediction blocked by security layer | missing artifacts, SHA mismatch, load failure (`logger.exception`) |
| `src/security/security_layer.py` | `hydraulics.security` | guarded predict lifecycle | — (violations raise `SecurityError`) | auth/validation failures |
| `src/core/model_registry.py` | `[REGISTRY]` | artifact registered | — | — |
| `src/quality/data_quality.py` | `[DATA-QUALITY]` | checks passed with measured rates | missing-rate budget exceeded, drift detected | schema violations, fully-missing column |

Errors are typed: the security layer raises `SecurityError`, the data-quality module
raises `DataQualityError`, and the API converts security failures into HTTP 400
responses rather than crashing:

```python
try:
    clean = _gateway.guarded_predict(client_id="frontend", api_key=x_api_key,
                                     payload=payload, predict_fn=predict_fn,
                                     bounds=HYDRAULIC_BOUNDS)
except SecurityError as e:
    logger.warning("Prediction blocked: %s", str(e))
    raise HTTPException(status_code=400, detail=str(e))
```

Example log line from the data-quality module (WARNING level firing on injected
drift during tests):

```
2026-08-12 17:51:41 [DATA-QUALITY] WARNING Drift detected on oil_temp_mean_c: PSI=1.877 (threshold 0.25)
```

## 2.4 Task 4 — Code formatting and linting (before / after)

Toolchain: **flake8** (linting), **black** (formatting), **isort** (import
ordering), run over `src/`, `tests/`, `api_server.py`, `train_and_save.py`.
Full reports are committed under `docs/lint/`.

**Before** (`docs/lint/*_before.txt`):

```
flake8 : 20 violations (F401 unused imports, E501 long lines, E402 imports
         not at top, E702 multiple statements, E225 spacing, E741 ambiguous name)
black  : "8 files would be reformatted, 11 files would be left unchanged."
isort  : 7 files with incorrectly sorted imports
```

Representative flake8-before excerpt:

```
src/api/server.py:26:1: F401 'src.security_layer.validate_sensor_payload' imported but unused
src/api/server.py:102:36: E702 multiple statements on one line (semicolon)
src/rag/maintenance_advisor.py:121:39: E741 ambiguous variable name 'l'
train_and_save.py:57:1: E402 module level import not at top of file
```

**Actions:** `isort --profile black` and `black --line-length 100` applied to the
whole codebase; remaining semantic issues (unused imports, ambiguous variable
name, import placement) fixed by hand.

**After** (`docs/lint/*_after.txt`):

```
flake8 : 0 violations
black  : "24 files would be left unchanged."
isort  : clean (no errors)
```

## 2.5 Task 5 — REST API design (FastAPI)

`src/api/server.py` exposes the inference functionality with explicit
request/response schemas and status codes. Interactive documentation is
auto-generated at `http://localhost:8000/docs`.

| Endpoint | Method | Request schema | Response schema | Status codes |
|---|---|---|---|---|
| `/health` | GET | — | service status, model SHA-256, targets, machine types | 200 |
| `/predict` | POST | `SensorReading` (10 typed fields with examples) | `PredictionResponse` (per-component results, stability, RAG guidance, latency) | 200; 400 invalid API key / out-of-range input; 422 malformed body (FastAPI validation) |

Good-practice elements:

- **Typed contracts:** Pydantic models (`SensorReading`, `ComponentResult`,
  `PredictionResponse`) validate every request and serialize every response.
- **Meaningful status codes:** security violations → 400 with a clear `detail`;
  schema violations → 422 automatically; healthy service → 200.
- **Security at the boundary:** API-key header, per-client rate limiting,
  physical-bounds input validation and an audit trail wrap every prediction.
- **Operational hygiene:** model warm-up at startup, artifact SHA verification
  before serving, latency reported in every response, CORS configured for the
  frontend, Docker health check on `/health`.

Application screenshots (frontend consuming the API):

![Sensor input form](../screenshots/app_input.png)

![Prediction result with RAG guidance](../screenshots/app_prediction.png)

\newpage

# 3. Objective 2 — Quality Assurance

## 3.1 Task 6 — Test types implemented with pytest

The suite contains **37 tests** across **three distinct types** (evidence:
`docs/lint/pytest_report.txt`, `37 passed`):

1. **Unit tests** (`tests/test_predictive_maintenance.py`) — security primitives
   in isolation: input validation, API-key auth, rate limiter, SHA-256 model
   integrity, tamper-evident audit trail, TF-IDF retriever ranking.
2. **Integration tests** (same file, `TestClient(app)`) — the real FastAPI app
   with real trained artifacts: `/health` liveness and `/predict` rejecting a
   bad API key end-to-end through the security gateway.
3. **Data-validation tests** (`tests/test_data_quality.py`) — the shipped
   dataset against the schema, injected corruption, missing-value budgets and
   drift detection (details in 3.3).

## 3.2 Task 7 — ML-specific tests

**7a. Model training** (`tests/test_ml_training.py`):

- `test_model_can_overfit_small_batch` — a high-capacity RandomForest must reach
  ≥99% accuracy on a 64-row batch; failure indicates broken features/labels or a
  mis-wired training loop.
- `test_training_loss_decreases_with_iterations` — training log-loss over
  `GradientBoostingClassifier.staged_predict_proba` must be monotonically
  non-increasing and end below its starting value.
- `test_production_pipeline_beats_majority_baseline` — the exact production
  preprocessing+model pipeline must beat the majority-class baseline on held-out
  data, proving end-to-end learning.

**7b. Model inference** (`tests/test_ml_inference.py`):

- **Shape/range checks** — the multi-output condition model returns shape
  `(n, 4)` with classes inside each target's label set; stability
  `predict_proba` lies in [0, 1] and rows sum to 1.
- **Invariance tests** — identical inputs give identical outputs (determinism),
  and an unseen `machine_type` ("MoonCrawler-9000") is handled gracefully by
  `OneHotEncoder(handle_unknown="ignore")` instead of crashing.
- **Directional test** — severely degraded sensors (oil 95 °C, vibration
  12 mm/s, cooling 25%) must **not decrease** the predicted instability
  probability relative to a healthy reading.

## 3.3 Task 8 — Model-quality and data-quality metrics

**a. Model quality** — measured on a 20% stratified held-out split by
`train_and_save.py` and persisted into `model_registry/schema.json`:

| Target | Accuracy | Macro-F1 |
|---|---|---|
| cooler_condition | 0.937 | 0.931 |
| valve_condition | 0.890 | 0.884 |
| pump_leakage | 0.936 | 0.914 |
| accumulator_pressure | 0.881 | 0.875 |
| stability_flag (real-time model) | 0.909 | — |

Two metric families are used: **accuracy** (overall correctness) and
**macro-F1** (robustness to class imbalance across degradation levels — critical
here because failure classes are rarer than healthy classes).

**b. Data quality** — implemented in `src/quality/data_quality.py` and measured
on the shipped dataset (10,000 cycles):

| Metric | Definition | Measured value |
|---|---|---|
| Schema validation | 17 columns: presence, dtype family, physical ranges (e.g. oil temp within −20…130 °C) and allowed label sets | **PASS** — 0 violations |
| Missing-value rate | per-column NaN rate vs 5% budget | worst rate **0.0%** (training additionally median-imputes noisy sensor columns defensively) |
| Drift (PSI) | Population Stability Index, first half of fleet history vs second half; threshold 0.25 | pressure 0.014, flow 0.012, oil temp 0.014, vibration 0.012 — **all stable** |

The same PSI metric fires correctly when drift is injected in tests
(+25 °C fleet-wide oil-temperature shift → PSI = 1.88 > 0.25, flagged with a
WARNING log).

## 3.4 Task 9 — Testing in production & security consideration

**Production experimentation approach — shadow deployment, then canary.**
For a maintenance system, wrong predictions have asymmetric costs (a missed
failure strands a machine in the field), so we would first deploy a new model in
**shadow mode**: the API keeps serving the current `model_registry` version
while the candidate model receives a mirrored copy of every `/predict` request
and its outputs are only logged. Because ground truth (actual component
condition found at service time) arrives with delay, shadow predictions are
joined against maintenance records to compare accuracy/macro-F1 offline —
with zero user impact. Once the candidate matches or beats the champion, a
**canary release** routes a small slice of fleet traffic (e.g. one machine type,
5% of requests) to it behind the same API contract, monitored on the metrics we
already log per request (latency, stability rate, audit trail) and on the PSI
drift monitor from `src/quality`. Docker + the model registry make rollback a
one-line revert to the previous checksummed artifact.

**Security consideration — input validation against adversarial/corrupted
inputs (implemented).** Every `/predict` request passes through
`validate_sensor_payload` with `HYDRAULIC_BOUNDS`: each sensor value must be a
finite number inside its physically plausible range, otherwise the request is
rejected with HTTP 400 before it ever reaches the model. This blocks NaN/Inf
payloads, out-of-range values that would push the model far outside its training
distribution, and type-confusion attacks. Defence in depth around it: API-key
authentication, per-client rate limiting (30 req/s) against scraping/DoS,
SHA-256 registry verification so tampered model artifacts refuse to load, and a
hash-chained audit trail making prediction history tamper-evident.

\newpage

# 4. How to run

```bash
# full stack (frontend + API):
docker compose up --build          # frontend :5173, API docs :8000/docs

# training (produces model_registry/):
python train_and_save.py

# tests:
pytest -q                          # 37 passed

# lint gates:
flake8 --max-line-length=100 src tests api_server.py train_and_save.py
black --check --line-length 100 src tests api_server.py train_and_save.py
isort --check-only --profile black --line-length 100 src tests
```

# 5. Evidence index (in repository)

| Artifact | Path |
|---|---|
| Lint reports before/after | `docs/lint/flake8_before.txt`, `flake8_after.txt`, `black_before.txt`, `black_after.txt`, `isort_before.txt`, `isort_after.txt` |
| Pytest report (37 passed) | `docs/lint/pytest_report.txt` |
| Model metrics | `model_registry/schema.json` |
| Research notebook | `notebook/105.ipynb` |
| Production package | `src/`, `train_and_save.py`, `api_server.py` |
| New QA modules/tests | `src/quality/`, `tests/test_ml_training.py`, `tests/test_ml_inference.py`, `tests/test_data_quality.py` |
| App screenshots | `docs/screenshots/` |
