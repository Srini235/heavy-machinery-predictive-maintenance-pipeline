# Predictive Maintenance of Mobile Hydraulic Systems

> Repository: **`heavy-machinery-predictive-maintenance-pipeline`** — project topic is
> **Predictive Maintenance of Mobile Hydraulic Systems**.

An ML-based system that reads hydraulic sensor data (pressure, temperature, vibration, flow,
oil debris) from a fleet of excavators and predicts which machine needs maintenance **only when
it actually needs it** — a smarter "check-engine light" that says *which* component, *how
confident*, and *why*, and then retrieves the matching repair procedure.

---
### Academic Submission Details
- **Course:** AIMLCZG546 - Software Engineering for Machine Learning
- **Assignments:** Assignment 1 & Assignment 2 (Weightage: 10 Marks each)
- **Group Details:** Group No. 105

<div align="center">

**Team Contribution — Software Engineering for Machine Learning · Group 105**

| Serial No | BITS ID | Student Name | Contribution % |
| :---: | :---: | :--- | :--- |
| 1 | 2024AC05744 | Srinivasan R | 100 |
| 2 | 2024AC05100 | Vineet Kumar  | 100 |
| 3 | 2024AD05482 | Vibhav Sharma | 100 |
| 4 | 2024AC05064 | Aman Kushwah | 100 |

</div>

---
## What's in this repo

| Area | Files |
| :--- | :--- |
| **Solution notebook** (model, patterns, security, RAG — executed) | `notebook/105.ipynb` |
| **GR4ML report** (Objective 1) | `readme_others/GR4ML_REPORT.md` |
| **Architecture diagram** (Objective 2, ML + non-ML) | `docs/screenshots/architecture_diagram.png` |
| **Web app** — React (shadcn/ui) frontend + FastAPI backend | `frontend/`, `api_server.py`, `train_and_save.py` |
| **Reusable modules** | `src/security_layer.py`, `src/maintenance_advisor.py` |
| **Data quality module** (schema / missing values / PSI drift) | `src/quality/data_quality.py` |
| **Knowledge base** (RAG) | `data/hydraulic_maintenance_manual.md` |
| **Tests** (37 passing — unit, integration, ML training/inference, data validation) | `tests/` |
| **Lint & test reports** (before/after evidence, Assignment 2) | `docs/lint/` |
| **Assignment 2 submission report** | `docs/submission/105.docx` |
| **Docker pipeline** | `Dockerfile`, `frontend/Dockerfile`, `docker-compose.yml` |
| **CI pipeline** (lint gates → train → full test suite) | `.github/workflows/ci_pipeline.yml` |
| **Docs** | `readme_others/RUN.md`, `readme_others/FRONTEND_README.md` |
| **Screenshots** | `docs/screenshots/` |

## Architecture patterns
- **Architectural:** Pipe-and-Filter (data pipeline) · Microservices (decoupled services)
- **Design / MLOps:** Model Registry · Batch-vs-Real-time Serving

## Stepwise Architecture & Implementation (where to find it)

1. Data ingestion & layout
	- File: `data/hydraulic_fleet_telemetry.csv`
	- Notes: dataset is a one-row-per-cycle synthetic telemetry export used for training.

2. Training pipeline (idempotent)
	- File: `train_and_save.py`
	- Steps implemented:
	  - Idempotency guard checks `model_registry/index.json` and artifact sha256s.
	  - Loads data only when retraining is required (or `--force` used).
	  - Builds a `ColumnTransformer` + `Pipeline` for preprocessing and model.
	  - Trains multi-output condition model + stability classifier.
	  - Saves artifacts to `model_registry/` and registers them.

3. Model registry (artifact management)
	- File: `src/core/model_registry.py`
	- Notes: lightweight on-disk registry mapping filenames -> sha256 + metadata.

4. Security & inference checks
	- File: `src/security/security_layer.py` and facade `src/security_layer.py`
	- Responsibilities: API-key auth, input validation, model integrity verification,
	  audit logging and rate limiting. The facade keeps imports stable for callers.

5. Serving / API
	- File: `api_server.py` (wrapper) and `src/api/server.py`
	- Behavior: On startup the API validates model files against the registry and
	  refuses to start if checksums do not match, ensuring deployment integrity.

6. Observability
	- Optional MLflow logging is included in `train_and_save.py` when
	  `MLFLOW_TRACKING_URI` is set; metrics and artifacts are recorded per run.

7. Tests (37 total)
	- `tests/test_predictive_maintenance.py` — unit tests (security primitives, RAG
	  retriever) and API integration tests via FastAPI `TestClient`.
	- `tests/test_ml_training.py` — ML training tests: overfit-a-small-batch sanity
	  check, boosting loss decreases per iteration, pipeline beats majority baseline.
	- `tests/test_ml_inference.py` — ML inference tests: output shape/class-set and
	  probability-range checks, determinism/unseen-category invariance, directional
	  test (degraded sensors must not lower predicted risk).
	- `tests/test_data_quality.py` — data validation tests against the real dataset
	  and corrupted copies (schema, missing-value budget, PSI drift).

8. Data quality metrics
	- File: `src/quality/data_quality.py` — schema validation, per-column
	  missing-value rates, and Population Stability Index (PSI) drift detection,
	  with INFO/WARNING/ERROR logging. Used by the data-validation tests and
	  suitable for scheduled production monitoring.

9. Code style & CI
	- The codebase is kept clean under `flake8` (max line length 100), `black`
	  and `isort` (before/after reports in `docs/lint/`). The GitHub Actions
	  workflow (`.github/workflows/ci_pipeline.yml`) enforces the lint gates,
	  retrains the models, and runs the full 37-test suite on every push/PR.

See `DESIGN.md` for a short architecture summary, implemented patterns, security
improvements, and open checklist items for future hardening.

## Quality requirements
1. **Robustness to noisy data** — ≥ 95% of clean accuracy retained under 15% corrupted readings
2. **Low latency** — safety-critical stability inference in under 100 ms
3. **Explainability** — every flag names its top contributing sensors
4. **Security** — input validation, API-key auth, model integrity, audit logging, rate limiting

## How to run
This project is folder-name independent for runtime execution. The code imports `src` as a package and uses absolute root-relative paths in the backend, so you can run commands from the repository root regardless of the parent folder name.

See **[RUN.md](readme_others/RUN.md)** for full steps. Quick start:

```bash
# Option A — Docker (one command)
docker compose up --build            # web: http://localhost:5173  · api: http://localhost:8000/docs

# Option B — API + training from repository root
python train_and_save.py              # creates model_registry artifacts
python -m uvicorn api_server:app --reload --port 8000

# Option B alternative — force retrain
python train_and_save.py --force
```

Run the tests (train first — the API integration tests load `model_registry/`):
```bash
python train_and_save.py
python -m pytest tests/ -q                                   # 37 passed
```

Run the lint gates (same commands CI enforces):
```bash
flake8 --max-line-length=100 src tests api_server.py train_and_save.py
black --check --line-length 100 src tests api_server.py train_and_save.py
isort --check-only --profile black --line-length 100 src tests api_server.py train_and_save.py
```
