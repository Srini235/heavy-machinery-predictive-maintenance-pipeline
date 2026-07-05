# Predictive Maintenance of Mobile Hydraulic Systems

> Repository name is `mediclaim-ai-insurance-agent` (retained from the original repo); the
> project topic is **Predictive Maintenance of Mobile Hydraulics**.

An ML-based system that reads hydraulic sensor data (pressure, temperature, vibration, flow,
oil debris) from a fleet of excavators and predicts which machine needs maintenance **only when
it actually needs it** — a smarter "check-engine light" that says *which* component, *how
confident*, and *why*, and then retrieves the matching repair procedure.

---
### Academic Submission Details
- **Course:** AIMLCZG546 - Software Engineering for Machine Learning
- **Assignment:** Assignment 1 (Weightage: 10 Marks)
- **Group Details:** Group No. 105

<div align="center">

**Team Contribution — Software Engineering for Machine Learning · Assignment 1 · Group 105**

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
| **Solution notebook** (model, patterns, security, RAG — executed) | `105.ipynb`, `105.html` |
| **GR4ML report** (Objective 1) | `GR4ML_REPORT.md` |
| **Architecture diagram** (Objective 2, ML + non-ML) | `architecture_diagram.png` |
| **Web app** — React (shadcn/ui) frontend + FastAPI backend | `frontend/`, `api_server.py`, `train_and_save.py` |
| **Reusable modules** | `src/security_layer.py`, `src/maintenance_advisor.py` |
| **Knowledge base** (RAG) | `data/hydraulic_maintenance_manual.md` |
| **Tests** (18 passing) | `tests/test_predictive_maintenance.py` |
| **Docker pipeline** | `Dockerfile`, `frontend/Dockerfile`, `docker-compose.yml` |
| **Docs** | `RUN.md`, `FRONTEND_README.md`, `AMAN_CONTRIBUTION.md` |
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

7. Tests
	- File: `tests/test_predictive_maintenance.py` — unit and integration tests
	  validating the pipeline and inference behaviors.

See `DESIGN.md` for a short architecture summary, implemented patterns, security
improvements, and open checklist items for future hardening.
3. **Explainability** — every flag names its top contributing sensors
4. **Security** — input validation, API-key auth, model integrity, audit logging, rate limiting

## How to run
See **[RUN.md](RUN.md)** for full steps. Quick start:

```bash
# Option A — Docker (one command)
docker compose up --build            # web: http://localhost:5173  · api: http://localhost:8000/docs

# Option B — notebook only
pip install -r requirements-notebook.txt
jupyter notebook 105.ipynb           # or open 105.html
```

Run the tests:
```bash
python3 -m pytest tests/test_predictive_maintenance.py -q     # 18 passed
```
