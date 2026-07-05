# Design Overview

This document summarizes the project architecture, implemented patterns, security improvements, and open checklist items for future hardening.

## Implemented patterns

### Architectural patterns
- **Pipe-and-Filter**: `train_and_save.py` builds an ML pipeline using `ColumnTransformer` and `Pipeline` to separate preprocessing from model logic.
- **Microservices / decoupled services**: The React frontend is decoupled from the FastAPI backend (`frontend/` + `api_server.py` / `src/api/server.py`).
- **Batch vs Real-time Serving**: Batch training is performed by `train_and_save.py`; real-time prediction is served by the API.
- **Model Registry**: `src/core/model_registry.py` stores artifact metadata and checksums in `model_registry/index.json`.

### Design patterns
- **Registry**: The `ModelRegistry` class manages artifact metadata and supports idempotency checks.
- **Pipeline**: The training stage composes `ColumnTransformer` and `Pipeline` for data preprocessing and model training.
- **Facade**: `src/security_layer.py` re-exports security primitives and `SecureInferenceGateway` composes authentication, validation, rate limiting, and audit logging.
- **Single Responsibility / Modularization**: Training, serving, security, and artifact management are separate modules.
- **Strategy / Dependency Injection**: `SecureInferenceGateway.guarded_predict` accepts a `predict_fn` callback.
- **Idempotent Guard**: `train_and_save.py` checks existing artifacts and registry checksums before retraining.

## High-level architecture

```text
[ data/hydraulic_fleet_telemetry.csv ]
             |
             v
     [ train_and_save.py ]
             | saves artifacts to
             v
 [ model_registry/index.json + .joblib + schema.json ]
             |
             v
 [ src/api/server.py ] --uses--> [SecureInferenceGateway] --> [model integrity checks]
             |
             v
      [ frontend request flow ]
```

## Security improvements

- Input validation against physically plausible bounds.
- API-key authentication using hashed keys and constant-time comparison.
- Model integrity / anti-tamper checks using SHA-256.
- Tamper-evident audit logging via chained hashes.
- Request rate limiting with a sliding window.

## Open checklist items

- [ ] Persist `AuditTrail` entries to durable storage instead of keeping them in memory.
- [ ] Replace the in-memory `RateLimiter` with a shared store (Redis, DB) for distributed deployments.
- [ ] Add atomic file writes for model artifacts and schema files.
- [ ] Add signature files for artifact metadata and optionally sign registry index entries.
- [ ] Move API secret management to environment-only secrets or a secret manager.
- [ ] Add CI checks for corrupted model registry / SHA mismatch during startup.
- [ ] Add integration tests for security failure modes, including invalid API key, out-of-range payload, and rate-limit enforcement.

## Notes on folder naming

The project code is intentionally folder-name independent. The important requirement is that the repository root is on Python's import path when running commands like `python train_and_save.py` or `python -m uvicorn api_server:app --reload --port 8000`.

`train_and_save.py` detects the repository root by walking upward from its own file location and loads `data/hydraulic_fleet_telemetry.csv` from there. `api_server.py` now resolves the `model_registry/` path using `Path(__file__).resolve().parent`, so it does not depend on the current working directory.

The README examples still use `cd mediclaim-ai-insurance-agent` because that is the local workspace path in this review copy, but the code itself works from any root folder name.
