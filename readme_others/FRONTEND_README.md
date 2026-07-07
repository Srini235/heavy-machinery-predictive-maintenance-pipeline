# Web App — React (shadcn/ui) + FastAPI

Author: Aman Kushwah (2024AC05064) — Group 105

A working web application for the predictive-maintenance models: enter per-cycle hydraulic
telemetry and get each component's condition, circuit stability, and RAG repair guidance (with an
optional LLM recommendation). This is the **"application" the assignment asks you to screenshot.**

- **Frontend:** React + Vite + TypeScript + Tailwind + **shadcn/ui** (`frontend/`)
- **Backend:** FastAPI (`api_server.py`) — serves the multi-output condition model + stability
  model, guarded by the security layer, enriched by the RAG + OpenRouter LLM advisor.

Screenshots are in `docs/screenshots/` (`app_input.png`, `app_prediction.png`).

---

## Prerequisites
- Python 3.10+ (backend) and Node.js 18+ (frontend). Both confirmed working in this repo.

## Run it locally (two terminals)

**Terminal 1 — backend (port 8000):**
```bash
cd heavy-machinery-predictive-maintenance-pipeline

# one-time: venv + deps
python3 -m venv venv
source venv/bin/activate                 # Windows: venv\Scripts\activate
pip install -r requirements-api.txt

# (optional) enable the LLM recommendation via OpenRouter
cp .env.example .env                      # paste your OPENROUTER_API_KEY into .env
set -a; . ./.env; set +a

# train + save both models (creates model_registry/*.joblib from the dataset CSV)
python3 train_and_save.py

# start the API
uvicorn api_server:app --reload --port 8000
#   docs at http://localhost:8000/docs
```

**Terminal 2 — frontend (port 5173):**
```bash
cd heavy-machinery-predictive-maintenance-pipeline/frontend
npm install          # one-time
npm run dev
#   open http://localhost:5173
```

Then open **http://localhost:5173**, pick a preset (Healthy / Degraded) or type sensor values +
machine type, and press **Assess Machine**.

> Tip: open `http://localhost:5173/?demo=1` to auto-run one assessment on load (used for screenshots).

## What the app shows
- **Machine type** dropdown + **9 sensor inputs** (per-cycle telemetry).
- **Circuit stability** — stable / unstable (real-time stability model).
- **Component conditions** — cooler / valve / pump / accumulator, each OK or Attention
  (multi-output condition model).
- **Repair guidance** — the RAG advisor retrieves the procedure for the worst-flagged component;
  with an OpenRouter key set it also shows an **AI recommendation (RAG + LLM)**.
- **Security** — the API validates the 9-sensor ranges, checks the API key, rate-limits, and audits
  every request (out-of-range values return HTTP 400).

## LLM (OpenRouter) — optional
- Set `OPENROUTER_API_KEY` (in `.env`) to enable the generated recommendation. The backend reports
  `llm_enabled` at `/health`. Without a key it falls back to the retrieved procedure text, so the
  app still works fully.
- The `/predict` `latency_ms` measures **model inference only** (the real-time path); the LLM call
  is a separate enrichment.

## Docker (one command, both services)
```bash
docker compose up --build          # web: http://localhost:5173 · API: http://localhost:8000/docs
# to enable the LLM inside the container, pass the key and add an env block (see RUN.md):
# OPENROUTER_API_KEY=sk-or-... docker compose up --build
```

## How it maps to the architecture
- The **React app** is the non-ML presentation layer.
- The **FastAPI backend** is the real-time inference + batch scoring microservice.
- It reuses `src/security_layer.py` (Security Gateway) and `src/maintenance_advisor.py`
  (RAG Retriever + LLM) — the same components in the architecture diagram.

## Production build (optional)
```bash
cd frontend && npm run build      # outputs frontend/dist/
```
