# How to Run — Predictive Maintenance of Mobile Hydraulics (Group 105)

There are two ways to run the full application (React frontend + FastAPI backend):

- **Option A — Docker (one command)** — recommended, nothing to install except Docker.
- **Option B — Run separately (no Docker)** — run backend and frontend in two terminals.

After it's up:
- **Web app:** http://localhost:5173
- **API docs:** http://localhost:8000/docs

---

## Option A — Docker (one command)

**Prerequisite:** Docker Desktop / Docker Engine with Compose v2.

```bash
cd heavy-machinery-predictive-maintenance-pipeline

# build both images and start the whole stack
docker compose up --build
```

That's it. Compose builds two services:
- `backend` (FastAPI) — trains + loads the model, serves the API on port **8000**.
- `frontend` (React, built and served by nginx) on port **5173**.

Open **http://localhost:5173**, pick a preset (Healthy / Degraded) or type sensor values, and
click **Assess Machine**.

**Stop it:**
```bash
docker compose down          # Ctrl+C first if running in the foreground
```

**Rebuild after code changes:**
```bash
docker compose up --build
```

> To enable the LLM recommendation inside Docker, set the key on the host first:
> `OPENROUTER_API_KEY=sk-or-... docker compose up --build` (compose passes it through to the backend).

---

## Option B — Run separately (no Docker)

**Prerequisites:** Python 3.10+ and Node.js 18+.

### Terminal 1 — Backend (port 8000)
```bash
cd heavy-machinery-predictive-maintenance-pipeline

# one-time setup
python3 -m venv venv
source venv/bin/activate                 # Windows: venv\Scripts\activate
pip install -r requirements-api.txt

# (optional) enable the RAG advisor's LLM generation via OpenRouter
cp .env.example .env                      # then paste your OPENROUTER_API_KEY into .env
set -a; . ./.env; set +a                  # load it into the shell (Linux/Mac)

# train + save the models once (creates model_registry/*.joblib from the dataset)
python3 train_and_save.py

# start the API
uvicorn api_server:app --reload --port 8000
```

> **LLM is optional.** With `OPENROUTER_API_KEY` set, the app shows an AI-generated repair
> recommendation (RAG + LLM). Without it, it falls back to the retrieved procedure text — the app
> still works fully. Get a free key at https://openrouter.ai.

### Terminal 2 — Frontend (port 5173)
```bash
cd heavy-machinery-predictive-maintenance-pipeline/frontend

npm install        # one-time
npm run dev
```

Open **http://localhost:5173**.

---

## Also available (analysis / evidence)

- **Notebook** (the full ML solution + patterns + security + RAG, executed):
  ```bash
  pip install -r requirements-notebook.txt
  jupyter notebook notebook/105.ipynb
  ```
- **Tests** (18 tests):
  ```bash
  python3 -m pytest tests/test_predictive_maintenance.py -q
  ```

---

## Ports summary

| Service | URL | Port |
|:--|:--|:--|
| React web app | http://localhost:5173 | 5173 |
| FastAPI backend | http://localhost:8000 | 8000 |
| API interactive docs | http://localhost:8000/docs | 8000 |

## Troubleshooting

- **"Could not reach the API" in the web app** → the backend isn't running or not on port 8000.
  Start it (Option A or B, Terminal 1) and confirm http://localhost:8000/health returns `healthy`.
- **Port already in use** → stop whatever is using 8000/5173, or change the published port in
  `docker-compose.yml` (Docker) / the `--port` flag / Vite config (separate).
- **Docker build slow the first time** → normal; images are cached for subsequent runs.
- **Out-of-range sensor value returns an error** → that's the security layer working as designed
  (input validation rejects physically-impossible readings with HTTP 400).
