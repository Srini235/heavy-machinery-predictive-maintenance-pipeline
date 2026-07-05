# Contribution Notes — Aman Kushwah (2024AC05064), Group 105

This documents what I built/merged for Assignment I (*Predictive Maintenance of Mobile Hydraulic
Systems*) and how a teammate can run and verify it. Written so anyone can pick it up quickly.

---

## 1. What I did (summary)

- **Merged the team's master notebook into `105.ipynb`** — kept all its premium GR4ML content
  (graphviz Business / Analytics / Data-Prep views), the multi-output model, Model Registry,
  batch/real-time serving and drift monitor, filled in the **Group 105** details, and **appended
  four sections of my own**: Security, RAG advisor, architectural-pattern clarification, and the
  deployed web app.
- **Exported the dataset** to `data/hydraulic_fleet_telemetry.csv` (10,000 rows).
- **Built a working web app** (React + FastAPI) so we have real application screenshots.
- **Added the RAG Maintenance Advisor with an LLM** (OpenRouter) + an offline fallback.
- **Application-wide security layer** + **18 pytest tests** + **Docker** one-command run.

---

## 2. Files I added / changed

| File | What it is |
|:--|:--|
| `105.ipynb` / `105.html` | Final solution notebook (master content + my sections), executed + HTML export |
| `merge_notebook.py` | Builds `105.ipynb` from the master notebook + exports the dataset CSV |
| `data/hydraulic_fleet_telemetry.csv` | The dataset (40 machines × 250 cycles = 10,000 rows) |
| `data/hydraulic_maintenance_manual.md` | Knowledge base for the RAG advisor |
| `src/security_layer.py` | Reusable security controls (used by notebook **and** API) |
| `src/maintenance_advisor.py` | RAG advisor — TF-IDF retrieval + optional OpenRouter LLM generation |
| `train_and_save.py` | Trains the multi-output condition model + stability model, saves to `model_registry/` |
| `api_server.py` | FastAPI backend serving both models + security + RAG/LLM |
| `frontend/` | React + Vite + shadcn/ui web app |
| `tests/test_predictive_maintenance.py` | 18 tests (model, security, RAG) |
| `Dockerfile`, `frontend/Dockerfile`, `docker-compose.yml` | One-command full-stack run |
| `docs/screenshots/` | App + architecture screenshots |
| `.env.example` | Template for the OpenRouter key (`.env` is gitignored) |

---

## 3. The model & dataset (what teammates should know)

**Dataset** — synthetic fleet telemetry modelled on the UCI *Condition Monitoring of Hydraulic
Systems* dataset. Features: `operating_hours`, `pressure_mean_bar`, `pressure_std_bar`,
`flow_mean_lpm`, `oil_temp_mean_c`, `vibration_rms_mms`, `motor_power_kw`, `pump_speed_mean_rpm`,
`cooling_efficiency_pct`, `machine_type`.

**Two models** (matching the two GR4ML PredictionGoals):
- **Multi-output condition model** → `cooler_condition`, `valve_condition`, `pump_leakage`,
  `accumulator_pressure` (batch use case)
- **Stability model** → `stability_flag` (lightweight, real-time use case)

**Metrics:** cooler 94% · valve 89% · pump 94% · accumulator 88% · stability 91%.

---

## 4. Assignment coverage

- **Objective 1 (GR4ML):** formal Business / Analytics Design / Data Preparation views + SMART
  top-3 quality requirements (in `105.ipynb`; also written up in `GR4ML_REPORT.md`).
- **Objective 2 (Architecture):** architecture diagram (`architecture_diagram.png`); **architectural
  patterns** = Pipe-and-Filter + Microservices; **design/MLOps patterns** = Model Registry +
  Batch-vs-Real-time Serving.
- **Quality requirements demonstrated:** Robustness (15% corruption, ~99% retained), Latency
  (stability model < 100 ms), Explainability (feature importances). **Security** added as a 4th.
- **Application with screenshots:** the React + FastAPI web app.

---

## 5. How to run it (teammate quick-start)

**Easiest — Docker (one command):**
```bash
cd mediclaim-ai-insurance-agent
docker compose up --build          # web: http://localhost:5173 · API: http://localhost:8000/docs
```

**Without Docker (two terminals):**
```bash
# Terminal 1 — backend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements-api.txt
cp .env.example .env                # (optional) paste your OpenRouter key for the LLM
set -a; . ./.env; set +a
python3 train_and_save.py
uvicorn api_server:app --port 8000

# Terminal 2 — frontend
cd frontend && npm install && npm run dev
```

**Notebook only:**
```bash
pip install -r requirements-notebook.txt
jupyter notebook 105.ipynb          # or open 105.html
```

Full details and troubleshooting are in **`RUN.md`**.

---

## 6. How to test it

```bash
# 18 automated tests — expect "18 passed"
python3 -m pytest tests/test_predictive_maintenance.py -q

# prove the RAG advisor retrieves the right procedure
python3 -c "from src.maintenance_advisor import MaintenanceAdvisor
print(MaintenanceAdvisor().advise('pump_leakage', k=1)[0]['procedure'])"   # -> Internal pump leakage
```

---

## 7. Notes for whoever continues tomorrow
- **RAG LLM is optional.** With `OPENROUTER_API_KEY` set (in `.env`), the advisor generates a
  tailored recommendation via OpenRouter; without it, it uses the offline TF-IDF retrieval. So the
  app always runs. Say *"ChromaDB in production (Session-5 lab), TF-IDF + OpenRouter here."*
- **Rotate the OpenRouter key** before final submission (it was shared over chat). It lives only in
  the gitignored `.env`; `.env.example` holds a placeholder.
- `merge_notebook.py` regenerates `105.ipynb` from the master notebook — re-run it if the master or
  my sections change. `train_and_save.py` regenerates the models.
- Old insurance code has been removed; the repo now contains only the predictive-maintenance project.
