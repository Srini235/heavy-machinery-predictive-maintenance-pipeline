"""
merge_notebook.py — build the final 105.ipynb by combining:
  * the team's master notebook (premium GR4ML content, graphviz diagrams,
    multi-output model, registry, batch/real-time serving, drift) — outputs preserved
  * Group 105 details filled in
  * appended sections authored by Aman: Security layer, RAG advisor,
    architectural-pattern clarification, and the deployed React/FastAPI web app.

It also exports the master's synthetic dataset to data/hydraulic_fleet_telemetry.csv.

Run:  python3 merge_notebook.py
"""
import io
import os
import base64
import contextlib
import nbformat as nbf

REPO = os.path.dirname(os.path.abspath(__file__))
# The master notebook lives one level up (outside the repo). 105.ipynb / 105.html are already
# committed, so you only need to re-run this if the master notebook or the appended sections change.
# Override the location with the MASTER_NOTEBOOK env var if it sits elsewhere.
MASTER = os.getenv("MASTER_NOTEBOOK", os.path.join(
    os.path.dirname(REPO), "GroupXX_AssignmentI_MobileHydraulics_PredictiveMaintenance 1.ipynb"))
if not os.path.exists(MASTER):
    raise SystemExit(f"Master notebook not found at {MASTER}\n"
                     "105.ipynb is already committed — you usually don't need to re-run this. "
                     "If you do, set MASTER_NOTEBOOK to the master notebook path.")

nb = nbf.read(MASTER, as_version=4)

# --------------------------------------------------------------------------- #
# 1. Export the master's synthetic dataset to CSV (run its data-gen cells)
# --------------------------------------------------------------------------- #
def export_dataset():
    ns = {}
    # cell 15 (imports/setup) + cell 16 (data generation) build `df`
    for idx in (15, 16):
        src = "".join(nb.cells[idx].source)
        with contextlib.redirect_stdout(io.StringIO()):
            exec(src, ns)
    df = ns["df"]
    os.makedirs(os.path.join(REPO, "data"), exist_ok=True)
    out = os.path.join(REPO, "data", "hydraulic_fleet_telemetry.csv")
    df.to_csv(out, index=False)
    print(f"Dataset exported: {out}  ({df.shape[0]} rows x {df.shape[1]} cols)")

export_dataset()

# --------------------------------------------------------------------------- #
# 2. Fill Group 105 details (replace the placeholder title cell)
# --------------------------------------------------------------------------- #
TITLE = """# AIMLCZG546 — Software Engineering for Machine Learning
# Assignment I: Predictive Maintenance of Mobile Hydraulic Systems

**Domain:** Mobile Hydraulics / Automotive / Embedded Systems — Off-highway construction & mining equipment (excavators, telehandlers, backhoe loaders)

**Application:** IoT-based Predictive Maintenance & Fault Diagnosis for Mobile Hydraulic Systems, modeled end-to-end using **GR4ML** and implemented as a working ML system.

---
### Group Details

**Group No.: 105**

| Sl. No | BITS ID | Name | Contribution (Qualitative) | Contribution (%) |
|---|---|---|---|---|
| 1 | 2024AC05744 | Srinivasan R | System architecture design, Model Registry pattern, Saga/orchestration prototype | 25 |
| 2 | 2024AC05100 | Vineet Kumar | Data pipeline, Docker/infrastructure, CI | 25 |
| 3 | 2024AD05482 | Vibhav Sharma | GR4ML requirements engineering (Business / Analytics / Data Prep views), quality-attribute analysis | 25 |
| 4 | 2024AC05064 | Aman Kushwah | Model training & evaluation, Security layer, RAG maintenance advisor, Batch/Real-time serving, React + FastAPI web app, tests, documentation | 25 |
"""
nb.cells[0] = nbf.v4.new_markdown_cell(TITLE)

# --------------------------------------------------------------------------- #
# 3. Build the appended sections (authored by Aman)
# --------------------------------------------------------------------------- #
def run_capture(src_code):
    """Execute self-contained demo code, return captured stdout text."""
    buf = io.StringIO()
    ns = {}
    with contextlib.redirect_stdout(buf):
        exec(src_code, ns)
    return buf.getvalue()


def code_with_stream(src_code, capture=True):
    cell = nbf.v4.new_code_cell(src_code)
    if capture:
        text = run_capture(src_code)
        cell.outputs = [nbf.v4.new_output("stream", name="stdout", text=text)]
        cell.execution_count = 1
    return cell


def image_cell(src_code, image_path, caption_alt="figure"):
    cell = nbf.v4.new_code_cell(src_code)
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    cell.outputs = [nbf.v4.new_output(
        "execute_result",
        data={"image/png": b64, "text/plain": [f"<{caption_alt}>"]},
        execution_count=1, metadata={})]
    cell.execution_count = 1
    return cell


new_cells = []

# ---- Security ----
new_cells.append(nbf.v4.new_markdown_cell(
"""# 14. Application-Wide Security Controls

Beyond model quality, a fleet-facing predictive-maintenance service must be **secure**. A single
reusable security layer (`src/security_layer.py`) is imported by both this notebook and the
serving API, so the whole application is protected consistently. Five controls are implemented
and demonstrated below.

| # | Control | Threat mitigated |
|:--:|:--|:--|
| 1 | Input validation (range checks) | bad-data / adversarial injection |
| 2 | API-key authentication | unauthorized access |
| 3 | Model integrity (SHA-256) | model tampering / supply-chain |
| 4 | Audit logging (hash chain) | non-repudiation, tamper-evidence |
| 5 | Rate limiting (sliding window) | abuse / denial-of-service |
"""))

SECURITY_DEMO = r'''import sys, os, joblib
sys.path.insert(0, ".")
from src.security_layer import (validate_sensor_payload, HYDRAULIC_BOUNDS, ApiKeyAuthenticator,
    RateLimiter, AuditTrail, compute_file_sha256, verify_model_integrity, SecurityError)

# a valid per-cycle hydraulic reading (same 9-sensor schema the model + API use)
good = {"operating_hours":900,"pressure_mean_bar":165,"pressure_std_bar":6.5,"flow_mean_lpm":7.2,
        "oil_temp_mean_c":68,"vibration_rms_mms":3.5,"motor_power_kw":18,
        "pump_speed_mean_rpm":1400,"cooling_efficiency_pct":78}
print("1. INPUT VALIDATION")
print("   valid payload accepted:", bool(validate_sensor_payload(good, bounds=HYDRAULIC_BOUNDS)))
for bad, why in [({**good,"pressure_mean_bar":99999}, "out-of-range pressure"),
                 ({**good,"oil_temp_mean_c":float("nan")}, "NaN oil temperature")]:
    try: validate_sensor_payload(bad, bounds=HYDRAULIC_BOUNDS)
    except SecurityError as e: print(f"   BLOCKED {why}: {e}")

print("2. API-KEY AUTHENTICATION")
auth = ApiKeyAuthenticator(["fleet-ops-secret-key"])
print("   valid key accepted:", auth.authenticate("fleet-ops-secret-key"))
try: auth.authenticate("attacker-guess")
except SecurityError as e: print(f"   BLOCKED invalid key: {e}")

print("3. MODEL INTEGRITY (SHA-256)")
os.makedirs("model_registry", exist_ok=True)
p = "model_registry/_sec_demo.joblib"; joblib.dump({"weights": 1}, p)
chk = compute_file_sha256(p); print("   registered SHA-256:", chk[:24], "...")
print("   verify untampered:", verify_model_integrity(p, chk))
open(p, "ab").write(b"\x00")   # tamper
try: verify_model_integrity(p, chk)
except SecurityError as e: print("   BLOCKED tampered model:", str(e)[:60], "...")
os.remove(p)

print("4. AUDIT LOGGING (tamper-evident hash chain)")
a = AuditTrail(); a.record("tech-42","predict","machine=EX-118"); a.record("tech-42","predict","machine=EX-204")
print("   chain valid:", a.verify_chain())
a._entries[0]["detail"] = "machine=HACKED"
print("   tamper detected (chain invalid):", not a.verify_chain())

print("5. RATE LIMITING (max 3 / sec per client)")
rl = RateLimiter(max_calls=3, window_seconds=1.0); allowed = 0
for i in range(5):
    try: rl.check("client", now=100.0+i*0.1); allowed += 1
    except SecurityError: pass
print(f"   {allowed}/5 requests allowed within the window (expected 3)")
print("\nAll five security controls enforced application-wide via src/security_layer.py")
'''
new_cells.append(code_with_stream(SECURITY_DEMO))

# ---- RAG advisor ----
new_cells.append(nbf.v4.new_markdown_cell(
"""# 15. RAG Maintenance Advisor (Retrieval-Augmented Guidance)

A prediction is more actionable when paired with **what to do about it**. When the batch model
flags a component (`cooler_condition`, `valve_condition`, `pump_leakage`, `accumulator_pressure`),
the **Maintenance Advisor** retrieves the matching repair procedure from a maintenance-manual
knowledge base (`data/hydraulic_maintenance_manual.md`) by semantic similarity — the *retrieve*
stage of a Retrieval-Augmented Generation (RAG) workflow.

> **Vector store:** production would use **ChromaDB** (as in the Session-5 lab); this prototype
> uses a lightweight **TF-IDF + cosine similarity** retriever (scikit-learn) so it runs on any
> machine with no heavy dependencies while demonstrating the identical retrieve-by-similarity
> pattern. Swapping in ChromaDB changes only the retriever class, not the interface.
"""))

RAG_DEMO = r'''import sys
sys.path.insert(0, ".")
from src.maintenance_advisor import MaintenanceAdvisor, load_knowledge_base

kb = load_knowledge_base()
print(f"Knowledge base loaded: {len(kb)} maintenance procedures\n")
advisor = MaintenanceAdvisor()

# first component: full RAG (retrieve -> LLM generate); rest: retrieval only (keeps this fast)
for i, component in enumerate(["pump_leakage", "cooler_condition", "valve_condition", "accumulator_pressure"]):
    top = advisor.advise(component, k=1, use_llm=(i == 0))[0]
    print(f"MODEL FLAG: {component}")
    print(f"  -> retrieved procedure : {top['procedure']}  (relevance {top['relevance']})")
    if top.get("llm_recommendation"):
        print(f"  -> LLM recommendation  : {top['llm_recommendation'][:170]}...")
    else:
        print(f"  -> guidance            : {top['guidance'][:110]}...")
    print()
'''
new_cells.append(code_with_stream(RAG_DEMO))

# ---- Architectural patterns clarification ----
new_cells.append(nbf.v4.new_markdown_cell(
"""# 16. Architectural Patterns — Pipe-and-Filter & Microservices

Sections 10–11 implement two **MLOps / design patterns** — *Model Registry* and
*Batch-vs-Real-time Serving*. For Objective 2's requirement of **architectural patterns**, the
system additionally realises two classic architectural patterns:

**Pipe-and-Filter (data pipeline).** The Data Preparation View (Section 4) is implemented as a
chain of independent filters — **Ingest → Clean → Feature-Extract → Normalize** — each doing one
job and passing its output to the next (`sklearn` `Pipeline`/`ColumnTransformer` in Section 8).
This is the same pattern as the course's Session-4 lab.

**Microservices.** Training, batch scoring, real-time inference, and the security gateway are
**independently deployable services** with well-defined contracts (Sections 11–12 + the FastAPI
service). Each can be scaled and updated independently — the real-time API scales on request rate,
the batch scorer on data volume.

| Pattern | Type | Where realised |
|:--|:--|:--|
| **Pipe-and-Filter** | Architectural | Data preparation pipeline (Section 8) |
| **Microservices** | Architectural | Batch / Real-time / Security services (Sections 11–12, web app) |
| **Model Registry** | Design / MLOps | Section 10 |
| **Batch vs Real-time Serving** | Design / MLOps | Section 11 |
"""))

# ---- Web app ----
new_cells.append(nbf.v4.new_markdown_cell(
"""# 17. Deployed Web Application (React + FastAPI + Docker)

The model is served through a working web application so a non-technical user (a field technician)
can assess a machine and get an explained verdict plus repair guidance — this is the *"application
with screenshots"* required by the submission guidelines.

- **Frontend:** React + Vite + shadcn/ui (`frontend/`)
- **Backend:** FastAPI (`api_server.py`) — serves the model behind the security layer + RAG advisor
- **One-command run:** `docker compose up --build`  → web app at `http://localhost:5173`

The technician enters live sensor readings, clicks **Assess Machine**, and receives the verdict,
confidence, top contributing sensors, and the retrieved repair procedure.
"""))

new_cells.append(image_cell(
    'from IPython.display import Image, display\n'
    'display(Image("docs/screenshots/app_prediction.png"))  # live prediction result',
    os.path.join(REPO, "docs/screenshots/app_prediction.png"),
    caption_alt="web app — live prediction"))

# --------------------------------------------------------------------------- #
# 4. Insert the new cells just before the Conclusion (# 13. Conclusion)
# --------------------------------------------------------------------------- #
insert_at = len(nb.cells)
for i, c in enumerate(nb.cells):
    if c.cell_type == "markdown" and "13. Conclusion" in "".join(c.source):
        insert_at = i
        break
nb.cells[insert_at:insert_at] = new_cells

# --------------------------------------------------------------------------- #
# 5. Save 105.ipynb
# --------------------------------------------------------------------------- #
out = os.path.join(REPO, "105.ipynb")
nbf.write(nb, out)
print(f"Wrote {out} with {len(nb.cells)} cells "
      f"(inserted {len(new_cells)} new cells before Conclusion at index {insert_at})")
