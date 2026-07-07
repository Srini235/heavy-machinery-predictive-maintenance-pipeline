"""
maintenance_advisor.py — RAG "Maintenance Advisor" for the
Predictive-Maintenance-for-Mobile-Hydraulics system.

Author: Aman Kushwah (2024AC05064) — Group 105

When the predictive model flags a machine/component, this module retrieves the most
relevant repair procedure from the maintenance-manual knowledge base and returns it to
the technician. This is the *retrieval* half of a Retrieval-Augmented Generation (RAG)
workflow (retrieve -> ground -> advise).

Vector store note:
    In production this would be backed by **ChromaDB** (a vector database) with sentence
    embeddings. For this self-contained prototype we use a lightweight **TF-IDF + cosine
    similarity** retriever (scikit-learn) so it runs on any laptop with zero heavy
    dependencies or model downloads, while demonstrating the identical retrieve-by-similarity
    pattern. Swapping in ChromaDB later only changes the Retriever class, not the interface.
"""
from __future__ import annotations

import os
import re
import json
import urllib.request
from dataclasses import dataclass
from typing import List, Optional, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent.parent

DEFAULT_KB_PATH = str(ROOT / "data" / "hydraulic_maintenance_manual.md")

# OpenRouter (https://openrouter.ai) — optional LLM generation step of the RAG.
# If OPENROUTER_API_KEY is set, the retrieved procedure is passed to an LLM which
# writes a tailored recommendation (retrieve -> ground -> generate). Without a key,
# the system falls back to the retrieved text so it always runs offline.
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
# Primary model (overridable via env) + fallbacks tried in order when a free model
# is temporarily rate-limited (429) or unavailable.
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemma-4-26b-a4b-it:free")
FALLBACK_MODELS = [
    "google/gemma-4-26b-a4b-it:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "google/gemma-4-31b-it:free",
    "liquid/lfm-2.5-1.2b-instruct:free",
]


def _post_openrouter(api_key: str, model: str, prompt: str, timeout: int):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
    }).encode("utf-8")
    req = urllib.request.Request(
        OPENROUTER_URL, data=body,
        headers={"Authorization": f"Bearer {api_key}",
                 "Content-Type": "application/json",
                 "HTTP-Referer": "https://github.com/Srini235/SE4ML_Assignment_01_Group105",
                 "X-Title": "Hydraulics Predictive Maintenance (Group 105)"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.load(resp)
    return data["choices"][0]["message"]["content"].strip()


def generate_with_llm(component: str, procedure_title: str, procedure_text: str,
                      timeout: int = 40) -> Optional[str]:
    """Ground an LLM (via OpenRouter) on the retrieved procedure and return a concise
    technician recommendation. Returns None if no API key is set or all models fail
    (caller then falls back to the retrieved procedure text).
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return None
    prompt = (
        "You are a hydraulic maintenance assistant for off-highway machines. "
        f"The predictive model flagged the component '{component.replace('_', ' ')}'. "
        "Using ONLY the maintenance procedure below as grounding, write a short (3-4 sentence), "
        "actionable recommendation for the field technician. Do not invent part numbers.\n\n"
        f"Procedure - {procedure_title}:\n{procedure_text}"
    )
    # try the primary model first, then the fallbacks (dedup, preserve order)
    tried = []
    for model in [OPENROUTER_MODEL] + FALLBACK_MODELS:
        if model in tried:
            continue
        tried.append(model)
        try:
            text = _post_openrouter(api_key, model, prompt, timeout)
            if text and len(text) > 20:          # guard against safety-classifier junk
                return text
        except Exception:
            continue
    return None


@dataclass
class Document:
    """One retrievable maintenance procedure."""
    title: str
    text: str


def load_knowledge_base(path: str = DEFAULT_KB_PATH) -> List[Document]:
    """Parse the markdown manual into one Document per '## ' section."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"knowledge base not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    docs: List[Document] = []
    # split on level-2 headings; keep the heading as the title
    for block in re.split(r"\n##\s+", raw):
        block = block.strip()
        if not block or block.startswith("#"):
            continue
        lines = block.splitlines()
        title = lines[0].strip()
        body = " ".join(l.strip() for l in lines[1:] if l.strip())
        if body:
            docs.append(Document(title=title, text=f"{title}. {body}"))
    return docs


class TfidfRetriever:
    """Lightweight vector retriever (TF-IDF + cosine similarity).

    Interface mirrors a ChromaDB collection: `add` then `retrieve`.
    """

    def __init__(self):
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._docs: List[Document] = []
        self._matrix = None

    def add(self, docs: List[Document]) -> "TfidfRetriever":
        self._docs = list(docs)
        self._matrix = self._vectorizer.fit_transform([d.text for d in self._docs])
        return self

    def retrieve(self, query: str, k: int = 1) -> List[Tuple[Document, float]]:
        """Return the top-k (document, similarity_score) for the query."""
        if self._matrix is None:
            raise RuntimeError("retriever is empty; call add() first")
        q_vec = self._vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self._matrix)[0]
        ranked = sorted(zip(self._docs, sims), key=lambda x: x[1], reverse=True)
        return ranked[:k]


# Map each model target/flag to a natural-language query for retrieval.
COMPONENT_QUERIES = {
    "cooler_condition":     "cooler efficiency drop high oil temperature thermal load",
    "valve_condition":      "directional valve sticking switching fault pressure fluctuation",
    "pump_leakage":         "internal pump leakage falling flow rate volumetric efficiency",
    "accumulator_pressure": "accumulator pre-charge pressure loss pump cycling",
    "vibration":            "high vibration cavitation loose mounting bearing wear",
    "oil_debris":           "oil contamination particle count debris filter replacement",
}


class MaintenanceAdvisor:
    """Ties the model's verdict to a concrete, retrieved repair procedure."""

    def __init__(self, kb_path: str = DEFAULT_KB_PATH):
        self.retriever = TfidfRetriever().add(load_knowledge_base(kb_path))

    def advise(self, flagged_component: str, extra_symptoms: str = "", k: int = 1,
               use_llm: bool = True):
        """Retrieve the top-k procedures for a flagged component/symptom set.

        If use_llm is True and OPENROUTER_API_KEY is set, the top procedure is passed to
        an LLM (via OpenRouter) which writes a tailored recommendation; the result is added
        as `llm_recommendation`. Otherwise the retrieved text is the guidance (offline).
        """
        base_query = COMPONENT_QUERIES.get(
            flagged_component, flagged_component.replace("_", " "))
        query = f"{base_query} {extra_symptoms}".strip()
        hits = self.retriever.retrieve(query, k=k)
        results = []
        for rank, (doc, score) in enumerate(hits):
            item = {"procedure": doc.title,
                    "relevance": round(float(score), 3),
                    "guidance": doc.text}
            if use_llm and rank == 0:                       # generate only for the top hit
                llm = generate_with_llm(flagged_component, doc.title, doc.text)
                if llm:
                    item["llm_recommendation"] = llm
            results.append(item)
        return results
