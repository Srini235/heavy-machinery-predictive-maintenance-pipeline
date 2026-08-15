"""RAG Maintenance-Advisor tests — retrieval quality, separated from the
predictive-maintenance model tests (QA feedback).

Author: Group 105

Run from the repo root:   pytest -m rag -q
"""

import pytest

from src.maintenance_advisor import (
    Document,
    MaintenanceAdvisor,
    TfidfRetriever,
    load_knowledge_base,
)
from tests._metrics import record_metric

pytestmark = pytest.mark.rag


def test_knowledge_base_loads():
    kb = load_knowledge_base()
    record_metric("rag.knowledge_base_documents", len(kb))
    assert len(kb) >= 5
    assert all(isinstance(d, Document) and d.text for d in kb)


def test_retriever_ranks_relevant_doc_first():
    retriever = TfidfRetriever().add(
        [
            Document("Pump", "internal pump leakage falling flow rate volumetric efficiency"),
            Document("Cooler", "cooler efficiency drop high oil temperature thermal load"),
        ]
    )
    hits = retriever.retrieve("pump leakage low flow", k=1)
    assert hits[0][0].title == "Pump"
    assert hits[0][1] > 0


@pytest.mark.parametrize(
    "component,expected_keyword",
    [
        ("pump_leakage", "pump"),
        ("cooler_condition", "cooler"),
        ("accumulator_pressure", "accumulator"),
    ],
)
def test_advisor_retrieves_matching_procedure(component, expected_keyword):
    advisor = MaintenanceAdvisor()
    result = advisor.advise(component, k=1)[0]
    record_metric(f"rag.retrieval_relevance.{component}", result["relevance"])
    assert expected_keyword in result["procedure"].lower()
    assert 0.0 <= result["relevance"] <= 1.0
