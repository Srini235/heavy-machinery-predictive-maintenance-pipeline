from fastapi import APIRouter
from src.api.schemas import ClaimInput, AdjudicationOutput
from src.core.orchestrator import SagaOrchestrator

router = APIRouter(prefix="/api/v1")
orchestrator = SagaOrchestrator()

@router.post("/adjudicate", response_model=AdjudicationOutput)
async def adjudicate_insurance_claim(claim: ClaimInput):
    """
    Ingestion portal for cashless medical claims. 
    Triggers the orchestrated Saga distributed transaction workflow.
    """
    response = await orchestrator.execute_claim_lifecycle(claim)
    return response