import pytest
from src.core.orchestrator import SagaOrchestrator
from src.api.schemas import ClaimInput

@pytest.mark.asyncio
async def test_saga_orchestration_compensating_loop():
    """Verifies that the Saga engine executes compensating rollbacks on data anomalies."""
    orchestrator = SagaOrchestrator()
    
    # Formulate an extreme request profile to intentionally force an out-of-bounds rule crash
    invalid_payload = ClaimInput(
        policy_holder_id="POL_9988",
        customer_age=45,
        policy_deductible=1000,
        claim_amount=3000000.0, # Massive value to trigger out-of-bounds crash validation logic
        past_claims_count=1,
        incident_hour=12
    )
    
    response = await orchestrator.execute_claim_lifecycle(invalid_payload)
    
    # Verify that the Saga engine intercepted the Pydantic crash error gracefully 
    # instead of letting the application crash with a 500 error.
    assert response.status_code == "REJECTED_COMPLIANCE_VIOLATION"
    assert response.approved_room_rent_allocation == 0.0
    assert "Saga Rollback Triggered" in response.disclaimer