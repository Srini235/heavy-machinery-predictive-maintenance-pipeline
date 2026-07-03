import uuid
import httpx
from fastapi import APIRouter, HTTPException
from src.api.schemas import ClaimInput, AdjudicationOutput
from src.core.database import ClaimsLedgerDatabase # Your PostgreSQL layer

router = APIRouter(prefix="/api/v1")
db_command_side = ClaimsLedgerDatabase() # CQRS: Command Side (Writes)

FRAUD_SERVICE_URL = "http://127.0.0.1:8001/api/v1/evaluate-risk"
RAG_SERVICE_URL = "http://127.0.0.1:8002/api/v1/compliance-check"

@router.post("/adjudicate", response_model=AdjudicationOutput)
async def process_claim_saga_workflow(payload: ClaimInput):
    tx_id = str(uuid.uuid4())
    
    # ---------------------------------------------------------------------
    # CQRS COMMAND: Create initial uncommitted ledger state row
    # ---------------------------------------------------------------------
    db_command_side.insert_initial_transaction_state(tx_id, payload.policy_holder_id, payload.claim_amount)
    
    async with httpx.AsyncClient() as client:
        # SAGA STEP 1: Microservice Communication to Fraud Agent Node
        try:
            fraud_payload = {
                "customer_age": payload.customer_age, "policy_deductible": payload.policy_deductible,
                "claim_amount": payload.claim_amount, "past_claims_count": payload.past_claims_count,
                "incident_hour": payload.incident_hour
            }
            f_res = await client.post(FRAUD_SERVICE_URL, json=fraud_payload, timeout=5.0)
            fraud_data = f_res.json()
        except Exception:
            # COMPENSATING ACTION
            db_command_side.execute_compensating_rollback(tx_id, "FRAUD_SERVICE_TIMEOUT")
            raise HTTPException(status_code=503, detail="Saga Aborted: Fraud Node down.")

        fraud_risk = fraud_data["fraud_probability"]
        if fraud_data["suspect_anomaly"]:
            # SAGA EARLY EXIT COMPENSATING STEP
            db_command_side.execute_compensating_rollback(tx_id, "HIGH_FRAUD_RISK")
            return AdjudicationOutput(
                transaction_id=tx_id, fraud_risk_score=fraud_risk, status="REJECTED_AUDIT_FLAGGED",
                approved_room_rent_allocation=0.0, disclaimer="Suspended due to behavioral fraud risk flags."
            )

        # SAGA STEP 2: Microservice Communication to RAG Agent Node
        try:
            rag_payload = {"claim_amount": payload.claim_amount}
            r_res = await client.post(RAG_SERVICE_URL, json=rag_payload, timeout=5.0)
            rag_data = r_res.json()
        except Exception:
            db_command_side.execute_compensating_rollback(tx_id, "RAG_SERVICE_TIMEOUT")
            raise HTTPException(status_code=503, detail="Saga Aborted: RAG Node down.")

        calculated_rent = rag_data["allocated_room_rent"]
        
        # Synthetic simulation to verify Pydantic guardrail failures locally
        if payload.claim_amount > 2500000:
            calculated_rent = 9999.0

        # SAGA STEP 3: CQRS Final Commit or Failsafe Error Rollback
        try:
            output = AdjudicationOutput(
                transaction_id=tx_id, fraud_risk_score=fraud_risk, status="CLAIM_APPROVED_SUCCESSFULLY",
                approved_room_rent_allocation=calculated_rent, disclaimer=rag_data["disclaimer"]
            )
            # CQRS WRITE COMMAND
            db_command_side.commit_approved_transaction(tx_id, calculated_rent)
            return output
        except ValueError as pydantic_err:
            # SAGA COMPENSATING ACTION
            db_command_side.execute_compensating_rollback(tx_id, "COMPLIANCE_CEILING_BREACH")
            return AdjudicationOutput(
                transaction_id=tx_id, fraud_risk_score=fraud_risk, status="REJECTED_COMPLIANCE_VIOLATION",
                approved_room_rent_allocation=0.0, disclaimer=f"Saga Rollback Triggered: {str(pydantic_err)}"
            )