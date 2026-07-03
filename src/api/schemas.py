from pydantic import BaseModel, Field, field_validator

class ClaimInput(BaseModel):
    policy_holder_id: str
    customer_age: int
    policy_deductible: int
    claim_amount: float
    past_claims_count: int
    incident_hour: int

class AdjudicationOutput(BaseModel):
    transaction_id: str
    fraud_risk_score: float
    status: str
    approved_room_rent_allocation: float
    disclaimer: str

    @field_validator("approved_room_rent_allocation")
    @classmethod
    def enforce_cap_rules(cls, value: float) -> float:
        if value > 5000.0:
            raise ValueError("CRITICAL OUT-OF-BOUNDS ERROR: Room rent allocation breaches the statutory ₹5,000 ceiling.")
        return value