import os
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Mediclaim-AI: Fraud Classification Node")
MODEL_PATH = "models/fraud_xgboost.model"

class FraudInput(BaseModel):
    customer_age: int
    policy_deductible: int
    claim_amount: float
    past_claims_count: int
    incident_hour: int

class FraudOutput(BaseModel):
    fraud_probability: float
    suspect_anomaly: bool

@app.post("/api/v1/evaluate-risk", response_model=FraudOutput)
async def evaluate_risk(payload: FraudInput):
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=500, detail="XGBoost model file missing.")
    
    bst = xgb.Booster()
    bst.load_model(MODEL_PATH)
    
    features = [[payload.customer_age, payload.policy_deductible, payload.claim_amount, payload.past_claims_count, payload.incident_hour]]
    dmatrix_row = xgb.DMatrix(features)
    prob = float(bst.predict(dmatrix_row)[0])
    
    return FraudOutput(fraud_probability=prob, suspect_anomaly=prob >= 0.65)

if __name__ == "__main__":
    import uvicorn
    print("Launching Isolated Fraud Microservice Node on Port 8001...")
    uvicorn.run("src.agents.fraud_predictor:app", host="127.0.0.1", port=8001, reload=True)