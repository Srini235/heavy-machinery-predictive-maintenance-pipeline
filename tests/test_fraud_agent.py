import os
import pytest
import xgboost as xgb
from src.agents.fraud_predictor import FraudPredictorAgent

def test_fraud_predictor_inference_boundaries():
    """Verifies the XGBoost predictor is accurately loading weights and reading matrices."""
    agent = FraudPredictorAgent(model_path="models/fraud_xgboost.model")
    
    # Skip test execution gracefully if the user hasn't run train.py pipeline block yet
    if not os.path.exists("models/fraud_xgboost.model"):
        pytest.skip("Model binary weights file not built yet. Skipping safety checks.")
        
    # Evaluate a baseline benign normal request profile
    benign_score = agent.evaluate_claim_risk(age=35, deductible=1000, amount=12000.0, claims_count=0, hour=14)
    assert isinstance(benign_score, float)
    assert 0.0 <= benign_score <= 1.0
    
    # Evaluate an anomalous high-risk claim profile matching our synthetic injection rules
    fraudulent_score = agent.evaluate_claim_risk(age=22, deductible=500, amount=145000.0, claims_count=4, hour=2)
    assert fraudulent_score > benign_score