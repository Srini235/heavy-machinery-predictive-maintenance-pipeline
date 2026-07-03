# Cell 1: Install Core MLOps and API Packaging Dependencies
!pip install xgboost pandas scikit-learn fastapi uvicorn mlflow PyYAML

# Cell 2: Trigger the complete Pipe-and-Filter Training Pipeline Stage
from src.train import run_pipe_and_filter_training
run_pipe_and_filter_training()

# Cell 3: Boot the Microservice Application Interface directly inside the notebook environment
import subprocess
print("Starting the mediclaim-ai-insurance-agent application listener context...")
!python -m src.main