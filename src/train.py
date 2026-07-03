import os
import yaml
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.xgboost

class PipelineFilter:
    """Abstract Base Class defining our Pipe-and-Filter data streaming interface."""
    def set_next(self, next_filter):
        self.next_filter = next_filter
        return next_filter

    def process(self, data):
        raise NotImplementedError

class DataIngestionFilter(PipelineFilter):
    def process(self, data):
        print("[Filter 1/3]: Executing Data Ingestion and DVC version hashing setup...")
        os.makedirs("data", exist_ok=True)
        np.random.seed(data["config"]["model"]["seed"])
        n_samples = 2500
        
        synthetic_dataset = {
            "customer_age": np.random.randint(18, 75, n_samples),
            "policy_deductible": np.random.choice([500, 1000, 2000], n_samples),
            "claim_amount": np.random.randint(5000, 150000, n_samples),
            "past_claims_count": np.random.randint(0, 5, n_samples),
            "incident_hour": np.random.randint(0, 24, n_samples),
            "fraud_reported": np.random.choice([0, 1], n_samples, p=[0.88, 0.12])
        }
        df = pd.DataFrame(synthetic_dataset)
        # Introduce a high-correlation anomaly for XGBoost to extract
        df.loc[(df["incident_hour"] < 4) & (df["claim_amount"] > 110000), "fraud_reported"] = 1
        
        csv_path = "data/insurance_claims.csv"
        df.to_csv(csv_path, index=False)
        
        data["dataframe"] = df
        return self.next_filter.process(data)

class DataPreprocessingFilter(PipelineFilter):
    def process(self, data):
        print("[Filter 2/3]: Partitioning matrices into training vectors...")
        df = data["dataframe"]
        X = df.drop(columns=["fraud_reported"])
        y = df["fraud_reported"]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=data["config"]["model"]["seed"]
        )
        
        data["splits"] = {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}
        return self.next_filter.process(data)

class ModelTrainerFilter(PipelineFilter):
    def process(self, data):
        print("[Filter 3/3]: Launching training runs under active MLflow monitoring...")
        cfg = data["config"]
        splits = data["splits"]
        
        mlflow.set_tracking_uri(cfg["paths"]["mlflow_uri"])
        mlflow.set_experiment("mediclaim_ai_insurance_agent_pipeline")
        
        with mlflow.start_run():
            mlflow.xgboost.autolog()
            
            dtrain = xgb.DMatrix(splits["X_train"], label=splits["y_train"])
            params = {
                "max_depth": cfg["model"]["max_depth"],
                "eta": cfg["model"]["eta"],
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "seed": cfg["model"]["seed"]
            }
            
            bst = xgb.train(params, dtrain, num_boost_round=cfg["model"]["num_boost_round"])
            
            os.makedirs(os.path.dirname(cfg["paths"]["model_output"]), exist_ok=True)
            bst.save_model(cfg["paths"]["model_output"])
            print(f"Pipeline complete. Compiled artifact registered at: {cfg['paths']['model_output']}")
        return data

def run_pipe_and_filter_training():
    with open("config/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
        
    # Instantiate pipeline architecture nodes
    ingest = DataIngestionFilter()
    preprocess = DataPreprocessingFilter()
    train = ModelTrainerFilter()
    
    # Chain filters together
    ingest.set_next(preprocess).set_next(train)
    
    # Execute stream processing pipeline
    pipeline_payload = {"config": cfg}
    ingest.process(pipeline_payload)

if __name__ == "__main__":
    run_pipe_and_filter_training()