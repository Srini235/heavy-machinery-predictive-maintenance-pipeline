import os
from pydantic_settings import BaseSettings
from pydantic import Field

class AppEnvironmentSettings(BaseSettings):
    """Manages system environment vars, API tokens, and database isolation ports."""
    PROJECT_NAME: str = "mediclaim-ai-insurance-agent"
    APP_ENV: str = Field(default="development", env="APP_ENV")
    
    # Database Connections Parameters
    POSTGRES_HOST: str = Field(default="127.0.0.1", env="POSTGRES_HOST")
    POSTGRES_PORT: int = Field(default=5432, env="POSTGRES_PORT")
    POSTGRES_DB: str = Field(default="mediclaim_ledger", env="POSTGRES_DB")
    
    # Telemetry tracking variables
    MLFLOW_TRACKING_URI: str = Field(default="sqlite:///mlflow.db", env="MLFLOW_TRACKING_URI")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Instantiate clean shared config context node
settings = AppEnvironmentSettings()