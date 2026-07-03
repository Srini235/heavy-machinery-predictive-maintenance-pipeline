import uvicorn
from fastapi import FastAPI
from config.settings import settings
from src.core.orchestrator import router as saga_router

app = FastAPI(
    title="Mediclaim-AI Ingress Core",
    description="Central Saga Orchestrator & Ingress Gateway Service Node"
)

# Attach the orchestrated Saga endpoint routes
app.include_router(saga_router)

@app.get("/health")
def system_health_check():
    """Basic cluster infrastructure live ping response."""
    return {
        "status": "HEALTHY",
        "environment": settings.APP_ENV,
        "telemetry_registry": "SQLITE_MLFLOW"
    }

if __name__ == "__main__":
    print("Booting Central Ingress Gateway Service on 127.0.0.1:8000...")
    uvicorn.run("src.main:app", host="127.0.0.1", port=8000, reload=True)