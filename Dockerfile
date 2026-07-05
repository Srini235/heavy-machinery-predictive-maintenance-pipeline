# Backend image — FastAPI serving the predictive-maintenance model.
# Group 105 · Predictive Maintenance of Mobile Hydraulics
FROM python:3.11-slim

WORKDIR /app

# system libs needed by scikit-learn / xgboost runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# install python deps first (better layer caching)
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# copy the application code + dataset + RAG knowledge base
COPY src/ ./src/
COPY data/ ./data/
COPY api_server.py train_and_save.py ./

# train + persist the model at build time so the API starts instantly
RUN python train_and_save.py

EXPOSE 8000

CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
