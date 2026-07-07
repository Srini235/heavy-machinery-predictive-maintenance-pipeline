#!/bin/bash

set -e

echo "=========================================="
echo "Heavy Machinery Predictive Maintenance"
echo "=========================================="

echo "[1/3] Stopping existing containers..."
docker compose down

echo "[2/3] Building and starting application..."
docker compose up -d --build

echo "[3/3] Running containers:"
docker compose ps

echo ""
echo "Application started successfully."
echo "Frontend : http://localhost:5173"
echo "Backend  : http://localhost:8000"