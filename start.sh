#!/bin/bash

# Update and install Redis if not already installed
if ! command -v redis-server &> /dev/null
then
    echo "Redis not found. Installing Redis..."
    apt-get update
    apt-get install -y redis-server
fi

# Start Redis service
echo "Starting Redis server..."
service redis-server start

# Create jobs directory
mkdir -p jobs

# Stop any existing processes to avoid port conflicts
pkill -f "uvicorn app.main:app"
pkill -f "celery -A app.worker.celery_app"

# Start FastAPI server in background
echo "Starting FastAPI server..."
nohup uvicorn app.main:app --host 0.0.0.0 --port 8000 > api.log 2>&1 &

# Start Celery Worker in background (concurrency 1 for GPU)
echo "Starting Celery Worker..."
nohup celery -A app.worker.celery_app worker --loglevel=info --concurrency=1 > worker.log 2>&1 &

# Start Celery Beat for scheduled cleanup
echo "Starting Celery Beat..."
nohup celery -A app.worker.celery_app beat --loglevel=info > beat.log 2>&1 &

echo "========================================="
echo "All services started successfully!"
echo "API is running at http://localhost:8000/docs"
echo "Check api.log, worker.log, and beat.log for details."
echo "========================================="
