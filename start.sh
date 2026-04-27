#!/bin/bash
set -e

echo "========================================="
echo " TRIBEv2 Inference Pipeline Setup"
echo "========================================="

# --- Step 1: System Dependencies ---
echo ""
echo "[1/5] Installing system dependencies (ffmpeg, redis)..."
apt-get update -qq
apt-get install -y -qq ffmpeg redis-server git

# --- Step 2: Install Python packages ---
echo ""
echo "[2/5] Installing Python packages (uv for speed)..."
pip install -q uv

# Install lightweight web packages first (fast)
uv pip install --system fastapi==0.110.0 uvicorn==0.27.1 python-multipart==0.0.9 celery==5.3.6 redis==5.0.2 SQLAlchemy==2.0.27

# Install tribev2 (this is the large one - takes a few minutes)
echo ""
echo "      Installing TRIBEv2 and dependencies (this may take a few minutes)..."
uv pip install --system "tribev2[plotting] @ git+https://github.com/facebookresearch/tribev2.git"

# --- Step 3: Start Redis ---
echo ""
echo "[3/5] Starting Redis server..."
service redis-server start

# --- Step 4: Create jobs directory ---
echo ""
echo "[4/5] Creating jobs directory..."
mkdir -p jobs

# --- Step 5: Start services ---
echo ""
echo "[5/5] Starting background services..."

# Stop any existing processes to avoid port conflicts
pkill -f "uvicorn app.main:app" || true
pkill -f "celery -A app.worker.celery_app" || true

sleep 1

# Start FastAPI server in background
nohup uvicorn app.main:app --host 0.0.0.0 --port 8000 > api.log 2>&1 &
echo "      FastAPI started (PID $!)"

# Start Celery Worker in background (concurrency 1 for GPU safety)
nohup celery -A app.worker.celery_app worker --loglevel=info --concurrency=1 > worker.log 2>&1 &
echo "      Celery Worker started (PID $!)"

# Start Celery Beat for scheduled cleanup
nohup celery -A app.worker.celery_app beat --loglevel=info > beat.log 2>&1 &
echo "      Celery Beat started (PID $!)"

# Wait a moment and verify API is up
sleep 3
if curl -s http://localhost:8000/docs > /dev/null; then
    echo ""
    echo "========================================="
    echo " All services are running!"
    echo " API Docs: http://localhost:8000/docs"
    echo " Logs: api.log | worker.log | beat.log"
    echo "========================================="
else
    echo ""
    echo "[ERROR] API failed to start. Check api.log:"
    tail -20 api.log
fi
