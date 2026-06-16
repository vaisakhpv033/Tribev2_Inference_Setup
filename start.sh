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

echo "      Installing completely locked environment from requirements-frozen.txt..."
uv pip install --system -r requirements-frozen.txt --extra-index-url https://download.pytorch.org/whl/cu124 --index-strategy unsafe-best-match

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
nohup celery -A app.worker.celery_app worker --loglevel=info --pool=solo > worker.log 2>&1 &
echo "      Celery Worker started (PID $!)"

# Start Celery Beat for scheduled cleanup
nohup celery -A app.worker.celery_app beat --loglevel=info > beat.log 2>&1 &
echo "      Celery Beat started (PID $!)"

# Wait a moment and verify API is up
sleep 20
if (echo > /dev/tcp/localhost/8000) 2>/dev/null; then
    echo ""
    echo "========================================="
    echo " All services are running!"
    echo " API Docs: http://0.0.0.0:8000/docs"
    echo " Expose port 8000 via RunPod to access externally."
    echo " Logs: api.log | worker.log | beat.log"
    echo "========================================="
else
    echo ""
    echo "[ERROR] API failed to start. Check api.log:"
    tail -20 api.log
fi
