#!/bin/bash
set -e

echo "========================================="
echo " TRIBEv2 Inference Pipeline Setup"
echo "========================================="

# --- Register this script to run on pod restart (via cron @reboot) ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CRON_JOB="@reboot cd $SCRIPT_DIR && bash start.sh >> /workspace/startup.log 2>&1"
( crontab -l 2>/dev/null | grep -qF "start.sh" ) || ( crontab -l 2>/dev/null; echo "$CRON_JOB" ) | crontab -
service cron start > /dev/null 2>&1 || true

# --- Generate .env from RunPod environment variables (if not already present) ---
if [ ! -f .env ]; then
    echo "Generating .env from environment variables..."
    cat > .env <<EOF
DATABASE_URL=sqlite:///./tribev2_jobs.db
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0
HF_TOKEN=${HF_TOKEN:-}
EOF
fi

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
nohup celery -A app.worker.celery_app worker --loglevel=info --pool=solo > worker.log 2>&1 &
echo "      Celery Worker started (PID $!)"

# Start Celery Beat for scheduled cleanup
nohup celery -A app.worker.celery_app beat --loglevel=info > beat.log 2>&1 &
echo "      Celery Beat started (PID $!)"

# Wait a moment and verify API is up
sleep 4
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
