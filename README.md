# TRIBEv2 Inference API

A production-ready, asynchronous REST API that runs the **TRIBEv2** multimodal brain encoding model on a GPU-accelerated RunPod instance. Upload an ad video, receive a `job_id` immediately, poll for completion, and download the raw brain activity predictions as a `.npz` file.

---

## What is TRIBEv2?

[TRIBEv2](https://github.com/facebookresearch/tribev2) (TRImodal Brain Encoder v2) is a deep learning model developed by Meta (Facebook Research) that simulates fMRI brain responses to video and audio stimuli. It combines:

- **LLaMA 3.2** — Language/text understanding
- **V-JEPA2** — Visual scene understanding
- **Wav2Vec-BERT** — Audio and speech processing

Given a video file, the model outputs a second-by-second matrix of shape `(n_seconds, 20484)` representing predicted brain activation across the cortical surface. This can be used to measure neural engagement, attention, and emotional response to video content.

---

## Architecture

```
Client (EC2 / Browser / Postman)
        │
        │  HTTP Requests
        ▼
┌───────────────────┐
│   FastAPI (8000)  │  — Job management & file serving
└────────┬──────────┘
         │ Dispatch Task
         ▼
┌───────────────────┐
│   Celery Worker   │  — Runs TRIBEv2 model on GPU (1 video at a time)
│   (--pool=solo)   │
└────────┬──────────┘
         │
    ┌────┴─────┐
    │          │
┌───┴──┐  ┌───┴──────┐
│Redis │  │  SQLite  │  — Job queue / Status tracking
└──────┘  └──────────┘
```

**Key design decisions:**
- `--pool=solo` prevents CUDA re-initialization errors in forked subprocesses
- The model is loaded into GPU memory once on first request and stays resident
- Jobs and their files are automatically deleted after 6 hours

---

## Deploying on RunPod

### Step 1: Create a New Pod

In the **RunPod Console**, click **Deploy** and configure:

| Setting | Recommended Value |
|---|---|
| **GPU** | A40 (48 GB VRAM) or RTX 4090 |
| **Container Image** | `runpod/pytorch:2.2.1-py3.11-cuda12.1.1-devel-ubuntu22.04` |
| **Container Disk** | 20 GB |
| **Volume Disk** | 50 GB (required for model weights cache) |
| **Expose HTTP Ports** | `8000, 8888` |

### Step 2: Set Environment Variables

> **CRITICAL:** You must set the `HF_TOKEN` environment variable before deploying. Without it, the worker cannot download the TRIBEv2 model weights from HuggingFace.

In the **Environment Variables** section of the pod configuration, add:

| Variable | Value |
|---|---|
| `HF_TOKEN` | Your HuggingFace access token (get it from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)) |

> You must also have been **granted access** to the [facebook/tribev2](https://huggingface.co/facebook/tribev2) model on HuggingFace. If you haven't, request access at the model page first.

### Step 3: Set the Container Start Command

In the **Container Start Command** field, paste the following:

```bash
bash -c "if [ -d /workspace/Tribev2_Inference_Setup ]; then cd /workspace/Tribev2_Inference_Setup && git pull; else cd /workspace && git clone https://github.com/vaisakhpv033/Tribev2_Inference_Setup.git; fi && cd /workspace/Tribev2_Inference_Setup && bash start.sh && sleep infinity"
```

This command will:
1. Clone the repository (or pull the latest changes if already present)
2. Run `start.sh` to install dependencies and start all services
3. Keep the container alive with `sleep infinity`

### Step 4: Deploy and Wait

Click **Deploy**. The first deployment will take **5-10 minutes** to:
- Install system packages (`ffmpeg`, `redis-server`)
- Install Python dependencies via `uv`
- Download and cache the TRIBEv2 model weights (~1 GB)

Subsequent restarts are much faster since dependencies are cached on the volume.

---

## Verifying the Deployment

Once the pod is running, find your public API URL. In the RunPod Console:
- Your pod ID is shown in the URL or pod list (e.g., `yiv0wf1ie046jg`)
- Your API is available at: `https://<pod-id>-8000.proxy.runpod.net`

### Check the API is Running

Open your browser and navigate to:

```
https://<your-pod-id>-8000.proxy.runpod.net/docs
```

You should see the **Swagger UI** — an interactive documentation page where you can test all endpoints directly in the browser.

![Swagger UI](https://fastapi.tiangolo.com/img/index/index-01-swagger-ui-simple.png)

If the page loads, your API is live and ready.

---

## API Endpoints

### `POST /api/v1/jobs/analyze`

Upload a video file for brain activity analysis.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: `video` — the `.mp4` video file to analyze

**Example (curl):**
```bash
curl -X POST https://<your-pod-id>-8000.proxy.runpod.net/api/v1/jobs/analyze \
  -F "video=@/path/to/your/ad_video.mp4"
```

**Response:**
```json
{
  "job_id": "c9b740a1-2d3e-4f5a-b6c7-d8e9f0a1b2c3",
  "status": "PENDING"
}
```

> Save the `job_id` — you will need it to check status and download results.

---

### `GET /api/v1/jobs/{job_id}/status`

Poll this endpoint to check the current status of a job.

**Request:**
- Method: `GET`
- URL Parameter: `job_id` — the ID returned from the analyze endpoint

**Example (curl):**
```bash
curl https://<your-pod-id>-8000.proxy.runpod.net/api/v1/jobs/c9b740a1-2d3e-4f5a-b6c7-d8e9f0a1b2c3/status
```

**Response:**
```json
{
  "job_id": "c9b740a1-2d3e-4f5a-b6c7-d8e9f0a1b2c3",
  "status": "COMPLETED",
  "created_at": "2026-04-27T10:00:00",
  "updated_at": "2026-04-27T10:22:15",
  "error_message": null
}
```

**Possible Status Values:**

| Status | Meaning |
|---|---|
| `PENDING` | Job submitted, waiting in the queue |
| `STARTED` | Model is actively processing the video |
| `COMPLETED` | Inference finished, result is ready to download |
| `FAILED` | An error occurred during processing (see `error_message`) |
| `DELETED` | Job files were cleaned up (older than 6 hours) |

> The model typically takes **15-25 minutes** to process a video. Poll this endpoint every 30-60 seconds.

---

### `GET /api/v1/jobs/{job_id}/result`

Download the brain activity prediction file once the job is `COMPLETED`.

**Request:**
- Method: `GET`
- URL Parameter: `job_id`

**Example (curl):**
```bash
curl -O https://<your-pod-id>-8000.proxy.runpod.net/api/v1/jobs/c9b740a1-2d3e-4f5a-b6c7-d8e9f0a1b2c3/result
```

**Response:** A binary `.npz` file download named `tribe_predictions_<job_id>.npz`

**Error Responses:**
- `400` — Job is not yet completed (still `PENDING` or `STARTED`)
- `410 Gone` — Job files have been automatically deleted after 6 hours
- `404` — Job ID not found

---

## Understanding the Output

The downloaded `.npz` file contains two arrays:

```python
import numpy as np

data = np.load("tribe_predictions_<job_id>.npz", allow_pickle=True)

preds    = data['preds']    # shape: (n_seconds, 20484)
segments = data['segments'] # list of time segment objects
```

| Field | Shape | Description |
|---|---|---|
| `preds` | `(n_seconds, 20484)` | Predicted fMRI BOLD signal for each second of the video across 20,484 cortical surface vertices |
| `segments` | `list` | Time segment metadata (start time, duration, trigger index) |

Use the **Destrieux Brain Atlas** via `nilearn` to map the raw vertex values to named brain regions (Visual Cortex, Auditory Cortex, Orbitofrontal/Reward regions, etc.).

---

## Monitoring Logs

SSH into your pod and check the log files for debugging:

```bash
# FastAPI web server logs
tail -f api.log

# Celery worker logs (model inference progress)
tail -f worker.log

# Celery beat scheduler logs (cleanup tasks)
tail -f beat.log

# Startup log (runs on pod restart)
tail -f /workspace/startup.log
```

---

## Automatic Cleanup

Jobs older than **6 hours** are automatically cleaned up by a scheduled background task:
- The video file and `.npz` result file are **deleted from disk** to free storage
- The job record is **kept in the database** with status `DELETED` for historical tracking
- Attempting to download the result of a deleted job returns a `410 Gone` response

---

## Development / Local Testing

The API and Celery worker can be run without a GPU for development purposes. The `tribev2` import failure is handled gracefully — the API will start normally, and jobs will fail with a descriptive error message if the model is not available.

---

## Tech Stack

| Component | Technology |
|---|---|
| Web Framework | FastAPI |
| Background Tasks | Celery |
| Message Broker | Redis |
| Database | SQLite (via SQLAlchemy) |
| AI Model | TRIBEv2 (Meta / Facebook Research) |
| GPU Runtime | PyTorch + CUDA |
| Package Management | `uv` (fast pip alternative) |
