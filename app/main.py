import os
import shutil
import httpx
from pathlib import Path
from urllib.parse import urlparse
from pydantic import BaseModel, HttpUrl
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from .database import engine, Base, get_db
from .models import Job
from .config import settings

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="TRIBEv2 Video Processing API")

# Ensure jobs directory exists
Path(settings.JOBS_DIR).mkdir(parents=True, exist_ok=True)

# Import celery task here to avoid circular imports
from .worker import process_video_task

@app.post("/api/v1/jobs/analyze")
async def analyze_video(video: UploadFile = File(...), db: Session = Depends(get_db)):
    # 1. Create Job record
    job = Job(video_filename=video.filename)
    db.add(job)
    db.commit()
    db.refresh(job)

    job_id_str = str(job.id)

    # 2. Setup job directory and save video
    job_dir = Path(settings.JOBS_DIR) / job_id_str
    job_dir.mkdir(parents=True, exist_ok=True)
    video_path = job_dir / video.filename

    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    # 3. Dispatch Celery Task
    process_video_task.delay(job_id_str)

    return {"job_id": job_id_str, "status": job.status}


class VideoUrlRequest(BaseModel):
    video_url: HttpUrl


@app.post("/api/v1/jobs/analyze-url")
async def analyze_video_from_url(request: VideoUrlRequest, db: Session = Depends(get_db)):
    video_url_str = str(request.video_url)

    # Derive a safe filename from the URL path
    parsed_path = urlparse(video_url_str).path
    filename = Path(parsed_path).name or "video.mp4"
    if not filename.lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".webm")):
        filename = filename + ".mp4"

    # 1. Create Job record
    job = Job(video_filename=filename)
    db.add(job)
    db.commit()
    db.refresh(job)

    job_id_str = str(job.id)

    # 2. Setup job directory and download video
    job_dir = Path(settings.JOBS_DIR) / job_id_str
    job_dir.mkdir(parents=True, exist_ok=True)
    video_path = job_dir / filename

    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=300) as client:
            async with client.stream("GET", video_url_str) as response:
                if response.status_code != 200:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Failed to download video from URL. Server responded with {response.status_code}."
                    )
                with open(video_path, "wb") as f:
                    async for chunk in response.aiter_bytes(chunk_size=1024 * 1024):
                        f.write(chunk)
    except httpx.RequestError as e:
        # Clean up the empty job dir on network failure
        shutil.rmtree(job_dir, ignore_errors=True)
        job.status = "FAILED"
        job.error_message = f"Network error while downloading video: {str(e)}"
        db.commit()
        raise HTTPException(status_code=400, detail=f"Could not reach the provided URL: {str(e)}")

    # 3. Dispatch Celery Task
    process_video_task.delay(job_id_str)

    return {"job_id": job_id_str, "status": job.status, "filename": filename}


@app.get("/api/v1/jobs/{job_id}/status")
def get_job_status(job_id: str, db: Session = Depends(get_db)):
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return {
        "job_id": str(job.id),
        "status": job.status,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "error_message": job.error_message
    }

@app.get("/api/v1/jobs/{job_id}/result")
def get_job_result(job_id: str, db: Session = Depends(get_db)):
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status == "DELETED":
        raise HTTPException(status_code=410, detail="Result has been deleted by automated cleanup")
    if job.status != "COMPLETED":
        raise HTTPException(status_code=400, detail=f"Job is currently {job.status}")
    if not job.result_filepath or not os.path.exists(job.result_filepath):
        raise HTTPException(status_code=404, detail="Result file not found on server")

    return FileResponse(
        path=job.result_filepath, 
        filename=f"tribe_predictions_{job_id}.npz",
        media_type="application/octet-stream"
    )
