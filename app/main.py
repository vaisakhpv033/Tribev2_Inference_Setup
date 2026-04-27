import os
import shutil
from pathlib import Path
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
