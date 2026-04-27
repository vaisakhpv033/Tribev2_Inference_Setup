import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from celery import Celery
from celery.schedules import crontab
from sqlalchemy.orm import Session
import numpy as np

from .config import settings
from .database import SessionLocal
from .models import Job

# Try to import TribeModel, fail gracefully if not in heavy environment
try:
    from tribev2.demo_utils import TribeModel
    from huggingface_hub import login
    HAS_TRIBE = True
except ImportError:
    HAS_TRIBE = False
    print("WARNING: tribev2 module not found. Inference will not work unless running in the proper environment.")

celery_app = Celery(
    "tribev2_worker",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND
)

# Load the model globally at worker startup so it stays in GPU memory
tribe_model = None

@celery_app.on_after_configure.connect
def setup_periodic_tasks(sender, **kwargs):
    # Runs every hour
    sender.add_periodic_task(
        crontab(minute=0),
        cleanup_old_jobs_task.s(),
        name="cleanup_old_jobs_every_hour"
    )

def get_model():
    global tribe_model
    if tribe_model is None and HAS_TRIBE:
        if settings.HF_TOKEN:
            print("Logging into HuggingFace Hub...")
            login(token=settings.HF_TOKEN)
            
        print("Loading TRIBEv2 model into GPU memory...")
        CACHE_FOLDER = Path("./cache")
        CACHE_FOLDER.mkdir(exist_ok=True)
        tribe_model = TribeModel.from_pretrained(
            "facebook/tribev2",
            cache_folder=CACHE_FOLDER,
        )
        print("Model successfully loaded!")
    return tribe_model

@celery_app.task(bind=True, max_retries=1)
def process_video_task(self, job_id_str: str):
    db: Session = SessionLocal()
    job = db.query(Job).filter(Job.id == job_id_str).first()
    
    if not job:
        db.close()
        return

    try:
        # Update status to STARTED
        job.status = "STARTED"
        db.commit()

        if not HAS_TRIBE:
            raise Exception("TRIBEv2 library is not installed in this environment.")

        model = get_model()
        if not model:
            raise Exception("Failed to load TRIBEv2 model.")

        job_dir = Path(settings.JOBS_DIR) / job_id_str
        video_path = job_dir / job.video_filename
        npz_path = job_dir / "image_predictions.npz"

        # 1. Extract Events
        df = model.get_events_dataframe(video_path=str(video_path))
        
        # 2. Predict Brain Activity
        preds, segments = model.predict(events=df)

        # 3. Save Output
        np.savez(str(npz_path), preds=preds, segments=segments)

        # Update DB
        job.status = "COMPLETED"
        job.result_filepath = str(npz_path)
        db.commit()

    except Exception as e:
        job.status = "FAILED"
        job.error_message = str(e)
        db.commit()
        raise e
    finally:
        db.close()

@celery_app.task
def cleanup_old_jobs_task():
    db: Session = SessionLocal()
    try:
        # Find jobs older than 6 hours that are COMPLETED or FAILED
        six_hours_ago = datetime.utcnow() - timedelta(hours=6)
        old_jobs = db.query(Job).filter(
            Job.status.in_(["COMPLETED", "FAILED"]),
            Job.created_at < six_hours_ago
        ).all()

        for job in old_jobs:
            job_dir = Path(settings.JOBS_DIR) / str(job.id)
            
            # Physically delete the directory
            if job_dir.exists() and job_dir.is_dir():
                shutil.rmtree(job_dir)
            
            # Update database record
            job.status = "DELETED"
            job.deleted_at = datetime.utcnow()
            
        db.commit()
    except Exception as e:
        print(f"Error during cleanup: {e}")
        db.rollback()
    finally:
        db.close()
