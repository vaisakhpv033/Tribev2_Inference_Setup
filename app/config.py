import os

class Settings:
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/tribev2_jobs")
    CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
    CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")
    JOBS_DIR = os.getenv("JOBS_DIR", "jobs")
    HF_TOKEN = os.getenv("HF_TOKEN")

settings = Settings()
