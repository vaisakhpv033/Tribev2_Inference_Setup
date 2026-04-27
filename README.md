# TRIBEv2 Async Inference Pipeline

This is the production-ready asynchronous pipeline for TRIBEv2 video processing.

## Architecture
- **FastAPI**: Web server handling requests.
- **Celery**: Background task queue.
- **PostgreSQL**: Job tracking and metadata storage.
- **Redis**: Message broker for Celery.
- **Docker Compose**: Container orchestration.

## Setup Instructions
1. Ensure your `.env` file is populated with your `HF_TOKEN`.
2. Run the application using Docker Compose:
```bash
docker-compose up -d --build
```
3. The API will be available at `http://localhost:8000/docs`.
