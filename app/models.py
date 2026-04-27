import uuid
from datetime import datetime
from sqlalchemy import Column, String, Text, DateTime
from .database import Base

def generate_uuid():
    return str(uuid.uuid4())

class Job(Base):
    __tablename__ = "jobs"

    id = Column(String, primary_key=True, default=generate_uuid, index=True)
    status = Column(String, default="PENDING", index=True) # PENDING, STARTED, COMPLETED, FAILED, DELETED
    video_filename = Column(String, nullable=False)
    result_filepath = Column(String, nullable=True)
    error_message = Column(Text, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    deleted_at = Column(DateTime, nullable=True)
