from pydantic_settings import BaseSettings
from typing import Optional
import torch

class Settings(BaseSettings):
    # PINECONE
    PINECONE_API_KEY: str
    PINECONE_INDEX_NAME: str = "faces-index"
    PINECONE_ENVIRONMENT: Optional[str] = None
    
    # FACE RECOGNITION
    SIMILARITY_THRESHOLD: float = 0.51
    TOP_K: int = 1
    IMAGE_SIZE: int = 160
    MARGIN: int = 20
    
    # DEVICE
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # APP
    PROJECT_NAME: str = "Zero-Shot Face Recognition API"
    VERSION: str = "1.0.0"

    # DB
    DATABASE_URL: str = "postgresql://postgres:admin@localhost:5432/attendance_db"

    model_config = {
        "env_file": ".env",
        "extra": "ignore"
    }

settings = Settings()
