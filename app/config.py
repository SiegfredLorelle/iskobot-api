from dotenv import load_dotenv
import os

load_dotenv()

class Config:
    # Google Cloud Platform
    REGION = os.getenv("REGION", "")
    GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "")
    DB_INSTANCE_NAME = os.getenv("DB_INSTANCE_NAME", "")

    # Database
    DB_USER = os.getenv("DB_USER", "")
    DB_NAME = os.getenv("DB_NAME", "")
    DB_PASS = os.getenv("DB_PASS", "")
    DB_HOST = os.getenv("DB_HOST", "")
    DB_PORT = os.getenv("DB_PORT", "")

    CONQUI_XTTS_ID = os.getenv("CONQUI_XTTS_ID", "")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
