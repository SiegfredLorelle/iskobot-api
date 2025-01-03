from dotenv import load_dotenv
import os

load_dotenv()

class Config:
    REGION = os.getenv("REGION", "")
    DB_INSTANCE_NAME = os.getenv("DB_INSTANCE_NAME", "")
    DB_USER = os.getenv("DB_USER", "")
    DB_NAME = os.getenv("DB_NAME", "")
    DB_PASS = os.getenv("DB_PASS", "")
    GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "")
