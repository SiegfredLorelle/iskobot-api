from google.cloud import storage
from app.config import Config

class GCSHandler:
    def __init__(self):
        self.client = storage.Client()
        self.bucket = self.client.bucket(Config.GCS_BUCKET_NAME)

    def list_pdf_files(self):
        """List all PDFs in the root of the GCS bucket"""
        blobs = self.bucket.list_blobs()
        return [blob for blob in blobs if blob.name.lower().endswith('.pdf')]