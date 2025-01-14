from google.cloud import storage
from app.config import Config

class GCSHandler:
    def __init__(self):
        """Initialize the GCS client and specify the bucket."""
        self.client = storage.Client()
        self.bucket = self.client.bucket(Config.GCS_BUCKET_NAME)

    def list_files(self):
        """List all files in the bucket."""
        try:
            return list(self.bucket.list_blobs())
        except Exception as e:
            print(f"Error listing files in bucket: {str(e)}")
            return []

    def list_files_by_extension(self, extensions):
        """List files in the bucket that match the given extensions."""
        if not extensions:
            print("No extensions provided. Returning all files.")
            return self.list_files()

        try:
            return [
                file for file in self.list_files()
                if file.name.split(".")[-1].lower() in extensions
            ]
        except Exception as e:
            print(f"Error filtering files by extension: {str(e)}")
            return []
