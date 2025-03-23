from supabase import create_client, Client
from app.config import Config

class SupabaseStorageHandler:
    def __init__(self):
        """Initialize the Supabase client and specify the bucket."""
        # Create Supabase client
        self.supabase: Client = create_client(
            Config.SUPABASE_URL,
            Config.SUPABASE_KEY
        )
        # Get the storage bucket (equivalent to GCS bucket)
        self.bucket = self.supabase.storage.from_(Config.SUPABASE_BUCKET_NAME)
        print(self.bucket)

    def list_files(self):
        """List all files in the bucket."""
        try:
            # List all files in the bucket
            response = self.bucket.list()
            # Return list of file objects (each has 'name', 'id', etc.)
            return response
        except Exception as e:
            print(f"Error listing files in bucket: {str(e)}")
            return []

    def list_files_by_extension(self, extensions):
        """List files in the bucket that match the given extensions."""
        if not extensions:
            print("No extensions provided. Returning all files.")
            return self.list_files()

        try:
            # Get all files and filter by extension
            files = self.list_files()
            return [
                file for file in files
                if file["name"].split(".")[-1].lower() in extensions
            ]
        except Exception as e:
            print(f"Error filtering files by extension: {str(e)}")
            return []

# Example usage
if __name__ == "__main__":
    handler = SupabaseStorageHandler()
    all_files = handler.list_files()
    print("All files:", [file["name"] for file in all_files])
    pdf_files = handler.list_files_by_extension(["pdf", "PDF"])
    print("PDF files:", [file["name"] for file in pdf_files])