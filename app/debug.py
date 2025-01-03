import os
from google.cloud.sql.connector import Connector
import pg8000
from langchain_community.vectorstores.pgvector import PGVector
from langchain_google_vertexai import VertexAIEmbeddings
from google.cloud import storage
from dotenv import load_dotenv

load_dotenv()

def list_all_files_in_bucket(bucket_name):
    """List all files in the bucket to help with debugging"""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    print(f"\nBucket name: {bucket.name}")
    print("\nListing ALL files in bucket (no prefix):")
    all_files = list(bucket.list_blobs())
    for blob in all_files:
        print(f"- {blob.name}")
    
    print("\nListing files with 'pdfs/' prefix:")
    pdfs_files = list(bucket.list_blobs(prefix="pdfs/"))
    for blob in pdfs_files:
        print(f"- {blob.name}")
    
    print("\nListing files with 'PDFs/' prefix (capital):")
    PDFs_files = list(bucket.list_blobs(prefix="PDFs/"))
    for blob in PDFs_files:
        print(f"- {blob.name}")
    
    return all_files, pdfs_files, PDFs_files

def main():
    # Get bucket name from environment
    bucket_name = os.getenv("GCS_BUCKET_NAME")
    if not bucket_name:
        print("Error: GCS_BUCKET_NAME environment variable is not set")
        return
    
    print(f"Using bucket name: {bucket_name}")
    
    try:
        all_files, pdfs_files, PDFs_files = list_all_files_in_bucket(bucket_name)
        
        print("\nSummary:")
        print(f"Total files in bucket: {len(all_files)}")
        print(f"Files in 'pdfs/' folder: {len(pdfs_files)}")
        print(f"Files in 'PDFs/' folder: {len(PDFs_files)}")
        
        # Check for PDF files in root directory
        pdf_files_root = [blob for blob in all_files if blob.name.lower().endswith('.pdf')]
        if pdf_files_root:
            print("\nFound PDF files in root directory:")
            for blob in pdf_files_root:
                print(f"- {blob.name}")
    
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        
        # Check if it's a permissions error
        if "permission" in str(e).lower():
            print("\nThis might be a permissions issue. Please verify:")
            print("1. Your application has the Storage Object Viewer role")
            print("2. Your credentials are properly set up")
            print("3. You're using the correct project and bucket")

if __name__ == "__main__":
    main()