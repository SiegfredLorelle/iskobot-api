# app/routes/rag.py
from fastapi import APIRouter, HTTPException, Depends, status, File, UploadFile, Form
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, HttpUrl, validator
from supabase import Client
from typing import List, Optional
from datetime import datetime, timezone
import uuid
import logging
from app.config import Config
from app.utils.auth_utils import verify_jwt_token, get_current_user
import mimetypes
import requests
from bs4 import BeautifulSoup
import asyncio
import aiohttp
from fastapi import Path

# Initialize router
router = APIRouter(prefix="/rag", tags=["RAG Management"])

# Security scheme
security = HTTPBearer()

logger = logging.getLogger(__name__)

# Pydantic models
class FileResponse(BaseModel):
    id: str
    name: str
    size: int
    type: str
    uploaded_at: datetime
    vectorized: bool

class WebsiteRequest(BaseModel):
    url: HttpUrl
    
    @validator('url')
    def validate_url(cls, v):
        if not str(v).startswith(('http://', 'https://')):
            raise ValueError('URL must start with http:// or https://')
        return v

# Re-enabled and updated WebsiteResponse model to match your provided schema
class WebsiteResponse(BaseModel):
    id: str
    url: str
    last_scraped: Optional[datetime]
    status: str
    vectorized: bool
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime]

class WebsiteCreate(BaseModel):
    url: HttpUrl
class VectorizationToggleRequest(BaseModel):
    vectorized: bool

# Allowed file types
ALLOWED_FILE_TYPES = {
    'image/jpeg', 'image/jpg', 'image/png', 'application/pdf'
}

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

def get_supabase_client() -> Client:
    """Dependency to get Supabase client"""
    from supabase import create_client
    return create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)

# --- File Management Routes ---

@router.post("/files/upload", response_model=List[FileResponse])
async def upload_files(
    files: List[UploadFile] = File(...),
    supabase: Client = Depends(get_supabase_client)
):
    """Upload multiple files to Supabase Storage and record metadata in rag_files table."""
    try:

        uploaded_files = []
        
        for file in files:
            if file.content_type not in ALLOWED_FILE_TYPES:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"File type {file.content_type} not allowed. Allowed types: DOCX, PDF, PPTX"
                )
            
            file_content = await file.read()
            
            if len(file_content) > MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"File {file.filename} exceeds maximum size of 10MB"
                )
            
            file_id = str(uuid.uuid4())
            file_extension = file.filename.split('.')[-1] if '.' in file.filename else ''
            storage_filename = f"{file_id}.{file_extension}" 
            
            storage_response = supabase.storage.from_("iskobot-documents-2.0-lms-only").upload(
                storage_filename,
                file_content,
                {
                    "content-type": file.content_type,
                    "upsert": False 
                }
            )

            if hasattr(storage_response, 'error') and storage_response.error:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to upload {file.filename}: {storage_response.error.message}"
                )
            
            file_record = {
                "id": file_id,
                "name": file.filename,
                "storage_name": storage_filename,
                "size": len(file_content),
                "type": file.content_type,
                "uploaded_at": datetime.now(timezone.utc).isoformat(),
                "vectorized": False,
            }

            
            db_response = supabase.table("rag_files").insert(file_record).execute()
            
            if hasattr(db_response, 'error') and db_response.error:
                supabase.storage.from_("iskobot-documents-2.0-lms-only").remove([storage_filename])
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to save file metadata for {file.filename}: {db_response.error.message}"
                )
            
            uploaded_files.append(FileResponse(
                id=file_id,
                name=file.filename,
                size=len(file_content),
                type=file.content_type,
                uploaded_at=datetime.now(timezone.utc),
                vectorized=False,
            ))
        
        return uploaded_files
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File upload error: {str(e)}") 
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="File upload failed"
        )


@router.get("/files", response_model=List[FileResponse])
async def get_user_files(
    supabase: Client = Depends(get_supabase_client)
):
    """Get all files from the rag_files table."""
    
    try:
        response = supabase.table("rag_files").select("id, name, size, type, uploaded_at, vectorized").execute()
        
        files_data = []
        if response.data: 
            for file_record in response.data:
                uploaded_at_str = file_record["uploaded_at"]
                uploaded_at = datetime.fromisoformat(uploaded_at_str.replace('Z', '+00:00')) if uploaded_at_str else datetime.now(timezone.utc)

                files_data.append(
                    FileResponse(
                        id=file_record["id"],
                        name=file_record["name"],
                        size=file_record["size"],
                        type=file_record["type"],
                        uploaded_at=uploaded_at,
                        vectorized=file_record["vectorized"],
                    )
                )
        
        return files_data 

    except Exception as e:
        logger.error(f"Error retrieving files from rag_files table: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Could not retrieve files: {e}")


@router.delete("/files/{file_id}", status_code=204)
async def delete_file(
    file_id: str = Path(..., description="The ID of the file to delete"),
    supabase: Client = Depends(get_supabase_client)
):
    """Delete a file from the rag_files table and Supabase storage."""
    try:
        # Step 1: Get the file metadata
        file_query = supabase.table("rag_files").select("name").eq("id", file_id).single().execute()
        file_data = file_query.data

        if not file_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="File not found in database"
            )

        file_name = file_data["name"]

        # Step 2: Delete from Supabase Storage
        try:
            file_query = supabase.table("rag_files").select("storage_name").eq("id", file_id).single().execute()
            file_data = file_query.data
            storage_name = file_data["storage_name"]
            # Delete directly
            supabase.storage.from_("iskobot-documents-2.0-lms-only").remove([storage_name])
        except Exception as storage_err:
            logger.warning(f"Storage delete warning for {file_name}: {storage_err}")

        # Step 3: Delete metadata from rag_files table
        db_response = supabase.table("rag_files").delete().eq("id", file_id).execute()
        if not db_response.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="File metadata not found"
            )

        return  # 204 No Content

    except Exception as e:
        logger.error(f"Error deleting file {file_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting file: {str(e)}"
        )

## Web Source Management Routes
@router.get("/websites", response_model=List[WebsiteResponse])
async def get_all_websites(
    supabase: Client = Depends(get_supabase_client)
):
    """Get all web sources from the rag_web_sources table."""
    try:
        # Query the rag_web_sources table for all columns
        response = supabase.table("rag_websites").select("*").execute()
        
        websites_data = []
        if response.data:
            for record in response.data:
                # Convert string timestamps from Supabase to datetime objects
                last_scraped_dt = datetime.fromisoformat(record["last_scraped"].replace('Z', '+00:00')) if record.get("last_scraped") else None
                created_at_dt = datetime.fromisoformat(record["created_at"].replace('Z', '+00:00'))
                updated_at_dt = datetime.fromisoformat(record["updated_at"].replace('Z', '+00:00')) if record.get("updated_at") else None

                websites_data.append(
                    WebsiteResponse(
                        id=record["id"],
                        url=record["url"],
                        last_scraped=last_scraped_dt,
                        status=record["status"],
                        vectorized=record["vectorized"],
                        error_message=record["error_message"],
                        created_at=created_at_dt,
                        updated_at=updated_at_dt,
                    )
                )
        
        return websites_data

    except Exception as e:
        logger.error(f"Error retrieving web sources: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not retrieve web sources: {e}"
        )
    

@router.delete("/websites/{website_id}", status_code=204)
async def delete_website(
    website_id: str = Path(..., description="The ID of the website to delete"),
    supabase: Client = Depends(get_supabase_client)
):
    """Delete a website from the rag_websites table."""
    try:
        response = supabase.table("rag_websites").delete().eq("id", website_id).execute()

        # If no records were deleted, raise 404
        if not response.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Website not found"
            )

        return  # 204 No Content

    except Exception as e:
        logger.error(f"Error deleting website {website_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting website: {str(e)}"
        )
    

@router.post("/websites", status_code=201)
async def add_website(
    website: WebsiteCreate,
    supabase: Client = Depends(get_supabase_client)
):
    try:
        now = datetime.utcnow().isoformat()

        new_website = {
            "id": str(uuid.uuid4()),
            "url": str(website.url),
            "status": "pending",
            "vectorized": False,
            "created_at": now,
            "updated_at": now,
            "last_scraped": None,
            "error_message": None,
        }

        response = supabase.table("rag_websites").insert(new_website).execute()

        if not response.data:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to insert website"
            )

        return response.data[0]

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error adding website: {e}"
        )
