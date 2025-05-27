# from fastapi import APIRouter, HTTPException, status
# from app.jobs.vectorstore_ingestor import run_vectorstore_ingestor

# router = APIRouter()

# @router.post("/ingest")
# async def ingest_documents():
#     """
#     Run full document and web source ingestion.
#     """
#     try:
#         stats = run_vectorstore_ingestor()
#         return {"message": "Ingestion completed", "details": stats}
#     except Exception as e:
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Ingestion failed: {str(e)}"
#         )


# app/routes/rag.py
from fastapi import APIRouter, HTTPException, status, BackgroundTasks
from sse_starlette.sse import EventSourceResponse
import asyncio
from contextlib import asynccontextmanager
from app.document_processing.preprocess_documents import preprocess_document, SupabaseBlob
from app.database.vectorstore import initialize_vectorstore
from app.scraper.process_web_sources import process_web_sources
from app.storage.supabase_storage_handler import SupabaseStorageHandler
from app.document_processing.chunking import create_chunks
from tqdm import tqdm
from app.config import Config
from supabase import create_client, Client
from datetime import datetime, timezone
import json
from fastapi import APIRouter, HTTPException, status, BackgroundTasks
from sse_starlette.sse import EventSourceResponse
import asyncio
from contextlib import asynccontextmanager
import threading
import time


router = APIRouter()
ingestion_progress = {
    "percentage": 0,
    "message": "",
    "active": False,
    "error": None
}


def run_ingestion_task():
    """Synchronous task runner in a separate thread"""
    global ingestion_progress
    try:
        # Initialize progress
        ingestion_progress.update({
            "active": True,
            "percentage": 0,
            "message": "Starting ingestion...",
            "error": None
        })

        # ---------- PHASE 1: File Processing ---------- 
        ingestion_progress.update({
            "message": "Listing files...",
            "percentage": 5
        })
        
        gcs_handler = SupabaseStorageHandler()
        supabase = create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)
        supported_files = gcs_handler.list_files_by_extension(["pdf", "docx", "pptx"])
        total_files = len(supported_files)
        all_chunks = []

        # ---------- PHASE 2: Process Files ----------
        ingestion_progress.update({
            "message": "Processing documents...",
            "percentage": 10
        })

        for idx, file in enumerate(supported_files):
            # Update progress for each file
            progress = 10 + int((idx/total_files)*25)
            ingestion_progress.update({
                "message": f"Processing document {idx+1}/{total_files}",
                "percentage": progress
            })

            try:
                file_name = file['name']
                print(f"\nProcessing: {file_name}")
                file_content = gcs_handler.bucket.download(file_name)
                blob = SupabaseBlob(file_content, file_name)
                result = preprocess_document(blob, file_name.split(".")[-1].lower())
                chunks = create_chunks(result["text"], result["metadata"])
                print(f"Created {len(chunks)} chunks from {file_name}\n")
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"Error processing {file_name}: {str(e)}")

        # ---------- PHASE 3: Web Scraping ----------
        ingestion_progress.update({
            "message": "Scraping websites...",
            "percentage": 40
        })
        #         # web_sources = [
        #         #     "https://sites.google.com/view/pupous",
        #         #     "https://pupsinta.freshservice.com/support/solutions",
        #         #     "https://www.pup.edu.ph/"
        #         #     # Add more URLs as needed
        #         # ]
        try:
            response = supabase.table("rag_websites").select("url").execute()
            web_sources = [row["url"] for row in response.data]
            current_time = datetime.now(timezone.utc).isoformat()
            
            # Update last_scraped timestamps
            for item in response.data:
                if "url" in item:
                    supabase.table("rag_websites").update({"last_scraped": current_time}).eq("url", item["url"]).execute()
            
            # Process web sources
            web_documents = process_web_sources(web_sources)
            for doc in web_documents:
                chunks = create_chunks(doc.page_content, doc.metadata)
                all_chunks.extend(chunks)
                print(f"Created {len(chunks)} chunks from web sources")

        except Exception as e:
            raise Exception(f"Web processing failed: {e}")

        # ---------- PHASE 4: Store Chunks ----------
        ingestion_progress.update({
            "message": "Storing chunks...",
            "percentage": 70
        })

        if all_chunks:
            try:
                store = initialize_vectorstore(for_ingestion=True)
                batch_size = 20
                total_batches = (len(all_chunks) + batch_size - 1) // batch_size
                
                for batch_idx in range(total_batches):
                    # Update batch progress
                    progress = 70 + int((batch_idx/total_batches)*30)
                    ingestion_progress.update({
                        "message": f"Saving batch {batch_idx+1}/{total_batches}",
                        "percentage": progress
                    })
                    
                    # Process batch
                    start_idx = batch_idx * batch_size
                    end_idx = start_idx + batch_size
                    batch = all_chunks[start_idx:end_idx]
                    texts = [chunk.page_content for chunk in batch]
                    metadatas = [chunk.metadata for chunk in batch]
                    store.add_texts(texts=texts, metadatas=metadatas)
                    print(f"Successfully saved batch {batch_idx + 1}/{(len(all_chunks) + batch_size - 1)//batch_size}")

            except Exception as e:
                raise Exception(f"Chunking failed: {e}")

        # ---------- COMPLETION ----------
        ingestion_progress.update({
            "percentage": 100,
            "message": "Ingestion complete!",
            "active": False
        })

    except Exception as e:
        ingestion_progress.update({
            "error": str(e),
            "active": False
        })
        print(f"Ingestion failed: {str(e)}")

@router.post("/ingest")
async def start_ingestion():
    if ingestion_progress["active"]:
        raise HTTPException(status_code=400, detail="Ingestion already in progress")
    
    # Reset progress state
    ingestion_progress.update({
        "percentage": 0,
        "message": "",
        "active": False,
        "error": None
    })

    # Run in separate thread to avoid blocking
    thread = threading.Thread(target=run_ingestion_task)
    thread.start()
    
    return {"message": "Ingestion started"}

@router.get("/ingest/stream")
async def ingestion_progress_stream():
    async def event_generator():
        last_percentage = -1
        while True:
            current = ingestion_progress.copy()
            
            if current["error"]:
                yield {
                    "event": "error",
                    "data": json.dumps({"error": current["error"]})
                }
                break
                
            if current["percentage"] != last_percentage:
                yield {
                    "event": "progress",
                    "data": json.dumps({
                        "percentage": current["percentage"],
                        "message": current["message"]
                    })
                }
                last_percentage = current["percentage"]
                
            if current["percentage"] >= 100 or not current["active"]:
                yield {"event": "complete", "data": json.dumps({})}
                break
                
            await asyncio.sleep(0.5)

    return EventSourceResponse(event_generator())