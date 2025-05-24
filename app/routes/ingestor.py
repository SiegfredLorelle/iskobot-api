from fastapi import APIRouter, HTTPException, status
from app.jobs.vectorstore_ingestor import run_vectorstore_ingestor

router = APIRouter()

@router.post("/ingest")
async def ingest_documents():
    """
    Run full document and web source ingestion.
    """
    try:
        stats = run_vectorstore_ingestor()
        return {"message": "Ingestion completed", "details": stats}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {str(e)}"
        )
