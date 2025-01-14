from app.database import RateLimitedEmbeddings
from langchain_community.vectorstores.pgvector import PGVector
from .connector import get_db_connection
from langchain_google_vertexai import VertexAIEmbeddings

def initialize_vectorstore(for_ingestion=False):
    """Initialize the vector store with improved rate limiting for ingestion."""
    base_embeddings = VertexAIEmbeddings(model_name="text-embedding-005")
    embedding_function = RateLimitedEmbeddings(
        base_embeddings, 
        for_ingestion=for_ingestion,
        batch_size=5,  # Process 5 documents at a time
        base_delay=2   # Wait 2 seconds between batches
    )
    
    return PGVector(
        connection_string="postgresql+pg8000://",
        use_jsonb=True,
        engine_args=dict(
            creator=get_db_connection,
        ),
        embedding_function=embedding_function,
        pre_delete_collection=for_ingestion
    )