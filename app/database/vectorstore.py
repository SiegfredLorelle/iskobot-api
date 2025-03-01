from app.database.RateLimitedEmbeddings import RateLimitedEmbeddings
from langchain_community.vectorstores.pgvector import PGVector
from app.database.connector import get_db_connection
from langchain_google_vertexai import VertexAIEmbeddings

def initialize_vectorstore(for_ingestion=False):
    """Initialize the vector store with improved rate limiting for ingestion."""
    # Default embedding function
    embedding_function = VertexAIEmbeddings(
            model_name="text-embedding-005"
        )

    if for_ingestion:
        embedding_function = RateLimitedEmbeddings(
            base_embeddings=embedding_function, 
            for_ingestion=for_ingestion,
            batch_size=5,  # Process 5 documents at a time
            base_delay=2   # Wait 2 seconds between batches
        )
    
    # Ensure PGVector still gets a valid embedding function
    return PGVector(
        connection_string="postgresql+pg8000://",
        use_jsonb=True,
        engine_args=dict(
            creator=get_db_connection,
        ),
        embedding_function=embedding_function,
        pre_delete_collection=for_ingestion
    )
