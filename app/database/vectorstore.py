from app.database.RateLimitedEmbeddings import RateLimitedEmbeddings
from langchain_community.vectorstores.pgvector import PGVector
from app.database.connector import get_db_connection
import google.generativeai as genai

def initialize_vectorstore(for_ingestion=False):
    """Initialize the vector store with improved rate limiting for ingestion."""
    # Default embedding function
    genai.configure(api_key="AIzaSyAj3zJJttO6dtjC8NIybsozUHcnKgX7eAQ")
    def embedding_function(text):
        result = genai.models.embed_content(
            model="gemini-embedding-exp-03-07",
            contents=text
        )
        return result.embeddings

    if for_ingestion:
        embedding_function = RateLimitedEmbeddings(
            base_embeddings=embedding_function, 
            for_ingestion=for_ingestion,
            batch_size=5,  # Process 5 documents at a time
            base_delay=2   # Wait 2 seconds between batches
        )
    
    # Ensure PGVector still gets a valid embedding function
    return PGVector(
        connection_string="postgresql+psycopg2://",
        use_jsonb=True,
        engine_args=dict(
            creator=get_db_connection,
        ),
        embedding_function=embedding_function,
        pre_delete_collection=for_ingestion
    )



# Run the test
if __name__ == "__main__":
    # Initialize the vector store
    try:
        vs = initialize_vectorstore()
        print(f"Vetor Store initialized!")
    except Exception as e:
        print(f"Error initializing vector store: {e}")