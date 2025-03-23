# from app.database.GeminiEmbeddings import GeminiEmbeddings
from app.database.RateLimitedEmbeddings import RateLimitedEmbeddings
from langchain_community.vectorstores.pgvector import PGVector
from app.database.connector import get_db_connection
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.config import Config

def initialize_vectorstore(for_ingestion=False):
    """Initialize the vector store with improved rate limiting for ingestion."""
    # Default embedding function
    embedding_function = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key = Config.GEMINI_API_KEY
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
    import time

    vectorstore = initialize_vectorstore()
    
    # Test single query embedding
    test_text = "This is a test sentence."
    embedding = vectorstore.embedding_function.embed_query(test_text)
    print("Single query embedding:", embedding[:5], "...")  # Print first few values
    print("Embedding length:", len(embedding))

    # Test batch embedding with rate limiting
    texts = ["Sentence 1", "Sentence 2", "Sentence 3", "Sentence 4", "Sentence 5", "Sentence 6"]
    vectorstore = initialize_vectorstore(for_ingestion=True)

    start_time = time.time()
    embeddings = vectorstore.embedding_function.embed_documents(texts)
    end_time = time.time()

    print(f"Time taken for batch embedding: {end_time - start_time:.2f} seconds")
    print("First batch embedding sample:", embeddings[0][:5], "...")  # Print first few values