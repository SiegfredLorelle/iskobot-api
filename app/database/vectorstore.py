from langchain_community.vectorstores.pgvector import PGVector
from .connector import get_db_connection
from langchain_google_vertexai import VertexAIEmbeddings
import time
from langchain.embeddings.base import Embeddings

from langchain.embeddings.base import Embeddings
import time
from tenacity import retry, wait_exponential, stop_after_attempt
import numpy as np

class RateLimitedEmbeddings(Embeddings):
    """Custom embeddings class with improved rate limiting capabilities."""
    
    def __init__(self, base_embeddings, for_ingestion=False, 
                 batch_size=5, base_delay=2):
        self.base_embeddings = base_embeddings
        self.for_ingestion = for_ingestion
        self.batch_size = batch_size
        self.base_delay = base_delay    

    @retry(wait=wait_exponential(multiplier=2, min=4, max=60),
           stop=stop_after_attempt(5))
    def _get_embeddings_with_retry(self, texts):
        """Get embeddings with retry logic."""
        return self.base_embeddings.embed_documents(texts)

    def embed_documents(self, texts):
        """Embed multiple documents with batching and rate limiting."""
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            
            # Add base delay between batches
            if i > 0:
                time.sleep(self.base_delay)
            
            try:
                batch_embeddings = self._get_embeddings_with_retry(batch_texts)
                all_embeddings.extend(batch_embeddings)
                
                # Print progress
                print(f"Processed batch {i//self.batch_size + 1}/{(len(texts) + self.batch_size - 1)//self.batch_size}")
                
            except Exception as e:
                print(f"Error processing batch {i//self.batch_size + 1}: {str(e)}")
                raise e
                
        return all_embeddings
    
    def embed_query(self, text):
        """Embed a single query."""
        return self.base_embeddings.embed_query(text)

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