import os
from google.cloud.sql.connector import Connector
import pg8000
from langchain_community.vectorstores.pgvector import PGVector
from langchain_google_vertexai import VertexAIEmbeddings
from google.cloud import storage
from dotenv import load_dotenv
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re

load_dotenv()

def clean_text(text: str) -> str:
    """Clean extracted text by removing extra whitespace and unwanted characters."""
    # Replace multiple newlines with a single newline
    text = re.sub(r'\n+', '\n', text)
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    # Remove any non-printable characters
    text = ''.join(char for char in text if char.isprintable())
    return text.strip()

def extract_text_from_pdf(blob) -> dict:
    """Extract text from PDF with metadata."""
    print(f"Processing: {blob.name}")
    metadata = {
        "source": blob.name,
        "page_numbers": []
    }
    
    with blob.open("rb") as f:
        doc = fitz.open(stream=f.read(), filetype="pdf")
        text_by_page = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            # Extract text with formatting details
            text = page.get_text("text")
            cleaned_text = clean_text(text)
            
            if cleaned_text.strip():  # Only add non-empty pages
                text_by_page.append(cleaned_text)
                metadata["page_numbers"].append(page_num + 1)
        
        doc.close()
    
    return {
        "text": "\n".join(text_by_page),
        "metadata": metadata
    }

def create_chunks(text: str, metadata: dict) -> list:
    """Create optimized chunks from text using RecursiveCharacterTextSplitter."""
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", "!", "?", ",", " "],
        chunk_size=500,          # Adjust based on your needs
        chunk_overlap=50,        # Slight overlap to maintain context
        length_function=len,
        is_separator_regex=False
    )
    
    texts = text_splitter.create_documents(
        texts=[text],
        metadatas=[metadata]
    )
    
    return texts

def main():
    # Initialize Storage Client
    client = storage.Client()
    bucket_name = os.getenv("GCS_BUCKET_NAME", "")
    bucket = client.bucket(bucket_name)
    print(f"Bucket name: {bucket}")

    # List all PDFs in the root of the GCS bucket
    blobs = bucket.list_blobs()
    pdf_files = [blob for blob in blobs if blob.name.lower().endswith('.pdf')]
    print(f"Number of PDFs found: {len(pdf_files)}")
    
    # Set up PGVector instance
    connector = Connector()
    
    def getconn() -> pg8000.dbapi.Connection:
        return connector.connect(
            os.getenv("DB_INSTANCE_NAME", ""),
            "pg8000",
            user=os.getenv("DB_USER", ""),
            password=os.getenv("DB_PASS", ""),
            db=os.getenv("DB_NAME", ""),
        )

    store = PGVector(
        connection_string="postgresql+pg8000://",
        use_jsonb=True,
        engine_args=dict(
            creator=getconn,
        ),
        embedding_function=VertexAIEmbeddings(
            model_name="textembedding-gecko@003"
        ),
        pre_delete_collection=True
    )

    # Process each PDF
    all_chunks = []
    total_chunks = 0
    
    for pdf in pdf_files:
        try:
            print(f"\nProcessing PDF: {pdf.name}")
            
            # Extract text and metadata
            result = extract_text_from_pdf(pdf)
            
            # Create chunks
            chunks = create_chunks(result["text"], result["metadata"])
            
            print(f"Created {len(chunks)} chunks from {pdf.name}")
            total_chunks += len(chunks)
            all_chunks.extend(chunks)
            
        except Exception as e:
            print(f"Error processing {pdf.name}: {str(e)}")

    # Store chunks in database
    if all_chunks:
        try:
            texts = [chunk.page_content for chunk in all_chunks]
            metadatas = [chunk.metadata for chunk in all_chunks]
            
            ids = store.add_texts(texts=texts, metadatas=metadatas)
            print(f"\nSuccessfully saved {len(ids)} chunks to database")
            print(f"Average chunk size: {sum(len(t) for t in texts) / len(texts):.0f} characters")
            
        except Exception as e:
            print(f"Error saving to database: {str(e)}")
    else:
        print("No chunks to save to the database")

if __name__ == "__main__":
    main()