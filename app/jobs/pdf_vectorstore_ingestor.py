from google.cloud import storage
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
from app.database.vectorstore import initialize_vectorstore
from app.config import Config
from app.storage.gcs import GCSHandler

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
        chunk_size=1_000,          # Adjust based on your needs
        chunk_overlap=100,        # Slight overlap to maintain context
        length_function=len,
        is_separator_regex=False
    )
    
    texts = text_splitter.create_documents(
        texts=[text],
        metadatas=[metadata]
    )
    
    return texts

def main():
    # Get all PDFs from Google Cloud Storage
    gcs_handler = GCSHandler()
    pdf_files = gcs_handler.list_pdf_files()
    print(f"Number of PDFs found: {len(pdf_files)}")
    
    # Set up PGVector instance
    store = initialize_vectorstore(delete_on_insert=True)

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
    print("\nStoring chunks to database")
    if all_chunks:
        try:
            texts = [chunk.page_content for chunk in all_chunks]
            metadatas = [chunk.metadata for chunk in all_chunks]
            
            ids = store.add_texts(texts=texts, metadatas=metadatas)
            print(f"uccessfully saved {len(ids)} chunks to database")
            print(f"Average chunk size: {sum(len(t) for t in texts) / len(texts):.0f} characters")
            
        except Exception as e:
            print(f"Error saving to database: {str(e)}")
    else:
        print("No chunks to save to the database")

if __name__ == "__main__":
    main()