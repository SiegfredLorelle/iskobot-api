from app.document_processing.preprocess_documents import preprocess_document, SupabaseBlob
from app.database.vectorstore import initialize_vectorstore
from app.scraper.process_web_sources import process_web_sources
from app.storage.supabase_storage_handler import SupabaseStorageHandler
from app.document_processing.chunking import create_chunks
from tqdm import tqdm

def run_vectorstore_ingestor():
    # Initialize storage handler
    gcs_handler = SupabaseStorageHandler()
    
    # List all supported files
    supported_files = gcs_handler.list_files_by_extension(["pdf", "docx", "pptx"])
    print(f"Number of files found: {len(supported_files)}")

    # Set up PGVector instance
    store = initialize_vectorstore(for_ingestion=True)

    # Process files from GCS
    all_chunks = []
    for file in tqdm(supported_files, desc="Processing files in storage"):
        try:
            file_name = file['name']
            # print(f"\nProcessing: {file_name}")
            file_type = file_name.split(".")[-1].lower()
            # Download file
            file_content = gcs_handler.bucket.download(file_name)
            blob = SupabaseBlob(file_content, file_name)
            result = preprocess_document(blob, file_type)
            chunks = create_chunks(result["text"], result["metadata"])
            print(f"Created {len(chunks)} chunks from {file_name}\n")
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"Error processing {file_name}: {str(e)}")

    # Process web sources
    web_sources = [
        "https://sites.google.com/view/pupous",
        "https://pupsinta.freshservice.com/support/solutions",
        "https://www.pup.edu.ph/"
        # Add more URLs as needed
    ]
    
    print("\nProcessing web sources")
    web_documents = process_web_sources(web_sources)
    for doc in web_documents:
        chunks = create_chunks(doc.page_content, doc.metadata)
        all_chunks.extend(chunks)
    print(f"Created {len(chunks)} chunks from web sources")

    stats = {"total_chunks": len(all_chunks), "batches_saved": 0}
    
    # Store chunks in the database
    print("\nStoring chunks to the database")
    if all_chunks:
        try:
            batch_size = 20  # Adjust based on your needs
            for i in range(0, len(all_chunks), batch_size):
                batch = all_chunks[i:i + batch_size]
                texts = [chunk.page_content for chunk in batch]
                metadatas = [chunk.metadata for chunk in batch]
                ids = store.add_texts(texts=texts, metadatas=metadatas)
                stats["batches_saved"] += 1
                print(f"Successfully saved batch {i//batch_size + 1}/{(len(all_chunks) + batch_size - 1)//batch_size}")
            
            stats["avg_chunk_size"] = sum(len(c.page_content) for c in all_chunks) // len(all_chunks)
            print(f"Successfully saved all {len(all_chunks)} chunks to the database")
            print(f"Average chunk size: {sum(len(c.page_content) for c in all_chunks) / len(all_chunks):.0f} characters")
        except Exception as e:
            print(f"Error saving to database: {str(e)}")
    else:
        print("No chunks to save to the database")
    return stats

if __name__ == "__main__":
    run_vectorstore_ingestor()