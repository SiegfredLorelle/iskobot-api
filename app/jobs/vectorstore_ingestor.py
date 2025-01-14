from app.document_processing.preprocess_documents import preprocess_document
from app.database.vectorstore import initialize_vectorstore
from app.storage.GCSHandler import GCSHandler
from app.document_processing.chunking import create_chunks
from tqdm import tqdm

def main():
    # Initialize GCS handler
    gcs_handler = GCSHandler()
    
    # List all supported files
    supported_files = gcs_handler.list_files_by_extension(["pdf", "docx", "pptx"])
    print(f"Number of files found: {len(supported_files)}")

    # Set up PGVector instance
    store = initialize_vectorstore(for_ingestion=True)

    # Process each file
    all_chunks = []
    for file in tqdm(supported_files, desc="Processing files"):
        try:
            print(f"\nProcessing: {file.name}")
            file_type = file.name.split(".")[-1].lower()
            result = preprocess_document(file, file_type)
            chunks = create_chunks(result["text"], result["metadata"])
            print(f"Created {len(chunks)} chunks from {file.name}")
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"Error processing {file.name}: {str(e)}")

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
                print(f"Successfully saved batch {i//batch_size + 1}/{(len(all_chunks) + batch_size - 1)//batch_size}")
            print(f"Successfully saved all {len(all_chunks)} chunks to the database")
            print(f"Average chunk size: {sum(len(c.page_content) for c in all_chunks) / len(all_chunks):.0f} characters")
        except Exception as e:
            print(f"Error saving to database: {str(e)}")
    else:
        print("No chunks to save to the database")

if __name__ == "__main__":
    main()
