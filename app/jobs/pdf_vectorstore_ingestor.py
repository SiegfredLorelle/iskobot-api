from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.database.vectorstore import initialize_vectorstore
from app.storage.gcs import GCSHandler
from app.document_processing.pdf import extract_text_from_pdf
from app.document_processing.chunking import create_chunks

def main():
    # Get all PDFs from Google Cloud Storage
    gcs_handler = GCSHandler()
    pdf_files = gcs_handler.list_pdf_files()
    print(f"Number of PDFs found: {len(pdf_files)}")

    # Set up PGVector instance
    store = initialize_vectorstore(delete_on_insert=True)

    # Process each PDF
    all_chunks = []
    for pdf in pdf_files:
        try:
            print(f"\nProcessing PDF: {pdf.name}")
            result = extract_text_from_pdf(pdf)
            chunks = create_chunks(result["text"], result["metadata"])
            print(f"Created {len(chunks)} chunks from {pdf.name}")
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