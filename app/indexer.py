import os
from google.cloud.sql.connector import Connector
import pg8000
from langchain_community.vectorstores.pgvector import PGVector
from langchain_google_vertexai import VertexAIEmbeddings
from google.cloud import storage
from dotenv import load_dotenv
import fitz


load_dotenv()

# Retrieve all Cloud Run release notes from BigQuery 
client = storage.Client()
bucket_name = os.getenv("GCS_BUCKET_NAME", "")
bucket = client.bucket(bucket_name)
print(f"bucket name: {bucket}")

# List all PDFs in the GCS bucket
blobs = bucket.list_blobs()
pdf_files = [blob for blob in blobs if blob.name.lower().endswith('.pdf')]
print(f"Number of PDFs found: {len(pdf_files)}")
for pdf in pdf_files:
    print(f"Found PDF: {pdf.name}")

# Set up a PGVector instance 
connector = Connector()

def getconn() -> pg8000.dbapi.Connection:
    conn: pg8000.dbapi.Connection = connector.connect(
        os.getenv("DB_INSTANCE_NAME", ""),
        "pg8000",
        user=os.getenv("DB_USER", ""),
        password=os.getenv("DB_PASS", ""),
        db=os.getenv("DB_NAME", ""),
    )
    return conn

store = PGVector(
    connection_string="postgresql+pg8000://",
    use_jsonb=True,
    engine_args=dict(
        creator=getconn,
    ),
    embedding_function=VertexAIEmbeddings(
        model_name="text-embedding-005"
    ),
    pre_delete_collection=True  
)

# # Save all release notes into the Cloud SQL database
# texts = list(row["release_note"] for row in rows)
# ids = store.add_texts(texts)

# print(f"Done saving: {len(ids)} release notes")



# Function to extract text from a PDF file
def extract_text_from_pdf(blob):
    print(f"Processing: {blob.name}")
    with blob.open("rb") as f:
        doc = fitz.open(stream=f.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
    return text

# Extract text from each PDF and save to PGVector
texts = []
for pdf in pdf_files:
    try:
        print(f"Extracting text from: {pdf.name}")
        text = extract_text_from_pdf(pdf)
        texts.append(text)
        print(f"Successfully extracted text from: {pdf.name}")
    except Exception as e:
        print(f"Error processing {pdf.name}: {str(e)}")

# Store extracted text into the Cloud SQL database
if texts:
    try:
        ids = store.add_texts(texts)
        print(f"Successfully saved: {len(ids)} PDFs")
        print("Document IDs:", ids)
    except Exception as e:
        print(f"Error saving to database: {str(e)}")
else:
    print("No texts to save to the database")