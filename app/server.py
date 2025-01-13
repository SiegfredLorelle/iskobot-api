from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes
from langchain_google_vertexai import VertexAI
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from app.database.vectorstore import initialize_vectorstore
from app.models.query_request import QueryRequest
from app.models.query_response import QueryResponse
from groq import Groq
import tempfile
import os
import logging

# Set up logging for debugging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

# Initialize Groq client
client = Groq()

# CORS Middleware setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://iskobot-ui.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
)

# (1) Initialize VectorStore
vectorstore = initialize_vectorstore()

# (2) Build retriever
def format_docs(docs):
    formatted_docs = []
    for doc in docs:
        source = doc.metadata.get('source', 'Unknown')
        content = doc.page_content
        formatted_docs.append(f"Source: {source}\nContent: {content}")
    return "\n\n---\n\n".join(formatted_docs)

notes_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 5,
        "score_threshold": 0.5
    }
) | format_docs

# (3) Create prompt template
prompt_template = PromptTemplate.from_template(
    """You are an expert answering questions. 
Use the provided documentation to answer questions.
Give a concise answer.
If the answer isn't clear from the documents, say so.

Documentation: {notes}

Question: {query}
Your answer: """)
    
# (4) Initialize LLM
llm = VertexAI(
    model_name="gemini-1.0-pro-002",
    temperature=0.2,
    max_output_tokens=200,
    top_k=40,
    top_p=0.95
)

# (5) Chain everything together
chain = (
    RunnableParallel({
        "notes": notes_retriever,
        "query": RunnablePassthrough()
    })
    | prompt_template
    | llm
    | StrOutputParser()
)

# Redirect root to playground
@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/playground")

# Handle query requests
@app.post("/query", response_model=QueryRequest)
async def get_answers_from_query(request: QueryRequest):
    answer = await chain.ainvoke(request.query)
    response = QueryResponse(response=answer)
    return JSONResponse(content=response.dict())

# Transcribe audio input
@app.post("/transcribe")
async def transcribe_speech(audio_file: UploadFile = File(...)):
    try:
        # Debug: Log information about the incoming file
        logger.debug(f"Received file: {audio_file.filename}, size: {audio_file.size} bytes")
        logger.debug(f"Content type: {audio_file.content_type}")

        # Check if the file is indeed a valid WAV file
        if not audio_file.filename.endswith(".wav"):
            logger.error("Invalid file format. Expected .wav file.")
            raise HTTPException(status_code=422, detail="Invalid file format. Only .wav files are allowed.")

        # Create a temporary file to store the uploaded audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            content = await audio_file.read()
            logger.debug(f"Read {len(content)} bytes of data from the file")

            # Debug: Ensure that the file content is being written to the temporary file correctly
            temp_file.write(content)
            temp_path = temp_file.name

        # Debug: Log the path of the temporary file
        logger.debug(f"Temporary file created at: {temp_path}")

        try:
            # Open the temporary file and transcribe
            with open(temp_path, "rb") as file:
                logger.debug("Sending file to transcription service...")
                transcription = client.audio.transcriptions.create(
                    file=(temp_path, file.read()),
                    model="whisper-large-v3-turbo",
                    response_format="verbose_json",
                )
                transcription_text = transcription.text

            # Return transcription result
            logger.debug(f"Transcription result: {transcription_text}")
            return {"transcription": transcription_text}

        finally:
            # Clean up: remove the temporary file
            os.unlink(temp_path)
            logger.debug(f"Temporary file {temp_path} removed.")

    except Exception as e:
        # Log the error before raising it
        logger.error(f"Error during transcription: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during transcription: {str(e)}")

# Add routes for the chain
add_routes(app, chain)

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)