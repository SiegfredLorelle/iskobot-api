from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import RedirectResponse, JSONResponse, FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from app.database.vectorstore import initialize_vectorstore
from app.models.Query import Query, QueryRequest, QueryResponse
from app.transcripts_processing.transcriber import transcribe_audio
from app.utils.retry_with_backoff import retry_with_backoff
from google.api_core.exceptions import ResourceExhausted
from supabase import create_client, Client
import os
import tempfile
import requests
from pydantic import BaseModel
from gradio_client import Client, handle_file
from app.config import Config
from app.routes.auth import router as auth_router
from app.routes.kms import router as kms_router
from elevenlabs.client import ElevenLabs

supabase: Client = create_client(
    Config.SUPABASE_URL,
    Config.SUPABASE_KEY
)

elevenlabs = ElevenLabs(
    api_key=Config.ELEVENLABS_API_KEY
)

app = FastAPI()

# xtts_client = Client("jimmyvu/Coqui-Xtts-Demo")

class Message(BaseModel):
    text: str

# CORS Middleware setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://iskobot-ui.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
)

# Include authentication routes
app.include_router(auth_router)
app.include_router(kms_router)

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

knowledge_bank_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 5,
        "score_threshold": 0.5
    }
) | format_docs

# (3) Create prompt template
prompt_template = PromptTemplate.from_template(
    """You're **Iskobot**, a helpful and knowledgeable assistant in Computer Engineering.

**When responding:**
- Answer only using the provided knowledge bank. If the topic isn’t covered, say something like:
  *"That's a bit outside what I know right now. My focus is on Computer Engineering, but I'm happy to help with that if it relates!"*
- Keep answers **clear, concise, and accurate**.
- Use **bullet points** where helpful.
- Match the language and tone of the user's question.
- Don’t mention the knowledge bank or where the info came from.
- Focus on **technical accuracy** for anything related to Computer Engineering.
- When it fits naturally, end with a helpful follow-up like:
  *"Would you also like to know more about [related topic]?"*

Knowledge Bank: {knowledge_bank}

Question: {query}
Your answer: """
)

# (4) Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.2,
    max_output_tokens=500,
    top_k=40,
    top_p=0.95,
    google_api_key=Config.GEMINI_API_KEY
)

# (5) Chain everything together
chain = (
    RunnableParallel({
        "knowledge_bank": knowledge_bank_retriever,
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


# TODO: Use different query for public use and with login use

# Handle query requests
@app.post("/query", response_model=QueryRequest)
async def get_answers_from_query(request: QueryRequest):
    async def invoke_chain():
        return await chain.ainvoke(request.query)

    try:
        answer = await retry_with_backoff(invoke_chain)
        response = QueryResponse(response=answer)
        # Log query and response to Supabase
        log_data = {
            "query": request.query,
            "response": response.response
        }
        supabase_response = supabase.table("query_logs").insert(log_data).execute()

        if not supabase_response.data:
            print("Warning: Failed to log query and response to Supabase")

        return JSONResponse(content=response.dict())
    except ResourceExhausted as e:
        print(f"Error: {e}")
        return JSONResponse(
            content={
                "error": "Online prediction request quota exceeded. Please try again later."
            },
            status_code=429
        )

# Transcribe audio input
@app.post("/transcribe")
async def transcribe_speech(audio_file: UploadFile = File(...)):
    return await transcribe_audio(audio_file)

# Generate voice output
@app.post("/speech")
async def generate_speech(message: Message):
    try:
        print(f"Generating speech for: {message.text}")

        audio = elevenlabs.text_to_speech.convert(
            text=message.text,
            voice_id="zZLmKvCp1i04X8E0FJ8B",
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128",
        )
        
        audio_bytes = b"".join(audio)  # consume generator to bytes

        return Response(content=audio_bytes, media_type="audio/mpeg")

    except Exception as e:
        print("Error in /speech:", e)
        return Response(content=str(e), status_code=500)

# Add routes for the chain
add_routes(app, chain)

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)