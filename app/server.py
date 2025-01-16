from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes
from langchain_google_vertexai import VertexAI
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from app.database.vectorstore import initialize_vectorstore
from app.models.QueryRequest import QueryRequest
from app.models.QueryResponse import QueryResponse
from app.transcripts_processing.transcriber import transcribe_audio
import os
import tempfile
import requests
from pydantic import BaseModel
from gradio_client import Client

app = FastAPI()

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

knowledge_bank_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 5,
        "score_threshold": 0.5
    }
) | format_docs

# (3) Create prompt template
prompt_template = PromptTemplate.from_template(
    """You are Iskobot, an expert in Computer Engineering.
Refer to the provided knowledge bank to answer questions.
Provide a brief and clear answer.
If the answer isn't clear from your knowledge bank, acknowledge that you don't have sufficient information.
If the question is asked in a different language, translate your answer into the same language.

Knowledge Bank: {knowledge_bank}

Question: {query}
Your answer: """)

# (4) Initialize LLM
llm = VertexAI(
    model_name="gemini-1.5-pro-002",
    temperature=0.2,
    max_output_tokens=500,
    top_k=40,
    top_p=0.95
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

# Handle query requests
@app.post("/query", response_model=QueryRequest)
async def get_answers_from_query(request: QueryRequest):
    answer = await chain.ainvoke(request.query)
    response = QueryResponse(response=answer)
    return JSONResponse(content=response.dict())

# Transcribe audio input
@app.post("/transcribe")
async def transcribe_speech(audio_file: UploadFile = File(...)):
    return await transcribe_audio(audio_file)

@app.post("/generate_speech")
async def generate_speech(message: Message):
    try:
        text = message.text

        # Create a temporary file for the reference audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
            reference_audio_url = "https://github.com/overtheskyy/iskobot-voice/blob/main/iskobot.wav?raw=true"
            download_audio(reference_audio_url, temp_audio_file.name)

            # Generate speech from the provided text
            audio_output = generate_speech_from_text(text, temp_audio_file.name)

        # Delete the temporary file after use
        os.remove(temp_audio_file.name)

        return {audio_output}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def download_audio(url: str, local_path: str) -> None:
    try:
        response = requests.get(url)
        response.raise_for_status()

        with open(local_path, 'wb') as file:
            file.write(response.content)
        print(f"Audio file downloaded successfully to {local_path}")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to download audio: {str(e)}")

def generate_speech_from_text(text: str, reference_audio_path: str) -> str:
    """
    Generate speech from the provided text using a reference audio sample.
    """
    client = Client("https://coqui-xtts.hf.space/--replicas/5891u/")

    if not os.path.exists(reference_audio_path):
        raise FileNotFoundError(f"Reference audio file not found at: {reference_audio_path}")
    
    try:
        result = client.predict(
            text,
            "en",
            reference_audio_path,
            "",
            False,
            False,
            True,
            True,
            fn_index=1
        )
        return result
    except Exception as e:
        raise Exception(f"Error occurred while generating speech: {str(e)}")

# Add routes for the chain
add_routes(app, chain)

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)