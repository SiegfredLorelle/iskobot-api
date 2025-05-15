from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import RedirectResponse, JSONResponse, FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
#from langchain_core.prompts import PromptTemplate  # Removed
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
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.checkpoint.memory import MemorySaver

supabase: Client = create_client(
    Config.SUPABASE_URL,
    Config.SUPABASE_KEY
)

xtts_client = Client("jimmyvu/Coqui-Xtts-Demo")

app = FastAPI()

class Message(BaseModel):
    text: str
    
class QueryRequest(BaseModel):
    query: str
    thread_id: str | None = None

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

# (3) Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2,
    max_output_tokens=500,
    top_k=40,
    top_p=0.95,
    google_api_key=Config.GEMINI_API_KEY
)

# (4) Chain everything together
chain = (
    RunnableParallel({
        "knowledge_bank": knowledge_bank_retriever,
        "query": RunnablePassthrough()
    })
    | llm
    | StrOutputParser()
)

# (6) Message persistence
workflow = StateGraph(state_schema=MessagesState)

def call_model(state: MessagesState):
    user_query = state["messages"][-1].content

    # Retrieve knowledge
    knowledge_bank = knowledge_bank_retriever.invoke(user_query)

    # Use structured messages with system prompt instead of prompt template
    messages = [
        {"role": "system", "content": (
            "You are Iskobot, an expert in Computer Engineering.\n"
            "Refer to the provided knowledge bank to answer questions.\n"
            "Provide a brief and clear answer.\n"
            "If the answer isn't clear from your knowledge bank, acknowledge that you don't have sufficient information.\n"
            "If the question is asked in a different language, translate your answer into the same language.\n"
            f"\nKnowledge Bank:\n{knowledge_bank}"
        )},
        *state["messages"]
    ]

    # Call Gemini with full structured context
    response = llm.invoke(messages)
    response_text = response.content

    return {
        "messages": [
            {"role": "assistant", "content": response_text}
        ]
    }
    
# Build graph
workflow.add_node("model", call_model)
workflow.add_edge(START, "model")

# Add memory
memory = MemorySaver()
graph_app = workflow.compile(checkpointer=memory)

# Redirect root to playground
@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/playground")

# Handle query requests
@app.post("/query", response_model=QueryRequest)
async def get_answers_from_query(request: QueryRequest):
    try:
        # Format input as message for MessagesState
        messages = [{"role": "user", "content": request.query}]

        # Run the LangGraph graph
        result = await graph_app.ainvoke(
            {"messages": messages},
            config={"configurable": {"thread_id": request.thread_id or "default"}}
        )
        response_content = result["messages"][-1].content

        # Log to Supabase
        response = QueryResponse(response=response_content)
        supabase_response = supabase.table("query_logs").insert({
            "query": request.query,
            "response": response.response
        }).execute()
        if not supabase_response.data:
            print("Warning: Failed to log query and response to Supabase")

        return JSONResponse(content=response.dict())

    except ResourceExhausted as e:
        print(f"Error: {e}")
        return JSONResponse(
            content={"error": "Online prediction quota exceeded."},
            status_code=429
        )

# Transcribe audio input
@app.post("/transcribe")
async def transcribe_speech(audio_file: UploadFile = File(...)):
    return await transcribe_audio(audio_file)

@app.post("/speech")
async def generate_speech(message: Message):
    try:
        print(f"Generating speech for: {message.text}")

        # Call the /generate_speech endpoint (correct one!)
        result = xtts_client.predict(
            input_text=message.text,
            speaker_reference_audio=handle_file("https://github.com/overtheskyy/iskobot-voice/raw/main/iskobot.wav"),
            enhance_speech=True,
            temperature=0.65,
            top_p=0.80,
            top_k=50,
            repetition_penalty=2.0,
            language="English",
            api_name="/generate_speech"
        )

        #Fix: unpack tuple if needed
        if isinstance(result, tuple):
            file_path = result[0]
        else:
            file_path = result

        # Return audio if valid
        if isinstance(file_path, str) and os.path.exists(file_path):
            with open(file_path, "rb") as f:
                audio_data = f.read()
            return Response(content=audio_data, media_type="audio/wav")
        else:
            print("Unexpected response from TTS client:", result)
            raise Exception("Speech generation failed or returned invalid file path.")

    except Exception as e:
        print("Error in /speech:", e)
        return Response(content=str(e), status_code=500)
    
    # TODO: add quota error handler

# Add routes for the chain
add_routes(app, chain)

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)