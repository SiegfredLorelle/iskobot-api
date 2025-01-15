from fastapi import FastAPI, UploadFile, File
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

app = FastAPI()

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

# Add routes for the chain
add_routes(app, chain)

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)