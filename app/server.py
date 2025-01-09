from fastapi import FastAPI
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

app = FastAPI()

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


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/playground")

@app.post("/query", response_model=QueryRequest)
async def get_answers_from_query(request: QueryRequest):
    answer = await chain.ainvoke(request.query)
    response = QueryResponse(response=answer)
    return JSONResponse(content=response.dict())


add_routes(app, chain)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)