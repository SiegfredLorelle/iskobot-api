from fastapi import FastAPI
from fastapi.responses import RedirectResponse, JSONResponse
from langserve import add_routes
from langchain_google_vertexai import VertexAI
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from app.database.vectorstore import initialize_vectorstore
from app.models.query_request import QueryRequest
from app.models.query_response import QueryResponse

from dotenv import load_dotenv

app = FastAPI()

# (1) Initialize VectorStore
vectorstore = initialize_vectorstore()

# (2) Build retriever


def concatenate_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


notes_retriever = vectorstore.as_retriever() | concatenate_docs

# (3) Create prompt template
prompt_template = PromptTemplate.from_template(
    """You are a Cloud Run expert answering questions. 
Use the retrieved release notes to answer questions
Give a concise answer, and if you are unsure of the answer, just say so.

Release notes: {notes}

Here is your question: {query}
Your answer: """)

# (4) Initialize LLM
llm = VertexAI(
    model_name="gemini-1.0-pro-002",
    temperature=0.2,
    max_output_tokens=100,
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
    response = QueryResponse(answer=answer)
    return JSONResponse(content=response.dict())


add_routes(app, chain)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
