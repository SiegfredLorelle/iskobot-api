from pydantic import BaseModel

class QueryResponse(BaseModel):
    response: str

class QueryRequest(BaseModel):
    query: str
    thread_id: str | None = None

class Query(BaseModel):
    query: str
    response: str