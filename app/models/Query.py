from pydantic import BaseModel

class QueryResponse(BaseModel):
    response: str

class QueryRequest(BaseModel):
    query: str

class Query(BaseModel):
    query: str
    response: str