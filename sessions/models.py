from pydantic import BaseModel
from typing import List, Optional
import uuid
from datetime import datetime
from enum import Enum

class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"

class MessageCreate(BaseModel):
    content: str
    metadata: Optional[dict] = {}

class Message(BaseModel):
    id: uuid.UUID
    session_id: uuid.UUID
    role: MessageRole
    content: str
    metadata: dict
    created_at: datetime

class SessionCreate(BaseModel):
    title: Optional[str] = None

class Session(BaseModel):
    id: uuid.UUID
    user_id: Optional[uuid.UUID] = None
    title: Optional[str]
    created_at: datetime
    updated_at: datetime
    is_active: bool
    message_count: Optional[int] = 0
    last_message: Optional[str] = None

class QueryWithSession(BaseModel):
    query: str
    session_id: Optional[uuid.UUID] = None

class ChatResponse(BaseModel):
    response: str
    session_id: uuid.UUID
    message_id: uuid.UUID
