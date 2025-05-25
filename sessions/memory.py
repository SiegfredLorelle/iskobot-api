from typing import List
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from .service import SessionService
from .models import MessageRole
import uuid

class SessionMemory:
    """Custom memory class that integrates with database"""
    
    def __init__(self, session_service: SessionService, session_id: uuid.UUID, user_id: uuid.UUID, window_size: int = 10):
        self.session_service = session_service
        self.session_id = session_id
        self.user_id = user_id
        self.window_size = window_size
        self._messages: List[BaseMessage] = []
        self._loaded = False
    
    async def load_history(self):
        """Load recent messages from database"""
        if self._loaded:
            return
        
        messages = await self.session_service.get_session_messages(self.session_id, self.user_id)
        # Keep only recent messages based on window size
        recent_messages = messages[-self.window_size:] if len(messages) > self.window_size else messages
        
        self._messages = []
        for msg in recent_messages:
            if msg.role == MessageRole.USER:
                self._messages.append(HumanMessage(content=msg.content))
            else:
                self._messages.append(AIMessage(content=msg.content))
        
        self._loaded = True
    
    def get_context(self) -> str:
        """Get conversation context as string"""
        if not self._messages:
            return ""
        
        context_parts = []
        for msg in self._messages:
            role = "Human" if isinstance(msg, HumanMessage) else "Assistant"
            context_parts.append(f"{role}: {msg.content}")
        
        return "\n".join(context_parts)
    
    def clear(self):
        """Clear memory cache"""
        self._messages = []
        self._loaded = False
