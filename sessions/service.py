from typing import List, Optional
import uuid
from .models import Session, Message, MessageRole, SessionCreate
from datetime import datetime
from fastapi import HTTPException

class SessionService:
    def __init__(self, supabase_client):
        self.supabase = supabase_client

    async def create_session(self, user_id: Optional[uuid.UUID], title: Optional[str] = None) -> Session:
        """Create a new chat session (authenticated or anonymous)"""
        session_data = {
            "title": title or "New Chat",
            "is_active": True,
            "user_id": str(user_id) if user_id else None
        }
        
        result = self.supabase.table("sessions").insert(session_data).execute()
        if not result.data:
            raise HTTPException(status_code=500, detail="Failed to create session")
        
        return Session(**result.data[0])

    async def get_session(self, session_id: uuid.UUID, user_id: Optional[uuid.UUID] = None) -> Optional[Session]:
        """Get a session with optional user validation"""
        query = self.supabase.table("sessions").select("*").eq("id", str(session_id))
        
        if user_id:
            query = query.eq("user_id", str(user_id))
        
        result = query.execute()
        if not result.data:
            return None
        
        return Session(**result.data[0])

    async def get_user_sessions(self, user_id: uuid.UUID, limit: int = 50) -> List[Session]:
        """Get all sessions for an authenticated user"""
        result = (
            self.supabase.table("sessions")
            .select("*, messages!inner(*)")
            .eq("user_id", str(user_id))
            .eq("is_active", True)
            .order("updated_at", desc=True)
            .limit(limit)
            .execute()
        )
        
        return [self._format_session(session_data) for session_data in result.data]

    async def get_session_messages(self, session_id: uuid.UUID, user_id: Optional[uuid.UUID] = None) -> List[Message]:
        """Get messages for a session (with optional user validation)"""
        # Verify session access
        session = await self.get_session(session_id, user_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        result = (
            self.supabase.table("messages")
            .select("*")
            .eq("session_id", str(session_id))
            .order("created_at")
            .execute()
        )
        
        return [Message(**msg) for msg in result.data]

    async def add_message(self, session_id: uuid.UUID, role: MessageRole, content: str) -> Message:
        """Add a message to a session"""
        message_data = {
            "session_id": str(session_id),
            "role": role.value,
            "content": content
        }
        
        result = self.supabase.table("messages").insert(message_data).execute()
        if not result.data:
            raise HTTPException(status_code=500, detail="Failed to add message")

        # Update session timestamp
        self.supabase.table("sessions").update({
            "updated_at": datetime.utcnow().isoformat()
        }).eq("id", str(session_id)).execute()
        
        return Message(**result.data[0])

    async def update_session_title(self, session_id: uuid.UUID, title: str, user_id: Optional[uuid.UUID] = None) -> bool:
        """Update session title (authenticated users only)"""
        if not user_id:
            raise HTTPException(status_code=403, detail="Anonymous sessions cannot be modified")

        result = (
            self.supabase.table("sessions")
            .update({"title": title})
            .eq("id", str(session_id))
            .eq("user_id", str(user_id))
            .execute()
        )
        
        return bool(result.data)

    async def delete_session(self, session_id: uuid.UUID, user_id: Optional[uuid.UUID] = None) -> bool:
        """Delete a session (authenticated users only)"""
        if not user_id:
            raise HTTPException(status_code=403, detail="Anonymous sessions cannot be deleted")

        result = (
            self.supabase.table("sessions")
            .update({"is_active": False})
            .eq("id", str(session_id))
            .eq("user_id", str(user_id))
            .execute()
        )
        
        return bool(result.data)

    def _format_session(self, session_data: dict) -> Session:
        """Format session data with message counts"""
        messages = session_data.get("messages", [])
        return Session(
            id=session_data["id"],
            user_id=session_data["user_id"],
            title=session_data["title"],
            created_at=session_data["created_at"],
            updated_at=session_data["updated_at"],
            is_active=session_data.get("is_active", True),
            message_count=len(messages),
            last_message=next(
                (msg["content"] for msg in reversed(messages) 
                if msg["role"] == "user"),  # Matches your existing logic
                None
            )
        )