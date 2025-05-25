from fastapi import APIRouter, Depends, HTTPException
from typing import List
import uuid
from .models import Session, Message, SessionCreate, QueryWithSession, ChatResponse
from .service import SessionService
from .auth import AuthService
from .memory import SessionMemory

def create_sessions_router(supabase_client, chain, retry_with_backoff):
    """Factory function to create the sessions router with dependencies"""
    router = APIRouter(prefix="/sessions", tags=["sessions"])
    
    # Create service instances with injected dependencies
    session_service = SessionService(supabase_client)
    auth_service = AuthService(supabase_client)

    @router.post("/", response_model=Session)
    async def create_session(
        session_data: SessionCreate,
        user_id: uuid.UUID = Depends(auth_service.get_current_user)
    ):
        """Create a new chat session"""
        try:
            session = await session_service.create_session(user_id, session_data.title)
            return session
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/", response_model=List[Session])
    async def get_sessions(
        user_id: uuid.UUID = Depends(auth_service.get_current_user),
        limit: int = 50
    ):
        """Get user's chat sessions"""
        try:
            sessions = await session_service.get_user_sessions(user_id, limit)
            return sessions
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/{session_id}/messages", response_model=List[Message])
    async def get_session_messages(
        session_id: uuid.UUID,
        user_id: uuid.UUID = Depends(auth_service.get_current_user)
    ):
        """Get messages for a specific session"""
        try:
            messages = await session_service.get_session_messages(session_id, user_id)
            return messages
        except Exception as e:
            if "not found" in str(e).lower():
                raise HTTPException(status_code=404, detail="Session not found")
            raise HTTPException(status_code=500, detail=str(e))

    @router.put("/{session_id}/title")
    async def update_session_title(
        session_id: uuid.UUID,
        title_data: dict,  # Accept JSON body instead of query parameter
        user_id: uuid.UUID = Depends(auth_service.get_current_user)
    ):
        """Update session title"""
        try:
            title = title_data.get("title")
            if not title:
                raise HTTPException(status_code=400, detail="Title is required")
            
            success = await session_service.update_session_title(session_id, user_id, title)
            if not success:
                raise HTTPException(status_code=404, detail="Session not found")
            return {"message": "Title updated successfully"}
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.delete("/{session_id}")
    async def delete_session(
        session_id: uuid.UUID,
        user_id: uuid.UUID = Depends(auth_service.get_current_user)
    ):
        """Delete a session"""
        try:
            success = await session_service.delete_session(session_id, user_id)
            if not success:
                raise HTTPException(status_code=404, detail="Session not found")
            return {"message": "Session deleted successfully"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return router