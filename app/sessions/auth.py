from fastapi import Header, HTTPException, Depends
import uuid
from typing import Optional

class AuthService:
    def __init__(self, supabase_client):
        self.supabase = supabase_client
    
    async def get_current_user(self, authorization: Optional[str] = Header(None)) -> uuid.UUID:
        """Extract user ID from JWT token"""
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(
                status_code=401, 
                detail="Missing or invalid authorization header"
            )
        
        token = authorization.split(" ")[1]
        try:
            # Verify JWT token with Supabase
            user = self.supabase.auth.get_user(token)
            return uuid.UUID(user.user.id)
        except Exception as e:
            raise HTTPException(status_code=401, detail="Invalid token")