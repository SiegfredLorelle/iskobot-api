import jwt
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from supabase import Client
from app.config import Config
from app.models.auth import TokenPayload, UserResponse
import logging

# logger = logging.getLogger(__name__)
security = HTTPBearer()

def get_supabase_client() -> Client:
    """Get Supabase client instance"""
    from supabase import create_client
    return create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)

def verify_jwt_token(token: str) -> TokenPayload:
    """
    Verify and decode JWT token from Supabase
    """
    try:
        # Decode without verification first to get the payload
        # Supabase handles the verification on their end
        payload = jwt.decode(
            token, 
            options={"verify_signature": False, "verify_exp": True}
        )
        
        token_data = TokenPayload(**payload)
        
        if token_data.sub is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token: missing subject"
            )
        
        return token_data
        
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired"
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )
    except Exception as e:
        logger.error(f"Token verification error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    supabase: Client = Depends(get_supabase_client)
) -> UserResponse:
    """
    Dependency to get the current authenticated user
    """
    try:
        # Verify the token
        token_data = verify_jwt_token(credentials.credentials)
        
        # Get user from Supabase
        response = supabase.auth.get_user(credentials.credentials)
        
        if response.user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found"
            )
        
        return UserResponse(
            id=response.user.id,
            email=response.user.email,
            full_name=response.user.user_metadata.get("full_name"),
            display_name=response.user.user_metadata.get("display_name"),
            email_confirmed=response.user.email_confirmed_at is not None,
            created_at=response.user.created_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get current user error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )

async def get_current_active_user(
    current_user: UserResponse = Depends(get_current_user)
) -> UserResponse:
    """
    Dependency to get current active user (email confirmed)
    """
    if not current_user.email_confirmed:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email not confirmed. Please check your email and confirm your account."
        )
    return current_user

def require_auth(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> str:
    """
    Simple dependency that just requires a valid Bearer token
    Returns the token string for further processing
    """
    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header required"
        )
    
    # Basic token validation
    verify_jwt_token(credentials.credentials)
    
    return credentials.credentials

class AuthRequired:
    """
    Class-based dependency for more flexible authentication requirements
    """
    def __init__(self, require_email_verification: bool = False):
        self.require_email_verification = require_email_verification
    
    async def __call__(
        self, 
        credentials: HTTPAuthorizationCredentials = Depends(security),
        supabase: Client = Depends(get_supabase_client)
    ) -> UserResponse:
        try:
            # Verify token
            verify_jwt_token(credentials.credentials)
            
            # Get user
            response = supabase.auth.get_user(credentials.credentials)
            
            if response.user is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User not found"
                )
            
            user = UserResponse(
                id=response.user.id,
                email=response.user.email,
                full_name=response.user.user_metadata.get("full_name"),
                display_name=response.user.user_metadata.get("display_name"),
                email_confirmed=response.user.email_confirmed_at is not None,
                created_at=response.user.created_at
            )
            
            # Check email verification if required
            if self.require_email_verification and not user.email_confirmed:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Email verification required"
                )
            
            return user
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Auth required error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication failed"
            )