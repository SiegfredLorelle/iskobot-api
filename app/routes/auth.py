from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
from supabase import Client
from app.config import Config
from app.models.auth import (
    SignUpRequest, 
    SignInRequest, 
    AuthResponse, 
    UserResponse,
    PasswordResetRequest,
    PasswordUpdateRequest
)
from app.utils.auth_utils import verify_jwt_token
import logging

# Initialize router
router = APIRouter(prefix="/auth", tags=["Authentication"])

# Security scheme
security = HTTPBearer()

logger = logging.getLogger(__name__)

def get_supabase_client() -> Client:
    """Dependency to get Supabase client"""
    from supabase import create_client
    return create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)

@router.post("/signup", response_model=AuthResponse)
async def sign_up(
    request: SignUpRequest,
    supabase: Client = Depends(get_supabase_client)
):
    """Register a new user"""
    try:
        # Sign up user with Supabase Auth
        response = supabase.auth.sign_up({
            "email": request.email,
            "password": request.password,
            "options": {
                "data": {
                    "full_name": request.full_name,
                    "display_name": request.display_name
                }
            }
        })
        
        if response.user is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to create user account"
            )
        
        return AuthResponse(
            access_token=response.session.access_token if response.session else None,
            refresh_token=response.session.refresh_token if response.session else None,
            user=UserResponse(
                id=response.user.id,
                email=response.user.email,
                full_name=response.user.user_metadata.get("full_name"),
                display_name=response.user.user_metadata.get("display_name"),
                email_confirmed=response.user.email_confirmed_at is not None,
                created_at=response.user.created_at
            ),
            message="User created successfully. Please check your email for verification."
        )
        
    except Exception as e:
        logger.error(f"Sign up error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Registration failed: {str(e)}"
        )

@router.post("/signin", response_model=AuthResponse)
async def sign_in(
    request: SignInRequest,
    supabase: Client = Depends(get_supabase_client)
):
    """Sign in an existing user"""
    try:
        response = supabase.auth.sign_in_with_password({
            "email": request.email,
            "password": request.password
        })
        
        if response.user is None or response.session is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )
        
        return AuthResponse(
            access_token=response.session.access_token,
            refresh_token=response.session.refresh_token,
            user=UserResponse(
                id=response.user.id,
                email=response.user.email,
                full_name=response.user.user_metadata.get("full_name"),
                display_name=response.user.user_metadata.get("display_name"),
                email_confirmed=response.user.email_confirmed_at is not None,
                created_at=response.user.created_at
            ),
            message="Sign in successful"
        )
        
    except Exception as e:
        logger.error(f"Sign in error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )

@router.post("/signout")
async def sign_out(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    supabase: Client = Depends(get_supabase_client)
):
    """Sign out the current user"""
    try:
        # Set the session token for the request
        supabase.auth.set_session(credentials.credentials, "")
        
        # Sign out
        supabase.auth.sign_out()
        
        return {"message": "Successfully signed out"}
        
    except Exception as e:
        logger.error(f"Sign out error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Sign out failed"
        )

@router.post("/refresh")
async def refresh_token(
    request: dict,
    supabase: Client = Depends(get_supabase_client)
):
    """Refresh access token using refresh token"""
    try:
        refresh_token = request.get("refresh_token")
        if not refresh_token:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Refresh token is required"
            )
        
        response = supabase.auth.refresh_session(refresh_token)
        
        if response.session is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )
        
        return {
            "access_token": response.session.access_token,
            "refresh_token": response.session.refresh_token,
            "expires_in": response.session.expires_in
        }
        
    except Exception as e:
        logger.error(f"Token refresh error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token refresh failed"
        )

@router.get("/me", response_model=UserResponse)
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    supabase: Client = Depends(get_supabase_client)
):
    """Get current user information"""
    try:
        # Verify and decode the JWT token
        user_data = verify_jwt_token(credentials.credentials)
        
        # Get user from Supabase
        response = supabase.auth.get_user(credentials.credentials)
        
        if response.user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token"
            )
        
        return UserResponse(
            id=response.user.id,
            email=response.user.email,
            full_name=response.user.user_metadata.get("full_name"),
            display_name=response.user.user_metadata.get("display_name"),
            email_confirmed=response.user.email_confirmed_at is not None,
            created_at=response.user.created_at
        )
        
    except Exception as e:
        logger.error(f"Get user error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )

@router.post("/forgot-password")
async def forgot_password(
    request: PasswordResetRequest,
    supabase: Client = Depends(get_supabase_client)
):
    """Send password reset email"""
    try:
        supabase.auth.reset_password_email(request.email)
        
        return {"message": "Password reset email sent successfully"}
        
    except Exception as e:
        logger.error(f"Password reset error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to send password reset email"
        )

@router.post("/update-password")
async def update_password(
    request: PasswordUpdateRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    supabase: Client = Depends(get_supabase_client)
):
    """Update user password"""
    try:
        # Set the session token
        supabase.auth.set_session(credentials.credentials, "")
        
        # Update password
        response = supabase.auth.update_user({
            "password": request.new_password
        })
        
        if response.user is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to update password"
            )
        
        return {"message": "Password updated successfully"}
        
    except Exception as e:
        logger.error(f"Password update error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to update password"
        )

@router.post("/resend-confirmation")
async def resend_confirmation(
    request: dict,
    supabase: Client = Depends(get_supabase_client)
):
    """Resend email confirmation"""
    try:
        email = request.get("email")
        if not email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email is required"
            )
        
        supabase.auth.resend(type="signup", email=email)
        
        return {"message": "Confirmation email sent successfully"}
        
    except Exception as e:
        logger.error(f"Resend confirmation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to resend confirmation email"
        )