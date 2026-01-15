from fastapi import HTTPException, Security, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from ..config import settings
from typing import Optional

# We keep HTTPBearer for Swagger UI support, but we make it optional
security = HTTPBearer(auto_error=False)

def verify_token(request: Request, bearer: Optional[HTTPAuthorizationCredentials] = Security(security)):
    """
    Verifies the JWT token from:
    1. The 'token' HttpOnly cookie (Preferred)
    2. The Authorization header (Fallback)
    """
    token = None
    
    # 1. Try Cookie
    if "token" in request.cookies:
        token = request.cookies["token"]
    
    # 2. Try Header (Fallback)
    if not token and bearer:
        token = bearer.credentials
        
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM])
        
        # Security Check: Prevent usage of temporary 2FA tokens
        if payload.get("isTemp") is True:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Full authentication required (2FA not completed)",
            )
            
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )