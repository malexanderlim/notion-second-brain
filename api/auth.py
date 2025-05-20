import os
from authlib.integrations.starlette_client import OAuth
from dotenv import load_dotenv
from fastapi import Request, HTTPException, status

# Load environment variables from .env file
load_dotenv()

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
AUTHORIZED_USER_EMAIL = os.getenv("AUTHORIZED_USER_EMAIL") # We'll use this later in the callback
SESSION_SECRET_KEY = os.getenv("SESSION_SECRET_KEY") # We'll use this for session middleware

if not all([GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, SESSION_SECRET_KEY]):
    raise ValueError(
        "Missing critical OAuth or Session environment variables: "
        "GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, SESSION_SECRET_KEY must be set."
    )

# It's good practice to also check for AUTHORIZED_USER_EMAIL if your app logic strictly depends on it from the start.
# For now, we'll assume it might be checked more specifically where applied.
# if not AUTHORIZED_USER_EMAIL:
#     print("Warning: AUTHORIZED_USER_EMAIL is not set. Access control might be open or default to deny.")


oauth = OAuth()

oauth.register(
    name='google',
    client_id=GOOGLE_CLIENT_ID,
    client_secret=GOOGLE_CLIENT_SECRET,
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={
        'scope': 'openid email profile'
    }
)

# We will add more functions here later, e.g., for login, callback, logout, and getting current user.

# --- FastAPI Dependency for Protected Routes ---
async def get_current_active_user(request: Request):
    """
    Dependency to get the current active user from the session.
    Raises HTTPException if the user is not authenticated.
    """
    user = request.session.get("user")
    if not user:
        # It might be useful to log this attempt or provide a more specific error code/detail
        # if different unauthenticated scenarios should be handled differently by the client.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"}, # Though not strictly Bearer, it's a common hint
        )
    # We could also re-verify if user_email matches AUTHORIZED_USER_EMAIL here if we want to be extremely paranoid
    # or if the session could somehow be set without passing through our callback logic (unlikely with HTTPOnly cookies).
    # For now, trusting the session if 'user' is present, as it's set upon successful auth.
    return user # The route will receive this user dict 