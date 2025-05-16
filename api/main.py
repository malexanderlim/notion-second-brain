from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import List, Dict, Any, Optional, Union
import os
import logging # Import logging
import sys # Added for stderr
from dotenv import load_dotenv
from datetime import datetime, timezone # Added timezone
from contextlib import asynccontextmanager # Import asynccontextmanager

# --- Load Environment Variables First ---
load_dotenv()

# --- Vercel GCP Credentials Handling ---
# This block writes the GCP service account JSON from an environment variable
# to a temporary file, so GOOGLE_APPLICATION_CREDENTIALS can point to it.
GCP_SERVICE_ACCOUNT_JSON_CONTENT_ENV_VAR = "GCP_SERVICE_ACCOUNT_JSON_CONTENT"
# GOOGLE_APPLICATION_CREDENTIALS should be set in Vercel to e.g., /tmp/gcp-credentials.json
gcp_json_string = os.getenv(GCP_SERVICE_ACCOUNT_JSON_CONTENT_ENV_VAR)
google_creds_file_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

# Setup logger for the main application - BEFORE any potential logging in credential handling
# Ensure this is configured early so all logs are captured.
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper(),
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("api.main")

if gcp_json_string and google_creds_file_path:
    try:
        # Ensure the target directory for the credentials file exists
        os.makedirs(os.path.dirname(google_creds_file_path), exist_ok=True)
        with open(google_creds_file_path, 'w') as f:
            f.write(gcp_json_string)
        logger.info(f"Successfully wrote GCP service account credentials to {google_creds_file_path}")
    except Exception as e:
        logger.error(f"CRITICAL: Failed to write GCP service account credentials to {google_creds_file_path}: {e}", exc_info=True)
        # Depending on the desired behavior, you might want to exit or raise an error here
        # For now, it logs critical and continues, but GCS operations will likely fail.
elif not gcp_json_string:
    logger.warning(f"GCP_SERVICE_ACCOUNT_JSON_CONTENT environment variable not set. GCS operations may fail if credentials are not found elsewhere.")
elif not google_creds_file_path:
    logger.warning(f"GOOGLE_APPLICATION_CREDENTIALS environment variable not set to a file path. GCS operations may fail.")
# --- End Vercel GCP Credentials Handling ---

# --- Import RAG logic AFTER loading .env and setting up credentials/logging ---
# Ensure functions needed for initialization are imported
from .rag_query import perform_rag_query
# Import initialization functions directly from rag_initializer
from .rag_initializer import (
    load_rag_data, 
    initialize_openai_client, 
    initialize_anthropic_client,
    initialize_pinecone_client,
    get_openai_client,
    get_anthropic_client,
    get_last_sync_timestamp,
    RAGSystemNotInitializedError,
    LLMClientNotInitializedError
)

# --- FastAPI Lifespan Manager for Startup/Shutdown ---
@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    logger.critical("LIFESPAN FUNCTION ENTERED!")
    # Code here runs on startup
    logger.info("Application startup: Initializing RAG system and LLM clients...")
    
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

    openai_key_snippet = f"'{OPENAI_API_KEY[:5]}...'" if OPENAI_API_KEY else "None"
    anthropic_key_snippet = f"'{ANTHROPIC_API_KEY[:5]}...'" if ANTHROPIC_API_KEY else "None"
    pinecone_key_snippet = f"'{PINECONE_API_KEY[:5]}...'" if PINECONE_API_KEY else "None"

    logger.info(f"Lifespan: Attempting to initialize with OPENAI_API_KEY: {openai_key_snippet}")
    logger.info(f"Lifespan: Attempting to initialize with ANTHROPIC_API_KEY: {anthropic_key_snippet}")
    logger.info(f"Lifespan: Attempting to initialize with PINECONE_API_KEY: {pinecone_key_snippet}, Index: {PINECONE_INDEX_NAME}")

    try:
        logger.info("Lifespan: Loading RAG data...")
        load_rag_data() # Depends on GCS creds being set up before lifespan starts
        logger.info("Lifespan: Initializing OpenAI client...")
        initialize_openai_client(OPENAI_API_KEY)
        logger.info("Lifespan: Initializing Anthropic client...")
        initialize_anthropic_client(ANTHROPIC_API_KEY)
        logger.info("Lifespan: Initializing Pinecone client...")
        initialize_pinecone_client(PINECONE_API_KEY, PINECONE_INDEX_NAME)
        logger.info("Application startup: RAG system and LLM clients initialized successfully.")
    except Exception as e:
        logger.critical(f"Application startup: Failed to initialize RAG system or LLM clients: {e}", exc_info=True)
        # Optionally, re-raise or handle to prevent app from starting in a bad state
        # raise SystemExit(f"Failed to initialize critical components: {e}") from e
    
    yield
    
    # Code here runs on shutdown (optional)
    logger.info("Application shutdown: Cleaning up resources (if any)...")

# --- FastAPI App Instance with Lifespan Manager ---
logger.info("API_MAIN: About to instantiate FastAPI app.")
try:
    app = FastAPI(
        title="Notion Second Brain API",
        description="API for querying the Notion Second Brain RAG index.",
        version="0.1.0",
        lifespan=lifespan # Assign the lifespan manager
    )
    logger.info("API_MAIN: FastAPI app instantiated successfully.")
except Exception as e_app_create:
    logger.critical(f"API_MAIN: FAILED TO INSTANTIATE FastAPI app: {e_app_create}", exc_info=True)
    # Optionally raise to ensure Vercel sees a startup failure
    # raise

# --- Configure CORS --- (Essential for frontend interaction)
# Adjust origins based on your frontend development/production URLs
origins = [
    "http://localhost:5173",  # Default Vite dev server port
    "http://localhost:3000",  # Common alternative dev port
    # Add your frontend production URL here later, e.g.:
    # "https://your-frontend-app.vercel.app", 
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, # Allows specified origins
    allow_credentials=True, # Allows cookies/auth headers
    allow_methods=["POST", "GET", "OPTIONS"], # Allow only needed methods
    allow_headers=["Content-Type"], # Allow only necessary headers
)

# --- Pydantic Models for API Data Structure ---

class QueryRequest(BaseModel):
    query: str
    model_name: Optional[str] = None

class SourceDocument(BaseModel):
    title: str
    url: str
    id: Optional[str] = None
    date: Optional[str] = None
    score: Optional[float] = None

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceDocument]
    model_used: Optional[str] = None
    model_api_id_used: Optional[str] = None
    model_provider_used: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    estimated_cost_usd: Optional[float] = None

class LastUpdatedResponse(BaseModel):
    last_updated_timestamp: Optional[str] = None
    error: Optional[str] = None

class TranscriptionResponse(BaseModel):
    transcription: str
    error: Optional[str] = None

# --- API Endpoints ---

@app.get("/", tags=["General"])
def read_root():
    """Root endpoint providing basic API info."""
    return {"message": "Welcome to the Notion Second Brain API"}

@app.post("/api/query", response_model=QueryResponse, tags=["RAG"])
async def handle_query(request: QueryRequest):
    """Receives a user query, performs RAG, and returns the answer with sources."""
    user_query = request.query
    model_name = request.model_name # Extract model_name
    logger.info(f"Received query via API: '{user_query[:50]}...' using model: {model_name if model_name else 'default'}") # Log received query and model
    if not user_query:
        logger.warning("Received empty query.")
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        # --- Call the Refactored RAG Logic --- 
        rag_result = await perform_rag_query(user_query, model_name=model_name)
        
        # Extract results safely
        answer = rag_result.get("answer", "Error: Could not retrieve answer from RAG process.")
        sources_data = rag_result.get("sources", [])
        model_used = rag_result.get("model_used")
        model_api_id_used = rag_result.get("model_api_id_used")
        model_provider_used = rag_result.get("model_provider_used")
        input_tokens = rag_result.get("input_tokens")
        output_tokens = rag_result.get("output_tokens")
        estimated_cost_usd = rag_result.get("estimated_cost_usd")
        
        # Validate and structure the sources
        sources = []
        for source in sources_data:
            if isinstance(source, dict) and 'title' in source and 'url' in source:
                 try:
                     sources.append(SourceDocument(**source))
                 except Exception as pydantic_error: # Catch potential Pydantic validation errors
                     logger.warning(f"Skipping source due to validation error: {source}. Error: {pydantic_error}")
            else:
                logger.warning(f"Skipping invalid source format: {source}")

        logger.info(f"Successfully processed query with model: {model_used}. Returning answer and {len(sources)} sources.")
        return QueryResponse(
            answer=answer, 
            sources=sources, 
            model_used=model_used,
            model_api_id_used=model_api_id_used,
            model_provider_used=model_provider_used,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=estimated_cost_usd
        )

    except Exception as e:
        # Use logger for exceptions
        logger.error(f"Error processing query '{user_query[:50]}...': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal server error occurred while processing the query.")

@app.get("/api/last-updated", response_model=LastUpdatedResponse, tags=["General"])
def get_last_updated_timestamp():
    """Retrieves the timestamp of the last successfully processed entry from the index build process."""
    logger.info("API_MAIN: get_last_updated_timestamp CALLED.") # New Log
    
    # DATA_SOURCE_MODE logging is helpful but the core logic now relies on rag_initializer
    # data_source_mode = os.getenv('DATA_SOURCE_MODE', 'local').lower()
    # logger.info(f"API_MAIN: DATA_SOURCE_MODE is '{data_source_mode}'.")

    # Get timestamp from rag_initializer
    timestamp_str = get_last_sync_timestamp() # MODIFIED

    if timestamp_str is None:
        logger.warning(f"API_MAIN: Last sync timestamp not available from rag_initializer.")
        return LastUpdatedResponse(last_updated_timestamp=None, error="Timestamp information not available. Sync may not have run or file not found/readable.")
    
    if not timestamp_str: # Check if it's an empty string
        logger.warning(f"API_MAIN: Last sync timestamp from rag_initializer is empty.")
        return LastUpdatedResponse(last_updated_timestamp=None, error="Timestamp information is empty.")

    # Basic validation (could be more robust, e.g., regex or parse with datetime)
    # datetime.fromisoformat(timestamp_str) # Uncomment to validate format strictly
    logger.info(f"Successfully retrieved timestamp via rag_initializer: {timestamp_str}")
    return LastUpdatedResponse(last_updated_timestamp=timestamp_str)

@app.post("/api/transcribe", response_model=TranscriptionResponse, tags=["Transcription"])
async def transcribe_audio(file: UploadFile = File(...)):
    """Receives an audio file and returns its transcription using OpenAI Whisper."""
    logger.info(f"Received audio file for transcription: {file.filename}, content type: {file.content_type}")

    openai_client = get_openai_client() # Get the initialized client
    if not openai_client:
        logger.error("OpenAI client not initialized. Cannot transcribe audio.")
        raise HTTPException(status_code=500, detail="OpenAI client not available. Transcription service is down.")

    try:
        # Ensure the file pointer is at the beginning of the file stream
        await file.seek(0)
        file_bytes = await file.read()

        transcription_result = openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=(file.filename, file_bytes)
        )
        
        transcribed_text = transcription_result.text
        logger.info(f"Successfully transcribed audio file: {file.filename}. Transcription length: {len(transcribed_text)}")
        return TranscriptionResponse(transcription=transcribed_text)
    except Exception as e:
        logger.error(f"Error during audio transcription for file {file.filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to transcribe audio: {str(e)}")

# --- Optional: Add command to run the server easily ---
# uvicorn backend.main:app --reload --port 8000

# Remove placeholder asyncio import if no longer needed
# import asyncio 