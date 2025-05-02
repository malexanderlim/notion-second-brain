from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import os
import logging # Import logging
from dotenv import load_dotenv

# Refactor and import RAG logic
from backend.rag_query import perform_rag_query # Adjusted import path

load_dotenv()

# Setup logger for the main application
logger = logging.getLogger(__name__)
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper(), 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

app = FastAPI(
    title="Notion Second Brain API",
    description="API for querying the Notion Second Brain RAG index.",
    version="0.1.0"
)

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
    allow_methods=["POST", "GET"], # Allow only needed methods (e.g., POST for query)
    allow_headers=["Content-Type"], # Allow only necessary headers
)

# --- Pydantic Models for API Data Structure ---

class QueryRequest(BaseModel):
    query: str

class SourceDocument(BaseModel):
    title: str
    url: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceDocument]

# --- API Endpoints ---

@app.get("/", tags=["General"])
def read_root():
    """Root endpoint providing basic API info."""
    return {"message": "Welcome to the Notion Second Brain API"}

@app.post("/api/query", response_model=QueryResponse, tags=["RAG"])
async def handle_query(request: QueryRequest):
    """Receives a user query, performs RAG, and returns the answer with sources."""
    user_query = request.query
    logger.info(f"Received query via API: '{user_query[:50]}...'") # Log received query
    if not user_query:
        logger.warning("Received empty query.")
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        # --- Call the Refactored RAG Logic --- 
        rag_result = await perform_rag_query(user_query)
        
        # Extract results safely
        answer = rag_result.get("answer", "Error: Could not retrieve answer from RAG process.")
        sources_data = rag_result.get("sources", [])
        
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

        logger.info(f"Successfully processed query. Returning answer and {len(sources)} sources.")
        return QueryResponse(answer=answer, sources=sources)

    except Exception as e:
        # Use logger for exceptions
        logger.error(f"Error processing query '{user_query[:50]}...': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal server error occurred while processing the query.")

# --- Optional: Add command to run the server easily ---
# uvicorn backend.main:app --reload --port 8000

# Remove placeholder asyncio import if no longer needed
# import asyncio 