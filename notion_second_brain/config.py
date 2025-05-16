import os
from dotenv import load_dotenv, find_dotenv
import logging

# Detailed debug for .env loading
# print("DEBUG config.py: Attempting to find .env file...") # REMOVE
# dotenv_path = find_dotenv(usecwd=True) # REMOVE
# print(f"DEBUG config.py: dotenv_path found: {dotenv_path}") # REMOVE

# Load environment variables from .env file if it exists
# load_dotenv_status = load_dotenv(dotenv_path=dotenv_path if dotenv_path else None, override=True) # REMOVE
load_dotenv(override=True) # Simplify back, override was key
# print(f"DEBUG config.py: load_dotenv() status: {load_dotenv_status}") # REMOVE

# Check os.environ directly after load_dotenv for Pinecone keys
# print(f"DEBUG config.py: os.environ.get('PINECONE_API_KEY'): {os.environ.get('PINECONE_API_KEY')}") # REMOVE
# print(f"DEBUG config.py: os.environ.get('PINECONE_INDEX_NAME'): {os.environ.get('PINECONE_INDEX_NAME')}") # REMOVE
# print(f"DEBUG config.py: os.environ.get('OPENAI_API_KEY') is set: {bool(os.environ.get('OPENAI_API_KEY'))}") # REMOVE

NOTION_TOKEN = os.getenv("NOTION_TOKEN")
DATABASE_ID = os.getenv("NOTION_DATABASE_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Google Cloud Storage settings
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
GCS_EXPORT_PREFIX = os.getenv("GCS_EXPORT_PREFIX", "notion_exports") # Default prefix for exports
GCS_INDEX_ARTIFACTS_PREFIX = os.getenv("GCS_INDEX_ARTIFACTS_PREFIX", "index_artifacts") # Default prefix for index artifacts

# --- Logging Configuration ---
DEFAULT_LOG_FORMAT = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'

def setup_logging(level=logging.INFO):
    """Configures logging for the application."""
    formatter = logging.Formatter(DEFAULT_LOG_FORMAT)
    
    # Configure the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers (important if this function is called multiple times)
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
        
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Set requests/urllib3 logger level higher to avoid excessive debug messages
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    logging.info(f"Logging configured with level {logging.getLevelName(level)}")

# --- Config Value Checks (Keep these after setup_logging is defined) ---
if NOTION_TOKEN is None:
    # Use logging if available, otherwise raise error early
    try:
        logging.error("NOTION_TOKEN environment variable not set.")
    except NameError:
        pass # Logging might not be set up yet if config fails early
    raise ValueError("Error: NOTION_TOKEN environment variable not set. Please create a .env file or set the environment variable.")

if DATABASE_ID is None:
    try:
        logging.error("NOTION_DATABASE_ID environment variable not set.")
    except NameError:
        pass
    raise ValueError("Error: NOTION_DATABASE_ID environment variable not set. Please create a .env file or set the environment variable.")

# Note: We don't strictly *require* OPENAI_API_KEY for *all* operations (like export),
# so we don't raise an error here. Scripts using it (build_index, cli --query)
# should check for its existence themselves.
# if OPENAI_API_KEY is None:
#     try:
#         logging.warning("OPENAI_API_KEY environment variable not set. RAG features will fail.")
#     except NameError:
#         pass 