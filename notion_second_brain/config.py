import os
from dotenv import load_dotenv
import logging

# Load environment variables from .env file if it exists
load_dotenv()

NOTION_TOKEN = os.getenv("NOTION_TOKEN")
DATABASE_ID = os.getenv("NOTION_DATABASE_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

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