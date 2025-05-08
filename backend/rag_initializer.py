"""Manages shared RAG state (data, clients)."""
import faiss
import json
import os
import logging
from datetime import date

from openai import OpenAI
try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

# Import configuration constants
from backend.rag_config import (
    INDEX_PATH,
    MAPPING_PATH,
    METADATA_CACHE_PATH,
    DATABASE_SCHEMA_PATH
)

logger = logging.getLogger(__name__)

# --- Globals managed by this initializer module ---
# These will be imported and used by the main RAG logic (rag_query.py)
openai_client = None
anthropic_client = None
index = None
mapping_data_list: list[dict] | None = None
index_to_entry: dict[int, dict] | None = None
distinct_metadata_values: dict | None = None
schema_properties: dict | None = None

# Flag to prevent redundant loading
_rag_data_loaded = False

def load_rag_data():
    """Loads necessary data for RAG: index, mapping (list), metadata cache, schema.
       Uses a flag to prevent reloading if already loaded.
    """
    global index, mapping_data_list, index_to_entry, distinct_metadata_values, schema_properties
    global _rag_data_loaded
    
    if _rag_data_loaded:
        logger.debug("RAG data already loaded. Skipping reload.")
        return

    logger.info("--- Starting RAG Data Loading --- ")
    try:
        logger.info(f"Loading FAISS index from {INDEX_PATH}...")
        index = faiss.read_index(INDEX_PATH)
        logger.info(f"Index loaded successfully. Total vectors: {index.ntotal}")
    except Exception as e:
        logger.error(f"Fatal: Failed to load FAISS index from {INDEX_PATH}: {e}", exc_info=True)
        raise RuntimeError(f"Could not load FAISS index from {INDEX_PATH}") from e

    try:
        logger.info(f"Loading index mapping list from {MAPPING_PATH}...")
        with open(MAPPING_PATH, 'r') as f:
            loaded_mapping_data = json.load(f) # Load into temporary var first
            if not isinstance(loaded_mapping_data, list):
                raise TypeError(f"{MAPPING_PATH} does not contain a JSON list.")
        mapping_data_list = loaded_mapping_data # Assign to global var
        logger.info(f"Mapping list loaded successfully. Total entries: {len(mapping_data_list)}")

        if index and len(mapping_data_list) != index.ntotal:
            logger.warning(f"Mismatch: FAISS index has {index.ntotal} vectors, but mapping list has {len(mapping_data_list)} entries. Assuming list order corresponds to index order, but this could cause issues.")

        temp_index_to_entry = {} # Process into temporary var
        for i, entry in enumerate(mapping_data_list):
            if not isinstance(entry, dict):
                logger.warning(f"Skipping non-dictionary item at index {i} in mapping list: {entry}")
                continue
            
            entry['faiss_index'] = i 
            
            if 'entry_date' in entry and isinstance(entry['entry_date'], str):
                try:
                    entry['entry_date'] = date.fromisoformat(entry['entry_date'])
                except ValueError:
                    page_id_for_log = entry.get('page_id', f'index {i}')
                    logger.warning(f"Could not parse date string: {entry['entry_date']} for entry {page_id_for_log}")
                    entry['entry_date'] = None
            
            temp_index_to_entry[i] = entry
        index_to_entry = temp_index_to_entry # Assign to global var
        logger.info(f"Processed mapping list into index_to_entry lookup. Size: {len(index_to_entry)}")
            
    except FileNotFoundError:
        logger.error(f"Fatal: Mapping file not found at {MAPPING_PATH}")
        raise RuntimeError(f"Mapping file not found at {MAPPING_PATH}")
    except (json.JSONDecodeError, TypeError) as e:
        logger.error(f"Fatal: Failed to decode or process JSON list from {MAPPING_PATH}: {e}", exc_info=True)
        raise RuntimeError(f"Failed to decode or process {MAPPING_PATH}") from e
    except Exception as e:
        logger.error(f"Fatal: An unexpected error occurred loading/processing mapping from {MAPPING_PATH}: {e}", exc_info=True)
        raise RuntimeError(f"Unexpected error loading/processing {MAPPING_PATH}") from e

    temp_distinct_metadata = {}
    try:
        logger.info(f"Loading distinct metadata values from {METADATA_CACHE_PATH}...")
        if os.path.exists(METADATA_CACHE_PATH):
            with open(METADATA_CACHE_PATH, 'r') as f:
                cache_content = json.load(f)
                if "distinct_values" in cache_content and isinstance(cache_content["distinct_values"], dict):
                    loaded_distinct = cache_content["distinct_values"]
                    for key in loaded_distinct:
                        if isinstance(loaded_distinct[key], list):
                             temp_distinct_metadata[key] = set(loaded_distinct[key])
                        else:
                             temp_distinct_metadata[key] = loaded_distinct[key] # Keep as is if not list
                    logger.info("Distinct metadata values (from 'distinct_values' key) processed and loaded.")
                else:
                    logger.warning(f"'{METADATA_CACHE_PATH}' does not contain a 'distinct_values' dictionary. Proceeding without specific distinct values.")
        else:
            logger.warning(f"Metadata cache file not found at {METADATA_CACHE_PATH}. Proceeding without it.")
    except Exception as e:
        logger.error(f"Error loading metadata cache from {METADATA_CACHE_PATH}: {e}. Proceeding without cache.", exc_info=True)
    distinct_metadata_values = temp_distinct_metadata # Assign to global var

    temp_schema_props = None
    try:
        logger.info(f"Loading database schema from {DATABASE_SCHEMA_PATH}...")
        if os.path.exists(DATABASE_SCHEMA_PATH):
            with open(DATABASE_SCHEMA_PATH, 'r') as f:
                loaded_schema = json.load(f) 
                if not isinstance(loaded_schema, dict):
                     logger.error(f"Fatal: Content of {DATABASE_SCHEMA_PATH} is not a JSON dictionary.")
                     raise TypeError(f"Schema file {DATABASE_SCHEMA_PATH} root is not a dictionary.")
                temp_schema_props = loaded_schema
            logger.info("Database schema loaded.")
        else:
            logger.warning(f"Database schema file not found at {DATABASE_SCHEMA_PATH}. Query analysis may be less accurate.")
    except json.JSONDecodeError as e:
        logger.error(f"Fatal: Error decoding JSON from {DATABASE_SCHEMA_PATH}: {e}", exc_info=True)
        raise RuntimeError(f"Error decoding JSON from {DATABASE_SCHEMA_PATH}") from e
    except Exception as e:
        logger.error(f"Fatal: Error loading schema from {DATABASE_SCHEMA_PATH}: {e}", exc_info=True)
        raise RuntimeError(f"Error loading schema from {DATABASE_SCHEMA_PATH}") from e
    schema_properties = temp_schema_props # Assign to global var
    
    _rag_data_loaded = True # Set flag after successful loading
    logger.info("--- RAG Data Loading Complete --- ")

def initialize_openai_client(api_key: str | None):
    """Initializes the OpenAI client using the provided API key."""
    global openai_client
    if openai_client:
        logger.info("OpenAI client already initialized.")
        return

    # Enhanced logging for the API key
    if not api_key:
        logger.error("Fatal: OpenAI API key was NOT provided (None or empty string) during initialization.")
        # We will still raise ValueError, but the log is more specific.
        raise ValueError("OpenAI API key not provided.")
    else:
        # Log type and a portion of the key for debugging, being careful not to log the whole key.
        key_snippet = api_key[:4] + "..." + api_key[-4:] if len(api_key) > 8 else "key_too_short_to_snippet"
        logger.info(f"Attempting to initialize OpenAI client with API key of type {type(api_key)}, snippet: {key_snippet}")

    try:
        openai_client = OpenAI(api_key=api_key)
        logger.info("OpenAI client initialized successfully.") # Changed log message for clarity
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}", exc_info=True)
        raise

def initialize_anthropic_client(api_key: str | None):
    """Initializes the Anthropic client using the provided API key."""
    global anthropic_client
    if anthropic_client:
        logger.info("Anthropic client already initialized.")
        return
        
    if not Anthropic: # Check if Anthropic SDK was imported
        logger.warning("Anthropic SDK not installed. Anthropic models will be unavailable.")
        return # Do not set client if SDK not present
        
    if not api_key:
        logger.warning("Anthropic API key not provided. Anthropic models will be unavailable.")
        return # Do not set client if key not present
        
    try:
        anthropic_client = Anthropic(api_key=api_key)
        logger.info("Anthropic client initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize Anthropic client: {e}", exc_info=True)
        anthropic_client = None # Ensure client is None if initialization fails
        # Do not raise here, allow app to start, but Anthropic features will be disabled

# Function to get the current state (optional, provides controlled access)
def get_rag_globals() -> dict:
    """Returns a dictionary containing the current state of initialized RAG globals."""
    return {
        "openai_client": openai_client,
        "anthropic_client": anthropic_client,
        "index": index,
        "mapping_data_list": mapping_data_list,
        "index_to_entry": index_to_entry,
        "distinct_metadata_values": distinct_metadata_values,
        "schema_properties": schema_properties,
        "Anthropic": Anthropic # Return the class itself if needed elsewhere
    } 