"""Manages shared RAG state (data, clients)."""
# import faiss # Removed this line
import json
import os
import logging
import requests # Added
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional, Any

# Removed Vercel Blob SDK import block
# try:
#     from vercel_blob import download as vercel_download, put as vercel_put, list as vercel_list, VercelBlobException
# except ImportError:
#     vercel_download = None
#     vercel_put = None
#     vercel_list = None
#     VercelBlobException = None

from openai import OpenAI
from anthropic import Anthropic
from pinecone import Pinecone
# from langchain_community.vectorstores import FAISS # Removed this unused import
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import JSONLoader

from google.cloud import storage # Added for GCS

logger = logging.getLogger("api.rag_initializer")

# --- Constants for file names (used for both local and remote) ---
# These were previously directly used from rag_config, now centralized for easier reference
# INDEX_FILENAME = "index.faiss" # Removed this unused constant
MAPPING_FILENAME = "index_mapping.json"
METADATA_FILENAME = "metadata_cache.json"
SCHEMA_FILENAME = "schema.json"
LAST_SYNC_TIMESTAMP_FILENAME = "last_entry_update_timestamp.txt" # ADDED
# Note: last_entry_update_timestamp.txt is not loaded by this module directly.

# --- Globals managed by this initializer module ---
# These will be imported and used by the main RAG logic (rag_query.py)
openai_client: Optional[OpenAI] = None
anthropic_client: Optional[Anthropic] = None
pinecone_client: Optional[Pinecone] = None
pinecone_index_instance: Optional[Any] = None
index = None
mapping_data_list: list[dict] | None = None
index_to_entry: dict[int, dict] | None = None
distinct_metadata_values: dict | None = None
schema_properties: dict | None = None

# Flag to prevent redundant loading
_rag_data_loaded = False
_last_sync_timestamp_content: Optional[str] = None # ADDED

# --- Environment Variable Dependent Names for Vercel Blob ---
# Construct blob names using today's date from RAG_CONFIG
# Ensure RAG_CONFIG["today_date"] is set before this module is fully imported if used at module level
# Or, move these inside functions if RAG_CONFIG["today_date"] is determined dynamically elsewhere

# INDEX_BLOB_NAME = RAG_CONFIG["FAISS_INDEX_FILENAME"]
# DOCSTORE_BLOB_NAME = RAG_CONFIG["DOCSTORE_FILENAME"]
# TIMESTAMP_BLOB_NAME = RAG_CONFIG["LAST_ENTRY_UPDATE_TIMESTAMP_FILENAME"]

# --- GCS Configuration (from environment) ---
GCS_BUCKET_NAME_ENV = os.getenv("GCS_BUCKET_NAME")
GCS_INDEX_ARTIFACTS_PREFIX_ENV = os.getenv("GCS_INDEX_ARTIFACTS_PREFIX", "index_artifacts/")

# --- Custom Exceptions ---
class RAGSystemNotInitializedError(Exception):
    pass

class LLMClientNotInitializedError(Exception):
    pass

# --- GCS Helper Functions (Simplified for this module) ---
def _get_gcs_client_internal():
    try:
        return storage.Client()
    except Exception as e:
        logger.error(f"Failed to initialize GCS client in rag_initializer: {e}", exc_info=True)
        return None

def download_blob_from_gcs_to_local_path(bucket_name: str, blob_name: str, local_destination_path: Path) -> bool:
    storage_client = _get_gcs_client_internal()
    if not storage_client or not bucket_name:
        logger.error(f"GCS client or bucket name not available for downloading {blob_name}.")
        return False
    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        # Ensure parent directory for local_path exists
        local_destination_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Attempting to download GCS blob gs://{bucket_name}/{blob_name} to {local_destination_path}")
        blob.download_to_filename(str(local_destination_path))
        logger.info(f"Successfully downloaded GCS blob to {local_destination_path}")
        return True
    except Exception as e:
        logger.error(f"Error downloading GCS blob gs://{bucket_name}/{blob_name} to {local_destination_path}: {e}", exc_info=True)
        # Clean up potentially partially downloaded file
        if local_destination_path.exists():
            try:
                local_destination_path.unlink()
            except OSError:
                pass # Ignore errors during cleanup
        return False
# --- End GCS Helper Functions ---

# Remove old Vercel Blob download helper
# def download_file_from_vercel_blob(blob_url: str, local_path: Path, token: str | None = None): ...

def load_rag_data():
    """Loads necessary data for RAG: mapping, metadata cache, schema.
       Data can be loaded from local disk or GCS based on DATA_SOURCE_MODE.
    """
    global mapping_data_list, index_to_entry, distinct_metadata_values, schema_properties
    global _rag_data_loaded, index # index refers to pinecone_index_instance
    
    # Initialize path variables to ensure they are always defined in this scope
    current_mapping_path: Path | None = None
    current_metadata_path: Path | None = None
    current_schema_path: Path | None = None

    if _rag_data_loaded:
        logger.debug("RAG data already loaded. Skipping reload.")
        return

    if not pinecone_index_instance: # Check if Pinecone is ready
        logger.warning("Pinecone client/index not initialized. Data loading might be incomplete.")

    logger.info("--- Starting RAG Data Loading (Mapping, Schema, Metadata Cache) --- ")

    # DATA_SOURCE_MODE: 'local' (default) or 'gcs'
    data_source_mode = os.getenv('DATA_SOURCE_MODE', 'local').lower()
    logger.info(f"DATA_SOURCE_MODE set to: {data_source_mode}")

    if data_source_mode == 'gcs':
        if not GCS_BUCKET_NAME_ENV:
            logger.error("Fatal: DATA_SOURCE_MODE is 'gcs' but GCS_BUCKET_NAME env var is not set.")
            raise RuntimeError("GCS_BUCKET_NAME not set for GCS data source mode.")
        
        gcs_prefix = GCS_INDEX_ARTIFACTS_PREFIX_ENV
        if gcs_prefix and not gcs_prefix.endswith('/'):
            gcs_prefix += '/'

        remote_tmp_dir = Path("/tmp/rag_data_gcs") # Use a distinct temp dir
        try:
            remote_tmp_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Ensured temporary directory for GCS data: {remote_tmp_dir}")
        except OSError as e:
            logger.error(f"Fatal: Could not create temporary directory {remote_tmp_dir}: {e}", exc_info=True)
            raise RuntimeError(f"Failed to create temporary directory {remote_tmp_dir}") from e

        # Define which file maps to which local path variable
        files_config = {
            MAPPING_FILENAME: {"critical": True, "target_path_var": "current_mapping_path"},
            METADATA_FILENAME: {"critical": False, "target_path_var": "current_metadata_path"},
            SCHEMA_FILENAME: {"critical": False, "target_path_var": "current_schema_path"},
            LAST_SYNC_TIMESTAMP_FILENAME: {"critical": False, "target_path_var": "current_timestamp_path"} # ADDED
        }

        for filename, config_item in files_config.items():
            gcs_blob_name = f"{gcs_prefix}{filename}"
            local_tmp_path = remote_tmp_dir / filename
            
            logger.info(f"Requesting download of '{filename}' from GCS gs://{GCS_BUCKET_NAME_ENV}/{gcs_blob_name} to '{local_tmp_path}'...")
            download_successful = download_blob_from_gcs_to_local_path(
                bucket_name=GCS_BUCKET_NAME_ENV,
                blob_name=gcs_blob_name,
                local_destination_path=local_tmp_path
            )

            if download_successful:
                logger.info(f"Successfully downloaded '{filename}' from GCS to '{local_tmp_path}'.")
                # Assign the successful download path to the correct variable
                if config_item["target_path_var"] == "current_mapping_path":
                    current_mapping_path = local_tmp_path
                elif config_item["target_path_var"] == "current_metadata_path":
                    current_metadata_path = local_tmp_path
                elif config_item["target_path_var"] == "current_schema_path":
                    current_schema_path = local_tmp_path
                elif config_item["target_path_var"] == "current_timestamp_path": # ADDED BLOCK
                    # This is the timestamp file, read its content
                    try:
                        with open(local_tmp_path, 'r', encoding='utf-8') as ts_file:
                            _last_sync_timestamp_content = ts_file.read().strip()
                        logger.info(f"Successfully read timestamp content from {local_tmp_path}: '{_last_sync_timestamp_content}'")
                        # We don't need to assign to current_timestamp_path for further processing
                        # as its content is now in the global variable.
                        # Optionally delete the temp file if no longer needed locally
                        # local_tmp_path.unlink(missing_ok=True) 
                    except Exception as e_read_ts:
                        logger.error(f"Error reading downloaded timestamp file {local_tmp_path}: {e_read_ts}", exc_info=True)
                        _last_sync_timestamp_content = None # Ensure it's None on error
            else:
                if config_item["critical"]:
                    logger.error(f"Fatal: Failed to download critical file '{filename}' from GCS. Halting data load.")
                    raise RuntimeError(f"Failed to download critical file {filename} from GCS.")
                else:
                    logger.warning(f"Could not download optional file '{filename}' from GCS. Will proceed without it if possible.")
                    # Ensure path var is None if download failed for optional file
                    if config_item["target_path_var"] == "current_metadata_path":
                        current_metadata_path = None
                    elif config_item["target_path_var"] == "current_schema_path":
                        current_schema_path = None
                    elif config_item["target_path_var"] == "current_timestamp_path": # ADDED
                        _last_sync_timestamp_content = None # Ensure it's None if download failed
        
    elif data_source_mode == 'local':
        local_data_base_dir = Path(os.getenv('LOCAL_DATA_PATH', '.'))
        logger.info(f"Local data path set to: {local_data_base_dir.resolve()}")
        current_mapping_path = local_data_base_dir / MAPPING_FILENAME
        current_metadata_path = local_data_base_dir / METADATA_FILENAME
        current_schema_path = local_data_base_dir / SCHEMA_FILENAME
        # For local mode, try to load timestamp file directly
        local_timestamp_path = local_data_base_dir / LAST_SYNC_TIMESTAMP_FILENAME # ADDED
        if local_timestamp_path.exists(): # ADDED BLOCK
            try:
                with open(local_timestamp_path, 'r', encoding='utf-8') as ts_file:
                    _last_sync_timestamp_content = ts_file.read().strip()
                logger.info(f"Successfully read local timestamp content from {local_timestamp_path}: '{_last_sync_timestamp_content}'")
            except Exception as e_read_ts_local:
                logger.error(f"Error reading local timestamp file {local_timestamp_path}: {e_read_ts_local}", exc_info=True)
                _last_sync_timestamp_content = None
        else:
            logger.warning(f"Local timestamp file not found at {local_timestamp_path}. Last sync time will be unknown.")
            _last_sync_timestamp_content = None
    else:
        logger.error(f"Fatal: Invalid DATA_SOURCE_MODE: '{data_source_mode}'. Must be 'local' or 'gcs'.")
        raise ValueError(f"Invalid DATA_SOURCE_MODE: {data_source_mode}")

    # --- Load data using the determined paths ---
    if not pinecone_index_instance and not index: # Pinecone index is assigned to global 'index' later
        logger.warning("Pinecone index instance is not available. RAG system might not function correctly for retrieval.")

    try:
        # Ensure current_mapping_path is not None before proceeding, as it's critical
        if current_mapping_path is None:
            logger.error("Fatal: current_mapping_path was not set. This indicates a critical file was not loaded.")
            raise FileNotFoundError("Mapping file path was not determined.")
        
        logger.info(f"Loading index mapping list from {current_mapping_path}...")
        if not current_mapping_path.exists():
            raise FileNotFoundError(f"Mapping file not found at {current_mapping_path}. This is critical.")
        with open(current_mapping_path, 'r', encoding='utf-8') as f:
            loaded_mapping_file_content = json.load(f) 
            # Expecting format from build_index: {"description": ..., "entries": [...]}
            if isinstance(loaded_mapping_file_content, dict) and "entries" in loaded_mapping_file_content:
                mapping_data_list = loaded_mapping_file_content["entries"]
                if not isinstance(mapping_data_list, list):
                    raise TypeError(f"Key 'entries' in {current_mapping_path} is not a list.")
            elif isinstance(loaded_mapping_file_content, list): # Support old direct list format too
                logger.warning(f"Mapping file {current_mapping_path} is a direct list. Prefer format with 'entries' key.")
                mapping_data_list = loaded_mapping_file_content
            else:
                 raise TypeError(f"{current_mapping_path} does not contain a JSON list or a dict with an 'entries' list.")
        logger.info(f"Mapping list loaded successfully with {len(mapping_data_list)} entries.")
        
        # Convert entry_date strings to datetime.date objects
        for entry_map_item in mapping_data_list: # Use a different variable name to avoid confusion with 'entry' module
            date_str = entry_map_item.get("entry_date")
            if date_str and isinstance(date_str, str):
                try:
                    entry_map_item["entry_date"] = date.fromisoformat(date_str)
                except ValueError:
                    logger.warning(f"Could not parse date string '{date_str}' for entry_id {entry_map_item.get('page_id')}. Setting to None.")
                    entry_map_item["entry_date"] = None # Set to None if parsing fails
            elif isinstance(date_str, date): # Already a date object, do nothing
                pass
            else: # Missing or not a string/date, ensure it's None if not already
                if "entry_date" in entry_map_item and entry_map_item["entry_date"] is not None:
                    logger.debug(f"entry_date for {entry_map_item.get('page_id')} is {type(entry_map_item.get('entry_date'))}, not string or date. Setting to None.")
                entry_map_item["entry_date"] = None

        # Create index_to_entry from mapping_data_list (if FAISS was used, this mapped faiss index to entry)
        # For Pinecone, the 'pinecone_id' (page_id) is the key. This mapping might be less critical
        # if retrieval logic directly uses page_ids from Pinecone metadata.
        # However, RAG_SYSTEM_OVERVIEW.md refers to 'index_mapping.json' for mapping vector index to entry.
        # Let's assume it can still be useful for quick lookups by original list order if needed, or adapt later.
        index_to_entry = {i: entry for i, entry in enumerate(mapping_data_list)}
        logger.info(f"Created index_to_entry map with {len(index_to_entry)} entries.")

    except FileNotFoundError:
        logger.error(f"Fatal: Mapping file {current_mapping_path} not found. Ensure it was downloaded or exists locally.")
        raise
    except (json.JSONDecodeError, TypeError) as e:
        logger.error(f"Fatal: Failed to load or parse mapping data from {current_mapping_path}: {e}", exc_info=True)
        raise RuntimeError(f"Could not load/parse mapping data from {current_mapping_path}") from e

    try:
        # Ensure current_metadata_path is defined before trying to use it
        if current_metadata_path is None:
            logger.warning(f"Metadata cache path (current_metadata_path) is not set. Proceeding without it.")
            distinct_metadata_values = {}
        elif current_metadata_path.exists():
            logger.info(f"Loading metadata cache from {current_metadata_path}...")
            with open(current_metadata_path, 'r') as f:
                loaded_metadata_cache = json.load(f)
                if isinstance(loaded_metadata_cache, dict) and "distinct_values" in loaded_metadata_cache:
                    distinct_metadata_values = loaded_metadata_cache["distinct_values"]
                    logger.info(f"Metadata cache loaded successfully. Keys: {list(distinct_metadata_values.keys())}")
                else:
                    logger.warning(f"Metadata cache {current_metadata_path} is not in expected format (dict with 'distinct_values'). Skipping.")
                    distinct_metadata_values = {}
        else:
            logger.warning(f"Metadata cache file {current_metadata_path} not found. Proceeding without it.")
            distinct_metadata_values = {}
    except Exception as e:
        logger.error(f"Error loading metadata cache from {current_metadata_path if current_metadata_path else 'N/A'}: {e}", exc_info=True)
        distinct_metadata_values = {} # Default to empty if error

    try:
        # Ensure current_schema_path is defined
        if current_schema_path is None:
            logger.warning(f"Schema path (current_schema_path) is not set. Proceeding without schema properties.")
            schema_properties = {}
        elif current_schema_path.exists():
            logger.info(f"Loading database schema from {current_schema_path}...")
            with open(current_schema_path, 'r') as f:
                loaded_schema = json.load(f)
                # Expecting schema to be a dict, often with a top-level "properties" key from Notion API schema calls
                if isinstance(loaded_schema, dict):
                    if "properties" in loaded_schema: # Common structure for Notion DB schema
                        schema_properties = loaded_schema["properties"]
                    else: # Assume the loaded dict itself is the properties map
                        logger.warning(f"Schema file {current_schema_path} does not have a top-level 'properties' key. Using the whole object as schema_properties.")
                        schema_properties = loaded_schema
                    logger.info(f"Schema properties loaded. Keys: {list(schema_properties.keys())}")
                else:
                    logger.warning(f"Schema file {current_schema_path} did not load as a dictionary. Skipping.")
                    schema_properties = {}
        else:
            logger.warning(f"Schema file {current_schema_path} not found. Proceeding without schema properties.")
            schema_properties = {}
    except Exception as e:
        logger.error(f"Error loading schema from {current_schema_path if current_schema_path else 'N/A'}: {e}", exc_info=True)
        schema_properties = {} # Default to empty if error

    _rag_data_loaded = True
    logger.info("--- RAG Data Loading Completed ---")

def initialize_pinecone_client(pinecone_api_key: str | None, pinecone_index_name: str | None):
    """Initializes the Pinecone client and connects to the specified index."""
    global pinecone_client, pinecone_index_instance, index
    logger.info("Initializing Pinecone client...")

    # Use passed arguments instead of os.getenv()
    # pinecone_api_key = os.getenv("PINECONE_API_KEY")
    # pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")

    if not pinecone_api_key:
        logger.error("PINECONE_API_KEY not provided to initialize_pinecone_client. Pinecone client not initialized.")
        return
    if not pinecone_index_name:
        logger.error("PINECONE_INDEX_NAME not provided to initialize_pinecone_client. Pinecone client not initialized.")
        return
    # Environment variable for Pinecone host/environment might be needed if direct init by name fails
    # For pinecone-client v3.x, initialization with api_key is standard.
    # The host is resolved when pc.Index(name) is called.

    try:
        logger.info(f"Attempting to initialize Pinecone client with API key.")
        pc = Pinecone(api_key=pinecone_api_key)
        logger.info(f"Pinecone client initialized. Connecting to index: '{pinecone_index_name}'")
        
        # Check if index exists and is ready (optional, good practice)
        # if pinecone_index_name not in pc.list_indexes().names:
        #     logger.error(f"Pinecone index '{pinecone_index_name}' does not exist.")
        #     # Decide if to raise error or try to create it, for now, assume it exists.
        #     return

        index_instance = pc.Index(pinecone_index_name)
        # You can add a describe_index_stats() call here to confirm connection and get details
        # stats = index_instance.describe_index_stats()
        # logger.info(f"Successfully connected to Pinecone index '{pinecone_index_name}'. Stats: {stats}")
        
        pinecone_client = pc
        pinecone_index_instance = index_instance
        index = pinecone_index_instance # Assign to global index variable
        logger.info(f"Pinecone client and index '{pinecone_index_name}' instance are ready.")

    except Exception as e:
        logger.error(f"Failed to initialize Pinecone client or connect to index '{pinecone_index_name}': {e}", exc_info=True)
        pinecone_client = None
        pinecone_index_instance = None
        index = None # Ensure index is also None on failure

def initialize_openai_client(api_key: str | None):
    """Initializes the OpenAI client using the provided API key."""
    global openai_client
    logger.info("RAG_INITIALIZER: initialize_openai_client CALLED.") # New Log

    if openai_client:
        logger.info("RAG_INITIALIZER: OpenAI client already initialized. Skipping.")
        return

    if not api_key:
        logger.error("RAG_INITIALIZER: FATAL - OpenAI API key was NOT provided (None or empty string). Cannot initialize.")
        return
    
    key_snippet = api_key[:4] + "..." + api_key[-4:] if len(api_key) > 8 else "key_too_short_to_snippet"
    logger.info(f"RAG_INITIALIZER: Attempting to initialize OpenAI client with API key snippet: {key_snippet}, type: {type(api_key)}")

    try:
        logger.info("RAG_INITIALIZER: BEFORE OpenAI(api_key=...) call.")
        temp_client = OpenAI(api_key=api_key)
        logger.info(f"RAG_INITIALIZER: AFTER OpenAI(api_key=...) call. temp_client object: {str(temp_client)[:100]}...") # Log part of the object
        
        openai_client = temp_client
        logger.info(f"RAG_INITIALIZER: OpenAI client assigned to global. openai_client object: {str(openai_client)[:100]}...") # Log part of the global object
        if openai_client:
            logger.info("RAG_INITIALIZER: OpenAI client successfully SET.")
        else:
            logger.error("RAG_INITIALIZER: OpenAI client is NONE after assignment.")

    except Exception as e:
        logger.error(f"RAG_INITIALIZER: FAILED to initialize OpenAI client during OpenAI(api_key=...) call: {e}", exc_info=True)
        openai_client = None
        logger.error("RAG_INITIALIZER: openai_client set to None due to exception.") # New Log

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

def get_openai_client():
    """Returns the initialized OpenAI client instance."""
    global openai_client
    logger.info("RAG_INITIALIZER: get_openai_client CALLED.") # New Log
    if not openai_client:
        logger.warning("RAG_INITIALIZER: get_openai_client() called but openai_client global is None.")
    else:
        logger.info(f"RAG_INITIALIZER: get_openai_client() returning openai_client object: {str(openai_client)[:100]}...") # New Log
    return openai_client

def get_anthropic_client():
    """Returns the initialized Anthropic client instance."""
    global anthropic_client
    if not anthropic_client:
        logger.warning("get_anthropic_client() called but client is not initialized.")
    return anthropic_client

def get_last_sync_timestamp() -> Optional[str]: # ADDED FUNCTION
    """Returns the content of the last sync timestamp file, if loaded."""
    return _last_sync_timestamp_content

# Function to get the current state (optional, provides controlled access)
def get_rag_globals() -> dict:
    """Returns a dictionary containing the current state of initialized RAG globals."""
    return {
        "openai_client": openai_client,
        "anthropic_client": anthropic_client,
        "pinecone_client": pinecone_client,
        "pinecone_index_instance": pinecone_index_instance,
        "index": index,
        "mapping_data_list": mapping_data_list,
        "index_to_entry": index_to_entry,
        "distinct_metadata_values": distinct_metadata_values,
        "schema_properties": schema_properties,
        "Anthropic": Anthropic, # Return the class itself if needed elsewhere
        "last_sync_timestamp_content": _last_sync_timestamp_content
    } 