"""Manages shared RAG state (data, clients)."""
# import faiss # Removed this line
import json
import os
import logging
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional, Any

# Added for Vercel Blob integration
try:
    from vercel_blob import download as vercel_download, put as vercel_put, list as vercel_list, VercelBlobException
except ImportError:
    vercel_download = None
    vercel_put = None
    vercel_list = None
    VercelBlobException = None

from openai import OpenAI
from anthropic import Anthropic
from pinecone import Pinecone
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import JSONLoader

logger = logging.getLogger("api.rag_initializer")

# --- Constants for file names (used for both local and remote) ---
# These were previously directly used from rag_config, now centralized for easier reference
INDEX_FILENAME = "index.faiss"
MAPPING_FILENAME = "index_mapping.json"
METADATA_FILENAME = "metadata_cache.json"
SCHEMA_FILENAME = "schema.json"
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

# --- Environment Variable Dependent Names for Vercel Blob ---
# Construct blob names using today's date from RAG_CONFIG
# Ensure RAG_CONFIG["today_date"] is set before this module is fully imported if used at module level
# Or, move these inside functions if RAG_CONFIG["today_date"] is determined dynamically elsewhere

# INDEX_BLOB_NAME = RAG_CONFIG["FAISS_INDEX_FILENAME"]
# DOCSTORE_BLOB_NAME = RAG_CONFIG["DOCSTORE_FILENAME"]
# TIMESTAMP_BLOB_NAME = RAG_CONFIG["LAST_ENTRY_UPDATE_TIMESTAMP_FILENAME"]

# --- Custom Exceptions ---
class RAGSystemNotInitializedError(Exception):
    pass

class LLMClientNotInitializedError(Exception):
    pass

def load_rag_data():
    """Loads necessary data for RAG: index, mapping (list), metadata cache, schema.
       Uses a flag to prevent reloading if already loaded.
       Data can be loaded from local disk or Vercel Blob storage based on DATA_SOURCE_MODE.
    """
    global index, mapping_data_list, index_to_entry, distinct_metadata_values, schema_properties
    global _rag_data_loaded
    
    if _rag_data_loaded:
        logger.debug("RAG data already loaded. Skipping reload.")
        return

    # Ensure Pinecone client is initialized before loading data that might depend on it
    # (though for now, load_rag_data mainly loads mapping files, not directly using the index object)
    if not pinecone_index_instance: # Check if Pinecone is ready
        logger.warning("Pinecone client/index not initialized. Data loading might be incomplete if it were to rely on an active index connection immediately.")
        # Depending on strictness, could raise an error here or try to proceed with just file loading.
        # For now, we primarily load mapping files, schema, etc.

    logger.info("--- Starting RAG Data Loading (Mapping, Schema, Metadata Cache) --- ")

    data_source_mode = os.getenv('DATA_SOURCE_MODE', 'local').lower()
    logger.info(f"DATA_SOURCE_MODE set to: {data_source_mode}")

    # --- Determine base path for data files ---
    # These will be the actual paths used for loading, whether local or remote (temp)
    current_mapping_path: Path
    current_metadata_path: Path
    current_schema_path: Path

    if data_source_mode == 'remote':
        if not vercel_download or not VercelBlobException:
            logger.error("Fatal: Vercel Blob SDK (vercel-blob) is not installed, but DATA_SOURCE_MODE is 'remote'.")
            raise RuntimeError("Vercel Blob SDK not installed for remote data loading.")

        remote_tmp_dir = Path("/tmp/rag_data")
        try:
            remote_tmp_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Ensured temporary directory for remote data: {remote_tmp_dir}")
        except OSError as e:
            logger.error(f"Fatal: Could not create temporary directory {remote_tmp_dir}: {e}", exc_info=True)
            raise RuntimeError(f"Failed to create temporary directory {remote_tmp_dir}") from e

        blob_token = os.getenv("VERCEL_BLOB_ACCESS_TOKEN")
        if not blob_token:
            logger.error("Fatal: VERCEL_BLOB_ACCESS_TOKEN not set for remote data loading.")
            raise RuntimeError("VERCEL_BLOB_ACCESS_TOKEN not set.")

        # Define expected environment variables for blob URLs
        # These URLs point to the specific files in Vercel Blob storage
        file_env_vars_and_paths = {
            # INDEX_FILENAME: (os.getenv("FAISS_INDEX_BLOB_URL"), remote_tmp_dir / INDEX_FILENAME), # FAISS Index removed
            MAPPING_FILENAME: (os.getenv("MAPPING_JSON_BLOB_URL"), remote_tmp_dir / MAPPING_FILENAME),
            METADATA_FILENAME: (os.getenv("METADATA_CACHE_BLOB_URL"), remote_tmp_dir / METADATA_FILENAME),
            SCHEMA_FILENAME: (os.getenv("SCHEMA_JSON_BLOB_URL"), remote_tmp_dir / SCHEMA_FILENAME),
        }

        for filename, (url, local_tmp_path) in file_env_vars_and_paths.items():
            if not url:
                # For schema.json and metadata_cache.json, it might be optional if they don't exist.
                # However, index and mapping are critical.
                # if filename in [INDEX_FILENAME, MAPPING_FILENAME]: # INDEX_FILENAME removed from critical check
                if filename == MAPPING_FILENAME: # Only MAPPING_FILENAME is critical now from this list
                    logger.error(f"Fatal: Blob URL for critical file '{filename}' is not set in environment variables.")
                    raise RuntimeError(f"Missing Blob URL for {filename}.")
                else:
                    logger.warning(f"Blob URL for optional file '{filename}' is not set. Will attempt to proceed without it if possible.")
                    # Ensure the path points to something, even if it won't exist, so later checks handle it.
                    if filename == METADATA_FILENAME: current_metadata_path = local_tmp_path 
                    if filename == SCHEMA_FILENAME: current_schema_path = local_tmp_path
                    continue # Skip download if URL is not set for optional files

            logger.info(f"Downloading '{filename}' from Vercel Blob (URL: {url[:30]}...) to '{local_tmp_path}'...")
            try:
                vercel_download(url=url, path=local_tmp_path, token=blob_token)
                logger.info(f"Successfully downloaded '{filename}' to '{local_tmp_path}'.")
            except VercelBlobException as e:
                logger.error(f"Fatal: Failed to download '{filename}' from Vercel Blob (URL: {url}): {e}", exc_info=True)
                # If critical files fail, re-raise. For optional, we might allow proceeding.
                # if filename in [INDEX_FILENAME, MAPPING_FILENAME]: # INDEX_FILENAME removed
                if filename == MAPPING_FILENAME:
                    raise RuntimeError(f"Failed to download critical file {filename} from Vercel Blob.") from e
                logger.warning(f"Could not download optional file {filename}. Will proceed without it.")
            except Exception as e: # Catch any other unexpected errors during download
                logger.error(f"Fatal: An unexpected error occurred downloading '{filename}' from Vercel Blob: {e}", exc_info=True)
                # if filename in [INDEX_FILENAME, MAPPING_FILENAME]: # INDEX_FILENAME removed
                if filename == MAPPING_FILENAME:
                    raise RuntimeError(f"Unexpected error downloading critical file {filename} from Vercel Blob.") from e
                logger.warning(f"Unexpected error downloading optional file {filename}. Will proceed without it.")
            
            # Assign current paths after successful download or if optional and URL missing
            # if filename == INDEX_FILENAME: current_index_path = local_tmp_path # FAISS Index removed
            if filename == MAPPING_FILENAME: current_mapping_path = local_tmp_path
            elif filename == METADATA_FILENAME: current_metadata_path = local_tmp_path
            elif filename == SCHEMA_FILENAME: current_schema_path = local_tmp_path
        
        # Check if critical paths were set (they would be if download succeeded or if URL was missing but error was raised)
        # if not 'current_index_path' in locals() or not 'current_mapping_path' in locals(): # current_index_path check removed
        if not 'current_mapping_path' in locals():
             # This case should ideally be caught by earlier errors, but as a safeguard:
            logger.error("Fatal: Critical data paths (mapping) were not established in remote mode.")
            raise RuntimeError("Critical remote data paths not established.")
        if not 'current_metadata_path' in locals(): # Ensure it's defined even if download failed/skipped for optional
            current_metadata_path = remote_tmp_dir / METADATA_FILENAME
        if not 'current_schema_path' in locals(): # Ensure it's defined
            current_schema_path = remote_tmp_dir / SCHEMA_FILENAME


    elif data_source_mode == 'local':
        # In local mode, paths are relative to LOCAL_DATA_PATH or current dir by default
        # `rag_config.py` provides default *filenames* (e.g., "index.faiss")
        # `LOCAL_DATA_PATH` defaults to '.' (project root) if not set.
        local_data_base_dir = Path(os.getenv('LOCAL_DATA_PATH', '.'))
        logger.info(f"Local data path set to: {local_data_base_dir.resolve()}")

        # current_index_path = local_data_base_dir / DEFAULT_INDEX_FILENAME # FAISS Index removed
        current_mapping_path = local_data_base_dir / MAPPING_FILENAME
        current_metadata_path = local_data_base_dir / METADATA_FILENAME
        current_schema_path = local_data_base_dir / SCHEMA_FILENAME
    else:
        logger.error(f"Fatal: Invalid DATA_SOURCE_MODE: '{data_source_mode}'. Must be 'local' or 'remote'.")
        raise ValueError(f"Invalid DATA_SOURCE_MODE: {data_source_mode}")

    # --- Load data using the determined paths ---
    # FAISS Index loading is removed. The 'index' global will be the pinecone_index_instance.
    # try:
    #     logger.info(f"Loading FAISS index from {current_index_path}...")
    #     index = faiss.read_index(str(current_index_path)) # faiss expects string path
    #     logger.info(f"Index loaded successfully. Total vectors: {index.ntotal}")
    # except Exception as e:
    #     logger.error(f"Fatal: Failed to load FAISS index from {current_index_path}: {e}", exc_info=True)
    #     raise RuntimeError(f"Could not load FAISS index from {current_index_path}") from e

    global index # This now refers to the pinecone_index_instance
    if not pinecone_index_instance and not index: # Check if pinecone index is available
        logger.warning("Pinecone index instance is not available. RAG system might not function correctly for retrieval.")
        # Depending on use case, might want to raise error or allow proceeding if only mapping is needed for some ops

    try:
        logger.info(f"Loading index mapping list from {current_mapping_path}...")
        with open(current_mapping_path, 'r') as f:
            loaded_mapping_data = json.load(f) 
            if not isinstance(loaded_mapping_data, list):
                raise TypeError(f"{current_mapping_path} does not contain a JSON list.")
        mapping_data_list = loaded_mapping_data 
        logger.info(f"Mapping list loaded successfully. Total entries: {len(mapping_data_list)}")

        # The check against index.ntotal needs to be re-evaluated for Pinecone.
        # For now, we'll skip the direct comparison with Pinecone's total vectors here.
        # If pinecone_index_instance:
        #     try:
        #         stats = pinecone_index_instance.describe_index_stats()
        #         pinecone_total_vectors = stats.total_vector_count
        #         if len(mapping_data_list) != pinecone_total_vectors:
        #             logger.warning(f"Mismatch: Pinecone index has {pinecone_total_vectors} vectors, "
        #                            f"but mapping list has {len(mapping_data_list)} entries. "
        #                            "This could indicate an inconsistency.")
        #     except Exception as e:
        #         logger.warning(f"Could not get Pinecone index stats to verify mapping length: {e}")


        temp_index_to_entry = {} 
        for i, entry_data in enumerate(mapping_data_list): # Changed 'entry' to 'entry_data' to avoid conflict if 'entry' is a pinecone object
            if not isinstance(entry_data, dict):
                logger.warning(f"Skipping non-dictionary item at index {i} in mapping list: {entry_data}")
                continue
            
            # IMPORTANT: Pinecone uses string IDs.
            # We assume 'index_mapping.json' now contains entries where each dict
            # has a unique string ID (e.g., 'page_id' or 'vector_id') that was used for Pinecone.
            # This ID should be used to key 'index_to_entry'.
            # For MVP, let's assume 'page_id' is the string ID.
            # build_index.py will need to ensure this 'page_id' is used for upserting to Pinecone.

            vector_id = entry_data.get('page_id') # Assuming 'page_id' is the string ID. Make this configurable if needed.
                                                # Or, look for a dedicated 'vector_id' field if build_index.py adds one.

            if not vector_id:
                logger.warning(f"Entry at index {i} in mapping list is missing 'page_id' (expected as string vector ID). Skipping: {entry_data.get('title', 'N/A')}")
                continue
            
            if not isinstance(vector_id, str):
                # Attempt to stringify, but log a warning as this implies an issue with how index_mapping.json was created.
                logger.warning(f"Vector ID ('page_id': {vector_id}) for entry '{entry_data.get('title', 'N/A')}' is not a string. Attempting to cast. This should be fixed in build_index.py.")
                vector_id = str(vector_id)

            # entry_data['faiss_index'] = i # This is no longer relevant
            entry_data['vector_id'] = vector_id # Store the string ID used for Pinecone

            if 'entry_date' in entry_data and isinstance(entry_data['entry_date'], str):
                try:
                    entry_data['entry_date'] = date.fromisoformat(entry_data['entry_date'])
                except ValueError:
                    page_id_for_log = entry_data.get('page_id', f'index {i}')
                    logger.warning(f"Could not parse date string: {entry_data['entry_date']} for entry {page_id_for_log}")
                    entry_data['entry_date'] = None
            
            temp_index_to_entry[vector_id] = entry_data # Use string vector_id as key
        index_to_entry = temp_index_to_entry 
        logger.info(f"Processed mapping list into index_to_entry lookup. Size: {len(index_to_entry)} (keyed by string vector IDs)")
            
    except FileNotFoundError:
        logger.error(f"Fatal: Mapping file not found at {current_mapping_path}")
        raise RuntimeError(f"Mapping file not found at {current_mapping_path}")
    except (json.JSONDecodeError, TypeError) as e:
        logger.error(f"Fatal: Failed to decode or process JSON list from {current_mapping_path}: {e}", exc_info=True)
        raise RuntimeError(f"Failed to decode or process {current_mapping_path}") from e
    except Exception as e:
        logger.error(f"Fatal: An unexpected error occurred loading/processing mapping from {current_mapping_path}: {e}", exc_info=True)
        raise RuntimeError(f"Unexpected error loading/processing {current_mapping_path}") from e

    temp_distinct_metadata = {}
    try:
        logger.info(f"Loading distinct metadata values from {current_metadata_path}...")
        if current_metadata_path.exists(): # Use Path.exists()
            with open(current_metadata_path, 'r') as f:
                cache_content = json.load(f)
                if "distinct_values" in cache_content and isinstance(cache_content["distinct_values"], dict):
                    loaded_distinct = cache_content["distinct_values"]
                    for key in loaded_distinct:
                        if isinstance(loaded_distinct[key], list):
                             temp_distinct_metadata[key] = set(loaded_distinct[key])
                        else:
                             temp_distinct_metadata[key] = loaded_distinct[key] 
                    logger.info("Distinct metadata values (from 'distinct_values' key) processed and loaded.")
                else:
                    logger.warning(f"'{current_metadata_path}' does not contain a 'distinct_values' dictionary. Proceeding without specific distinct values.")
        else:
            logger.warning(f"Metadata cache file not found at {current_metadata_path}. Proceeding without it.")
    except Exception as e:
        logger.error(f"Error loading metadata cache from {current_metadata_path}: {e}. Proceeding without cache.", exc_info=True)
    distinct_metadata_values = temp_distinct_metadata

    temp_schema_props = None
    try:
        logger.info(f"Loading database schema from {current_schema_path}...")
        if current_schema_path.exists(): # Use Path.exists()
            with open(current_schema_path, 'r') as f:
                loaded_schema = json.load(f) 
                if not isinstance(loaded_schema, dict):
                     logger.error(f"Fatal: Content of {current_schema_path} is not a JSON dictionary.")
                     raise TypeError(f"Schema file {current_schema_path} root is not a dictionary.")
                temp_schema_props = loaded_schema
            logger.info("Database schema loaded.")
        else:
            logger.warning(f"Database schema file not found at {current_schema_path}. Query analysis may be less accurate.")
    except json.JSONDecodeError as e:
        logger.error(f"Fatal: Error decoding JSON from {current_schema_path}: {e}", exc_info=True)
        raise RuntimeError(f"Error decoding JSON from {current_schema_path}") from e
    except Exception as e:
        logger.error(f"Fatal: Error loading schema from {current_schema_path}: {e}", exc_info=True)
        raise RuntimeError(f"Error loading schema from {current_schema_path}") from e
    schema_properties = temp_schema_props 
    
    _rag_data_loaded = True 
    logger.info("--- RAG Data Loading Complete --- ")

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
    if openai_client:
        logger.info("OpenAI client already initialized. Skipping.") # Added skipping
        return

    # Enhanced logging for the API key
    if not api_key:
        logger.error("FATAL IN INITIALIZER: OpenAI API key was NOT provided (None or empty string). Cannot initialize.")
        # We won't raise here to see if other logs appear, but this is a critical failure point.
        return # Explicitly return if no key
    
    key_snippet = api_key[:4] + "..." + api_key[-4:] if len(api_key) > 8 else "key_too_short_to_snippet"
    logger.info(f"RAG_INITIALIZER: Attempting to initialize OpenAI client with API key snippet: {key_snippet}, type: {type(api_key)}")

    try:
        logger.info("RAG_INITIALIZER: BEFORE OpenAI(api_key=...) call") # LOG BEFORE
        temp_client = OpenAI(api_key=api_key) # Assign to temp var first
        logger.info(f"RAG_INITIALIZER: AFTER OpenAI(api_key=...) call. temp_client is: {'SET' if temp_client else 'NOT SET'}") # LOG AFTER
        
        openai_client = temp_client # Assign to global
        logger.info(f"RAG_INITIALIZER: OpenAI client assigned to global. openai_client is: {'SET' if openai_client else 'NOT SET'}")

    except Exception as e:
        logger.error(f"RAG_INITIALIZER: FAILED to initialize OpenAI client during OpenAI(api_key=...) call: {e}", exc_info=True)
        openai_client = None # Ensure it's None if init fails
        # We are not re-raising here to allow the app to potentially start and log more, 
        # but this is a critical failure for OpenAI functionality.

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
    if not openai_client:
        logger.warning("get_openai_client() called but client is not initialized.")
    return openai_client

def get_anthropic_client():
    """Returns the initialized Anthropic client instance."""
    global anthropic_client
    if not anthropic_client:
        logger.warning("get_anthropic_client() called but client is not initialized.")
    return anthropic_client

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
        "Anthropic": Anthropic # Return the class itself if needed elsewhere
    } 