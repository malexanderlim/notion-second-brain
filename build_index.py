# build_index.py

import argparse
import logging
import json
import os
import sys
import asyncio # Add asyncio import
# import numpy as np # Likely no longer needed
# import faiss # No longer needed
from openai import OpenAI, RateLimitError, APIError
import time
import glob # For handling file globs
from datetime import datetime # Ensure datetime is imported
from pinecone import Pinecone, PineconeException # Import Pinecone
import requests # Added
import mimetypes # Added
from google.cloud import storage # Added for GCS

# Removed vercel_blob import
# try:
#     import vercel_blob
# except ImportError:
#     vercel_blob = None

# Adjust path to import from the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from notion_second_brain import config # Use centralized logging and config

# Setup logger for this module
logger = logging.getLogger("build_index")

# Constants (Consider moving to config or making CLI args if more flexibility needed)
# DEFAULT_INPUT_JSON = "output/all_time.json" # No longer a single default
DEFAULT_INPUT_GLOB = "output/*.json" # Default to process all JSONs in output
# DEFAULT_INDEX_FILE = "index.faiss" # No longer saving a FAISS index file locally by default
DEFAULT_MAPPING_FILE = "index_mapping.json"
EMBEDDING_MODEL = "text-embedding-ada-002"
DEFAULT_LAST_ENTRY_TIMESTAMP_FILE = "last_entry_update_timestamp.txt" # New constant
DEFAULT_METADATA_CACHE_FILE = "metadata_cache.json" # Added for consistency
SCHEMA_FILE_NAME = "schema.json" # Added for consistency

# Simple retry logic for embedding API calls
MAX_EMBEDDING_RETRIES = 3
EMBEDDING_RETRY_DELAY = 5 # seconds
BATCH_SIZE = 100 # Number of texts to send in each embedding request
PINECONE_UPSERT_BATCH_SIZE = 100 # Batch size for upserting vectors to Pinecone

# Global Pinecone index instance
pinecone_index_instance = None

# --- GCS Configuration ---
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
GCS_EXPORT_PREFIX = os.getenv("GCS_EXPORT_PREFIX", "notion_exports/") # Raw data JSONs
GCS_INDEX_ARTIFACTS_PREFIX = os.getenv("GCS_INDEX_ARTIFACTS_PREFIX", "index_artifacts/") # Index files

# Ensure prefixes end with a slash
if GCS_EXPORT_PREFIX and not GCS_EXPORT_PREFIX.endswith('/'):
    GCS_EXPORT_PREFIX += '/'
if GCS_INDEX_ARTIFACTS_PREFIX and not GCS_INDEX_ARTIFACTS_PREFIX.endswith('/'):
    GCS_INDEX_ARTIFACTS_PREFIX += '/'

# --- GCS Helper Functions ---
def _get_gcs_client():
    try:
        return storage.Client()
    except Exception as e:
        logger.error(f"Failed to initialize GCS client: {e}", exc_info=True)
        return None

def list_gcs_blobs(bucket_name: str, prefix: str, file_extension: str = ".json"):
    storage_client = _get_gcs_client()
    if not storage_client or not bucket_name:
        logger.error("GCS client or bucket name not available for listing blobs.")
        return []
    
    blobs_data = []
    try:
        logger.info(f"Listing blobs in GCS bucket '{bucket_name}' with prefix '{prefix}' and extension '{file_extension}'")
        blobs = storage_client.list_blobs(bucket_name, prefix=prefix)
        for blob in blobs:
            if blob.name.endswith(file_extension) and blob.name != prefix: # Ensure it's not the prefix itself if it's a directory-like object
                logger.debug(f"Found GCS blob: {blob.name}")
                blobs_data.append({"name": blob.name, "updated": blob.updated})
        logger.info(f"Found {len(blobs_data)} blobs matching criteria.")
        # Sort by name (e.g., by date if filenames are sortable)
        return sorted(blobs_data, key=lambda b: b["name"])
    except Exception as e:
        logger.error(f"Error listing GCS blobs in gs://{bucket_name}/{prefix}: {e}", exc_info=True)
        return []

def download_gcs_blob_as_json(bucket_name: str, blob_name: str):
    storage_client = _get_gcs_client()
    if not storage_client or not bucket_name:
        return None
    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        logger.info(f"Downloading GCS blob: gs://{bucket_name}/{blob_name}")
        json_data = json.loads(blob.download_as_string())
        logger.debug(f"Successfully downloaded and parsed gs://{bucket_name}/{blob_name}")
        return json_data
    except Exception as e:
        logger.error(f"Error downloading GCS blob gs://{bucket_name}/{blob_name}: {e}", exc_info=True)
        return None

def upload_file_to_gcs(local_file_path: str, bucket_name: str, gcs_blob_name: str):
    storage_client = _get_gcs_client()
    if not storage_client or not bucket_name:
        logger.error(f"GCS client or bucket name not available for uploading {local_file_path}.")
        return None
    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(gcs_blob_name)
        blob.upload_from_filename(local_file_path)
        gcs_uri = f"gs://{bucket_name}/{gcs_blob_name}"
        logger.info(f"Successfully uploaded {local_file_path} to {gcs_uri}")
        return gcs_uri
    except Exception as e:
        logger.error(f"Error uploading {local_file_path} to GCS (gs://{bucket_name}/{gcs_blob_name}): {e}", exc_info=True)
        return None

def upload_content_to_gcs(content: str, bucket_name: str, gcs_blob_name: str, content_type: str = 'application/json'):
    storage_client = _get_gcs_client()
    if not storage_client or not bucket_name:
        logger.error(f"GCS client or bucket name not available for uploading content to {gcs_blob_name}.")
        return None
    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(gcs_blob_name)
        blob.upload_from_string(content, content_type=content_type)
        gcs_uri = f"gs://{bucket_name}/{gcs_blob_name}"
        logger.info(f"Successfully uploaded content to {gcs_uri}")
        return gcs_uri
    except Exception as e:
        logger.error(f"Error uploading content to GCS (gs://{bucket_name}/{gcs_blob_name}): {e}", exc_info=True)
        return None

def get_embedding(client: OpenAI, text: str, retries: int = MAX_EMBEDDING_RETRIES) -> list[float] | None:
    """Gets embedding for text using OpenAI API with retry logic."""
    attempt = 0
    while attempt < retries:
        try:
            # Ensure input is not empty
            if not text or not text.strip():
                 logger.warning("Attempted to get embedding for empty string. Skipping.")
                 return None
            
            response = client.embeddings.create(
                input=text,
                model=EMBEDDING_MODEL
            )
            return response.data[0].embedding
        except RateLimitError as e:
            attempt += 1
            logger.warning(f"Rate limit hit for embedding, attempt {attempt}/{retries}. Retrying in {EMBEDDING_RETRY_DELAY}s... Error: {e}")
            time.sleep(EMBEDDING_RETRY_DELAY)
        except APIError as e:
            # Handle potential context length errors specifically if possible
            if "context_length_exceeded" in str(e):
                 logger.error(f"Context length exceeded for text snippet: '{text[:100]}...'. Skipping entry.")
                 return None # Cannot retry this error
            attempt += 1
            logger.warning(f"API error during embedding, attempt {attempt}/{retries}. Retrying in {EMBEDDING_RETRY_DELAY}s... Error: {e}")
            time.sleep(EMBEDDING_RETRY_DELAY)
        except Exception as e:
            logger.error(f"Unexpected error getting embedding: {e}", exc_info=True)
            return None # Stop retrying on unexpected errors
    logger.error(f"Failed to get embedding after {retries} retries.")
    return None

def embed_batch(client: OpenAI, texts: list[str], retries: int = MAX_EMBEDDING_RETRIES) -> list[list[float]] | None:
    """Gets embeddings for a batch of texts using OpenAI API with retry logic."""
    if not texts:
        return [] # Return empty list if no texts provided
        
    attempt = 0
    while attempt < retries:
        try:
            response = client.embeddings.create(
                input=texts,
                model=EMBEDDING_MODEL
            )
            # Ensure the response structure is as expected and contains data
            if not response.data or len(response.data) != len(texts):
                 logger.error(f"Mismatched response length from OpenAI embedding API. Expected {len(texts)}, got {len(response.data) if response.data else 0}.")
                 # Decide how to handle: return None? Partial results? For now, treat as failure.
                 raise APIError(f"Mismatched response length. Expected {len(texts)}, got {len(response.data) if response.data else 0}.", response=None, body=None)

            # Extract embeddings in the correct order
            batch_embeddings = [item.embedding for item in response.data]
            return batch_embeddings
        
        except RateLimitError as e:
            attempt += 1
            logger.warning(f"Rate limit hit for embedding batch (size {len(texts)}), attempt {attempt}/{retries}. Retrying in {EMBEDDING_RETRY_DELAY}s... Error: {e}")
            time.sleep(EMBEDDING_RETRY_DELAY)
        except APIError as e:
             # Handle context length errors for the whole batch if possible (though less likely)
            if "context_length_exceeded" in str(e):
                 logger.error(f"Context length exceeded for the batch (size {len(texts)}). Cannot process batch. Error: {e}")
                 # This indicates one or more texts in the batch were too long. 
                 # We can't easily identify which one failed here. Return None to signal batch failure.
                 return None 
            attempt += 1
            logger.warning(f"API error during embedding batch (size {len(texts)}), attempt {attempt}/{retries}. Retrying in {EMBEDDING_RETRY_DELAY}s... Error: {e}")
            time.sleep(EMBEDDING_RETRY_DELAY)
        except Exception as e:
            logger.error(f"Unexpected error getting embedding batch (size {len(texts)}): {e}", exc_info=True)
            return None # Stop retrying on unexpected errors
            
    logger.error(f"Failed to get embedding batch after {retries} retries.")
    return None

def extract_and_save_metadata_cache(mapping_data: list[dict], cache_file_name: str, gcs_bucket: str, gcs_prefix: str):
    """Extracts distinct values for specified fields and saves them to GCS."""
    logger.info("Extracting distinct values for metadata cache...")
    # Define which fields to extract distinct values from 
    fields_to_extract = ["Family", "Friends", "Tags"]
    distinct_metadata_values = {field: set() for field in fields_to_extract}
    extraction_errors = 0
    
    for entry in mapping_data:
        for field in fields_to_extract:
            value = entry.get(field)
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, str) and item: 
                        distinct_metadata_values[field].add(item)
            elif value is not None and not isinstance(value, list):
                 if extraction_errors < 10: 
                      logger.warning(f"CacheGen: Expected '{field}' to be a list, found {type(value)}. Entry ID: {entry.get('page_id', 'N/A')}")
                 extraction_errors += 1

    final_cache_data = {field: sorted(list(values)) for field, values in distinct_metadata_values.items()}
        
    logger.info("Finished extracting distinct values for cache.")
    if extraction_errors > 0:
        logger.warning(f"CacheGen: Encountered {extraction_errors} unexpected data type issues.")

    cache_content = {
        "last_updated": datetime.now().isoformat(),
        "distinct_values": final_cache_data
    }
    
    gcs_blob_name = f"{gcs_prefix}{cache_file_name}"
    logger.info(f"Attempting to save metadata cache to GCS: gs://{gcs_bucket}/{gcs_blob_name}")
    
    cache_json_string = json.dumps(cache_content, indent=2)
    saved_uri = upload_content_to_gcs(cache_json_string, gcs_bucket, gcs_blob_name, 'application/json')

    if saved_uri:
        logger.info(f"Successfully saved metadata cache to {saved_uri}")
    else:
        logger.error(f"Failed to save metadata cache to GCS.")
        # Decide on error handling, e.g., raise exception or return status

async def main():
    parser = argparse.ArgumentParser(description="Build or update a Pinecone index from Notion data JSON files in GCS.")
    # Input is now implicitly from GCS via GCS_EXPORT_PREFIX
    # parser.add_argument(
    #     "-i", "--input",
    #     nargs='+', 
    #     default=[DEFAULT_INPUT_GLOB], # This default is no longer directly used for local files
    #     help=f"Path(s) or glob pattern(s) for input JSON files (default: '{DEFAULT_INPUT_GLOB}') - Now reads from GCS_EXPORT_PREFIX"
    # )
    
    # Output files are also managed via GCS prefixes
    # parser.add_argument(
    #     "--mapping-file",
    #     default=DEFAULT_MAPPING_FILE,
    #     help=f"Base name for the index-to-entry mapping JSON file (saved to GCS, default: {DEFAULT_MAPPING_FILE})"
    # )
    # parser.add_argument(
    #     "--last-entry-timestamp-file",
    #     default=DEFAULT_LAST_ENTRY_TIMESTAMP_FILE,
    #     help=f"Base name for the last processed entry timestamp file (saved to GCS, default: {DEFAULT_LAST_ENTRY_TIMESTAMP_FILE})"
    # )
    # parser.add_argument(
    #     "--metadata-cache-file", # New argument for consistency, though file name is fixed
    #     default=DEFAULT_METADATA_CACHE_FILE,
    #     help=f"Base name for the metadata cache file (saved to GCS, default: {DEFAULT_METADATA_CACHE_FILE})"
    # )
    parser.add_argument( # Keep this for schema file name, could also be env var
        "--schema-file-name",
        default=SCHEMA_FILE_NAME,
        help=f"Name of the schema JSON file to be uploaded/used (default: {SCHEMA_FILE_NAME})"
    )

    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Ignore existing index and mapping files (from GCS) and build from scratch."
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging (DEBUG level)."
    )
    parser.add_argument(
        "--pinecone-api-key",
        default=os.getenv("PINECONE_API_KEY"),
        help="Pinecone API key. Defaults to PINECONE_API_KEY env var."
    )
    parser.add_argument(
        "--pinecone-index-name",
        default=os.getenv("PINECONE_INDEX_NAME"),
        help="Pinecone index name. Defaults to PINECONE_INDEX_NAME env var."
    )
    parser.add_argument(
        "--openai-api-key",
        default=os.getenv("OPENAI_API_KEY"),
        help="OpenAI API key. Defaults to OPENAI_API_KEY env var."
    )
    # Add arguments for GCS if not solely relying on env vars (optional)
    # parser.add_argument("--gcs-bucket-name", default=GCS_BUCKET_NAME, help="GCS bucket name")
    # parser.add_argument("--gcs-export-prefix", default=GCS_EXPORT_PREFIX, help="GCS prefix for exported Notion JSONs")
    # parser.add_argument("--gcs-index-artifacts-prefix", default=GCS_INDEX_ARTIFACTS_PREFIX, help="GCS prefix for storing index artifacts")


    args = parser.parse_args()

    # Setup logging level
    log_level = logging.DEBUG if args.verbose else logging.INFO
    config.setup_logging(level=log_level) # Use centralized setup
    logger.setLevel(log_level) # Ensure this module's logger also respects the level

    if not GCS_BUCKET_NAME:
        logger.error("GCS_BUCKET_NAME environment variable is not set. Exiting.")
        sys.exit(1)
    if not GCS_EXPORT_PREFIX: # Should have a default but good to check
        logger.error("GCS_EXPORT_PREFIX is not configured. Exiting.")
        sys.exit(1)
    if not GCS_INDEX_ARTIFACTS_PREFIX: # Should have a default
        logger.error("GCS_INDEX_ARTIFACTS_PREFIX is not configured. Exiting.")
        sys.exit(1)
    
    openai_client = OpenAI(api_key=args.openai_api_key)
    pinecone_api_key = args.pinecone_api_key
    pinecone_index_name = args.pinecone_index_name

    if not openai_client.api_key:
        logger.error("OpenAI API key is missing. Please set OPENAI_API_KEY environment variable or use --openai-api-key.")
        sys.exit(1)
    if not pinecone_api_key:
        logger.error("Pinecone API key is missing. Please set PINECONE_API_KEY or use --pinecone-api-key.")
        sys.exit(1)
    if not pinecone_index_name:
        logger.error("Pinecone index name is missing. Please set PINECONE_INDEX_NAME or use --pinecone-index-name.")
        sys.exit(1)
        
    # Initialize Pinecone
    try:
        pinecone_client = Pinecone(api_key=pinecone_api_key)
        # Check if index exists, if not, create or inform (creation is usually separate)
        found_indexes = pinecone_client.list_indexes() # Returns a list of IndexDescription-like objects
        index_names = [index_description.name for index_description in found_indexes] # Extract names

        if pinecone_index_name not in index_names:
            logger.error(f"Pinecone index '{pinecone_index_name}' does not exist. Please create it first.")
            # Potentially add code here to create the index if desired, e.g.:
            # from pinecone import ServerlessSpec
            # DIMS = 1536 # Dimensions for text-embedding-ada-002
            # pinecone_client.create_index(pinecone_index_name, dimension=DIMS, metric='cosine', spec=ServerlessSpec(cloud='aws', region='us-east-1'))
            # logger.info(f"Pinecone index '{pinecone_index_name}' created.")
            sys.exit(1)
        global pinecone_index_instance
        pinecone_index_instance = pinecone_client.Index(pinecone_index_name)
        logger.info(f"Successfully connected to Pinecone index: {pinecone_index_name}")
    except PineconeException as e:
        logger.error(f"Pinecone initialization failed: {e}", exc_info=True)
        sys.exit(1)
    except Exception as e: # Catch any other unexpected errors during Pinecone init
        logger.error(f"An unexpected error occurred during Pinecone initialization: {e}", exc_info=True)
        sys.exit(1)

    all_entries_data = []
    processed_page_ids = set()  # To track IDs for incremental updates
    
    # --- Load existing mapping and timestamp from GCS if not force_rebuild ---
    # These filenames are now constants at the top of the file
    gcs_mapping_blob_name = f"{GCS_INDEX_ARTIFACTS_PREFIX}{DEFAULT_MAPPING_FILE}"
    gcs_timestamp_blob_name = f"{GCS_INDEX_ARTIFACTS_PREFIX}{DEFAULT_LAST_ENTRY_TIMESTAMP_FILE}"

    if not args.force_rebuild:
        logger.info(f"Attempting to load existing mapping data from {gcs_mapping_blob_name}")
        existing_mapping_json = download_gcs_blob_as_json(GCS_BUCKET_NAME, gcs_mapping_blob_name)
        if existing_mapping_json:
            all_entries_data = existing_mapping_json.get("entries", []) # Assuming mapping file has an "entries" key
            # Populate processed_page_ids from existing mapping
            for entry in all_entries_data:
                if 'page_id' in entry:
                    processed_page_ids.add(entry['page_id'])
            logger.info(f"Loaded {len(all_entries_data)} entries and {len(processed_page_ids)} page IDs from existing GCS mapping file.")
        else:
            logger.info("No existing mapping file found on GCS or failed to load. Starting fresh.")

        logger.info(f"Attempting to load last entry timestamp from {gcs_timestamp_blob_name}")
        # Timestamp is a plain text file
        storage_client = _get_gcs_client()
        last_entry_timestamp = None
        if storage_client:
            try:
                blob = storage_client.bucket(GCS_BUCKET_NAME).blob(gcs_timestamp_blob_name)
                if blob.exists():
                    last_entry_timestamp = blob.download_as_text()
                    logger.info(f"Loaded last entry timestamp from GCS: {last_entry_timestamp}")
                else:
                    logger.info("No last entry timestamp file found on GCS.")
            except Exception as e:
                logger.warning(f"Could not load last entry timestamp from GCS: {e}")
        else:
            logger.info("Force rebuild enabled. Ignoring existing GCS mapping and timestamp.")
            last_entry_timestamp = None # Reset timestamp for full rebuild

    # --- Load new/updated data from exported JSONs in GCS ---
    logger.info(f"Listing exported JSON files from GCS bucket '{GCS_BUCKET_NAME}' at prefix '{GCS_EXPORT_PREFIX}'")
    export_blobs = list_gcs_blobs(GCS_BUCKET_NAME, GCS_EXPORT_PREFIX, ".json")
    
    if not export_blobs:
        logger.warning(f"No JSON export files found in GCS at gs://{GCS_BUCKET_NAME}/{GCS_EXPORT_PREFIX}. Nothing to process.")
        # If there's existing data, we might still want to save it and the metadata cache
        if not all_entries_data: # No existing data and no new data
             logger.info("No existing data and no new data found. Exiting.")
             sys.exit(0) # Exit gracefully if no data at all
    
    new_entries_count = 0
    max_last_edited_time_this_run = last_entry_timestamp # Initialize with previously known max
    
    # Sort blobs by name, assuming it correlates with date for incremental processing
    # Already sorted by list_gcs_blobs
    
    # Limit processing if needed (e.g., only process files modified after last_entry_timestamp)
    # For simplicity, this example re-processes all JSONs from GCS and relies on page_id checks
    # More advanced: compare blob.updated with last_entry_timestamp if exports are idempotent per file.

    for blob_item in export_blobs:
        blob_name = blob_item["name"]
        logger.info(f"Processing GCS export file: gs://{GCS_BUCKET_NAME}/{blob_name}")
        data = download_gcs_blob_as_json(GCS_BUCKET_NAME, blob_name)
        
        if not data or "entries" not in data:
            logger.warning(f"Skipping GCS file {blob_name}: No data or 'entries' key missing.")
            continue
                
        current_file_entries = data["entries"]
        logger.info(f"Loaded {len(current_file_entries)} entries from {blob_name}.")

        for entry in current_file_entries:
            page_id = entry.get("page_id")
            if not page_id:
                logger.warning("Skipping entry with no page_id.")
                continue
                    
            # Check if page_id already processed (for incremental updates from mapping)
            if page_id in processed_page_ids and not args.force_rebuild:
                # Potentially update if entry is newer, but for now, skip if ID seen
                # More complex: compare last_edited_time of current entry with stored one
                logger.debug(f"Skipping already processed page_id: {page_id}")
                continue
                    
            # Track the latest 'last_edited_time'
            entry_last_edited_time = entry.get("last_edited_time")
            if entry_last_edited_time:
                if max_last_edited_time_this_run is None or entry_last_edited_time > max_last_edited_time_this_run:
                    max_last_edited_time_this_run = entry_last_edited_time
            
            # Add to list for embedding
            all_entries_data.append(entry) # Append new or updated entry
            if page_id not in processed_page_ids: # Add to set if truly new
                 processed_page_ids.add(page_id)
                 new_entries_count +=1
            # If page_id was in processed_page_ids but force_rebuild is true, it's already added
            # If it was seen, but this version is newer (and we implement update logic), it would be handled.

    logger.info(f"Total entries to process (including new/updated): {len(all_entries_data)}")
    logger.info(f"Added {new_entries_count} new entries for processing.")

    if not all_entries_data:
        logger.info("No entries to process after loading all data. Exiting.")
        sys.exit(0)

    # --- Prepare for Pinecone upsert ---
    # We need to map Pinecone vector IDs. Using page_id is good if unique and Pinecone allows these chars.
    # Pinecone IDs must be strings.
    
    texts_to_embed_batch = []
    metadata_for_pinecone_batch = []
    pinecone_vectors_batch = [] # For upserting

    # For the local mapping file, we still need a way to map FAISS-like integer index if we were using FAISS
    # But for Pinecone, the ID is the string page_id.
    # The index_mapping.json will now store entries that are IN Pinecone.
    final_mapping_data_for_json = [] 

    logger.info(f"Starting embedding and Pinecone upsert process for {len(all_entries_data)} total entries...")
    
    # Explicitly re-initialize batch lists just before the loop as a diagnostic measure
    texts_to_embed_batch = []
    metadata_for_pinecone_batch = []
    # pinecone_vectors_batch is initialized before the loop and reset within, should be fine.

    # Filter all_entries_data to only include entries that need embedding/upserting
    # If not force_rebuild, an entry might be in all_entries_data from a previous run
    # but already exist in Pinecone. A more robust check would be needed here,
    # e.g., by querying Pinecone for existing IDs if necessary or relying on page_id uniqueness.
    # For this version, we assume if it's in all_entries_data after the loading phase, it needs processing.

    for i, entry in enumerate(all_entries_data):
        page_id = entry.get("page_id")
        content = entry.get("content", "")
        title = entry.get("title", "[No Title]")

        if not page_id:
            logger.warning(f"Entry at index {i} has no page_id. Skipping.")
            continue
        if not content.strip():
            logger.warning(f"Entry with page_id {page_id} ('{title}') has no content. Skipping.")
            continue
        
        texts_to_embed_batch.append(content)
        # Pinecone metadata should not be too large. Select key fields.
        # All fields from entry are candidates for the index_mapping.json
        pinecone_meta = {
            "title": title,
            "entry_date": entry.get("entry_date"),
            "page_id": page_id,
            "url": entry.get("url"),
            # Add other small, filterable fields if needed, ensure they are of supported types
            "tags": entry.get("tags", []), # Example: ensure it's a list of strings
            "family": entry.get("family", []),
            "friends": entry.get("friends", []),
        }
        # Sanitize metadata for Pinecone (e.g. no None values, correct types)
        pinecone_meta = {k: v for k, v in pinecone_meta.items() if v is not None}

        metadata_for_pinecone_batch.append(pinecone_meta)
        
        # The full entry data is saved in the mapping file
        # Add a temporary 'pinecone_id' for clarity, which is the page_id
        entry_for_mapping = entry.copy() # Avoid modifying original dict if it's referenced elsewhere
        entry_for_mapping['pinecone_id'] = page_id 
        final_mapping_data_for_json.append(entry_for_mapping)


        if len(texts_to_embed_batch) >= BATCH_SIZE or i == len(all_entries_data) - 1:
            logger.info(f"Processing batch of {len(texts_to_embed_batch)} entries for embedding (Total processed so far: {i+1})")
            batch_embeddings = embed_batch(openai_client, texts_to_embed_batch)

            if batch_embeddings and len(batch_embeddings) == len(texts_to_embed_batch):
                for k_batch, emb in enumerate(batch_embeddings):
                    if emb:
                        # ID for pinecone is page_id
                        vector_id = metadata_for_pinecone_batch[k_batch]["page_id"] 
                        pinecone_vectors_batch.append({
                            "id": vector_id,
                            "values": emb,
                            "metadata": metadata_for_pinecone_batch[k_batch]
                        })
                    else:
                        logger.warning(f"No embedding returned for an entry in batch (original index around {i - len(texts_to_embed_batch) + k_batch}). Skipping.")
                
                # Upsert to Pinecone in batches
                if pinecone_vectors_batch:
                    for j in range(0, len(pinecone_vectors_batch), PINECONE_UPSERT_BATCH_SIZE):
                        batch_to_upsert = pinecone_vectors_batch[j:j+PINECONE_UPSERT_BATCH_SIZE]
                        try:
                            logger.info(f"Upserting batch of {len(batch_to_upsert)} vectors to Pinecone index '{pinecone_index_name}'")
                            pinecone_index_instance.upsert(vectors=batch_to_upsert)
                            logger.debug(f"Successfully upserted {len(batch_to_upsert)} vectors.")
                        except PineconeException as e:
                            logger.error(f"Pinecone upsert failed: {e}", exc_info=True)
                            # Decide on error handling: continue, retry batch, or exit
                        except Exception as e:
                            logger.error(f"Unexpected error during Pinecone upsert: {e}", exc_info=True)
                    pinecone_vectors_batch = [] # Reset for next main batch

            else:
                logger.error(f"Failed to get embeddings for batch starting around original index {i - len(texts_to_embed_batch)}. Skipping this batch.")
            
            texts_to_embed_batch = []
            metadata_for_pinecone_batch = []


    logger.info("Finished processing all entries for Pinecone.")

    # --- Save mapping, metadata cache, and timestamp to GCS ---
    # Mapping File
    mapping_content_for_gcs = {
        "description": "Mapping of entries processed and stored in Pinecone, including their full content and metadata.",
        "generated_at": datetime.now().isoformat(),
        "pinecone_index_name": pinecone_index_name,
        "total_entries": len(final_mapping_data_for_json),
        "entries": final_mapping_data_for_json
    }
    mapping_json_string = json.dumps(mapping_content_for_gcs, indent=2)
    gcs_mapping_blob_name = f"{GCS_INDEX_ARTIFACTS_PREFIX}{DEFAULT_MAPPING_FILE}"
    upload_content_to_gcs(mapping_json_string, GCS_BUCKET_NAME, gcs_mapping_blob_name, 'application/json')

    # Metadata Cache
    # The mapping_data for cache generation should be what's actually in Pinecone
    extract_and_save_metadata_cache(final_mapping_data_for_json, DEFAULT_METADATA_CACHE_FILE, GCS_BUCKET_NAME, GCS_INDEX_ARTIFACTS_PREFIX)

    # Last Entry Timestamp
    if max_last_edited_time_this_run:
        gcs_timestamp_blob_name = f"{GCS_INDEX_ARTIFACTS_PREFIX}{DEFAULT_LAST_ENTRY_TIMESTAMP_FILE}"
        upload_content_to_gcs(max_last_edited_time_this_run, GCS_BUCKET_NAME, gcs_timestamp_blob_name, 'text/plain')
    else:
        logger.warning("No last_edited_time found in any processed entry. Timestamp file will not be updated/created on GCS.")

    # Schema file - Assuming schema.json is present locally or generated by another process
    # and needs to be uploaded to GCS for the backend to use.
    # This part needs clarification on how schema.json is sourced.
    # If it's static or generated by cli.py, it needs to be present for upload here.
    local_schema_file_path = args.schema_file_name # e.g., "schema.json"
    if os.path.exists(local_schema_file_path):
        gcs_schema_blob_name = f"{GCS_INDEX_ARTIFACTS_PREFIX}{args.schema_file_name}"
        upload_file_to_gcs(local_schema_file_path, GCS_BUCKET_NAME, gcs_schema_blob_name)
    else:
        logger.warning(f"Schema file '{local_schema_file_path}' not found locally. Cannot upload to GCS.")


    logger.info(f"Index build process complete. Artifacts uploaded to GCS gs://{GCS_BUCKET_NAME}/{GCS_INDEX_ARTIFACTS_PREFIX}")


if __name__ == "__main__":
    # Configure basic logging if running standalone for setup, before config.setup_logging takes over
    if not logger.hasHandlers():
         logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    
    # Note: Pinecone operations can be slow. Consider if main() should be async if there are many awaits.
    # For now, Pinecone client itself is synchronous. Batch embedding is also blocking.
    # If GCS client operations were async, then asyncio.run would be needed.
    # For now, storage.Client() is synchronous.
    asyncio.run(main()) 