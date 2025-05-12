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

# Simple retry logic for embedding API calls
MAX_EMBEDDING_RETRIES = 3
EMBEDDING_RETRY_DELAY = 5 # seconds
BATCH_SIZE = 100 # Number of texts to send in each embedding request
PINECONE_UPSERT_BATCH_SIZE = 100 # Batch size for upserting vectors to Pinecone

# Global Pinecone index instance
pinecone_index_instance = None

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

def extract_and_save_metadata_cache(mapping_data: list[dict], cache_file_path: str):
    """Extracts distinct values for specified fields and saves them to a cache file."""
    logger.info("Extracting distinct values for metadata cache...")
    # Define which fields to extract distinct values from 
    fields_to_extract = ["Family", "Friends", "Tags"] # Removed "Food"
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

    # Convert sets to sorted lists 
    final_cache_data = {field: sorted(list(values)) for field, values in distinct_metadata_values.items()}
        
    logger.info("Finished extracting distinct values for cache.")
    if extraction_errors > 0:
        logger.warning(f"CacheGen: Encountered {extraction_errors} unexpected data type issues.")

    # Prepare final cache object
    cache_content = {
        "last_updated": datetime.now().isoformat(),
        "distinct_values": final_cache_data
    }
    
    # Save the cache
    try:
        with open(cache_file_path, 'w', encoding='utf-8') as f:
            json.dump(cache_content, f, indent=2)
        logger.info(f"Successfully saved metadata cache to {cache_file_path}")
    except Exception as e:
        logger.error(f"Failed to save metadata cache to {cache_file_path}: {e}", exc_info=True)

# +++ Helper function for Vercel Blob Upload +++
def upload_file_to_vercel_blob(local_file_path: str, blob_pathname: str, token: str):
    VERCEL_BLOB_STORE_ID_URL_PART = os.getenv("VERCEL_BLOB_STORE_ID_URL_PART")
    if not VERCEL_BLOB_STORE_ID_URL_PART:
        logger.error("VERCEL_BLOB_STORE_ID_URL_PART environment variable not set. Cannot upload.")
        return None

    # Ensure blob_pathname does not start with a slash for URL construction,
    # but preserve it for user-facing messages if needed.
    clean_blob_pathname = blob_pathname.lstrip('/')
    upload_url = f"https://{VERCEL_BLOB_STORE_ID_URL_PART}/{clean_blob_pathname}"
    
    content_type, _ = mimetypes.guess_type(local_file_path)
    if content_type is None:
        content_type = "application/octet-stream" # Default MIME type

    headers = {
        "Authorization": f"Bearer {token}",
        # Vercel's JS SDK @vercel/blob uses x-api-version and other x- headers.
        # For basic PUT, Content-Type is standard.
        # We'll add 'x-content-type' as per @vercel/blob's observed behavior,
        # and 'x-add-random-suffix' which is a common and useful option.
        "x-content-type": content_type,
        "x-add-random-suffix": "true", # Default in JS SDK, usually good. Set to "false" to disable.
        # "x-cache-control-max-age": "31536000", # Optional: 1 year, Vercel's default
    }

    with open(local_file_path, "rb") as f:
        file_data = f.read()
    
    logger.info(f"Attempting to PUT to Vercel Blob: {upload_url} for local file: {local_file_path}")
    logger.debug(f"Upload headers (excluding Authorization): {{'x-content-type': '{headers['x-content-type']}', 'x-add-random-suffix': '{headers['x-add-random-suffix']}'}}")
    
    try:
        response = requests.put(upload_url, headers=headers, data=file_data, timeout=60) # 60s timeout
        response.raise_for_status() 
        response_json = response.json()
        logger.info(f"Successfully uploaded {local_file_path} as {blob_pathname}. Blob URL: {response_json.get('url')}")
        return response_json
    except requests.exceptions.HTTPError as e_http:
        logger.error(f"HTTP error uploading {local_file_path} to Vercel Blob at {upload_url}: {e_http}")
        if e_http.response is not None:
            logger.error(f"Response status: {e_http.response.status_code}, Response text: {e_http.response.text}")
        return None
    except requests.exceptions.RequestException as e_req:
        logger.error(f"RequestException (e.g., connection error, timeout) uploading {local_file_path} to Vercel Blob at {upload_url}: {e_req}")
        return None
    except json.JSONDecodeError as e_json:
        logger.error(f"Failed to decode JSON response after uploading {local_file_path} to Vercel Blob. Status: {response.status_code}, Response text: {response.text[:200]}...")
        return None # Or return a partial success if appropriate
# --- End Helper function ---

async def main():
    parser = argparse.ArgumentParser(description="Build or update a Pinecone index from exported Notion data JSON files.") # Updated description
    parser.add_argument(
        "-i", "--input",
        nargs='+', # Accept one or more arguments
        default=[DEFAULT_INPUT_GLOB],
        help=f"Path(s) or glob pattern(s) for input JSON files (default: '{DEFAULT_INPUT_GLOB}')"
    )
    # parser.add_argument(
    #     "--index-file",
    #     default=DEFAULT_INDEX_FILE,
    #     help=f"Path to load/save the FAISS index file (default: {DEFAULT_INDEX_FILE})"
    # ) # This argument is no longer used for local FAISS file
    parser.add_argument(
        "--mapping-file",
        default=DEFAULT_MAPPING_FILE,
        help=f"Path to load/save the index-to-entry mapping JSON file (default: {DEFAULT_MAPPING_FILE})"
    )
    parser.add_argument(
        "--last-entry-timestamp-file",
        default=DEFAULT_LAST_ENTRY_TIMESTAMP_FILE,
        help=f"Path to load/save the last processed entry timestamp file (default: {DEFAULT_LAST_ENTRY_TIMESTAMP_FILE})"
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Ignore existing index and mapping files and build from scratch."
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging (DEBUG level)."
    )
    parser.add_argument(
        "--upload-to-blob",
        action="store_true",
        help="Upload generated index files to Vercel Blob storage."
    )

    args = parser.parse_args()

    # --- Setup Logging ---
    log_level = logging.DEBUG if args.verbose else logging.INFO
    config.setup_logging(level=log_level)

    logger.info(f"Starting Pinecone index build/update process...") # Updated log message
    logger.info(f"Input pattern(s): {args.input}")
    # logger.info(f"Index file: {args.index_file}") # No longer relevant for local FAISS file
    logger.info(f"Mapping file: {args.mapping_file}")
    logger.info(f"Last entry timestamp file: {args.last_entry_timestamp_file}") # Log new file
    logger.info(f"Force rebuild: {args.force_rebuild}")
    logger.info(f"Batch size: {BATCH_SIZE}")
    logger.info(f"Upload to Vercel Blob: {args.upload_to_blob}") # Log the new argument

    # --- Resolve Input Files ---
    input_files = []
    for pattern in args.input:
        resolved_files = glob.glob(pattern)
        if not resolved_files:
            logger.warning(f"Input pattern '{pattern}' did not match any files.")
        input_files.extend(resolved_files)
    
    if not input_files:
        logger.error("No valid input JSON files found based on input patterns. Aborting.")
        sys.exit(1)
        
    # Sort files to process them in a consistent order (e.g., chronologically)
    input_files.sort()
    logger.info(f"Found {len(input_files)} input files to process: {input_files}")

    # --- Initialize Pinecone Client and Index Instance ---
    global pinecone_index_instance
    try:
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
        if not pinecone_api_key or not pinecone_index_name:
            logger.critical("PINECONE_API_KEY and PINECONE_INDEX_NAME environment variables must be set.")
            sys.exit(1)
        
        logger.info(f"Initializing Pinecone client and connecting to index: '{pinecone_index_name}'...")
        pc = Pinecone(api_key=pinecone_api_key)
        # Ensure index exists, or handle creation if desired (TDD assumes index exists)
        # if pinecone_index_name not in pc.list_indexes().names:
        #     logger.critical(f"Pinecone index '{pinecone_index_name}' does not exist. Please create it first.")
        #     sys.exit(1)
        pinecone_index_instance = pc.Index(pinecone_index_name)
        logger.info(f"Successfully connected to Pinecone index: '{pinecone_index_name}'. Index stats: {pinecone_index_instance.describe_index_stats()}")
    except PineconeException as e:
        logger.critical(f"Pinecone initialization failed: {e}", exc_info=True)
        sys.exit(1)
    except Exception as e:
        logger.critical(f"An unexpected error occurred during Pinecone initialization: {e}", exc_info=True)
        sys.exit(1)

    # --- Load or Initialize Index and Mapping ---
    # index = None # This was for FAISS index object
    mapping_data = []
    processed_page_ids = set()
    current_max_last_edited_time = None # Reset if error

    if args.force_rebuild:
        logger.info("Force rebuild requested. Clearing existing data from Pinecone index and local mapping.")
        try:
            index_stats = pinecone_index_instance.describe_index_stats()
            if index_stats.total_vector_count > 0:
                logger.info(f"Deleting all vectors from Pinecone index '{pinecone_index_name}' as it has {index_stats.total_vector_count} vectors...")
                pinecone_index_instance.delete(delete_all=True)
                logger.info("All vectors deleted from Pinecone index.")
            else:
                logger.info(f"Pinecone index '{pinecone_index_name}' is already empty. No deletion needed.")
            
            # Reset local data structures
            mapping_data = []
            processed_page_ids = set()
            current_max_last_edited_time = None # Reset timestamp for full rebuild
            # Also delete local mapping and timestamp files to ensure clean state
            if os.path.exists(args.mapping_file):
                os.remove(args.mapping_file)
                logger.info(f"Removed existing local mapping file: {args.mapping_file}")
            if os.path.exists(args.last_entry_timestamp_file):
                os.remove(args.last_entry_timestamp_file)
                logger.info(f"Removed existing local timestamp file: {args.last_entry_timestamp_file}")
        except PineconeException as e:
            logger.critical(f"Failed to delete all vectors from Pinecone index '{pinecone_index_name}': {e}", exc_info=True)
            sys.exit(1)
        except Exception as e:
            logger.critical(f"An unexpected error occurred during force_rebuild cleanup: {e}", exc_info=True)
            sys.exit(1)
    else:
        # Try to load existing mapping file to continue incrementally
        if os.path.exists(args.mapping_file):
            logger.info(f"Attempting to load existing mapping from {args.mapping_file}...")
            try:
                with open(args.mapping_file, 'r', encoding='utf-8') as f_map:
                    loaded_mapping = json.load(f_map)
                    if isinstance(loaded_mapping, list):
                        mapping_data = loaded_mapping
                        # Populate processed_page_ids from the loaded mapping_data
                        # Assuming each entry in mapping_data has a 'page_id'
                        for entry in mapping_data:
                            if isinstance(entry, dict) and 'page_id' in entry:
                                processed_page_ids.add(str(entry['page_id'])) # Ensure string ID
                        logger.info(f"Loaded {len(mapping_data)} entries from existing mapping. Processed {len(processed_page_ids)} page IDs.")
                    else:
                        logger.warning(f"Mapping file {args.mapping_file} does not contain a list. Starting fresh.")
                        mapping_data = [] # Reset if format is incorrect
                        processed_page_ids = set()
            except json.JSONDecodeError:
                logger.warning(f"Could not decode JSON from mapping file {args.mapping_file}. Starting fresh.")
                mapping_data = []
                processed_page_ids = set()
            except Exception as e:
                logger.warning(f"Error loading mapping file {args.mapping_file}: {e}. Starting fresh.", exc_info=True)
                mapping_data = []
                processed_page_ids = set()
        else:
            logger.info("No existing mapping file found. Starting fresh.")
            mapping_data = []
            processed_page_ids = set()

    # Dimension is fixed by Pinecone index (e.g., 1536 for text-embedding-ada-002)
    # No need to load FAISS index to get dimension or existing vectors from it.
    # dimension = None 

    # --- Load existing last_entry_update_timestamp if not forcing rebuild ---
    # This logic is now placed before attempting to load mapping_data
    # if not args.force_rebuild and os.path.exists(args.last_entry_timestamp_file):
    # ... (this block was moved up)

    # if not args.force_rebuild and os.path.exists(args.index_file) and os.path.exists(args.mapping_file):
    #     logger.info(f"Attempting to load existing index from {args.index_file} and mapping from {args.mapping_file}...")
    #     try:
    #         # Load FAISS index
    #         index = faiss.read_index(args.index_file)
    #         dimension = index.d
    #         logger.info(f"FAISS index loaded. Dimension: {dimension}, Total vectors: {index.ntotal}")
            
    #         # Load mapping data
    #         with open(args.mapping_file, 'r') as f_map:
    #             mapping_data = json.load(f_map)
    #         if len(mapping_data) != index.ntotal:
    #             logger.warning(
    #                 f"Mismatch: FAISS index has {index.ntotal} vectors, mapping has {len(mapping_data)} entries. "
    #                 "This might lead to issues. Consider --force-rebuild."
    #             )
    #         # Populate processed_page_ids from the loaded mapping_data
    #         for entry in mapping_data:
    #             if 'page_id' in entry:
    #                 processed_page_ids.add(entry['page_id'])
    #         logger.info(f"Loaded {len(mapping_data)} entries from mapping. Processed {len(processed_page_ids)} page IDs.")

    #     except Exception as e:
    #         logger.error(f"Error loading existing index/mapping: {e}. Consider --force-rebuild.", exc_info=True)
    #         logger.warning("Proceeding by initializing a new index/mapping.")
    #         index = None
    #         mapping_data = []
    #         processed_page_ids = set()
    # else:
    #     if args.force_rebuild:
    #         logger.info("Force rebuild requested. Initializing new index and mapping.")
    #     else:
    #         logger.info("No existing index/mapping found or files missing. Initializing new index and mapping.")
    #     # index object will be created once the dimension is known
    #     mapping_data = []
    #     processed_page_ids = set()

    # --- Initialize OpenAI Client ---
    logger.info("Initializing OpenAI client...")
    try:
        if not config.OPENAI_API_KEY:
             logger.error("OPENAI_API_KEY not found in environment or .env file.")
             sys.exit(1)
        openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
    except ValueError as e:
        logger.error(f"OpenAI Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}", exc_info=True)
        sys.exit(1)

    # --- Process Files, Generate Embeddings, Update Index --- 
    logger.info("Processing input files incrementally...")
    total_entries_processed_this_run = 0
    total_entries_skipped_this_run = 0
    total_embeddings_added_this_run = 0
    files_processed_count = 0
    logged_one_example = False # Flag to log only once

    # Initialize overall_max_last_edited_time with the loaded value or None
    overall_max_last_edited_time = current_max_last_edited_time

    # --- Batching for embeddings and Pinecone upserts ---
    texts_to_embed_batch = []
    metadata_for_embedding_batch = [] # Stores corresponding entry dicts for the texts

    # --- Main Processing Loop ---
    for file_path in input_files:
        logger.info(f"Processing file: {file_path}...")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if not isinstance(data, dict) or "entries" not in data or not isinstance(data["entries"], list):
                    logger.error(f"JSON file {file_path} is not in the expected format (object with 'entries' list). Skipping.")
                    continue
                
                entries_in_file = data["entries"]
                logger.info(f"Found {len(entries_in_file)} entries in {file_path}.")
                
                file_entries_processed = 0
                file_entries_skipped_due_to_timestamp = 0
                file_entries_skipped_due_to_missing_id = 0
                file_entries_skipped_due_to_duplicate = 0
                file_embeddings_added = 0

                for entry_data in entries_in_file:
                    page_id = entry_data.get("page_id")
                    if not page_id:
                        logger.warning(f"Entry missing 'page_id' in file {file_path}. Skipping: {entry_data.get('title', 'N/A')}")
                        file_entries_skipped_due_to_missing_id += 1
                        total_entries_skipped_this_run += 1
                        continue
                    
                    page_id_str = str(page_id) # Ensure string ID

                    # Incremental update: Check last_edited_time
                    entry_last_edited_str = entry_data.get("last_edited_time")
                    if entry_last_edited_str and current_max_last_edited_time:
                        try:
                            entry_last_edited_dt = datetime.fromisoformat(entry_last_edited_str)
                            if entry_last_edited_dt <= current_max_last_edited_time:
                                # Only skip if page_id already processed to handle re-runs where timestamp file exists but mapping doesn't
                                if page_id_str in processed_page_ids:
                                    file_entries_skipped_due_to_timestamp += 1
                                    total_entries_skipped_this_run += 1
                                    continue 
                                else:
                                    logger.debug(f"Entry {page_id_str} is old but not in processed_page_ids, processing anyway.")
                        except ValueError:
                            logger.warning(f"Could not parse last_edited_time '{entry_last_edited_str}' for page {page_id_str}. Processing entry.")

                    if page_id_str in processed_page_ids and not args.force_rebuild: # Check if already processed (and not forcing rebuild)
                        logger.debug(f"Page ID {page_id_str} already processed and not forcing rebuild. Skipping.")
                        file_entries_skipped_due_to_duplicate +=1
                        total_entries_skipped_this_run +=1
                        continue

                    content_to_embed = entry_data.get("content", "")
                    if not content_to_embed.strip():
                        logger.warning(f"Entry {page_id_str} has no content. Skipping embedding.")
                        file_entries_skipped_due_to_missing_id +=1 # Or a different counter for no content
                        total_entries_skipped_this_run +=1
                        continue
                    
                    texts_to_embed_batch.append(content_to_embed)
                    # Store the original entry_data; it contains page_id and other metadata for mapping and Pinecone
                    metadata_for_embedding_batch.append(entry_data) 

                    if len(texts_to_embed_batch) >= BATCH_SIZE:
                        # Process the batch for embeddings and then prepare for Pinecone upsert
                        num_added_from_batch = await process_and_upsert_batch(
                            openai_client,
                            texts_to_embed_batch,
                            metadata_for_embedding_batch,
                            mapping_data, # list to append successful mappings
                            processed_page_ids, # set to add successful page_ids
                            pinecone_index_name # Pass pinecone_index_name
                        )
                        total_embeddings_added_this_run += num_added_from_batch
                        file_embeddings_added += num_added_from_batch
                        texts_to_embed_batch = []
                        metadata_for_embedding_batch = []
                    
                    file_entries_processed += 1
                    total_entries_processed_this_run +=1

                    # Update overall_max_last_edited_time
                    if entry_last_edited_str:
                        try:
                            entry_dt = datetime.fromisoformat(entry_last_edited_str)
                            if overall_max_last_edited_time is None or entry_dt > overall_max_last_edited_time:
                                overall_max_last_edited_time = entry_dt
                        except ValueError:
                            pass # Already warned

                logger.info(f"Finished processing {file_path}: "
                            f"{file_entries_processed} entries considered for processing, "
                            f"{file_embeddings_added} new embeddings added/updated in Pinecone, "
                            f"{file_entries_skipped_due_to_timestamp} skipped (timestamp), "
                            f"{file_entries_skipped_due_to_missing_id} skipped (missing id/content), "
                            f"{file_entries_skipped_due_to_duplicate} skipped (duplicate ID).")
                files_processed_count +=1

        except FileNotFoundError:
            logger.error(f"Input file {file_path} not found. Skipping.")
        except json.JSONDecodeError:
            logger.error(f"Could not decode JSON from {file_path}. Skipping.")
        except Exception as e:
            logger.error(f"Unexpected error processing file {file_path}: {e}", exc_info=True)

    # Process any remaining texts in the last batch
    if texts_to_embed_batch:
        logger.info(f"Processing final batch of {len(texts_to_embed_batch)} entries...")
        num_added_from_batch = await process_and_upsert_batch(
            openai_client,
            texts_to_embed_batch,
            metadata_for_embedding_batch,
            mapping_data,
            processed_page_ids,
            pinecone_index_name # Pass pinecone_index_name
        )
        total_embeddings_added_this_run += num_added_from_batch
        texts_to_embed_batch = []
        metadata_for_embedding_batch = []

    logger.info(f"--- Indexing Summary ---")
    logger.info(f"Processed {files_processed_count} files.")
    logger.info(f"Total entries considered for processing this run: {total_entries_processed_this_run}")
    logger.info(f"Total new embeddings added/updated in Pinecone: {total_embeddings_added_this_run}")
    logger.info(f"Total entries skipped (timestamp, duplicate, missing_id/content): {total_entries_skipped_this_run}")
    
    if pinecone_index_instance:
        try:
            final_stats = pinecone_index_instance.describe_index_stats()
            logger.info(f"Final Pinecone index '{pinecone_index_name}' stats: {final_stats}")
        except Exception as e:
            logger.warning(f"Could not retrieve final Pinecone index stats: {e}")

    # --- Save Mapping Data ---
    logger.info(f"Saving updated mapping data to {args.mapping_file} ({len(mapping_data)} entries)...")
    try:
        with open(args.mapping_file, 'w', encoding='utf-8') as f_map_out:
            json.dump(mapping_data, f_map_out, indent=4)
        logger.info("Mapping data saved successfully.")
    except Exception as e:
        logger.error(f"Error saving mapping data: {e}", exc_info=True)

    # --- Save Last Entry Update Timestamp ---
    if overall_max_last_edited_time:
        logger.info(f"Saving last entry update timestamp ({overall_max_last_edited_time.isoformat()}) to {args.last_entry_timestamp_file}...")
        try:
            with open(args.last_entry_timestamp_file, 'w', encoding='utf-8') as f_ts_out:
                f_ts_out.write(overall_max_last_edited_time.isoformat())
            logger.info("Last entry update timestamp saved successfully.")
        except Exception as e:
            logger.error(f"Error saving last entry update timestamp: {e}", exc_info=True)
    else:
        logger.info("No new maximum last_edited_time determined in this run; timestamp file not updated.")

    # --- Save Metadata Cache ---
    # (Assuming extract_and_save_metadata_cache is defined elsewhere or called if needed)
    # For this refactor, focusing on Pinecone. This function might need an update if it relies on FAISS specific details.
    # It seems to operate on mapping_data, so it should be mostly fine.
    metadata_cache_file = os.path.join(os.path.dirname(args.mapping_file), "metadata_cache.json") # Assuming it's in the same dir
    extract_and_save_metadata_cache(mapping_data, metadata_cache_file)


    # --- Upload to Vercel Blob if requested ---
    if args.upload_to_blob:
        logger.info("Uploading generated files to Vercel Blob storage...")
        blob_token = os.getenv("VERCEL_BLOB_ACCESS_TOKEN")
        if not blob_token:
            logger.error("VERCEL_BLOB_ACCESS_TOKEN not set. Cannot upload.")
        elif not os.getenv("VERCEL_BLOB_STORE_ID_URL_PART"):
            logger.error("VERCEL_BLOB_STORE_ID_URL_PART environment variable not set. Cannot construct upload URL.")
        else:
            # Define files to upload and their desired pathnames in the blob store
            # The pathname is relative to the root of your blob store.
            files_to_upload_info = [
                {"local_path": args.mapping_file, "blob_pathname": os.path.basename(args.mapping_file)},
                {"local_path": args.last_entry_timestamp_file, "blob_pathname": os.path.basename(args.last_entry_timestamp_file)},
                {"local_path": metadata_cache_file, "blob_pathname": os.path.basename(metadata_cache_file)},
                # Add schema.json if it's generated and needs to be uploaded
                # {"local_path": "schema.json", "blob_pathname": "schema.json"}, 
            ]

            successful_uploads = 0
            for file_info in files_to_upload_info:
                local_p = file_info["local_path"]
                blob_pn = file_info["blob_pathname"]
                
                if not os.path.exists(local_p):
                    logger.warning(f"Local file {local_p} for blob pathname {blob_pn} does not exist. Skipping upload.")
                    continue
                
                logger.info(f"Preparing to upload {local_p} to Vercel Blob as {blob_pn}...")
                upload_response = upload_file_to_vercel_blob(
                    local_file_path=local_p,
                    blob_pathname=blob_pn,
                    token=blob_token
                )
                if upload_response and upload_response.get("url"):
                    logger.info(f"Successfully uploaded {blob_pn}. Blob URL: {upload_response['url']}")
                    # If you need to store these URLs (e.g., in env vars for the backend), this is where you'd get them.
                    # For now, we just log. The backend will download using predefined pathnames.
                    successful_uploads += 1
                else:
                    logger.error(f"Failed to upload {blob_pn} from {local_p}.")
            
            if successful_uploads == len([f for f in files_to_upload_info if os.path.exists(f["local_path"])]):
                logger.info("All existing local files successfully uploaded to Vercel Blob.")
            else:
                logger.warning("Some files failed to upload to Vercel Blob. Check logs for details.")
        logger.info("Vercel Blob upload process finished (or skipped).")


    logger.info("Build process finished.")


async def process_and_upsert_batch(openai_client, texts_to_embed, metadata_list, mapping_data_ref, processed_page_ids_ref, pinecone_index_name: str):
    """
    Embeds a batch of texts and upserts them to Pinecone.
    Updates mapping_data_ref and processed_page_ids_ref with successful entries.
    Returns the number of embeddings successfully added.
    """
    global pinecone_index_instance # Ensure access to the global Pinecone index instance
    embeddings_added_count = 0

    if not texts_to_embed:
        return 0

    logger.debug(f"Embedding batch of {len(texts_to_embed)} texts...")
    batch_embeddings = embed_batch(openai_client, texts_to_embed)

    if batch_embeddings is None or len(batch_embeddings) != len(texts_to_embed):
        logger.error(f"Failed to get embeddings for batch or mismatched length. Skipping {len(texts_to_embed)} entries.")
        # Note: metadata_list corresponds to texts_to_embed, so all these are skipped.
        return 0 # No embeddings added from this batch

    vectors_to_upsert = []
    successful_metadata_for_mapping = []

    for i, embedding in enumerate(batch_embeddings):
        if embedding is None:
            logger.warning(f"Got None embedding for text: '{texts_to_embed[i][:50]}...'. Skipping this entry.")
            continue

        entry_metadata = metadata_list[i] # Original entry data
        page_id = entry_metadata.get("page_id")
        if not page_id: # Should have been caught earlier, but double check
            logger.error(f"CRITICAL: Entry metadata missing 'page_id' at embedding stage. Text: '{texts_to_embed[i][:50]}...'. Skipping.")
            continue
        
        page_id_str = str(page_id)

        # Prepare metadata for Pinecone upsert.
        # Crucially include page_id for filtering in rag_query.py
        pinecone_meta = {
            "page_id": page_id_str,
            "title": entry_metadata.get("title", "Untitled"),
            # Add other relevant, filterable metadata if desired:
            # "entry_date": entry_metadata.get("entry_date_str_iso"), # If converted to ISO string
            # "tags": entry_metadata.get("Tags", []) # Example if Tags are simple list of strings
        }
        # Ensure all metadata values are Pinecone compatible (str, num, bool, or list of str)
        # For example, if 'entry_date_str_iso' is added, ensure it's an ISO string.

        vectors_to_upsert.append({
            "id": page_id_str,       # String ID
            "values": embedding,     # Dense vector
            "metadata": pinecone_meta # Metadata payload
        })
        successful_metadata_for_mapping.append(entry_metadata) # This entry is ready for mapping_data

    if not vectors_to_upsert:
        logger.info("No valid vectors prepared in this batch for Pinecone upsert.")
        return 0

    # Upsert to Pinecone in batches (though `vectors_to_upsert` is already one batch from embedding)
    # Pinecone client's upsert can handle lists up to a certain size effectively.
    # If vectors_to_upsert itself becomes extremely large (many thousands), then further sub-batching here might be needed.
    # For now, assuming PINECONE_UPSERT_BATCH_SIZE is for controlling sub-batches if this list is huge.
    # The current BATCH_SIZE for embeddings is 100, so vectors_to_upsert will have at most 100 items.
    # This should be fine for a single Pinecone upsert call.
    
    try:
        logger.debug(f"Upserting {len(vectors_to_upsert)} vectors to Pinecone index '{pinecone_index_name}'...")
        
        # Sub-batching for Pinecone upsert if vectors_to_upsert is larger than PINECONE_UPSERT_BATCH_SIZE
        for j in range(0, len(vectors_to_upsert), PINECONE_UPSERT_BATCH_SIZE):
            batch_to_upsert_pinecone = vectors_to_upsert[j:j + PINECONE_UPSERT_BATCH_SIZE]
            metadata_sub_batch = successful_metadata_for_mapping[j:j + PINECONE_UPSERT_BATCH_SIZE]

            if not batch_to_upsert_pinecone: continue

            upsert_response = await asyncio.to_thread(pinecone_index_instance.upsert, vectors=batch_to_upsert_pinecone)
            
            # Process response and update mapping_data_ref and processed_page_ids_ref
            # for the successfully upserted items in this sub-batch
            # Assuming upsert_response.upserted_count gives us info, or we assume success if no error.
            # For now, if no exception, assume all in batch_to_upsert_pinecone were successful.
            current_batch_upserted_count = upsert_response.upserted_count if hasattr(upsert_response, 'upserted_count') else len(batch_to_upsert_pinecone)
            logger.debug(f"Pinecone upsert response for sub-batch: Upserted count = {current_batch_upserted_count}")

            for k_idx, original_entry_meta in enumerate(metadata_sub_batch):
                # This assumes the order is maintained and success of the batch implies success for all its items.
                # More robust error handling per vector might be needed for production if Pinecone provides it.
                page_id_str_for_map = str(original_entry_meta["page_id"])
                
                # Add to main mapping_data if it's a new ID or if we are rebuilding (force_rebuild handled earlier for clearing)
                # If it's an update to an existing ID, we might need to find and replace in mapping_data_ref
                # For simplicity, if ID exists, we assume its metadata in mapping_data_ref is current or doesn't need changing here.
                # The main purpose of mapping_data is to map ID to full content.
                # If an entry is re-embedded and re-upserted, its content is what was just processed.
                
                existing_entry_index = -1
                if page_id_str_for_map in processed_page_ids_ref: # If ID was already processed (e.g. in a previous file, or loaded mapping)
                    # Find and update existing entry in mapping_data_ref
                    for idx, map_entry in enumerate(mapping_data_ref):
                        if str(map_entry.get("page_id")) == page_id_str_for_map:
                            existing_entry_index = idx
                            break
                    if existing_entry_index != -1:
                        logger.debug(f"Updating existing entry in mapping_data for page_id: {page_id_str_for_map}")
                        mapping_data_ref[existing_entry_index] = original_entry_meta # Replace with the new metadata
                    else: # Should not happen if page_id_str_for_map is in processed_page_ids_ref from loading
                        logger.warning(f"page_id {page_id_str_for_map} was in processed_page_ids_ref but not found in mapping_data_ref. Appending.")
                        mapping_data_ref.append(original_entry_meta)
                else: # New entry
                    mapping_data_ref.append(original_entry_meta)

                processed_page_ids_ref.add(page_id_str_for_map)
                embeddings_added_count +=1 # Count actual successful items based on original metadata count for this batch
        
        logger.info(f"Successfully upserted {embeddings_added_count} vectors from batch to Pinecone.")

    except PineconeException as e:
        logger.error(f"Pinecone API error during upsert: {e}", exc_info=True)
        # Decide how to handle partial failures, for now, assume batch failed if error occurs
        return 0 # No embeddings counted as added from this batch on error
    except Exception as e:
        logger.error(f"Unexpected error during Pinecone upsert: {e}", exc_info=True)
        return 0
        
    return embeddings_added_count


if __name__ == "__main__":
    asyncio.run(main()) 