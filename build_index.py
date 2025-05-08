# build_index.py

import argparse
import logging
import json
import os
import sys
import numpy as np
import faiss
from openai import OpenAI, RateLimitError, APIError
import time
import glob # For handling file globs
from datetime import datetime # Ensure datetime is imported

# Adjust path to import from the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from notion_second_brain import config # Use centralized logging and config

# Setup logger for this module
logger = logging.getLogger("build_index")

# Constants (Consider moving to config or making CLI args if more flexibility needed)
# DEFAULT_INPUT_JSON = "output/all_time.json" # No longer a single default
DEFAULT_INPUT_GLOB = "output/*.json" # Default to process all JSONs in output
DEFAULT_INDEX_FILE = "index.faiss"
DEFAULT_MAPPING_FILE = "index_mapping.json"
EMBEDDING_MODEL = "text-embedding-ada-002"
DEFAULT_LAST_ENTRY_TIMESTAMP_FILE = "last_entry_update_timestamp.txt" # New constant

# Simple retry logic for embedding API calls
MAX_EMBEDDING_RETRIES = 3
EMBEDDING_RETRY_DELAY = 5 # seconds
BATCH_SIZE = 100 # Number of texts to send in each embedding request

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

def main():
    parser = argparse.ArgumentParser(description="Build or update a FAISS index from exported Notion data JSON files.")
    parser.add_argument(
        "-i", "--input",
        nargs='+', # Accept one or more arguments
        default=[DEFAULT_INPUT_GLOB],
        help=f"Path(s) or glob pattern(s) for input JSON files (default: '{DEFAULT_INPUT_GLOB}')"
    )
    parser.add_argument(
        "--index-file",
        default=DEFAULT_INDEX_FILE,
        help=f"Path to load/save the FAISS index file (default: {DEFAULT_INDEX_FILE})"
    )
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

    args = parser.parse_args()

    # --- Setup Logging ---
    log_level = logging.DEBUG if args.verbose else logging.INFO
    config.setup_logging(level=log_level)

    logger.info(f"Starting index build/update process...")
    logger.info(f"Input pattern(s): {args.input}")
    logger.info(f"Index file: {args.index_file}")
    logger.info(f"Mapping file: {args.mapping_file}")
    logger.info(f"Last entry timestamp file: {args.last_entry_timestamp_file}") # Log new file
    logger.info(f"Force rebuild: {args.force_rebuild}")
    logger.info(f"Batch size: {BATCH_SIZE}")

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

    # --- Load or Initialize Index and Mapping ---
    index = None
    mapping_data = []
    processed_page_ids = set()
    dimension = None # Will be determined later
    current_max_last_edited_time = None

    # --- Load existing last_entry_update_timestamp if not forcing rebuild ---
    if not args.force_rebuild and os.path.exists(args.last_entry_timestamp_file):
        try:
            with open(args.last_entry_timestamp_file, 'r', encoding='utf-8') as f_ts:
                timestamp_str = f_ts.read().strip()
                if timestamp_str:
                    current_max_last_edited_time = datetime.fromisoformat(timestamp_str)
                    logger.info(f"Loaded existing last entry update timestamp: {current_max_last_edited_time.isoformat()}")
        except Exception as e:
            logger.warning(f"Could not read or parse existing last entry timestamp file at {args.last_entry_timestamp_file}: {e}. Will determine from scratch.")
            current_max_last_edited_time = None # Reset if error


    if not args.force_rebuild and os.path.exists(args.index_file) and os.path.exists(args.mapping_file):
        logger.info(f"Attempting to load existing index from {args.index_file} and mapping from {args.mapping_file}...")
        try:
            index = faiss.read_index(args.index_file)
            dimension = index.d # Get dimension from loaded index
            logger.info(f"Loaded existing FAISS index with {index.ntotal} vectors (Dimension: {dimension}).")
            
            with open(args.mapping_file, 'r', encoding='utf-8') as f:
                mapping_data = json.load(f)
            logger.info(f"Loaded existing mapping data with {len(mapping_data)} entries.")
            
            # Populate processed IDs set
            for i, item in enumerate(mapping_data):
                page_id = item.get('page_id')
                if page_id:
                    processed_page_ids.add(page_id)
                else:
                    logger.warning(f"Entry at index {i} in existing mapping file has no page_id. Skipping.")
            logger.info(f"Loaded {len(processed_page_ids)} unique page IDs from existing mapping.")

            # Sanity check: number of vectors in index should match mapping entries
            if index.ntotal != len(mapping_data):
                 logger.warning(f"Mismatch between index vectors ({index.ntotal}) and mapping entries ({len(mapping_data)}). Checkpoint might be corrupt. Consider --force-rebuild.")
                 # Decide how to handle: exit, force rebuild, or proceed cautiously?
                 # For now, proceed cautiously, but log a strong warning.

        except FileNotFoundError:
             logger.warning("Index/mapping file not found despite check (race condition?). Initializing new index.")
             index = None
             mapping_data = []
             processed_page_ids = set()
        except Exception as e:
            logger.error(f"Error loading existing index/mapping: {e}. Consider --force-rebuild.", exc_info=True)
            # Decide how to handle: exit or initialize new?
            logger.warning("Proceeding by initializing a new index/mapping.")
            index = None
            mapping_data = []
            processed_page_ids = set()
    else:
        if args.force_rebuild:
            logger.info("Force rebuild requested. Initializing new index and mapping.")
        else:
            logger.info("No existing index/mapping found or files missing. Initializing new index and mapping.")
        # index object will be created once the dimension is known
        mapping_data = []
        processed_page_ids = set()

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

    for file_path in input_files:
        files_processed_count += 1
        logger.info(f"--- Processing file {files_processed_count}/{len(input_files)}: {file_path} ---")
        entries_in_file = 0
        embeddings_added_from_file = 0
        skipped_in_file = 0
        
        # Batch processing buffers for the current file
        batch_texts_to_embed = [] # Renamed for clarity
        batch_metadata_for_mapping = [] # Renamed for clarity

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                current_entries = data.get('entries', [])
            entries_in_file = len(current_entries)
            if not current_entries:
                logger.warning(f"No 'entries' found in {file_path}. Skipping file.")
                continue 
            logger.info(f"Loaded {entries_in_file} entries from {file_path}. Processing in batches of {BATCH_SIZE}...")
        except FileNotFoundError:
            logger.error(f"Input file disappeared: {file_path}. Skipping.")
            continue
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {file_path}: {e}. Skipping file.")
            continue
        except Exception as e:
            logger.error(f"Unexpected error loading {file_path}: {e}. Skipping file.", exc_info=True)
            continue

        # Function to process a completed batch
        def process_embedding_batch(texts_to_embed, metadata_to_map):
            nonlocal index, dimension, mapping_data, processed_page_ids, total_embeddings_added_this_run, embeddings_added_from_file, skipped_in_file, total_entries_skipped_this_run, logged_one_example
            
            if not texts_to_embed:
                return

            logger.debug(f"Processing batch of {len(texts_to_embed)} entries...")
            batch_embeddings = embed_batch(openai_client, texts_to_embed)

            if batch_embeddings is None: # Whole batch failed
                logger.error(f"Failed to get embeddings for batch of size {len(texts_to_embed)}. Skipping these entries.")
                skipped_in_file += len(texts_to_embed)
                return
                
            if len(batch_embeddings) != len(texts_to_embed):
                 logger.error(f"Mismatched results from embed_batch. Expected {len(texts_to_embed)}, got {len(batch_embeddings)}. Skipping batch.")
                 skipped_in_file += len(texts_to_embed)
                 return

            # Process successful embeddings from the batch
            vectors_to_add = []
            mappings_to_add = []
            ids_processed_in_batch = []
            
            for i, embedding in enumerate(batch_embeddings):
                entry_data = metadata_to_map[i]
                page_id = entry_data['page_id'] # Should exist based on earlier checks

                if embedding is None: # Should ideally not happen if batch succeeded, but check just in case
                     logger.warning(f"Embed_batch succeeded but got None embedding for entry {i} (ID: {page_id}). Skipping.")
                     skipped_in_file += 1
                     continue
                
                # Initialize index if first embedding
                if index is None:
                    dimension = len(embedding)
                    logger.info(f"First embedding successful. Detected dimension: {dimension}")
                    try:
                        index = faiss.IndexFlatL2(dimension)
                        logger.info("FAISS index initialized.")
                    except Exception as e:
                        logger.critical(f"Failed to initialize FAISS index: {e}. Aborting.", exc_info=True)
                        sys.exit(1)
                
                # Check dimension consistency
                elif len(embedding) != dimension:
                     logger.error(f"Inconsistent embedding dimension! Expected {dimension}, got {len(embedding)} for entry (ID: {page_id}). Skipping.")
                     skipped_in_file += 1
                     continue
                
                # If valid, prepare for adding
                vectors_to_add.append(embedding)
                
                # --- Create the dictionary for index_mapping.json --- 
                # Start with the essential fields
                mapping_entry = {
                    "page_id": page_id,
                    "title": entry_data.get('title', '[No Title]'),
                    "entry_date": entry_data.get('entry_date'),
                    "source_file": os.path.basename(file_path),
                    "content": entry_data.get('content', '') # Ensure full content is stored
                }
                # Add other relevant metadata fields if they exist in the source entry_data
                # This ensures Family, Friends, Tags, etc. are carried over
                # Use Consistent Capitalization matching Notion Properties / Cache Extraction
                for key in ["Family", "Friends", "Tags", "Food", "AI summary"]:
                    # Check lowercase key from source (transformers.py output)
                    source_key = key.lower().replace(' ', '_') # e.g., "AI summary" -> "ai_summary"
                    # Special case for AI summary potentially
                    if key == "AI summary": source_key = "ai_summary"
                        
                    if source_key in entry_data:
                        mapping_entry[key] = entry_data[source_key]
                # --- End dictionary creation ---

                mappings_to_add.append(mapping_entry)
                ids_processed_in_batch.append(page_id)

            # Add batch to index and mapping if any vectors are valid
            if vectors_to_add:
                try:
                    embeddings_np = np.array(vectors_to_add).astype('float32')
                    index.add(embeddings_np)
                    mapping_data.extend(mappings_to_add)
                    processed_page_ids.update(ids_processed_in_batch)
                    
                    count_added = len(vectors_to_add)
                    embeddings_added_from_file += count_added
                    total_embeddings_added_this_run += count_added
                    logger.debug(f"Added batch of {count_added} embeddings. Index size: {index.ntotal}")
                except Exception as e:
                     logger.error(f"Error adding batch of embeddings to index: {e}. Corresponding mapping entries might not be added correctly.", exc_info=True)
                     skipped_in_file += len(vectors_to_add) # Count these as skipped if add fails
                     # Note: State might be inconsistent here (partially added mapping?) - checkpointing helps mitigate.
        
        # --- Loop through entries, build batches ---   
        for i, entry in enumerate(current_entries):
            total_entries_processed_this_run += 1
            page_id = entry.get('page_id')
            entry_title = entry.get('title', '[No Title]') # Get title early for logging
            entry_last_edited_time_str = entry.get("last_edited_time") # Get last_edited_time

            # --- Skip checks (ID, duplicate, content) ---
            if not page_id:
                logger.warning(f"Entry {i+1}/{entries_in_file} in {file_path} (Title: '{entry_title}') missing page_id. Skipping.")
                skipped_in_file += 1
                continue

            if page_id in processed_page_ids:
                logger.debug(f"Page ID {page_id} (Title: '{entry_title}') already processed. Skipping duplicate.")
                skipped_in_file += 1
                continue

            content = entry.get('content', '')
            # Skip if the main content block is empty/whitespace
            if not content or not content.strip():
                logger.warning(f"Entry {i+1}/{entries_in_file} (ID: {page_id}, Title: '{entry_title}') has empty/blank block content. Skipping embedding.")
                skipped_in_file += 1
                continue
            
            # --- Track overall_max_last_edited_time ---
            if entry_last_edited_time_str:
                try:
                    # Notion's last_edited_time is usually like "2024-04-29T22:06:00.000Z"
                    # Convert 'Z' to '+00:00' for fromisoformat if necessary
                    entry_dt = datetime.fromisoformat(entry_last_edited_time_str.replace('Z', '+00:00'))
                    if overall_max_last_edited_time is None or entry_dt > overall_max_last_edited_time:
                        overall_max_last_edited_time = entry_dt
                        logger.debug(f"New overall_max_last_edited_time: {overall_max_last_edited_time.isoformat()} from page {page_id}")
                except ValueError as ve:
                    logger.warning(f"Could not parse last_edited_time '{entry_last_edited_time_str}' for page {page_id} (Title: '{entry_title}'). Error: {ve}. Skipping for timestamp tracking.")
            # --- End Track overall_max_last_edited_time ---
            
            # --- Construct enriched text for embedding --- 
            metadata_parts = []
            # Use the specific keys identified from transformers.py
            if entry_title != '[No Title]':
                 metadata_parts.append(f"Title: {entry_title}")
            if entry.get('entry_date'):
                 metadata_parts.append(f"Date: {entry['entry_date']}")
            if entry.get('tags'):
                 metadata_parts.append(f"Tags: {", ".join(entry['tags'])}")
            if entry.get('food'):
                 metadata_parts.append(f"Food: {", ".join(entry['food'])}")
            if entry.get('friends'):
                 metadata_parts.append(f"Friends: {", ".join(entry['friends'])}")
            if entry.get('family'):
                 metadata_parts.append(f"Family: {", ".join(entry['family'])}")
            if entry.get('ai_summary'):
                 metadata_parts.append(f"AI Summary: {entry['ai_summary']}")
                 
            # Combine metadata and content
            # Add newlines between parts for clarity and potentially better embedding distinction
            combined_text_for_embedding = "\n".join(metadata_parts) + "\n\nContent:\n" + content
            
            # --- >>> ADD DEBUG LOGGING (ONCE) <<< ---
            if not logged_one_example:
                logger.info("--- EXAMPLE EMBEDDING TEXT START ---")
                logger.info(combined_text_for_embedding)
                logger.info("--- EXAMPLE EMBEDDING TEXT END ---")
                logged_one_example = True
            # --- >>> END DEBUG LOGGING <<< ---
            
            # --- Add to current batch --- 
            batch_texts_to_embed.append(combined_text_for_embedding) 
            batch_metadata_for_mapping.append(entry) 
            
            # If batch is full, process it
            if len(batch_texts_to_embed) >= BATCH_SIZE:
                # Pass the correct buffers to the processing function
                process_embedding_batch(batch_texts_to_embed, batch_metadata_for_mapping)
                # Clear the batch buffers
                batch_texts_to_embed = []
                batch_metadata_for_mapping = []
        
        # Process any remaining items in the last batch for this file
        if batch_texts_to_embed:
            logger.info(f"Processing final batch of {len(batch_texts_to_embed)} entries for {file_path}...")
            process_embedding_batch(batch_texts_to_embed, batch_metadata_for_mapping)
            batch_texts_to_embed = [] 
            batch_metadata_for_mapping = []
        
        # Update total skips and log file summary
        total_entries_skipped_this_run += skipped_in_file
        logger.info(f"Finished processing {file_path}. Added {embeddings_added_from_file} new embeddings. Skipped {skipped_in_file} entries (duplicates/errors/empty). Index size: {index.ntotal if index else 0}.")

        # --- Save Checkpoint After Each File ---
        if index is not None and embeddings_added_from_file > 0: # Only save if something was potentially added from this file
            logger.info(f"Saving checkpoint after processing {file_path}... Index size: {index.ntotal}, Mapping size: {len(mapping_data)}")
            checkpoint_saved = False
            try:
                faiss.write_index(index, args.index_file)
                logger.info(f"Checkpoint index saved successfully to {args.index_file}.")
                checkpoint_saved = True 
            except Exception as e:
                logger.error(f"Failed to save checkpoint FAISS index after {file_path}: {e}", exc_info=True)
                logger.warning("Continuing to next file despite index checkpoint save failure.")

            if checkpoint_saved: 
                try:
                    with open(args.mapping_file, 'w', encoding='utf-8') as f:
                        json.dump(mapping_data, f, indent=2)
                    logger.info(f"Checkpoint mapping data saved successfully to {args.mapping_file}.")
                except Exception as e:
                    logger.error(f"Failed to save checkpoint mapping data after {file_path}: {e}", exc_info=True)
                    logger.warning("Continuing despite mapping checkpoint save failure. Index/mapping may be out of sync!")
        elif embeddings_added_from_file == 0:
             logger.info(f"No new embeddings added from {file_path}. Skipping checkpoint save.")

    # --- Final Save Check (Consolidated) ---
    logger.info("Performing final save check...")
    if index is not None: 
        logger.info(f"Finalizing save of index ({index.ntotal} vectors) and mapping ({len(mapping_data)} entries)...")
        final_save_ok = True
        try:
            faiss.write_index(index, args.index_file)
            logger.info(f"Final index saved successfully to {args.index_file}.")
        except Exception as e:
            logger.error(f"Failed to save final FAISS index: {e}", exc_info=True)
            final_save_ok = False
        
        try:
            with open(args.mapping_file, 'w', encoding='utf-8') as f:
                json.dump(mapping_data, f, indent=2)
            logger.info(f"Final mapping data saved successfully to {args.mapping_file}.")
        except Exception as e:
            logger.error(f"Failed to save final mapping data: {e}", exc_info=True)
            final_save_ok = False
            
        if not final_save_ok:
             logger.error("One or both final save operations failed. Index and mapping may be corrupt or out of sync!")
    else:
         logger.warning("No embeddings generated or index not initialized. Nothing to save.")

    # --- Final Save & Summary --- 
    final_mapping_data = mapping_data # Assuming mapping_data holds the final full map here
    
    # --- >>> NEW: Generate and Save Metadata Cache <<< ---
    metadata_cache_file = "metadata_cache.json" # Define cache filename
    extract_and_save_metadata_cache(final_mapping_data, metadata_cache_file)
    # --- >>> END: Generate and Save Metadata Cache <<< ---

    # --- Summary Logging ---
    logger.info("--- Index build/update process finished ---")
    logger.info(f"Total entries processed across files: {total_entries_processed_this_run}")
    logger.info(f"Total entries skipped (duplicates/errors/empty): {total_entries_skipped_this_run}")
    logger.info(f"Total new embeddings added to index: {total_embeddings_added_this_run}")
    logger.info(f"Final index size: {index.ntotal if index else 0} vectors")
    logger.info(f"Final mapping size: {len(mapping_data)} entries")

    if index is None or index.ntotal == 0:
        logger.warning("No vectors were added to the index. Index file and mapping will not be saved.")
        # Also don't save metadata cache or last entry timestamp if nothing was indexed
    else:
        # Save FAISS index
        try:
            faiss.write_index(index, args.index_file)
            logger.info(f"FAISS index saved to {args.index_file} with {index.ntotal} vectors.")
        except Exception as e:
            logger.error(f"Error saving FAISS index: {e}", exc_info=True)

        # Save mapping data
        try:
            with open(args.mapping_file, 'w', encoding='utf-8') as f:
                json.dump(mapping_data, f, indent=2)
            logger.info(f"Index-to-entry mapping saved to {args.mapping_file} with {len(mapping_data)} entries.")
        except Exception as e:
            logger.error(f"Error saving mapping data: {e}", exc_info=True)
            
        # Save metadata cache
        extract_and_save_metadata_cache(mapping_data, "metadata_cache.json") # Assuming fixed name for now

        # Save the overall maximum last_edited_time
        if overall_max_last_edited_time:
            try:
                with open(args.last_entry_timestamp_file, 'w', encoding='utf-8') as f_ts:
                    f_ts.write(overall_max_last_edited_time.isoformat())
                logger.info(f"Last processed entry timestamp saved to {args.last_entry_timestamp_file}: {overall_max_last_edited_time.isoformat()}")
            except Exception as e:
                logger.error(f"Error saving last processed entry timestamp: {e}", exc_info=True)
        else:
            logger.info("No last_edited_time was found in processed entries. Timestamp file not updated/created.")


    logger.info("Index build/update process finished.")

if __name__ == "__main__":
    # Check for OPENAI_API_KEY early
    try:
        _ = config.OPENAI_API_KEY 
    except ValueError as e:
         print(f"Error: {e}", file=sys.stderr)
         sys.exit(1)
    main() 