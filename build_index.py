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

# Adjust path to import from the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from notion_second_brain import config # Use centralized logging and config

# Setup logger for this module
logger = logging.getLogger("build_index")

# Constants (Consider moving to config or making CLI args if more flexibility needed)
DEFAULT_INPUT_JSON = "output/all_time.json" # Assuming the full export is run first
DEFAULT_INDEX_FILE = "index.faiss"
DEFAULT_MAPPING_FILE = "index_mapping.json"
EMBEDDING_MODEL = "text-embedding-ada-002"
# Simple retry logic for embedding API calls
MAX_EMBEDDING_RETRIES = 3
EMBEDDING_RETRY_DELAY = 5 # seconds

def get_embedding(client: OpenAI, text: str, retries: int = MAX_EMBEDDING_RETRIES) -> list[float] | None:
    """Gets embedding for text using OpenAI API with retry logic."""
    attempt = 0
    while attempt < retries:
        try:
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
            attempt += 1
            logger.warning(f"API error during embedding, attempt {attempt}/{retries}. Retrying in {EMBEDDING_RETRY_DELAY}s... Error: {e}")
            time.sleep(EMBEDDING_RETRY_DELAY)
        except Exception as e:
            logger.error(f"Unexpected error getting embedding: {e}", exc_info=True)
            return None # Stop retrying on unexpected errors
    logger.error(f"Failed to get embedding after {retries} retries.")
    return None

def main():
    parser = argparse.ArgumentParser(description="Build a FAISS index from exported Notion data.")
    parser.add_argument(
        "-i", "--input",
        default=DEFAULT_INPUT_JSON,
        help=f"Path to the input JSON file containing processed entries (default: {DEFAULT_INPUT_JSON})"
    )
    parser.add_argument(
        "--index-file",
        default=DEFAULT_INDEX_FILE,
        help=f"Path to save the FAISS index file (default: {DEFAULT_INDEX_FILE})"
    )
    parser.add_argument(
        "--mapping-file",
        default=DEFAULT_MAPPING_FILE,
        help=f"Path to save the index-to-entry mapping JSON file (default: {DEFAULT_MAPPING_FILE})"
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

    logger.info(f"Starting index build process...")
    logger.info(f"Input JSON: {args.input}")
    logger.info(f"Output Index: {args.index_file}")
    logger.info(f"Output Mapping: {args.mapping_file}")

    # --- Load Input Data ---
    logger.info(f"Loading entries from {args.input}...")
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            data = json.load(f)
            entries = data.get('entries', [])
        if not entries:
            logger.error(f"No 'entries' found in {args.input}. Ensure the export was successful.")
            sys.exit(1)
        logger.info(f"Loaded {len(entries)} entries.")
    except FileNotFoundError:
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {args.input}: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error loading input file: {e}", exc_info=True)
        sys.exit(1)

    # --- Initialize OpenAI Client ---
    logger.info("Initializing OpenAI client...")
    try:
        # Ensure API key is available (will raise ValueError from config if not)
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

    # --- Generate Embeddings and Mapping ---
    logger.info("Generating embeddings and creating mapping...")
    embeddings = []
    mapping_data = []
    skipped_entries = 0
    dimension = None # Will be set by the first successful embedding

    for i, entry in enumerate(entries):
        content_to_embed = entry.get('content', '')
        if not content_to_embed:
            logger.warning(f"Entry {i} (ID: {entry.get('page_id', 'N/A')}) has no content. Skipping embedding.")
            skipped_entries += 1
            continue

        logger.debug(f"Generating embedding for entry {i} (ID: {entry.get('page_id', 'N/A')})...")
        embedding = get_embedding(openai_client, content_to_embed)

        if embedding:
            if dimension is None:
                dimension = len(embedding)
                logger.info(f"Detected embedding dimension: {dimension}")
            elif len(embedding) != dimension:
                 logger.error(f"Inconsistent embedding dimension! Expected {dimension}, got {len(embedding)} for entry {i}. Aborting.")
                 sys.exit(1)
            
            embeddings.append(embedding)
            # Store essential info for retrieval context
            mapping_data.append({
                "page_id": entry.get('page_id'),
                "title": entry.get('title', '[No Title]'),
                "entry_date": entry.get('entry_date'), # Or created_time if preferred
                "content_preview": content_to_embed[:200] + ("..." if len(content_to_embed) > 200 else "") # Store preview for context
            })
        else:
            logger.error(f"Failed to generate embedding for entry {i} (ID: {entry.get('page_id', 'N/A')}). Skipping.")
            skipped_entries += 1

    if not embeddings:
        logger.error("No embeddings were generated. Cannot build index.")
        sys.exit(1)

    logger.info(f"Generated {len(embeddings)} embeddings. Skipped {skipped_entries} entries.")

    # --- Build FAISS Index ---
    logger.info(f"Building FAISS index (Dimension: {dimension})...")
    try:
        index = faiss.IndexFlatL2(dimension) # Using simple L2 distance
        embeddings_np = np.array(embeddings).astype('float32') # FAISS requires float32 numpy array
        index.add(embeddings_np)
        logger.info(f"FAISS index built successfully. Total vectors: {index.ntotal}")
    except Exception as e:
        logger.error(f"Failed to build FAISS index: {e}", exc_info=True)
        sys.exit(1)

    # --- Save Index and Mapping ---
    logger.info(f"Saving FAISS index to {args.index_file}...")
    try:
        faiss.write_index(index, args.index_file)
        logger.info(f"Index saved successfully.")
    except Exception as e:
        logger.error(f"Failed to save FAISS index: {e}", exc_info=True)
        sys.exit(1)

    logger.info(f"Saving mapping data to {args.mapping_file}...")
    try:
        with open(args.mapping_file, 'w', encoding='utf-8') as f:
            json.dump(mapping_data, f, indent=2)
        logger.info(f"Mapping data saved successfully.")
    except Exception as e:
        logger.error(f"Failed to save mapping data: {e}", exc_info=True)
        sys.exit(1)

    logger.info("Index build process completed.")

if __name__ == "__main__":
    # Check for OPENAI_API_KEY early in config import before logging setup
    # This happens implicitly when config is imported, it raises ValueError if missing.
    try:
        _ = config.OPENAI_API_KEY
    except ValueError as e:
         print(f"Error: {e}", file=sys.stderr)
         sys.exit(1)
    main() 