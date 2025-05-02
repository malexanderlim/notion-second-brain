import argparse
import logging
from datetime import datetime, date, timedelta
import sys
import os
import json # Import json for pretty printing
import numpy as np
import faiss
from openai import OpenAI, RateLimitError, APIError
import time

# Adjust path to import from the package
# This assumes cli.py is in the root and the package is notion_second_brain/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from notion_second_brain.notion.api import NotionClient
from notion_second_brain.notion.extractors import extract_page_content
# Remove unused import of local filters module
# from notion_second_brain.processing import filters 
from notion_second_brain.storage import json_storage
from notion_second_brain import config # To check if config loaded ok
from notion_second_brain.processing.transformers import transform_page_to_simple_dict # Import the new transformer

# --- RAG Constants ---
INDEX_FILE = "index.faiss"
MAPPING_FILE = "index_mapping.json"
EMBEDDING_MODEL = "text-embedding-ada-002" # Must match build_index.py
COMPLETION_MODEL = "gpt-3.5-turbo" # Or "gpt-4o", etc.
MAX_EMBEDDING_RETRIES = 3
EMBEDDING_RETRY_DELAY = 5 # seconds
TOP_K = 3 # Number of results to retrieve

# --- Logging Setup ---
# Removed old basicConfig setup
# Centralized logging setup will be called from main()
logger = logging.getLogger("cli") # Still get a logger specific to this module

# --- Helper Function for Embedding (Duplicated from build_index for MVP) ---
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

# --- Helper Function for Notion Filter ---
def build_notion_filter_from_args(args):
    """Builds a Notion API filter dictionary based on parsed arguments."""
    period_filter = None
    target_dt = None # Keep track of the primary date for filename generation

    try:
        prop = args.date_property
        is_standard_prop = prop != "created_time" # Check if we're using a custom property or the metadata timestamp
        filter_key = "date" if is_standard_prop else "created_time"
        timestamp_key = "property" if is_standard_prop else "timestamp"

        if args.period == 'day':
            target_dt = date.fromisoformat(args.date) if args.date else date.today()
            period_filter = {timestamp_key: prop, filter_key: {"equals": target_dt.isoformat()}}
        
        elif args.period == 'week':
            target_dt = date.fromisoformat(args.date) if args.date else date.today()
            start_of_week = target_dt - timedelta(days=target_dt.weekday()) # Monday
            end_of_week = start_of_week + timedelta(days=6) # Sunday
            period_filter = {
                "and": [
                    {timestamp_key: prop, filter_key: {"on_or_after": start_of_week.isoformat()}},
                    {timestamp_key: prop, filter_key: {"on_or_before": end_of_week.isoformat()}}
                ]
            }

        elif args.period == 'month':
            year, month = args.year, args.month
            target_dt = date(year, month, 1) # Date object for filename generation
            start_date = date(year, month, 1)
            next_month = date(year, month + 1, 1) if month < 12 else date(year + 1, 1, 1)
            end_date = next_month - timedelta(days=1)
            period_filter = {
                "and": [
                    {timestamp_key: prop, filter_key: {"on_or_after": start_date.isoformat()}},
                    {timestamp_key: prop, filter_key: {"on_or_before": end_date.isoformat()}}
                ]
            }

        elif args.period == 'year':
            year = args.year
            target_dt = date(year, 1, 1) # Date object for filename generation
            start_date = date(year, 1, 1)
            end_date = date(year, 12, 31)
            period_filter = {
                "and": [
                    {timestamp_key: prop, filter_key: {"on_or_after": start_date.isoformat()}},
                    {timestamp_key: prop, filter_key: {"on_or_before": end_date.isoformat()}}
                ]
            }

        elif args.period == 'range':
            start_dt = date.fromisoformat(args.start_date)
            end_dt = date.fromisoformat(args.end_date) if args.end_date else date.today()
            target_dt = start_dt # Use start date for filename
            period_filter = {
                "and": [
                    {timestamp_key: prop, filter_key: {"on_or_after": start_dt.isoformat()}},
                    {timestamp_key: prop, filter_key: {"on_or_before": end_dt.isoformat()}}
                ]
            }
        
        # 'all' period results in period_filter being None
        
    except ValueError as e:
        logger.error(f"Invalid date format provided: {e}")
        raise # Re-raise to be caught in main
    except Exception as e:
        logger.error(f"Error building Notion filter: {e}")
        raise # Re-raise to be caught in main

    return (period_filter, target_dt)

# --- Argument Parsing ---
def parse_arguments():
    parser = argparse.ArgumentParser(description="Extract or query journal entries from Notion.")

    # --- Action Group ---
    action_group = parser.add_argument_group(title='Actions (choose one)')
    # Make actions mutually exclusive for clarity
    mut_ex_group = action_group.add_mutually_exclusive_group()
    mut_ex_group.add_argument(
        "--query",
        type=str,
        help="Ask a natural language question about your journal entries."
    )
    mut_ex_group.add_argument(
        "--export",
        action="store_true",
        help="Export entries based on filters (default action if no other action specified)."
    )
    mut_ex_group.add_argument(
        "--test-connection",
        action="store_true",
        help="Test the connection to the Notion API and exit."
    )
    mut_ex_group.add_argument(
        "--schema",
        action="store_true",
        help="Retrieve database schema and save to schema.json and exit."
    )

    # --- Filtering Group (Only relevant for --export) ---
    filter_group = parser.add_argument_group(title='Filtering Options (for --export)')
    filter_group.add_argument(
        "-p", "--period",
        choices=['day', 'week', 'month', 'year', 'all', 'range'],
        default='all',
        help="Time period to filter entries for export (default: all). Ignored for --query."
    )
    filter_group.add_argument(
        "--date",
        help="Target date (YYYY-MM-DD). Defaults to today. Ignored for --query."
    )
    filter_group.add_argument(
        "--month",
        type=int,
        help="Target month (1-12). Requires --year. Ignored for --query."
    )
    filter_group.add_argument(
        "--year",
        type=int,
        help="Target year (YYYY). Ignored for --query."
    )
    filter_group.add_argument(
        "--start-date",
        help="Start date for range (YYYY-MM-DD). Ignored for --query."
    )
    filter_group.add_argument(
        "--end-date",
        help="End date for range (YYYY-MM-DD). Defaults to today. Ignored for --query."
    )
    filter_group.add_argument(
        "--date-property",
        default="created_time",
        help="Notion property for date filtering (default: created_time). Ignored for --query."
    )
    
    # --- Output/Config Group ---
    output_group = parser.add_argument_group(title='Output & Configuration')
    output_group.add_argument(
        "-o", "--output-dir",
        default="output",
        help="Directory for saving export JSON files (default: output). Ignored for --query."
    )
    output_group.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging (DEBUG level)."
    )
    
    args = parser.parse_args()

    # Default action is export if no action flag is explicitly given
    is_action_specified = args.query or args.test_connection or args.schema or args.export
    if not is_action_specified:
        args.export = True

    # Basic validation (only for export)
    if args.export:
        if args.period == 'month' and not (args.year and args.month):
            parser.error("--period 'month' requires --year and --month when exporting.")
        if args.period == 'year' and not args.year:
            parser.error("--period 'year' requires --year when exporting.")
        if args.period == 'range' and not args.start_date:
             parser.error("--period 'range' requires --start-date when exporting.")
             
    log_level = logging.DEBUG if args.verbose else logging.INFO
    return args, log_level

# --- RAG Query Function ---
def handle_query(args):
    logger.info(f"Handling query: '{args.query}'")

    # --- Check for index files ---
    if not os.path.exists(INDEX_FILE) or not os.path.exists(MAPPING_FILE):
        logger.error(f"Index files not found ({INDEX_FILE}, {MAPPING_FILE}).")
        logger.error(f"Please run 'python build_index.py' first to create the index.")
        sys.exit(1)

    # --- Load Index and Mapping ---
    logger.info(f"Loading FAISS index from {INDEX_FILE}...")
    try:
        index = faiss.read_index(INDEX_FILE)
        logger.info(f"Index loaded. Contains {index.ntotal} vectors.")
    except Exception as e:
        logger.error(f"Failed to load FAISS index: {e}", exc_info=True)
        sys.exit(1)

    logger.info(f"Loading mapping data from {MAPPING_FILE}...")
    try:
        with open(MAPPING_FILE, 'r', encoding='utf-8') as f:
            mapping_data = json.load(f)
        logger.info(f"Mapping data loaded for {len(mapping_data)} entries.")
    except FileNotFoundError:
        logger.error(f"Mapping file not found: {MAPPING_FILE}") # Should be caught earlier, but belt-and-suspenders
        sys.exit(1)
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {MAPPING_FILE}: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error loading mapping file: {e}", exc_info=True)
        sys.exit(1)

    # --- Initialize OpenAI Client ---
    logger.info("Initializing OpenAI client for query...")
    try:
        if not config.OPENAI_API_KEY:
             logger.error("OPENAI_API_KEY not found.")
             sys.exit(1)
        openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
    except ValueError as e:
        logger.error(f"OpenAI Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}", exc_info=True)
        sys.exit(1)

    # --- Embed Query ---
    logger.info("Generating embedding for the query...")
    query_embedding = get_embedding(openai_client, args.query)
    if query_embedding is None:
        logger.error("Failed to generate embedding for the query. Aborting.")
        sys.exit(1)
    query_vector = np.array([query_embedding]).astype('float32')

    # --- Search Index ---
    logger.info(f"Searching index for top {TOP_K} relevant entries...")
    try:
        distances, indices = index.search(query_vector, TOP_K)
        logger.debug(f"Search results (indices): {indices[0]}")
        logger.debug(f"Search results (distances): {distances[0]}")
    except Exception as e:
        logger.error(f"Error searching FAISS index: {e}", exc_info=True)
        sys.exit(1)

    # --- Retrieve and Format Context ---
    logger.info("Retrieving context from mapping data...")
    context = ""
    retrieved_count = 0
    for i, idx in enumerate(indices[0]):
        if idx < 0 or idx >= len(mapping_data):
             logger.warning(f"Search returned invalid index {idx}, skipping.")
             continue
        entry_data = mapping_data[idx]
        context += f"---\nEntry {i+1} (Page ID: {entry_data.get('page_id')}, Date: {entry_data.get('entry_date', 'N/A')}, Title: {entry_data.get('title')})\nContent Preview: {entry_data.get('content_preview')}\n---\n"
        retrieved_count += 1
    
    if retrieved_count == 0:
        logger.warning("Could not retrieve any valid entries for the found indices.")
        print("Sorry, I couldn't find relevant information to answer that.") # User-friendly message
        sys.exit(0)

    # --- Construct Prompt ---
    prompt = f"""You are a helpful assistant answering questions based on provided journal entries.

Use the following journal entries to answer the question below:
--- START CONTEXT ---
{context}
--- END CONTEXT ---

Question: {args.query}
Answer:"""
    logger.debug(f"Constructed Prompt:\n{prompt}")

    # --- Call OpenAI Chat Completion ---
    logger.info("Sending request to OpenAI for answer generation...")
    try:
        response = openai_client.chat.completions.create(
            model=COMPLETION_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant answering questions based on provided journal entries."},
                {"role": "user", "content": prompt} # Send the whole constructed prompt as user message
            ],
            temperature=0.7, # Adjust creativity as needed
        )
        answer = response.choices[0].message.content
        logger.info("Received answer from OpenAI.")
    except RateLimitError as e:
         logger.error(f"OpenAI API rate limit exceeded during chat completion: {e}")
         print("\nError: OpenAI API rate limit exceeded. Please try again later.")
         sys.exit(1)
    except APIError as e:
         logger.error(f"OpenAI API error during chat completion: {e}")
         print("\nError: OpenAI API returned an error. Please check your connection or API key.")
         sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error calling OpenAI chat completion: {e}", exc_info=True)
        print("\nError: An unexpected error occurred while generating the answer.")
        sys.exit(1)

    # --- Print Result ---
    print("\nAnswer:")
    print(answer)

# --- Export Function (Existing Logic Refactored) ---
def handle_export(args, client, db_id):
    logger.info("Starting export process...")
    # --- Build Filter --- 
    notion_filter = None
    target_dt_for_filename = None
    if args.period != 'all':
        logger.info(f"Building Notion filter for period: {args.period}")
        try:
            notion_filter, target_dt_for_filename = build_notion_filter_from_args(args)
            logger.debug(f"Constructed Notion filter: {notion_filter}")
        except Exception as e:
            sys.exit(1)
    else:
        logger.info("No time period filter specified (fetching all entries).")

    # --- Query Database ---
    logger.info("Querying Notion database...")
    try:
        filtered_pages = client.query_database(database_id=db_id, filter_params=notion_filter)
    except Exception as e:
        logger.error(f"Failed to query database: {e}")
        sys.exit(1)

    if not filtered_pages:
        logger.warning(f"No pages found matching the criteria for period: {args.period}")
        sys.exit(0)

    logger.info(f"Retrieved {len(filtered_pages)} page entries. Processing...")
    
    # --- Process Filtered Pages --- 
    processed_entries = []
    for i, page in enumerate(filtered_pages):
        page_id = page.get('id')
        page_title_prop = page.get('properties', {}).get('Name', {}).get('title', [{}])[0].get('plain_text', '[No Title]')
        logger.debug(f"Processing page {i+1}/{len(filtered_pages)}: ID {page_id} ({page_title_prop})")
        if not page_id:
            logger.warning("Skipping entry with no ID.")
            continue
        
        try:
            simple_entry_data = transform_page_to_simple_dict(page)
            blocks = client.retrieve_block_children(page_id)
            content = extract_page_content(blocks)
            simple_entry_data['content'] = content 
            processed_entries.append(simple_entry_data)
        except Exception as e:
            logger.error(f"Failed to process page ID {page_id}: {e}")
            continue 

    logger.info(f"Successfully processed {len(processed_entries)} entries.")

    # --- Saving ---
    filename = json_storage.generate_filename(args.period, dt=target_dt_for_filename, year=args.year)
    metadata = {
        "export_time": datetime.now().isoformat(),
        "period_filter": args.period,
        "date_property_used": args.date_property,
    }
    if args.period != 'all':
        metadata["period_details"] = { 
            k: v for k, v in vars(args).items() 
            if k in ['date', 'month', 'year', 'start_date', 'end_date'] and v is not None
        }
    logger.info(f"Saving {len(processed_entries)} entries to {os.path.join(args.output_dir, filename)}...")
    saved_path = json_storage.save_entries_to_json(
        processed_entries, filename, args.output_dir, metadata
    )
    if saved_path:
        logger.info(f"Successfully saved entries to {saved_path}")
    else:
        logger.error("Failed to save entries.")
        sys.exit(1)

    logger.info("Extraction process completed.")

# --- Main Logic ---
def main():
    args, log_level = parse_arguments()
    config.setup_logging(level=log_level)

    # --- Action Dispatch --- 
    if args.query:
        # Check for OpenAI key before proceeding with query
        if not config.OPENAI_API_KEY:
            logger.error("OPENAI_API_KEY not found in environment or .env file. Cannot run query.")
            sys.exit(1)
        handle_query(args)
        sys.exit(0)
    
    # For non-query actions, initialize Notion client
    logger.info("Initializing Notion client...")
    try:
        if not config.NOTION_TOKEN or not config.DATABASE_ID:
             logger.error("Missing NOTION_TOKEN or DATABASE_ID in environment or .env file.")
             sys.exit(1)
        client = NotionClient(token=config.NOTION_TOKEN)
        db_id = config.DATABASE_ID
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to initialize Notion client: {e}")
        sys.exit(1)

    # --- Handle other actions ---
    if args.test_connection:
        logger.info("Testing Notion connection...")
        if client.test_connection(database_id=db_id):
            logger.info("Notion connection successful!")
        else:
            logger.error(f"Notion connection failed. Check token and database ID.")
            sys.exit(1) # Exit with error on failure
        sys.exit(0) # Exit successfully after test

    elif args.schema:
        output_filename = "schema.json"
        logger.info(f"Retrieving schema for database ID: {db_id} and saving to {output_filename}")
        try:
            schema_data = client.retrieve_database_schema(database_id=db_id)
            if schema_data and 'properties' in schema_data:
                properties_data = schema_data['properties']
                with open(output_filename, 'w') as f:
                    json.dump(properties_data, f, indent=2)
                logger.info(f"Successfully saved database properties to {output_filename}")
                sys.exit(0)
            elif schema_data:
                logger.warning("Schema retrieved but no 'properties' key found. Saving entire schema object.")
                with open(output_filename, 'w') as f:
                    json.dump(schema_data, f, indent=2)
                logger.info(f"Successfully saved entire schema object to {output_filename}")
                sys.exit(1) 
            else:
                 logger.error("Failed to retrieve database schema.")
                 sys.exit(1)
        except Exception as e:
            logger.error(f"An error occurred while retrieving or saving schema: {e}")
            sys.exit(1)
    
    elif args.export:
        # Refactored export logic into its own function
        handle_export(args, client, db_id)
        sys.exit(0)
    
    else:
        # Should not be reached if default action logic is correct
        logger.error("No valid action specified.")
        sys.exit(1)

if __name__ == "__main__":
    main() 