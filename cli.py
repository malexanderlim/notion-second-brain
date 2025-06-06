import argparse
import logging
from datetime import datetime, date, timedelta
import sys
import os
import json # Import json for pretty printing
# import numpy as np # Moved to rag_pipeline
# import faiss # Moved to rag_pipeline
# from openai import OpenAI, RateLimitError, APIError # Moved to rag_pipeline
# import time # Moved to rag_pipeline

# Adjust path to import from the package
# This assumes cli.py is in the root and the package is notion_second_brain/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from notion_second_brain.notion.api import NotionClient
# from notion_second_brain.notion.extractors import extract_page_content # Moved to data_exporter
# from notion_second_brain.storage import json_storage # Moved to data_exporter
from notion_second_brain import config # To check if config loaded ok
# from notion_second_brain.processing.transformers import transform_page_to_simple_dict # Moved to data_exporter
from notion_second_brain.data_exporter import handle_export as run_export_action
from notion_second_brain.data_exporter import create_notion_filter_for_specific_month
from notion_second_brain.processing.transformers import transform_page_to_simple_dict # Need this for initial date scan
from notion_second_brain.notion import api as notion_api_utils # For get_database_cached
# from notion_second_brain.rag_pipeline import handle_query as run_query_action # Removed old import

# Imports for RAG logic and initialization
from api.rag_query import execute_rag_query_sync as run_query_action
from api.rag_initializer import (
    load_rag_data,
    initialize_openai_client,
    initialize_anthropic_client,
    initialize_pinecone_client
)
# DEFAULT_FINAL_ANSWER_MODEL_KEY can be imported from rag_config if needed for help text
from api.rag_config import DEFAULT_FINAL_ANSWER_MODEL_KEY

# --- RAG Constants --- # MOVED to rag_pipeline.py
# INDEX_FILE = "index.faiss"
# MAPPING_FILE = "index_mapping.json"
# EMBEDDING_MODEL = "text-embedding-ada-002" # Must match build_index.py
# COMPLETION_MODEL = "gpt-4o" # Changed from gpt-3.5-turbo
# MAX_EMBEDDING_RETRIES = 3
# EMBEDDING_RETRY_DELAY = 5 # seconds
# TOP_K = 15 # Number of results to retrieve - Increased from 5
# QUERY_ANALYSIS_MODEL = "gpt-4o-mini"

# --- Logging Setup ---
# Removed old basicConfig setup
# Centralized logging setup will be called from main()
logger = logging.getLogger("cli") # Still get a logger specific to this module

# --- Helper Function for Embedding (Duplicated from build_index for MVP) --- # MOVED to rag_pipeline.py
# def get_embedding(client: OpenAI, text: str, retries: int = MAX_EMBEDDING_RETRIES) -> list[float] | None:
#     """Gets embedding for text using OpenAI API with retry logic."""
#     attempt = 0
#     while attempt < retries:
#         try:
#             response = client.embeddings.create(
#                 input=text,
#                 model=EMBEDDING_MODEL
#             )
#             return response.data[0].embedding
#         except RateLimitError as e:
#             attempt += 1
#             logger.warning(f"Rate limit hit for embedding, attempt {attempt}/{retries}. Retrying in {EMBEDDING_RETRY_DELAY}s... Error: {e}")
#             time.sleep(EMBEDDING_RETRY_DELAY)
#         except APIError as e:
#             attempt += 1
#             logger.warning(f"API error during embedding, attempt {attempt}/{retries}. Retrying in {EMBEDDING_RETRY_DELAY}s... Error: {e}")
#             time.sleep(EMBEDDING_RETRY_DELAY)
#         except Exception as e:
#             logger.error(f"Unexpected error getting embedding: {e}", exc_info=True)
#             return None # Stop retrying on unexpected errors
#     logger.error(f"Failed to get embedding after {retries} retries.")
#     return None

# --- NEW: Query Analysis Function --- # MOVED to rag_pipeline.py
# def analyze_query_for_filters(client: OpenAI, query: str, schema_properties: dict, distinct_metadata_values: dict | None) -> dict | None:
#     """Analyzes the user query using an LLM to extract structured filters based on the Notion schema and distinct values.
# 
#     Args:
#         client: OpenAI client instance.
#         query: The user's natural language query.
#         schema_properties: Dictionary of property names and their details from schema.json.
#         distinct_metadata_values: Dictionary containing lists of known distinct values for key fields (e.g., Family, Friends, Tags). Can be None.
# 
#     Returns:
#         A dictionary containing structured filters (e.g., date_range, filters list) or None if failed.
#     """
#     logger.info("Analyzing query to extract potential metadata filters...")
#     
#     # --- Prepare schema and distinct value info for prompt --- 
#     field_descriptions = []
#     for name, details in schema_properties.items():
#         # Skip adding Food details to the prompt for now
#         if name == "Food": 
#             continue 
#             
#         field_type = details.get('type', 'unknown')
#         desc = f"- {name} (type: {field_type})"
#         if distinct_metadata_values and name in distinct_metadata_values:
#             known_values = distinct_metadata_values[name]
#             if known_values:
#                  max_values_to_show = 50 
#                  values_str = ", ".join(known_values[:max_values_to_show])
#                  if len(known_values) > max_values_to_show:
#                      values_str += f", ... ({len(known_values) - max_values_to_show} more)"
#                  desc += f" | Known values: [{values_str}]"
#         field_descriptions.append(desc)
#     schema_prompt_part = "\n".join(field_descriptions)
# 
#     # System Prompt (Remove specific mention of Food if any - checking...) 
#     # Current prompt doesn't explicitly mention Food, focuses on Family/Friends/Tags, so it's okay.
#     system_prompt = (
#         "You are a query analysis assistant. Your task is to analyze the user query and the available Notion database fields "
#         "(including known values for some fields) to extract structured filters. Identify potential entities like names, tags, dates, or date ranges mentioned in the query "
#         "and map them to the most relevant field based on the provided schema AND the known values. Format the output as a JSON object. "
#         "Recognize date ranges (like 'last year', '2024', 'next month', 'June 2023'). For date ranges, output a 'date_range' key "
#         "with 'start' and 'end' sub-keys in 'YYYY-MM-DD' format. For specific field value filters, output a 'filters' key containing "
#         "a list of objects, where each object has 'field' (the Notion property name) and 'contains' (the value extracted from the query). "
#         "**Important:** Names of people are typically found in the 'Family' (relation) or 'Friends' (relation) fields. Use the 'Known values' list provided for these fields to help map names accurately. Map person names to THESE fields unless the query specifically asks about the entry's title (the 'Name' field). "
#         "If a name could belong to either 'Family' or 'Friends' based on known values or context, include filters for BOTH fields. "
#         "Match keywords mentioned in the query to the 'Known values' for the 'Tags' field where appropriate. "
#         "If no specific filters are identified, return an empty JSON object {}."
#     )
#     
#     user_prompt = f"""Available Notion Fields (with known values for some):
# --- SCHEMA & VALUES START ---
# {schema_prompt_part}
# --- SCHEMA & VALUES END ---
# 
# User Query: "{query}"
# 
# Analyze the query based ONLY on the schema, known values, and query provided, following the mapping guidelines carefully. Output the structured filters as a JSON object:
# """
#     
#     logger.debug(f"Query Analysis System Prompt: {system_prompt}")
#     logger.debug(f"Query Analysis User Prompt:\n{user_prompt}")
# 
#     try:
#         response = client.chat.completions.create(
#             model=QUERY_ANALYSIS_MODEL, 
#             messages=[
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": user_prompt}
#             ],
#             temperature=0.1, # Low temperature for factual extraction
#             response_format={ "type": "json_object" } # Request JSON output
#         )
#         analysis_result = response.choices[0].message.content
#         logger.info("Received query analysis result from LLM.")
#         logger.debug(f"Raw analysis result: {analysis_result}")
#         
#         # Parse the JSON result
#         filter_data = json.loads(analysis_result)
#         # Basic validation (can be improved)
#         if not isinstance(filter_data, dict):
#             raise ValueError("Analysis result is not a dictionary.")
#         if 'date_range' in filter_data and (not isinstance(filter_data['date_range'], dict) or 
#                                            'start' not in filter_data['date_range'] or 
#                                            'end' not in filter_data['date_range']):
#              raise ValueError("Invalid 'date_range' format.")
#         if 'filters' in filter_data and not isinstance(filter_data['filters'], list):
#              raise ValueError("Invalid 'filters' format.")
#              
#         logger.info(f"Parsed filter data: {json.dumps(filter_data)}")
#         return filter_data
#         
#     except json.JSONDecodeError as e:
#         logger.error(f"Failed to parse JSON from query analysis LLM response: {e}")
#         logger.error(f"Raw response was: {analysis_result}")
#         return None
#     except Exception as e:
#         logger.error(f"Error during query analysis LLM call: {e}", exc_info=True)
#         return None

# --- Argument Parsing ---
def parse_arguments():
    parser = argparse.ArgumentParser(description="Extract or query journal entries from Notion.")

    # --- Action Group ---
    action_group = parser.add_argument_group(title='Actions (choose one)')
    mut_ex_group = action_group.add_mutually_exclusive_group()
    
    # --- Query Action --- 
    query_group = mut_ex_group.add_argument_group(title='Query Action') # Group query related args
    query_group.add_argument(
        "--query",
        type=str,
        help="Ask a natural language question about your journal entries."
    )
    query_group.add_argument(
        "--model", 
        type=str, 
        # Use the imported default as an example in the help text
        help=f"Specify the AI model for the query (e.g., {DEFAULT_FINAL_ANSWER_MODEL_KEY}, gpt-4o, claude-3-5-haiku-20241022). See backend/rag_query.py MODEL_CONFIG for options."
    )
    
    # --- Export Action --- 
    mut_ex_group.add_argument(
        "--export",
        action="store_true",
        help="Export entries based on filters (default action if no other action specified)."
    )
    # --- Test Connection Action ---
    mut_ex_group.add_argument(
        "--test-connection",
        action="store_true",
        help="Test the connection to the Notion API and exit."
    )
    # --- Schema Action --- 
    mut_ex_group.add_argument(
        "--schema",
        action="store_true",
        help="Retrieve database schema and save to schema.json and exit."
    )

    # --- Filtering Group (Only relevant for --export) ---
    filter_group = parser.add_argument_group(title='Filtering Options (for --export)')
    filter_period_group = filter_group.add_mutually_exclusive_group() 
    filter_period_group.add_argument(
        "--export-month",
        type=str,
        metavar="YYYY-MM",
        help="Export entries for a specific month (e.g., 2024-01). Mutually exclusive with other period/date args."
    )
    # Group for the older period/date flags
    try: # Wrap in try-except as nesting groups is deprecated and might raise errors in future versions
      old_period_flags = filter_period_group.add_argument_group(title='Legacy Period/Date Filters') 
    except Exception:
      old_period_flags = filter_period_group # Fallback if nesting fails
      logger.warning("Could not create nested argument group due to potential deprecation. Legacy filter flags might display differently.")

    old_period_flags.add_argument(
        "-p", "--period",
        choices=['day', 'week', 'month', 'year', 'all', 'range'],
        default='all',
        help="Time period filter (e.g., day, week, month, year, all, range). Default: all. Use --export-month YYYY-MM for preferred monthly export."
    )
    old_period_flags.add_argument(
        "--date",
        help="Target date (YYYY-MM-DD) for --period day/week. Defaults to today."
    )
    old_period_flags.add_argument(
        "--month", # Legacy month arg
        type=int,
        help="Target month (1-12). Requires --year. Used only with --period month."
    )
    old_period_flags.add_argument(
        "--year", # Legacy year arg
        type=int,
        help="Target year (YYYY). Used with --period month/year."
    )
    old_period_flags.add_argument(
        "--start-date",
        help="Start date for --period range (YYYY-MM-DD)."
    )
    old_period_flags.add_argument(
        "--end-date",
        help="End date for --period range (YYYY-MM-DD). Defaults to today."
    )

    # This applies to all filters within the export group
    filter_group.add_argument(
        "--date-property",
        default="last_edited_time", # Changed default
        help="Notion property for date filtering (default: last_edited_time). Recommended: last_edited_time for sync."
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

    # Default action is export if no other mutually exclusive action is specified
    is_action_specified = args.query or args.export or args.test_connection or args.schema
    if not is_action_specified:
        args.export = True
        logger.info("No action specified, defaulting to --export.")
        # If defaulting to export, ensure some filter is usable
        if not args.export_month and args.period == 'all':
             logger.info("Defaulting export to --period all. Use --export-month or other filters for specific periods.")

    # Validation (only for export)
    if args.export:
        # Check for conflicting period arguments (mutex group should handle, but double-check)
        if args.export_month and (args.period != 'all' or args.date or args.month or args.year or args.start_date or args.end_date):
             parser.error("--export-month cannot be used with --period, --date, --month, --year, --start-date, or --end-date.")
        
        # Validation for legacy period flags if --export-month is NOT used
        if not args.export_month:
            if args.period == 'month' and not (args.year and args.month):
                parser.error("--period 'month' requires --year and --month when exporting.")
            if args.period == 'year' and not args.year:
                parser.error("--period 'year' requires --year when exporting.")
            if args.period == 'range' and not args.start_date:
                 parser.error("--period 'range' requires --start-date when exporting.")
                
        # Validate export_month format if provided
        if args.export_month:
            try:
                year_str, month_str = args.export_month.split('-')
                year = int(year_str)
                month_val = int(month_str) # Renamed to avoid conflict with datetime.month
                if not (1 <= month_val <= 12) or len(year_str) != 4 or len(month_str) != 2:
                    raise ValueError()
                datetime(year, month_val, 1) # Check if it's a valid date start
            except (ValueError, TypeError):
                parser.error("--export-month must be in YYYY-MM format (e.g., 2024-01).")

    log_level = logging.DEBUG if args.verbose else logging.INFO
    return args, log_level

# --- Main Logic ---
def main():
    args, log_level = parse_arguments()
    config.setup_logging(level=log_level)
    
    # --- Initialize Core Components (RAG Data, LLM Clients) --- 
    # These are needed for query, and loading them early avoids redundant loads
    # and ensures clients are ready.
    try:
        logger.info("Loading RAG data (index, mapping, cache, schema)...")
        load_rag_data()
        logger.info("RAG data loaded.")
    except Exception as e:
        logger.error(f"Fatal: Failed to load RAG data on startup: {e}", exc_info=True)
        print(f"Error: Could not load essential RAG data: {e}")
        sys.exit(1)
        
    # Initialize LLM clients (using keys from config)
    # These functions handle missing keys gracefully (log warnings)
    logger.info("Initializing LLM clients...")
    initialize_openai_client(config.OPENAI_API_KEY)
    # Safely get ANTHROPIC_API_KEY, pass None if not found
    initialize_anthropic_client(getattr(config, 'ANTHROPIC_API_KEY', None))
    logger.info("Initializing Pinecone client for CLI...")

    # Debug log to check config values
    pinecone_key_from_config = getattr(config, 'PINECONE_API_KEY', None)
    pinecone_index_from_config = getattr(config, 'PINECONE_INDEX_NAME', None)
    logger.debug(f"CLI: Pinecone Key from config: {'SET' if pinecone_key_from_config else 'NOT SET'}")
    logger.debug(f"CLI: Pinecone Index from config: {'SET' if pinecone_index_from_config else 'NOT SET'}")

    initialize_pinecone_client(
        pinecone_key_from_config,
        pinecone_index_from_config
    )
    logger.info("LLM and Pinecone clients initialized (if keys were provided).")

    # --- Notion Client Initialization ---
    notion_client = None
    logger.info("Initializing Notion client for requested action...")
    try:
        notion_client = NotionClient(token=config.NOTION_TOKEN)
        logger.info("Notion client initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize Notion client: {e}", exc_info=True)
        sys.exit(1)

    # Determine DB ID to use
    db_id_to_use = config.DATABASE_ID
    if not db_id_to_use:
        logger.error("NOTION_DATABASE_ID is not set in .env or config.py. Cannot proceed.")
        sys.exit(1)
    logger.info(f"Using Notion Database ID: {db_id_to_use}")

    # --- Action Handling ---
    action_taken = False
    if args.query:
        # Ensure RAG data and clients are loaded before querying
        logger.info("Loading RAG data (index, mapping, cache, schema)...")
        load_rag_data() # Loads mapping, schema, metadata cache
        logger.info("RAG data loaded.")
        
        logger.info("Initializing LLM clients for query...")
        initialize_openai_client() 
        initialize_anthropic_client()
        # Pinecone already initialized by load_rag_data if needed by DATA_SOURCE_MODE
        logger.info("LLM clients initialized (if keys were provided).")

        logger.info(f"Executing query: '{args.query}' with model: '{args.model if args.model else 'default defined in RAG'}'")
        run_query_action(query_text=args.query, llm_model_name_override=args.model)
        action_taken = True

    elif args.test_connection:
        logger.info("Testing Notion API connection...")
        try:
            test_db_info = notion_client.retrieve_database_schema(db_id_to_use)
            db_title = test_db_info.get('title', [{'plain_text': '[Unknown Database]'}])[0].get('plain_text', '[Unknown Database]')
            logger.info(f"Successfully connected to Notion. Database title: '{db_title}'")
            logger.info(f"Full database info (first 200 chars): {str(test_db_info)[:200]}...")
        except Exception as e:
            logger.error(f"Notion API connection test failed: {e}", exc_info=True)
        action_taken = True
        
    elif args.schema:
        logger.info(f"Retrieving schema for database ID: {db_id_to_use}...")
        try:
            db_info = notion_client.retrieve_database_schema(db_id_to_use) # type: ignore
            # exporter.save_schema_locally(db_info, "schema.json") # This was already commented
            
            # The problematic call below is removed.
            # exporter.handle_schema_retrieval_and_save(config=config, db_id=db_id_to_use) # type: ignore

            # Save the full schema to a local file
            schema_filename = "schema.json"
            with open(schema_filename, 'w') as f:
                json.dump(db_info, f, indent=4)
            logger.info(f"Database schema saved to {schema_filename}")
            
            # Also attempt to save to GCS if configured (as done by build_index.py)
            if config.GCS_BUCKET_NAME and config.GCS_INDEX_ARTIFACTS_PREFIX:
                from google.cloud import storage
                try:
                    storage_client = storage.Client()
                    bucket = storage_client.bucket(config.GCS_BUCKET_NAME)
                    blob_name = f"{config.GCS_INDEX_ARTIFACTS_PREFIX.rstrip('/')}/{schema_filename}"
                    blob = bucket.blob(blob_name)
                    blob.upload_from_filename(schema_filename)
                    logger.info(f"Schema also uploaded to GCS: gs://{config.GCS_BUCKET_NAME}/{blob_name}")
                except Exception as gcs_e:
                    logger.warning(f"Failed to upload schema to GCS: {gcs_e}")
            else:
                logger.info("GCS bucket/prefix not configured for schema upload.")

        except Exception as e:
            logger.error(f"Failed to retrieve or save database schema: {e}", exc_info=True)
        action_taken = True

    # Default to export action if no other specific action is chosen by flags
    # or if --export is explicitly set.
    if args.export or not action_taken:
        logger.info("Export action initiated.")
        # If a specific month or period is given via args, let handle_export use that directly.
        if args.export_month or args.period != 'all' or args.date or args.month or args.year or args.start_date:
            logger.info("Exporting for a specific period defined by arguments.")
            run_export_action(args, notion_client, db_id_to_use)
        else:
            # --- New: Export all data, month by month sequentially ---
            logger.info("Exporting all data, month by month sequentially...")
            try:
                # 1. Get DB schema to know the 'Date' property name and for transform_page_to_simple_dict
                logger.info("Fetching database schema properties...")
                db_info_full = notion_client.retrieve_database_schema(database_id=db_id_to_use)
                if not db_info_full:
                    logger.error(f"Failed to retrieve database schema for {db_id_to_use}. Cannot proceed.")
                    sys.exit(1)
                schema_properties = db_info_full.get('properties', {})
                
                # Find the actual name of the 'Date' property if it's different, for now assume 'Date'
                # This could be made more robust by looking for type 'date' in schema_properties
                # For now, we'll assume it's literally named 'Date' as per prior logs.
                date_property_name_in_notion = "Date" 
                if date_property_name_in_notion not in schema_properties:
                    logger.error(f"The assumed date property '{date_property_name_in_notion}' not found in database schema. Cannot proceed with monthly export.")
                    logger.error(f"Available properties: {list(schema_properties.keys())}")
                    sys.exit(1)
                logger.info(f"Using date property '{date_property_name_in_notion}' for monthly filtering.")

                # 2. Fetch all pages (minimal data) to determine the overall date range
                logger.info("Fetching all page entries from Notion to determine date range...")
                # Use the notion_client directly as it handles pagination
                all_raw_pages_for_daterange = notion_client.query_database(database_id=db_id_to_use, filter_params=None)
                
                if not all_raw_pages_for_daterange:
                    logger.info("No pages found in the database. Nothing to export.")
                    sys.exit(0)
                
                logger.info(f"Retrieved {len(all_raw_pages_for_daterange)} raw pages for date range analysis. Processing them...")
                
                # Transform to simple dicts to easily access the 'entry_date'
                # Corrected: transform_page_to_simple_dict only takes one argument: page
                processed_for_daterange = [
                    transform_page_to_simple_dict(page) 
                    for page in all_raw_pages_for_daterange
                ]

                entry_dates = []
                for entry in processed_for_daterange:
                    date_str = entry.get('entry_date')
                    if date_str and isinstance(date_str, str):
                        try:
                            entry_dates.append(date.fromisoformat(date_str))
                        except ValueError:
                            logger.warning(f"Could not parse date string '{date_str}' for entry_id {entry.get('page_id')} during date range scan.")
                    elif isinstance(date_str, date): # If already a date object
                        entry_dates.append(date_str)
                
                if not entry_dates:
                    logger.info("No valid dates found in entries. Cannot determine export range.")
                    sys.exit(0)

                min_date = min(entry_dates)
                max_date = max(entry_dates)
                logger.info(f"Overall data range found: from {min_date.isoformat()} to {max_date.isoformat()}.")

                # 3. Generate list of YYYY-MM strings for the range
                year_months_to_export = []
                current_loop_date = min_date
                while current_loop_date <= max_date:
                    year_months_to_export.append(current_loop_date.strftime("%Y-%m"))
                    # Move to the first day of the next month to continue generating unique YYYY-MM
                    if current_loop_date.month == 12:
                        current_loop_date = current_loop_date.replace(year=current_loop_date.year + 1, month=1, day=1)
                    else:
                        current_loop_date = current_loop_date.replace(month=current_loop_date.month + 1, day=1)
                
                year_months_to_export = sorted(list(set(year_months_to_export))) # Ensure uniqueness and order

                logger.info(f"Will export data for the following months: {year_months_to_export}")

                # 4. Loop through each month and export
                for ym_str in year_months_to_export:
                    logger.info(f"--- Starting export for month: {ym_str} ---")
                    
                    # Create the specific Notion filter for this month targeting the 'Date' property
                    # The date_property_name_in_notion was determined above
                    notion_filter_for_month = create_notion_filter_for_specific_month(
                        year_month_str=ym_str,
                        date_property_name=date_property_name_in_notion 
                    )

                    if not notion_filter_for_month:
                        logger.warning(f"Could not generate Notion filter for {ym_str}. Skipping this month.")
                        continue
                    
                    current_month_date_obj = datetime.strptime(ym_str, "%Y-%m").date()

                    # Call handle_export with the specific filter and period name
                    run_export_action(
                        args=args, # Pass original args, though filter/period are now overridden
                        notion_client_instance=notion_client,
                        db_id=db_id_to_use,
                        notion_filter_override=notion_filter_for_month,
                        period_name_override=ym_str,
                        target_dt_for_filename_override=current_month_date_obj
                    )
                    logger.info(f"--- Successfully completed export for month: {ym_str} ---")
                
                logger.info("Completed sequential monthly export for all data.")

            except Exception as e:
                logger.error(f"Error during sequential monthly export: {e}", exc_info=True)
                sys.exit(1)
        action_taken = True # Mark that export action was handled

    if not action_taken:
        # This path should ideally not be reached if default action logic is correct
        # and all actions are handled above.
        logger.error("No valid action specified or action could not be dispatched.")
        parser.print_help() # Print help if no valid action found
        sys.exit(1)

if __name__ == "__main__":
    main() 