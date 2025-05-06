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
COMPLETION_MODEL = "gpt-4o" # Changed from gpt-3.5-turbo
MAX_EMBEDDING_RETRIES = 3
EMBEDDING_RETRY_DELAY = 5 # seconds
TOP_K = 15 # Number of results to retrieve - Increased from 5
QUERY_ANALYSIS_MODEL = "gpt-4o-mini"

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

# --- NEW: Query Analysis Function ---
def analyze_query_for_filters(client: OpenAI, query: str, schema_properties: dict, distinct_metadata_values: dict | None) -> dict | None:
    """Analyzes the user query using an LLM to extract structured filters based on the Notion schema and distinct values.

    Args:
        client: OpenAI client instance.
        query: The user's natural language query.
        schema_properties: Dictionary of property names and their details from schema.json.
        distinct_metadata_values: Dictionary containing lists of known distinct values for key fields (e.g., Family, Friends, Tags). Can be None.

    Returns:
        A dictionary containing structured filters (e.g., date_range, filters list) or None if failed.
    """
    logger.info("Analyzing query to extract potential metadata filters...")
    
    # --- Prepare schema and distinct value info for prompt --- 
    field_descriptions = []
    for name, details in schema_properties.items():
        # Skip adding Food details to the prompt for now
        if name == "Food": 
            continue 
            
        field_type = details.get('type', 'unknown')
        desc = f"- {name} (type: {field_type})"
        if distinct_metadata_values and name in distinct_metadata_values:
            known_values = distinct_metadata_values[name]
            if known_values:
                 max_values_to_show = 50 
                 values_str = ", ".join(known_values[:max_values_to_show])
                 if len(known_values) > max_values_to_show:
                     values_str += f", ... ({len(known_values) - max_values_to_show} more)"
                 desc += f" | Known values: [{values_str}]"
        field_descriptions.append(desc)
    schema_prompt_part = "\n".join(field_descriptions)

    # System Prompt (Remove specific mention of Food if any - checking...) 
    # Current prompt doesn't explicitly mention Food, focuses on Family/Friends/Tags, so it's okay.
    system_prompt = (
        "You are a query analysis assistant. Your task is to analyze the user query and the available Notion database fields "
        "(including known values for some fields) to extract structured filters. Identify potential entities like names, tags, dates, or date ranges mentioned in the query "
        "and map them to the most relevant field based on the provided schema AND the known values. Format the output as a JSON object. "
        "Recognize date ranges (like 'last year', '2024', 'next month', 'June 2023'). For date ranges, output a 'date_range' key "
        "with 'start' and 'end' sub-keys in 'YYYY-MM-DD' format. For specific field value filters, output a 'filters' key containing "
        "a list of objects, where each object has 'field' (the Notion property name) and 'contains' (the value extracted from the query). "
        "**Important:** Names of people are typically found in the 'Family' (relation) or 'Friends' (relation) fields. Use the 'Known values' list provided for these fields to help map names accurately. Map person names to THESE fields unless the query specifically asks about the entry's title (the 'Name' field). "
        "If a name could belong to either 'Family' or 'Friends' based on known values or context, include filters for BOTH fields. "
        "Match keywords mentioned in the query to the 'Known values' for the 'Tags' field where appropriate. "
        "If no specific filters are identified, return an empty JSON object {}."
    )
    
    user_prompt = f"""Available Notion Fields (with known values for some):
--- SCHEMA & VALUES START ---
{schema_prompt_part}
--- SCHEMA & VALUES END ---

User Query: "{query}"

Analyze the query based ONLY on the schema, known values, and query provided, following the mapping guidelines carefully. Output the structured filters as a JSON object:
"""
    
    logger.debug(f"Query Analysis System Prompt: {system_prompt}")
    logger.debug(f"Query Analysis User Prompt:\n{user_prompt}")

    try:
        response = client.chat.completions.create(
            model=QUERY_ANALYSIS_MODEL, 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1, # Low temperature for factual extraction
            response_format={ "type": "json_object" } # Request JSON output
        )
        analysis_result = response.choices[0].message.content
        logger.info("Received query analysis result from LLM.")
        logger.debug(f"Raw analysis result: {analysis_result}")
        
        # Parse the JSON result
        filter_data = json.loads(analysis_result)
        # Basic validation (can be improved)
        if not isinstance(filter_data, dict):
            raise ValueError("Analysis result is not a dictionary.")
        if 'date_range' in filter_data and (not isinstance(filter_data['date_range'], dict) or 
                                           'start' not in filter_data['date_range'] or 
                                           'end' not in filter_data['date_range']):
             raise ValueError("Invalid 'date_range' format.")
        if 'filters' in filter_data and not isinstance(filter_data['filters'], list):
             raise ValueError("Invalid 'filters' format.")
             
        logger.info(f"Parsed filter data: {json.dumps(filter_data)}")
        return filter_data
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from query analysis LLM response: {e}")
        logger.error(f"Raw response was: {analysis_result}")
        return None
    except Exception as e:
        logger.error(f"Error during query analysis LLM call: {e}", exc_info=True)
        return None

# --- Helper Function for Notion Filter ---
def build_notion_filter_from_args(args):
    """Builds a Notion API filter dictionary based on parsed arguments."""
    period_filter = None
    target_dt = None # Keep track of the primary date for filename generation
    period_name_for_file = args.period # Default period name for filename

    try:
        prop = args.date_property
        # Determine keys based on whether it's a standard property or a metadata timestamp
        is_standard_prop = prop not in ("created_time", "last_edited_time") 
        filter_key = "date" if is_standard_prop else prop # Use prop name directly for timestamp keys
        timestamp_key = "property" if is_standard_prop else "timestamp"

        # --- NEW: Handle --export-month first ---
        if args.export_month:
            period_name_for_file = "month" # Set for filename generation
            year_str, month_str = args.export_month.split('-')
            year = int(year_str)
            month = int(month_str)
            if not (1 <= month <= 12):
                raise ValueError("Month must be between 01 and 12.")
            
            target_dt = date(year, month, 1) # Use first of month for filename consistency
            start_date = date(year, month, 1)
            # Calculate next month correctly, handling year change
            if month == 12:
                next_month_start = date(year + 1, 1, 1)
            else:
                next_month_start = date(year, month + 1, 1)
            end_date = next_month_start - timedelta(days=1)

            period_filter = {
                "and": [
                    {timestamp_key: prop, filter_key: {"on_or_after": start_date.isoformat()}},
                    {timestamp_key: prop, filter_key: {"on_or_before": end_date.isoformat()}}
                ]
            }
            logger.info(f"Using export month {args.export_month}, filtering on '{prop}' between {start_date} and {end_date}")

        # --- Existing period logic (only if --export-month is not used) ---
        elif args.period == 'day':
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

        # Note: This specific 'month' period is now less likely used if --export-month is preferred
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
        logger.error(f"Invalid date format provided or error parsing month string: {e}")
        raise # Re-raise to be caught in main
    except Exception as e:
        logger.error(f"Error building Notion filter: {e}")
        raise # Re-raise to be caught in main

    # Return the calculated filter, the representative date, and the period name for the filename
    return (period_filter, target_dt, period_name_for_file)

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
    
    # Define period/date args first before making them mutually exclusive with --export-month
    filter_period_group = filter_group.add_mutually_exclusive_group() 
    filter_period_group.add_argument(
        "--export-month",
        type=str,
        metavar="YYYY-MM",
        help="Export entries for a specific month (e.g., 2024-01). Mutually exclusive with other period/date args."
    )
    # Group for the older period/date flags
    # Note: The `dest` is not strictly needed here but can help avoid confusion if we print args
    old_period_flags = filter_period_group.add_argument_group(title='Legacy Period/Date Filters') 
    old_period_flags.add_argument(
        "-p", "--period",
        choices=['day', 'week', 'month', 'year', 'all', 'range'],
        # Default changed - if --export-month isn't used, maybe 'all' is better?
        # Or maybe no default, forcing either --export-month or --period?
        # Let's stick with 'all' for now for backward compatibility.
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

    # This applies to all filters
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

    # Default action is export if no action flag is explicitly given
    is_action_specified = args.query or args.test_connection or args.schema or args.export
    if not is_action_specified:
        args.export = True
        logger.info("No action specified, defaulting to --export.")
        # If defaulting to export, ensure some filter is usable
        if not args.export_month and args.period == 'all':
             logger.info("Defaulting export to --period all. Use --export-month or other filters for specific periods.")

    # Basic validation (only for export)
    if args.export:
        # Check if conflicting period arguments were somehow passed (should be caught by mutex group, but belt-and-suspenders)
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
                month = int(month_str)
                if not (1 <= month <= 12) or len(year_str) != 4 or len(month_str) != 2:
                    raise ValueError()
                datetime(year, month, 1) # Check if it's a valid date start
            except (ValueError, TypeError):
                parser.error("--export-month must be in YYYY-MM format (e.g., 2024-01).")

    log_level = logging.DEBUG if args.verbose else logging.INFO
    return args, log_level

# --- RAG Query Function ---
def handle_query(args):
    logger.info(f"Handling query: '{args.query}'")

    # --- Initialize OpenAI Client (needed for safety check first) --- 
    logger.info("Initializing OpenAI client for query processing...") # Updated log message
    try:
        if not config.OPENAI_API_KEY:
             logger.error("OPENAI_API_KEY not found.")
             sys.exit(1)
        openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}", exc_info=True)
        sys.exit(1)

    # --- Proceed with RAG pipeline only if query is safe ---
    # --- Check for prerequisite files (Index, Mapping, Schema, Cache) --- 
    schema_file = "schema.json"
    metadata_cache_file = "metadata_cache.json"
    prereqs = [INDEX_FILE, MAPPING_FILE, schema_file, metadata_cache_file]
    missing = [f for f in prereqs if not os.path.exists(f)]
    if missing:
        logger.error(f"Prerequisite files not found: {', '.join(missing)}.")
        if schema_file in missing: logger.error(f"Please run 'python cli.py --schema' first.")
        if INDEX_FILE in missing or MAPPING_FILE in missing or metadata_cache_file in missing:
             logger.error(f"Please run 'python build_index.py' first (it generates index, mapping, and metadata cache).")
        sys.exit(1)

    # --- Load Schema --- 
    logger.info(f"Loading schema from {schema_file}...")
    try:
        with open(schema_file, 'r') as f:
            schema_properties = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load or parse schema file {schema_file}: {e}")
        sys.exit(1)

    # --- Load Metadata Cache --- 
    logger.info(f"Loading metadata cache from {metadata_cache_file}...")
    distinct_metadata_values = None
    try:
        with open(metadata_cache_file, 'r') as f:
            metadata_cache = json.load(f)
        distinct_metadata_values = metadata_cache.get('distinct_values')
        cache_time = metadata_cache.get('last_updated', 'N/A')
        if not distinct_metadata_values:
             logger.error("'distinct_values' key not found or empty in metadata cache file.")
             sys.exit(1)
        logger.info(f"Metadata cache loaded (Last Updated: {cache_time})")
        logger.debug(f"Loaded distinct values: {json.dumps(distinct_metadata_values, indent=2)}")
    except Exception as e:
        logger.error(f"Failed to load or parse metadata cache file {metadata_cache_file}: {e}")
        sys.exit(1)

    # --- Load Index and Mapping --- 
    logger.info(f"Loading FAISS index from {INDEX_FILE}...")
    try:
        index = faiss.read_index(INDEX_FILE)
        logger.info(f"Index loaded. Contains {index.ntotal} vectors.")
    except Exception as e:
        logger.error(f"Failed to load FAISS index: {e}", exc_info=True)
        sys.exit(1)

    logger.info(f"Loading mapping data from {MAPPING_FILE}... (This might take a moment for large files)")
    try:
        with open(MAPPING_FILE, 'r', encoding='utf-8') as f:
            mapping_data = json.load(f)
        logger.info(f"Mapping data loaded for {len(mapping_data)} entries.")
    except Exception as e:
        logger.error(f"Failed to load or parse mapping file {MAPPING_FILE}: {e}")
        sys.exit(1)

    # --- Analyze Query for Filters (Now passes distinct_metadata_values) ---
    logger.info("Analyzing query to extract potential metadata filters...")
    extracted_filters = analyze_query_for_filters(openai_client, args.query, schema_properties, distinct_metadata_values)
    if extracted_filters is None:
        logger.warning("Proceeding without metadata filters due to analysis error.")
        extracted_filters = {} 

    # --- Pre-filter Mapping Data (Uses extracted_filters) --- 
    logger.info("Pre-filtering mapping data based on query analysis...")
    filtered_indices = list(range(len(mapping_data))) 

    if extracted_filters: 
        logger.info(f"Applying filters: {json.dumps(extracted_filters)}")
        potential_indices = list(range(len(mapping_data)))

        # --- Apply Date Range Filter (Logic remains the same) --- 
        date_range = extracted_filters.get('date_range')
        if date_range:
            try:
                start_date_str = date_range.get('start')
                end_date_str = date_range.get('end')
                start_date = datetime.fromisoformat(start_date_str).date() if start_date_str else None
                end_date = datetime.fromisoformat(end_date_str).date() if end_date_str else None
                logger.info(f"Applying date filter: {start_date} to {end_date}")
                
                temp_indices_date = []
                for i in potential_indices:
                    entry_date_str = mapping_data[i].get('entry_date')
                    if not entry_date_str: continue # Skip if entry has no date
                    try:
                        entry_date = datetime.fromisoformat(entry_date_str).date()
                        passes_date = True
                        if start_date and entry_date < start_date:
                            passes_date = False
                        if end_date and entry_date > end_date:
                            passes_date = False
                        if passes_date:
                            temp_indices_date.append(i)
                    except ValueError:
                        logger.warning(f"Could not parse entry_date '{entry_date_str}' for index {i}. Skipping date filter for this entry.")
                        continue # Skip if date format is wrong
                potential_indices = temp_indices_date # Update potential indices
                logger.debug(f"Indices after date filter: {len(potential_indices)}")
            except Exception as e:
                logger.error(f"Error applying date range filter {date_range}: {e}", exc_info=True)
                # Decide whether to proceed without date filter or halt
                logger.warning("Proceeding without date filter due to error.")
        
        # --- Apply Specific Field Filters (Modified for Family/Friends OR logic) --- 
        field_filters = extracted_filters.get('filters', [])
        if field_filters:
            logger.info(f"Applying {len(field_filters)} field filters...")
            # Separate potential Family/Friends filters for the same name
            name_filters = {}
            other_filters = []
            for f in field_filters:
                field = f.get('field')
                value = f.get('contains')
                if field in ["Family", "Friends"] and value:
                    if value not in name_filters:
                        name_filters[value] = []
                    name_filters[value].append(field)
                else:
                    other_filters.append(f)

            # Apply the combined Family/Friends OR filters first
            temp_indices_names = list(potential_indices)
            for name, fields in name_filters.items():
                logger.debug(f"Applying combined filter for name '{name}' in fields {fields}")
                indices_passing_name_filter = []
                for i in temp_indices_names:
                    entry_data = mapping_data[i]
                    passes_name = False
                    # Check if name exists in *any* of the specified fields (OR logic)
                    for field in fields:
                        field_value = entry_data.get(field)
                        if isinstance(field_value, list):
                            if any(isinstance(item, str) and name.lower() in item.lower() for item in field_value):
                                passes_name = True
                                break # Found in one field, no need to check others for this entry
                    if passes_name:
                        indices_passing_name_filter.append(i)
                temp_indices_names = indices_passing_name_filter
                logger.debug(f"Indices after combined name filter for '{name}': {len(temp_indices_names)}")
                if not temp_indices_names: break
            
            potential_indices = temp_indices_names # Update after name filters

            # Apply other filters sequentially (AND logic)
            if potential_indices: # Only proceed if name filters didn't yield empty set
                temp_indices_others = list(potential_indices)
                for f_filter in other_filters:
                    # ... (logic for applying individual non-name filters as before) ...
                    field_name = f_filter.get('field')
                    contains_value = f_filter.get('contains')
                    if not field_name or contains_value is None:
                         logger.warning(f"Skipping invalid filter object: {f_filter}")
                         continue
                    # ... (existing string/list contains check logic) ...
                    # ... (update temp_indices_others) ...
                    logger.debug(f"Applying filter: Field '{field_name}' must contain '{contains_value}'")
                    indices_passing_this_filter = []
                    for i in temp_indices_others:
                        entry_data = mapping_data[i]
                        field_value = entry_data.get(field_name)
                        passes_filter = False
                        if field_value is not None:
                            if isinstance(field_value, str):
                                if contains_value.lower() in field_value.lower():
                                    passes_filter = True
                            elif isinstance(field_value, list):
                                if any(isinstance(item, str) and contains_value.lower() in item.lower() for item in field_value):
                                    passes_filter = True
                        if passes_filter:
                            indices_passing_this_filter.append(i)
                    temp_indices_others = indices_passing_this_filter
                    logger.debug(f"Indices after filter '{field_name}' contains '{contains_value}': {len(temp_indices_others)}")
                    if not temp_indices_others: break
                potential_indices = temp_indices_others # Update after other filters

        filtered_indices = potential_indices

    else:
        logger.info("No metadata filters extracted or applied.")
         
    if not filtered_indices:
        logger.warning("Pre-filtering resulted in zero matching entries.")
        print("Sorry, no entries matched the specific filters (like dates or tags) in your query.")
        sys.exit(0)
    else:
         logger.info(f"Found {len(filtered_indices)} potentially relevant entries after pre-filtering.")
    # --- >>> END: Pre-filter Mapping Data <<< ---

    # --- Embed Query --- 
    logger.info("Generating embedding for the query...")
    query_embedding = get_embedding(openai_client, args.query)
    if query_embedding is None:
        logger.error("Failed to generate embedding for the query. Aborting.")
        sys.exit(1)
    query_vector = np.array([query_embedding]).astype('float32')

    # --- >>> MODIFIED: Search Index (Targeted) <<< ---
    logger.info(f"Searching index for top {TOP_K} relevant entries WITHIN the {len(filtered_indices)} pre-filtered set...")
    
    distances = np.array([[]], dtype='float32')
    indices = np.array([[]], dtype='int64')
    
    try:
        # 1. Create the IDSelector for the filtered indices
        # Ensure indices are int64 for FAISS
        filtered_ids_array = np.array(filtered_indices, dtype='int64') 
        if filtered_ids_array.size == 0:
             logger.warning("Filtered ID array is empty, cannot perform search.")
             # Handle case where filtering yields empty results but wasn't caught earlier
        else:
             id_selector = faiss.IDSelectorBatch(filtered_ids_array)
             
             # 2. Create SearchParameters and set the selector
             # Assuming IndexFlatL2 which doesn't have specific params beyond the selector
             search_params = faiss.SearchParameters()
             search_params.sel = id_selector
             
             # 3. Perform the targeted search
             # We need to ensure TOP_K is not larger than the number of items we are searching
             actual_k = min(TOP_K, len(filtered_ids_array))
             if actual_k == 0:
                 logger.warning("Effective K is 0 after filtering, skipping search.")
             else:
                 logger.debug(f"Performing targeted search with k={actual_k}")
                 distances, indices = index.search(query_vector, actual_k, params=search_params)
                 logger.debug(f"Targeted search results (indices relative to full index): {indices[0]}")
                 logger.debug(f"Targeted search results (distances): {distances[0]}")

    except AttributeError as e:
         # This might happen if the faiss version doesn't support SearchParameters or sel
         logger.error(f"FAISS Attribute Error: {e}. Your FAISS version might not support subset search via SearchParameters. Falling back to non-filtered search.")
         # Fallback to original search if subset search fails
         distances, indices = index.search(query_vector, TOP_K)
    except Exception as e:
        logger.error(f"Error during targeted FAISS search: {e}", exc_info=True)
        # Decide whether to fallback or exit
        logger.warning("Falling back to non-filtered search due to error.")
        distances, indices = index.search(query_vector, TOP_K)
    
    # Check if search returned any results
    if indices.size == 0 or indices[0].size == 0:
         logger.warning("Targeted search returned no results.")
         # Optional: Add a user message here?
         # We might still have filtered_indices, but semantic search found nothing in them.
         # For now, proceed to context retrieval which will likely be empty.
         pass 
    # --- >>> END: Search Index (Targeted) <<< ---

    # --- Retrieve and Format Context (using the indices from the targeted search) ---
    logger.info("Retrieving context from mapping data...")
    # The indices returned by search (even targeted) are relative to the full index (0 to N-1)
    # and should directly map to our mapping_data list if built correctly.
    context = ""
    retrieved_count = 0
    retrieved_sources = []
    # Check if indices is not empty before iterating
    if indices.size > 0:
        for i, idx in enumerate(indices[0]): 
            # Basic check: idx should be non-negative. 
            # It should also be less than len(mapping_data), which is implicitly handled 
            # by the fact that IDSelectorBatch was created from valid indices within that range.
            if idx < 0: # Should not happen with IDSelectorBatch result
                 logger.warning(f"Search returned negative index {idx}, skipping.")
                 continue
            # The rest of the context retrieval loop remains the same
            try:
                entry_data = mapping_data[idx]
                page_id = entry_data.get('page_id', 'N/A')
                entry_date = entry_data.get('entry_date', 'N/A')
                entry_title = entry_data.get('title', '[No Title]')
                full_content = entry_data.get('content', '')
                if not full_content:
                     logger.warning(f"Retrieved entry (ID: {page_id}) has empty content in mapping file. Skipping for context.")
                     continue
                source_file = entry_data.get('source_file', 'N/A')
                
                context += f"---\nEntry {i+1} Metadata: (Page ID: {page_id}, Date: {entry_date}, Title: {entry_title}, SourceFile: {source_file})\nFull Content:\n{full_content}\n---"
                retrieved_sources.append(f"ID: {page_id}, Title: {entry_title}, Date: {entry_date}, SourceFile: {source_file}, Distance: {distances[0][i]:.4f}")
                retrieved_count += 1
            except IndexError:
                 # This *shouldn't* happen if indices are correct, but good to have safety.
                 logger.warning(f"IndexError retrieving data for search result index {idx}. Mapping data length: {len(mapping_data)}.")
                 continue
            except Exception as e:
                 logger.error(f"Unexpected error retrieving context for index {idx}: {e}", exc_info=True)
                 continue
             
    # --- LOG THE RETRIEVED CONTEXT --- 
    logger.info(f"Retrieved {retrieved_count} entries for context.")
    if retrieved_count > 0:
        logger.info("--- Retrieved Context START ---")
        logger.info("Retrieved Sources (ID, Title, Date, SourceFile, Distance):")
        for source in retrieved_sources:
            logger.info(f"  - {source}")
        logger.debug(f"Context string sent to LLM:\n{context}") 
        logger.info("--- Retrieved Context END ---")
    # --- END CONTEXT LOGGING ---
    
    if retrieved_count == 0:
        # ... (handling no results) ...
        # Now more likely if targeted search yields nothing OR if filter was empty
        print("Sorry, I couldn't find relevant information matching your query and filters.") 
        sys.exit(0)

    # --- Construct Prompt (Refined) ---
    # System prompt instructing the model on how to behave and use the context.
    system_prompt = ( 
        "You are a helpful assistant answering questions based ONLY on the provided journal entries. "
        "Analyze the metadata (Title, Date, Tags, Family, Friends, etc.) and the Content of each entry carefully. "
        "**Assumption:** Assume that if a person's name appears in the 'Family' or 'Friends' metadata for an entry, the author saw or was with that person on that entry's date, even if the content doesn't explicitly state it. Use this assumption when answering questions about seeing people or frequency of contact. "
        "Do not make assumptions or use external knowledge beyond this specific instruction. "
        "**Safety Guardrail:** Avoid generating responses that contain harmful, inappropriate, or overly sensitive personal information. Specifically, do not output details related to illicit substances or illegal activities, even if mentioned in the context. "
        "If the answer cannot be found in the provided entries even with the assumption, say so."
    )
    
    # User prompt containing the context and the actual question.
    # Updated to be more explicit about comprehensiveness and desired output format.
    user_prompt = f"""Here are the relevant journal entries:
--- START CONTEXT ---
{context}
--- END CONTEXT ---

Based *only* on the context provided above, answer the following question comprehensively.

**Instructions for Formatting:**
- When listing items (like sakes, events, etc.) found in the entries:
  - Include the date (from the 'Date:' metadata associated with that entry).
  - Provide a markdown link to the original Notion journal entry formatted as `Journal Entry: [Title of Entry](Notion Link)`.
  - Construct the Notion Link using the 'Page ID:' metadata like this: `https://www.notion.so/<page_id_without_hyphens>` (e.g., if Page ID is 'a1b2c3d4-e5f6-7890-abcd-ef1234567890', the link is `https://www.notion.so/a1b2c3d4e5f67890abcdef1234567890`). Use the 'Title:' metadata for the link text.

Question: {args.query}
Answer:"""
    
    logger.debug(f"System Prompt: {system_prompt}")
    logger.debug(f"User Prompt:\n{user_prompt}")

    # --- Call OpenAI Chat Completion --- 
    logger.info(f"Sending request to OpenAI ({COMPLETION_MODEL}) for answer generation...") # Log the model being used
    try:
        response = openai_client.chat.completions.create(
            model=COMPLETION_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.5, # Lower temperature for more factual answers
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
    
    # --- Determine Filter and Filename --- 
    notion_filter = None
    target_dt_for_filename = None
    period_name = 'all' # Default if no specific period filter is applied
    try:
        notion_filter, target_dt_for_filename, period_name = build_notion_filter_from_args(args)
        logger.debug(f"Constructed Notion filter: {notion_filter}")
    except Exception as e:
        logger.error(f"Failed to build Notion filter: {e}")
        sys.exit(1)

    # --- Query Database ---
    logger.info(f"Querying Notion database (Filter: {json.dumps(notion_filter) if notion_filter else 'None - fetching all'}) based on property '{args.date_property}'...")
    try:
        filtered_pages = client.query_database(database_id=db_id, filter_params=notion_filter)
    except Exception as e:
        logger.error(f"Failed to query database: {e}")
        sys.exit(1)

    if not filtered_pages:
        logger.warning(f"No pages found matching the criteria for period: {period_name}, filter: {notion_filter}")
        # Optionally create an empty file to mark completion?
        # filename = json_storage.generate_filename(period_name, dt=target_dt_for_filename)
        # json_storage.save_entries_to_json([], filename, args.output_dir, {})
        sys.exit(0) # Exit gracefully if no pages found

    logger.info(f"Retrieved {len(filtered_pages)} page entries. Processing...")
    
    # --- Process Filtered Pages --- 
    processed_entries = []
    for i, page in enumerate(filtered_pages):
        page_id = page.get('id')
        # Safely get title
        title_list = page.get('properties', {}).get('Name', {}).get('title', [])
        page_title = title_list[0].get('plain_text', '[No Title]') if title_list else '[No Title]'
        
        logger.debug(f"Processing page {i+1}/{len(filtered_pages)}: ID {page_id} ({page_title})")
        if not page_id:
            logger.warning("Skipping entry with no ID.")
            continue
        
        try:
            # Transform first (gets basic props like date, id)
            simple_entry_data = transform_page_to_simple_dict(page)
            # Then fetch content and add it
            blocks = client.retrieve_block_children(page_id)
            content = extract_page_content(blocks)
            simple_entry_data['content'] = content 
            processed_entries.append(simple_entry_data)
        except Exception as e:
            logger.error(f"Failed to process page ID {page_id} ('{page_title}'): {e}")
            continue # Continue with the next page if one fails

    logger.info(f"Successfully processed {len(processed_entries)} entries.")

    # --- Saving ---
    # Note: generate_filename now takes period_name which build_notion_filter_from_args returns
    filename = json_storage.generate_filename(period_name, dt=target_dt_for_filename) 
    metadata = {
        "export_time": datetime.now().isoformat(),
        "period_filter_used": period_name,
        "date_property_used": args.date_property,
        "notion_filter_applied": notion_filter, # Log the actual filter used
        "export_month_arg": args.export_month if args.export_month else None,
        "total_entries_processed": len(processed_entries),
        "total_entries_retrieved": len(filtered_pages)
    }
    # Add specific date args if they were used (primarily for non --export-month)
    if args.period != 'all' and not args.export_month:
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