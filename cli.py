import argparse
import logging
from datetime import datetime, date, timedelta
import sys
import os
import json # Import json for pretty printing

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

# --- Logging Setup ---
# Removed old basicConfig setup
# Centralized logging setup will be called from main()
logger = logging.getLogger("cli") # Still get a logger specific to this module

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
    parser = argparse.ArgumentParser(description="Extract journal entries from Notion.")

    # --- Action Group ---
    action_group = parser.add_argument_group(title='Actions')
    action_group.add_argument(
        "--test-connection",
        action="store_true",
        help="Test the connection to the Notion API and exit."
    )
    action_group.add_argument(
        "--schema",
        action="store_true",
        help="Retrieve and display the database schema (properties) and exit."
    )

    # --- Filtering Group ---
    filter_group = parser.add_argument_group(title='Filtering Options')
    filter_group.add_argument(
        "-p", "--period",
        choices=['day', 'week', 'month', 'year', 'all', 'range'],
        default='all',
        help="Time period to filter entries for (default: all). Ignored if --schema is used."
    )
    filter_group.add_argument(
        "--date",
        help="Target date for 'day' or 'week' period (YYYY-MM-DD). Defaults to today. Ignored if --schema is used."
    )
    filter_group.add_argument(
        "--month",
        type=int,
        help="Target month (1-12) for 'month' period. Requires --year. Ignored if --schema is used."
    )
    filter_group.add_argument(
        "--year",
        type=int,
        help="Target year (YYYY) for 'month' or 'year' period. Ignored if --schema is used."
    )
    filter_group.add_argument(
        "--start-date",
        help="Start date for 'range' period (YYYY-MM-DD). Ignored if --schema is used."
    )
    filter_group.add_argument(
        "--end-date",
        help="End date for 'range' period (YYYY-MM-DD). Defaults to today if start date is given. Ignored if --schema is used."
    )
    filter_group.add_argument(
        "--date-property",
        default="created_time",
        help="Notion property to use for date filtering (default: created_time). Can be 'created_time' or the name of a date property in your database. Ignored if --schema is used."
    )
    
    # --- Output/Config Group ---
    output_group = parser.add_argument_group(title='Output & Configuration')
    output_group.add_argument(
        "-o", "--output-dir",
        default="output",
        help="Directory to save the output JSON files (default: output). Ignored if --schema is used."
    )
    output_group.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging (DEBUG level)."
    )
    
    args = parser.parse_args()

    # Basic validation (only if not testing connection or getting schema)
    if not args.test_connection and not args.schema:
        if args.period == 'month' and not (args.year and args.month):
            parser.error("--period 'month' requires --year and --month.")
        if args.period == 'year' and not args.year:
            parser.error("--period 'year' requires --year.")
        if args.period == 'range' and not args.start_date:
             parser.error("--period 'range' requires --start-date.")
             
    # Set logging level based on verbosity
    # The actual logger configuration is moved to main()
    # We just determine the level here.
    log_level = logging.DEBUG if args.verbose else logging.INFO

    return args, log_level # Return log_level as well

# --- Main Logic ---
def main():
    args, log_level = parse_arguments() # Get args and desired log level

    # --- Setup Logging FIRST ---
    # Use the centralized function from config.py
    config.setup_logging(level=log_level) 

    # --- Now proceed with the rest of the script ---
    logger.info("Initializing Notion client...")
    try:
        # Make sure NOTION_TOKEN and DATABASE_ID are loaded
        if not config.NOTION_TOKEN or not config.DATABASE_ID:
             # Logging is now available before this check
             logger.error("Missing NOTION_TOKEN or DATABASE_ID in environment or .env file.")
             sys.exit(1)
        
        # --- DEBUG: Print the loaded Database ID ---
        # Remove this debug line now that the ID issue is resolved
        # logger.info(f"Using Database ID from config: {config.DATABASE_ID}") 
        # --- END DEBUG ---

        client = NotionClient(token=config.NOTION_TOKEN)
        db_id = config.DATABASE_ID # Use the configured DB ID
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to initialize Notion client: {e}")
        sys.exit(1)

    if args.test_connection:
        logger.info("Testing Notion connection...")
        if client.test_connection(database_id=db_id):
            logger.info("Notion connection successful!")
            sys.exit(0)
        else:
            logger.error(f"Notion connection failed. Check token and database ID.")
            sys.exit(1)

    if args.schema:
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
                # If 'properties' key is missing, save the whole schema object for inspection
                logger.warning("Schema retrieved but no 'properties' key found. Saving entire schema object.")
                with open(output_filename, 'w') as f:
                    json.dump(schema_data, f, indent=2)
                logger.info(f"Successfully saved entire schema object to {output_filename}")
                sys.exit(1) # Exit with error code since properties were missing
            else:
                 logger.error("Failed to retrieve database schema.")
                 sys.exit(1)
        except Exception as e:
            logger.error(f"An error occurred while retrieving or saving schema: {e}")
            sys.exit(1)

    # --- Build Filter --- (Only if not testing connection or getting schema)
    notion_filter = None
    target_dt_for_filename = None # Store the date needed for filename generation
    if args.period != 'all':
        logger.info(f"Building Notion filter for period: {args.period}")
        try:
            notion_filter, target_dt_for_filename = build_notion_filter_from_args(args)
            logger.debug(f"Constructed Notion filter: {notion_filter}")
        except Exception as e:
            # Error already logged in helper function
            sys.exit(1)
    else:
         logger.info("No time period filter specified (fetching all entries).")

    # --- Query Database --- 
    logger.info("Querying Notion database...")
    try:
        # Pass the constructed filter (if any) to the API client
        # The filter_params argument in NotionClient.query_database expects the inner filter object
        filtered_pages = client.query_database(filter_params=notion_filter) 
    except Exception as e:
        logger.error(f"Failed to query database: {e}")
        sys.exit(1)

    if not filtered_pages:
        logger.warning(f"No pages found in the database matching the filter criteria for period: {args.period}")
        sys.exit(0)

    logger.info(f"Retrieved {len(filtered_pages)} page entries matching the criteria. Processing...")
    
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
            # 1. Transform the raw page object to the simple dict structure
            simple_entry_data = transform_page_to_simple_dict(page)

            # 2. Fetch the block content
            blocks = client.retrieve_block_children(page_id)
            content = extract_page_content(blocks)
            
            # 3. Add the extracted content to the simple dict
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
        # Add more relevant metadata based on args
    }
    if args.period != 'all':
        metadata["period_details"] = { 
            k: v for k, v in vars(args).items() 
            if k in ['date', 'month', 'year', 'start_date', 'end_date'] and v is not None
        }

    logger.info(f"Saving {len(processed_entries)} entries to {os.path.join(args.output_dir, filename)}...")
    saved_path = json_storage.save_entries_to_json(
        processed_entries, # Save the list of simplified dicts
        filename, 
        args.output_dir,
        metadata
    )

    if saved_path:
        logger.info(f"Successfully saved entries to {saved_path}")
    else:
        logger.error("Failed to save entries.")
        sys.exit(1)

    logger.info("Extraction process completed.")

if __name__ == "__main__":
    main() 