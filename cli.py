import argparse
import logging
from datetime import datetime, date, timedelta
import sys
import os

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
log_format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("cli")
# Set requests logger level higher to avoid excessive debug messages
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

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

    parser.add_argument(
        "--test-connection",
        action="store_true",
        help="Test the connection to the Notion API and exit."
    )
    parser.add_argument(
        "-p", "--period",
        choices=['day', 'week', 'month', 'year', 'all', 'range'],
        default='all',
        help="Time period to filter entries for (default: all)."
    )
    parser.add_argument(
        "--date",
        help="Target date for 'day' or 'week' period (YYYY-MM-DD). Defaults to today."
    )
    parser.add_argument(
        "--month",
        type=int,
        help="Target month (1-12) for 'month' period. Requires --year."
    )
    parser.add_argument(
        "--year",
        type=int,
        help="Target year (YYYY) for 'month' or 'year' period."
    )
    parser.add_argument(
        "--start-date",
        help="Start date for 'range' period (YYYY-MM-DD)."
    )
    parser.add_argument(
        "--end-date",
        help="End date for 'range' period (YYYY-MM-DD). Defaults to today if start date is given."
    )
    parser.add_argument(
        "--date-property",
        default="created_time",
        help="Notion property to use for date filtering (default: created_time). Can be 'created_time' or the name of a date property in your database."
    )
    parser.add_argument(
        "-o", "--output-dir",
        default="output",
        help="Directory to save the output JSON files (default: output)."
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging (DEBUG level)."
    )
    
    args = parser.parse_args()

    # Basic validation
    if args.period == 'month' and not (args.year and args.month):
        parser.error("--period 'month' requires --year and --month.")
    if args.period == 'year' and not args.year:
        parser.error("--period 'year' requires --year.")
    if args.period == 'range' and not args.start_date:
         parser.error("--period 'range' requires --start-date.")
         
    # Set logging level based on verbosity
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        for handler in logging.getLogger().handlers:
            handler.setLevel(logging.DEBUG)
        logger.info("Verbose logging enabled.")

    return args

# --- Main Logic ---
def main():
    args = parse_arguments()

    logger.info("Initializing Notion client...")
    try:
        client = NotionClient()
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to initialize Notion client: {e}")
        sys.exit(1)

    if args.test_connection:
        logger.info("Testing Notion connection...")
        if client.test_connection():
            logger.info("Notion connection successful!")
            sys.exit(0)
        else:
            logger.error("Notion connection failed. Check token and database ID.")
            sys.exit(1)

    # --- Build Filter --- 
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