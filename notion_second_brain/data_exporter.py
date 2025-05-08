import logging
from datetime import datetime, date, timedelta
import sys
import os
import json # For json.dumps in logging if needed, and for Notion filter logging

# Assuming these are relative to the package root if this file is notion_second_brain/data_exporter.py
from .notion.api import NotionClient # Relative import
from .notion.extractors import extract_page_content # Relative import
from .storage import json_storage # Relative import
from .processing.transformers import transform_page_to_simple_dict # Relative import
# config is not directly used by these functions, logging setup is handled by the caller (cli.py)

logger = logging.getLogger(__name__)

# --- Helper Function for Notion Filter (Moved from cli.py) ---
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
            year, month_val = args.year, args.month # Renamed month to month_val to avoid conflict
            target_dt = date(year, month_val, 1) # Date object for filename generation
            start_date = date(year, month_val, 1)
            next_month_dt = date(year, month_val + 1, 1) if month_val < 12 else date(year + 1, 1, 1) # Renamed next_month to next_month_dt
            end_date = next_month_dt - timedelta(days=1)
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
        raise # Re-raise to be caught by the caller
    except Exception as e:
        logger.error(f"Error building Notion filter: {e}")
        raise # Re-raise to be caught by the caller

    # Return the calculated filter, the representative date, and the period name for the filename
    return (period_filter, target_dt, period_name_for_file)


# --- Export Function (Moved from cli.py) ---
# Renamed client_in to notion_client_instance to avoid confusion with other 'client' variables if any
def handle_export(args, notion_client_instance: NotionClient, db_id: str):
    logger.info("Starting export process...")

    # --- Determine Filter and Filename ---
    notion_filter = None
    target_dt_for_filename = None
    period_name = 'all' # Default if no specific period filter is applied
    try:
        # Calls the build_notion_filter_from_args now local to this module
        notion_filter, target_dt_for_filename, period_name = build_notion_filter_from_args(args)
        logger.debug(f"Constructed Notion filter: {json.dumps(notion_filter) if notion_filter else 'None'}")
    except Exception as e:
        # Error already logged by build_notion_filter_from_args
        # logger.error(f"Failed to build Notion filter: {e}")
        sys.exit(1) # Exit if filter building fails fundamentally

    # --- Query Database ---
    logger.info(f"Querying Notion database (Filter: {json.dumps(notion_filter) if notion_filter else 'None - fetching all'}) based on property '{args.date_property}'...")
    try:
        filtered_pages = notion_client_instance.query_database(database_id=db_id, filter_params=notion_filter)
    except Exception as e:
        logger.error(f"Failed to query database: {e}", exc_info=True)
        sys.exit(1)

    if not filtered_pages:
        logger.warning(f"No pages found matching the criteria for period: {period_name}, filter: {json.dumps(notion_filter) if notion_filter else 'None'}")
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
            blocks = notion_client_instance.retrieve_block_children(page_id)
            content = extract_page_content(blocks)
            simple_entry_data['content'] = content
            processed_entries.append(simple_entry_data)
        except Exception as e:
            logger.error(f"Failed to process page ID {page_id} ('{page_title}'): {e}", exc_info=True)
            continue # Continue with the next page if one fails

    logger.info(f"Successfully processed {len(processed_entries)} entries.")

    # --- Saving ---
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
    if args.period != 'all' and not args.export_month:
        metadata["period_details"] = {
            k: v for k, v in vars(args).items()
            if k in ['date', 'month', 'year', 'start_date', 'end_date'] and v is not None
        }
    output_file_path = os.path.join(args.output_dir, filename)
    logger.info(f"Saving {len(processed_entries)} entries to {output_file_path}...")
    saved_path = json_storage.save_entries_to_json(
        processed_entries, filename, args.output_dir, metadata
    )
    if saved_path:
        logger.info(f"Successfully saved entries to {saved_path}")
    else:
        logger.error(f"Failed to save entries to {output_file_path}.") # More specific error
        sys.exit(1)

    logger.info("Extraction process completed.") 