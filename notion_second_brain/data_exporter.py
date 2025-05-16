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

GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
GCS_EXPORT_PREFIX = os.getenv("GCS_EXPORT_PREFIX", "notion_exports") # Default prefix

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


def create_notion_filter_for_specific_month(year_month_str: str, date_property_name: str = "Date") -> dict | None:
    """
    Creates a Notion API filter for a specific month.

    Args:
        year_month_str: The month in "YYYY-MM" format.
        date_property_name: The name of the date property in Notion.

    Returns:
        A dictionary representing the Notion API filter, or None if input is invalid.
    """
    try:
        year, month = map(int, year_month_str.split('-'))
        if not (1 <= month <= 12):
            logger.error(f"Invalid month value in {year_month_str}. Month must be 1-12.")
            return None

        start_date = date(year, month, 1)
        # Calculate the first day of the next month, then subtract one day to get the last day of the current month
        if month == 12:
            next_month_start = date(year + 1, 1, 1)
        else:
            next_month_start = date(year, month + 1, 1)
        end_date = next_month_start - timedelta(days=1)

        filter_dict = {
            "and": [
                {"property": date_property_name, "date": {"on_or_after": start_date.isoformat()}},
                {"property": date_property_name, "date": {"on_or_before": end_date.isoformat()}}
            ]
        }
        logger.info(f"Created filter for property '{date_property_name}' for month {year_month_str}: {start_date.isoformat()} to {end_date.isoformat()}")
        return filter_dict
    except ValueError:
        logger.error(f"Invalid year_month_str format: '{year_month_str}'. Expected YYYY-MM.")
        return None
    except Exception as e:
        logger.error(f"Error creating month filter for {year_month_str}: {e}", exc_info=True)
        return None


# --- Export Function (Moved from cli.py) ---
# Renamed client_in to notion_client_instance to avoid confusion with other 'client' variables if any
def handle_export(args, 
                  notion_client_instance: NotionClient, 
                  db_id: str,
                  # New optional parameters to allow direct filter/period specification
                  notion_filter_override: dict | None = None, 
                  period_name_override: str | None = None,
                  target_dt_for_filename_override: date | None = None):
    logger.info("Starting export process...")

    # --- Determine Filter and Filename ---
    notion_filter = notion_filter_override
    target_dt_for_filename = target_dt_for_filename_override
    period_name = period_name_override

    if notion_filter is None: # If no override, use args to build it
        logger.info("No filter override provided, building filter from args...")
        try:
            # Calls the build_notion_filter_from_args now local to this module
            # This will use args.date_property which might be 'last_edited_time' or 'Date'
            # and args.export_month, args.period etc.
            notion_filter, target_dt_for_filename, period_name = build_notion_filter_from_args(args)
            logger.debug(f"Constructed Notion filter from args: {json.dumps(notion_filter) if notion_filter else 'None'}")
        except Exception as e:
            # Error already logged by build_notion_filter_from_args
            sys.exit(1) # Exit if filter building fails fundamentally
    else:
        logger.info(f"Using provided notion_filter_override for period: {period_name if period_name else 'N/A'}")
        if not period_name: # If filter is overridden, period_name should also be.
             # Default to a generic name if period_name_override is missing but filter is present
            period_name = "custom_filter_export" 
            logger.warning("notion_filter_override provided without period_name_override. Using default.")
        if not target_dt_for_filename: # If filter is overridden, target_dt might also need to be.
            try:
                if period_name and '-' in period_name and len(period_name.split('-')) == 2: # YYYY-MM
                    year_str, month_str = period_name.split('-')
                    target_dt_for_filename = date(int(year_str), int(month_str), 1)
                else: # fallback
                    target_dt_for_filename = datetime.now().date()
            except:
                target_dt_for_filename = datetime.now().date()


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
    if not GCS_BUCKET_NAME:
        logger.error("GCS_BUCKET_NAME environment variable not set. Cannot save to GCS.")
        sys.exit(1)

    # Ensure GCS_EXPORT_PREFIX ends with a slash if it's not empty
    gcs_export_prefix_path = GCS_EXPORT_PREFIX
    if gcs_export_prefix_path and not gcs_export_prefix_path.endswith('/'):
        gcs_export_prefix_path += '/'

    # If a specific filter was applied (not exporting 'all' data), save as a single file
    if notion_filter is not None: # This implies a specific period/month was requested
        base_filename = json_storage.generate_filename(period_name, dt=target_dt_for_filename)
        gcs_blob_name = f"{gcs_export_prefix_path}{base_filename}"
        
        metadata_for_gcs = {
            "export_time": datetime.now().isoformat(),
            "period_filter_used": period_name,
            "date_property_used": args.date_property,
            "notion_filter_applied": notion_filter,
            "export_month_arg": args.export_month if args.export_month else None,
            "total_entries_processed": len(processed_entries),
            "total_entries_retrieved_for_this_file": len(filtered_pages) # Renamed for clarity
        }
        if args.period != 'all' and not args.export_month: # This condition might be redundant if notion_filter is not None
            metadata_for_gcs["period_details"] = {
                k: v for k, v in vars(args).items()
                if k in ['date', 'month', 'year', 'start_date', 'end_date'] and v is not None
            }
        
        logger.info(f"Saving {len(processed_entries)} entries (for period '{period_name}') to GCS bucket '{GCS_BUCKET_NAME}' as blob '{gcs_blob_name}'...")
        
        saved_gcs_uri = json_storage.save_entries_to_gcs(
            entries=processed_entries,
            gcs_bucket_name=GCS_BUCKET_NAME,
            gcs_blob_name=gcs_blob_name,
            metadata=metadata_for_gcs
        )
        
        if saved_gcs_uri:
            logger.info(f"Successfully saved entries for period '{period_name}' to {saved_gcs_uri}")
        else:
            logger.error(f"Failed to save entries for period '{period_name}' to GCS (gs://{GCS_BUCKET_NAME}/{gcs_blob_name}).")
            sys.exit(1)

    else: # Exporting 'all' data (notion_filter is None), so group by month and save multiple files
        logger.info("Exporting all data: Grouping entries by month and saving to separate GCS blobs...")
        entries_by_month = {}
        for entry in processed_entries:
            entry_date_str = entry.get('entry_date') # Assumes YYYY-MM-DD format
            if entry_date_str:
                try:
                    entry_dt_obj = date.fromisoformat(entry_date_str)
                    month_key = entry_dt_obj.strftime('%Y-%m') # YYYY-MM
                    if month_key not in entries_by_month:
                        entries_by_month[month_key] = []
                    entries_by_month[month_key].append(entry)
                except ValueError:
                    logger.warning(f"Could not parse date '{entry_date_str}' for entry ID {entry.get('page_id')}. Skipping for monthly grouping.")
            else:
                logger.warning(f"Entry ID {entry.get('page_id')} has no 'entry_date'. Skipping for monthly grouping.")

        if not entries_by_month:
            logger.warning("No entries could be grouped by month. Nothing will be saved.")
            # sys.exit(0) # Or handle as an error depending on expectations

        total_files_saved = 0
        total_entries_saved_across_files = 0

        for month_key, monthly_entries_list in entries_by_month.items():
            if not monthly_entries_list:
                logger.info(f"Skipping month {month_key} as it has no entries after filtering/processing issues.")
                continue

            # Generate filename like 'notion_export_month_YYYY-MM.json'
            # We need a date object for the filename generation, use the 1st of the month_key
            year_str, month_str = month_key.split('-')
            month_dt_for_filename = date(int(year_str), int(month_str), 1)
            
            base_filename = json_storage.generate_filename("month", dt=month_dt_for_filename) # Use "month" as period_name
            gcs_blob_name_monthly = f"{gcs_export_prefix_path}{base_filename}"

            metadata_for_gcs_monthly = {
                "export_time": datetime.now().isoformat(),
                "export_type": "all_data_monthly_batch",
                "month_exported": month_key,
                "date_property_used": args.date_property, # Original date property used for Notion query
                "total_entries_in_this_file": len(monthly_entries_list)
            }
            
            logger.info(f"Saving {len(monthly_entries_list)} entries for month '{month_key}' to GCS bucket '{GCS_BUCKET_NAME}' as blob '{gcs_blob_name_monthly}'...")
            
            saved_gcs_uri_monthly = json_storage.save_entries_to_gcs(
                entries=monthly_entries_list,
                gcs_bucket_name=GCS_BUCKET_NAME,
                gcs_blob_name=gcs_blob_name_monthly,
                metadata=metadata_for_gcs_monthly
            )

            if saved_gcs_uri_monthly:
                logger.info(f"Successfully saved entries for month '{month_key}' to {saved_gcs_uri_monthly}")
                total_files_saved += 1
                total_entries_saved_across_files += len(monthly_entries_list)
            else:
                logger.error(f"Failed to save entries for month '{month_key}' to GCS (gs://{GCS_BUCKET_NAME}/{gcs_blob_name_monthly}).")
                # Decide if to sys.exit(1) here or continue with other months

        logger.info(f"Completed monthly export of all data. Saved {total_files_saved} monthly files, containing a total of {total_entries_saved_across_files} entries.")
        if total_entries_saved_across_files != len(processed_entries):
             logger.warning(f"Discrepancy in entry count: Processed {len(processed_entries)} entries, but saved {total_entries_saved_across_files} across monthly files. Some entries might have been skipped due to missing/invalid dates.")


    logger.info("Extraction process completed.") 