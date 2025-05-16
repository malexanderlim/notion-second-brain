import json
import os
import logging
from datetime import date, datetime
from typing import List, Dict, Any
from google.cloud import storage # Added for GCS

logger = logging.getLogger(__name__)

def save_entries_to_gcs(
    entries: List[Dict[str, Any]],
    gcs_bucket_name: str,
    gcs_blob_name: str, # Full path in the bucket, e.g., "exports/2023-10.json"
    metadata: Dict[str, Any] | None = None
) -> str | None:
    """Saves a list of journal entries to a JSON file in Google Cloud Storage.

    Args:
        entries: The list of entry dictionaries to save.
        gcs_bucket_name: The GCS bucket name.
        gcs_blob_name: The full path/name for the blob in the GCS bucket.
        metadata: Optional dictionary with metadata to include at the top level of the JSON file.

    Returns:
        The GCS URI (gs://bucket_name/blob_name) of the saved file, or None if an error occurred.
    """
    if not gcs_blob_name.endswith('.json'):
        gcs_blob_name += '.json'

    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(gcs_bucket_name)
        blob = bucket.blob(gcs_blob_name)

        data_to_save = {}
        if metadata:
            data_to_save['metadata'] = metadata
        data_to_save['entries'] = entries

        json_data = json.dumps(data_to_save, ensure_ascii=False, indent=4, default=str)
        
        blob.upload_from_string(json_data, content_type='application/json')
        
        gcs_uri = f"gs://{gcs_bucket_name}/{gcs_blob_name}"
        logger.info(f"Successfully saved {len(entries)} entries to {gcs_uri}")
        return gcs_uri
    except Exception as e: # Catching a broader exception for GCS operations
        logger.error(f"Error writing to GCS (gs://{gcs_bucket_name}/{gcs_blob_name}): {e}", exc_info=True)
        return None

def generate_filename(period: str, dt: date | None = None, year: int | None = None) -> str:
    """Generates a filename based on the time period.
    
    Args:
        period: The time period ('day', 'week', 'month', 'year', 'all', 'range').
                Note: 'range' will likely use the start date and fallback filename for now.
        dt: The specific date, often the start date of the period (used for day, week, month, range).
        year: The specific year (used primarily for 'year' period, can be derived from dt for others).

    Returns:
        A formatted filename string (e.g., '2023-10-26.json', '2023-W43.json', '2024-01.json').
    """
    try:
        if period == 'day' and dt:
            return f"{dt.isoformat()}.json"
        elif period == 'week' and dt:
            # ISO week date: YYYY-Www (e.g., 2023-W43)
            return f"{dt.isocalendar().year}-W{dt.isocalendar().week:02d}.json" 
        elif period == 'month' and dt: # Simplified: rely only on dt when period is month
            return f"{dt.year}-{dt.month:02d}.json"
        elif period == 'year' and dt: # Also handle year using dt if available
            return f"{dt.year}.json"   
        elif period == 'year' and year: # Fallback if only year is known
            return f"{year}.json"
        elif period == 'all':
            return "all_time.json"
        else: # Handles 'range' or cases where dt/year might be missing unexpectedly
            period_qualifier = f"_{period}" if period else ""
            date_qualifier = f"_{dt.isoformat()}" if dt else ""
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            fallback_name = f"export{period_qualifier}{date_qualifier}_{timestamp}.json"
            logger.warning(f"Could not determine specific filename format for period '{period}' with dt='{dt}', year='{year}'. Using fallback: {fallback_name}")
            return fallback_name
    except Exception as e:
        logger.error(f"Error generating filename for period '{period}', dt='{dt}', year='{year}': {e}", exc_info=True)
        # Fallback in case of unexpected errors during formatting
        return f"export_error_{datetime.now().strftime('%Y%m%d%H%M%S')}.json" 