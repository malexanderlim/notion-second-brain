import json
import os
import logging
from datetime import date, datetime
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def save_entries_to_json(
    entries: List[Dict[str, Any]], 
    filename: str, 
    output_dir: str = 'output', 
    metadata: Dict[str, Any] | None = None
) -> str | None:
    """Saves a list of journal entries to a JSON file.

    Args:
        entries: The list of entry dictionaries to save.
        filename: The desired name for the JSON file (e.g., '2023-10-26.json').
        output_dir: The directory where the JSON file should be saved. Defaults to 'output'.
        metadata: Optional dictionary with metadata to include at the top level of the JSON file.

    Returns:
        The full path to the saved file, or None if an error occurred.
    """
    if not filename.endswith('.json'):
        filename += '.json'

    output_path = os.path.join(output_dir, filename)

    try:
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Prepare data structure, optionally including metadata
        data_to_save = {}
        if metadata:
            data_to_save['metadata'] = metadata
        data_to_save['entries'] = entries

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=4, default=str) # Use default=str for non-serializable types like datetime
        
        logger.info(f"Successfully saved {len(entries)} entries to {output_path}")
        return output_path
    except IOError as e:
        logger.error(f"Error writing to JSON file {output_path}: {e}")
        return None
    except TypeError as e:
        logger.error(f"Error serializing data to JSON for {output_path}: {e}. Check for non-serializable types.")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred while saving to {output_path}: {e}")
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