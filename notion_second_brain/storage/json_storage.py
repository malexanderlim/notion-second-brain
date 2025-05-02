import json
import os
import logging
from datetime import date
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
        period: The time period ('day', 'week', 'month', 'year', 'all').
        dt: The specific date for day/week periods.
        year: The specific year for year/month periods.

    Returns:
        A formatted filename string (e.g., '2023-10-26.json', '2023-W43.json').
    """
    if period == 'day' and dt:
        return f"{dt.isoformat()}.json"
    elif period == 'week' and dt:
        # ISO week date: YYYY-Www (e.g., 2023-W43)
        return f"{dt.isocalendar().year}-W{dt.isocalendar().week:02d}.json" 
    elif period == 'month' and year and dt: # Need dt for month number
        return f"{year}-{dt.month:02d}.json"
    elif period == 'year' and year:
        return f"{year}.json"
    elif period == 'all':
        return "all_time.json"
    else:
        logger.warning(f"Could not generate filename for period '{period}' with provided date/year info.")
        # Fallback or raise error?
        return f"export_{datetime.now().strftime('%Y%m%d%H%M%S')}.json" 