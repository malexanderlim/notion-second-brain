from datetime import datetime, date, timedelta
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def get_entry_date(entry: Dict[str, Any], date_property: str = 'created_time') -> date | None:
    """Extracts the date part from a specified date/datetime string property in an entry."""
    date_str = entry.get(date_property)
    if not date_str:
        # Look inside properties if it's a page object with a specific date property
        properties = entry.get('properties', {})
        for prop_name, prop_value in properties.items():
            if prop_value.get('type') == 'date' and prop_value.get('date'):
                date_str = prop_value['date'].get('start')
                break # Use the first date property found
            elif prop_value.get('type') == 'created_time': # Fallback to page created time
                date_str = entry.get('created_time')
                break

    if date_str:
        try:
            # Handle ISO 8601 format (YYYY-MM-DDTHH:MM:SS.mmmZ or YYYY-MM-DD)
            dt_obj = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            return dt_obj.date()
        except ValueError:
            try:
                # Handle just date format YYYY-MM-DD
                return date.fromisoformat(date_str)
            except ValueError:
                logger.warning(f"Could not parse date string: {date_str} in entry {entry.get('id')}")
                return None
    else:
        logger.warning(f"Entry {entry.get('id')} does not have a usable date property ('{date_property}' or date type property).")
        return None

def filter_entries_by_day(entries: List[Dict[str, Any]], target_date: date, date_property: str = 'created_time') -> List[Dict[str, Any]]:
    """Filters entries created or dated on a specific day."""
    return [entry for entry in entries if get_entry_date(entry, date_property) == target_date]

def filter_entries_by_week(entries: List[Dict[str, Any]], target_date: date, date_property: str = 'created_time') -> List[Dict[str, Any]]:
    """Filters entries created or dated within the week (Mon-Sun) of the target date."""
    start_of_week = target_date - timedelta(days=target_date.weekday()) # Monday
    end_of_week = start_of_week + timedelta(days=6) # Sunday
    return [
        entry for entry in entries 
        if (entry_date := get_entry_date(entry, date_property)) 
        and start_of_week <= entry_date <= end_of_week
    ]

def filter_entries_by_month(entries: List[Dict[str, Any]], target_year: int, target_month: int, date_property: str = 'created_time') -> List[Dict[str, Any]]:
    """Filters entries created or dated within a specific month and year."""
    return [
        entry for entry in entries 
        if (entry_date := get_entry_date(entry, date_property)) 
        and entry_date.year == target_year 
        and entry_date.month == target_month
    ]

def filter_entries_by_year(entries: List[Dict[str, Any]], target_year: int, date_property: str = 'created_time') -> List[Dict[str, Any]]:
    """Filters entries created or dated within a specific year."""
    return [
        entry for entry in entries 
        if (entry_date := get_entry_date(entry, date_property)) 
        and entry_date.year == target_year
    ]

def filter_entries_by_date_range(entries: List[Dict[str, Any]], start_date: date, end_date: date, date_property: str = 'created_time') -> List[Dict[str, Any]]:
    """Filters entries created or dated within a specific date range (inclusive)."""
    return [
        entry for entry in entries 
        if (entry_date := get_entry_date(entry, date_property)) 
        and start_date <= entry_date <= end_date
    ] 