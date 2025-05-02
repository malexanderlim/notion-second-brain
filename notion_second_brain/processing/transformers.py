import logging
from datetime import date
from typing import Dict, Any, List

from ..notion.extractors import extract_text_from_rich_text_array # Reuse helper

logger = logging.getLogger(__name__)

def get_property_value(properties: Dict[str, Any], prop_name: str, prop_type: str) -> Any:
    """Safely extracts a value from a Notion properties object based on name and type."""
    prop_data = properties.get(prop_name)
    if not prop_data or prop_data.get('type') != prop_type:
        logger.debug(f"Property '{prop_name}' of type '{prop_type}' not found or type mismatch.")
        return None

    try:
        if prop_type == "title":
            return extract_text_from_rich_text_array(prop_data['title'])
        elif prop_type == "date":
            date_info = prop_data.get('date')
            return date_info.get('start') if date_info else None
        elif prop_type == "multi_select":
            return [item.get('name') for item in prop_data.get('multi_select', [])]
        elif prop_type == "rich_text":
            return extract_text_from_rich_text_array(prop_data['rich_text'])
        # Add more type handlers as needed (select, number, files, relation, etc.)
        else:
            logger.warning(f"Unhandled property type '{prop_type}' for property '{prop_name}'.")
            return None
    except KeyError as e:
        logger.error(f"KeyError accessing data for property '{prop_name}' (type '{prop_type}'): {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error processing property '{prop_name}': {e}")
        return None

def transform_page_to_simple_dict(page: Dict[str, Any]) -> Dict[str, Any]:
    """Transforms a raw Notion page object into a simpler dictionary for indexing/querying."""
    properties = page.get('properties', {})
    
    # Extract data using the helper function
    title = get_property_value(properties, 'Name', 'title') or '[No Title]'
    entry_date = get_property_value(properties, 'Date', 'date')
    tags = get_property_value(properties, 'Tags', 'multi_select') or []
    food = get_property_value(properties, 'Food', 'multi_select') or []
    friends = get_property_value(properties, 'Friends', 'multi_select') or []
    family = get_property_value(properties, 'Family', 'multi_select') or []
    ai_summary = get_property_value(properties, 'AI summary', 'rich_text') or ""
    
    # Add other known/expected properties here...

    simple_data = {
        "page_id": page.get('id'),
        "created_time": page.get('created_time'),
        "last_edited_time": page.get('last_edited_time'),
        "url": page.get('url'),
        "entry_date": entry_date, # Explicitly named date from property
        "title": title,
        "tags": tags,
        "food": food,
        "friends": friends,
        "family": family,
        "ai_summary": ai_summary,
        # Placeholder for block content, to be added later
        "content": ""
    }

    return {k: v for k, v in simple_data.items() if v is not None} # Clean out None values if desired 