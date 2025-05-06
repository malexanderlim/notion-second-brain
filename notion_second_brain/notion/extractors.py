import logging

logger = logging.getLogger(__name__)

def extract_text_from_rich_text_array(rich_text_array: list) -> str:
    """Extracts and concatenates plain text from a Notion rich_text array."""
    return "".join([item.get('plain_text', '') for item in rich_text_array])

def extract_text_from_block(block: dict) -> str:
    """Extracts text content from a single Notion block object.

    Args:
        block: A dictionary representing a Notion block object.

    Returns:
        A string containing the extracted text, or an empty string if no text.
    """
    block_type = block.get('type')
    text_content = ""

    if not block_type:
        logger.warning("Block has no type: %s", block.get('id'))
        return ""

    try:
        if block_type == 'paragraph' and 'paragraph' in block:
            text_content = extract_text_from_rich_text_array(block['paragraph']['rich_text'])
        elif block_type.startswith('heading_') and block_type in block:
            text_content = extract_text_from_rich_text_array(block[block_type]['rich_text'])
        elif block_type == 'bulleted_list_item' and 'bulleted_list_item' in block:
            # Add a bullet point prefix for clarity
            text = extract_text_from_rich_text_array(block['bulleted_list_item']['rich_text'])
            text_content = f"- {text}" if text else ""
        elif block_type == 'numbered_list_item' and 'numbered_list_item' in block:
            # Add a number prefix (Note: Notion doesn't provide the number)
            text = extract_text_from_rich_text_array(block['numbered_list_item']['rich_text'])
            text_content = f"1. {text}" if text else "" # Placeholder number
        elif block_type == 'to_do' and 'to_do' in block:
            checked = block['to_do'].get('checked', False)
            prefix = "[x]" if checked else "[ ]"
            text = extract_text_from_rich_text_array(block['to_do']['rich_text'])
            text_content = f"{prefix} {text}" if text else ""
        elif block_type == 'toggle' and 'toggle' in block:
            text_content = extract_text_from_rich_text_array(block['toggle']['rich_text'])
            # Note: Content inside the toggle (children blocks) is not extracted here.
            # This would require recursively fetching child blocks.
        elif block_type == 'quote' and 'quote' in block:
            text_content = extract_text_from_rich_text_array(block['quote']['rich_text'])
        elif block_type == 'callout' and 'callout' in block:
            text_content = extract_text_from_rich_text_array(block['callout']['rich_text'])
        elif block_type == 'divider':
            text_content = "---" # Represent divider
        elif block_type == 'code' and 'code' in block:
            text_content = extract_text_from_rich_text_array(block['code']['rich_text'])
            # Optionally add language if needed: block['code'].get('language')
        elif block_type in ['child_page', 'child_database', 'embed', 'image', 'video', 'file', 'pdf', 'bookmark', 'equation', 'table_of_contents', 'breadcrumb', 'column_list', 'column', 'link_preview', 'synced_block', 'template', 'link_to_page', 'table', 'table_row']:
            # These blocks either don't have direct text or require special handling (like fetching children)
            logger.debug(f"Skipping block type '{block_type}' for direct text extraction: {block.get('id')}")
            pass 
        else:
            # Log unhandled block types
            logger.warning(f"Unhandled block type '{block_type}' encountered: {block.get('id')}")

    except KeyError as e:
        logger.error(f"KeyError extracting text from block type '{block_type}' (ID: {block.get('id')}): {e}")
    except Exception as e:
        logger.error(f"Unexpected error extracting text from block (ID: {block.get('id')}): {e}")

    # Add a newline for separation between block texts
    return text_content + "\n" if text_content else ""

def extract_page_content(page_blocks: list) -> str:
    """Extracts and concatenates text from a list of Notion blocks (typically a page's content)."""
    full_text = ""
    for block in page_blocks:
        full_text += extract_text_from_block(block)
    return full_text.strip() # Remove trailing newline

# Note: To get the content of a page, you first need to use the `blocks.children.list` endpoint
# using the page ID. The `query_database` result only gives page metadata, not block content.
# The NotionClient would need a new method like `retrieve_block_children(block_id)`. 