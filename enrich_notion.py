import typer
import notion_client
import os
import json
import time
import datetime
from dotenv import load_dotenv
from rich.console import Console
from rich.prompt import Prompt
# Import your LLM client library, e.g.:
# import openai
# from anthropic import Anthropic
from openai import OpenAI, RateLimitError, APIError # Add OpenAI imports

# Initialize console for rich output
console = Console()

# --- Configuration Loading ---
def load_configuration():
    """Loads API keys and Database ID from .env file."""
    load_dotenv()
    api_key = os.getenv("NOTION_TOKEN")
    llm_key = os.getenv("OPENAI_API_KEY")
    db_id = os.getenv("NOTION_DATABASE_ID")

    if not api_key:
        console.print("[bold red]Error: NOTION_TOKEN not found in .env file.[/bold red]")
        raise typer.Exit(code=1)
    if not db_id:
        console.print("[bold red]Error: NOTION_DATABASE_ID not found in .env file.[/bold red]")
        raise typer.Exit(code=1)
    # Add check for OPENAI_API_KEY
    if not llm_key:
        console.print("[bold yellow]Warning: OPENAI_API_KEY not found in .env file. LLM suggestions will be skipped.[/bold yellow]")

    return api_key, llm_key, db_id

def load_schema(schema_path="schema.json"):
    """Loads the database schema and extracts multi-select properties."""
    try:
        with open(schema_path, 'r') as f:
            schema = json.load(f)
        # Assuming schema.json structure is { "PropName": { "type": "multi_select", "multi_select": { "options": [...] } }, ... }
        multi_select_props = {
            prop_name: data
            for prop_name, data in schema.items()
            if data.get("type") == "multi_select"
        }
        if not multi_select_props:
             console.print(f"[bold yellow]Warning: No 'multi_select' properties found in {schema_path}[/bold yellow]")
        return multi_select_props
    except FileNotFoundError:
        console.print(f"[bold red]Error: Schema file not found at {schema_path}[/bold red]")
        raise typer.Exit(code=1)
    except json.JSONDecodeError:
        console.print(f"[bold red]Error: Could not decode JSON from {schema_path}[/bold red]")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[bold red]Error loading schema: {e}[/bold red]")
        raise typer.Exit(code=1)

# --- Notion API Helpers ---
def initialize_notion_client(api_key):
    """Initializes and returns the Notion client."""
    return notion_client.Client(auth=api_key)

def get_database_pages(notion, database_id, start_cursor=None, date_filter=None):
    """Fetches a batch of pages from the database, applying optional date filter."""
    query_params = {
        "database_id": database_id,
        "page_size": 100 # Max page size
    }
    if start_cursor:
        query_params["start_cursor"] = start_cursor
    if date_filter:
        query_params["filter"] = date_filter
        
    try:
        response = notion.databases.query(**query_params)
        return response
    except notion_client.APIResponseError as e:
        console.print(f"[bold red]Notion API Error querying database: {e}[/bold red]")
        return None # Or raise an exception

def get_page_text_content(notion, page_id):
    """Retrieves and concatenates text content from a page's blocks."""
    all_text = []
    next_cursor = None
    while True:
        try:
            response = notion.blocks.children.list(block_id=page_id, start_cursor=next_cursor, page_size=100)
            blocks = response.get("results", [])
            for block in blocks:
                block_type = block.get("type")
                if block_type in ["paragraph", "heading_1", "heading_2", "heading_3", 
                                  "bulleted_list_item", "numbered_list_item", "quote", "callout"]:
                    # Extract text from rich_text arrays within these block types
                    rich_text_elements = block.get(block_type, {}).get("rich_text", [])
                    for element in rich_text_elements:
                        if element.get("type") == "text":
                            all_text.append(element.get("text", {}).get("content", ""))
            
            if response.get("has_more"):
                next_cursor = response.get("next_cursor")
                time.sleep(0.5) # Small delay for pagination
            else:
                break
        except notion_client.APIResponseError as e:
            console.print(f"[bold red]Notion API Error fetching blocks for page {page_id}: {e}[/bold red]")
            break # Stop fetching blocks for this page on error
    
    return " ".join(all_text).strip()

def update_page_properties(notion, page_id, properties_payload):
    """Updates the properties of a specific page."""
    try:
        notion.pages.update(page_id=page_id, properties=properties_payload)
        return True
    except notion_client.APIResponseError as e:
        console.print(f"[bold red]Notion API Error updating page {page_id}: {e}[/bold red]")
        return False

# --- LLM Interaction ---
# Define the model to use
LLM_MODEL = "gpt-4o-mini" # Or "gpt-4o"

def get_llm_suggestions(page_content: str, current_properties: dict, available_options: dict, llm_key: str) -> dict | None:
    """Gets metadata suggestions from the specified OpenAI model."""
    if not llm_key:
        console.print("[yellow]Skipping LLM analysis: LLM_API_KEY not provided.[/yellow]")
        return None # Skip if no LLM key
    if not page_content:
        console.print("[yellow]Skipping LLM analysis: Page content is empty.[/yellow]")
        return None

    try:
        client = OpenAI(api_key=llm_key)
    except Exception as e:
        console.print(f"[bold red]Error initializing OpenAI client: {e}[/bold red]")
        return None

    # --- Construct the Prompt ---

    # System prompt defining the task, rules, and output format
    system_prompt = """You are an intelligent Notion metadata assistant. Your task is to analyze text from a Notion page and suggest relevant, *additional* multi-select options from a predefined list for specific properties.

You will be given:
1.  The page's text content.
2.  The currently selected options for relevant multi-select properties ('Current Selections').
3.  The complete list of available options for those properties ('Available Options').

Based ONLY on the provided page text:
- Identify options from the 'Available Options' list that are strongly relevant to the text.
- Suggest ONLY those relevant options that are NOT already present in the 'Current Selections' for that specific property.
- Do NOT suggest options already selected.
- Do NOT suggest options not present in the 'Available Options' list.
- Do NOT make up new options or categories.
- Focus on high confidence and direct relevance found within the text. Analyze the context carefully. For example, if the text mentions 'dinner with Randy Ang', and 'Randy Ang' is an available option under 'Friends', you should suggest adding 'Randy Ang' to 'Friends' if it's not already there. Similarly, if the text describes a hiking trip, suggest the 'Hike' tag if available and not selected.

Respond ONLY with a single, valid JSON object.
- The keys of the JSON object MUST be the property names (e.g., "Tags", "Food", "Family", "Friends").
- The values MUST be lists of the suggested option names (strings) to add for that property.
- Only include properties in the JSON object for which you have valid suggestions.
- If no new, relevant options are identified for ANY property, return an empty JSON object: {}
"""

    # User prompt providing the specific data for the current page
    user_prompt = f"""Analyze the following Notion page data:

--- PAGE TEXT START ---
{page_content}
--- PAGE TEXT END ---

--- CURRENT SELECTIONS START ---
{json.dumps(current_properties, indent=2)}
--- CURRENT SELECTIONS END ---

--- AVAILABLE OPTIONS START ---
{json.dumps(available_options, indent=2)}
--- AVAILABLE OPTIONS END ---

Suggest relevant options to add based on the rules provided. Respond ONLY with the JSON object.
"""

    console.print(f"[grey50]Sending request to LLM ({LLM_MODEL})...[/grey50]")

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2, # Lower temperature for more deterministic suggestions
            response_format={ "type": "json_object" } # Request JSON output
        )
        llm_output_str = response.choices[0].message.content
        console.print(f"[grey50]Received LLM response.[/grey50]")

        # Attempt to parse the JSON response
        suggestions = json.loads(llm_output_str)

        # Basic validation of the returned structure
        if not isinstance(suggestions, dict):
            console.print(f"[bold red]LLM Error: Response was not a JSON object.[/bold red]\nRaw response:\n{llm_output_str}")
            return None

        validated_suggestions = {}
        for prop, opts in suggestions.items():
            if prop in available_options: # Ensure suggested property exists
                 if isinstance(opts, list):
                      # Filter suggested options to ensure they are valid strings and exist in available_options
                      valid_opts = [str(opt) for opt in opts if isinstance(opt, str) and opt in available_options[prop]]
                      # Only add if there are valid options AND they are not already in current selections
                      current_opts_for_prop = set(current_properties.get(prop, []))
                      new_valid_opts = [opt for opt in valid_opts if opt not in current_opts_for_prop]
                      if new_valid_opts:
                           validated_suggestions[prop] = new_valid_opts
                 else:
                      console.print(f"[bold yellow]LLM Warning: Suggestions for property '{prop}' was not a list. Skipping.[/bold yellow]")
            else:
                 console.print(f"[bold yellow]LLM Warning: Suggested property '{prop}' not found in available options. Skipping.[/bold yellow]")

        return validated_suggestions # Return the validated suggestions

    except json.JSONDecodeError:
        console.print(f"[bold red]LLM Error: Failed to decode JSON response.[/bold red]\nRaw response:\n{llm_output_str}")
        return None
    except RateLimitError as e:
         console.print(f"[bold red]LLM API Error: Rate limit exceeded. {e}[/bold red]")
         # Consider adding a delay and retry here if needed
         return None
    except APIError as e:
         console.print(f"[bold red]LLM API Error: {e}[/bold red]")
         return None
    except Exception as e:
        console.print(f"[bold red]LLM Error: An unexpected error occurred: {e}[/bold red]")
        # Log the full traceback for debugging if possible
        import traceback
        traceback.print_exc()
        return None

# --- User Interaction ---
def prompt_user_for_updates(page_title, current_props, suggestions):
    """Displays suggestions and prompts the user for approval."""
    console.print(f"\n--- Suggestions for: [bold cyan]{page_title}[/bold cyan] ---")
    
    prompt_lines = []
    suggestion_map = {} # Maps index to (prop_name, option_name)
    current_index = 1

    for prop_name, suggested_options in suggestions.items():
        if not suggested_options: continue
        current_options_str = ", ".join(current_props.get(prop_name, []))
        console.print(f"[bold]{prop_name}[/bold]:")
        console.print(f"  Current: [grey50]{current_options_str if current_options_str else 'None'}[/grey50]")
        console.print(f"  Suggested New:")
        for option in suggested_options:
            console.print(f"    [green]({current_index}) {option}[/green]")
            suggestion_map[str(current_index)] = (prop_name, option)
            current_index += 1
        prompt_lines.append(f"{prop_name}: {len(suggested_options)}")
        
    if not suggestion_map:
        console.print("[yellow]No valid suggestions to display.[/yellow]")
        return None

    prompt_text = (
        f"Apply suggestions? Enter numbers (e.g., 1,3), 'all', or press Enter to skip: "
    )
    response = Prompt.ask(prompt_text, default="", show_default=False).strip().lower() # Hide default for cleaner prompt

    if not response:
        return None
    
    selected_updates = {prop: [] for prop in suggestions.keys()}
    
    if response == 'all':
        for prop_name, option_name in suggestion_map.values():
            selected_updates[prop_name].append(option_name)
    else:
        selected_indices = [idx.strip() for idx in response.split(',')]
        invalid_selections = []
        for idx in selected_indices:
            if idx in suggestion_map:
                prop_name, option_name = suggestion_map[idx]
                # Avoid adding duplicates if user enters same number twice
                if option_name not in selected_updates[prop_name]:
                     selected_updates[prop_name].append(option_name)
            elif idx: # Only warn if it's not an empty string resulting from trailing comma etc.
                invalid_selections.append(idx)

        if invalid_selections:
             console.print(f"[yellow]Ignoring invalid selection(s): {', '.join(invalid_selections)}[/yellow]")

    # Filter out properties with no selected options
    return {prop: opts for prop, opts in selected_updates.items() if opts}

# --- Helper Functions ---
def construct_date_filter(property_name="Date", start_date_str=None, end_date_str=None, year=None, month=None):
    """Constructs a Notion API date filter dictionary."""
    filters = []
    try:
        if start_date_str:
            start_dt = datetime.datetime.strptime(start_date_str, "%Y-%m-%d").date()
            filters.append({"property": property_name, "date": {"on_or_after": start_dt.isoformat()}})
        if end_date_str:
            end_dt = datetime.datetime.strptime(end_date_str, "%Y-%m-%d").date()
            filters.append({"property": property_name, "date": {"on_or_before": end_dt.isoformat()}})
        
        if year:
            if month:
                # Specific month
                start_of_month = datetime.date(year, month, 1)
                if month == 12:
                    end_of_month = datetime.date(year, 12, 31)
                else:
                    end_of_month = datetime.date(year, month + 1, 1) - datetime.timedelta(days=1)
                filters.append({"property": property_name, "date": {"on_or_after": start_of_month.isoformat()}})
                filters.append({"property": property_name, "date": {"on_or_before": end_of_month.isoformat()}})
            else:
                # Entire year
                start_of_year = datetime.date(year, 1, 1)
                end_of_year = datetime.date(year, 12, 31)
                filters.append({"property": property_name, "date": {"on_or_after": start_of_year.isoformat()}})
                filters.append({"property": property_name, "date": {"on_or_before": end_of_year.isoformat()}})

    except ValueError as e:
        console.print(f"[bold red]Invalid date format or value: {e}. Use YYYY-MM-DD for dates, 1-12 for month.[/bold red]")
        raise typer.Exit(code=1)

    if not filters:
        return None
    elif len(filters) == 1:
        return filters[0]
    else:
        # If multiple date conditions exist (e.g., year and month, or start and end), combine with AND
        return {"and": filters}

# --- Main Application Logic ---
app = typer.Typer(help="Analyzes Notion pages and suggests metadata enrichment using an LLM.")

@app.command()
def enrich_pages(
    schema_file: str = typer.Option("schema.json", help="Path to the schema JSON file."),
    start_date: str = typer.Option(None, "--start", help="Start date (YYYY-MM-DD) to filter pages (inclusive)."),
    end_date: str = typer.Option(None, "--end", help="End date (YYYY-MM-DD) to filter pages (inclusive)."),
    year: int = typer.Option(None, "--year", help="Specific year (YYYY) to filter pages."),
    month: int = typer.Option(None, "--month", help="Specific month (1-12) to filter pages (requires --year)."),
    limit: int = typer.Option(None, "--limit", help="Process only the first N pages found (after date filtering)."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Run without making any updates to Notion."),
    date_property: str = typer.Option("Date", help="Name of the date property in Notion to filter by."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress non-essential output (like pages skipped or with no suggestions).") # Added quiet flag
):
    """Analyzes Notion pages within a date range and suggests metadata enrichment."""
    console.print("[bold blue]Starting Notion Page Enrichment...[/bold blue]")
    if dry_run:
        console.print("[bold yellow]Dry Run Mode: No changes will be made to Notion.[/bold yellow]")
    if quiet:
        console.print("[grey50]Quiet mode enabled.[/grey50]")

    # --- Argument Validation ---
    if month and not year:
        console.print("[bold red]Error: --month option requires --year option.[/bold red]")
        raise typer.Exit(code=1)
    if month and not (1 <= month <= 12):
         console.print("[bold red]Error: --month must be between 1 and 12.[/bold red]")
         raise typer.Exit(code=1)
    if (start_date or end_date) and (year or month):
        console.print("[bold yellow]Warning: Using --start/--end dates. Ignoring --year/--month.[/bold yellow]")
        year, month = None, None # Prioritize start/end over year/month

    # 1. Load Config & Schema
    api_key, llm_key, db_id = load_configuration()
    multi_select_props = load_schema(schema_file)
    if not multi_select_props:
        console.print("[bold red]Error: No multi-select properties found in schema. Cannot proceed.[/bold red]")
        raise typer.Exit(code=1)
        
    # Create a map of available option names for easy lookup by the LLM prompt
    available_options = {prop: [opt['name'] for opt in data['multi_select']['options']]
                         for prop, data in multi_select_props.items()}

    # 2. Initialize Clients
    notion = initialize_notion_client(api_key)
    # LLM client is initialized within get_llm_suggestions

    # 3. Construct Date Filter (using the specified date_property)
    date_filter = construct_date_filter(property_name=date_property, start_date_str=start_date, end_date_str=end_date, year=year, month=month)

    # 4. Fetch Pages (handle pagination and apply date_filter)
    if not quiet:
        console.print(f"Fetching pages from Notion (filtering by '{date_property}')...")
    all_pages = []
    start_cursor = None
    pages_fetched_count = 0
    api_error_occurred = False
    while True:
        page_batch_response = get_database_pages(notion, db_id, start_cursor, date_filter=date_filter)
        if page_batch_response is None:
            api_error_occurred = True
            break # Stop fetching if API error occurred

        batch_results = page_batch_response.get('results', [])
        all_pages.extend(batch_results)
        pages_fetched_count += len(batch_results)
        if not quiet:
             console.print(f"Fetched {pages_fetched_count} pages...")

        # Check limit after extending the list
        if limit and pages_fetched_count >= limit:
            if not quiet:
                console.print(f"Reached page limit ({limit}).")
            all_pages = all_pages[:limit] # Trim excess pages if limit is precise
            break

        if not page_batch_response.get('has_more'):
            break # No more pages
        start_cursor = page_batch_response.get('next_cursor')
        
        # Add delay *before* next request to respect rate limits (slightly safer)
        time.sleep(0.5) # Reduced sleep slightly, Notion official limit is 3 req/sec avg

    if api_error_occurred:
        console.print("[bold red]Exiting due to Notion API error during page fetch.[/bold red]")
        raise typer.Exit(code=1)

    if not all_pages:
        # Always print this, even in quiet mode, as it indicates no work will be done.
        console.print("[yellow]No pages found matching the criteria.[/yellow]")
        raise typer.Exit(code=0) # Exit cleanly if no pages found

    if not quiet:
        console.print(f"Processing {len(all_pages)} pages total.")

    # 5. Process Each Page
    updated_count = 0
    skipped_count = 0
    llm_failures = 0
    suggestions_found_count = 0 # Track pages with suggestions
    for i, page in enumerate(all_pages):
        page_id = page['id']
        page_title = "Untitled"
        page_date_str = "N/A"
        has_suggestions = False # Flag for this page
        try:
            # Extract Title (handle potential lack of title)
            title_list = page.get('properties', {}).get('Name', {}).get('title', [])
            if title_list:
                page_title = title_list[0].get('plain_text', 'Untitled')
            # Extract Date (handle potential lack of date or different date formats)
            date_info = page.get('properties', {}).get(date_property, {}).get('date', {}) # Use specified date_property
            if date_info and date_info.get('start'):
                 page_date_str = date_info.get('start')
        except Exception as e:
             # Keep warning even in quiet mode?
             console.print(f"[yellow]Warning: Could not extract title/date for page {page_id}: {e}[/yellow]")

        if not quiet: # Only print analysis start if not quiet
             console.print(f"\n--- ({i+1}/{len(all_pages)}) Analyzing: [bold cyan]{page_title}[/bold cyan] (Date: {page_date_str}, ID: {page_id}) ---")

        # Extract current multi-select values
        current_props = {}
        for prop_name in multi_select_props.keys():
            try:
                # Ensure the property exists in the page data before accessing 'multi_select'
                prop_data = page.get('properties', {}).get(prop_name)
                if prop_data and prop_data.get('type') == 'multi_select' and 'multi_select' in prop_data:
                     current_props[prop_name] = [opt['name'] for opt in prop_data['multi_select']]
                else:
                     current_props[prop_name] = [] # Property might be missing or not multi-select type in this specific page
            except KeyError:
                current_props[prop_name] = [] # Fallback if property name itself is missing

        # Get page content
        page_content = get_page_text_content(notion, page_id)
        if not page_content:
            if not quiet:
                console.print("[yellow]Skipping page, no text content found.[/yellow]")
            skipped_count += 1
            continue

        # Get LLM suggestions
        suggestions = get_llm_suggestions(page_content, current_props, available_options, llm_key)

        if suggestions is None:
            # Always print LLM errors?
            console.print(f"[yellow]Skipping page '{page_title}' ({page_id}) due to LLM error or lack of API key.[/yellow]")
            skipped_count += 1
            llm_failures += 1
            continue

        if not suggestions:
            if not quiet:
                console.print("No new valid suggestions from LLM.")
            skipped_count += 1
            continue
        else:
            has_suggestions = True
            suggestions_found_count += 1
            # If quiet, print the analysis header only when suggestions are found
            if quiet:
                 console.print(f"\n--- ({i+1}/{len(all_pages)}) Suggestions Found: [bold cyan]{page_title}[/bold cyan] (Date: {page_date_str}, ID: {page_id}) ---")

        # Prompt user (Always prompt if suggestions exist)
        user_selections = prompt_user_for_updates(page_title, current_props, suggestions)

        # Update page if selections made and not dry_run
        if user_selections:
            properties_payload = {}
            needs_update = False
            for prop_name, new_opts_to_add in user_selections.items():
                # Ensure the property exists in current_props fetched from the page, otherwise default to empty list
                current_opts_set = set(current_props.get(prop_name, [])) 
                
                # Combine ensures uniqueness automatically
                combined_opts_set = current_opts_set.union(new_opts_to_add)

                # Check if the combined set is different from the current set
                if combined_opts_set != current_opts_set:
                    # Sort for consistent Notion display order (optional but nice)
                    sorted_combined_opts = sorted(list(combined_opts_set)) 
                    properties_payload[prop_name] = {'multi_select': [{'name': name} for name in sorted_combined_opts]}
                    needs_update = True

            if properties_payload and needs_update:
                if not dry_run:
                    # Print update message even if quiet?
                    console.print(f"Updating page [bold]{page_title}[/bold] (ID: {page_id})...")
                    time.sleep(0.5)
                    success = update_page_properties(notion, page_id, properties_payload)
                    if success:
                        console.print("[green]Page updated successfully.[/green]")
                        updated_count += 1
                    else:
                        console.print("[red]Failed to update page.[/red]")
                        skipped_count += 1 # Count update failure as skipped
                    time.sleep(0.5)
                else:
                    # Print dry run message even if quiet?
                    console.print(f"[yellow](Dry Run) Would update page [bold]{page_title}[/bold] with:[/yellow]")
                    console.print(json.dumps(properties_payload, indent=2))
                    updated_count += 1
            else:
                if not quiet: # Only print if not quiet
                     console.print("No actual changes selected or needed for this page.")
                skipped_count += 1 # No changes selected counts as skipped
        else:
            if not quiet: # Only print if not quiet
                console.print("Skipping update for this page (no selections)." )
            skipped_count += 1 # No user selection counts as skipped

    # Always print the final summary
    console.print(f"\n[bold green]Enrichment process complete.[/bold green]")
    console.print(f" - Pages processed: {len(all_pages)}")
    console.print(f" - Pages with suggestions found: {suggestions_found_count}")
    console.print(f" - Pages updated (or would update in dry run): {updated_count}")
    console.print(f" - Pages skipped (no content, no suggestions, user skip, API error, update failure): {skipped_count}")
    if llm_failures > 0:
         console.print(f" - LLM suggestion failures: {llm_failures}")

if __name__ == "__main__":
    app() 