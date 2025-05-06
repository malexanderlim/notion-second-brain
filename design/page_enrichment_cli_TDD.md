# Technical Design Document: Notion Page Enrichment CLI

**Metadata:**

*   `Title:` Notion Page Enrichment CLI Tool
*   `Status:` Proposed
*   `Author(s):` AI Assistant (based on user idea)
*   `Date:` 2024-07-27
*   `Related Docs:` `schema.json`

**1. Introduction / Motivation / Problem:**

Many existing pages within the target Notion database lack comprehensive metadata (specifically multi-select properties like `Tags`, `Food`, `Family`, `Friends`) compared to more recent entries. This inconsistency hinders effective searching, filtering, and retrieval of information. This document proposes a command-line interface (CLI) tool that utilizes a Large Language Model (LLM) to analyze page content and suggest relevant, existing metadata options (defined in `schema.json`) to enrich older pages, thereby improving data consistency and usability.

**2. Goals:**

*   Develop a CLI tool that interacts with the Notion API and an LLM API.
*   Fetch pages from a specified Notion database.
*   For each page, extract text content and existing multi-select property values.
*   Use an LLM to analyze the text content and suggest *additional*, relevant options from the pre-defined choices in `schema.json` for multi-select properties (`Tags`, `Food`, `Family`, `Friends`).
*   Present the current page details and LLM suggestions to the user via the CLI.
*   Allow the user to selectively approve or reject suggested additions for each page.
*   Update the Notion page properties via the API based on user approval.
*   Handle Notion API pagination and rate limits gracefully.
*   Load configuration (API keys, Database ID, `schema.json` path) securely.

**3. Non-Goals:**

*   Creating *new* tag or property options not already present in `schema.json`.
*   Updating property types other than the specified multi-selects (unless explicitly added later).
*   Performing complex analysis of non-text block types (images, embeds, etc.). Content analysis will primarily focus on concatenated text from blocks.
*   Fully automated updates without user validation.
*   Providing a graphical user interface (GUI).

**4. Proposed Design / Technical Solution:**

The tool will be a Python script run from the command line.

*   **Configuration:**
    *   Use environment variables or a configuration file (e.g., `.env`) to store Notion API Key, LLM API Key, and the target Notion Database ID.
    *   Load the `schema.json` file at startup to get the definitions and available options for multi-select properties.
*   **Core Logic:**
    1.  **Initialization:** Load config, initialize Notion client (e.g., using `notion-client`) and LLM client (e.g., OpenAI, Anthropic).
    2.  **Page Retrieval:** Query the Notion database using `POST /v1/databases/{database_id}/query`. Handle pagination to fetch all pages or a user-defined subset/date range.
    3.  **Page Processing Loop:** Iterate through fetched pages.
        *   **Data Extraction:** Get page ID, title, date, and current values for relevant multi-select properties (`Tags`, `Food`, `Family`, `Friends`).
        *   **Content Retrieval:** Fetch page blocks using `GET /v1/blocks/{page_id}/children`. Handle pagination if necessary. Concatenate text content from supported block types (paragraph, headings, bullets, etc.) into a single string.
        *   **LLM Interaction:**
            *   Construct a prompt for the LLM. The prompt will include:
                *   The concatenated page text content.
                *   The *currently selected* options for each relevant multi-select property.
                *   The *full list of available options* for each relevant multi-select property (from `schema.json`).
                *   Clear instructions to identify *only relevant options* from the *available list* that are *not already selected* and appear to be strongly supported by the text content. Instruct the LLM to output suggestions in a structured format (e.g., JSON).
            *   Send the prompt to the LLM API.
        *   **Suggestion Parsing:** Parse the LLM response to extract suggested additions for each property.
    4.  **User Interaction:**
        *   If suggestions exist, display: Page Title, Date, Current Properties, Suggested New Properties (grouped by property type).
        *   Prompt the user to select which suggestions to apply (e.g., comma-separated list like `Tags:TagA,Food:FoodC`, `all`, or Enter to skip). Use a library like `rich` or `inquirer` for a better CLI experience.
    5.  **Page Update:**
        *   If the user approves additions:
            *   Construct the `properties` payload for the Notion API. This involves merging the existing multi-select options with the user-approved new ones, ensuring no duplicates.
            *   Call `PATCH /v1/pages/{page_id}` with the updated `properties` object.
            *   Report success or failure to the user.
        *   Implement delays between API calls (fetch and patch) to avoid rate limiting.
*   **Schema Handling:** The tool will dynamically read `schema.json` to determine which properties are multi-select and what their valid options are. This makes the tool adaptable if the schema changes.
*   **Error Handling:** Implement robust error handling for API requests (Notion & LLM), file loading, and user input.

*   **4.1 Script Outline (Python - using `typer`, `notion-client`, `python-dotenv`, `rich`):**
    ```python
    import typer
    import notion_client
    import os
    import json
    import time
    from dotenv import load_dotenv
    from rich.console import Console
    from rich.prompt import Prompt
    # Potentially import LLM client library (e.g., openai, anthropic)

    # Initialize console for rich output
    console = Console()

    # --- Configuration Loading ---
    def load_configuration():
        load_dotenv()
        # Fetch NOTION_API_KEY, LLM_API_KEY, NOTION_DATABASE_ID
        # Handle missing keys
        return api_key, llm_key, db_id

    def load_schema(schema_path="schema.json"):
        # Load schema.json
        # Extract multi-select properties and their options
        # Return a structure like: {'Tags': [opt1, opt2], 'Food': [optA, optB], ...}
        return multi_select_props

    # --- Notion API Helpers ---
    def initialize_notion_client(api_key):
        return notion_client.Client(auth=api_key)

    def get_database_pages(notion, database_id, start_cursor=None, date_filter=None):
        # Call notion.databases.query with pagination handling
        # Return list of page objects
        pass

    def get_page_text_content(notion, page_id):
        # Call notion.blocks.children.list with pagination
        # Extract text from relevant block types (paragraph, heading_*, bulleted_list_item, etc.)
        # Concatenate into a single string
        # Return text content
        pass

    def update_page_properties(notion, page_id, properties_payload):
        # Call notion.pages.update(page_id=page_id, properties=properties_payload)
        # Return success/failure
        pass

    # --- LLM Interaction ---
    def get_llm_suggestions(page_content, current_properties, available_options):
        # Initialize LLM client (if not already global)
        # Construct the detailed prompt (as described in TDD Section 4)
        # Call LLM API
        # Parse the response (expecting structured output like JSON)
        # Handle potential LLM errors or empty responses
        # Return suggestions like: {'Tags': ['NewTag1'], 'Food': ['NewFoodA']}
        pass

    # --- User Interaction ---
    def prompt_user_for_updates(page_title, current_props, suggestions):
        # Use rich console to display current info and suggestions clearly
        # Example prompt:
        # console.print(f"Processing: [bold]{page_title}[/bold]")
        # console.print("Current Tags:", current_props.get('Tags', []))
        # console.print("Suggested New Tags:", suggestions.get('Tags', []))
        # ... other properties ...
        # response = Prompt.ask("Enter selections (e.g., 'Tags:TagA,Food:FoodC'), 'all', or press Enter to skip")
        # Parse user response
        # Return parsed selections or None
        pass

    # --- Main Application Logic ---
    app = typer.Typer()

    @app.command()
    def enrich_pages(
        schema_file: str = typer.Option("schema.json", help="Path to the schema JSON file."),
        start_date: str = typer.Option(None, help="Start date (YYYY-MM-DD) to filter pages."),
        end_date: str = typer.Option(None, help="End date (YYYY-MM-DD) to filter pages."),
        year: int = typer.Option(None, help="Specific year (YYYY) to filter pages."),
        month: int = typer.Option(None, help="Specific month (1-12) to filter pages (requires --year).")
    ):
        """Analyzes Notion pages within a date range and suggests metadata enrichment."""
        console.print("[bold blue]Starting Notion Page Enrichment...[/bold blue]")
        
        # Validate date arguments (e.g., month requires year)
        # Construct Notion API date filter based on provided arguments
        date_filter = None
        if start_date and end_date:
             # Create filter for date range
             pass # Placeholder
        elif year and month:
             # Create filter for specific month
             pass # Placeholder
        elif year:
             # Create filter for specific year
             pass # Placeholder

        # 1. Load Config & Schema
        api_key, llm_key, db_id = load_configuration()
        multi_select_props = load_schema(schema_file)
        available_options = {prop: [opt['name'] for opt in data['multi_select']['options']] 
                             for prop, data in multi_select_props.items()}

        # 2. Initialize Clients
        notion = initialize_notion_client(api_key)
        # Initialize LLM client here if needed
        
        # 3. Fetch Pages (handle pagination and apply date_filter)
        all_pages = []
        start_cursor = None
        while True:
            # Modify get_database_pages to accept and use the date_filter
            page_batch = get_database_pages(notion, db_id, start_cursor, date_filter=date_filter)
            all_pages.extend(page_batch['results'])
            if not page_batch['has_more']:
                break
            start_cursor = page_batch['next_cursor']
            console.print(f"Fetched {len(all_pages)} pages...")
            time.sleep(1) # Basic rate limiting

        console.print(f"Processing {len(all_pages)} pages within the specified date range.")

        # 4. Process Each Page
        for page in all_pages:
            page_id = page['id']
            page_title = "Untitled" # Extract title properly
            try:
                page_title = page['properties']['Name']['title'][0]['plain_text']
            except (KeyError, IndexError): pass
            
            console.print(f"\n--- Analyzing: [bold cyan]{page_title}[/bold cyan] (ID: {page_id}) ---")

            # Extract current multi-select values
            current_props = {}
            for prop_name in multi_select_props.keys():
                try:
                    current_props[prop_name] = [opt['name'] for opt in page['properties'][prop_name]['multi_select']]
                except KeyError: current_props[prop_name] = []

            # Get page content
            page_content = get_page_text_content(notion, page_id)
            if not page_content:
                console.print("[yellow]Skipping page, no text content found.[/yellow]")
                continue
            
            # Get LLM suggestions
            suggestions = get_llm_suggestions(page_content, current_props, available_options)
            if not suggestions:
                console.print("No new suggestions from LLM.")
                continue

            # Prompt user
            user_selections = prompt_user_for_updates(page_title, current_props, suggestions)

            # Update page if selections made
            if user_selections:
                properties_payload = {}
                # Construct payload by merging current + selected new options
                for prop_name, new_opts in user_selections.items():
                    combined_opts = list(set(current_props.get(prop_name, []) + new_opts))
                    properties_payload[prop_name] = {'multi_select': [{'name': name} for name in combined_opts]}
                
                if properties_payload:
                    console.print(f"Updating page {page_id}...")
                    success = update_page_properties(notion, page_id, properties_payload)
                    if success:
                        console.print("[green]Page updated successfully.[/green]")
                    else:
                        console.print("[red]Failed to update page.[/red]")
                    time.sleep(1) # Basic rate limiting
            else:
                console.print("Skipping update for this page.")

        console.print("\n[bold green]Enrichment process complete.[/bold green]")

    if __name__ == "__main__":
        app()

**5. Alternatives Considered:**

*   **Manual Enrichment:** Too time-consuming and prone to inconsistency for a large number of pages.
*   **Fully Automated Updates:** Risky due to potential LLM inaccuracies; could lead to incorrect metadata being added without oversight.
*   **Simple Keyword Matching:** Less effective than LLM analysis for understanding context and identifying relevant but not explicitly mentioned concepts (e.g., inferring "Vacation" tag from descriptions of travel).

The proposed semi-automated CLI approach balances efficiency with user control.

**6. Impact / Risks / Open Questions:**

*   **Impact:** Improved data consistency, enhanced search/retrieval capabilities within the Notion database.
*   **Risks:**
    *   **LLM Cost/Quality:** API calls to the LLM provider will incur costs. Suggestion quality depends on the model and prompt.
    *   **Notion API Rate Limits:** Processing many pages quickly could hit rate limits. Requires careful pacing.
    *   **Prompt Engineering:** Getting the LLM to reliably suggest only relevant, *new* tags from the *existing* list requires careful prompt design and potentially iteration.
    *   **Schema Changes:** Significant changes to `schema.json` structure might require updates to the tool's logic.
*   **Open Questions:**
    *   What is the optimal LLM prompt for balancing accuracy and recall?
    *   What are the acceptable API cost and processing time constraints?
    *   Should specific pages be excluded (e.g., based on creation date or existing tags)?

**7. (Optional) Implementation Plan / Phases:**

1.  **Setup:** Project structure, dependency management (`requirements.txt`), basic CLI framework (`argparse`, `typer`).
2.  **Notion Integration:** Implement functions to fetch database schema, fetch pages, fetch block content, and update page properties. Handle authentication and rate limits.
3.  **LLM Integration:** Implement function to interact with the chosen LLM API, including prompt construction and response parsing.
4.  **Core Logic:** Integrate Notion and LLM parts, implement the main page processing loop.
5.  **User Interface:** Develop the CLI interaction for displaying suggestions and capturing user input.
6.  **Testing & Refinement:** Test with various pages, refine prompts, add error handling.

**8. (Optional) Future Work / Follow-on:**

*   Support for suggesting updates to other property types (e.g., select, relation).
*   More sophisticated content analysis (e.g., handling images with OCR, analyzing linked pages).
*   Option for batch processing/approval.
*   Integration into a more automated workflow (e.g., triggered by page updates). 