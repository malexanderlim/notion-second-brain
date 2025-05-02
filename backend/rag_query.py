import faiss
import json
import os
import logging
from datetime import date, datetime, timedelta
import time
import asyncio # Ensure asyncio is imported if used for async operations

from openai import OpenAI, RateLimitError, APIError

# --- Configuration ---
# Consider moving these to a config file or environment variables
INDEX_PATH = "index.faiss"
MAPPING_PATH = "index_mapping.json"
METADATA_CACHE_PATH = "metadata_cache.json"
DATABASE_SCHEMA_PATH = "database_schema.json" # Assuming schema is dumped here
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_CHAT_MODEL = "gpt-4o" # Default chat model
QUERY_ANALYSIS_MODEL = "gpt-4o" # Model for analyzing query filters
FINAL_ANSWER_MODEL = "gpt-4o" # Model for generating the final answer
TOP_K = 15 # Default number of results to retrieve initially
MAX_EMBEDDING_RETRIES = 3
EMBEDDING_RETRY_DELAY = 5 # seconds

# Setup logger for this module
logger = logging.getLogger(__name__)
# Configure logging basic settings if not configured by the main app
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Globals (Load once) ---
openai_client = None
index = None
mapping_data = None
distinct_metadata_values = None
schema_properties = None

def load_rag_data():
    """Loads necessary data for RAG: index, mapping, metadata cache, schema."""
    global index, mapping_data, distinct_metadata_values, schema_properties
    
    try:
        logger.info(f"Loading FAISS index from {INDEX_PATH}...")
        index = faiss.read_index(INDEX_PATH)
        logger.info(f"Index loaded successfully. Total vectors: {index.ntotal}")
    except Exception as e:
        logger.error(f"Fatal: Failed to load FAISS index from {INDEX_PATH}: {e}", exc_info=True)
        raise RuntimeError(f"Could not load FAISS index from {INDEX_PATH}") from e

    try:
        logger.info(f"Loading index mapping from {MAPPING_PATH}...")
        with open(MAPPING_PATH, 'r') as f:
            mapping_data = json.load(f)
        logger.info(f"Mapping loaded successfully. Total entries: {len(mapping_data)}")
        # Convert string dates back to date objects if needed (ensure consistency)
        for entry in mapping_data.values():
            if 'Entry Date' in entry and isinstance(entry['Entry Date'], str):
                try:
                    entry['Entry Date'] = date.fromisoformat(entry['Entry Date'])
                except ValueError:
                    logger.warning(f"Could not parse date string: {entry['Entry Date']} for entry ID {entry.get('id')}")
                    entry['Entry Date'] = None # Or handle differently
    except FileNotFoundError:
        logger.error(f"Fatal: Mapping file not found at {MAPPING_PATH}")
        raise RuntimeError(f"Mapping file not found at {MAPPING_PATH}")
    except json.JSONDecodeError as e:
        logger.error(f"Fatal: Failed to decode JSON from {MAPPING_PATH}: {e}", exc_info=True)
        raise RuntimeError(f"Failed to decode JSON from {MAPPING_PATH}") from e
    except Exception as e:
        logger.error(f"Fatal: An unexpected error occurred loading mapping from {MAPPING_PATH}: {e}", exc_info=True)
        raise RuntimeError(f"Unexpected error loading {MAPPING_PATH}") from e

    try:
        logger.info(f"Loading distinct metadata values from {METADATA_CACHE_PATH}...")
        if os.path.exists(METADATA_CACHE_PATH):
            with open(METADATA_CACHE_PATH, 'r') as f:
                distinct_metadata_values = json.load(f)
                # Convert lists back to sets if they were stored as lists
                for key in distinct_metadata_values:
                    if isinstance(distinct_metadata_values[key], list):
                         distinct_metadata_values[key] = set(distinct_metadata_values[key])
            logger.info("Distinct metadata values cache loaded.")
        else:
            logger.warning(f"Metadata cache file not found at {METADATA_CACHE_PATH}. Proceeding without it.")
            distinct_metadata_values = None
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {METADATA_CACHE_PATH}: {e}. Proceeding without metadata cache.", exc_info=True)
        distinct_metadata_values = None
    except Exception as e:
        logger.error(f"Error loading metadata cache from {METADATA_CACHE_PATH}: {e}. Proceeding without metadata cache.", exc_info=True)
        distinct_metadata_values = None

    try:
        logger.info(f"Loading database schema from {DATABASE_SCHEMA_PATH}...")
        if os.path.exists(DATABASE_SCHEMA_PATH):
            with open(DATABASE_SCHEMA_PATH, 'r') as f:
                schema_data = json.load(f)
                schema_properties = schema_data.get('properties', {}) # Assuming schema is stored under 'properties' key
            logger.info("Database schema loaded.")
        else:
            logger.error(f"Fatal: Database schema file not found at {DATABASE_SCHEMA_PATH}. Cannot perform accurate query analysis.")
            raise RuntimeError(f"Database schema file not found at {DATABASE_SCHEMA_PATH}")
    except json.JSONDecodeError as e:
        logger.error(f"Fatal: Error decoding JSON from {DATABASE_SCHEMA_PATH}: {e}", exc_info=True)
        raise RuntimeError(f"Error decoding JSON from {DATABASE_SCHEMA_PATH}") from e
    except Exception as e:
        logger.error(f"Fatal: Error loading schema from {DATABASE_SCHEMA_PATH}: {e}", exc_info=True)
        raise RuntimeError(f"Error loading schema from {DATABASE_SCHEMA_PATH}") from e

def initialize_openai_client():
    """Initializes the OpenAI client using the API key from environment variables."""
    global openai_client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("Fatal: OPENAI_API_KEY environment variable not set.")
        raise ValueError("OPENAI_API_KEY environment variable not set.")
    openai_client = OpenAI(api_key=api_key)
    logger.info("OpenAI client initialized.")

# Ensure data is loaded and client is initialized when module is loaded
try:
    load_rag_data()
    initialize_openai_client()
except Exception as e:
    logger.critical(f"Failed to initialize RAG module: {e}", exc_info=True)
    # Depending on the application structure, might want to exit or prevent app start
    # For now, log critical error. The API endpoint will fail if globals are None.

# --- Helper Functions (Copied/Adapted from cli.py) ---

async def get_embedding(text: str, retries: int = MAX_EMBEDDING_RETRIES) -> list[float] | None:
    """Gets the embedding for the given text using the OpenAI API, with retries."""
    if not openai_client:
        logger.error("OpenAI client not initialized. Cannot get embedding.")
        return None
    attempt = 0
    while attempt < retries:
        try:
            response = await asyncio.to_thread(
                openai_client.embeddings.create,
                input=text,
                model=OPENAI_EMBEDDING_MODEL
            )
            # Ensure response structure is as expected
            if response.data and len(response.data) > 0 and response.data[0].embedding:
                logger.debug(f"Successfully obtained embedding for text snippet: '{text[:50]}...'")
                return response.data[0].embedding
            else:
                logger.error(f"Unexpected embedding response structure: {response}")
                return None # Or raise an error
        except RateLimitError as e:
            attempt += 1
            logger.warning(f"Rate limit hit during embedding request (attempt {attempt}/{retries}). Retrying in {EMBEDDING_RETRY_DELAY}s... Error: {e}")
            if attempt >= retries:
                logger.error(f"Max retries ({retries}) reached for embedding request due to rate limiting.")
                return None
            await asyncio.sleep(EMBEDDING_RETRY_DELAY)
        except APIError as e:
            attempt += 1
            logger.warning(f"API error during embedding request (attempt {attempt}/{retries}). Retrying in {EMBEDDING_RETRY_DELAY}s... Error: {e}")
            if attempt >= retries:
                logger.error(f"Max retries ({retries}) reached for embedding request due to API error.")
                return None
            await asyncio.sleep(EMBEDDING_RETRY_DELAY)
        except Exception as e:
            logger.error(f"Unexpected error getting embedding: {e}", exc_info=True)
            return None # Or raise a specific exception
    return None

async def analyze_query_for_filters(query: str) -> dict | None:
    """Analyzes the user query using an LLM to extract structured date and field filters."""
    if not openai_client:
        logger.error("OpenAI client not initialized. Cannot analyze query.")
        return None
    if not schema_properties:
        logger.error("Database schema not loaded. Cannot analyze query accurately.")
        return None # Cannot proceed without schema

    # Prepare schema and distinct value info for prompt
    field_descriptions = []
    # Use the globally loaded schema and distinct values
    current_schema = schema_properties or {}
    current_distinct_values = distinct_metadata_values or {}

    for name, details in current_schema.items():
        field_type = details.get('type', 'unknown')
        desc = f"- {name} (type: {field_type})"
        if name in current_distinct_values:
            known_values = current_distinct_values[name]
            if known_values:
                max_values_to_show = 50
                # Ensure known_values is a list or set before joining
                if isinstance(known_values, (list, set)):
                    values_list = list(known_values)
                    values_str = ", ".join(values_list[:max_values_to_show])
                    if len(values_list) > max_values_to_show:
                        values_str += f", ... ({len(values_list) - max_values_to_show} more)"
                    desc += f" | Known values: [{values_str}]"
                else:
                    logger.warning(f"Expected list or set for distinct values of '{name}', got {type(known_values)}")
        field_descriptions.append(desc)
    schema_prompt_part = "\n".join(field_descriptions)

    system_prompt = (
        "You are a query analysis assistant. Your task is to analyze the user query and the available Notion database fields "
        "(including known values for some fields) to extract structured filters. Identify potential entities like names, tags, dates, or date ranges mentioned in the query "
        "and map them to the most relevant field based on the provided schema AND the known values. Format the output as a JSON object. "
        "Recognize date ranges (like 'last year', '2024', 'next month', 'June 2023'). For date ranges, output a 'date_range' key "
        "with 'start' and 'end' sub-keys in 'YYYY-MM-DD' format. For specific field value filters, output a 'filters' key containing "
        "a list of objects, where each object has 'field' (the Notion property name) and 'contains' (the value extracted from the query). "
        "**Important:** Names of people are typically found in the 'Family' (relation) or 'Friends' (relation) fields. Use the 'Known values' list provided for these fields to help map names accurately. Map person names to THESE fields unless the query specifically asks about the entry's title (the 'Name' field). "
        "If a name could belong to either 'Family' or 'Friends' based on known values or context, include filters for BOTH fields. "
        "Match keywords mentioned in the query to the 'Known values' for the 'Tags' field where appropriate. "
        "If no specific filters are identified, return an empty JSON object {}."
    )

    user_prompt = f"""Available Notion Fields (with known values for some):
--- SCHEMA & VALUES START ---
{schema_prompt_part}
--- SCHEMA & VALUES END ---

User Query: "{query}"

Analyze the query based ONLY on the schema, known values, and query provided, following the mapping guidelines carefully. Output the structured filters as a JSON object:
"""

    logger.debug(f"Query Analysis System Prompt: {system_prompt}")
    logger.debug(f"Query Analysis User Prompt:\n{user_prompt}")

    try:
        response = await asyncio.to_thread(
             openai_client.chat.completions.create,
             model=QUERY_ANALYSIS_MODEL,
             messages=[
                 {"role": "system", "content": system_prompt},
                 {"role": "user", "content": user_prompt}
             ],
             temperature=0.1,
             response_format={ "type": "json_object" }
         )
        analysis_result = response.choices[0].message.content
        logger.info("Received query analysis result from LLM.")
        logger.debug(f"Raw analysis result: {analysis_result}")

        # Parse the JSON result
        filter_data = json.loads(analysis_result)
        # Basic validation
        if not isinstance(filter_data, dict):
            raise ValueError("Analysis result is not a dictionary.")
        if 'date_range' in filter_data and (not isinstance(filter_data['date_range'], dict) or
                                           'start' not in filter_data['date_range'] or
                                           'end' not in filter_data['date_range']):
             logger.warning(f"Invalid 'date_range' format received: {filter_data.get('date_range')}")
             filter_data.pop('date_range', None) # Remove invalid range
        if 'filters' in filter_data and not isinstance(filter_data['filters'], list):
             logger.warning(f"Invalid 'filters' format received: {filter_data.get('filters')}")
             filter_data.pop('filters', None) # Remove invalid filters

        logger.info(f"Parsed filter data: {json.dumps(filter_data)}")
        return filter_data

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from query analysis LLM response: {e}")
        logger.error(f"Raw response was: {analysis_result}")
        return None # Indicate failure to parse
    except Exception as e:
        logger.error(f"Error during query analysis LLM call: {e}", exc_info=True)
        return None # Indicate general failure

# --- Main RAG Query Function ---

async def perform_rag_query(user_query: str) -> dict:
    """
    Performs the full RAG process:
    1. Analyzes the query for filters.
    2. Pre-filters mapping data based on filters.
    3. Performs FAISS search (potentially targeted).
    4. Retrieves context.
    5. Generates a final answer using an LLM.
    Returns a dictionary containing the answer and source documents.
    """
    if not all([openai_client, index, mapping_data, schema_properties]):
         logger.error("RAG system not fully initialized. Cannot perform query.")
         # Consider raising an exception or returning a specific error structure
         return {"answer": "Error: RAG system not initialized.", "sources": []}

    start_time = time.time()
    logger.info(f"Starting RAG process for query: '{user_query}'")

    # 1. Analyze Query for Filters
    logger.info("Analyzing query for potential filters...")
    filter_analysis = await analyze_query_for_filters(user_query)
    if filter_analysis is None:
        logger.warning("Query analysis failed. Proceeding without pre-filtering.")
        filter_analysis = {}

    # 2. Pre-filter Mapping Data
    filtered_indices = set(range(index.ntotal)) # Start with all indices
    filtered_mapping_indices = [] # FAISS indices (0 to ntotal-1) corresponding to filtered mapping entries

    if filter_analysis:
        logger.info(f"Applying filters based on analysis: {json.dumps(filter_analysis)}")
        date_range = filter_analysis.get('date_range')
        field_filters = filter_analysis.get('filters', [])

        # Map FAISS index (0..ntotal-1) to mapping entry key (Notion Page ID)
        # Assuming mapping_data keys correspond to the order FAISS index was built
        # THIS IS A POTENTIALLY FRAGILE ASSUMPTION. A mapping file from FAISS index -> Notion ID is better.
        # For now, assuming list(mapping_data.keys()) gives the order. Need to verify index build process.
        # Let's reconstruct the mapping from index ID to page ID IF POSSIBLE from mapping_data
        # Assuming mapping_data structure is { "faiss_index": int, "id": "notion_page_id", ... }
        # If not, this filtering won't work correctly. NEED TO CHECK build_index.py
        # --- TEMPORARY ASSUMPTION: mapping_data values contain 'faiss_index' ---
        # Rebuild index_to_page_id map if mapping_data values have 'faiss_index'
        index_to_page_id = {}
        page_id_to_faiss_index = {}
        mapping_has_faiss_index = all('faiss_index' in entry for entry in mapping_data.values())

        if mapping_has_faiss_index:
            for entry in mapping_data.values():
                idx = entry.get('faiss_index')
                page_id = entry.get('id')
                if idx is not None and page_id is not None:
                    index_to_page_id[idx] = page_id
                    page_id_to_faiss_index[page_id] = idx
            logger.info(f"Reconstructed index_to_page_id map (size: {len(index_to_page_id)})")
        else:
            # Fallback (less reliable): Assume order matches index.keys()
            logger.warning("Mapping data does not contain 'faiss_index'. Assuming order matches FAISS index. This might be inaccurate.")
            page_ids_in_order = list(mapping_data.keys())
            index_to_page_id = {i: page_id for i, page_id in enumerate(page_ids_in_order) if i < index.ntotal}
            page_id_to_faiss_index = {page_id: i for i, page_id in index_to_page_id.items()}


        # Filter logic (modify as needed based on actual mapping_data structure)
        candidate_page_ids = set(mapping_data.keys()) # Start with all page IDs

        # Apply date filter
        if date_range and date_range.get('start') and date_range.get('end'):
            try:
                start_date_filter = date.fromisoformat(date_range['start'])
                end_date_filter = date.fromisoformat(date_range['end'])
                logger.info(f"Applying date filter: {start_date_filter} to {end_date_filter}")
                date_filtered_ids = set()
                for page_id, entry in mapping_data.items():
                    entry_date = entry.get('Entry Date') # Already parsed to date object hopefully
                    if entry_date and isinstance(entry_date, date):
                        if start_date_filter <= entry_date <= end_date_filter:
                             date_filtered_ids.add(page_id)
                candidate_page_ids.intersection_update(date_filtered_ids)
                logger.info(f"After date filter, {len(candidate_page_ids)} candidates remain.")
            except ValueError:
                logger.warning(f"Invalid date format in date_range filter: {date_range}. Skipping date filter.")

        # Apply field filters
        if field_filters:
             name_filters = [] # Special handling for OR logic across Family/Friends
             other_filters = []
             for f in field_filters:
                 field = f.get('field')
                 contains = f.get('contains')
                 if field and contains:
                     if field in ['Family', 'Friends']:
                         name_filters.append(f)
                     else:
                         other_filters.append(f)
                 else:
                     logger.warning(f"Skipping invalid field filter format: {f}")

             # Apply 'other' filters (AND logic)
             for f in other_filters:
                 field = f['field']
                 contains = f['contains'].lower() # Case-insensitive comparison for contains
                 logger.info(f"Applying filter: Field '{field}' contains '{contains}'")
                 field_filtered_ids = set()
                 for page_id in list(candidate_page_ids): # Iterate over copy
                     entry = mapping_data.get(page_id)
                     if entry and field in entry:
                         value = entry[field]
                         # Handle different types: string, list/set of strings
                         match = False
                         if isinstance(value, str):
                             if contains in value.lower():
                                 match = True
                         elif isinstance(value, (list, set)):
                             if any(contains in str(item).lower() for item in value):
                                 match = True
                         elif value is not None: # Handle other types if necessary
                             if contains in str(value).lower():
                                 match = True
                         
                         if match:
                            field_filtered_ids.add(page_id)
                 candidate_page_ids.intersection_update(field_filtered_ids)
                 logger.info(f"After filter '{field}' contains '{contains}', {len(candidate_page_ids)} candidates remain.")

             # Apply 'name' filters (OR logic across Family/Friends)
             if name_filters:
                 logger.info(f"Applying name filters (OR logic for Family/Friends): {name_filters}")
                 name_filtered_ids = set()
                 names_to_check = {f['contains'].lower() for f in name_filters}
                 
                 for page_id in list(candidate_page_ids):
                      entry = mapping_data.get(page_id)
                      if entry:
                           matches_name = False
                           # Check Family field
                           family_val = entry.get('Family')
                           if isinstance(family_val, (list, set)):
                                if any(name in str(item).lower() for item in family_val for name in names_to_check):
                                    matches_name = True
                           elif isinstance(family_val, str):
                                if any(name in family_val.lower() for name in names_to_check):
                                    matches_name = True
                                    
                           # Check Friends field if not already matched
                           if not matches_name:
                               friends_val = entry.get('Friends')
                               if isinstance(friends_val, (list, set)):
                                    if any(name in str(item).lower() for item in friends_val for name in names_to_check):
                                        matches_name = True
                               elif isinstance(friends_val, str):
                                    if any(name in friends_val.lower() for name in names_to_check):
                                        matches_name = True

                           if matches_name:
                               name_filtered_ids.add(page_id)
                 candidate_page_ids.intersection_update(name_filtered_ids)
                 logger.info(f"After name filters, {len(candidate_page_ids)} candidates remain.")


        # Convert final candidate page IDs to FAISS indices
        filtered_mapping_indices = [page_id_to_faiss_index[pid] for pid in candidate_page_ids if pid in page_id_to_faiss_index]
        
        if not filtered_mapping_indices:
            logger.warning("Pre-filtering resulted in zero candidates. Will search entire index.")
            filtered_indices = set(range(index.ntotal)) # Reset to all if filtering empty
        else:
            logger.info(f"Pre-filtering successful. Targeting {len(filtered_mapping_indices)} specific indices for search.")
            # Use IDSelectorBatch for targeted search
            filtered_indices = faiss.IDSelectorBatch(filtered_mapping_indices)

    else:
        logger.info("No filters extracted or applied. Searching entire index.")
        filtered_indices = None # Signal to search all

    # 3. Get Query Embedding
    logger.info("Getting embedding for the user query...")
    query_embedding = await get_embedding(user_query)
    if query_embedding is None:
        logger.error("Failed to get embedding for the query.")
        return {"answer": "Error: Could not process query embedding.", "sources": []}

    # 4. Perform FAISS Search
    k = TOP_K # Number of nearest neighbors to retrieve
    query_embedding_np = np.array([query_embedding]).astype('float32') # FAISS expects numpy array

    try:
        logger.info(f"Performing FAISS search (k={k})...")
        distances, indices = index.search(query_embedding_np, k, params=None if filtered_indices is None else faiss.SearchParametersIVF(sel=filtered_indices)) # Pass selector if available
        
        # Handle case where search returns fewer than k results or empty
        if indices.size == 0 or indices[0][0] == -1: # Check if any valid indices returned
             logger.warning("FAISS search returned no results.")
             retrieved_indices = []
        else:
             # Filter out invalid indices (-1) which can happen with selectors
             retrieved_indices = [idx for idx in indices[0] if idx != -1]
             logger.info(f"FAISS search completed. Retrieved {len(retrieved_indices)} potential indices.")
             
    except Exception as e:
        logger.error(f"Error during FAISS search: {e}", exc_info=True)
        return {"answer": "Error: Failed during similarity search.", "sources": []}

    # 5. Retrieve Context & Source Information
    context_parts = []
    sources = []
    retrieved_page_ids = set() # Track unique page IDs retrieved

    if not retrieved_indices:
        logger.warning("No relevant documents found after search.")
        # Handle this case - maybe return a specific message
        # return {"answer": "I couldn't find any specific entries matching your query.", "sources": []}

    logger.info("Retrieving content for top results...")
    for i in retrieved_indices:
        if i not in index_to_page_id:
             logger.warning(f"FAISS index {i} not found in index_to_page_id mapping. Skipping.")
             continue
             
        page_id = index_to_page_id[i]
        if page_id in retrieved_page_ids:
            continue # Already processed this page

        entry_data = mapping_data.get(page_id)
        if entry_data:
            retrieved_page_ids.add(page_id)
            content = entry_data.get('content', '')
            title = entry_data.get('Name', f"Untitled Entry ({page_id})")
            url = entry_data.get('url', '')
            entry_date_str = entry_data.get('Entry Date') # Expecting date object or None
            if isinstance(entry_date_str, date):
                 entry_date_str = entry_date_str.strftime('%Y-%m-%d') # Format for prompt

            # Construct context string
            context_entry = f"--- Entry Start ---
"
            context_entry += f"Title: {title}
"
            if entry_date_str:
                context_entry += f"Date: {entry_date_str}
"
            # Include other relevant metadata? e.g., Tags? Family? Friends? Be concise.
            tags = entry_data.get('Tags')
            if tags: context_entry += f"Tags: {', '.join(tags)}
"
            family = entry_data.get('Family')
            if family: context_entry += f"Family: {', '.join(family)}
"
            friends = entry_data.get('Friends')
            if friends: context_entry += f"Friends: {', '.join(friends)}
"
            context_entry += f"Content:
{content}
"
            context_entry += f"--- Entry End ---

"
            context_parts.append(context_entry)

            # Add to sources list
            if url: # Only add if URL is present
                sources.append({"title": title, "url": url})
        else:
            logger.warning(f"Page ID {page_id} found in search results but not in mapping_data.")

    context_string = "".join(context_parts)
    if not context_string:
         logger.warning("No context could be retrieved for the search results.")
         # Decide how to handle this - return specific message or let LLM try without context?
         # For now, let LLM try but it will likely fail gracefully.
         # return {"answer": "I found some potential matches, but couldn't retrieve their content.", "sources": []}


    # 6. Generate Final Answer using LLM
    logger.info("Generating final answer using LLM...")

    # Prepare final prompt
    # Refine based on experimentation (like in cli.py)
    final_system_prompt = (
        "You are a helpful assistant answering questions based on provided text snippets from personal journal entries. "
        "Use ONLY the provided context snippets to answer the user's query. Do not make assumptions or use external knowledge. "
        "Synthesize the information from the snippets to provide a concise and accurate answer. "
        "If the context contains relevant dates, names, or tags, mention them in your answer. "
        "If the context explicitly mentions someone being 'tagged', treat it as confirmation that the event or mention occurred (e.g., 'tagged = seen'). "
        "If the provided context does not contain the answer, state that you couldn't find the information in the provided entries. "
        "Format dates clearly (e.g., YYYY-MM-DD). "
        "Be helpful and conversational."
    )
    final_user_prompt = f"""Context from relevant journal entries:
--- CONTEXT START ---
{context_string if context_string else "No context was retrieved."}
--- CONTEXT END ---

User Query: "{user_query}"

Based *only* on the provided context, answer the user's query:
"""
    logger.debug(f"Final Answer System Prompt: {final_system_prompt}")
    logger.debug(f"Final Answer User Prompt Start:\nContext Length: {len(context_string)}\nQuery: {user_query}")

    try:
        response = await asyncio.to_thread(
            openai_client.chat.completions.create,
            model=FINAL_ANSWER_MODEL,
            messages=[
                {"role": "system", "content": final_system_prompt},
                {"role": "user", "content": final_user_prompt}
            ],
            temperature=0.5 # Allow for some synthesis/natural language generation
        )
        final_answer = response.choices[0].message.content
        logger.info("Successfully generated final answer.")

    except Exception as e:
        logger.error(f"Error during final answer LLM call: {e}", exc_info=True)
        final_answer = "Error: Failed to generate the final answer."
        # Keep sources even if answer generation fails? Or return empty sources too? Let's keep them.

    end_time = time.time()
    logger.info(f"RAG process completed in {end_time - start_time:.2f} seconds.")

    return {"answer": final_answer, "sources": sources}


# Need numpy for FAISS search input
import numpy as np 