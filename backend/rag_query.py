import faiss
import json
import os
import logging
from datetime import date, datetime, timedelta
import time
import asyncio
import numpy as np

from openai import OpenAI, RateLimitError, APIError

# --- Configuration ---
INDEX_PATH = "index.faiss"
MAPPING_PATH = "index_mapping.json"
METADATA_CACHE_PATH = "metadata_cache.json"
DATABASE_SCHEMA_PATH = "schema.json" # Corrected schema path
OPENAI_EMBEDDING_MODEL = "text-embedding-ada-002"
FINAL_ANSWER_MODEL = "gpt-4o"
QUERY_ANALYSIS_MODEL = "gpt-4o"
TOP_K = 15
MAX_EMBEDDING_RETRIES = 3
EMBEDDING_RETRY_DELAY = 5

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Globals (Load once) ---
openai_client = None
index = None
mapping_data_list: list[dict] | None = None # Expecting list of entry dicts
index_to_entry: dict[int, dict] | None = None # Map FAISS index to entry dict
distinct_metadata_values: dict | None = None
schema_properties: dict | None = None

def load_rag_data():
    """Loads necessary data for RAG: index, mapping (list), metadata cache, schema."""
    global index, mapping_data_list, index_to_entry, distinct_metadata_values, schema_properties
    
    try:
        logger.info(f"Loading FAISS index from {INDEX_PATH}...")
        index = faiss.read_index(INDEX_PATH)
        logger.info(f"Index loaded successfully. Total vectors: {index.ntotal}")
    except Exception as e:
        logger.error(f"Fatal: Failed to load FAISS index from {INDEX_PATH}: {e}", exc_info=True)
        raise RuntimeError(f"Could not load FAISS index from {INDEX_PATH}") from e

    try:
        logger.info(f"Loading index mapping list from {MAPPING_PATH}...")
        with open(MAPPING_PATH, 'r') as f:
            mapping_data_list = json.load(f)
            if not isinstance(mapping_data_list, list):
                raise TypeError(f"{MAPPING_PATH} does not contain a JSON list.")
        logger.info(f"Mapping list loaded successfully. Total entries: {len(mapping_data_list)}")

        # Check consistency between mapping list length and FAISS index size
        if len(mapping_data_list) != index.ntotal:
            logger.warning(f"Mismatch: FAISS index has {index.ntotal} vectors, but mapping list has {len(mapping_data_list)} entries. Assuming list order corresponds to index order, but this could cause issues.")

        # Process the list into a dictionary mapping FAISS index to entry
        index_to_entry = {}
        for i, entry in enumerate(mapping_data_list):
            if not isinstance(entry, dict):
                logger.warning(f"Skipping non-dictionary item at index {i} in mapping list: {entry}")
                continue
            
            # Store the original index as 'faiss_index' for potential future use/consistency
            entry['faiss_index'] = i 
            
            # Convert date string if present
            if 'entry_date' in entry and isinstance(entry['entry_date'], str):
                try:
                    entry['entry_date'] = date.fromisoformat(entry['entry_date'])
                except ValueError:
                    page_id_for_log = entry.get('page_id', f'index {i}')
                    logger.warning(f"Could not parse date string: {entry['entry_date']} for entry {page_id_for_log}")
                    entry['entry_date'] = None
            
            index_to_entry[i] = entry # Map list index i to the entry dict

        logger.info(f"Processed mapping list into index_to_entry lookup. Size: {len(index_to_entry)}")
            
    except FileNotFoundError:
        logger.error(f"Fatal: Mapping file not found at {MAPPING_PATH}")
        raise RuntimeError(f"Mapping file not found at {MAPPING_PATH}")
    except (json.JSONDecodeError, TypeError) as e:
        logger.error(f"Fatal: Failed to decode or process JSON list from {MAPPING_PATH}: {e}", exc_info=True)
        raise RuntimeError(f"Failed to decode or process {MAPPING_PATH}") from e
    except Exception as e:
        logger.error(f"Fatal: An unexpected error occurred loading/processing mapping from {MAPPING_PATH}: {e}", exc_info=True)
        raise RuntimeError(f"Unexpected error loading/processing {MAPPING_PATH}") from e

    # --- Load distinct metadata and schema (Corrected Path & Optional Schema) ---
    try:
        logger.info(f"Loading distinct metadata values from {METADATA_CACHE_PATH}...")
        if os.path.exists(METADATA_CACHE_PATH):
            with open(METADATA_CACHE_PATH, 'r') as f:
                distinct_metadata_values = json.load(f)
                for key in distinct_metadata_values:
                    if isinstance(distinct_metadata_values[key], list):
                         distinct_metadata_values[key] = set(distinct_metadata_values[key])
            logger.info("Distinct metadata values cache loaded.")
        else:
            logger.warning(f"Metadata cache file not found at {METADATA_CACHE_PATH}. Proceeding without it.")
            distinct_metadata_values = None
    except Exception as e:
        logger.error(f"Error loading metadata cache from {METADATA_CACHE_PATH}: {e}. Proceeding without cache.", exc_info=True)
        distinct_metadata_values = None

    try:
        logger.info(f"Loading database schema from {DATABASE_SCHEMA_PATH}...")
        if os.path.exists(DATABASE_SCHEMA_PATH):
            with open(DATABASE_SCHEMA_PATH, 'r') as f:
                schema_properties = json.load(f) 
                if not isinstance(schema_properties, dict):
                     logger.error(f"Fatal: Content of {DATABASE_SCHEMA_PATH} is not a JSON dictionary.")
                     raise TypeError(f"Schema file {DATABASE_SCHEMA_PATH} root is not a dictionary.")
            logger.info("Database schema loaded.")
        else:
            logger.warning(f"Database schema file not found at {DATABASE_SCHEMA_PATH}. Query analysis may be less accurate.")
            schema_properties = None
    except json.JSONDecodeError as e:
        logger.error(f"Fatal: Error decoding JSON from {DATABASE_SCHEMA_PATH}: {e}", exc_info=True)
        raise RuntimeError(f"Error decoding JSON from {DATABASE_SCHEMA_PATH}") from e
    except Exception as e:
        logger.error(f"Fatal: Error loading schema from {DATABASE_SCHEMA_PATH}: {e}", exc_info=True)
        raise RuntimeError(f"Error loading schema from {DATABASE_SCHEMA_PATH}") from e

def initialize_openai_client(api_key: str | None):
    """Initializes the OpenAI client using the provided API key."""
    global openai_client
    if not api_key:
        logger.error("Fatal: OpenAI API key was not provided during initialization.")
        raise ValueError("OpenAI API key not provided.")
    try:
        openai_client = OpenAI(api_key=api_key)
        logger.info("OpenAI client initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}", exc_info=True)
        raise

# --- Helper Functions ---
# (get_embedding remains the same)
async def get_embedding(text: str, retries: int = MAX_EMBEDDING_RETRIES) -> list[float] | None:
    if not openai_client: logger.error("OpenAI client not initialized."); return None
    attempt = 0
    while attempt < retries:
        try:
            response = await asyncio.to_thread(openai_client.embeddings.create, input=text, model=OPENAI_EMBEDDING_MODEL)
            if response.data and len(response.data) > 0 and response.data[0].embedding:
                logger.debug(f"Successfully obtained embedding for text snippet: '{text[:50]}...'")
                return response.data[0].embedding
            else: logger.error(f"Unexpected embedding response structure: {response}"); return None
        except RateLimitError as e:
            attempt += 1; logger.warning(f"Rate limit hit (attempt {attempt}/{retries}). Retrying in {EMBEDDING_RETRY_DELAY}s... Error: {e}")
            if attempt >= retries: logger.error(f"Max retries reached for embedding due to rate limiting."); return None
            await asyncio.sleep(EMBEDDING_RETRY_DELAY)
        except APIError as e:
            attempt += 1; logger.warning(f"API error (attempt {attempt}/{retries}). Retrying in {EMBEDDING_RETRY_DELAY}s... Error: {e}")
            if attempt >= retries: logger.error(f"Max retries reached for embedding due to API error."); return None
            await asyncio.sleep(EMBEDDING_RETRY_DELAY)
        except Exception as e: logger.error(f"Unexpected error getting embedding: {e}", exc_info=True); return None
    return None

async def analyze_query_for_filters(query: str) -> dict | None:
    """Analyzes query to extract structured filters."""
    if not openai_client: logger.error("OpenAI client not initialized."); return None
    if not schema_properties: logger.warning("Database schema not loaded. Query analysis will proceed without schema context."); current_schema = {}
    else: current_schema = schema_properties
    field_descriptions = []
    current_distinct_values = distinct_metadata_values or {}
    for name, details in current_schema.items():
        field_type = details.get('type', 'unknown'); desc = f"- {name} (type: {field_type})"
        if name in current_distinct_values:
            known_values = current_distinct_values[name]
            if known_values and isinstance(known_values, (list, set)):
                max_values_to_show = 50; values_list = list(known_values); values_str = ", ".join(values_list[:max_values_to_show])
                if len(values_list) > max_values_to_show: values_str += f", ... ({len(values_list) - max_values_to_show} more)"
                desc += f" | Known values: [{values_str}]"
            elif known_values: logger.warning(f"Expected list/set for distinct values of '{name}', got {type(known_values)}")
        field_descriptions.append(desc)
    schema_prompt_part = "\n".join(field_descriptions) if field_descriptions else "No schema information available."
    system_prompt = (
        "You are a query analysis assistant. Your task is to analyze the user query and the available Notion database fields "
        "(including known values for some fields, if provided) to extract structured filters. Identify potential entities like names, tags, dates, or date ranges mentioned in the query "
        "and map them to the most relevant field based on the provided schema AND the known values (if available). Format the output as a JSON object. "
        "Recognize date ranges (like 'last year', '2024', 'next month', 'June 2023'). For date ranges, output a 'date_range' key "
        "with 'start' and 'end' sub-keys in 'YYYY-MM-DD' format. For specific field value filters, output a 'filters' key containing "
        "a list of objects, where each object has 'field' (the Notion property name) and 'contains' (the value extracted from the query). "
        "**Important:** Names of people are typically found in the 'Family' (relation) or 'Friends' (relation) fields. Use the 'Known values' list provided for these fields to help map names accurately (if available). Map person names to THESE fields unless the query specifically asks about the entry's title (the 'Name' field). "
        "If a name could belong to either 'Family' or 'Friends' based on known values or context, include filters for BOTH fields (if Family/Friends fields exist in schema). "
        "Match keywords mentioned in the query to the 'Known values' for the 'Tags' field where appropriate (if Tags field exists in schema). "
        "If no specific filters are identified, return an empty JSON object {}."
    )
    user_prompt = f"""Available Notion Fields (with known values for some):
--- SCHEMA & VALUES START ---
{schema_prompt_part}
--- SCHEMA & VALUES END ---

User Query: "{query}"

Analyze the query based ONLY on the schema (if provided), known values (if provided), and query provided, following the mapping guidelines carefully. Output the structured filters as a JSON object:
"""
    logger.debug(f"Query Analysis System Prompt: {system_prompt}")
    logger.debug(f"Query Analysis User Prompt:\n{user_prompt}")
    try:
        response = await asyncio.to_thread(openai_client.chat.completions.create, model=QUERY_ANALYSIS_MODEL, messages=[
                {"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}
            ], temperature=0.1, response_format={ "type": "json_object" })
        analysis_result = response.choices[0].message.content
        logger.info("Received query analysis result from LLM."); logger.debug(f"Raw analysis result: {analysis_result}")
        filter_data = json.loads(analysis_result)
        if not isinstance(filter_data, dict): raise ValueError("Analysis result is not a dictionary.")
        if 'date_range' in filter_data and (not isinstance(filter_data['date_range'], dict) or 'start' not in filter_data['date_range'] or 'end' not in filter_data['date_range']):
             logger.warning(f"Invalid 'date_range' format: {filter_data.get('date_range')}"); filter_data.pop('date_range', None)
        if 'filters' in filter_data and not isinstance(filter_data['filters'], list):
             logger.warning(f"Invalid 'filters' format: {filter_data.get('filters')}"); filter_data.pop('filters', None)
        logger.info(f"Parsed filter data: {json.dumps(filter_data)}"); return filter_data
    except json.JSONDecodeError as e: logger.error(f"Failed to parse JSON from query analysis: {e}"); logger.error(f"Raw response: {analysis_result}"); return None
    except Exception as e: logger.error(f"Error during query analysis LLM call: {e}", exc_info=True); return None

# --- Main RAG Query Function ---

async def perform_rag_query(user_query: str) -> dict:
    """Performs the full RAG process assuming mapping data is a list."""
    # Use the globally loaded and processed data
    if not all([openai_client, index, index_to_entry]): # Check core components
         logger.error("RAG system core components (client, index, index->entry mapping) not fully initialized.")
         return {"answer": "Error: RAG system not initialized.", "sources": []}

    start_time = time.time()
    logger.info(f"Starting RAG process for query: '{user_query}'")

    # 1. Analyze Query for Filters
    logger.info("Analyzing query for potential filters...")
    filter_analysis = await analyze_query_for_filters(user_query)
    if filter_analysis is None: filter_analysis = {}; logger.warning("Query analysis failed. Proceeding without pre-filtering.")

    # 2. Pre-filter based on index_to_entry data
    target_faiss_indices_selector = None # Use IDSelectorBatch if filtering occurs
    # Start with all potential FAISS indices (0 to ntotal-1)
    candidate_indices = list(range(index.ntotal)) 

    if filter_analysis:
        logger.info(f"Applying filters based on analysis: {json.dumps(filter_analysis)}")
        date_range = filter_analysis.get('date_range')
        field_filters = filter_analysis.get('filters', [])

        # Apply date filter
        if date_range and date_range.get('start') and date_range.get('end'):
            try:
                start_date_filter = date.fromisoformat(date_range['start'])
                end_date_filter = date.fromisoformat(date_range['end'])
                logger.info(f"Applying date filter: {start_date_filter} to {end_date_filter}")
                new_candidate_indices = []
                for idx in candidate_indices:
                    # Use index_to_entry to get data for the current FAISS index
                    entry = index_to_entry.get(idx) 
                    entry_date = entry.get('entry_date') if entry else None
                    if entry_date and isinstance(entry_date, date):
                        if start_date_filter <= entry_date <= end_date_filter:
                            new_candidate_indices.append(idx)
                candidate_indices = new_candidate_indices
                logger.info(f"After date filter, {len(candidate_indices)} candidates remain.")
            except ValueError:
                logger.warning(f"Invalid date format in filter: {date_range}. Skipping date filter.")

        # Apply field filters (using AND logic between different fields, OR for Family/Friends)
        if field_filters:
            name_filters = [f for f in field_filters if f.get('field') in ['Family', 'Friends'] and f.get('contains')]
            other_filters = [f for f in field_filters if f.get('field') not in ['Family', 'Friends'] and f.get('field') and f.get('contains')]

            # Apply 'other' filters (AND logic)
            for f in other_filters:
                field = f['field']; contains = f['contains'].lower()
                logger.info(f"Applying filter: Field '{field}' contains '{contains}'")
                new_candidate_indices = []
                for idx in candidate_indices:
                    entry = index_to_entry.get(idx) # Get data via FAISS index
                    value = entry.get(field) if entry else None
                    match = False
                    if isinstance(value, str): match = contains in value.lower()
                    elif isinstance(value, (list, set)): match = any(contains in str(item).lower() for item in value)
                    elif value is not None: match = contains in str(value).lower()
                    if match: new_candidate_indices.append(idx)
                candidate_indices = new_candidate_indices
                logger.info(f"After filter '{field}', {len(candidate_indices)} candidates remain.")

            # Apply 'name' filters (OR logic)
            if name_filters:
                logger.info(f"Applying name filters (OR logic): {name_filters}")
                names_to_check = {f['contains'].lower() for f in name_filters}
                new_candidate_indices = []
                for idx in candidate_indices:
                    entry = index_to_entry.get(idx) # Get data via FAISS index
                    if not entry: continue
                    matches_name = False
                    for field in ['Family', 'Friends']:
                        value = entry.get(field)
                        if isinstance(value, (list, set)): matches_name = any(name in str(item).lower() for item in value for name in names_to_check)
                        elif isinstance(value, str): matches_name = any(name in value.lower() for name in names_to_check)
                        if matches_name: break
                    if matches_name: new_candidate_indices.append(idx)
                candidate_indices = new_candidate_indices
                logger.info(f"After name filters, {len(candidate_indices)} candidates remain.")

        # Final set of FAISS indices to target
        if not candidate_indices:
            logger.warning("Pre-filtering resulted in zero candidate indices.")
            # Return specific message if filters were applied
            if date_range or field_filters:
                 return {"answer": "Based on your filters (dates, names, tags), no matching entries could be found.", "sources": []}
            else: # Should not happen if filter_analysis was true, but indicates no results anyway
                target_faiss_indices_selector = None # Search all if no filters and somehow candidates are empty
        else:
            logger.info(f"Pre-filtering successful. Targeting {len(candidate_indices)} specific indices.")
            target_faiss_indices_selector = faiss.IDSelectorBatch(np.array(candidate_indices, dtype='int64'))
    else:
        logger.info("No filters extracted or applied. Searching entire index.")
        # target_faiss_indices_selector remains None

    # 3. Get Query Embedding
    logger.info("Getting embedding for the user query...")
    query_embedding = await get_embedding(user_query)
    if query_embedding is None: return {"answer": "Error: Could not process query embedding.", "sources": []}

    # 4. Perform FAISS Search
    k = TOP_K
    query_embedding_np = np.array([query_embedding]).astype('float32')
    retrieved_indices = []
    try:
        logger.info(f"Performing FAISS search (k={k})...")
        search_params = faiss.SearchParametersIVF(sel=target_faiss_indices_selector) if target_faiss_indices_selector else None
        distances, indices_result = index.search(query_embedding_np, k, params=search_params)
        
        if indices_result.size > 0:
             valid_indices = [idx for idx in indices_result[0] if idx != -1]
             if valid_indices:
                 retrieved_indices = valid_indices
                 logger.info(f"FAISS search completed. Retrieved {len(retrieved_indices)} potential indices.")
             else: logger.warning("FAISS search returned only invalid (-1) indices.")
        else: logger.warning("FAISS search returned no results.")
             
    except Exception as e:
        logger.error(f"Error during FAISS search: {e}", exc_info=True)
        return {"answer": "Error: Failed during similarity search.", "sources": []}

    # 5. Retrieve Context & Source Information using index_to_entry
    context_parts = []; sources = []; retrieved_faiss_indices = set()
    if not retrieved_indices: logger.warning("No relevant documents found after search.")

    logger.info("Retrieving content for top results...")
    for idx in retrieved_indices:
        if idx in retrieved_faiss_indices: continue # Skip duplicates from search if any
        entry_data = index_to_entry.get(idx)
        if entry_data:
            retrieved_faiss_indices.add(idx)
            content = entry_data.get('content', '')
            page_id = entry_data.get('page_id') 
            page_id_for_log = page_id if page_id else f'index {idx}'
            title = entry_data.get('title', f"Untitled Entry ({page_id_for_log})") 
            
            # >>> ADD LOGGING SIMILAR TO CLI <<<
            entry_date_value = entry_data.get('entry_date')
            entry_date_str_for_log = None
            if isinstance(entry_date_value, date):
                 entry_date_str_for_log = entry_date_value.strftime('%Y-%m-%d')
            elif isinstance(entry_date_value, str):
                 entry_date_str_for_log = entry_date_value
            # Find the distance if available (requires searching with distances)
            # Note: The current `index.search` call doesn't seem to retain distances alongside indices easily here.
            # We'll log without distance for now, which is the crucial part.
            logger.info(f"  - Retrieved Context: ID: {page_id_for_log}, Title: {title}, Date: {entry_date_str_for_log}")
            # >>> END LOGGING <<<
            
            entry_date_value = entry_data.get('entry_date') 
            entry_date_obj = None
            entry_date_str = None
            if isinstance(entry_date_value, date):
                entry_date_obj = entry_date_value
                entry_date_str = entry_date_obj.strftime('%Y-%m-%d')
            elif isinstance(entry_date_value, str):
                entry_date_str = entry_date_value
                try:
                    entry_date_obj = date.fromisoformat(entry_date_value) 
                except ValueError:
                    logger.warning(f"Context: Could not parse date string: {entry_date_value} for entry {page_id_for_log}")

            # Construct context string
            context_entry = f"""--- Entry Start ---\nTitle: {title}\n"""
            if page_id: context_entry += f"Page ID: {page_id}\n" 
            if entry_date_str: context_entry += f"Date: {entry_date_str}\n"
            tags = entry_data.get('Tags') 
            if tags and isinstance(tags, list): context_entry += f"Tags: {', '.join(tags)}\n"
            family = entry_data.get('Family')
            if family and isinstance(family, list): context_entry += f"Family: {', '.join(family)}\n"
            friends = entry_data.get('Friends')
            if friends and isinstance(friends, list): context_entry += f"Friends: {', '.join(friends)}\n"
            context_entry += f"""Content:
{content}
--- Entry End ---

"""
            context_parts.append(context_entry)
            
            # Construct URL from page_id if page_id exists
            constructed_url = None
            if page_id:
                page_id_no_hyphens = page_id.replace("-", "")
                constructed_url = f"https://www.notion.so/{page_id_no_hyphens}"
            
            # Append source info if URL exists
            if constructed_url:
                sources.append({"title": title, "url": constructed_url})
            else:
                # Optionally log if a source is skipped due to missing page_id/url
                logger.debug(f"Skipping source for entry '{title}' due to missing page_id/URL.")

        else:
            logger.warning(f"FAISS index {idx} found in search results but not in index_to_entry mapping.")

    context_string = "".join(context_parts)
    if not context_string: logger.warning("No context could be retrieved for the search results.")

    # 6. Generate Final Answer using LLM
    logger.info("Generating final answer using LLM...")
    # Align System and User prompts with CLI for formatting
    final_system_prompt = ( 
        "You are a helpful assistant answering questions based ONLY on the provided journal entries. "
        "Analyze the metadata (Title, Date, Tags, Family, Friends, etc.) and the Content of each entry carefully. "
        "**Assumption:** Assume that if a person's name appears in the 'Family' or 'Friends' metadata for an entry, the author saw or was with that person on that entry's date, even if the content doesn't explicitly state it. Use this assumption when answering questions about seeing people or frequency of contact. "
        "Do not make assumptions or use external knowledge beyond this specific instruction. "
        "**Safety Guardrail:** Avoid generating responses that contain harmful, inappropriate, or overly sensitive personal information. Specifically, do not output details related to illicit substances or illegal activities, even if mentioned in the context. "
        "If the answer cannot be found in the provided entries even with the assumption, say so."
    )
    final_user_prompt = f"""Here are the relevant journal entries:
--- START CONTEXT ---
{context_string if context_string else "No context was retrieved."}
--- END CONTEXT ---

Based *only* on the context provided above, answer the following question comprehensively.

**Instructions for Formatting:**
- If the answer includes specific dates or references entries, please include:
  - The relevant date (e.g., `Date: YYYY-MM-DD`).
  - A markdown link to the most relevant original Notion journal entry formatted as `Journal Entry: [Title of Entry](Notion Link)`.
  - Construct the Notion Link using the 'Page ID:' metadata like this: `https://www.notion.so/<page_id_without_hyphens>` (e.g., if Page ID is 'a1b2c3d4-e5f6-7890-abcd-ef1234567890', the link is `https://www.notion.so/a1b2c3d4e5f67890abcdef1234567890`). Use the 'Title:' metadata for the link text.
  - Place these details clearly within or directly following the main answer text.

Question: {user_query}
Answer:"""
    logger.debug(f"Final Answer System Prompt: {final_system_prompt}")
    logger.debug(f"Final Answer User Prompt Start:\nContext Length: {len(context_string)}\nQuery: {user_query}")
    final_answer = "Error: Default answer generation failure."
    try:
        response = await asyncio.to_thread(openai_client.chat.completions.create, model=FINAL_ANSWER_MODEL, messages=[
                {"role": "system", "content": final_system_prompt}, {"role": "user", "content": final_user_prompt}
            ], temperature=0.5)
        final_answer = response.choices[0].message.content
        logger.info("Successfully generated final answer.")
    except Exception as e:
        logger.error(f"Error during final answer LLM call: {e}", exc_info=True)
        final_answer = "Error: Failed to generate the final answer due to an LLM issue."

    end_time = time.time()
    logger.info(f"RAG process completed in {end_time - start_time:.2f} seconds.")

    return {"answer": final_answer, "sources": sources} 