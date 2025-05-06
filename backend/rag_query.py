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
                cache_content = json.load(f) # Load the entire cache content
                if "distinct_values" in cache_content and isinstance(cache_content["distinct_values"], dict):
                    distinct_metadata_values = cache_content["distinct_values"] # Assign the nested dictionary
                    for key in distinct_metadata_values: # Now iterate over "Family", "Friends", "Tags"
                        if isinstance(distinct_metadata_values[key], list):
                             distinct_metadata_values[key] = set(distinct_metadata_values[key]) # Convert lists to sets
                    logger.info("Distinct metadata values (from 'distinct_values' key) processed and loaded.")
                else:
                    logger.warning(f"'{METADATA_CACHE_PATH}' does not contain a 'distinct_values' dictionary. Proceeding without specific distinct values.")
                    distinct_metadata_values = {} 
        else:
            logger.warning(f"Metadata cache file not found at {METADATA_CACHE_PATH}. Proceeding without it.")
            distinct_metadata_values = {} 
    except Exception as e:
        logger.error(f"Error loading metadata cache from {METADATA_CACHE_PATH}: {e}. Proceeding without cache.", exc_info=True)
        distinct_metadata_values = {} 

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
    
    current_date_str = date.today().isoformat()

    field_descriptions = []
    current_distinct_values = distinct_metadata_values or {}
    for name, details in current_schema.items():
        field_type = details.get('type', 'unknown'); desc = f"- {name} (type: {field_type})"
        if name in current_distinct_values:
            known_values = current_distinct_values[name]
            if name == "Tags":
                logger.info(f"[Query Analysis] Distinct values for 'Tags' field being used: {known_values}")
            if known_values and isinstance(known_values, (list, set)):
                max_values_to_show = 50; values_list = list(known_values); values_str = ", ".join(values_list[:max_values_to_show])
                if len(values_list) > max_values_to_show: values_str += f", ... ({len(values_list) - max_values_to_show} more)"
                desc += f" | Known values: [{values_str}]"
            elif known_values: logger.warning(f"Expected list/set for distinct values of '{name}', got {type(known_values)}")
        field_descriptions.append(desc)
    schema_prompt_part = "\n".join(field_descriptions) if field_descriptions else "No schema information available."
    
    system_prompt = (
        "You are a query analysis assistant. Your primary task is to extract structured filters from the user's query based on the provided Notion database schema and known values. "
        "Format the output as a JSON object. "
        f"Today's date is {current_date_str}. Use this as the reference for any relative date calculations (e.g., 'last month', 'past 6 months'). "
        
        "**Date Range Extraction (Highest Priority):** "
        "First, always check for date ranges. Phrases like 'last year', 'in 2024', 'next month', 'June 2023', or specific start/end dates should be converted to a 'date_range' key "
        "with 'start' and 'end' sub-keys in 'YYYY-MM-DD' format. For a single year like 'in 2024', this means start: '2024-01-01' and end: '2024-12-31'. "
        "If a date range is identified, include it in the output. "

        "**Field Value Filters (Names, Tags, etc.):** "
        "Next, identify other potential entities like names or tags. For these, output a 'filters' key containing "
        "a list of objects, where each object has 'field' (the Notion property name) and 'contains' (the value extracted from the query). "
        
        "**Specific Field Guidance:** "
        "1. **Names:** Names of people are typically found in the 'Family' or 'Friends' fields. Use the 'Known values' for these fields to map names accurately. If a name could belong to either, include filters for BOTH. Do not map names to the entry's title (the 'Name' field) unless the query specifically asks about titles. "
        "2. **Tags:** Match keywords to 'Known values' for the 'Tags' field. Additionally: "
        "   - If the query mentions preparing food (e.g., 'cooked', 'made food') AND a 'Cooking' tag exists in 'Known values' for 'Tags', create a filter: `{'field': 'Tags', 'contains': 'Cooking'}`. "
        "   - If the query mentions dining out (e.g., 'ate at restaurant') AND a 'Restaurant' tag exists in 'Known values' for 'Tags', create a filter: `{'field': 'Tags', 'contains': 'Restaurant'}`. "
        "   Only use these semantic tag mappings if the target tag (e.g., 'Cooking', 'Restaurant') is explicitly listed in the 'Known values'. Do not invent tags. "
        
        "If no specific filters (date range or field filters) are identified, return an empty JSON object {}. Prioritize extracting a date range if present in the query."
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
    if not all([openai_client, index, index_to_entry, mapping_data_list]):
         logger.error("RAG system core components (client, index, index->entry mapping, mapping_data_list) not fully initialized.")
         return {"answer": "Error: RAG system not initialized.", "sources": []}

    start_time = time.time()
    logger.info(f"Starting RAG process for query: '{user_query}'")

    filter_analysis = await analyze_query_for_filters(user_query)
    if filter_analysis is None:
        filter_analysis = {}
        logger.warning("Query analysis failed. Proceeding without pre-filtering.")

    # --- Initialize filter-related variables ---
    name_filter_fields = {'Family', 'Friends'} 
    active_name_filters = [] 
    name_filters_were_active = False 
    fallback_triggered_due_to_names = False 
    
    tag_filter_fields = {'Tags'}  
    validated_active_tag_filters = [] 
    llm_suggested_filters_for_tag_fields = [] 
    tag_filters_were_active = False 
    fallback_triggered_due_to_tags = False 
    llm_attempted_tag_filtering_but_all_were_invalid = False

    non_name_or_tag_field_filters = []
    raw_llm_filters = filter_analysis.get('filters', [])

    # --- Filter Validation and Grouping Logic ---
    if raw_llm_filters:
        logger.debug(f"Raw LLM filters: {json.dumps(raw_llm_filters)}")
        temp_potential_tag_filters = []
        for f_filter in raw_llm_filters:
            field_name = f_filter.get('field')
            if field_name in name_filter_fields:
                active_name_filters.append(f_filter) 
                name_filters_were_active = True 
            elif field_name in tag_filter_fields:
                llm_suggested_filters_for_tag_fields.append(f_filter) 
                tag_filters_were_active = True 
                temp_potential_tag_filters.append(f_filter)
            else:
                non_name_or_tag_field_filters.append(f_filter)
        
        if temp_potential_tag_filters: 
            any_valid_tag_found_for_actual_filtering = False
            for pt_filter in temp_potential_tag_filters:
                tag_field_name = pt_filter.get('field')
                tag_value = pt_filter.get('contains')
                valid_options_for_field = distinct_metadata_values.get(tag_field_name, set()) if distinct_metadata_values else set()
                if tag_value and isinstance(tag_value, str) and \
                   tag_value.lower() in {opt.lower() for opt in valid_options_for_field if isinstance(opt, str)}:
                    validated_active_tag_filters.append(pt_filter) 
                    any_valid_tag_found_for_actual_filtering = True
                    logger.debug(f"Validated tag filter: {pt_filter}")
                else:
                    logger.warning(f"LLM suggested invalid tag filter, discarding from active filtering: {pt_filter}. Valid options for '{tag_field_name}': {valid_options_for_field}")
            if not any_valid_tag_found_for_actual_filtering and tag_filters_were_active:
                llm_attempted_tag_filtering_but_all_were_invalid = True
    
    # --- Apply Filters for Metadata Count ---
    if not mapping_data_list:
        logger.error("mapping_data_list is not populated. Cannot perform filtering.")
        return {"answer": "Error: RAG system data not fully loaded.", "sources": []}

    # Start with all entries, then filter down for the metadata count
    current_candidates_for_metadata_count: list[dict] = list(mapping_data_list)
    logger.debug(f"Initial candidates for metadata count: {len(current_candidates_for_metadata_count)}")

    if filter_analysis: 
        date_range_filter = filter_analysis.get('date_range')
        if date_range_filter and date_range_filter.get('start') and date_range_filter.get('end'):
            try:
                start_date = date.fromisoformat(date_range_filter['start'])
                end_date = date.fromisoformat(date_range_filter['end'])
                logger.info(f"Applying date filter for metadata count: {start_date.isoformat()} to {end_date.isoformat()}")
                current_candidates_for_metadata_count = [
                    entry for entry in current_candidates_for_metadata_count
                    if entry.get('entry_date') and isinstance(entry['entry_date'], date) and
                       start_date <= entry['entry_date'] <= end_date
                ]
                logger.info(f"After date filter, {len(current_candidates_for_metadata_count)} candidates remain for metadata count.")
            except ValueError as e:
                logger.warning(f"Invalid date format for metadata count: {date_range_filter}. Error: {e}")
        
        if non_name_or_tag_field_filters:
            logger.info(f"Applying other (non-name, non-tag) filters for metadata count: {json.dumps(non_name_or_tag_field_filters)}")
            for f_filter in non_name_or_tag_field_filters:
                field_name, contains_value = f_filter.get('field'), f_filter.get('contains')
                if not field_name or not contains_value: continue
                current_candidates_for_metadata_count = [
                    e for e in current_candidates_for_metadata_count
                    if field_name in e and isinstance(e.get(field_name), list) and 
                       any(contains_value.lower() in str(i).lower() for i in e[field_name])
                ]
            logger.info(f"After other filters, {len(current_candidates_for_metadata_count)} for metadata count.")

        candidates_before_tag_filter = list(current_candidates_for_metadata_count)
        if validated_active_tag_filters:
            logger.info(f"Applying validated tag filters for metadata count: {json.dumps(validated_active_tag_filters)}")
            for f in validated_active_tag_filters:
                fn, cv = f.get('field'), f.get('contains')
                current_candidates_for_metadata_count = [
                    e for e in current_candidates_for_metadata_count
                    if fn in e and isinstance(e.get(fn), list) and any(cv.lower() in str(i).lower() for i in e[fn])
                ]
            if not current_candidates_for_metadata_count and validated_active_tag_filters:
                logger.warning("Validated tags led to 0 for metadata count. Triggering tag fallback.")
                current_candidates_for_metadata_count = candidates_before_tag_filter
                fallback_triggered_due_to_tags = True
            logger.info(f"After validated tags, {len(current_candidates_for_metadata_count)} for metadata count. Fallback: {fallback_triggered_due_to_tags}")
        elif llm_attempted_tag_filtering_but_all_were_invalid:
            logger.warning("LLM tags all invalid. Triggering tag fallback for metadata count.")
            fallback_triggered_due_to_tags = True
            # current_candidates_for_metadata_count remains as it was after date/other filters
            logger.info(f"Using {len(current_candidates_for_metadata_count)} for metadata count after invalid tag fallback.")
            
        candidates_before_name_filter = list(current_candidates_for_metadata_count)
        if active_name_filters:
            logger.info(f"Applying name filters for metadata count: {json.dumps(active_name_filters)}")
            temp_list = []
            for entry in current_candidates_for_metadata_count:
                if any(name_f.get('field') in entry and 
                       isinstance(entry.get(name_f.get('field')), list) and
                       any(str(name_f.get('contains')).lower() in str(item).lower() for item in entry[name_f.get('field')]) 
                       for name_f in active_name_filters):
                    temp_list.append(entry)
            current_candidates_for_metadata_count = temp_list
            if not current_candidates_for_metadata_count and name_filters_were_active:
                logger.warning("Name filters led to 0 for metadata count. Triggering name fallback.")
                current_candidates_for_metadata_count = candidates_before_name_filter
                fallback_triggered_due_to_names = True
            logger.info(f"After name filters, {len(current_candidates_for_metadata_count)} for metadata count. Fallback: {fallback_triggered_due_to_names}")

    metadata_based_interaction_count = len(current_candidates_for_metadata_count)
    logger.info(f"Final metadata-based candidate count: {metadata_based_interaction_count}")

    person_names_for_prompt = "the person(s) mentioned"
    if active_name_filters:
        names = list(set(f.get('contains') for f in active_name_filters if f.get('contains')))
        if names: person_names_for_prompt = " and ".join(names)
    
    is_how_many_times_person_query = "how many times" in user_query.lower() and active_name_filters

    # --- Introduce new flag and logic for quantitative tag queries ---
    is_how_many_times_tag_query = False
    tag_names_for_prompt = ""
    if "how many times" in user_query.lower() and validated_active_tag_filters and not is_how_many_times_person_query:
        is_how_many_times_tag_query = True
        tags = list(set(f.get('contains') for f in validated_active_tag_filters if f.get('contains')))
        if tags:
            tag_names_for_prompt = " and ".join(tags)
        # This log helps confirm the new query type detection
        logger.info(f"Quantitative tag query detected for tag(s): '{tag_names_for_prompt}'. Metadata count: {metadata_based_interaction_count}.")

    if metadata_based_interaction_count == 0:
        logger.warning("Metadata filtering resulted in zero candidates.")
        # Construct more specific "no results" messages based on fallbacks if needed
        answer = f"Based on your query filters, no matching entries were found for {person_names_for_prompt if active_name_filters else 'your criteria'}."
        if fallback_triggered_due_to_names:
            answer = f"No entries were found explicitly tagged with '{person_names_for_prompt}' for the given period, even after broadening the search."
        # Add similar specific message for tag fallback if desired
        return {"answer": answer, "sources": []}

    # --- Semantic Search for Exemplars (using TOP_K, e.g., 15) ---
    faiss_indices_for_exemplar_search = [e['faiss_index'] for e in current_candidates_for_metadata_count if 'faiss_index' in e]
    if not faiss_indices_for_exemplar_search:
        logger.warning("No FAISS indices available from metadata-filtered candidates for exemplar search.")
        if is_how_many_times_person_query:
            return {"answer": f"Based on your journal entries, there appear to be {metadata_based_interaction_count} interactions with {person_names_for_prompt}. However, no specific details could be retrieved for examples.", "sources": []}
        return {"answer": "Metadata matches found, but no content could be retrieved for examples.", "sources": []}

    unique_faiss_indices_for_exemplars = sorted(list(set(faiss_indices_for_exemplar_search)))
    exemplar_selector = faiss.IDSelectorBatch(np.array(unique_faiss_indices_for_exemplars, dtype=np.int64))
    
    query_embedding = await get_embedding(user_query)
    if query_embedding is None:
        logger.error("Failed to get embedding for user query.")
        # Return count if available for quant person query
        if is_how_many_times_person_query:
            return {"answer": f"I found {metadata_based_interaction_count} entries for {person_names_for_prompt}, but could not process the query fully to provide details.", "sources": []}
        return {"answer": "Error: Could not process query embedding.", "sources": []}
    
    query_embedding_np = np.array([query_embedding], dtype=np.float32)
    effective_k_exemplars = min(TOP_K, len(unique_faiss_indices_for_exemplars))
    logger.info(f"Performing FAISS search for {effective_k_exemplars} exemplars from {len(unique_faiss_indices_for_exemplars)} candidates...")

    exemplar_distances, exemplar_retrieved_indices_raw = (np.array([]), np.array([[]]))
    if effective_k_exemplars > 0:
        try:
            sp = faiss.SearchParameters(); sp.sel = exemplar_selector
            exemplar_distances, exemplar_retrieved_indices_raw = index.search(query_embedding_np, k=effective_k_exemplars, params=sp)
        except Exception as e:
            logger.error(f"FAISS search for exemplars failed: {e}", exc_info=True)
            if is_how_many_times_person_query:
                return {"answer": f"Based on your journal entries, there were {metadata_based_interaction_count} interactions with {person_names_for_prompt}. Error retrieving examples: {str(e)}", "sources": []}
            return {"answer": f"Error during semantic search for examples: {str(e)}", "sources": []}

    exemplar_context_for_llm = []
    exemplar_sources_for_response = []
    if exemplar_retrieved_indices_raw.size > 0 and exemplar_retrieved_indices_raw[0].size > 0 and exemplar_retrieved_indices_raw[0][0] != -1:
        for i, faiss_idx in enumerate(exemplar_retrieved_indices_raw[0]):
            entry_data = index_to_entry.get(int(faiss_idx))
            if entry_data:
                content = entry_data.get("content", ""); title = entry_data.get("title", "Untitled Entry")
                page_id = entry_data.get("page_id")
                url = entry_data.get("url", "")
                if not url and page_id: url = f"https://www.notion.so/{str(page_id).replace('-', '')}"
                
                entry_date_obj = entry_data.get("entry_date")
                entry_date_str = entry_date_obj.isoformat() if entry_date_obj else "Unknown date"
                
                exemplar_context_for_llm.append(f"Document (ID: {faiss_idx}, Title: {title}, Date: {entry_date_str}, Page ID: {page_id if page_id else 'N/A'}):\n{content}\n---")
                
                dist = float(exemplar_distances[0][i]) if exemplar_distances.ndim > 1 and i < exemplar_distances.shape[1] else 0.0
                exemplar_sources_for_response.append({"title": title, "url": url, "id": str(faiss_idx), "date": entry_date_str, "distance": dist})
            else: 
                logger.warning(f"Could not find entry data for exemplar FAISS index: {int(faiss_idx)}")
    else:
        logger.warning("FAISS search for exemplars returned no results or invalid indices.")

    if not exemplar_context_for_llm:
        if is_how_many_times_person_query:
            return {"answer": f"Based on your journal entries, you interacted with {person_names_for_prompt} {metadata_based_interaction_count} times. No specific examples could be detailed.", "sources": exemplar_sources_for_response}
        elif is_how_many_times_tag_query: # New branch
            return {"answer": f"Based on your journal entries, you engaged in activities related to {tag_names_for_prompt} {metadata_based_interaction_count} times. No specific examples could be detailed.", "sources": exemplar_sources_for_response}
        else:
             return {"answer": "Found metadata matches, but no specific content could be detailed for examples.", "sources": exemplar_sources_for_response}

    exemplar_context_str = "\\n\\n".join(exemplar_context_for_llm)
    
    # --- Define Prompt Templates ---
    base_system_message_template_general = (
        "You are an AI assistant functioning as a 'Second Brain.' Your purpose is to help the user recall and synthesize information from their journal entries. "
        "Respond in a natural, reflective, and narrative style. Organize your answers clearly, often using bolded thematic titles to highlight key memories or points, each followed by descriptive details. "
        "If the user's query asks a quantitative question (e.g., 'List Y favorite items'), attempt to directly answer this based on the provided document exemplars. "
        "For 'List X items...' queries, provide the number of items requested if possible from the exemplars. If you find fewer, list those found and state that. "
        "After addressing any quantitative aspect directly, you can then elaborate with narrative details as appropriate. "
        "Cite specific journal entries when you draw information from them, as per the detailed formatting instructions provided."
    )

    base_system_message_template_quant_person = (
        "You are an AI assistant functioning as a 'Second Brain.' "
        "The user asked how many times they interacted with {person_name_s}. Based on their journal entries (pre-filtered by metadata), this occurred **{interaction_count}** times during the specified period. " # The count is from metadata.
        "Your primary task is to **first state this count clearly**. "
        "Then, provide a natural, reflective, and narrative summary of *some* of these interactions, drawing examples from the {exemplar_count} most textually relevant document exemplars provided below. "
        "Organize your examples clearly, often using bolded thematic titles. "
        "Cite specific journal entries (from the exemplars) when you draw information from them, as per the detailed formatting instructions."
    )

    base_system_message_template_quant_tag = (
        "You are an AI assistant functioning as a 'Second Brain.' "
        "The user asked how many times they engaged in activities related to '{tag_name_s}'. Based on their journal entries (pre-filtered by metadata), this occurred **{interaction_count}** times during the specified period. " # The count is from metadata.
        "Your primary task is to **first state this count clearly**. "
        "Then, provide a natural, reflective, and narrative summary of *some* of these occurrences, drawing examples from the {exemplar_count} most textually relevant document exemplars provided below. "
        "Organize your examples clearly, often using bolded thematic titles. "
        "Cite specific journal entries (from the exemplars) when you draw information from them, as per the detailed formatting instructions."
    )

    # Shared formatting instructions, with dynamic parts for quant person queries
    formatting_instructions_template = (
        "\n\n**Detailed Instructions for Structuring Your Answer and Referencing Sources:**\n\n"
        "{quantitative_person_intro}" # Placeholder for specific count instruction
        "1.  **Overall Narrative Flow:** Your answer should read like a helpful, reflective summary, as if recalling memories. \n\n"
        "2.  **Introduction (If appropriate):** Begin with a brief introductory sentence or two that generally addresses the user's query before diving into specific examples or themes.\n\n"
        "3.  **Thematic Sections (Main Content/Examples):**\n"
        "    *   Identify key themes, events, or memories from the provided document exemplars relevant to the query.\n"
        "    *   For each distinct theme/event you choose to highlight, start with a **bolded title**.\n"
        "    *   Follow with a descriptive paragraph elaborating on that theme/event, drawing information from the exemplars.\n\n"
        "4.  **Citing Sources (from Exemplars):**\n"
        "    *   When you include specific details or direct recollections from an exemplar journal entry, you MUST cite that entry.\n"
        "    *   Incorporate the citation naturally: create a markdown hyperlink `[Title of Entry](URL)`. \n"
        "    *   The `URL` part of the hyperlink should be constructed using the 'Page ID' provided in each document's context. To do this: take the 'Page ID' value, remove any hyphens from it, and then append the result to `https://www.notion.so/`. (For example, if a document's context shows 'Page ID: abc-123-def', the URL you construct and use in the citation will be `https://www.notion.so/abc123def`). \n"
        "    *   IMMEDIATELY follow the `[Title of Entry](URL)` hyperlink with its corresponding date (also from the document's context, labeled 'Date') in parentheses, formatted nicely (e.g., `(January 7, 2025)`). Do NOT include the word \"Date:\" before this parenthetical date.\n\n"
        "5.  **Concluding Thought (If appropriate).**\n\n"
        "**Specific Guidance for Non-\"How Many Times Person\" Quantitative Queries:**\n"
        "{general_quantitative_guidance}" # Placeholder for list guidance
        "**Important Reminders:**\n"
        "- Maintain a conversational and engaging tone.\n"
        "- Ensure the narrative flows logically.\n"
        "- Do NOT use markdown code blocks for any part of this."
    )
    
    # --- Assemble Final Prompt ---
    current_base_system_message = ""
    specific_formatting_quant_intro = "" 
    specific_formatting_general_quant_guidance = (
        "*   If the user's primary question is quantitative (and not a 'how many times' query specifically handled by other instructions to state a count first), **you MUST provide the direct numerical answer or list first based on the exemplars.**\n"
        "*   **For \"List X items...\" queries:** Provide the number of items requested if possible from the exemplars. If you find fewer, list those and state that. \n"
        "    *   Example: \"Here are 3 hikes you mentioned enjoying: 1. [Hike A](URL) (Date), 2. [Hike B](URL) (Date), 3. [Hike C](URL) (Date).\".\n"
        "*   If the context (i.e., the number of provided exemplar documents) doesn't allow for the requested number of items for a list, clearly state that.\n"
    )

    if is_how_many_times_person_query:
        logger.info(f"Quantitative person query for '{person_names_for_prompt}'. Metadata count: {metadata_based_interaction_count}. Exemplars: {len(exemplar_context_for_llm)}.")
        current_base_system_message = base_system_message_template_quant_person.format(
            person_name_s=person_names_for_prompt, 
            interaction_count=metadata_based_interaction_count,
            exemplar_count=len(exemplar_context_for_llm)
        )
        specific_formatting_quant_intro = (
            f"**Stating the Count:** Begin your answer by clearly stating: 'Based on your journal entries, you interacted with {person_names_for_prompt} **{metadata_based_interaction_count}** times during the specified period.' "
            f"Then, provide a narrative using the {len(exemplar_context_for_llm)} exemplars.\n\n"
        )
        specific_formatting_general_quant_guidance = "" # Not needed if specific quant intro is used

    elif is_how_many_times_tag_query: # New branch for quantitative tag queries
        logger.info(f"Quantitative tag query for '{tag_names_for_prompt}'. Metadata count: {metadata_based_interaction_count}. Exemplars: {len(exemplar_context_for_llm)}.")
        current_base_system_message = base_system_message_template_quant_tag.format(
            tag_name_s=tag_names_for_prompt,
            interaction_count=metadata_based_interaction_count,
            exemplar_count=len(exemplar_context_for_llm)
        )
        specific_formatting_quant_intro = (
            f"**Stating the Count:** Begin your answer by clearly stating: 'Based on your journal entries, you engaged in activities related to {tag_names_for_prompt} **{metadata_based_interaction_count}** times during the specified period.' "
            f"Then, provide a narrative using the {len(exemplar_context_for_llm)} exemplars.\n\n"
        )
        specific_formatting_general_quant_guidance = "" # Not needed

    else: # General query
        logger.info(f"General query type. Exemplars: {len(exemplar_context_for_llm)}.")
        current_base_system_message = base_system_message_template_general
        # general_quantitative_guidance is already set as a default for this case

    current_formatting_instructions = formatting_instructions_template.format(
        quantitative_person_intro=specific_formatting_quant_intro,
        general_quantitative_guidance=specific_formatting_general_quant_guidance
    )
    
    final_system_prompt = current_base_system_message + current_formatting_instructions

    # Apply Fallback Modifications
    if fallback_triggered_due_to_names: 
        fallback_intro = f"Note: The initial metadata search for '{person_names_for_prompt}' yielded no direct tags, so the search was broadened. The count of {metadata_based_interaction_count} (if applicable) and the exemplars reflect this. Focus on content for '{person_names_for_prompt}'.\n\n"
        final_system_prompt = fallback_intro + final_system_prompt
    elif fallback_triggered_due_to_tags: 
        original_llm_tags_str = "the specified tag(s)/concept(s)" 
        if llm_suggested_filters_for_tag_fields: 
            tag_values = list(set(f.get('contains') for f in llm_suggested_filters_for_tag_fields if f.get('contains')))
            if tag_values: original_llm_tags_str = " and ".join(tag_values)
        fallback_intro = f"Note: The initial metadata search for tags '{original_llm_tags_str}' was broadened. The count of {metadata_based_interaction_count} (if applicable) and exemplars reflect this. Focus on concepts related to '{original_llm_tags_str}'.\n\n"
        final_system_prompt = fallback_intro + final_system_prompt

    user_message_for_final_llm = f"User Query: \"{user_query}\"\n\nRelevant Documents (exemplars for narrative):\n{exemplar_context_str}"
    
    logger.info("Sending request to final LLM for answer generation...")
    # logger.debug(f"Final LLM System Prompt:\n{final_system_prompt}")

    try:
        response = await asyncio.to_thread(
            openai_client.chat.completions.create,
            model=FINAL_ANSWER_MODEL,
            messages=[
                {"role": "system", "content": final_system_prompt},
                {"role": "user", "content": user_message_for_final_llm}
            ],
            temperature=0.5 
        )
        answer = response.choices[0].message.content
        logger.info("Received answer from final LLM.")
    except Exception as e:
        logger.error(f"Error during final LLM call: {e}", exc_info=True)
        # answer_on_llm_error = "Error generating final answer." # This variable is not used.
        if is_how_many_times_person_query:
            return {"answer": f"Based on your records, there are {metadata_based_interaction_count} relevant entries for {person_names_for_prompt}. However, an error occurred generating a detailed summary: {str(e)}", "sources": exemplar_sources_for_response}
        elif is_how_many_times_tag_query: # New branch
            return {"answer": f"Based on your records, there are {metadata_based_interaction_count} relevant entries for activities related to {tag_names_for_prompt}. However, an error occurred generating a detailed summary: {str(e)}", "sources": exemplar_sources_for_response}
        return {"answer": f"An error occurred: {str(e)}", "sources": exemplar_sources_for_response}

    end_time = time.time()
    logger.info(f"RAG process completed in {end_time - start_time:.2f} seconds.")
    logger.info(f"Final sources for response (exemplars): {json.dumps(exemplar_sources_for_response, indent=2)}")
    return {"answer": answer, "sources": exemplar_sources_for_response}