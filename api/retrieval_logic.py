"""Handles embedding generation, vector search, and metadata filtering logic.

Provides functions for:
- Generating text embeddings using OpenAI.
- Performing FAISS similarity searches.
- Applying structured metadata filters (date, tags, names) with fallback logic.
"""
import logging
import asyncio
import numpy as np
import json
from datetime import date, timedelta
from typing import Optional

# Imports for get_embedding - to be reviewed if they are all needed here
# or if some (like RateLimitError, APIError) are handled by a shared OpenAI client wrapper later.
# from openai import RateLimitError, APIError # Assuming these are exceptions from the openai package

# Import static configurations directly
from .rag_config import (
    MODEL_CONFIG, 
    MAX_EMBEDDING_RETRIES, 
    EMBEDDING_RETRY_DELAY,
    OPENAI_EMBEDDING_MODEL_ID # If get_embedding's default model_key is used from here
)
# Import rag_initializer to access shared, stateful components like clients
from . import rag_initializer

logger = logging.getLogger(__name__)

async def get_embedding(text: str, embedding_model_key: str, **kwargs) -> dict:
    """Generates an embedding for the given text using the specified model key.
    Returns a dictionary containing the embedding vector, input tokens, and any error.
    Uses openai_client from rag_initializer module.
    Accepts MODEL_CONFIG, MAX_EMBEDDING_RETRIES, EMBEDDING_RETRY_DELAY directly or via kwargs.
    """
    # Retrieve necessary components
    current_openai_client = rag_initializer.openai_client # Always from rag_initializer
    
    # Configs can be passed via kwargs (for testing) or use imported defaults
    current_model_config = kwargs.get('MODEL_CONFIG', MODEL_CONFIG)
    current_max_retries = kwargs.get('MAX_EMBEDDING_RETRIES', MAX_EMBEDDING_RETRIES)
    current_retry_delay = kwargs.get('EMBEDDING_RETRY_DELAY', EMBEDDING_RETRY_DELAY)

    embedding_response = {
        "embedding": None,
        "input_tokens": 0,
        "error": None
    }

    selected_embedding_config = current_model_config.get(embedding_model_key)
    if not selected_embedding_config:
        logger.error(f"Embedding model configuration not found for key: {embedding_model_key}")
        embedding_response["error"] = f"Config for embedding model {embedding_model_key} not found."
        return embedding_response

    if selected_embedding_config["provider"] != "openai":
        logger.error(f"Embedding generation currently only supports OpenAI models. Requested: {embedding_model_key}")
        embedding_response["error"] = "Embeddings only support OpenAI currently."
        return embedding_response

    if not current_openai_client:
        logger.error("OpenAI client (from rag_initializer) not initialized for embeddings.")
        embedding_response["error"] = "OpenAI client not initialized for embeddings."
        return embedding_response

    attempt = 0
    while attempt < current_max_retries:
        try:
            # Use the client from rag_initializer
            api_response = await asyncio.to_thread(
                current_openai_client.embeddings.create, 
                input=text, 
                model=selected_embedding_config['api_id']
            )
            if api_response.data and len(api_response.data) > 0 and api_response.data[0].embedding:
                embedding_response["embedding"] = api_response.data[0].embedding
                if api_response.usage:
                    embedding_response["input_tokens"] = api_response.usage.prompt_tokens
                    logger.info(f"OpenAI Embedding Usage ({selected_embedding_config['api_id']}): Input={api_response.usage.prompt_tokens}")
                logger.debug(f"Successfully obtained embedding for text snippet: '{text[:50]}...'")
                return embedding_response
            else: 
                logger.error(f"Unexpected embedding response structure: {api_response}")
                embedding_response["error"] = "Unexpected embedding response structure."
                return embedding_response
        except Exception as e: # Broader catch for OpenAI specific errors like RateLimitError, APIError
            attempt += 1
            logger.warning(f"Error during embedding (attempt {attempt}/{current_max_retries}), model {selected_embedding_config['api_id']}. Retrying in {current_retry_delay}s... Error: {e}")
            if attempt >= current_max_retries: 
                embedding_response["error"] = f"Max retries reached for embedding ({selected_embedding_config['api_id']}). Last error: {e}"
                logger.error(embedding_response["error"])
                return embedding_response
            await asyncio.sleep(current_retry_delay)
    
    if not embedding_response["error"]:
        embedding_response["error"] = f"Failed to get embedding for model {selected_embedding_config['api_id']} after all retries."
    return embedding_response

async def perform_pinecone_search(query_embedding: list[float],
                                  top_k: int,
                                  filter_payload: Optional[dict] = None # For Pinecone metadata filters
                                 ) -> tuple[list[str], dict[str, float]]:
    """Performs a semantic search using Pinecone.

    Args:
        query_embedding: The query embedding vector (list of floats).
        top_k: The number of top results to retrieve.
        filter_payload: Optional dictionary for Pinecone metadata filtering.
                         Example: {"genre": "drama", "year": {"$gte": 2020}}
                         Or for specific IDs: {"vector_id_field_name": {"$in": ["id1", "id2"]}}
                         (The exact field name for ID filtering depends on how metadata is indexed in Pinecone)

    Returns:
        A tuple containing:
            - A list of retrieved document string IDs.
            - A dictionary mapping document string IDs to their similarity scores.
        Returns empty list and dict if search fails or no results.
    """
    if not rag_initializer.pinecone_index_instance:
        logger.error("Pinecone index instance not initialized. Cannot perform search.")
        return [], {}
    
    if top_k == 0:
        logger.info("Top_k for Pinecone search is 0, returning empty results.")
        return [], {}

    logger.info(f"Performing Pinecone search for top_k={top_k} results...")
    if filter_payload:
        logger.info(f"Using filter payload: {json.dumps(filter_payload)}")

    try:
        query_response = await asyncio.to_thread(
            rag_initializer.pinecone_index_instance.query,
            vector=query_embedding,
            top_k=top_k,
            filter=filter_payload, # Pass the filter payload here
            include_values=False,  # We typically don't need the vector values back
            include_metadata=False # For MVP, we fetch metadata from index_mapping.json later
                                  # If metadata were stored in Pinecone and needed, set to True
        )
        
        retrieved_ids: list[str] = []
        scores: dict[str, float] = {}

        if query_response and query_response.matches:
            for match in query_response.matches:
                retrieved_ids.append(match.id)
                scores[match.id] = match.score
            logger.info(f"Pinecone search completed. Retrieved {len(retrieved_ids)} IDs.")
        else:
            logger.info("Pinecone search returned no matches or an empty response.")
            
        return retrieved_ids, scores

    except Exception as e:
        logger.error(f"Pinecone search failed: {e}", exc_info=True)
        return [], {}

# FAISS search logic will be added here later
async def perform_faiss_search(query_embedding_np: np.ndarray,
                               effective_k: int,
                               faiss_index: faiss.Index,
                               selector: faiss.IDSelector | None = None
                               ) -> tuple[np.ndarray, np.ndarray]:
    """Performs a FAISS search with an optional selector.

    Args:
        query_embedding_np: The query embedding as a NumPy array.
        effective_k: The number of neighbors to search for.
        faiss_index: The loaded FAISS index object.
        selector: An optional FAISS IDSelector to restrict the search.

    Returns:
        A tuple containing distances and retrieved indices (both NumPy arrays).
        Returns empty arrays if effective_k is 0 or search fails.
    """
    if effective_k == 0:
        logger.info("Effective K for FAISS search is 0, returning empty results.")
        return np.array([]), np.array([[]])

    logger.info(f"Performing FAISS search for {effective_k} exemplars...")
    try:
        search_params = None
        if selector:
            search_params = faiss.SearchParameters(); 
            search_params.sel = selector
            logger.debug(f"FAISS search will use IDSelectorBatch with {selector.nbits if hasattr(selector, 'nbits') else 'N/A'} bits / {selector.nt if hasattr(selector, 'nt') else 'N/A'} IDs.")
        
        # FAISS search is CPU-bound, can be run in a thread for async compatibility if needed,
        # but usually fast enough not to block significantly for typical k.
        # For now, direct call assuming it's acceptable.
        distances, retrieved_indices = faiss_index.search(query_embedding_np, k=effective_k, params=search_params)
        logger.info(f"FAISS search completed. Retrieved {retrieved_indices.shape[1] if retrieved_indices.size > 0 else 0} indices.")
        return distances, retrieved_indices
    except Exception as e:
        logger.error(f"FAISS search failed: {e}", exc_info=True)
        # Return empty arrays consistent with no results, error logged.
        return np.array([]), np.array([[]]) 

def apply_metadata_filters(mapping_data_list: list[dict],
                             filter_analysis: dict,
                             distinct_metadata_values: dict | None,
                             user_query: str # Needed for is_how_many_times queries
                             ) -> dict:
    """Applies metadata filters to the mapping data list and determines query type flags.

    Args:
        mapping_data_list: The full list of data entries.
        filter_analysis: The output from the query_analyzer module.
        distinct_metadata_values: Dictionary of distinct metadata values for validation.
        user_query: The original user query string.

    Returns:
        A dictionary containing:
            - current_candidates_for_metadata_count: list[dict]
            - metadata_based_interaction_count: int
            - active_name_filters: list[dict]
            - name_filters_were_active: bool
            - fallback_triggered_due_to_names: bool
            - validated_active_tag_filters: list[dict]
            - llm_suggested_filters_for_tag_fields: list[dict] # To retain what LLM suggested
            - tag_filters_were_active: bool
            - fallback_triggered_due_to_tags: bool
            - llm_attempted_tag_filtering_but_all_were_invalid: bool
            - person_names_for_prompt: str
            - tag_names_for_prompt: str
            - is_how_many_times_person_query: bool
            - is_how_many_times_tag_query: bool
    """
    if not mapping_data_list:
        logger.error("mapping_data_list is not populated. Cannot perform filtering.")
        # Return a default/error state for all expected keys
        return {
            "current_candidates_for_metadata_count": [],
            "metadata_based_interaction_count": 0,
            "active_name_filters": [], "name_filters_were_active": False, "fallback_triggered_due_to_names": False,
            "validated_active_tag_filters": [], "llm_suggested_filters_for_tag_fields": [], "tag_filters_were_active": False, 
            "fallback_triggered_due_to_tags": False, "llm_attempted_tag_filtering_but_all_were_invalid": False,
            "person_names_for_prompt": "", "tag_names_for_prompt": "",
            "is_how_many_times_person_query": False, "is_how_many_times_tag_query": False
        }

    # --- Initialize filter-related variables ---
    name_filter_fields = {'Family', 'Friends'}
    active_name_filters: list[dict] = []
    name_filters_were_active: bool = False
    fallback_triggered_due_to_names: bool = False
    
    tag_filter_fields = {'Tags'}  
    validated_active_tag_filters: list[dict] = []
    llm_suggested_filters_for_tag_fields: list[dict] = [] 
    tag_filters_were_active: bool = False
    fallback_triggered_due_to_tags: bool = False
    llm_attempted_tag_filtering_but_all_were_invalid: bool = False

    non_name_or_tag_field_filters: list[dict] = []
    raw_llm_filters = filter_analysis.get('filters', [])

    # --- Filter Validation and Grouping Logic ---
    if raw_llm_filters:
        logger.debug(f"Raw LLM filters from query analysis: {json.dumps(raw_llm_filters)}")
        temp_potential_tag_filters = []
        for f_filter in raw_llm_filters:
            field_name = f_filter.get('field')
            if field_name in name_filter_fields:
                active_name_filters.append(f_filter) 
                name_filters_were_active = True 
            elif field_name in tag_filter_fields:
                llm_suggested_filters_for_tag_fields.append(f_filter) # Store all LLM suggested tags
                tag_filters_were_active = True 
                temp_potential_tag_filters.append(f_filter)
            else:
                non_name_or_tag_field_filters.append(f_filter)
        
        if temp_potential_tag_filters: 
            any_valid_tag_found_for_actual_filtering = False
            # Ensure distinct_metadata_values is not None before using .get()
            safe_distinct_metadata_values = distinct_metadata_values or {}
            for pt_filter in temp_potential_tag_filters:
                tag_field_name = pt_filter.get('field')
                tag_value = pt_filter.get('contains')
                # Use safe_distinct_metadata_values
                valid_options_for_field = safe_distinct_metadata_values.get(tag_field_name, set())
                if tag_value and isinstance(tag_value, str) and \
                   tag_value.lower() in {str(opt).lower() for opt in valid_options_for_field if isinstance(opt, str)}:
                    validated_active_tag_filters.append(pt_filter) 
                    any_valid_tag_found_for_actual_filtering = True
                    logger.debug(f"Validated tag filter: {pt_filter}")
                else:
                    logger.warning(f"LLM suggested invalid tag filter, discarding from active filtering: {pt_filter}. Valid options for '{tag_field_name}': {valid_options_for_field}")
            if not any_valid_tag_found_for_actual_filtering and tag_filters_were_active:
                llm_attempted_tag_filtering_but_all_were_invalid = True
    
    # --- Apply Filters for Metadata Count ---
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
                    # Ensure entry_date is a date object before comparison
                    if entry.get('entry_date') and isinstance(entry.get('entry_date'), date) and
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
                       any(str(contains_value).lower() in str(i).lower() for i in e[field_name])
                ]
            logger.info(f"After other filters, {len(current_candidates_for_metadata_count)} for metadata count.")

        candidates_before_tag_filter = list(current_candidates_for_metadata_count)
        if validated_active_tag_filters:
            logger.info(f"Applying validated tag filters for metadata count: {json.dumps(validated_active_tag_filters)}")
            for f in validated_active_tag_filters:
                fn, cv = f.get('field'), f.get('contains')
                current_candidates_for_metadata_count = [
                    e for e in current_candidates_for_metadata_count
                    if fn in e and isinstance(e.get(fn), list) and any(str(cv).lower() in str(i).lower() for i in e[fn])
                ]
            if not current_candidates_for_metadata_count and validated_active_tag_filters: # Check implies tag_filters_were_active
                logger.warning("Validated tags led to 0 for metadata count. Triggering tag fallback.")
                current_candidates_for_metadata_count = candidates_before_tag_filter
                fallback_triggered_due_to_tags = True
            logger.info(f"After validated tags, {len(current_candidates_for_metadata_count)} for metadata count. Fallback: {fallback_triggered_due_to_tags}")
        elif llm_attempted_tag_filtering_but_all_were_invalid:
            logger.warning("LLM tags all invalid. Triggering tag fallback for metadata count (no filtering applied for tags).")
            # current_candidates_for_metadata_count remains as it was after date/other filters
            fallback_triggered_due_to_tags = True # Indicate that a fallback occurred due to invalid tags
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

    is_how_many_times_tag_query = False
    tag_names_for_prompt = ""
    if "how many times" in user_query.lower() and validated_active_tag_filters and not is_how_many_times_person_query:
        is_how_many_times_tag_query = True
        tags = list(set(f.get('contains') for f in validated_active_tag_filters if f.get('contains')))
        if tags:
            tag_names_for_prompt = " and ".join(tags)
        logger.info(f"Quantitative tag query detected for tag(s): '{tag_names_for_prompt}'. Metadata count: {metadata_based_interaction_count}.")

    return {
        "current_candidates_for_metadata_count": current_candidates_for_metadata_count,
        "metadata_based_interaction_count": metadata_based_interaction_count,
        "active_name_filters": active_name_filters,
        "name_filters_were_active": name_filters_were_active,
        "fallback_triggered_due_to_names": fallback_triggered_due_to_names,
        "validated_active_tag_filters": validated_active_tag_filters,
        "llm_suggested_filters_for_tag_fields": llm_suggested_filters_for_tag_fields,
        "tag_filters_were_active": tag_filters_were_active,
        "fallback_triggered_due_to_tags": fallback_triggered_due_to_tags,
        "llm_attempted_tag_filtering_but_all_were_invalid": llm_attempted_tag_filtering_but_all_were_invalid,
        "person_names_for_prompt": person_names_for_prompt,
        "tag_names_for_prompt": tag_names_for_prompt,
        "is_how_many_times_person_query": is_how_many_times_person_query,
        "is_how_many_times_tag_query": is_how_many_times_tag_query
    } 