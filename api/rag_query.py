"""Orchestrates the RAG (Retrieval-Augmented Generation) pipeline.

Coordinates calls to modules for query analysis, embedding, retrieval,
metadata filtering, prompt construction, LLM interaction, and cost calculation.
Provides the main `perform_rag_query` async function and a synchronous wrapper
`execute_rag_query_sync` for CLI usage.
Relies on shared components initialized by `rag_initializer`.
"""
import faiss
import json
import os
import logging
from datetime import date, datetime, timedelta
import time
import asyncio
import numpy as np
import sys

from openai import OpenAI, RateLimitError, APIError
try:
    from anthropic import Anthropic, APIStatusError, APIConnectionError
except ImportError:
    Anthropic = None
    APIStatusError = None
    APIConnectionError = None

from .rag_config import (
    INDEX_PATH,
    MAPPING_PATH,
    METADATA_CACHE_PATH,
    DATABASE_SCHEMA_PATH,
    OPENAI_EMBEDDING_MODEL_ID,
    DEFAULT_QUERY_ANALYSIS_MODEL_KEY,
    DEFAULT_FINAL_ANSWER_MODEL_KEY,
    MODEL_CONFIG,
    TOP_K,
    MAX_EMBEDDING_RETRIES,
    EMBEDDING_RETRY_DELAY
)

from .query_analyzer import analyze_query_for_filters
from .retrieval_logic import get_embedding, perform_faiss_search, apply_metadata_filters
from .prompt_constructor import construct_final_prompts
from .llm_interface import generate_final_answer
from .cost_utils import calculate_estimated_cost

# Import the rag_initializer module itself to access its globals dynamically
from . import rag_initializer
# Functions from rag_initializer used by execute_rag_query_sync or other entry points
from .rag_initializer import load_rag_data

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# All global RAG data and client variables are now accessed via rag_initializer.variable_name

# --- Main RAG Query Function ---

async def perform_rag_query(user_query: str, model_name: str | None = None) -> dict:
    """Performs the full RAG process.
    Assumes clients and RAG data are initialized and loaded via rag_initializer.
    Accesses shared RAG components via the rag_initializer module.
    """
    
    # --- Initialize response structure with defaults ---
    response_data = {
        "answer": "Error: RAG process did not complete.",
        "sources": [],
        "model_used": model_name if model_name else DEFAULT_FINAL_ANSWER_MODEL_KEY,
        "model_api_id_used": None,
        "model_provider_used": None,
        "input_tokens": 0, # Initialize to 0
        "output_tokens": 0, # Initialize to 0
        "estimated_cost_usd": 0.0
    }

    # --- Determine model for final answer ---
    final_answer_model_key = model_name if model_name and model_name in MODEL_CONFIG else DEFAULT_FINAL_ANSWER_MODEL_KEY
    selected_model_config = MODEL_CONFIG.get(final_answer_model_key)

    # This log will help confirm which model key is being resolved
    logger.info(f"API request for model_name: '{model_name}'. Resolved to final_answer_model_key: '{final_answer_model_key}'")

    if not selected_model_config:
        logger.error(f"Model configuration not found for key: {final_answer_model_key}. Cannot proceed.")
        response_data["answer"] = f"Error: Model configuration for '{final_answer_model_key}' not found."
        return response_data
        
    response_data["model_used"] = final_answer_model_key # The user-facing name
    response_data["model_api_id_used"] = selected_model_config["api_id"]
    response_data["model_provider_used"] = selected_model_config["provider"]

    # --- Check client initialization based on provider ---
    active_client = None
    if selected_model_config["provider"] == "openai":
        active_client = rag_initializer.openai_client # Access via module
    elif selected_model_config["provider"] == "anthropic":
        active_client = rag_initializer.anthropic_client # Access via module
    # Add other providers here

    if not active_client:
        error_msg = f"LLM client for provider '{selected_model_config['provider']}' (model: {final_answer_model_key}) is not initialized."
        logger.error(error_msg)
        response_data["answer"] = error_msg
        return response_data

    # Check if RAG data from rag_initializer is loaded
    if not all([rag_initializer.index, rag_initializer.index_to_entry, rag_initializer.mapping_data_list]): # Access via module
         logger.error("RAG system core components (index, index->entry mapping, mapping_data_list) not fully initialized. Ensure rag_initializer.load_rag_data() was called.")
         response_data["answer"] = "Error: RAG system not initialized. Core data missing."
         return response_data

    start_time = time.time()
    logger.info(f"Starting RAG process for query: '{user_query}' using model: {final_answer_model_key} (API ID: {selected_model_config['api_id']})")

    # Initialize token counters for this query
    total_input_tokens = 0
    total_output_tokens = 0

    # --- Query Analysis ---
    query_analysis_model_key_to_use = DEFAULT_QUERY_ANALYSIS_MODEL_KEY # Could be made configurable later
    # For query analysis, we might want to stick to a more powerful model or make it independently selectable.
    # If we used gpt-4o-mini for analysis, its capabilities for that specific task might be different.
    # The DEFAULT_QUERY_ANALYSIS_MODEL_KEY is still gpt-4o.
    
    analysis_input_tokens = 0
    analysis_output_tokens = 0
    filter_analysis = {}

    # Call analyze_query_for_filters with the chosen model key
    analysis_result = await analyze_query_for_filters(
        query=user_query, 
        query_analysis_model_key=query_analysis_model_key_to_use,
        openai_client=rag_initializer.openai_client, # Access via module
        schema_properties=rag_initializer.schema_properties, # Access via module
        distinct_metadata_values=rag_initializer.distinct_metadata_values, # Access via module
        MODEL_CONFIG=MODEL_CONFIG 
    )
    
    if analysis_result and not analysis_result.get("error"):
        filter_analysis = analysis_result.get("filters", {})
        analysis_input_tokens = analysis_result.get("input_tokens", 0)
        analysis_output_tokens = analysis_result.get("output_tokens", 0)
        total_input_tokens += analysis_input_tokens
        total_output_tokens += analysis_output_tokens
        logger.info(f"Query analysis successful. Tokens: In={analysis_input_tokens}, Out={analysis_output_tokens}")
    else:
        error_detail = analysis_result.get("error", "Unknown error") if analysis_result else "No response"
        logger.warning(f"Query analysis failed or was skipped. Error: {error_detail}. Proceeding without pre-filtering.")
        # Potentially set a specific error message in response_data if this is critical
        # For now, it proceeds with empty filter_analysis

    # --- Apply Metadata Filters ---
    metadata_filter_results = apply_metadata_filters(
        mapping_data_list=rag_initializer.mapping_data_list, # Access via module
        filter_analysis=filter_analysis,
        distinct_metadata_values=rag_initializer.distinct_metadata_values, # Access via module
        user_query=user_query
    )

    current_candidates_for_metadata_count = metadata_filter_results["current_candidates_for_metadata_count"]
    metadata_based_interaction_count = metadata_filter_results["metadata_based_interaction_count"]
    active_name_filters = metadata_filter_results["active_name_filters"]
    name_filters_were_active = metadata_filter_results["name_filters_were_active"]
    fallback_triggered_due_to_names = metadata_filter_results["fallback_triggered_due_to_names"]
    validated_active_tag_filters = metadata_filter_results["validated_active_tag_filters"]
    llm_suggested_filters_for_tag_fields = metadata_filter_results["llm_suggested_filters_for_tag_fields"]
    tag_filters_were_active = metadata_filter_results["tag_filters_were_active"]
    fallback_triggered_due_to_tags = metadata_filter_results["fallback_triggered_due_to_tags"]
    llm_attempted_tag_filtering_but_all_were_invalid = metadata_filter_results["llm_attempted_tag_filtering_but_all_were_invalid"]
    person_names_for_prompt = metadata_filter_results["person_names_for_prompt"]
    tag_names_for_prompt = metadata_filter_results["tag_names_for_prompt"]
    is_how_many_times_person_query = metadata_filter_results["is_how_many_times_person_query"]
    is_how_many_times_tag_query = metadata_filter_results["is_how_many_times_tag_query"]

    if metadata_based_interaction_count == 0:
        logger.warning("Metadata filtering resulted in zero candidates.")
        # Construct more specific "no results" messages based on fallbacks if needed
        answer = f"Based on your query filters, no matching entries were found for {person_names_for_prompt if active_name_filters else 'your criteria'}."
        if fallback_triggered_due_to_names:
            answer = f"No entries were found explicitly tagged with '{person_names_for_prompt}' for the given period, even after broadening the search."
        # Add similar specific message for tag fallback if desired
        response_data["answer"] = answer
        # response_data["sources"] is already []
        # TODO: Calculate cost even for failed queries based on analysis tokens if any
        return response_data

    # --- Semantic Search for Exemplars (using TOP_K, e.g., 15) ---
    faiss_indices_for_exemplar_search = [e['faiss_index'] for e in current_candidates_for_metadata_count if 'faiss_index' in e]
    if not faiss_indices_for_exemplar_search:
        logger.warning("No FAISS indices available from metadata-filtered candidates for exemplar search.")
        answer_text = "Metadata matches found, but no content could be retrieved for examples."
        if is_how_many_times_person_query:
            answer_text = f"Based on your journal entries, there appear to be {metadata_based_interaction_count} interactions with {person_names_for_prompt}. However, no specific details could be retrieved for examples."
        response_data["answer"] = answer_text
        return response_data

    unique_faiss_indices_for_exemplars = sorted(list(set(faiss_indices_for_exemplar_search)))
    exemplar_selector = faiss.IDSelectorBatch(np.array(unique_faiss_indices_for_exemplars, dtype=np.int64))
    
    # Embedding uses a fixed model for now
    # Consider if embedding model choice should also be configurable
    embedding_input_tokens = 0 # Placeholder for embedding token count
    
    embedding_result = await get_embedding(
        text=user_query, 
        embedding_model_key=OPENAI_EMBEDDING_MODEL_ID, 
        openai_client=rag_initializer.openai_client, # Access via module
        MODEL_CONFIG=MODEL_CONFIG,
        MAX_EMBEDDING_RETRIES=MAX_EMBEDDING_RETRIES,
        EMBEDDING_RETRY_DELAY=EMBEDDING_RETRY_DELAY
    )
    query_embedding_vector = None

    if embedding_result and not embedding_result.get("error"):
        query_embedding_vector = embedding_result.get("embedding")
        embedding_input_tokens = embedding_result.get("input_tokens", 0)
        total_input_tokens += embedding_input_tokens
        logger.info(f"Query embedding successful. Tokens: In={embedding_input_tokens}")
    else:
        error_detail = embedding_result.get("error", "Unknown error") if embedding_result else "No embedding response"
        logger.error(f"Failed to get embedding for user query. Error: {error_detail}")
        answer_text = "Error: Could not process query embedding."
        if is_how_many_times_person_query:
            answer_text = f"I found {metadata_based_interaction_count} entries for {person_names_for_prompt}, but could not process the query fully to provide details."
        response_data["answer"] = answer_text
        # Update cost with any tokens used so far (e.g., analysis tokens)
        current_cost = 0.0
        query_analysis_config_for_cost = MODEL_CONFIG.get(query_analysis_model_key_to_use)
        if query_analysis_config_for_cost:
            current_cost += (analysis_input_tokens * query_analysis_config_for_cost.get("cost_per_input_token", 0))
            current_cost += (analysis_output_tokens * query_analysis_config_for_cost.get("cost_per_output_token", 0))
        response_data["estimated_cost_usd"] = round(current_cost, 6)
        return response_data

    if not query_embedding_vector: # Double check after error logging if vector is still None
        # This case should ideally be caught by the error handling above, but as a safeguard:
        logger.error("Query embedding vector is None even after get_embedding call completed (unexpected state).")
        response_data["answer"] = "Error: Failed to obtain query embedding vector."
        # Recalculate cost as above
        current_cost = 0.0
        query_analysis_config_for_cost = MODEL_CONFIG.get(query_analysis_model_key_to_use)
        if query_analysis_config_for_cost:
            current_cost += (analysis_input_tokens * query_analysis_config_for_cost.get("cost_per_input_token", 0))
            current_cost += (analysis_output_tokens * query_analysis_config_for_cost.get("cost_per_output_token", 0))
        response_data["estimated_cost_usd"] = round(current_cost, 6)
        return response_data
    
    query_embedding_np = np.array([query_embedding_vector], dtype=np.float32)
    effective_k_exemplars = min(TOP_K, len(unique_faiss_indices_for_exemplars))
    logger.info(f"Performing FAISS search for {effective_k_exemplars} exemplars from {len(unique_faiss_indices_for_exemplars)} candidates...")

    exemplar_distances, exemplar_retrieved_indices_raw = (np.array([]), np.array([[]]))
    if effective_k_exemplars > 0:
        exemplar_distances, exemplar_retrieved_indices_raw = await perform_faiss_search(
            query_embedding_np=query_embedding_np,
            effective_k=effective_k_exemplars,
            faiss_index=rag_initializer.index, # Access via module
            selector=exemplar_selector
        )
        # Error handling for FAISS search is now within perform_faiss_search,
        # it returns empty arrays on failure, so we check the results.
        if exemplar_retrieved_indices_raw.size == 0:
            logger.error(f"FAISS search for exemplars returned no results or failed (handled in perform_faiss_search).")
            # Decide if specific error message is needed or if existing logic handles it
            if is_how_many_times_person_query:
                answer_text = f"Based on your journal entries, there were {metadata_based_interaction_count} interactions with {person_names_for_prompt}. However, an error occurred while trying to retrieve specific examples."
                response_data["answer"] = answer_text
                return response_data
            answer_text = "Error during semantic search for examples (search returned no valid indices)."
            response_data["answer"] = answer_text
            return response_data

    exemplar_context_for_llm = []
    exemplar_sources_for_response = []
    if exemplar_retrieved_indices_raw.size > 0 and exemplar_retrieved_indices_raw[0].size > 0 and exemplar_retrieved_indices_raw[0][0] != -1:
        for i, faiss_idx in enumerate(exemplar_retrieved_indices_raw[0]):
            entry_data = rag_initializer.index_to_entry.get(int(faiss_idx)) # Access via module
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
        answer_text = "Found metadata matches, but no specific content could be detailed for examples."
        if is_how_many_times_person_query:
            answer_text = f"Based on your journal entries, you interacted with {person_names_for_prompt} {metadata_based_interaction_count} times. No specific examples could be detailed."
        elif is_how_many_times_tag_query: # New branch
            answer_text = f"Based on your journal entries, you engaged in activities related to {tag_names_for_prompt} {metadata_based_interaction_count} times. No specific examples could be detailed."
        
        response_data["answer"] = answer_text
        response_data["sources"] = exemplar_sources_for_response # Sources might still be useful
        return response_data

    exemplar_context_str = "\n\n".join(exemplar_context_for_llm)
    
    final_system_prompt, user_message_for_final_llm = construct_final_prompts(
        user_query=user_query,
        exemplar_context_str=exemplar_context_str,
        is_how_many_times_person_query=is_how_many_times_person_query,
        person_names_for_prompt=person_names_for_prompt,
        metadata_based_interaction_count=metadata_based_interaction_count,
        len_exemplar_context_for_llm=len(exemplar_context_for_llm), # Pass length directly
        is_how_many_times_tag_query=is_how_many_times_tag_query,
        tag_names_for_prompt=tag_names_for_prompt,
        fallback_triggered_due_to_names=fallback_triggered_due_to_names,
        fallback_triggered_due_to_tags=fallback_triggered_due_to_tags,
        llm_suggested_filters_for_tag_fields=llm_suggested_filters_for_tag_fields
    )
    
    # logger.info("Sending request to final LLM for answer generation...") # This log is now in prompt_constructor
    # logger.debug(f"Final LLM System Prompt:\n{final_system_prompt}")

    # --- LLM Call for Final Answer (using llm_interface) ---
    final_answer_input_tokens = 0
    final_answer_output_tokens = 0
    llm_error_message = None
    
    generated_answer, llm_input_tokens, llm_output_tokens, llm_error = await generate_final_answer(
        final_system_prompt=final_system_prompt,
        user_message_for_final_llm=user_message_for_final_llm,
        selected_model_config=selected_model_config,
        openai_client=rag_initializer.openai_client, # Access via module
        anthropic_client=rag_initializer.anthropic_client, # Access via module
        Anthropic=rag_initializer.Anthropic # Access via module
    )
    
    final_answer_input_tokens = llm_input_tokens
    final_answer_output_tokens = llm_output_tokens
    
    total_input_tokens += final_answer_input_tokens
    total_output_tokens += final_answer_output_tokens
    logger.info(f"Total Tokens for Query: Input={total_input_tokens}, Output={total_output_tokens}")

    if llm_error:
        logger.error(f"Error during final LLM call: {llm_error}")
        error_answer = f"An error occurred: {llm_error}"
        # Construct specific error messages based on query type if desired
        if is_how_many_times_person_query:
            error_answer = f"Based on your records, there are {metadata_based_interaction_count} relevant entries for {person_names_for_prompt}. However, an error occurred generating a detailed summary: {llm_error}"
        elif is_how_many_times_tag_query: 
            error_answer = f"Based on your records, there are {metadata_based_interaction_count} relevant entries for activities related to {tag_names_for_prompt}. However, an error occurred generating a detailed summary: {llm_error}"
        response_data["answer"] = error_answer
        llm_error_message = llm_error # Store error message if needed elsewhere
    elif generated_answer is not None:
        response_data["answer"] = generated_answer
        logger.info("Received answer from final LLM via interface.")
    else:
        # Handle unexpected case where no error but answer is None
        unknown_error = "LLM interface returned no answer and no error."
        logger.error(unknown_error)
        response_data["answer"] = f"An unknown error occurred during final answer generation."
        llm_error_message = unknown_error

    response_data["input_tokens"] = total_input_tokens
    response_data["output_tokens"] = total_output_tokens

    # --- Cost Calculation --- (Now delegated to cost_utils)
    estimated_cost = calculate_estimated_cost(
        query_analysis_model_key=query_analysis_model_key_to_use,
        analysis_input_tokens=analysis_input_tokens,
        analysis_output_tokens=analysis_output_tokens,
        embedding_model_key=OPENAI_EMBEDDING_MODEL_ID, # Using the default embedding model key
        embedding_input_tokens=embedding_input_tokens,
        final_answer_model_config=selected_model_config, # Pass the config for the final answer model
        final_answer_input_tokens=final_answer_input_tokens,
        final_answer_output_tokens=final_answer_output_tokens,
        MODEL_CONFIG=MODEL_CONFIG # Pass the main MODEL_CONFIG dict
    )
    response_data["estimated_cost_usd"] = estimated_cost
    # logger.info(f"Estimated cost for the query: ${response_data['estimated_cost_usd']:.6f}") # Log moved inside calculate_estimated_cost

    end_time = time.time()
    logger.info(f"RAG process completed in {end_time - start_time:.2f} seconds.")
    logger.info(f"Final sources for response (exemplars): {json.dumps(exemplar_sources_for_response, indent=2)}")
    
    response_data["sources"] = exemplar_sources_for_response
    return response_data

# --- Synchronous Wrapper for CLI --- 
def execute_rag_query_sync(args):
    """Synchronous wrapper to execute the RAG query process for the CLI.
    Ensures RAG data and clients are initialized via rag_initializer functions.
    """
    logger.info("Executing RAG query via synchronous wrapper for CLI...")
    
    user_query = args.query
    model_name = getattr(args, 'model', None) 
    
    # Ensure data is loaded by calling the function from rag_initializer.
    # load_rag_data is imported directly for this call.
    # rag_initializer.load_rag_data() handles the _rag_data_loaded flag.
    try:
        load_rag_data() # This is the directly imported function
    except Exception as e:
        logger.error(f"Failed to load RAG data in sync wrapper via rag_initializer: {e}", exc_info=True)
        print(f"Error: Failed to load necessary RAG data files. {e}")
        sys.exit(1) # Exit if core data cannot be loaded

    # Client initialization should happen at app start (e.g., in cli.py or backend/main.py).
    # perform_rag_query will check if clients (accessed via rag_initializer.openai_client etc.) are set.
    
    try:
        logger.info(f"Running perform_rag_query for query: \"{user_query}\" with model: {model_name if model_name else 'Default'}")
        response_data = asyncio.run(perform_rag_query(user_query, model_name))
        
        print("\nAnswer:")
        print(response_data.get("answer", "No answer returned."))
        
        # Optionally print sources and cost for CLI users as well
        sources = response_data.get("sources", [])
        if sources:
            print("\nSources Used:")
            for source in sources:
                # Format similar to backend, maybe simpler for CLI
                title = source.get('title', 'Unknown Title')
                date_str = source.get('date', 'Unknown Date')
                url = source.get('url', '')
                print(f"- {title} ({date_str}) {f'[Link: {url}]' if url else ''}")

        cost = response_data.get("estimated_cost_usd", 0.0)
        tokens_in = response_data.get("input_tokens", 0)
        tokens_out = response_data.get("output_tokens", 0)
        model_actually_used = response_data.get("model_used", "Unknown")
        provider = response_data.get("model_provider_used", "Unknown")
        
        print("\n--- Query Stats ---")
        print(f"Model Used: {model_actually_used} ({provider})" ) 
        print(f"Tokens: {tokens_in} (prompt) / {tokens_out} (completion)")
        print(f"Estimated Cost: ${cost:.6f} USD")

    except Exception as e:
        logger.error(f"Error running asyncio RAG query: {e}", exc_info=True)
        print(f"\nError occurred during query processing: {e}")
        # Don't sys.exit here, let the main cli loop handle exit codes if necessary