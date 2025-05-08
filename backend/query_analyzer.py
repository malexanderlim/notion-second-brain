"""Analyzes user queries using an LLM to extract structured metadata filters.

This module takes a natural language query and uses a configured LLM
(currently OpenAI models only) to identify potential date ranges and
filters based on fields defined in the Notion database schema and known distinct values.
It relies on shared components (client, schema, distinct values) from rag_initializer.
"""
import json
import logging
import asyncio
from datetime import date

# Import MODEL_CONFIG directly as it's static configuration
from backend.rag_config import MODEL_CONFIG 
# Import rag_initializer to access shared, stateful components like clients and data
import backend.rag_initializer as rag_initializer

# Removed commented out notes regarding imports/dependencies

logger = logging.getLogger(__name__)

async def analyze_query_for_filters(query: str, query_analysis_model_key: str, **kwargs) -> dict:
    """Analyzes the user query using an LLM to extract structured filters.
    Uses openai_client, schema_properties, distinct_metadata_values from rag_initializer module.
    Accepts MODEL_CONFIG directly or can fetch from rag_config if not passed.
    """
    # Retrieve necessary components from rag_initializer or kwargs
    # Clients are always from rag_initializer
    current_openai_client = rag_initializer.openai_client
    
    # Data can be passed via kwargs (primarily for testing) or fetched from rag_initializer
    current_schema_properties = kwargs.get('schema_properties', rag_initializer.schema_properties)
    current_distinct_metadata_values = kwargs.get('distinct_metadata_values', rag_initializer.distinct_metadata_values)
    
    # MODEL_CONFIG can be passed or imported directly (as it's static)
    # If passed via kwargs, it would override the imported one for this function call.
    current_model_config = kwargs.get('MODEL_CONFIG', MODEL_CONFIG)

    # Default response structure including token counts and error fields
    analysis_response = {
        "filters": {},
        "input_tokens": 0,
        "output_tokens": 0,
        "error": None
    }

    if not current_openai_client:
        logger.error("OpenAI client (from rag_initializer) not initialized for query analysis.")
        analysis_response["error"] = "OpenAI client not initialized."
        return analysis_response

    selected_analysis_model_config = current_model_config.get(query_analysis_model_key)
    if not selected_analysis_model_config:
        logger.error(f"Query analysis model configuration not found for key: {query_analysis_model_key}")
        analysis_response["error"] = f"Config for query analysis model {query_analysis_model_key} not found."
        return analysis_response
    
    if selected_analysis_model_config["provider"] != "openai":
        logger.error(f"Query analysis currently only supports OpenAI models. Requested: {query_analysis_model_key}")
        analysis_response["error"] = f"Query analysis only supports OpenAI models. Requested provider: {selected_analysis_model_config['provider']}."
        return analysis_response

    logger.info(f"Analyzing query with {query_analysis_model_key} (API ID: {selected_analysis_model_config['api_id']}) to extract potential metadata filters...")
    
    field_descriptions = []
    if current_schema_properties: # Check if schema_properties is available
        for name, details in current_schema_properties.items():
            if name == "Food": continue # Skip Food details for now
            field_type = details.get('type', 'unknown')
            desc = f"- {name} (type: {field_type})"
            if current_distinct_metadata_values and name in current_distinct_metadata_values:
                known_values = current_distinct_metadata_values.get(name, set()) # Ensure it's a set or empty set
                if known_values and isinstance(known_values, set): # Double check it's a set
                    max_values_to_show = 50
                    # Sort for consistent prompting, convert set to list first
                    sorted_known_values = sorted(list(known_values))
                    values_str = ", ".join(sorted_known_values[:max_values_to_show])
                    if len(sorted_known_values) > max_values_to_show:
                        values_str += f", ... ({len(sorted_known_values) - max_values_to_show} more)"
                    desc += f" | Known values: [{values_str}]"
            field_descriptions.append(desc)
    schema_prompt_part = "\n".join(field_descriptions) if field_descriptions else "No specific field schema available."

    current_date_str = date.today().isoformat()
    system_prompt = (
        f"Today\'s date is {current_date_str}. Use this as the reference for any relative date calculations (e.g., \'last month\', \'past 6 months\'). "
        "You are a query analysis assistant. Your task is to analyze the user query and the available Notion database fields "
        "(including known values for some fields) to extract structured filters. Identify potential entities like names, tags, dates, or date ranges mentioned in the query "
        "and map them to the most relevant field based on the provided schema AND the known values. Format the output as a JSON object. "
        "Recognize date ranges (like 'last year', '2024', 'next month', 'June 2023'). For date ranges, output a 'date_range' key "
        "with 'start' and 'end' sub-keys in 'YYYY-MM-DD' format. For specific field value filters, output a 'filters' key containing "
        "a list of objects, where each object has 'field' (the Notion property name) and 'contains' (the value extracted from the query). "
        "**Important:** Names of people are typically found in the 'Family' (relation) or 'Friends' (relation) fields. Use the 'Known values' list provided for these fields to help map names accurately. Map person names to THESE fields unless the query specifically asks about the entry\'s title (the 'Name' field). "
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
        # Use the client from rag_initializer, called within asyncio.to_thread
        response = await asyncio.to_thread(
            current_openai_client.chat.completions.create,
            model=selected_analysis_model_config['api_id'], 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            response_format={ "type": "json_object" }
        )
        analysis_result_content = response.choices[0].message.content
        logger.info("Received query analysis result from LLM.")
        logger.debug(f"Raw analysis result: {analysis_result_content}")
        
        if response.usage:
            analysis_response["input_tokens"] = response.usage.prompt_tokens
            analysis_response["output_tokens"] = response.usage.completion_tokens
            logger.info(f"OpenAI Query Analysis Usage ({selected_analysis_model_config['api_id']}): Input={response.usage.prompt_tokens}, Output={response.usage.completion_tokens}")

        filter_data = json.loads(analysis_result_content)
        if not isinstance(filter_data, dict):
            raise ValueError("Analysis result is not a dictionary.")
        # Further validation can be added here if needed

        analysis_response["filters"] = filter_data
        logger.info(f"Parsed filter data: {json.dumps(filter_data)}")
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from query analysis LLM response: {e}")
        logger.error(f"Raw response was: {analysis_result_content if 'analysis_result_content' in locals() else 'unavailable'}")
        analysis_response["error"] = "Failed to parse LLM response for query analysis."
    except Exception as e:
        logger.error(f"Error during query analysis LLM call: {e}", exc_info=True)
        analysis_response["error"] = f"Error during query analysis: {str(e)}"
        
    return analysis_response 