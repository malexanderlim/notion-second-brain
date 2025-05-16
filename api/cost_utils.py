"""Utility for calculating estimated LLM API costs based on token usage."""
import logging

# Import necessary config values if they are not passed directly
# from backend.rag_config import MODEL_CONFIG, OPENAI_EMBEDDING_MODEL_ID

logger = logging.getLogger(__name__)

def calculate_estimated_cost(query_analysis_model_key: str,
                               analysis_input_tokens: int,
                               analysis_output_tokens: int,
                               embedding_model_key: str,
                               embedding_input_tokens: int,
                               final_answer_model_config: dict, # Config of the model used for the final answer
                               final_answer_input_tokens: int,
                               final_answer_output_tokens: int,
                               MODEL_CONFIG: dict # Pass the global MODEL_CONFIG
                               ) -> float:
    """Calculates the estimated cost of a RAG query based on token usage.

    Args:
        query_analysis_model_key: The key for the query analysis model used.
        analysis_input_tokens: Input tokens for query analysis.
        analysis_output_tokens: Output tokens for query analysis.
        embedding_model_key: The key for the embedding model used.
        embedding_input_tokens: Input tokens for embedding.
        final_answer_model_config: The specific config dictionary for the final answer model.
        final_answer_input_tokens: Input tokens for final answer generation.
        final_answer_output_tokens: Output tokens for final answer generation.
        MODEL_CONFIG: The main dictionary containing all model configurations.

    Returns:
        The estimated cost in USD, rounded to 6 decimal places.
    """
    cost = 0.0
    
    # Cost for query analysis
    query_analysis_config_for_cost = MODEL_CONFIG.get(query_analysis_model_key)
    if query_analysis_config_for_cost:
        cost += (analysis_input_tokens * query_analysis_config_for_cost.get("cost_per_input_token", 0))
        cost += (analysis_output_tokens * query_analysis_config_for_cost.get("cost_per_output_token", 0))
        logger.debug(f"Cost component (Query Analysis - {query_analysis_model_key}): Tokens In={analysis_input_tokens}, Out={analysis_output_tokens}, Cost Added: ${(analysis_input_tokens * query_analysis_config_for_cost.get('cost_per_input_token', 0)) + (analysis_output_tokens * query_analysis_config_for_cost.get('cost_per_output_token', 0)):.6f}")
    else:
        logger.warning(f"Could not find cost config for query analysis model: {query_analysis_model_key}")

    # Cost for embedding
    embedding_config_for_cost = MODEL_CONFIG.get(embedding_model_key)
    if embedding_config_for_cost:
        cost += (embedding_input_tokens * embedding_config_for_cost.get("cost_per_input_token", 0))
        # Assuming embedding output cost is negligible or included in input cost
        logger.debug(f"Cost component (Embedding - {embedding_model_key}): Tokens In={embedding_input_tokens}, Cost Added: ${(embedding_input_tokens * embedding_config_for_cost.get('cost_per_input_token', 0)):.6f}")
    else:
        logger.warning(f"Could not find cost config for embedding model: {embedding_model_key}")

    # Cost for final answer generation
    if final_answer_model_config: # Check if config was successfully retrieved earlier
        cost += (final_answer_input_tokens * final_answer_model_config.get("cost_per_input_token", 0))
        cost += (final_answer_output_tokens * final_answer_model_config.get("cost_per_output_token", 0))
        logger.debug(f"Cost component (Final Answer - {final_answer_model_config.get('api_id', 'Unknown')}): Tokens In={final_answer_input_tokens}, Out={final_answer_output_tokens}, Cost Added: ${(final_answer_input_tokens * final_answer_model_config.get('cost_per_input_token', 0)) + (final_answer_output_tokens * final_answer_model_config.get('cost_per_output_token', 0)):.6f}")
    else:
        # This case should ideally not happen if final_answer_model_config is derived correctly before calling
        logger.error("Missing final answer model config for cost calculation.")

    final_cost = round(cost, 6)
    logger.info(f"Total estimated cost calculated: ${final_cost:.6f}")
    return final_cost 