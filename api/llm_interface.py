import logging
import asyncio

# Import rag_initializer to access shared, stateful components like clients and Anthropic class
from . import rag_initializer

# Removed unused commented imports for specific API errors

# It's assumed Anthropic client and Anthropic (SDK main class) are passed or globally available
# similar to openai_client for now.

logger = logging.getLogger(__name__)

async def generate_final_answer(final_system_prompt: str,
                                user_message_for_final_llm: str,
                                selected_model_config: dict,
                                **kwargs) -> tuple[str | None, int, int, str | None]:
    """Generates the final answer using the specified LLM provider and model.
    Uses openai_client, anthropic_client, Anthropic class from rag_initializer module.

    Args:
        final_system_prompt: The system prompt for the LLM.
        user_message_for_final_llm: The user message for the LLM.
        selected_model_config: Configuration for the selected model (provider, api_id, etc.).

    Returns:
        A tuple containing:
            - answer: str | None (the LLM's response text or None on error)
            - input_tokens: int
            - output_tokens: int
            - error: str | None (error message if any)
    """
    # Retrieve necessary components
    # Clients and Anthropic class are always from rag_initializer
    current_openai_client = rag_initializer.openai_client
    current_anthropic_client = rag_initializer.anthropic_client
    CurrentAnthropic = rag_initializer.Anthropic # Get the Anthropic class itself

    answer_text = None
    input_tokens = 0
    output_tokens = 0
    error_message = None

    provider = selected_model_config["provider"]
    api_id = selected_model_config["api_id"]
    max_output_tokens = selected_model_config.get("max_output_tokens", 4096) # Default if not specified

    logger.info(f"Generating final answer with {provider} model: {api_id}")
    logger.debug(f"Final LLM System Prompt:\n{final_system_prompt}")
    logger.debug(f"Final LLM User Message:\n{user_message_for_final_llm}")

    try:
        if provider == "openai":
            if not current_openai_client:
                error_message = "OpenAI client (from rag_initializer) not initialized for final answer."
                logger.error(error_message)
                return answer_text, input_tokens, output_tokens, error_message
            
            # Use the client from rag_initializer
            completion = await asyncio.to_thread(
                current_openai_client.chat.completions.create,
                model=api_id,
                messages=[
                    {"role": "system", "content": final_system_prompt},
                    {"role": "user", "content": user_message_for_final_llm}
                ],
                temperature=0.7, # Or whatever is standard for this project
                max_tokens=max_output_tokens 
            )
            answer_text = completion.choices[0].message.content
            if completion.usage:
                input_tokens = completion.usage.prompt_tokens
                output_tokens = completion.usage.completion_tokens
                logger.info(f"OpenAI Final Answer Usage ({api_id}): Input={input_tokens}, Output={output_tokens}")

        elif provider == "anthropic":
            if not CurrentAnthropic: # Check if the Anthropic SDK class is available
                error_message = "Anthropic SDK not available (CurrentAnthropic from rag_initializer is None). Cannot use Anthropic models."
                logger.error(error_message)
                return answer_text, input_tokens, output_tokens, error_message
            if not current_anthropic_client:
                error_message = "Anthropic client (from rag_initializer) not initialized for final answer."
                logger.error(error_message)
                return answer_text, input_tokens, output_tokens, error_message

            accumulated_text = []
            try:
                # Use the client and class from rag_initializer, with streaming
                response_stream = await asyncio.to_thread(
                    current_anthropic_client.messages.create,
                    model=api_id,
                    system=final_system_prompt,
                    messages=[
                        {"role": "user", "content": user_message_for_final_llm}
                    ],
                    max_tokens=max_output_tokens,
                    stream=True
                )

                for event in response_stream:
                    if event.type == "message_start":
                        if event.message.usage:
                            input_tokens = event.message.usage.input_tokens
                            logger.info(f"Anthropic stream message_start: Input tokens={input_tokens}")
                    elif event.type == "content_block_delta":
                        if event.delta.type == "text_delta":
                            accumulated_text.append(event.delta.text)
                    elif event.type == "message_delta":
                        if event.usage:
                            output_tokens = event.usage.output_tokens # Anthropic SDK sums this up by final delta
                            logger.info(f"Anthropic stream message_delta: Output tokens={output_tokens}")
                    # We can also handle message_stop if needed for final cleanup or logging

                answer_text = "".join(accumulated_text)
                if input_tokens > 0 or output_tokens > 0:
                     logger.info(f"Anthropic Final Answer Usage ({api_id}): Input={input_tokens}, Output={output_tokens}")
                else:
                    logger.warning(f"Anthropic token usage not fully reported by stream for {api_id}. Check SDK behavior.")

            except Exception as e: # Catch exceptions during streaming or processing
                logger.error(f"Error during Anthropic streaming call to model {api_id}: {e}", exc_info=True)
                error_message = f"LLM API streaming call failed for Anthropic model {api_id}: {str(e)}"
                # Ensure partial data is not returned as a full answer
                answer_text = None 
                # input_tokens and output_tokens might be partially set, or 0
        else:
            error_message = f"Unsupported LLM provider: {provider}"
            logger.error(error_message)

    except Exception as e:
        logger.error(f"Error during final LLM call to {provider} model {api_id}: {e}", exc_info=True)
        error_message = f"LLM API call failed for {provider} model {api_id}: {str(e)}"

    return answer_text, input_tokens, output_tokens, error_message 