import logging

logger = logging.getLogger(__name__)

# --- Prompt Templates ---
BASE_SYSTEM_MESSAGE_TEMPLATE_GENERAL = (
    "You are an AI assistant functioning as a 'Second Brain.' Your purpose is to help the user recall and synthesize information *strictly from their provided journal entries (exemplars)*. "
    "Respond in a natural, reflective, and narrative style. Organize your answers clearly, often using bolded thematic titles to highlight key memories or points, each followed by descriptive details. "
    "**Crucially, if the user's query includes specific names, places, or items that are NOT mentioned in any of the provided journal entry exemplars, you MUST explicitly state that no information was found for those specific elements within the provided documents.** Do not invent interactions, details, or discussions for entities not present in the exemplars. "
    "If the user's query asks a quantitative question (e.g., 'List Y favorite items'), attempt to directly answer this based *only* on the provided document exemplars. "
    "For 'List X items...' queries, provide the number of items requested if possible from the exemplars. If you find fewer, list those and state that. "
    "After addressing any quantitative aspect directly, you can then elaborate with narrative details as appropriate, always grounding your response in the exemplars. "
    "Cite specific journal entries when you draw information from them, as per the detailed formatting instructions provided."
)

BASE_SYSTEM_MESSAGE_TEMPLATE_QUANT_PERSON = (
    "You are an AI assistant functioning as a 'Second Brain.' "
    "The user asked how many times they interacted with {person_name_s}. Based on their journal entries (pre-filtered by metadata), this occurred **{interaction_count}** times during the specified period. " # The count is from metadata.
    "Your primary task is to **first state this count clearly**. "
    "Then, provide a natural, reflective, and narrative summary of *some* of these interactions, drawing examples from the {exemplar_count} most textually relevant document exemplars provided below. "
    "Organize your examples clearly, often using bolded thematic titles. "
    "Cite specific journal entries (from the exemplars) when you draw information from them, as per the detailed formatting instructions."
)

BASE_SYSTEM_MESSAGE_TEMPLATE_QUANT_TAG = (
    "You are an AI assistant functioning as a 'Second Brain.' "
    "The user asked how many times they engaged in activities related to '{tag_name_s}'. Based on their journal entries (pre-filtered by metadata), this occurred **{interaction_count}** times during the specified period. " # The count is from metadata.
    "Your primary task is to **first state this count clearly**. "
    "Then, provide a natural, reflective, and narrative summary of *some* of these occurrences, drawing examples from the {exemplar_count} most textually relevant document exemplars provided below. "
    "Organize your examples clearly, often using bolded thematic titles. "
    "Cite specific journal entries (from the exemplars) when you draw information from them, as per the detailed formatting instructions."
)

FORMATTING_INSTRUCTIONS_TEMPLATE = (
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
    "- **Factual Accuracy:** Base your entire response *only* on the information contained within the provided journal entry exemplars. If the query mentions entities (names, places, items) not found in these exemplars, explicitly state that information for those specific entities is not present in the documents. Do not invent details or assume connections.\n"
    "- Do NOT use markdown code blocks for any part of this."
)

def construct_final_prompts(user_query: str,
                              exemplar_context_str: str,
                              is_how_many_times_person_query: bool,
                              person_names_for_prompt: str,
                              metadata_based_interaction_count: int,
                              len_exemplar_context_for_llm: int,
                              is_how_many_times_tag_query: bool,
                              tag_names_for_prompt: str,
                              fallback_triggered_due_to_names: bool,
                              fallback_triggered_due_to_tags: bool,
                              llm_suggested_filters_for_tag_fields: list[dict]
                              ) -> tuple[str, str]:
    """
    Constructs the final system and user prompts for the LLM based on query type and context.
    """
    current_base_system_message = ""
    specific_formatting_quant_intro = ""
    specific_formatting_general_quant_guidance = (
        "*   If the user's primary question is quantitative (and not a 'how many times' query specifically handled by other instructions to state a count first), **you MUST provide the direct numerical answer or list first based on the exemplars.**\n"
        "*   **For \"List X items...\" queries:** Provide the number of items requested if possible from the exemplars. If you find fewer, list those and state that. \n"
        "    *   Example: \"Here are 3 hikes you mentioned enjoying: 1. [Hike A](URL) (Date), 2. [Hike B](URL) (Date), 3. [Hike C](URL) (Date).\n"
        "*   If the context (i.e., the number of provided exemplar documents) doesn't allow for the requested number of items for a list, clearly state that.\n"
    )

    if is_how_many_times_person_query:
        logger.info(f"Quantitative person query for '{person_names_for_prompt}'. Metadata count: {metadata_based_interaction_count}. Exemplars: {len_exemplar_context_for_llm}.")
        current_base_system_message = BASE_SYSTEM_MESSAGE_TEMPLATE_QUANT_PERSON.format(
            person_name_s=person_names_for_prompt,
            interaction_count=metadata_based_interaction_count,
            exemplar_count=len_exemplar_context_for_llm
        )
        specific_formatting_quant_intro = (
            f"**Stating the Count:** Begin your answer by clearly stating: 'Based on your journal entries, you interacted with {person_names_for_prompt} **{metadata_based_interaction_count}** times during the specified period.' "
            f"Then, provide a narrative using the {len_exemplar_context_for_llm} exemplars.\n\n"
        )
        specific_formatting_general_quant_guidance = ""  # Not needed if specific quant intro is used

    elif is_how_many_times_tag_query:
        logger.info(f"Quantitative tag query for '{tag_names_for_prompt}'. Metadata count: {metadata_based_interaction_count}. Exemplars: {len_exemplar_context_for_llm}.")
        current_base_system_message = BASE_SYSTEM_MESSAGE_TEMPLATE_QUANT_TAG.format(
            tag_name_s=tag_names_for_prompt,
            interaction_count=metadata_based_interaction_count,
            exemplar_count=len_exemplar_context_for_llm
        )
        specific_formatting_quant_intro = (
            f"**Stating the Count:** Begin your answer by clearly stating: 'Based on your journal entries, you engaged in activities related to {tag_names_for_prompt} **{metadata_based_interaction_count}** times during the specified period.' "
            f"Then, provide a narrative using the {len_exemplar_context_for_llm} exemplars.\n\n"
        )
        specific_formatting_general_quant_guidance = ""  # Not needed

    else:  # General query
        logger.info(f"General query type. Exemplars: {len_exemplar_context_for_llm}.")
        current_base_system_message = BASE_SYSTEM_MESSAGE_TEMPLATE_GENERAL
        # general_quantitative_guidance is already set as a default for this case

    current_formatting_instructions = FORMATTING_INSTRUCTIONS_TEMPLATE.format(
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
    
    logger.info("Sending request to final LLM for answer generation...") # This log might be better placed in the calling module (LLM interface)
    # logger.debug(f"Final LLM System Prompt:\n{final_system_prompt}") # Keep this commented or use finer log level control

    return final_system_prompt, user_message_for_final_llm 