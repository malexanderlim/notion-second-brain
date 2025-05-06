# Technical Design Document: Model Selection & Cost Estimation

## 1. Overview

This document outlines the design for implementing a feature that allows users to select the language model for their queries and understand the associated capabilities and estimated costs. Initially, this will support various OpenAI models, with the architecture allowing for future expansion to other providers like Anthropic.

## 2. Goals

*   Provide users with the ability to choose from a selection of available Large Language Models (LLMs) for their queries.
*   Offer users transparency into the relative capabilities (e.g., context window, strengths) of different models, where feasible.
*   Give users an estimate of the monetary cost associated with a query, based on the selected model and token usage.
*   Enable future expansion to include models from other providers (e.g., Anthropic).
*   Ensure the backend can dynamically use the selected model.

## 3. Non-Goals

*   Implementing a full-fledged, real-time cost tracking and billing system. The cost displayed will be an *estimate*.
*   Providing exhaustive details or comparisons for every model parameter; focus will be on key differentiators relevant to user choice (e.g., cost, general capability).
*   Dynamic fetching of model pricing information in real-time for the MVP. Pricing data will be initially hardcoded or managed through a simple configuration.
*   User-specific API key input through the UI for the MVP. The backend will use globally configured API keys.

## 4. Proposed Solution

### 4.1. Backend Changes

#### 4.1.1. API Endpoint Modification (`/api/query`)
The existing `/api/query` POST endpoint will be modified to accept a new optional parameter in its JSON payload:
`model_name` (string): Specifies the desired model (e.g., "gpt-4o", "gpt-3.5-turbo"). If not provided, a default model will be used (e.g., "gpt-4o").

The API response will be augmented to include:
`model_used` (string): The actual model that processed the query.
`input_tokens` (integer): Number of tokens in the input to the LLM.
`output_tokens` (integer): Number of tokens in the LLM's response.
`estimated_cost_usd` (float): Estimated cost of the query in USD.

#### 4.1.2. Model Configuration & Handling (`backend/rag_query.py` and `notion_second_brain/config.py`)
*   A dictionary or configuration structure will map `model_name` strings to their specific API identifiers and pricing information. This could be in `config.py` or a separate JSON/YAML configuration file.
    Example:
    ```python
    MODEL_CONFIG = {
        "gpt-4o": {"api_id": "gpt-4o", "input_cost_per_mtok": 5.00, "output_cost_per_mtok": 15.00, "context_window_tokens": 128000, "description": "Latest, most capable model."},
        "gpt-3.5-turbo": {"api_id": "gpt-3.5-turbo", "input_cost_per_mtok": 0.50, "output_cost_per_mtok": 1.50, "context_window_tokens": 16385, "description": "Fast, cost-effective model."},
        # Future:
        # "claude-3-opus": {"api_id": "claude-3-opus-20240229", "input_cost_per_mtok": 15.00, "output_cost_per_mtok": 75.00, ...},
    }
    ```
*   The RAG query logic in `backend/rag_query.py` will:
    *   Read the `model_name` from the request.
    *   Use the corresponding `api_id` when making calls to the OpenAI (and eventually Anthropic) client.
    *   Ensure API clients are instantiated correctly based on the model provider.
*   An `ANTHROPIC_API_KEY` will be added to `.env.example` and loaded via `config.py` for future Anthropic integration.

#### 4.1.3. Token Counting & Cost Estimation
*   The backend will need to count input and output tokens for the LLM call.
    *   For OpenAI, the response object often includes token usage.
    *   For Anthropic, similar mechanisms or client-side tokenization (e.g., using `tiktoken` for OpenAI-compatible counting if necessary, or Anthropic's own tokenizer) might be needed.
*   The cost will be estimated using: `(input_tokens / 1_000_000 * input_cost_per_mtok) + (output_tokens / 1_000_000 * output_cost_per_mtok)`.
*   This logic will reside in `backend/rag_query.py` or a helper utility.

### 4.2. Frontend Changes (`frontend/src/App.tsx`)

#### 4.2.1. Model Selector Component
*   A `Select` dropdown component (from `shadcn/ui`) will be added to the UI.
*   The options will be populated from a list of available models (e.g., "OpenAI GPT-4o", "OpenAI GPT-3.5 Turbo"). This list could be fetched from a new backend endpoint or be hardcoded initially. For MVP, hardcoding is acceptable.
*   The selected model will be stored in the React component's state.
*   This selected model name will be passed in the `model_name` field of the `/api/query` request.

#### 4.2.2. Displaying Model Information
*   Optionally, to help users choose, brief descriptions or key characteristics (e.g., "Fastest", "Most Capable", "Lower Cost") could be displayed next to model names in the dropdown or as a tooltip. This information would come from the `MODEL_CONFIG`.

#### 4.2.3. Displaying Query Cost & Model Used
*   After a query response is received, the UI will display:
    *   The model used for the query (from `response.data.model_used`).
    *   The estimated cost (from `response.data.estimated_cost_usd`), formatted appropriately (e.g., "$0.0012").
    *   Optionally, token counts (`input_tokens`, `output_tokens`) can also be displayed.

### 4.3. Data Model / Configuration

*   **Model Pricing and Details:** As described in 4.1.2, a configuration (likely Python dict in `config.py` or a JSON file loaded by `config.py`) will store:
    *   User-friendly name (e.g., "OpenAI GPT-4o")
    *   API identifier (e.g., "gpt-4o")
    *   Input cost per million tokens
    *   Output cost per million tokens
    *   Context window size (optional, for display/information)
    *   Short description (optional, for display)
    *   Provider (e.g., "OpenAI", "Anthropic")

## 5. Future Considerations

*   **Dynamic Pricing Updates:** Fetching model pricing dynamically from provider APIs or a maintained external source.
*   **User-Specific API Keys:** Allowing users to enter their own API keys for model providers.
*   **More Model Providers:** Expanding to include other LLM providers (e.g., Cohere, AI21).
*   **Streaming Support:** Ensuring cost estimation works correctly with streaming responses (may require final token count after stream completion).
*   **More Granular Model Capabilities Display:** Showing more detailed model specs if deemed useful.
*   **Error Handling:** Graceful error handling if a selected model is unavailable or an API key is missing/invalid.

## 6. Open Questions

*   Where should the `MODEL_CONFIG` be stored? (Initial thought: `config.py` for simplicity, or a `models.json` in the `backend` directory).
*   How will the list of available models be populated in the frontend dropdown for the MVP? (Initial thought: Hardcoded in `App.tsx`, synchronized with `MODEL_CONFIG` keys). A dedicated backend endpoint is a more robust long-term solution.
*   Default model if `model_name` is not specified by the client? (Proposal: `gpt-4o`).
*   How to handle tokenization for cost estimation if the API response doesn't provide it (especially for Anthropic if their client differs significantly)? (Research needed for Anthropic; OpenAI usually provides this).

---
This TDD is a living document and will be updated as the feature evolves. 