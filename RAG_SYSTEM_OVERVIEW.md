# RAG System Overview: Notion Second Brain

This document provides a comprehensive overview of the Retrieval Augmented Generation (RAG) system built for the Notion Second Brain project. It details the end-to-end data flow, indexing process, and the query processing logic.

## 1. High-Level System Architecture

The system connects to a Notion database, extracts journal entries, processes them, builds a searchable index, and provides a query interface via a backend API (supporting model selection) and a frontend UI.

```mermaid
graph TD
    A[Notion Database] --> B(Data Export via cli.py to GCS);
    B --> C{JSON Files per Month (on GCS)};
    C --> D[Indexing Process via build_index.py (uses GCS data, targets Pinecone & GCS for artifacts)];
    D -- Upserts vectors --> Pinecone_DB[Pinecone Vector DB];
    D -- Saves artifact --> F[Entry Mapping (index_mapping.json on GCS)];
    D -- Saves artifact --> G[Metadata Cache (metadata_cache.json on GCS)];
    D -- Saves artifact --> Schema[Schema (schema.json on GCS)];
    D -- Saves artifact --> LastUpdated[Last Update Timestamp (last_entry_update_timestamp.txt on GCS)];
    H[User Query via Frontend UI] --> I{Backend API (FastAPI - Modular RAG Backend)};
    I -- Queries --> Pinecone_DB;
    I -- Uses --> F;
    I -- Uses --> G;
    I -- Uses --> Schema;
    I -- Uses --> J[LLM Providers (OpenAI, Anthropic)];
    I --> K[Answer with Cost & Model Info to Frontend UI];
```

**Components:**
-   **Notion Database:** The source of journal entries.
-   **`cli.py`:** Command-line interface for setup operations (exporting data as monthly JSONs to Google Cloud Storage under `GCS_EXPORT_PREFIX`, testing connection, retrieving and uploading `schema.json` to GCS) and for initiating queries. Query execution now delegates to the core RAG logic within the `api` module.
-   **`build_index.py`:** Script responsible for processing the exported JSON files (from GCS), generating embeddings for entry content, upserting vectors to Pinecone, and creating/uploading several artifacts to Google Cloud Storage under `GCS_INDEX_ARTIFACTS_PREFIX`.
    -   `index_mapping.json`: (On GCS) A JSON file mapping each entry (identified by `page_id`) to its corresponding journal entry's metadata and full content. Includes `page_id`, `title`, `entry_date`, `content`, and extracted metadata like `Family`, `Friends`, `Tags`. This is used by the backend to retrieve context for the LLM.
    -   `metadata_cache.json`: (On GCS) Stores unique values for filterable metadata fields (e.g., all unique names in 'Family', all unique tags in 'Tags') to aid query analysis.
    -   `schema.json`: (On GCS) A representation of the Notion database schema, detailing property names and types. Uploaded by `cli.py --schema` and used by `build_index.py` and the backend.
    -   `last_entry_update_timestamp.txt`: (On GCS) A text file storing the ISO timestamp of the most recently processed entry during the last successful run of `build_index.py`. This is used by the frontend to display when the data was last synced.
-   **Backend API (`api/`):**
    -   **Entry Point (`api/main.py`):** FastAPI application handling HTTP requests, responses, CORS, and basic API structure. Initializes shared RAG state via `rag_initializer` on startup. Also includes Google OAuth authentication routes and session management.
    -   **Authentication Module (`api/auth.py`):** Contains logic for Google OAuth client setup, login/callback handling, user validation against `AUTHORIZED_USER_EMAIL`, and session data management. Provides a dependency for protecting routes.
    -   **RAG Orchestrator (`api/rag_query.py`):** Contains the main `perform_rag_query` function which orchestrates the entire RAG pipeline by calling specialized modules. Accesses shared state (clients, Pinecone index, data from GCS) dynamically from `rag_initializer`. Includes `execute_rag_query_sync` wrapper for CLI use.
    -   **Shared State Manager (`api/rag_initializer.py`):** Loads and holds shared components (Pinecone index client, mappings from GCS, metadata from GCS, schema from GCS, LLM clients) as module globals. Provides functions for initialization, accessed by entry points and other modules.
    -   **Query Analyzer (`api/query_analyzer.py`):** Handles LLM calls to analyze the user query and extract structured filters (date ranges, metadata), using schema and distinct values loaded from GCS.
    -   **Retrieval Logic (`api/retrieval_logic.py`):** Handles embedding generation, Pinecone searching (vector search + metadata filtering), and applying metadata filter fallbacks.
    -   **LLM Interface (`api/llm_interface.py`):** Handles interaction with different LLM providers (OpenAI, Anthropic) for final answer generation.
    -   **Prompt Constructor (`api/prompt_constructor.py`):** Assembles the final prompts for the LLM based on context, query type, and fallback status.
    -   **Cost Utility (`api/cost_utils.py`):** Calculates estimated query cost based on token usage across different steps.
    -   **Configuration (`api/rag_config.py`):** Defines constants, file paths, default models, and the `MODEL_CONFIG` dictionary (model properties, providers, pricing).
    -   Provides an endpoint (e.g., `/api/query`) to receive user queries, including an optional `model_name` for selecting the LLM. This endpoint, along with `/api/last-updated` and `/api/transcribe`, is now protected and requires authentication.
    -   Provides an endpoint (`/api/last-updated`) to retrieve the timestamp from `last_entry_update_timestamp.txt` (stored on GCS).
    -   Provides authentication endpoints:
        -   `GET /api/auth/login/google`: Initiates Google OAuth login.
        -   `GET /api/auth/callback/google`: Handles Google's callback, authenticates user, and sets session.
        -   `GET /api/auth/me`: Returns current authenticated user's status and details.
        -   `POST /api/auth/logout`: Clears user session.
-   **LLM Providers (OpenAI, Anthropic):** Used for generating text embeddings (e.g., OpenAI's `text-embedding-ada-002`) and for powering Large Language Model (LLM) calls (e.g., OpenAI's `gpt-4o`, `gpt-4o-mini`, Anthropic's `claude-3-5-haiku-20241022`) for query analysis and final answer generation.
-   **Frontend UI (`frontend/`):**
    -   React/Vite/TypeScript application.
    -   Manages authentication state using `AuthContext` (`frontend/src/contexts/AuthContext.tsx`).
    -   Handles routing and protected views in `frontend/src/App.tsx`.
    -   Provides a `LoginPage` (`frontend/src/pages/LoginPage.tsx`) for initiating login.
    -   Includes a `LogoutButton` (`frontend/src/components/auth/LogoutButton.tsx`).
    -   Provides an interface for users to select a model, input queries, and view answers with sources, model details, and estimated costs, after successful authentication.

## 2. Indexing Process (`build_index.py`)

1.  **Load Data:** Reads journal entries from JSON files previously exported by `cli.py --export-month YYYY-MM` (or `cli.py --export`). These JSON files are read from Google Cloud Storage (GCS) from the configured `GCS_EXPORT_PREFIX`. Can process multiple monthly files incrementally.
2.  **Content Preparation:** For each entry, relevant text content is extracted and combined.
3.  **Embedding Generation:** The combined text content of each entry is sent to the OpenAI API (using a model specified in `MODEL_CONFIG`, e.g., `text-embedding-ada-002`) to generate a vector embedding. Batch embedding requests are used for efficiency.
4.  **Pinecone Vector Upsert:**
    -   For each entry with valid content and an embedding, a vector is prepared for Pinecone.
    -   The `page_id` of the entry is used as the unique ID for the vector in Pinecone.
    -   Key metadata (e.g., `title`, `entry_date`, `tags`) is included with the vector for filtering capabilities within Pinecone.
    -   Vectors are upserted to the configured Pinecone index in batches for efficiency.
    -   No local `.faiss` index file is created or saved.
5.  **Index-to-Entry Mapping (`index_mapping.json`):**
    -   A list of dictionaries is created. Each dictionary contains the full data (including `page_id`, `title`, `entry_date`, original `content`, and all extracted metadata) for every entry that was processed and whose vector was upserted (or intended for upsert) to Pinecone.
    -   This mapping is crucial for the backend to retrieve the full context of entries whose `page_id`s are returned by a Pinecone search.
    -   This mapping is saved as `index_mapping.json` to GCS under `GCS_INDEX_ARTIFACTS_PREFIX`.
6.  **Metadata Cache (`metadata_cache.json`):**
    -   Iterates through all entries in the (final) `index_mapping.json` data.
    -   Extracts unique string values for key filterable list/multi-select fields (e.g., `Family`, `Friends`, `Tags`).
    -   Stores these efficiently (e.g., dictionary of sets, then saved as lists in JSON) in `metadata_cache.json` on GCS (under `GCS_INDEX_ARTIFACTS_PREFIX`). This helps the query analysis LLM map query terms to known metadata values.
7.  **Schema (`schema.json`):** This file, typically generated by `cli.py --schema` and uploaded to GCS (under `GCS_INDEX_ARTIFACTS_PREFIX`), defines the structure of the Notion DB properties. `build_index.py` may also re-upload a local copy if present, ensuring it's available on GCS for the backend.
8.  **Last Processed Entry Timestamp (`last_entry_update_timestamp.txt`):**
    -   During the processing of entries, `build_index.py` tracks the `last_edited_time` of each entry.
    -   After successfully processing all input files and saving other artifacts, it writes the most recent `last_edited_time` encountered to `last_entry_update_timestamp.txt` as an ISO format string to GCS (under `GCS_INDEX_ARTIFACTS_PREFIX`).
    -   If an existing timestamp file is present on GCS and not a forced rebuild, the script ensures this file always reflects the latest entry processed across all runs.
9.  **Incremental Updates:** The script supports loading an existing mapping from GCS (if not `force_rebuild`) to identify already processed `page_id`s. This allows it to primarily process and embed only new or updated entries before upserting to Pinecone and updating the artifacts on GCS.

## 3. Backend API Query Flow

The main orchestration happens in `api/rag_query.py` (`perform_rag_query`), called by `api/main.py` for API requests or `cli.py` (via `execute_rag_query_sync`) for command-line queries. Access to API endpoints like `/api/query` first requires successful authentication handled by `api/main.py` and `api/auth.py`. Shared state like clients and loaded data is managed by `api/rag_initializer.py`.

### 3.1. Request Reception & Initialization (and Authentication)

-   The FastAPI app (`main.py`, located in `api/`) receives a POST request to `/api/query` with `{"query": "...", "model_name": "..."}`. Before processing the query, the authentication dependency derived from `api/auth.py` verifies the user's session. If unauthenticated, a 401 error is returned.
-   On startup, `main.py` calls functions in `rag_initializer.py` (both in `api/`) to load data (initialize Pinecone index client, load mapping from GCS, metadata from GCS, schema from GCS) and initialize LLM clients (OpenAI, Anthropic) based on API keys. These are stored as globals within `rag_initializer`.
-   `rag_config.py` (in `api/`) holds static configuration like `MODEL_CONFIG`.
-   The `/api/last-updated` endpoint reads `last_entry_update_timestamp.txt` from GCS.

### 3.2. Query Analysis (`query_analyzer.py`)

-   **Goal:** Convert natural language query to structured filters.
-   **Process:** Orchestrated by `api/rag_query.py`.
    1.  `rag_query.py` calls `analyze_query_for_filters` in `api/query_analyzer.py`.
    2.  `query_analyzer.py` accesses the initialized `openai_client`, `schema_properties`, and `distinct_metadata_values` from `api/rag_initializer`.
    3.  It constructs a prompt (including current date, schema, distinct values) and sends the user query to the configured query analysis LLM (e.g., `gpt-4o`).
    4.  The LLM returns a JSON containing optional `date_range` and `filters` list.
-   **Output:** `query_analyzer.py` returns a dictionary to `api/rag_query.py` with the extracted `filters`, token counts, and any errors.

### 3.3. Pre-filtering Candidate Entries (`retrieval_logic.py`)

-   **Goal:** Narrow the search space using metadata.
-   **Process:** Orchestrated by `api/rag_query.py`.
    1.  `rag_query.py` calls `apply_metadata_filters` in `api/retrieval_logic.py`, passing the full entry list (from `api/rag_initializer.mapping_data_list`), the `filter_analysis` results, and distinct values (from `api/rag_initializer.distinct_metadata_values`).
    2.  `apply_metadata_filters` performs sequential filtering based on date range, other metadata (tags), and names, implementing the OR logic for names and the fallback mechanisms for tags and names if initial filtering yields zero results.
-   **Output:** `apply_metadata_filters` returns a dictionary to `api/rag_query.py` containing the final list of candidate entries for counting, the count itself, and various flags indicating active filters and whether fallbacks were triggered.

### 3.4. Semantic Search (`retrieval_logic.py`)

-   **If Candidate Indices Exist:** Orchestrated by `api/rag_query.py`.
    1.  **Query Embedding:** `rag_query.py` calls `get_embedding` in `api/retrieval_logic.py` to get the query vector using the OpenAI client from `api/rag_initializer`.
    2.  **Pinecone Search:** `rag_query.py` calls a search function in `api/retrieval_logic.py` (e.g., `perform_pinecone_search`). This function uses the Pinecone index client (from `api/rag_initializer.pinecone_index_client`) to query the index with the query vector and any metadata filters derived from pre-filtering (step 3.3).
-   **Output:** The Pinecone search returns the top `k` matching results, typically including `page_id`s, scores, and potentially metadata, to `api/rag_query.py`.

### 3.5. Context Retrieval (`rag_query.py`)

-   This logic remains within `api/rag_query.py`.
-   It uses the retrieved `page_id`s from Pinecone to look up the corresponding entry data (title, content, date, page_id) from the `mapping_data_list` (loaded from `index_mapping.json` on GCS via `api/rag_initializer`).
-   It constructs context strings for the LLM and prepares the `sources` list for the final response, generating Notion URLs from `page_id`s.

### 3.6. Final Answer Generation (LLM - `llm_interface.py`)

-   **Goal:** Generate the narrative answer using the selected model.
-   **Process:** Orchestrated by `api/rag_query.py`.
    1.  `rag_query.py` uses the `model_name` to get configuration from `MODEL_CONFIG` (defined in `api/rag_config.py`).
    2.  It calls `construct_final_prompts` (from `api/prompt_constructor.py`) to assemble the system and user prompts, incorporating retrieved context and flags (e.g., fallback status, query type).
    3.  It calls `generate_final_answer` in `api/llm_interface.py`, passing the prompts and model config.
    4.  `llm_interface.py` accesses the appropriate client (OpenAI or Anthropic) via `api/rag_initializer`, makes the API call using `asyncio.to_thread`, and returns the text answer, token counts, and any errors.
-   **Output:** `llm_interface.py` returns the generated text, tokens, and error status to `api/rag_query.py`. `rag_query.py` then calls `calculate_estimated_cost` (from `api/cost_utils.py`) to get the final cost.

### 3.7. Response Serialization & Return (`api/main.py`)

-   `api/rag_query.py` returns the final result dictionary (answer, sources, model info, tokens, cost) to `api/main.py`.
-   `api/main.py` validates and serializes this using Pydantic models (`QueryResponse`, `SourceDocument`) and sends the JSON response back to the client.

## 4. Frontend UI (`frontend/`)

-   **Authentication:**
    -   On load, `AuthProvider` calls `/api/auth/me` to check authentication status.
    -   `App.tsx` uses `isLoading` and `isAuthenticated` from `AuthContext` to either show a loading spinner, redirect to `/login` (if not authenticated), or display `MainAppLayout` (if authenticated).
    -   `LoginPage.tsx` provides a button to redirect to `/api/auth/login/google`.
    -   `LogoutButton.tsx` in `MainAppLayout.tsx` makes a POST request to `/api/auth/logout`.
-   **Query Submission:** User types a query and submits (only possible when authenticated and `MainAppLayout` is rendered).
-   **API Call:** An `axios` POST request is made to the backend's `/api/query` endpoint, including the `model_name` selected by the user and `withCredentials: true` to send session cookies.
-   **Response Handling:**
    -   **Answer Display:** The `answer` string from the API response is rendered using `ReactMarkdown`.
        -   Information about the `model_used`, `provider`, token counts, and `estimated_cost_usd` is displayed.
        -   Tailwind CSS `prose` classes are used for good default typography and markdown rendering.
        -   Custom component for `<a>` tags ensures links open in new tabs and have appropriate styling.
        -   Custom component for `<p>` tags improves paragraph spacing.
    -   **"Sources Used" Display:**
        -   The `sources` array from the API response is processed.
        -   Sources are sorted client-side by date (ascending).
        -   A `formatDate` utility converts "YYYY-MM-DD" strings into a human-readable format (e.g., "January 7, 2025").
        -   Displayed as a simple bulleted list, with each item showing:
            -   Hyperlinked `source.title`.
            -   The formatted `source.date` in parentheses, e.g., `(November 11, 2024)`.
-   **Loading/Error States:** The UI handles loading spinners and displays error messages from the API.
-   **Query Suggestion:** Basic logic to suggest adding a timeframe for queries using "recent" keywords without a year.
-   **Last Synced Display:** On component mount, an `axios` GET request is made to `/api/last-updated`. The retrieved timestamp (or an error message) is displayed, formatted for readability.

This overview should provide a solid understanding of the system's mechanics. 