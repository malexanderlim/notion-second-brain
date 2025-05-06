# RAG System Overview: Notion Second Brain

This document provides a comprehensive overview of the Retrieval Augmented Generation (RAG) system built for the Notion Second Brain project. It details the end-to-end data flow, indexing process, and the query processing logic.

## 1. High-Level System Architecture

The system connects to a Notion database, extracts journal entries, processes them, builds a searchable index, and provides a query interface via a backend API (supporting model selection) and a frontend UI.

```mermaid
graph TD
    A[Notion Database] --> B(Data Export via cli.py);
    B --> C{JSON Files per Month};
    C --> D[Indexing Process via build_index.py];
    D --> E[FAISS Vector Index (index.faiss)];
    D --> F[Entry Mapping (index_mapping.json)];
    D --> G[Metadata Cache (metadata_cache.json)];
    H[User Query via Frontend UI] --> I{Backend API (FastAPI - Supports Model Selection)};
    I -- Uses --> E;
    I -- Uses --> F;
    I -- Uses --> G;
    I -- Uses --> J[LLM Providers (OpenAI, Anthropic) for Embeddings & LLM Tasks];
    I --> K[Answer with Cost & Model Info to Frontend UI];
```

**Components:**
-   **Notion Database:** The source of journal entries.
-   **`cli.py`:** Command-line interface for various operations, including exporting data from Notion into JSON files (typically one per month of entries).
-   **`build_index.py`:** Script responsible for processing the exported JSON files, generating embeddings for entry content, creating a FAISS vector index, an index-to-entry mapping file, and a cache of distinct metadata values.
    -   `index.faiss`: The FAISS vector store containing embeddings of journal entry content.
    -   `index_mapping.json`: A JSON file mapping each vector index in FAISS back to its corresponding journal entry's metadata and content. Includes `page_id`, `title`, `entry_date`, full `content`, and extracted metadata like `Family`, `Friends`, `Tags`.
    -   `metadata_cache.json`: Stores unique values for filterable metadata fields (e.g., all unique names in 'Family', all unique tags in 'Tags') to aid query analysis.
    -   `schema.json`: A representation of the Notion database schema, detailing property names and types.
-   **Backend API (`backend/main.py`, `backend/rag_query.py`):**
    -   Built with FastAPI.
    -   Provides an endpoint (e.g., `/api/query`) to receive user queries, including an optional `model_name` for selecting the LLM.
    -   Orchestrates the RAG process using helper functions.
    -   Contains `MODEL_CONFIG` defining available models, their API IDs, providers, and pricing.
-   **LLM Providers (OpenAI, Anthropic):** Used for generating text embeddings (e.g., OpenAI's `text-embedding-ada-002`) and for powering Large Language Model (LLM) calls (e.g., OpenAI's `gpt-4o`, `gpt-4o-mini`, Anthropic's `claude-3-5-haiku-20241022`) for query analysis and final answer generation.
-   **Frontend UI (`frontend/src/App.tsx`):**
    -   React/Vite/TypeScript application.
    -   Provides an interface for users to select a model, input queries, and view answers with sources, model details, and estimated costs.

## 2. Indexing Process (`build_index.py`)

1.  **Load Data:** Reads journal entries from JSON files previously exported by `cli.py --export-month YYYY-MM`. Can process multiple monthly files incrementally.
2.  **Content Preparation:** For each entry, relevant text content is extracted and combined.
3.  **Embedding Generation:** The combined text content of each entry is sent to the OpenAI API (using a model specified in `MODEL_CONFIG`, e.g., `text-embedding-ada-002`) to generate a vector embedding. Batch embedding requests are used for efficiency.
4.  **FAISS Index Creation:**
    -   A FAISS index (typically `IndexFlatL2`) is created.
    -   Embeddings from all entries are added to this index.
    -   The index is saved to `index.faiss`.
5.  **Index-to-Entry Mapping (`index_mapping.json`):**
    -   A list of dictionaries is created. Each dictionary corresponds to an entry in the FAISS index (by its numerical index/order).
    -   Each dictionary stores: `page_id`, `title`, `entry_date` (as "YYYY-MM-DD" string), original `content`, extracted metadata fields (e.g., `Family`, `Friends`, `Tags` as lists of strings), and the `faiss_index` itself.
    -   This mapping is saved to `index_mapping.json`.
6.  **Metadata Cache (`metadata_cache.json`):**
    -   Iterates through all entries in `index_mapping.json`.
    -   Extracts unique string values for key filterable list/multi-select fields (e.g., `Family`, `Friends`, `Tags`).
    -   Stores these efficiently (e.g., dictionary of sets, then saved as lists in JSON) in `metadata_cache.json`. This helps the query analysis LLM map query terms to known metadata values.
7.  **Schema (`schema.json`):** While not directly created by `build_index.py`, this file (expected to be present) defines the structure of the Notion DB properties and is used by the query analysis step. It should be generated or maintained based on the actual Notion DB schema.
8.  **Incremental Updates:** The script supports loading an existing index and mapping to add new entries, allowing for checkpointing and incremental builds.

## 3. Backend API Query Flow

The primary logic resides in `backend/rag_query.py` (`perform_rag_query` function), orchestrated by `backend/main.py` which handles the API request/response cycle.

### 3.1. Request Reception & Initialization

-   The FastAPI app in `main.py` receives a POST request to `/api/query` with a JSON body like `{"query": "user's question", "model_name": "optional_model_key"}`.
-   On startup, `main.py` calls `load_rag_data()` (from `rag_query.py`) to load `index.faiss`, `index_mapping.json`, `metadata_cache.json`, and `schema.json` into global variables. It also initializes the OpenAI and Anthropic clients based on available API keys. The `MODEL_CONFIG` (defining model properties, API IDs, providers, and costs) is also available in `rag_query.py`.

### 3.2. Query Analysis (`analyze_query_for_filters` in `rag_query.py`)

-   **Goal:** Convert the natural language user query into structured filters.
-   **Process:**
    1.  The user query is sent to an LLM (e.g., `gpt-4o`, as specified by `DEFAULT_QUERY_ANALYSIS_MODEL_KEY` in `MODEL_CONFIG`).
    2.  The LLM prompt includes:
        -   The user's query.
        -   The current date (e.g., "Today's date is YYYY-MM-DD...") to aid in parsing relative date expressions (e.g., "last 6 months", "next week").
        -   The database schema from `schema.json` (property names and types).
        -   Distinct values from `metadata_cache.json` for relevant fields (e.g., known names in "Family", known tags in "Tags") to help the LLM map query terms accurately.
    3.  The LLM is instructed to output a JSON object containing:
        -   An optional `date_range` key with `start` and `end` sub-keys (in "YYYY-MM-DD" format).
        -   An optional `filters` key, which is a list of objects. Each object has:
            -   `field`: The Notion property name (e.g., "Family", "Tags").
            -   `contains`: The value extracted from the query (e.g., "Leo", "restaurant").
        -   The LLM is guided to map person names to "Family" or "Friends" fields. If a name could be in either, it may suggest filters for both.
-   **Output:** A dictionary including `filters`, `input_tokens`, `output_tokens`, and any `error`. Token usage for this step is tracked.

### 3.3. Pre-filtering Candidate Entries (`perform_rag_query` in `rag_query.py`)

-   **Goal:** Narrow down the search space from all journal entries to a relevant subset *before* performing expensive semantic search.
-   **Process:**
    1.  Starts with `current_candidates` as a list of all entries from `mapping_data_list` (loaded from `index_mapping.json`). Each entry dictionary contains its `faiss_index`, `entry_date` (as a Python `date` object after loading), and other metadata.
    2.  **Date Filter:** If `filter_analysis` provided a `date_range`, `current_candidates` are filtered to include only entries whose `entry_date` falls within this range.
    3.  **Other Metadata Filters (e.g., Tags):**
        -   Filters from `filter_analysis` that are *not* name-related (e.g., `{"field": "Tags", "contains": "restaurant"}`) are applied next.
        -   This typically involves checking if the `contains` value is present in the corresponding list-based metadata field of an entry (e.g., `entry['Tags']`). Logic is case-insensitive.
        -   `current_candidates` is updated.
        -   *(Future Enhancement: A "Tag Fallback Mechanism" is planned here. If this step results in zero candidates and tag filters were active, it would revert `current_candidates` to its state before tag filtering and set a flag.)*
    4.  **Name Filters (`Family`, `Friends`):**
        -   Filters specifically for configured name fields (e.g., "Family", "Friends") are applied.
        -   An OR logic is used: an entry matches if it matches *any* of the active name filters (e.g., Family contains "Leo" OR Friends contains "Leo").
        -   This also checks if the `contains` value is present in the list-based name field of an entry.
        -   `current_candidates` is updated.
    5.  **Name Filter Fallback Logic:**
        -   A copy of `current_candidates` is stored *before* applying the name filters (let's call this `candidates_before_name_filter`).
        -   If, after applying name filters, `current_candidates` becomes empty AND name filters were active:
            -   `current_candidates` is reverted to `candidates_before_name_filter`.
            -   A flag `fallback_triggered_due_to_names` is set to `True`.
            -   This ensures that if a name isn't explicitly tagged, the system can still search a broader set of entries (filtered by date and other tags) for mentions of that name in the content.
    6.  **Final Candidate Indices:** The `faiss_index` values from the final `current_candidates` list are collected. If this list is empty, a "no matching entries" message is prepared.

### 3.4. Semantic Search (`perform_rag_query`)

-   **If Candidate Indices Exist:**
    1.  **Query Embedding:** The original user query string is sent to an OpenAI embedding model (e.g., `text-embedding-ada-002` via `get_embedding`, configured in `MODEL_CONFIG`) to get its vector embedding. Token usage for this step is tracked.
    2.  **FAISS Search:**
        -   A `faiss.IDSelectorBatch` is created using the `final_candidate_faiss_indices`.
        -   The FAISS `index.search()` method is called, searching for the `query_embedding` within the subset of vectors specified by the `IDSelectorBatch`.
        -   It retrieves the top `k` (configurable, e.g., `TOP_K = 15`) most similar FAISS indices and their distances. `k` is capped by the number of actual candidate indices.
-   **Output:** A list of FAISS indices of the most relevant entries from the pre-filtered set.

### 3.5. Context Retrieval (`perform_rag_query`)

-   For each FAISS index retrieved from the search:
    1.  The full entry data (including `title`, `content`, `entry_date`, `page_id`) is looked up from the global `index_to_entry` dictionary (which was derived from `index_mapping.json`).
    2.  A context string is constructed for each entry, typically including its ID, title, date, and full content. Example: `Document (ID: {faiss_idx}, Title: {title}, Date: {entry_date_str}):\n{content}\n---`
    3.  Information for the "Sources Used" list is also prepared:
        -   `title`: Entry title.
        -   `url`: Notion URL. This is constructed on-the-fly using the `page_id` (e.g., `https://www.notion.so/{page_id_without_hyphens}`) if not already present in the entry data.
        -   `id`: The `faiss_idx` (or `page_id`).
        -   `date`: The `entry_date_str` ("YYYY-MM-DD").
        -   `distance`: (Optional) Semantic distance from FAISS search.
-   All context strings are combined into a single large string to be passed to the final LLM.

### 3.6. Final Answer Generation (LLM - `perform_rag_query`)

-   **Goal:** Generate a human-readable, narrative answer based on the retrieved context and the user's original query, using the user-selected (or default) model.
-   **Process:**
    1.  The `model_name` from the API request (or `DEFAULT_FINAL_ANSWER_MODEL_KEY`) is used to look up the model's details (including `api_id`, `provider`, and costs) from `MODEL_CONFIG`.
    2.  The appropriate LLM client (OpenAI or Anthropic) is selected based on the model's provider.
    3.  A prompt is constructed for the selected LLM. This prompt includes:
        -   **`base_system_message`:** Sets the persona and overall style. The specific base message varies based on the query type:
            -   **General Queries:** "You are an AI assistant functioning as a 'Second Brain.' Your purpose is to help the user recall and synthesize information from their journal entries. Respond in a natural, reflective, and narrative style. Organize your answers clearly, often using bolded thematic titles to highlight key memories or points, each followed by descriptive details. Cite specific journal entries when you draw information from them, as per the detailed formatting instructions provided."
            -   **Quantitative Person Queries (e.g., "How many times did I see X?"):** A specialized template is used. It first states the `metadata_based_interaction_count` for the person, then instructs the LLM to provide a narrative summary using exemplar documents.
            -   **Quantitative Tag Queries (e.g., "How many times did I cook?"):** A similar specialized template is used. It first states the `metadata_based_interaction_count` for the tag/activity, then instructs the LLM to provide a narrative summary using exemplar documents.
        -   **`formatting_instructions`:** Provides detailed rules for answer structure and citation.
            -   Emphasis on narrative flow, introduction, bolded thematic sections, and a conclusion.
            -   Specific citation style: `[Title of Entry](Notion URL) (Formatted Date)` (e.g., `(January 7, 2025)`), with the LLM expected to naturally hyperlink the title and append the formatted date.
            -   For quantitative person/tag queries, instructions explicitly guide the LLM to state the count first.
        -   **Context String:** The combined content from all retrieved documents (exemplars).
        -   **User Query:** The original user query.
    2.  **Fallback Prompt Modification:** If `fallback_triggered_due_to_names` or `fallback_triggered_due_to_tags` is true, the `final_system_prompt` is prepended with a message informing the LLM that the initial metadata search failed and the context is from a broader search, instructing it to specifically look for the named individuals or concepts in the content.
-   **Output:** The LLM's generated text response. Token usage (input and output) for this final generation step is tracked. The total estimated cost for the query (summing embedding, query analysis, and final answer generation tokens) is calculated using pricing from `MODEL_CONFIG`.

### 3.7. Response Serialization & Return (`backend/main.py`)

-   The `handle_query` function in `main.py` receives a dictionary from `perform_rag_query` containing the `answer`, `sources_data`, `model_used`, `model_api_id_used`, `model_provider_used`, `input_tokens`, `output_tokens`, and `estimated_cost_usd`.
-   It validates each item in `sources_data` and converts it into a `SourceDocument` Pydantic model instance.
-   The final response is structured using the `QueryResponse` Pydantic model, which now includes all the above fields, and sent back to the frontend as JSON.

## 4. Frontend UI (`frontend/src/App.tsx`)

-   **Query Submission:** User types a query and submits.
-   **API Call:** An `axios` POST request is made to the backend's `/api/query` endpoint, now including the `model_name` selected by the user.
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

This overview should provide a solid understanding of the system's mechanics. 