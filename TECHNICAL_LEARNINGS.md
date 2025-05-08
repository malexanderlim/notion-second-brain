# Technical Learnings: Notion Second Brain RAG Implementation

This document captures key technical insights and evolution during the development and debugging of the RAG (Retrieval-Augmented Generation) system for querying Notion journal entries.

## 1. Initial MVP RAG Limitations

*   **Metadata Blindness:** Embedding only the main `content` block means the system cannot effectively answer queries based on metadata (Tags, Family, Friends, Dates) unless that metadata is repeated *within* the content.
*   **Context Scarcity:** Retrieving only a small number (`TOP_K=3-5`) of the most semantically similar entries often provides insufficient context for the final LLM to synthesize answers, especially for frequency/counting questions ("how many times...") or questions requiring information spread across multiple entries.
*   **Content vs. Preview:** Storing only content previews (or truncated content) in the mapping file severely limits the LLM's ability to answer questions requiring details found later in an entry. The full content is necessary for comprehensive answers.

## 2. Improving Retrieval & Context

*   **Enriched Embeddings:** Including key metadata (Title, Date, Tags, People) *along with* the content in the text sent for embedding helps the vector search find entries relevant to metadata-focused queries.
*   **Full Content Mapping:** The `index_mapping.json` file *must* store the full content of each entry if the final LLM is expected to reason over it.
*   **Balancing `TOP_K`:** Increasing `TOP_K` retrieves more potentially relevant context, improving the LLM's synthesis ability. However, this directly increases the input token count for the final LLM call.

## 3. LLM Challenges & Prompting

*   **Context Window Limits:** Sending excessive context (high `TOP_K` combined with full entry content) can easily exceed LLM context window limits (e.g., `context_length_exceeded` errors for models like `gpt-3.5-turbo`).
*   **Model Capability:** More advanced models (e.g., `gpt-4o` vs. `gpt-3.5-turbo`) demonstrate significantly better performance in adhering to complex instructions, extracting information comprehensively from provided context, and synthesizing answers accurately.
*   **Explicit Instruction is Key:**
    *   Prompts must clearly instruct the LLM on its task, constraints (e.g., "answer ONLY from context"), desired output format (dates, links), and any necessary assumptions.
    *   **Injecting Assumptions:** For inferences like "tagged = seen", explicitly stating this assumption in the system prompt allows the LLM to bridge the gap between metadata and implicit meaning.

## 4. Hybrid RAG: Pre-filtering + Targeted Search

*   **Efficiency & Relevance:** A hybrid approach combining metadata pre-filtering with semantic search is highly effective.
    *   **Query Analysis:** An initial LLM call (e.g., `gpt-4o-mini`) analyzes the natural language query against the database schema (and known values) to extract structured filters (date ranges, field contains value).
    *   **Pre-filtering:** The extracted filters are applied to the full `mapping_data` *before* semantic search, drastically reducing the candidate pool.
    *   **Targeted Search:** FAISS's `IDSelectorBatch` allows performing the vector similarity search *only* within the pre-filtered subset of indices, ensuring retrieved results match hard criteria *and* semantic relevance.
*   **Query Analysis Nuances:**
    *   Requires careful prompt engineering to map natural language terms (names, relative dates, keywords) to the correct schema fields.
    *   Providing known distinct values (from a cache or the mapping data) significantly improves the LLM's ability to map query terms (like names) accurately to the correct fields (e.g., "Ron Lim" -> `Family`).
    *   **Managing "Known Values" in Prompts:** When providing lists of known values (e.g., all friends, all tags) directly in the query analysis prompt:
        *   **Signal vs. Noise:** Excessively long lists can act as "noise," making it harder for the LLM to identify the correct specific value mentioned in the user's query, even if that value is present in the list. The desired "signal" (the specific value) gets diluted.
        *   **Token Limits & Focus:** Very long lists also consume more tokens and can strain the LLM's ability to focus on all parts of the prompt effectively.
        *   **Dynamic Sizing:** A strategy is to dynamically adjust the number of examples shown for "known values" based on the field type. For instance, show a smaller number of examples (e.g., 10-20) for fields with potentially very long lists of unique values (like `Friends`), especially if alphabetical sorting can bring common matches to the forefront. For other fields (like `Tags`), a larger number of examples might be more beneficial. This helps balance providing enough context without overwhelming the model.

## 5. Data Handling & Consistency

*   **Schema Awareness:** Code must correctly handle the data types defined in the Notion schema (e.g., multi-select lists).
*   **Data Flow:** Ensure that all necessary data fields (including all relevant metadata and full content) are correctly propagated from the initial export (`cli.py` -> `transformers.py`), through embedding/mapping (`build_index.py`), and are available for context retrieval (`cli.py` -> `handle_query`). Omitting fields early breaks downstream steps.
*   **Case Sensitivity:** Inconsistent key casing (e.g., `family` vs. `Family`) between data storage and data retrieval/processing is a common source of bugs. Maintain consistency.
*   **Caching:** Caching extracted distinct metadata values (e.g., in `metadata_cache.json`) generated during the index build avoids redundant processing during query time.

## 6. Notion Specifics

*   **Value Aggregation:** The Notion API doesn't provide a direct way to get all unique values for a multi-select/relation property database-wide. This list must be aggregated by iterating through all page data (e.g., during the `build_index` process).
*   **Export Filtering:** Filtering exports by `last_edited_time` (useful for sync) means the `source_file` (e.g., `2024-06.json`) may contain entries whose `Date` property is from a different month (e.g., `2024-05-16`). This is expected behavior based on the filter used.

## 7. Cost & Model Choices

*   **Embeddings:** Batching embedding requests is generally faster and potentially more cost-effective than individual calls.
*   **Query Analysis:** Using a smaller, cheaper model (`gpt-4o-mini`) for the structured query analysis is efficient.
*   **Final Answer:** Using a more powerful (and expensive) model (`gpt-4o`) for the final synthesis step provides higher accuracy, but cost scales with the amount of context (`TOP_K` * content length) provided. There's a trade-off between context size, final model cost, and answer quality.

## Debugging CLI vs. Backend RAG Discrepancy

During the development of the web UI, a significant discrepancy was observed between the RAG results returned by the command-line interface (`cli.py`) and the FastAPI backend (`backend/rag_query.py`) for the same query.

The CLI consistently provided the correct answer and sources, while the backend initially failed to find the answer or returned incorrect/incomplete source information. The debugging process involved several steps:

1.  **Initial Check & Key Mismatch:** Compared backend logs (`INFO - Successfully processed query. Returning answer and X sources.`) with the UI output. Initial runs showed the backend returning 0 sources or sources titled "Untitled Entry". This pointed towards an issue in how the backend retrieved source metadata (`title`, `page_id`, `entry_date`) from the `index_mapping.json` data. Analysis of `build_index.py` confirmed the correct keys (`page_id`, `title`, `entry_date`), and `backend/rag_query.py` was updated accordingly in both the `load_rag_data` pre-parsing step and the context retrieval step.

2.  **Embedding Model Mismatch:** Even after fixing keys, the backend returned the *wrong* set of sources compared to the CLI. Investigation revealed that the index was built using `text-embedding-ada-002` (`build_index.py`), but the backend was generating query embeddings using `text-embedding-3-small` (`backend/rag_query.py`). This mismatch caused the similarity search to retrieve irrelevant documents. The backend's `OPENAI_EMBEDDING_MODEL` was changed to `text-embedding-ada-002` to align with the index.

3.  **CORS Preflight Issue:** After aligning embeddings, the frontend started receiving `400 Bad Request` errors on `OPTIONS` requests to the backend API. This indicated a CORS preflight failure. The `CORSMiddleware` configuration in `backend/main.py` was missing `"OPTIONS"` in its `allow_methods` list. Adding it resolved the browser's preflight check.

4.  **Prompt Discrepancy:** With the technical issues resolved, the backend now returned the correct *answer* but the formatting differed from the CLI output (e.g., lacking the inline Notion link). Comparing the final answer generation prompts revealed that the CLI's prompt included specific Markdown formatting instructions (`Journal Entry: [Title](URL)`) which were missing from the backend's prompt. The backend's `final_system_prompt` and `final_user_prompt` were updated to match the CLI's for consistent output formatting.

**Key Takeaways:**

*   **Consistency is Crucial:** Ensure absolute consistency between indexing and querying processes, especially regarding embedding models and data structure keys (mapping file).
*   **Logging Intermediate Steps:** Logging the retrieved context (titles, IDs) *before* sending it to the final LLM was vital for pinpointing the embedding mismatch.
*   **Check CORS Thoroughly:** Remember that browser preflight `OPTIONS` requests require explicit allowance in backend CORS configurations.
*   **Prompt Engineering Matters:** Subtle differences in prompts, especially formatting instructions, can significantly impact the LLM's output structure.
*   **Separate Concerns:** While refactoring from CLI to backend, meticulously verify that all relevant logic (including constants and prompt details) is transferred correctly. 