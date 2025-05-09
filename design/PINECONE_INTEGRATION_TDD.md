# Technical Design Document: Pinecone Integration

## 1. Metadata

*   **Title:** Integrate Pinecone for Vector Storage and Search
*   **Status:** Proposed
*   **Author(s):** AI Assistant (in collaboration with User)
*   **Date:** 2024-07-27
*   **Related Docs:** `design/MVP_RAG_TDD.md`, `TASKS.md`

## 2. Introduction / Motivation / Problem

The current RAG (Retrieval Augmented Generation) system for the "Notion Second Brain" project uses a local FAISS CPU-based index for vector storage and similarity search. While functional for local development, this approach has led to significant deployment issues on Vercel, specifically exceeding the 250MB unzipped serverless function size limit. The `faiss-cpu` library and its dependencies are substantial.

This TDD proposes migrating the vector storage and search capabilities to Pinecone, a managed vector database service. This change is primarily motivated by the need to reduce the deployment bundle size to successfully deploy the Python backend on Vercel. Additionally, Pinecone offers scalability, simplified management of the vector index, and potential performance benefits compared to a local FAISS index running in a constrained serverless environment.

## 3. Goals

*   Replace the local FAISS index with Pinecone for all vector storage and similarity search operations.
*   Successfully deploy the Python backend (FastAPI application) to Vercel without exceeding the serverless function size limits.
*   Ensure the RAG functionality remains intact, providing relevant context from Notion documents based on user queries.
*   Reduce the overall complexity of managing vector index files within the deployment package.
*   Maintain or improve query latency and search accuracy where possible.
*   Ensure the `build_index.py` script correctly populates the Pinecone index.
*   Ensure the query endpoint correctly retrieves vectors and associated metadata using Pinecone.

## 4. Non-Goals

*   Implementing advanced Pinecone features such as hybrid search (sparse-dense vectors), namespaces, or complex metadata filtering strategies beyond what is necessary for replicating current functionality in this initial iteration.
*   Changing the existing embedding model (e.g., OpenAI's text-embedding models) or the core logic for generating these embeddings.
*   A complete overhaul of the existing RAG query flow beyond adapting it to use the Pinecone client SDK instead of FAISS.
*   Implementing a real-time, continuous sync mechanism with Pinecone in this phase (the existing batch `build_index.py` script will be adapted).
*   Significant UI changes beyond ensuring the existing frontend continues to work with the Pinecone-backed RAG.

## 5. Proposed Design / Technical Solution

### 5.1. Pinecone Setup
1.  **Account and Index Creation:**
    *   The user will create a Pinecone account (if not already done).
    *   A new Pinecone index will be created (e.g., `notion-second-brain`).
    *   **Configuration:**
        *   **Dimensions:** Match the output dimensions of the current embedding model (e.g., 1536 for OpenAI `text-embedding-ada-002`, or 3072 for `text-embedding-3-large`). This needs to be confirmed based on the project's current embedding model.
        *   **Metric:** `cosine` (to match FAISS `IndexFlatL2` behavior for normalized embeddings).
        *   **Pod Type:** Start with a cost-effective pod type (e.g., `s1.x1` or `p1.x1` if available on free/starter tier) suitable for the expected data size and query load.
2.  **API Key and Environment Variables:**
    *   Obtain the Pinecone API Key and Environment identifier.
    *   These will be stored as environment variables: `PINECONE_API_KEY` and `PINECONE_ENVIRONMENT`.
    *   These variables must be added to `.env` for local development and configured in the Vercel deployment environment settings.

### 5.2. Dependency Management
1.  **`requirements.txt` (Root):**
    *   Add `pinecone-client` (e.g., `pinecone-client~=3.0.0`).
    *   Remove `faiss-cpu`.
2.  **`.vercelignore` / `vercel.json`:**
    *   Ensure no FAISS-related files or large, unnecessary model files are included in the deployment bundle. The `.vercelignore` should already be handling general exclusions.

### 5.3. Backend Code Modifications (`api/` directory)

#### 5.3.1. `api/rag_initializer.py`
1.  **Client Initialization:**
    *   Modify `initialize_clients()` or add a new function `initialize_pinecone_client()`:
        ```python
        # Example snippet
        from pinecone import Pinecone as PineconeClient # Ensure correct import alias
        
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        pinecone_environment = os.getenv("PINECONE_ENVIRONMENT") # This might be part of Pinecone() in v3+
        pinecone_index_name = os.getenv("PINECONE_INDEX_NAME", "notion-second-brain") # Add to env vars

        if not pinecone_api_key:
            logger.error("PINECONE_API_KEY not found in environment variables.")
            # Potentially raise an error or handle appropriately
            return None 
        
        # For Pinecone client v3.x and later:
        pc = PineconeClient(api_key=pinecone_api_key)
        # For older versions, environment might be needed in constructor:
        # pc = PineconeClient(api_key=pinecone_api_key, environment=pinecone_environment)
        
        # global pinecone_index_instance (or return and store elsewhere)
        # pinecone_index_instance = pc.Index(pinecone_index_name)
        # logger.info(f"Pinecone client initialized and connected to index '{pinecone_index_name}'.")
        # return pinecone_index_instance
        ```
    *   The Pinecone index instance should be made available globally or passed to where it's needed (similar to current OpenAI/Anthropic clients).
2.  **Data Loading (`load_rag_data`):**
    *   Remove FAISS index loading logic (`faiss.read_index(...)`).
    *   The `index_mapping.json` (mapping document metadata to vector IDs) will remain crucial. Its structure might need to ensure IDs are strings, as Pinecone prefers string IDs for vectors.
    *   The "index" object loaded here will now refer to the Pinecone index client instance.

#### 5.3.2. `api/retrieval_logic.py`
1.  **Semantic Search (`perform_semantic_search` or new `perform_pinecone_search`):**
    *   Replace FAISS `index.search()` calls with Pinecone `index.query()`.
    *   The `query()` method will take arguments like:
        *   `vector`: The query embedding.
        *   `top_k`: The number of results to retrieve.
        *   `(Optional)` `filter`: A dictionary for metadata filtering directly in Pinecone (see Future Work). For MVP, pre-filtering IDs from `index_mapping.json` might be simpler if we can't easily translate existing filter logic.
        *   `include_metadata`: Can be true if we store relevant metadata directly in Pinecone alongside vectors.
    *   **Input:** Query embedding, `top_k`, (optional) list of pre-filtered document IDs.
    *   **Process:**
        *   If a list of pre-filtered document IDs is provided (from metadata filtering on `index_mapping.json`), use Pinecone's `fetch(ids=[...])` if the goal is just to get these specific vectors, or use `query(..., id_filter=['$in', list_of_ids])` if further semantic ranking on this subset is needed. *Correction*: `fetch` retrieves by ID. For semantic search on a subset, one would typically use metadata filters in the `query` or, if not feasible, query broadly and then filter results.
        *   If no pre-filtered IDs, query normally:
            ```python
            # Example snippet
            # query_response = pinecone_index_instance.query(
            #     vector=query_embedding,
            #     top_k=top_k_value,
            #     include_values=False, # Typically don't need to retrieve the vectors themselves
            #     include_metadata=True # If metadata is stored in Pinecone
            # )
            # matches = query_response.get('matches', [])
            # retrieved_ids = [match['id'] for match in matches]
            # scores = {match['id']: match['score'] for match in matches} 
            ```
    *   **Output:** A list of document IDs and their similarity scores, compatible with the existing downstream logic that uses these IDs to fetch full content from `index_mapping.json`.

### 5.4. Indexing Script (`build_index.py`)
1.  **Initialization:** Initialize Pinecone client as in `api/rag_initializer.py`.
2.  **Vector Upsertion:**
    *   Remove FAISS index creation (`faiss.IndexFlatL2(d)`).
    *   Remove FAISS index saving (`faiss.write_index(...)`).
    *   After generating embeddings for documents:
        *   Prepare vectors for Pinecone: Each vector needs a unique string ID and the vector values. Metadata can optionally be stored with the vector in Pinecone.
            ```python
            # Example format for upserting
            # vectors_to_upsert = []
            # for i, (entry_id, embedding, metadata_dict) in enumerate(processed_entries):
            #     # Ensure entry_id is a string, unique, and consistent with index_mapping.json
            #     # metadata_dict could be a subset of what's in index_mapping.json if useful for Pinecone filtering
            #     vectors_to_upsert.append({
            #         "id": str(entry_id), # Ensure string ID
            #         "values": embedding,
            #         # "metadata": metadata_dict # Optional: store metadata here
            #     })
            ```
        *   Upsert vectors to Pinecone in batches to avoid request size limits and improve efficiency. Pinecone's `upsert` method handles batching well.
            ```python
            # Example snippet (inside loop or after collecting all vectors)
            # from pinecone.core.client.models import Vector # If using specific model types
            #
            # batch_size = 100 # Configurable
            # for i in range(0, len(vectors_to_upsert), batch_size):
            #     batch = vectors_to_upsert[i:i + batch_size]
            #     pinecone_index_instance.upsert(vectors=batch) 
            #     logger.info(f"Upserted batch {i//batch_size + 1} to Pinecone.")
            ```
3.  **`index_mapping.json`:**
    *   This file remains essential. It maps the vector IDs (now also Pinecone vector IDs) to the full document metadata (title, content, URL, original Notion properties).
    *   Ensure IDs used in `index_mapping.json` are consistent (and stringified) with those used in Pinecone.
4.  **Rebuild Logic (`--force-rebuild`):**
    *   If `--force-rebuild` is specified, the script should:
        1.  Optionally, delete all vectors from the Pinecone index to ensure a clean slate.
            `pinecone_index_instance.delete(delete_all=True)` (Use with caution).
        2.  Proceed with fetching data, embedding, and upserting all vectors.
    *   If not forcing a rebuild, the script might be adapted for incremental updates (see Future Work), or it might just re-upsert everything (Pinecone `upsert` will overwrite vectors with the same ID).

### 5.5. Configuration and Environment
*   Update `.env.example` with `PINECONE_API_KEY`, `PINECONE_ENVIRONMENT`, and `PINECONE_INDEX_NAME`.
*   Ensure these variables are set in Vercel project settings.
*   The Pinecone index dimension must be configured correctly in Pinecone UI to match the embedding model dimension.

## 6. Alternatives Considered

*   **Other Managed Vector Databases (Weaviate, ChromaDB Cloud, Qdrant Cloud, etc.):**
    *   *Pros:* Similar benefits in terms of offloading vector management and reducing bundle size.
    *   *Cons:* Each has its own API, client libraries, and pricing. Pinecone is well-established, known for ease of use in RAG, and has a lightweight Python client.
    *   *Reason for not choosing:* Pinecone was selected due to its popularity, good documentation, and perceived simplicity for this type of RAG application. The user also previously mentioned Pinecone as a potential solution.
*   **Further Optimizing FAISS/Dependencies:**
    *   *Pros:* Avoids introducing a new external service.
    *   *Cons:* Extensive attempts to reduce bundle size by excluding files and modifying `vercel.json` proved insufficient. The core `faiss-cpu` library and its native components are inherently large.
    *   *Reason for not choosing:* This path was exhausted and did not solve the Vercel size limit issue.
*   **Rewriting Backend in Node.js:**
    *   *Pros:* Node.js serverless functions might have different size characteristics or better tooling for tree-shaking large dependencies if a JS vector library was used.
    *   *Cons:* Would require a complete rewrite of the existing Python backend (RAG logic, API endpoints, Notion data processing), which is a significant effort.
    *   *Reason for not choosing:* A full rewrite is too high an effort compared to replacing one component (vector store) in the existing Python backend.

## 7. Impact / Risks / Open Questions

### 7.1. Impact
*   **Deployment Size:** Expected to be significantly reduced, resolving the Vercel 250MB limit.
*   **Dependencies:** `faiss-cpu` removed; `pinecone-client` added.
*   **Infrastructure:** Introduces a dependency on the external Pinecone service.
*   **Indexing Process:** Will involve network calls to Pinecone for upserting vectors. Indexing time might increase depending on network speed and batching efficiency.
*   **Query Process:** Will involve network calls to Pinecone for similarity search. Query latency might be affected (could be better or worse than local FAISS in a serverless environment, needs testing).
*   **Cost:** Pinecone is a commercial service. While a free tier exists, usage beyond its limits will incur costs. This needs to be monitored.
*   **Local Development:** Developers will need Pinecone credentials and an active index for local testing of RAG features.

### 7.2. Risks
*   **Latency:** Network latency to Pinecone could impact query response times. The chosen Pinecone region should ideally be close to the Vercel function region.
*   **API Rate Limits/Quotas:** Pinecone API usage (upserts, queries) is subject to rate limits and quotas based on the subscription tier. Batching and efficient client usage are important.
*   **Configuration Errors:** Incorrect API keys, environment, index name, or dimension mismatch can lead to runtime failures.
*   **Data Migration/Consistency:** Ensuring all existing embeddings are correctly upserted to Pinecone with consistent IDs matching `index_mapping.json` is critical.
*   **Vendor Lock-in:** Migrating to a managed service introduces a degree of vendor lock-in, though the core embedding logic remains independent.
*   **Security:** API keys must be managed securely (environment variables, Vercel secrets).

### 7.3. Open Questions
1.  **Pinecone Index Configuration:**
    *   What is the exact dimension of the embeddings currently used? (Must verify, e.g., 1536 for `text-embedding-ada-002`, 3072 for `text-embedding-3-large`, 768 for `text-embedding-3-small`).
    *   Which Pinecone region is optimal for Vercel deployment? (User to decide based on their Vercel region).
    *   What pod type and size are appropriate for the free/starter tier and expected load?
2.  **Vector ID Strategy:**
    *   Are the current IDs in `index_mapping.json` already strings and unique? If not, they need to be adapted (e.g., stringified integers, UUIDs). Pinecone requires string IDs.
3.  **Metadata Storage in Pinecone:**
    *   Should any metadata (e.g., document title, date, source URL) be stored directly with the vectors in Pinecone to allow for metadata filtering at query time via Pinecone's capabilities?
    *   *Initial approach:* Rely on `index_mapping.json` for metadata lookup post-retrieval.
    *   *Consideration for future:* Storing key filterable fields in Pinecone could optimize queries but adds complexity to upsertion.
4.  **Batch Upsert Size:** What is the optimal batch size for `build_index.py` when upserting to Pinecone? (Start with a default like 100, make it configurable).
5.  **Error Handling & Retries:** What specific retry mechanisms should be implemented for Pinecone API calls (e.g., for transient network issues)? (The `pinecone-client` might have some built-in retries).
6.  **Impact on `DATA_SOURCE_MODE=remote` and `vercel-blob`:** The original `vercel-blob` error might have been a red herring or related to an incomplete environment setup. With FAISS removed, this specific error should not reappear due_to_FAISS. If other parts of the app use `vercel-blob` and `DATA_SOURCE_MODE=remote`, that needs to be correctly configured independently. This TDD focuses on the vector store.

## 8. (Optional) Implementation Plan / Phases

This will be detailed in `TASKS.md`. Key phases include:
1.  Pinecone Account & Index Setup.
2.  Dependency Updates & Configuration.
3.  Refactor `build_index.py`.
4.  Refactor `api/rag_initializer.py` and `api/retrieval_logic.py`.
5.  Testing (Local and Staging/Preview on Vercel).
6.  Vercel Production Deployment.

## 9. (Optional) Future Work / Follow-on

*   **Advanced Pinecone Metadata Filtering:** Explore using Pinecone's metadata filtering capabilities directly in queries to potentially speed up filtered searches, instead of pre-filtering IDs from `index_mapping.json`.
*   **Incremental Updates to Pinecone:** Enhance `build_index.py` or create a new sync script for more efficient incremental updates to the Pinecone index (upserting new/changed documents, deleting removed ones) rather than always doing a full rebuild or full upsert.
*   **Pinecone Namespaces:** If multiple datasets or versions of the index are needed, explore using Pinecone namespaces.
*   **Performance Monitoring & Optimization:** Implement monitoring for Pinecone query latency and costs. Optimize `top_k` and other query parameters.
*   **Hybrid Search:** Investigate Pinecone's support for hybrid search (combining dense vector search with sparse vector search like BM25) if keyword-based matching becomes important.
*   **Error Handling and Resilience:** Add more robust error handling and retry logic for Pinecone interactions. 