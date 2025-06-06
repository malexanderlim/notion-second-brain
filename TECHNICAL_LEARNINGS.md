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
    *   **Targeted Search:** Pinecone's query capabilities allow performing vector similarity search combined with metadata filters, ensuring retrieved results match hard criteria *and* semantic relevance based on the pre-filtered candidate set.
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
*   **Caching:** Caching extracted distinct metadata values (e.g., in `metadata_cache.json` on GCS) generated during the index build avoids redundant processing during query time.

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

## 8. Vercel Monorepo Deployment Learnings (Frontend + Python Backend)

Deploying a monorepo with a Vite frontend (in `frontend/`) and a Python FastAPI backend (in `backend/`) to Vercel presented significant challenges, primarily around build configuration, path aliasing for the frontend, and Vercel's output directory expectations. This was further impacted by the migration to Google Cloud Storage (GCS) for backend data artifacts, which ultimately simplified parts of the Vercel deployment by reducing reliance on Vercel's build environment for data persistence.

*   **`.gitignore` is Critical:**
    *   **Issue:** An overly broad rule (`lib/`) in the root `.gitignore` unintentionally excluded the `frontend/src/lib/` directory (containing `utils.ts`) from Git tracking. This led to `ENOENT: no such file or directory` errors during Vercel's `vite build` step, as the aliased module (`@/lib/utils`) was correctly resolved to a path, but the file itself was not present in the build environment.
    *   **Learning:** Always meticulously audit `.gitignore` to ensure all necessary source code subdirectories are included, especially when adding new sub-projects or shared libraries within a monorepo.

*   **Vite Path Alias Resolution on Vercel:**
    *   **Issue:** Even with `frontend/src/lib/utils.ts` correctly tracked by Git, `vite build` on Vercel initially failed to resolve the `@/lib/utils` alias defined in `frontend/tsconfig.app.json` (`baseUrl: "."`, `paths: { "@/*": ["./src/*"] }`).
    *   **Attempts & Learnings:**
        1.  `vite-tsconfig-paths` plugin: This is the standard solution. Initial attempts without explicitly setting the `projects` option in `vite.config.ts` failed. Setting `tsconfigPaths({ projects: ['./tsconfig.app.json'] })` also failed, likely due to the then-hidden `.gitignore` issue masking true resolution problems.
        2.  Manual `resolve.alias` in `vite.config.ts`: Various attempts to manually define `{ find: '@', replacement: path.resolve(__dirname, './src') }` or even hyper-specific aliases like `{ find: '@/lib/utils', replacement: path.resolve(__dirname, './src/lib/utils.ts') }` also hit roadblocks, often manifesting as either `ENOENT` (if Vite couldn't resolve the path to a file with an extension) or "Rollup failed to resolve import" (if the alias itself wasn't recognized).
        3.  **Underlying Cause (Revealed after `.gitignore` fix):** The primary alias problem was the missing file. Once the file was present, `vite-tsconfig-paths` (with `projects` explicitly set) became the correct and working solution for alias resolution within the Vite build.

*   **Vercel Build Configuration (`vercel.json`) for Monorepos:**
    *   **Requirement:** For a monorepo with distinct frontend and backend builds, `vercel.json` is essential to define multiple `builds`. Relying on UI framework presets alone is insufficient.
    *   **Frontend Build:** Using `"src": "frontend/package.json"` and `"use": "@vercel/static-build"` is appropriate. The `config.distDir` for this build tells `@vercel/static-build` where its output is *relative to its source directory* (e.g., `distDir: "dist"` means `frontend/dist`).

*   **The "No Output Directory named 'dist' found" Error:**
    *   **Issue:** After Vite successfully built the frontend to `frontend/dist/`, Vercel threw an error: "No Output Directory named 'dist' found".
    *   **Cause:** Vercel's platform, especially with the "Other" framework preset, often defaults to looking for a *root-level* directory named `dist` (or `public`) for the final static assets, even if `builds[*].config.distDir` correctly points to a nested directory for an individual build step.
    *   **Learning & Solution (The "Move Output" Strategy):**
        1.  **Root `package.json`:** Create a `package.json` at the monorepo root.
        2.  **`vercel-build` Script:** In this root `package.json`, define a `"vercel-build"` script (e.g., `"cd frontend && npm install && npm run build && cd .. && rm -rf dist && mv frontend/dist dist"`). This script handles the frontend build and then moves its output (`frontend/dist`) to a `dist` directory at the monorepo root.
        3.  **Update `vercel.json`:**
            *   Change the frontend build `"src"` to point to the root `"package.json"`.
            *   Change its `config.distDir` to `"dist"` (as the `vercel-build` script now creates this at the root).
        4.  **Vercel Project UI Settings:** Set the "Output Directory" in the UI to `dist` (or clear the override and let it default, as Vercel often finds a root `dist`).
        5.  **Routing:** Ensure `routes` in `vercel.json` for SPA fallback now point relative to the new root `dist` (e.g., `"dest": "/index.html"`).
    *   **Rationale:** This strategy makes the project structure conform to Vercel's apparent default expectation for a root-level output directory when the "Other" preset is used, resolving the platform-level error.

*   **Iterative Debugging is Key:** Complex deployment issues often require isolating variables: first ensuring files are present (fixing `.gitignore`), then tackling alias resolution within the build tool (Vite), and finally addressing platform-level output directory expectations (Vercel). Each error message, however frustrating, provides a clue to the next layer of the problem. 

## 9. Cloud Migration: Google Cloud Storage (GCS) and Pinecone Integration

To enhance scalability, reliability, and prepare for serverless deployment (especially on Vercel), the project underwent a significant migration from local file storage and FAISS to Google Cloud Storage (GCS) for data exports and index artifacts, and Pinecone as the managed vector database.

**Benefits & Rationale:**
*   **Scalability & Reliability:** GCS provides a robust and scalable solution for storing growing volumes of monthly Notion JSON exports and the critical index artifacts (`index_mapping.json`, `metadata_cache.json`, `schema.json`, `last_entry_update_timestamp.txt`).
*   **Decoupling & Deployment:** Storing these artifacts on GCS decouples the application's data persistence from the local filesystem of the development or deployment environment. This is particularly beneficial for ephemeral environments like Vercel, where a persistent filesystem is not readily available for large, frequently updated index files.
*   **Managed Vector DB (Pinecone):** Migrating from a local FAISS index to Pinecone (a managed vector database service) offloads the complexities of index management, serving, and scaling. This simplifies the RAG backend and ensures high availability and performance for vector searches.

**Implementation Details & Learnings:**

*   **Configuration (`config.py` and `.env`):**
    *   New environment variables were introduced: `GCS_BUCKET_NAME`, `GCS_EXPORT_PREFIX`, `GCS_INDEX_ARTIFACTS_PREFIX`, `PINECONE_API_KEY`, and `PINECONE_INDEX_NAME`.
    *   Crucially, `config.py` was updated to load these variables. An initial `AttributeError: module 'notion_second_brain.config' has no attribute 'GCS_BUCKET_NAME'` was encountered and resolved by ensuring these variables were correctly defined in `.env` and loaded by `config.py`.

*   **Command Line Interface (`cli.py`):**
    *   **Schema Export:** The `python cli.py --schema` command was modified to directly upload the retrieved `schema.json` to the specified GCS bucket and prefix (`GCS_INDEX_ARTIFACTS_PREFIX`).
    *   **Data Export:** The `python cli.py --export` (and `--export-month`) commands were updated to save the monthly JSON export files directly to GCS under `GCS_EXPORT_PREFIX`, instead of locally.

*   **Index Building (`build_index.py`):**
    *   **Data Source:** The script now reads the monthly exported JSON files directly from GCS (`GCS_EXPORT_PREFIX`).
    *   **Vector Storage:** FAISS logic was entirely replaced with Pinecone integration. Embeddings are generated and then upserted in batches to the configured Pinecone index, using the `page_id` as the vector ID.
    *   **Artifact Storage:** All generated artifacts – `index_mapping.json`, `metadata_cache.json`, `last_entry_update_timestamp.txt`, and `schema.json` (if a local copy is updated/present) – are now uploaded to GCS under `GCS_INDEX_ARTIFACTS_PREFIX`.
    *   **Debugging `UnboundLocalError`:** A persistent `UnboundLocalError: cannot access local variable 'texts_to_embed_batch' where it is not associated with a value` (and similarly for `metadata_for_pinecone_batch`) occurred during the main processing loop. This was due to these batch lists being initialized outside the loop and potentially not being re-initialized if the first entry in `all_entries_data` was skipped (e.g., due to empty content). The error was resolved by explicitly re-initializing `texts_to_embed_batch = []` and `metadata_for_pinecone_batch = []` immediately before the `for entry in all_entries_data:` loop. This ensured they were always defined for the first iteration that actually appended to them.

*   **Backend RAG Initializer (`api/rag_initializer.py`):**
    *   The `load_rag_data` function was significantly updated to download `index_mapping.json`, `metadata_cache.json`, and `schema.json` from their respective GCS locations during application startup.
    *   It also initializes the Pinecone client for querying.

**Overall Impact:**
*   The data pipeline is now more robust and suitable for cloud deployment.
*   Operational overhead for managing local index files is eliminated.
*   The system is better prepared for scaling both data volume and query load. 

## 10. Debugging Voice Transcription (Frontend & Backend - OpenAI Whisper)

Troubleshooting voice transcription issues, particularly discrepancies between desktop and mobile (Chrome), involved several iterations and highlighted key aspects of both `MediaRecorder` behavior and OpenAI Whisper API interactions.

**Initial Problem:**
*   Voice queries worked on desktop (Chrome) but failed on mobile (Chrome).
*   Failures manifested as:
    1.  HTTP 400 errors from the backend's `/api/transcribe` endpoint, specifically "Invalid file format" errors from the Whisper API.
    2.  No errors, but completely incorrect transcriptions (e.g., English speech transcribed as Korean).

**Key Learnings & Solutions:**

1.  **Whisper API Filename Extension Requirement:**
    *   **Observation:** The OpenAI Whisper API heavily relies on the filename extension provided in the `file` parameter of its `transcriptions.create` method to correctly interpret the audio format, especially when raw bytes are sent.
    *   **Issue:** Initially, the frontend hardcoded the filename to `"voice_query.webm"`. If a mobile browser sent audio data in a different format (e.g., `audio/mp4`), this mismatch between the stated `webm` extension and the actual `mp4` content likely caused the "Invalid file format" error, even if MP4 is a supported format.
    *   **Solution Iteration 1 (Dynamic Extension):** Modify the frontend to dynamically determine the file extension based on the `MediaRecorder` blob's `mimeType` (e.g., `audio/mp4` -> `voice_query.mp4`). This aligns with Whisper API documentation.

2.  **Whisper API Language Misidentification:**
    *   **Observation:** Even with a (presumably) correct filename extension, mobile transcriptions were sometimes wildly inaccurate (e.g., English to Korean).
    *   **Issue:** Whisper was likely failing to auto-detect the language correctly from the mobile audio stream.
    *   **Solution:** Explicitly set the `language="en"` parameter in the backend call to `openai_client.audio.transcriptions.create`. This immediately resolved the language misidentification problem.

3.  **Standardizing `MediaRecorder` Output:**
    *   **Observation:** `MediaRecorder` output can vary significantly between browsers and operating systems (e.g., desktop Chrome might default to `audio/webm;codecs=opus`, while mobile Chrome on some OSes might produce `audio/mp4` or `audio/m4a`).
    *   **Strategy:** To improve consistency and leverage a well-supported, high-quality format:
        *   Attempt to force `MediaRecorder` to use `audio/webm;codecs=opus` with a specified bitrate (e.g., 128000 bps) in the frontend.
        *   Use `MediaRecorder.isTypeSupported()` to check if the preferred format is available.
        *   Implement fallbacks: if `audio/webm;codecs=opus` isn't supported, try generic `audio/webm`. If that also fails, allow the browser to use its default format.
        *   Crucially, the frontend should still use the *actual* `mimeType` reported by the `MediaRecorder` instance after recording (from `mediaRecorderRef.current.mimeType`) to create the `Blob` and determine the filename extension for the backend. This ensures that even if forcing fails, the backend receives an accurately described file.

4.  **Frontend-Backend Data Flow for Audio:**
    *   **Frontend:**
        *   Use `MediaRecorder` (attempting to force `audio/webm;codecs=opus`).
        *   On `onstop`, create a `Blob` using the *actual* MIME type from `mediaRecorder.mimeType`.
        *   Determine the correct file extension from this actual MIME type.
        *   Append the `Blob` to `FormData` with a filename that includes this correct extension (e.g., `voice_query.mp4`, `voice_query.webm`).
    *   **Backend (`/api/transcribe`):
        *   Receive the `UploadFile`.
        *   Pass `(file.filename, file_bytes)` and `language="en"` to `openai_client.audio.transcriptions.create`.

5.  **Diagnostic Logging:**
    *   **Frontend:** Log the `preferredOptions.mimeType` being attempted, whether it's supported, the `actualMimeType` from `mediaRecorder.mimeType` in the `onstop` event, and the `audioBlob.type` and `audioBlob.size`.
    *   **Backend:** Log `file.filename` and `file.content_type` (from `UploadFile`) and the client IP upon receiving a request at `/api/transcribe`. This helps verify what the server is actually receiving from different clients.

**Summary of Effective Fix:** The combination of explicitly setting `language="en"` on the backend and the frontend attempting to force `audio/webm;codecs=opus` while ensuring the submitted filename *always* matches the `MediaRecorder`'s actual output `mimeType` proved to be the most robust solution for achieving consistent and accurate voice transcriptions across desktop and mobile platforms. 