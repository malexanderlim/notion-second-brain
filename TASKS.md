# Notion Second Brain - Phase 1 MVP RAG Implementation

Tracking progress for the initial MVP RAG Demo and subsequent full index build.

## Completed Tasks

- [x] **1.1 Setup Project Structure**
  - [x] Create basic project scaffolding (`notion_second_brain/`, `tests/`, `docs/`)
  - [x] Create `requirements.txt`
  - [x] Create `.gitignore` for Python
  - [x] Create `.env.example`
  - [x] Restructure project to match README standard (package `notion_second_brain/`)
- [x] **1.2 Notion API Integration**
  - [x] Create `config.py` for loading environment variables (Notion Token, DB ID)
  - [x] Add `python-dotenv` to `requirements.txt`
  - [x] Create `notion/api.py` module
  - [x] Implement `NotionClient` class with base URL, headers, session
  - [x] Add `requests` to `requirements.txt`
  - [x] Implement `_request` helper method with error handling
  - [x] Implement `test_connection` method
  - [x] Implement `query_database` method with pagination
  - [x] Implement `retrieve_block_children` method with pagination
- [x] **1.3 Data Extraction and Processing**
  - [x] Create `notion/extractors.py` module
  - [x] Implement `extract_text_from_rich_text_array` helper
  - [x] Implement `extract_text_from_block` for various block types
  - [x] Implement `extract_page_content` to combine block texts
  - [x] Create `processing/filters.py` module
  - [x] Implement `get_entry_date` helper function
  - [x] Implement date-based filtering functions (`day`, `week`, `month`, `year`, `range`)
- [x] **1.4 JSON Export**
  - [x] Create `storage/json_storage.py` module
  - [x] Implement `generate_filename` for time periods
  - [x] Implement `save_entries_to_json` function
- [x] **3.1 Command Line Interface (Basic)**
  - [x] Create `cli.py` module in root
  - [x] Implement argument parsing (`argparse`) for basic options (period, dates, output, test, verbose)
  - [x] Implement main script logic: initialize client, query pages, retrieve blocks, extract content, filter, save to JSON.
  - [x] Optimize `cli.py` to use Notion API filters for queries (instead of local filtering)
  - [x] Transform raw Notion page object to simpler JSON structure before saving
- [x] **Phase 1 Refinements & Setup:**
  - [x] Initialize Git repository (`git init`)
  - [x] Create project virtual environment (e.g., `python3 -m venv venv`)
  - [x] Retrieve Notion database schema to identify all property types
  - [x] Add handlers for all used property types in `processing/transformers.py` (Verified existing code handles required types)
  - [x] Add basic logging configuration (Centralized in `config.py`)
  - [x] Implement better error handling in `cli.py` and core modules (API rate limit retry added)
- [x] **Project Setup & Documentation:**
  - [x] Create `design/` directory for technical design documents
  - [x] Create `design/MVP_RAG_TDD.md` outlining the MVP RAG plan
- [x] **Implement Hybrid RAG (Metadata Filtering with Dynamic Values + Semantic Search):**
  - [x] **Extract Distinct Metadata Values:**
    - [x] In `cli.py` (`handle_query`), after loading `index_mapping.json`, implement logic to iterate through it.
    - [x] Extract unique string values for key filterable list/multi-select fields (e.g., `Family`, `Friends`, `Tags`). Store efficiently (e.g., dict of sets).
    - [x] **Caching:** Store/load these distinct values in `build_index.py` and `cli.py` using `metadata_cache.json`.
  - [x] **Query Analysis:**
    - [x] Update `analyze_query_for_filters` prompt to accept and utilize distinct values.
    - [x] Ensure LLM correctly maps query terms (names, tags) using distinct values.
    - [x] Ensure LLM correctly extracts date ranges.
  - [x] **Pre-Filtering:**
    - [x] In `cli.py` (`handle_query`), use the parsed filters to pre-filter `mapping_data` based on dates and field values.
    - [x] Handle OR logic correctly for name filters across 'Family'/'Friends'.
  - [x] **Targeted Semantic Search:**
    - [x] Use FAISS `IDSelectorBatch` (or similar) to restrict `index.search` to the pre-filtered indices.
    - [x] Handle the case where pre-filtering yields zero results.
  - [x] **Final Answer Generation:**
    - [x] Use the results from the targeted search to retrieve context (full content).
    - [x] Send context + query to the final LLM (`gpt-4o`).
    - [x] Refine final prompt to handle specific inferences (e.g., tagged = seen) and formatting (dates, links).
- [x] Create `TECHNICAL_LEARNINGS.md` summarizing RAG development insights.

## MVP RAG - In Progress Tasks (Midnight Sprint!)

- [x] **Setup & Dependencies:**
  - [x] Add `openai`, `faiss-cpu` to `requirements.txt`
  - [x] Add `OPENAI_API_KEY` to `.env.example` (User must add to `.env`)
- [x] **Offline Indexing (`build_index.py`):**
  - [x] Create `build_index.py` script skeleton
  - [x] Load entries from JSON export
  - [x] Implement OpenAI embedding for entry content
  - [x] Implement FAISS index creation (`IndexFlatL2`)
  - [x] Create and save index-to-entry mapping file (`index_mapping.json`)
  - [x] Save FAISS index (`index.faiss`)
- [x] **CLI Query Handling (`cli.py`):**
  - [x] Add `--query` argument to `argparse`
  - [x] Add logic to `main()` to handle `--query` mode
  - [x] Load FAISS index and mapping file
  - [x] Implement OpenAI embedding for user query
  - [x] Implement FAISS similarity search (top k)
  - [x] Implement context retrieval from mapping
  - [x] Implement basic prompt construction
  - [x] Implement OpenAI chat completion call
  - [x] Print LLM response to console

## Future Tasks (Post-MVP)

(Based on README Phases/Plan & TDD - refine as needed)

- [ ] **Operationalization & Sync:**
  - [ ] Finalize strategy for ongoing synchronization (See `design/ROLLOUT_SYNC_PLAN_TDD.md`). Key considerations:
    - Confirm use of `last_edited_time` for detecting changes.
    - Determine frequency of sync runs (e.g., daily, weekly, manual trigger).
    - Design robust error handling for sync failures.
    - Consider how to handle page deletions (remove from index?).
  - [ ] Implement the chosen synchronization solution (likely involving scheduled runs of `cli.py --export-month` and `build_index.py`).

- [x] **Web Interface (Hyper-MVP):**
  - [ ] **Backend Setup:**
    - [x] Choose & setup backend framework (Flask/FastAPI) in `backend/` directory.
    - [x] Define `/api/query` POST endpoint structure.
    - [x] Refactor RAG query logic from `cli.py` into a reusable function in `backend/rag_query.py` (or similar).
    - [x] Implement index/mapping loading in the backend function.
    - [x] Implement query embedding, FAISS search, context retrieval logic in the backend function.
    - [x] Implement LLM call and response parsing in the backend function.
    - [x] Implement extraction of source document titles and URLs.
    - [x] Connect the `/api/query` endpoint to the refactored RAG logic.
    - [x] Add basic error handling (try-except blocks) to the endpoint.
    - [x] Create `backend/requirements.txt` (including Flask/FastAPI, openai, faiss-cpu, etc.).
    - [x] Add CORS middleware to the backend application.
    - [x] **Fix:** Ensure `sources` list in `rag_query.py` is correctly populated with `{'title': title, 'url': constructed_url}`.
  - [ ] **Frontend Setup:**
    - [x] Create `frontend/` directory.
    - [x] Initialize React + Vite + TS project (`npm create vite@latest frontend -- --template react-ts`).
    - [x] Install and configure Tailwind CSS.
    - [x] Initialize `shadcn/ui` (`npx shadcn@latest init`).
    - [x] Install necessary `shadcn/ui` components (e.g., `button`, `input`, `card`, `skeleton`).
    - [x] Install `axios`.
  - [ ] **Frontend UI Implementation:**
    - [x] Create main `App.tsx` component structure.
    - [x] Add `Input` for query and `Button` for submission.
    - [x] Add `Card` for displaying the answer.
    - [x] Implement `useState` hooks for query, loading, answer, sources, error.
    - [x] Implement the `handleSubmit` function to call the backend API.
    - [x] Implement loading state display (e.g., show `Skeleton`).
    - [x] Render the answer text within the `Card`.
    - [x] Render source links (title + URL) below the answer.
    - [x] Display a generic error message if the API call fails.
  - [ ] **Integration & Testing:**
    - [x] Run backend and frontend dev servers concurrently.
    - [x] Perform end-to-end test: Query -> Submit -> Loading -> Answer + Sources displayed. (Verified after resolving backend/CLI discrepancies).
    - [ ] Test error handling path (e.g., simulate backend failure).

- [ ] **Further RAG Enhancements:**
  - [ ] **Model Exploration:**
    - [ ] Evaluate alternative/newer embedding models (e.g., OpenAI text-embedding-3-small/large, local models like Sentence Transformers) for potential cost/performance/accuracy benefits.
    - [ ] Test alternative final answer generation models (e.g., other OpenAI models, Anthropic Claude models via API) for cost/accuracy trade-offs.
  - [ ] **Chunking Strategy:** Investigate alternatives to embedding entire entries. Explore fixed-size, sentence-based, or content-aware chunking to potentially improve retrieval specificity and reduce context size.
  - [ ] **Retrieval Tuning:**
    - [ ] Experiment with different `TOP_K` values and analyze the cost/accuracy trade-off more formally.
    - [ ] Consider dynamic `TOP_K` based on query type or initial filter results.
    - [ ] Explore re-ranking retrieved results before sending to LLM (e.g., using a cross-encoder).
  - [ ] **Prompt Engineering:**
    - [ ] Further refine system/user prompts for query analysis and final answer generation.
    - [ ] Experiment with different prompt structures (e.g., few-shot examples).
    - [ ] Improve handling of the "tagged = seen" assumption (e.g., injecting text notes into context, or implementing two-stage logic for counting vs. content queries).
  - [ ] **Query Processing:**
    - [ ] Implement query decomposition for complex questions involving multiple parts or constraints.
    - [ ] Enhance query safety check (e.g., more nuanced topic detection, stricter default behavior on error).
  - [ ] **Vector DB:** Evaluate migrating from local FAISS to a managed vector database (Pinecone, Weaviate, ChromaDB, etc.) for scalability and easier management, especially if syncing becomes frequent.
  - [ ] **Cost Management:** Implement token usage tracking and estimated cost calculation per query, potentially displaying it in the UI.
  - [ ] **Configuration:** Move more hardcoded values (models, paths, `TOP_K`, prompts) to a configuration file or environment variables.

- [ ] **Testing & Reliability:**
  - [ ] Implement comprehensive unit tests for core modules (API client, extractors, transformers, storage, RAG components).
  - [ ] Implement integration tests for `cli.py` export and query workflows.
  - [ ] Add specific tests for edge cases (empty results, API errors, invalid inputs, context limits).
  - [ ] Improve error handling and provide more informative user messages (e.g., when pre-filtering yields no results, when safety check fails, when LLM refuses to answer).

- [ ] **Data Handling:**
  - [ ] Re-evaluate handling of entries with empty content during indexing (currently skipped).
  - [ ] Enhance `extract_text_from_block` for more block types if needed based on journal usage (e.g., toggles, tables).

## Deprecated Tasks

- [-] **Debug RAG Metadata Query Accuracy:** (Superseded by Hybrid RAG approach)
  - [-] Verify embedded text structure: Add debug log to `build_index.py` for `combined_text_for_embedding`.
  - [-] (Conditional) Re-run `build_index.py --force-rebuild` if logged structure is incorrect.
  - [-] Increase retrieval count: Modify `cli.py` to set `TOP_K = 30`.
  - [-] Refine LLM guidance: Modify system prompt in `cli.py`.
  - [-] Test metadata-specific queries (e.g., involving names, tags, dates).

## Implementation Plan

**Current Focus:** Execute the "MVP RAG - In Progress Tasks" list ASAP for demo.

### Relevant Files

- `README.md` - Project overview and plan ✅
- `.gitignore` - Specifies intentionally untracked files ✅
- `requirements.txt` - Project dependencies (`python-dotenv`, `requests`) ⏳ (Needs update)
- `.env.example` - Example environment variable file ⏳ (Needs update)
- `cli.py` - Command-line interface script ⏳ (Needs RAG logic)
- `notion_second_brain/__init__.py` - Package marker ✅
- `notion_second_brain/config.py` - Handles loading environment variables & logging ✅
- `notion_second_brain/notion/__init__.py` - Notion sub-package marker ✅
- `notion_second_brain/notion/api.py` - Notion API client ✅
- `notion_second_brain/notion/extractors.py` - Extracts text from Notion blocks ✅
- `notion_second_brain/processing/__init__.py` - Processing sub-package marker ✅
- `notion_second_brain/processing/filters.py` - Filters entries by date ✅
- `notion_second_brain/storage/__init__.py` - Storage sub-package marker ✅
- `notion_second_brain/storage/json_storage.py` - Saves entries to JSON files ✅
- `.cursor/rules/task-list.mdc` - Cursor rule for task management ✅
- `notion_second_brain/processing/transformers.py` - Transforms raw Notion data to simpler format ✅
- `design/MVP_RAG_TDD.md` - Technical design for RAG MVP ✅
- `build_index.py` - (New) Script to create FAISS index ⏳ (Needs creation)
- `index.faiss` - (Generated) FAISS index file ⏳
- `index_mapping.json` - (Generated) Index mapping file ⏳
- `metadata_cache.json` - (Generated) Cache for distinct metadata values ✅
- `TECHNICAL_LEARNINGS.md` - Summary of RAG development insights ✅
- `design/WEB_UI_TDD.md` - Technical design for Web UI MVP ✅

(Future files)
- `tests/` - Directory for test code
- `docs/` - Directory for documentation
- `backend/` - Directory for backend API server code ✅
  - `backend/main.py` - FastAPI/Flask application entry point ✅
  - `backend/rag_query.py` - Refactored RAG query logic ✅
  - `backend/requirements.txt` - Backend Python dependencies ✅
- `frontend/` - Directory for React frontend code ✅
  - `frontend/src/App.tsx` - Main React application component ✅
  - `frontend/src/components/ui/` - shadcn/ui components ✅
  - `frontend/package.json` - Frontend Node.js dependencies ✅
  - `frontend/vite.config.ts` - Vite configuration ✅
  - `frontend/tailwind.config.js` - Tailwind CSS configuration ✅
  - `frontend/tsconfig.json` - TypeScript configuration ✅
  - `frontend/tsconfig.app.json` - TypeScript app configuration ✅
  - `frontend/src/lib/utils.ts` - shadcn utility functions ✅
  - `frontend/components.json` - shadcn configuration file ✅

- [x] **Implement Initial Indexing (Batch Rollout):**
  - [x] Modify `cli.py` to support `--month YYYY-MM` export argument & filtering logic.
  - [x] Create control script (`scripts/batch_export.py`) to run monthly exports.
  - [x] Modify `build_index.py` to process multiple JSON files incrementally.
  - [x] Modify `build_index.py` to load/save existing index/mapping for checkpointing.
  - [x] Modify `build_index.py` to use batch embedding requests.
  - [x] Execute batch export (`scripts/batch_export.py`) for all historical data.
  - [x] Execute index build (`build_index.py`) for all historical data.