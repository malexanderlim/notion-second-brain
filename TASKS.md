# Notion Second Brain - Phase 1 MVP RAG Implementation

Tracking progress for the initial MVP RAG Demo focusing on a basic query loop.

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

- [ ] **Phase 1 Testing - POSTPONED for MVP:**
  - [ ] Write unit tests for `notion/api.py` (`tests/test_notion_api.py`)
  - [ ] Write unit tests for `processing/filters.py`
  - [ ] Write unit tests for `notion/extractors.py` (`tests/test_data_processing.py`?)
  - [ ] Write unit tests for RAG components (`build_index.py`, CLI query logic)
  - [ ] Write integration tests for the `cli.py` extraction and query processes
  - [ ] Handle edge cases (e.g., empty database, invalid date formats, API rate limits, no JSON export, missing index files)
  - [ ] Improve logging and monitoring (especially for RAG)
  - [ ] Document code (docstrings) and API usage (`docs/api_usage.md`)
- [ ] **Phase 2: Data Processing and Storage:**
  - [ ] Enhance `extract_text_from_block` for more block types (e.g., toggles children, tables)
  - [ ] Add support for handling image references (e.g., save URL, download?)
  - [ ] Create summary statistics for journal entries
  - [ ] Implement topic extraction or categorization
  - [ ] Design a more robust storage system (beyond simple JSON files?)
  - [ ] Implement incremental sync functionality
  - [ ] Add versioning for exported data
  - [ ] Create a data update pipeline
- [ ] **Phase 2.3: Data Transformation for LLMs:**
  - [ ] Create embedding generation for entries
  - [ ] Implement chunking strategies
  - [ ] Design metadata structure for improved retrieval
  - [ ] Build utility functions for data preparation (`processing/transformers.py`?)
- [ ] **Phase 3: Query Interface & RAG Enhancements:**
  - [ ] **3.1 CLI Enhancements:**
    - [ ] Enhance `cli.py` query capabilities (e.g., specify k, show sources)
    - [ ] Implement basic search functionality in CLI (maybe redundant with RAG?)
    - [ ] Add reporting and statistics features to CLI
  - [ ] **3.2 LLM Integration (RAG) - Robust Implementation:**
    - [ ] Refine RAG system architecture (based on MVP learnings)
    - [ ] Evaluate & Choose/set up Vector Database (Pinecone, Weaviate, ChromaDB?)
    - [ ] Implement robust indexing pipeline (chunking, embedding, metadata storage)
    - [ ] Implement retrieval logic (semantic search + metadata filtering)
    - [ ] Integrate with OpenAI or Claude API (configurable?)
    - [ ] Improve prompt templates & context window management
- [ ] **Phase 4: Web Interface (Future):**
  - [ ] Design and implement Backend API (Flask/FastAPI)
  - [ ] Design and implement Frontend (React/Next.js)

- [ ] **Phase 5: Operationalization & Sync:**
  - [ ] Create `design/ROLLOUT_SYNC_PLAN_TDD.md` (Initial draft created)
  - [ ] Finalize strategy for initial indexing (1600+ entries)
  - [ ] Finalize strategy for ongoing synchronization
  - [ ] **Implement Initial Indexing (Batch Rollout):**
    - [ ] Modify `cli.py` to support `--month YYYY-MM` export argument & filtering logic.
    - [ ] Create control script (`scripts/batch_export.py`) to run monthly exports.
    - [ ] Modify `build_index.py` to process multiple JSON files incrementally.
    - [ ] Modify `build_index.py` to load/save existing index/mapping for checkpointing.
    - [ ] Modify `build_index.py` to use batch embedding requests.
    - [ ] Execute batch export (`scripts/batch_export.py`) for all historical data.
    - [ ] Execute index build (`build_index.py`) for all historical data.
  - [ ] Implement synchronization solution

- [ ] **Phase 4: Web Interface (Hyper-MVP):**
  - [ ] Choose simple web framework (e.g., Streamlit, Gradio, basic Flask)
  - [ ] Create basic UI with input box for query and area for displaying response
  - [ ] Create backend endpoint/logic to receive query
  - [ ] Reuse/adapt `cli.py` query logic (loading index, embedding, search, prompting, LLM call)
  - [ ] Display LLM response in UI
  - [ ] (Optional Stretch) Display token counts / estimated cost for OpenAI calls
  - [ ] (Optional Stretch) Add dropdown/option to select LLM model

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

(Future files)
- `tests/` - Directory for test code
- `docs/` - Directory for documentation

- [x] **Implement Initial Indexing (Batch Rollout):**
  - [x] Modify `cli.py` to support `--month YYYY-MM` export argument & filtering logic.
  - [x] Create control script (`scripts/batch_export.py`) to run monthly exports.
  - [x] Modify `build_index.py` to process multiple JSON files incrementally.
  - [x] Modify `build_index.py` to load/save existing index/mapping for checkpointing.
  - [x] Modify `build_index.py` to use batch embedding requests.
  - [x] Execute batch export (`scripts/batch_export.py`) for all historical data.
  - [x] Execute index build (`build_index.py`) for all historical data.