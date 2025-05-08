# notion-second-brain

A project to extract journal entries from Notion, process them, build a Retrieval Augmented Generation (RAG) system, and provide a web interface to query your personal knowledge with selectable LLMs and cost estimation.

## Project Overview

This project implements a full RAG pipeline that can:
1. Extract journal entries from Notion via their API.
2. Process and index this data using vector embeddings (FAISS) and metadata.
3. Offer a web-based chat interface to query your journal data.
4. Allow selection from multiple Large Language Models (e.g., OpenAI GPT series, Anthropic Claude series).
5. Provide token usage and estimated cost per query.
6. Support advanced query features like metadata filtering and semantic search.
7. Display the timestamp of the last synced journal entry in the UI.

## Technology Stack

- **Backend**: Python, FastAPI
- **Data Processing & Indexing**: Python, `requests`, `faiss-cpu`, `openai` (for embeddings)
- **LLM Integration**: OpenAI API, Anthropic API
- **Frontend**: React (Vite + TypeScript), Tailwind CSS, shadcn/ui, axios
- **Storage**: JSON files for data export, FAISS index, and mappings.

## Development Phases (Current Status)

Many of the initial phases outlined below have been completed or superseded by the current RAG implementation with a web UI.

### Phase 1: Notion Data Extraction (MVP) - LARGELY COMPLETE
- Functionality to connect to Notion, extract entries, and save them to JSON is implemented via `cli.py`.

### Phase 2: Data Processing and Storage - LARGELY COMPLETE
- `build_index.py` handles processing exported JSON, creating embeddings, building the FAISS index (`index.faiss`), creating entry-to-index mappings (`index_mapping.json`), and caching metadata (`metadata_cache.json`), and recording the timestamp of the latest processed entry (`last_entry_update_timestamp.txt`).

### Phase 3: Query Interface - LARGELY COMPLETE
- `cli.py` provides basic CLI query functionality.
- The FastAPI backend (`backend/main.py`, `backend/rag_query.py`) implements the full RAG pipeline, serving as the query engine for the web UI.
- Features include model selection, token counting, cost estimation, metadata filtering, semantic search, and retrieval of the last sync timestamp.

### Phase 4: Web Interface - LARGELY COMPLETE
- A React/Vite/TypeScript frontend (`frontend/`) provides a chat-like UI to query the RAG system.
- Users can select models, view answers, sources, query cost information, and see the date of the last synced journal entry.

## Implementation Plan

The initial week-by-week plan has been largely executed, leading to the current feature set.

## Cursor Development Guidelines

(This section can be updated if specific new guidelines for Cursor are needed)

## Getting Started

These steps assume you have Python 3, Node.js (with npm or yarn), and `git` installed.

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url> # Replace with your repo URL
    cd notion-second-brain
    ```

2.  **Setup Backend:**
    - Navigate to the `backend` directory: `cd backend`
    - Create and activate a Python virtual environment:
      ```bash
      python3 -m venv venv
      source venv/bin/activate # macOS/Linux
      # venv\Scripts\activate.bat # Windows Cmd
      # venv\Scripts\Activate.ps1 # Windows PowerShell
      ```
    - Install backend Python dependencies:
      ```bash
      pip install -r requirements.txt
      ```
    - Go back to the project root: `cd ..`

3.  **Set up Notion Integration:**
    - Go to [https://www.notion.so/my-integrations](https://www.notion.so/my-integrations) and create a new **internal** integration.
    - Copy the "Internal Integration Token".
    - Find the ID of the Notion database you want to use.
    - Share the database with the integration you created.

4.  **Configure Environment Variables:**
    - In the project root, copy `.env.example` to `.env`:
      ```bash
      cp .env.example .env
      ```
    - Open the `.env` file and add your keys:
      ```dotenv
      NOTION_TOKEN=secret_YOUR_NOTION_TOKEN
      NOTION_DATABASE_ID=YOUR_DATABASE_ID
      OPENAI_API_KEY=sk-YOUR_OPENAI_API_KEY
      ANTHROPIC_API_KEY=sk-YOUR_ANTHROPIC_API_KEY # Add your Anthropic key
      ```

5.  **Export Notion Data & Build Index:**
    - (If you haven't already) Export your Notion data using `cli.py`.
      For example, to export all data for a specific month (e.g., January 2024) to be used for indexing:
      ```bash
      python cli.py --export-month 2024-01 
      ```
      This creates `output/2024-01.json`. You might want to run this for all relevant months, or use `--period all` for a single comprehensive file (though monthly can be more manageable for very large journals).
    - Build the RAG index from the exported JSON files:
      ```bash
      python build_index.py 
      ```
      This will process all `*.json` files in the `output/` directory by default and create `index.faiss`, `index_mapping.json`, `metadata_cache.json`, and `last_entry_update_timestamp.txt` in the project root. Ensure `schema.json` is also present or correctly generated/placed in the root, as it's used by the RAG system.

6.  **Run the Backend Server:**
    - Ensure your Python virtual environment (created in step 2) is active.
    - From the project root directory:
      ```bash
      uvicorn backend.main:app --reload --port 8000
      ```
    - The API server should now be running on `http://localhost:8000`.

7.  **Run the Frontend Application:**
    - Open a new terminal.
    - Navigate to the `frontend` directory: `cd frontend`
    - Install frontend dependencies:
      ```bash
      npm install 
      # or yarn install
      ```
    - Start the frontend development server:
      ```bash
      npm run dev
      # or yarn dev
      ```
    - Open your browser and navigate to the URL provided (usually `http://localhost:5173`).

## Usage

1.  **Ensure your data is exported and indexed** (see steps 5 in "Getting Started").
2.  **Run the backend server** (step 6).
3.  **Run the frontend application** (step 7).
4.  Open the frontend URL in your browser.
5.  Select your desired Language Model from the dropdown.
6.  Type your query into the input box and press Enter or click "Query".
7.  View the answer, sources, model details, estimated cost, and the "Last Synced Entry" date.

### CLI Querying (Legacy/Alternative)

Basic querying via `cli.py` is still available if the index is built:
+The CLI query function now uses the same advanced RAG engine as the web UI (`backend/rag_query.py`). You can also specify a model using the `--model` flag.
+
 ```bash
 python cli.py --query "What did I do last weekend?"
+
+# Example using a specific model:
+python cli.py --query "Tell me about my umeshu project" --model claude-3-5-haiku-20241022
 ```
 
 ## Example Implementation Plan for Cursor

(This section can be updated for future feature development.)

Remember to:
- Use Cursor's AI features to help with implementation details.
- Create tasks for each component before implementing.
- Run tests frequently to verify functionality.
- Document your code and update the README as you progress.