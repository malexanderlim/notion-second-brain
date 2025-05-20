# notion-second-brain

A project to extract journal entries from Notion, process them, build a Retrieval Augmented Generation (RAG) system, and provide a web interface to query your personal knowledge with selectable LLMs and cost estimation.

## Project Overview

This project implements a full RAG pipeline that can:
1. Extract journal entries from Notion via their API.
2. Process and index this data using Pinecone for vector embeddings and metadata, with artifacts stored in Google Cloud Storage.
3. Offer a web-based chat interface to query your journal data.
4. Allow selection from multiple Large Language Models (e.g., OpenAI GPT series, Anthropic Claude series).
5. Provide token usage and estimated cost per query.
6. Support advanced query features like metadata filtering and semantic search.
7. Display the timestamp of the last synced journal entry in the UI.

## Technology Stack

- **Backend**: Python, FastAPI (with a modular structure: `rag_initializer`, `rag_query`, `query_analyzer`, `retrieval_logic`, `llm_interface`, `auth`, etc., located in `api/`)
- **Data Processing & Indexing**: Python, `requests`, `openai` (for embeddings), `pinecone-client`, `google-cloud-storage`
- **LLM Integration**: OpenAI API, Anthropic API
- **Authentication**: Google OAuth 2.0 (via Authlib on backend, React Context on frontend)
- **Frontend**: React (Vite + TypeScript), Tailwind CSS, shadcn/ui, axios
- **Storage**: Google Cloud Storage for monthly JSON data exports (from Notion) and for storing index artifacts (schema, entry mappings, metadata cache, last update timestamp). Pinecone for vector storage.

## Development Phases (Current Status)

Many of the initial phases outlined below have been completed or superseded by the current RAG implementation with a web UI.

### Phase 1: Notion Data Extraction (MVP) - LARGELY COMPLETE
- Functionality to connect to Notion, extract entries, and save them as monthly JSON files to Google Cloud Storage is implemented via `cli.py`.

### Phase 2: Data Processing and Storage - LARGELY COMPLETE
- `build_index.py` handles processing exported JSONs (read from Google Cloud Storage), creating OpenAI embeddings, upserting vectors to Pinecone, and storing related artifacts (entry mappings `index_mapping.json`, `metadata_cache.json`, `schema.json`, and `last_entry_update_timestamp.txt`) in Google Cloud Storage.

### Phase 3: Query Interface - LARGELY COMPLETE
- `cli.py` provides basic CLI query functionality (now leveraging the backend's RAG engine).
- The FastAPI backend (`api/main.py`, `api/rag_query.py`) implements the full RAG pipeline, serving as the query engine for the web UI.
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

2.  **Setup Backend (`api/` directory):**
    - Navigate to the `api` directory: `cd api`
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

4.  **Set up Google OAuth 2.0 Credentials:**
    - Go to the [Google Cloud Console](https://console.cloud.google.com/).
    - Create a new project or select an existing one.
    - Navigate to "APIs & Services" > "Credentials".
    - Click "Create Credentials" and select "OAuth client ID".
    - Choose "Web application" as the application type.
    - Configure the OAuth client ID:
        - **Authorized JavaScript origins**: Add your frontend URLs.
            - For local development: `http://localhost:5173` (or your frontend port if different)
            - For production (Vercel): Your Vercel deployment URL (e.g., `https://your-app-name.vercel.app`)
        - **Authorized redirect URIs**: Add your backend callback URLs.
            - For local development: `http://localhost:8000/api/auth/callback/google` (or your backend port if different)
            - For production (Vercel): `https://your-app-name.vercel.app/api/auth/callback/google`
    - Click "Create". Copy the "Client ID" and "Client secret".

5.  **Configure Environment Variables:**
    - In the project root, copy `.env.example` to `.env`:
      ```bash
      cp .env.example .env
      ```
    - Open the `.env` file and add your keys. It should look similar to this:
      ```dotenv
      # Notion Integration
      NOTION_TOKEN=secret_YOUR_NOTION_TOKEN
      NOTION_DATABASE_ID=YOUR_DATABASE_ID

      # LLM & Vector DB APIs
      OPENAI_API_KEY=sk-YOUR_OPENAI_API_KEY
      ANTHROPIC_API_KEY=sk-YOUR_ANTHROPIC_API_KEY
      PINECONE_API_KEY=YOUR_PINECONE_API_KEY
      PINECONE_INDEX_NAME=your-pinecone-index-name

      # Google Cloud Storage
      GCS_BUCKET_NAME=your-gcs-bucket-name
      # GCS_EXPORT_PREFIX=notion_exports/ (Optional, defaults to this)
      # GCS_INDEX_ARTIFACTS_PREFIX=index_artifacts/ (Optional, defaults to this)

      # Google OAuth 2.0 Credentials
      GOOGLE_CLIENT_ID=YOUR_GOOGLE_CLIENT_ID.apps.googleusercontent.com
      GOOGLE_CLIENT_SECRET=YOUR_GOOGLE_CLIENT_SECRET
      SESSION_SECRET_KEY=generate_a_strong_random_string_for_this
      AUTHORIZED_USER_EMAIL=your_authorized_google_email@example.com

      # Frontend URLs (for backend redirects)
      FRONTEND_URL=http://localhost:5173
      FRONTEND_LOGOUT_URL=http://localhost:5173/login
      ```
      **Important Security Notes:**
        - `SESSION_SECRET_KEY` should be a long, random, and unpredictable string. You can generate one using Python's `secrets.token_hex(32)`.
        - `AUTHORIZED_USER_EMAIL` restricts application access to only this Google account.
        - For production deployment (e.g., on Vercel), ensure `FRONTEND_URL` and `FRONTEND_LOGOUT_URL` are updated to your Vercel deployment URLs.

6.  **Export Notion Data & Build Index:**
    - (If you haven't already) Export your Notion data using `cli.py`. This will upload JSON files to your GCS bucket.
      For example, to export all data for a specific month (e.g., January 2024):
      ```bash
      python cli.py --export-month 2024-01 
      ```
      This saves `YYYY-MM.json` to your configured GCS bucket under the `GCS_EXPORT_PREFIX`. You might want to run this for all relevant months, or use the `--export` flag (no month argument) for a sequential export of all months containing data.
    - (Recommended) Ensure your Notion database schema is uploaded to GCS:
      ```bash
      python cli.py --schema
      ```
    - Build the RAG index. This will process `*.json` files from your `GCS_EXPORT_PREFIX` in GCS, generate embeddings, upsert to Pinecone, and save artifacts to `GCS_INDEX_ARTIFACTS_PREFIX`:
      ```bash
      python build_index.py 
      ```
      This creates `index_mapping.json`, `metadata_cache.json`, `last_entry_update_timestamp.txt`, and uploads/updates `schema.json` (if found locally or if `cli.py --schema` was run) in your GCS bucket under the `GCS_INDEX_ARTIFACTS_PREFIX`.

7.  **Run the Backend Server (`api/`):**
    - Ensure your Python virtual environment (created in step 2) is active.
    - From the project root directory:
      ```bash
      uvicorn api.main:app --reload --port 8000
      ```
    - The API server should now be running on `http://localhost:8000`.

8.  **Run the Frontend Application:**
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

1.  **Ensure your data is exported and indexed** (see step 6 in "Getting Started").
2.  **Run the backend server** (step 7).
3.  **Run the frontend application** (step 8).
4.  Open the frontend URL in your browser. You will be prompted to log in with your Google Account specified in `AUTHORIZED_USER_EMAIL`.
5.  After successful login, you can select your desired Language Model from the dropdown.
6.  Type your query into the input box and press Enter or click "Query".
7.  View the answer, sources, model details, estimated cost, and the "Last Synced Entry" date.
8.  Use the Logout button to end your session.

### CLI Querying (Legacy/Alternative)

Basic querying via `cli.py` is still available if the index is built.
-The CLI query function now uses the same advanced RAG engine as the web UI (`api/rag_query.py`). You can also specify a model using the `--model` flag.
-```bash
-python cli.py --query "What did I do last weekend?"
-# Example using a specific model:
-python cli.py --query "Tell me about my umeshu project" --model claude-3-5-haiku-20241022
+```bash
+python cli.py --query "What did I do last weekend?"
+# Example using a specific model:
+python cli.py --query "Tell me about my umeshu project" --model claude-3-haiku-20240307
 ```

## Example Implementation Plan for Cursor

(This section can be updated for future feature development.)

Remember to:
- Use Cursor's AI features to help with implementation details.
- Create tasks for each component before implementing.
- Run tests frequently to verify functionality.
- Document your code and update the README as you progress.

