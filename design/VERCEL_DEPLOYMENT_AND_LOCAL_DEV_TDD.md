# Vercel Deployment and Local Development Strategy TDD

*   **Status:** Proposed
*   **Author(s):** AI Assistant (based on user discussion)
*   **Date:** 2024-05-09
*   **Related Docs:** `RAG_SYSTEM_OVERVIEW.md`, `TASKS.md`

## 1. Introduction / Motivation / Problem

The Notion Second Brain RAG system is currently run locally via `start-backend.sh` and `start-frontend.sh`. This limits accessibility to when the user is at their primary computer. The goal is to deploy the application online for access from anywhere while maintaining robust security for sensitive journal data and preserving the ability to run and debug the system locally. This document outlines the strategy for deploying to Vercel, using Vercel Blob for data storage, and ensuring local development remains fully functional.

## 2. Goals

*   Deploy the React frontend (`frontend/`) to Vercel for global web access.
*   Deploy the FastAPI backend (`backend/`) to Vercel Serverless Functions.
*   Securely store and manage RAG data files (`index.faiss`, `index_mapping.json`, `metadata_cache.json`, `schema.json`, `last_entry_update_timestamp.txt`) using Vercel Blob.
*   Implement Vercel's Password Protection for the entire deployed application (frontend and backend APIs).
*   Keep the primary GitHub repository for the codebase public for portfolio purposes.
*   Maintain the ability to run the full export, indexing, and query pipeline locally for development, testing, and debugging.
*   Ensure the local `output/` directory remains the source for raw data exports (`YYYY-MM.json` files) for local index building.

## 3. Non-Goals

*   Implementing complex multi-user authentication systems (Vercel's built-in password protection is sufficient for initial MVP).
*   Migrating to a different vector database beyond FAISS for this phase.
*   Automated CI/CD pipelines for `build_index.py` to update Vercel Blob (manual local execution and upload is acceptable for now).
*   Public access to the deployed application; it should be private to the user.

## 4. Proposed Design / Technical Solution

### 4.1. Vercel Deployment

#### 4.1.1. Frontend (`frontend/`)
*   **Deployment:** The React/Vite frontend in the `frontend/` directory will be deployed as a static site on Vercel.
*   **Configuration:** Vercel will auto-detect the Vite project. Build command `npm run build` in `frontend/package.json`.
*   **API Communication:** The frontend will make API calls to the Vercel-hosted backend (e.g., `https://your-project.vercel.app/api/query`).

#### 4.1.2. Backend (`backend/`)
*   **Deployment:** The FastAPI application in `backend/main.py` will be deployed as Python Serverless Functions on Vercel.
*   **Configuration:**
    *   A `vercel.json` file in the project root will define the build for the Python backend and routing.
    *   `backend/requirements.txt` will list all Python dependencies.
    *   Environment variables (e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `VERCEL_BLOB_ACCESS_TOKEN`, `DATA_SOURCE`) will be configured in Vercel project settings.
*   **Data Handling:** See section 4.2.

#### 4.1.3. Security
*   **Deployment Protection:** Vercel's built-in Password Protection will be enabled for the entire project, covering both frontend and backend API routes.
*   **API Keys:** All sensitive API keys will be stored as environment variables in Vercel.

### 4.2. Data Storage (Vercel Blob)

*   **Files to Store:** `index.faiss`, `index_mapping.json`, `metadata_cache.json`, `schema.json`, `last_entry_update_timestamp.txt`.
*   **Storage Mechanism:** Vercel Blob will be used. These files will *not* be committed to the Git repository (and will remain in `.gitignore`).
*   **Upload Process (`build_index.py` - modified):**
    1.  The `build_index.py` script (run locally) will continue to read raw data from the local `output/` directory.
    2.  After generating the data files, `build_index.py` will be modified to:
        *   Authenticate with Vercel Blob using an access token (from a local environment variable or `.env` file).
        *   Upload the generated files to a designated Vercel Blob store.
*   **Download Process (`backend/rag_initializer.py` - modified for Vercel deployment):**
    1.  When a Vercel serverless function for the backend starts (cold start):
        *   It will check an environment variable `DATA_SOURCE`. If set to `remote` (or by default for Vercel environment):
            *   It will authenticate with Vercel Blob (using `VERCEL_BLOB_ACCESS_TOKEN` from Vercel env vars).
            *   Download the necessary data files from Blob storage into the function's temporary filesystem (e.g., `/tmp/`).
            *   The RAG logic will then load the index and mappings from these temporary local paths (e.g., `/tmp/index.faiss`).

### 4.3. Preserving Local Development & Execution

The goal is to allow `cli.py` and local testing of the backend (e.g., `uvicorn backend.main:app --reload --app-dir .`) to work seamlessly.

*   **Local `build_index.py`:**
    *   No changes to reading from `output/`.
    *   When run locally, it can *optionally* upload to Vercel Blob if configured with the necessary token and an argument/env var (e.g., `python build_index.py --upload-to-blob`). By default, it could just save files locally.
*   **Local Backend Data Source Configuration (`backend/rag_initializer.py`):**
    *   The `load_rag_data()` function (or equivalent initialization logic) will be modified to determine the data source path.
    *   An environment variable, `DATA_SOURCE_PATH` (or similar), will specify the local directory where `index.faiss`, `index_mapping.json`, etc., are stored (e.g., the project root or a `data/` subdirectory where `build_index.py` saves them locally).
    *   When running locally, `DATA_SOURCE` could be implicitly `local` or set explicitly.
    *   The `rag_initializer.py` will check:
        ```python
        # Example pseudo-code in rag_initializer.py
        import os

        DATA_SOURCE_MODE = os.getenv('DATA_SOURCE_MODE', 'local') # 'local' or 'remote'
        LOCAL_DATA_PATH = os.getenv('LOCAL_DATA_PATH', '.') # Path to local data files
        # VERCEL_BLOB_TOKEN = os.getenv('VERCEL_BLOB_TOKEN') # For remote

        if DATA_SOURCE_MODE == 'remote':
            # Download from Vercel Blob to a temporary path
            # data_files_path = "/tmp/" # or similar
            # ... download logic ...
        else: # 'local'
            data_files_path = LOCAL_DATA_PATH
            # Ensure files exist at data_files_path + "/index.faiss", etc.

        # Load index from data_files_path + "/index.faiss"
        # Load mapping from data_files_path + "/index_mapping.json"
        ```
*   **Local `cli.py` and Backend Testing:**
    *   Will use the local data loading logic described above.
    *   Developers will run `build_index.py` locally (without necessarily uploading to Blob) to generate/update local data files in `LOCAL_DATA_PATH`.
    *   Environment variables for local development (e.g., in a `.env` file loaded by `python-dotenv`) would point to local data and use local API keys.
    *   The `start-backend.sh` script for local dev will remain functional.

### 4.4. Workflow Summary

1.  **Local Development:**
    *   Run `python cli.py --export ...` to get raw data into `output/`.
    *   Run `python build_index.py` (defaulting to local save) to generate/update data files in `LOCAL_DATA_PATH`.
    *   Run frontend and backend locally, configured to use `DATA_SOURCE_MODE=local`.
2.  **Deploying/Updating Online Data:**
    *   Run `python build_index.py --upload-to-blob` (or equivalent) to update data in Vercel Blob.
    *   Commit code changes to Git.
    *   Push to the main branch to trigger Vercel deployment for frontend/backend code. The deployed backend will use `DATA_SOURCE_MODE=remote` and fetch from Blob.

## 5. Alternatives Considered

*   **Committing data files to a private Git repo:** Rejected as the goal is to keep the main codebase repo public.
*   **Using AWS S3 / Google Cloud Storage:** Viable alternatives, but Vercel Blob is chosen for tighter integration with the Vercel ecosystem, potentially simpler setup, and a suitable free tier for the project's current scale.
*   **Not supporting local data loading:** Rejected as it would hinder local development and debugging significantly.

## 6. Impact / Risks / Open Questions

*   **Impact:**
    *   `build_index.py` needs modification for Vercel Blob upload.
    *   `backend/rag_initializer.py` needs modification for conditional data loading (local vs. remote Blob).
    *   New environment variables need to be managed for local dev (`.env`) and Vercel deployment settings.
*   **Risks:**
    *   Correctly configuring Vercel Blob upload/download logic and permissions.
    *   Managing Vercel Blob access tokens securely.
    *   Ensuring the temporary file system in Vercel Serverless Functions has enough space and is writable for the downloaded data files (current total size ~12.4MB, which should be fine).
    *   Potential latency on cold starts if Vercel Blob downloads are slow, though for ~12.4MB this should be acceptable.
*   **Open Questions:**
    *   Exact Vercel Blob SDK/API usage details for Python in `build_index.py` and `rag_initializer.py`.
    *   Final naming and handling of environment variables for `DATA_SOURCE_MODE` and `LOCAL_DATA_PATH`.
    *   Strategy for handling `schema.json`: should it be bundled with the app code (since it's small and changes infrequently) or also uploaded to Blob? (Current proposal: upload to Blob for consistency, but bundling is an option).

## 7. Implementation Plan / Phases (High-Level)

1.  **Phase 1: Vercel Blob Integration & Backend Setup**
    *   Modify `build_index.py` to upload generated files to Vercel Blob.
    *   Modify `backend/rag_initializer.py` to download files from Vercel Blob when `DATA_SOURCE_MODE=remote`.
    *   Set up Vercel project, configure backend deployment, and add environment variables (excluding `DATA_SOURCE_MODE` for now, or setting to `remote`).
    *   Test backend deployment with data from Blob.
2.  **Phase 2: Frontend Deployment & Full System Test**
    *   Deploy frontend to Vercel.
    *   Configure Vercel Password Protection.
    *   End-to-end testing of the deployed application.
3.  **Phase 3: Local Development Adaptation**
    *   Refine `build_index.py` for optional local save vs. upload.
    *   Implement `DATA_SOURCE_MODE=local` logic in `backend/rag_initializer.py` using `LOCAL_DATA_PATH`.
    *   Update local development scripts/documentation (e.g., `.env.example`).
    *   Test local development workflow thoroughly.

## 8. Future Work / Follow-on

*   Automate `build_index.py` execution and Vercel Blob upload via a CI/CD pipeline (e.g., GitHub Actions on changes to `output/` or on a schedule).
*   More sophisticated error handling for Blob operations.
*   Monitoring for Vercel Blob usage and costs (though expected to be within free tier). 