# Vercel Deployment Tasks

This task list outlines the steps for deploying the Notion Second Brain application to Vercel, integrating Vercel Blob for data storage, and ensuring local development remains functional, based on the `design/VERCEL_DEPLOYMENT_AND_LOCAL_DEV_TDD.md` document.

## Phase 1: Vercel Blob Integration & Backend Setup

*   [ ] **1.1 Modify `build_index.py` for Vercel Blob Upload:**
    *   [ ] Add Vercel Blob SDK/client library to `backend/requirements.txt`.
    *   [ ] Implement authentication with Vercel Blob (using access token from env var/`.env`).
    *   [ ] Implement logic to upload generated files (`index.faiss`, `index_mapping.json`, `metadata_cache.json`, `schema.json`, `last_entry_update_timestamp.txt`) to a designated Vercel Blob store.
    *   [ ] Add command-line argument (e.g., `--upload-to-blob`) or environment variable to trigger upload.
    *   [ ] Ensure local saving functionality is preserved if upload is not triggered.
*   [ ] **1.2 Modify `backend/rag_initializer.py` for Remote Data Loading:**
    *   [ ] Add Vercel Blob SDK/client library if not already included via `build_index.py` changes.
    *   [ ] Implement logic to check `DATA_SOURCE_MODE` environment variable (default to `remote` for Vercel environment).
    *   [ ] If `DATA_SOURCE_MODE == 'remote'`:
        *   [ ] Authenticate with Vercel Blob (using `VERCEL_BLOB_ACCESS_TOKEN` from Vercel env vars).
        *   [ ] Download necessary data files from Blob storage to the function's temporary filesystem (e.g., `/tmp/`).
        *   [ ] Update RAG logic to load index and mappings from these temporary local paths.
    *   [ ] Ensure error handling for download failures.
*   [ ] **1.3 Vercel Project Setup & Backend Deployment:**
    *   [ ] Create a new project on Vercel.
    *   [ ] Link the GitHub repository.
    *   [ ] Configure the backend build settings in `vercel.json` (or Vercel UI):
        *   [ ] Specify Python runtime.
        *   [ ] Define build commands.
        *   [ ] Set up routes for FastAPI application (e.g., `/api/*` to `backend/main.py`).
    *   [ ] Add necessary environment variables to Vercel project settings:
        *   [ ] `OPENAI_API_KEY`
        *   [ ] `ANTHROPIC_API_KEY`
        *   [ ] `VERCEL_BLOB_ACCESS_TOKEN` (generate this from Vercel Blob settings)
        *   [ ] `DATA_SOURCE_MODE=remote` (or ensure it defaults to remote if not set locally).
        *   [ ] Any other required API keys or configurations.
    *   [ ] Ensure `backend/requirements.txt` is complete.
*   [ ] **1.4 Test Backend Deployment with Blob Data:**
    *   [ ] Manually run `build_index.py --upload-to-blob` locally to populate Vercel Blob.
    *   [ ] Trigger a deployment on Vercel.
    *   [ ] Test backend API endpoints (e.g., `/api/query`, `/api/sync-status`) to ensure they correctly load data from Vercel Blob and function as expected.
    *   [ ] Check Vercel deployment logs for any errors related to data loading or backend function execution.

## Phase 2: Frontend Deployment & Full System Test

*   [ ] **2.1 Deploy Frontend to Vercel:**
    *   [ ] Configure frontend build settings in Vercel (should auto-detect Vite):
        *   [ ] Specify `frontend/` as the root directory.
        *   [ ] Build command: `npm run build` (or as per `frontend/package.json`).
        *   [ ] Output directory: `frontend/dist`.
    *   [ ] Ensure frontend API calls point to the Vercel-hosted backend (e.g., relative `/api/query` or `https://your-project.vercel.app/api/query`). This might require an environment variable for the API base URL in the frontend, configured in Vercel.
*   [ ] **2.2 Configure Vercel Password Protection:**
    *   [ ] Enable Vercel's built-in Password Protection for the entire project in Vercel project settings.
    *   [ ] Set a strong password.
*   [ ] **2.3 End-to-End Testing of Deployed Application:**
    *   [ ] Access the deployed frontend URL.
    *   [ ] Enter the password when prompted.
    *   [ ] Test all application functionalities:
        *   [ ] Querying the RAG system.
        *   [ ] Voice input.
        *   [ ] Display of "last updated" timestamp.
        *   [ ] Manual sync trigger (if already implemented and intended for Vercel use).
        *   [ ] Model selection and cost estimation display.
    *   [ ] Verify that data is being fetched correctly from Vercel Blob via the backend.
    *   [ ] Test on different devices/browsers if possible.

## Phase 3: Local Development Adaptation

*   [ ] **3.1 Refine `build_index.py` for Local Save vs. Upload:**
    *   [ ] Ensure the script defaults to saving data files locally (e.g., to project root or a `data/` subfolder) if `--upload-to-blob` is not specified.
    *   [ ] Verify that local data saving paths are clear and configurable if necessary.
*   [ ] **3.2 Implement `DATA_SOURCE_MODE=local` in `backend/rag_initializer.py`:**
    *   [ ] Add logic to handle `DATA_SOURCE_MODE=local`.
    *   [ ] Introduce `LOCAL_DATA_PATH` environment variable (e.g., default to `.` or `./data/`).
    *   [ ] If `DATA_SOURCE_MODE == 'local'`, load RAG data files from the path specified by `LOCAL_DATA_PATH`.
    *   [ ] Ensure this logic works seamlessly when running `uvicorn backend.main:app --reload` locally.
*   [ ] **3.3 Update Local Development Scripts & Documentation:**
    *   [ ] Create or update `.env.example` to include:
        *   [ ] `DATA_SOURCE_MODE=local`
        *   [ ] `LOCAL_DATA_PATH=./data` (or chosen default)
        *   [ ] Placeholder for `VERCEL_BLOB_ACCESS_TOKEN` (for optional local upload testing).
    *   [ ] Update `README.md` or other development guides to explain the new local setup, environment variables, and workflow (e.g., run `build_index.py` for local data, then `start-backend.sh`).
    *   [ ] Ensure `start-backend.sh` and `start-frontend.sh` (if used) are compatible with these changes (e.g., load `.env` file).
*   [ ] **3.4 Test Local Development Workflow Thoroughly:**
    *   [ ] Run `python cli.py --export ...` to generate raw data.
    *   [ ] Run `python build_index.py` (without upload flag) to create/update local data files in `LOCAL_DATA_PATH`.
    *   [ ] Start the backend locally (`uvicorn` or `start-backend.sh`). Verify it loads data from `LOCAL_DATA_PATH`.
    *   [ ] Start the frontend locally (`npm run dev` or `start-frontend.sh`). Verify it communicates with the local backend.
    *   [ ] Test querying and all other features using the local setup.
    *   [ ] Test running `python build_index.py --upload-to-blob` locally and confirm it uploads to Vercel Blob correctly without interfering with the local data setup.

## Future Work / Follow-on (Considerations from Design Doc)

*   [ ] **CI/CD for `build_index.py`:**
    *   [ ] Design GitHub Action workflow to automate `build_index.py` execution and Vercel Blob upload (e.g., on changes to `output/` or a schedule).
    *   [ ] Securely manage `VERCEL_BLOB_ACCESS_TOKEN` in GitHub Actions secrets.
*   [ ] **Error Handling:**
    *   [ ] Implement more sophisticated error handling and retries for Vercel Blob operations in `build_index.py` and `rag_initializer.py`.
*   [ ] **Monitoring:**
    *   [ ] Set up monitoring for Vercel Blob usage and costs.
    *   [ ] Review Vercel function logs for performance and errors.
*   [ ] **Schema Handling:**
    *   [ ] Revisit decision on `schema.json` (Blob vs. bundled). If bundling is preferred, update scripts to reflect this and remove from Blob upload/download.
*   [ ] **Security:**
    *   [ ] Regularly review Vercel password protection and access token security.
    *   [ ] Consider if any other security hardening is needed for the Vercel deployment.

---
*This task list is a starting point and can be refined as the implementation progresses.* 