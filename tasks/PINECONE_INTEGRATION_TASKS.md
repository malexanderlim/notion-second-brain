## Pinecone Integration (Vercel Deployment Unblocker)

Corresponds to TDD: `design/PINECONE_INTEGRATION_TDD.md`

- [x] **Phase 1: Setup & Configuration**
  - [x] 1.1. User: Create Pinecone account and a new index (e.g., `notion-second-brain`).
    - [x] Configure index: dimensions (e.g., 1536 for `text-embedding-ada-002`), metric (`cosine`), pod type (e.g., `s1.x1`).
  - [x] 1.2. User: Obtain Pinecone API Key and Environment identifier.
  - [x] 1.3. Add `PINECONE_API_KEY`, `PINECONE_ENVIRONMENT`, `PINECONE_INDEX_NAME` to `.env.example`. (Note: `PINECONE_ENVIRONMENT` not strictly needed for client v3+)
  - [x] 1.4. User: Configure these environment variables in Vercel project settings.
  - [x] 1.5. Update root `requirements.txt`: Add `pinecone-client`, remove `faiss-cpu`.
  - [x] 1.6. Verify embedding model dimension used in the project and ensure it matches Pinecone index configuration.

- [x] **Phase 2: Backend Implementation (`api/` directory and `build_index.py`)**
  - [x] 2.1. **`api/rag_initializer.py` Modifications:**
    - [x] 2.1.1. Implement `initialize_pinecone_client()`:
      - [x] Initialize Pinecone client using new environment variables (passed as args).
      - [x] Connect to the Pinecone index and store/return the index instance.
      - [x] Call this initializer in `main.py` (FastAPI app startup) and `cli.py`.
    - [x] 2.1.2. Modify `load_rag_data()`:
      - [x] Remove FAISS index loading.
      - [x] Ensure `index_mapping.json` IDs are stringified if not already, for Pinecone compatibility.
      - [x] The \"index\" object passed around should now be the Pinecone index client instance.
  - [x] 2.2. **`api/retrieval_logic.py` Modifications:**
    - [x] 2.2.1. Update `perform_semantic_search` (or create `perform_pinecone_search`):
      - [x] Replace FAISS `index.search()` with Pinecone `pinecone_index_instance.query()`.
      - [x] Adapt to use `vector`, `top_k`. For MVP, `include_metadata=True` can be used if metadata is stored, or rely on `index_mapping.json` post-fetch. (Implemented with metadata filtering based on `page_id`)
      - [x] Ensure output (list of IDs, scores) is compatible with downstream logic.
  - [x] 2.3. **`build_index.py` Modifications:**
    - [x] 2.3.1. Initialize Pinecone client.
    - [x] 2.3.2. Remove FAISS index creation and saving.
    - [x] 2.3.3. Implement vector upsertion to Pinecone:
      - [x] Prepare vectors with string IDs (consistent with `index_mapping.json`) and embeddings.
      - [x] Upsert to Pinecone in batches (e.g., batch size 100, configurable).
      - [x] Log batch upsert progress.
    - [x] 2.3.4. Ensure `index_mapping.json` is saved with string IDs.
    - [x] 2.3.5. Update `--force-rebuild` logic:
      - [x] Optionally, `pinecone_index_instance.delete(delete_all=True)` if flag is present (and index is not empty).
      - [x] Proceed with full data fetch, embedding, and upsertion.
  - [x] 2.4. **`api/rag_query.py` Modifications:** (Implicitly covered by successful CLI query)
    - [x] Update to use `perform_pinecone_search`.
    - [x] Adapt logic for Pinecone's string IDs and metadata filtering.
  - [x] 2.5. **`cli.py` Modifications:**
    - [x] Initialize Pinecone client.
    - [x] Ensure it uses the new Pinecone-backed RAG system.
  - [x] 2.6. **`api/main.py` (Pydantic Models):**
    - [x] Add `score: Optional[float]` to `SourceDocument`.

- [ ] **Phase 3: Testing**
  - [ ] 3.1. **Local Testing:**
    - [x] Run `build_index.py --force-rebuild` and verify data in Pinecone console. (Completed)
    - [x] Test RAG queries via FastAPI backend locally (using Postman or frontend).
    - [x] Verify `cli.py --query "..."` functionality. (Completed)
  - [ ] 3.2. **Vercel Preview Deployment Testing:**
    - [ ] Deploy the branch to Vercel (preview environment).
    - [ ] Verify `build_index.py` can be run (if applicable to Vercel build, or run manually pointing to Vercel env). Often, indexing is a separate step before deployment or run via a one-off job. For Vercel, usually data is already in Pinecone.
    - [ ] Test deployed API RAG queries thoroughly using the frontend.
    - [ ] Check Vercel function logs for errors.
    - [ ] Confirm serverless function size is within limits.

- [ ] **Phase 4: Deployment & Documentation**
  - [ ] 4.1. Merge changes to the main deployment branch.
  - [ ] 4.2. Deploy to Vercel production.
  - [ ] 4.3. Monitor production application for any issues.
  - [ ] 4.4. Update `README.md` regarding the new Pinecone dependency and any changes to setup/deployment.
  - [ ] 4.5. Update `TECHNICAL_LEARNINGS.md` with insights from Pinecone integration.

- [ ] **Phase 5: Code Cleanup**
  - [ ] 5.1. Review all modified files for dead, stale, or commented-out code related to FAISS or previous implementations.
  - [ ] 5.2. Remove unnecessary imports and variables.
  - [ ] 5.3. Ensure logging is appropriate (remove excessive debug logs if not needed for production).
  - [ ] 5.4. Check `api/rag_config.py` for any FAISS-specific constants that can be removed or updated. 