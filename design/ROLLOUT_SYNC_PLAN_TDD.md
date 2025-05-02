# TDD: Index Rollout & Synchronization Plan

**Status:** Proposed
**Author:** AI Assistant (Gemini)
**Date:** 2025-05-01

## 1. Introduction & Motivation

The current MVP uses a manually generated index based on a one-time export. To make the RAG system useful long-term, we need a strategy to:
1.  Perform the initial indexing of the full dataset (~1600 entries).
2.  Keep the index reasonably up-to-date with new or modified entries in Notion.

This document explores options and proposes a plan for both initial rollout and ongoing synchronization.

## 2. Goals

*   Define a process for indexing the entire Notion database.
*   Define a strategy for updating the index as Notion data changes.
*   Consider trade-offs between freshness, complexity, cost, and performance.

## 3. Non-Goals

*   Real-time, instantaneous synchronization.
*   Complex conflict resolution logic.
*   Full implementation details (this doc focuses on strategy).

## 4. Proposed Design / Technical Solution (Initial Thoughts)

### 4.1 Initial Rollout (Indexing 1600+ Entries)

*   **Option A: Enhance `build_index.py`:**
    *   Modify `build_index.py` to handle potential API rate limits or timeouts during embedding generation for a large dataset.
    *   Add batching to embedding calls.
    *   Run `python cli.py --export --period all` first, then `python build_index.py --input output/all_time.json`.
    *   *Pros:* Reuses existing code.
    *   *Cons:* Might be slow; requires large initial JSON export; potentially fragile for very large datasets.
*   **Option B: Direct Notion Indexing Script:**
    *   Create a new script (`full_indexer.py`?) that queries *all* pages directly from Notion (handling pagination).
    *   For each page, retrieve block children, extract text, generate embedding, and add to FAISS index + mapping.
    *   *Pros:* Avoids intermediate large JSON file.
    *   *Cons:* More complex API interaction logic; needs careful rate limit handling; potentially slower than batch embedding if OpenAI API allows large batches.

### 4.2 Ongoing Synchronization

*   **Option A: Periodic Re-indexing:**
    *   Schedule a job (e.g., cron, GitHub Action) to periodically (daily? weekly?) run the full export and `build_index.py` process.
    *   *Pros:* Simplest to implement.
    *   *Cons:* Index can be stale; inefficient (re-embeds unchanged entries); potentially costly if run frequently.
*   **Option B: Incremental Updates (Event-Driven - Harder):**
    *   Requires a way to detect changes in Notion (Webhooks? Periodic polling?). Notion Webhooks (if available/suitable) might be complex.
    *   Polling: Periodically query Notion for pages modified since the last check (`last_edited_time` filter).
    *   For modified pages: Re-fetch content, re-embed, update/replace vector in FAISS (requires unique ID mapping, FAISS `remove_ids` / `add_with_ids`).
    *   For new pages: Fetch, embed, add.
    *   For deleted pages: Need strategy (e.g., periodic cleanup or check during retrieval).
    *   *Pros:* Index is fresher; more efficient potentially.
    *   *Cons:* Much more complex to implement reliably; requires state management (last sync time); handling deletions is tricky.
*   **Option C: Incremental Updates (Batch Polling):**
    *   Periodically query Notion for pages modified since last check.
    *   Export *only* these modified/new pages to a temporary JSON.
    *   Run a modified `build_index.py` that can *update* an existing FAISS index (remove old vectors, add new ones).
    *   *Pros:* Balances freshness and complexity; less complex than event-driven.
    *   *Cons:* Still requires state management and index update logic.

## 5. Alternatives Considered

*   Using a managed Vector DB service (Pinecone, etc.) - Might simplify sync later, but adds external dependency.

## 6. Impact / Risks / Open Questions

*   Cost of embedding 1600+ entries initially?
*   Cost of frequent re-embedding for sync?
*   Performance of FAISS with 1600+ entries?
*   Reliability of Notion API for large queries/frequent polling?
*   Complexity of implementing incremental updates vs. stale data tolerance?
*   How to handle Notion page deletions?

## 7. Dependencies to Add (Potentially)

*   Scheduling library (e.g., `schedule`, `apscheduler`) if polling/periodic sync is chosen.
*   Libraries for specific Vector DBs if not using FAISS. 