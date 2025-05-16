# Technical Design Document: Incremental Sync for Notion Second Brain

## 1. Metadata

*   **Title:** Incremental Sync for Notion Second Brain
*   **Status:** Proposed
*   **Author(s):** AI Assistant (Draft), Michael Lim (Owner)
*   **Date:** 2025-05-16
*   **Related Docs:** (Potentially link to GCS migration TDD if one existed)

## 2. Introduction / Motivation / Problem

Currently, updating the second brain requires a full re-export of Notion data, followed by a full rebuild of the search index. This process is:
*   **Time-consuming:** Fetching all entries from Notion and re-processing them takes significant time, especially as the journal grows.
*   **Inefficient:** Resources (CPU, network, API calls) are used to process data that hasn't changed.
*   **Costly:** Re-embedding unchanged content with OpenAI incurs unnecessary API costs.
*   **Poor User Experience (for updates):** The user has to wait for a lengthy process to see new entries reflected.

This TDD proposes an incremental sync mechanism to update only new or modified entries from Notion, making the sync process faster, more efficient, and cost-effective. The primary user is the owner of the Notion journal.

## 3. Goals

*   Implement a mechanism to identify and fetch only new or modified pages from Notion since the last successful sync.
*   Update the data stored in Google Cloud Storage (GCS) by modifying only the relevant monthly JSON files or by adding new ones.
*   Ensure the search index (Pinecone) is updated to reflect these changes with minimal re-indexing of unchanged content.
*   Significantly reduce the time and computational resources required for regular updates.
*   Minimize Notion API calls and OpenAI embedding costs during updates.
*   Provide a clear and reliable way to trigger and monitor the sync process.

## 4. Non-Goals

*   Real-time (sub-second) synchronization between Notion and the second brain. A batch-oriented approach is acceptable.
*   Automatic handling of complex merge conflicts if the same entry is modified in Notion and in the GCS JSONs simultaneously (the system assumes Notion is the source of truth).
*   Automatic detection and handling of structural changes to the Notion database schema itself (e.g., new properties, changed property types). This would still require manual adjustments to transformer/mapping code and potentially a full resync.
*   A UI for managing sync conflicts.

## 5. Proposed Design / Technical Solution

The core idea is to leverage Notion\'s `last_edited_time` property and maintain a record of the last successful sync.

### 5.1. Storing the Last Successful Sync Timestamp

*   A timestamp (ISO 8601 format, UTC) of the last successful incremental sync will be stored.
*   **Storage Location:** A simple text file in GCS, e.g., `gs://[GCS_BUCKET_NAME]/[GCS_INDEX_ARTIFACTS_PREFIX]/last_successful_export_sync.txt`.
    *   _Alternative:_ Could be a local file if `cli.py --export` is always run from the same environment, but GCS is more robust for potential future automation.

### 5.2. Modified Export Process (`cli.py`)

A new CLI flag, e.g., `python cli.py --export --incremental`, will trigger this mode. If `--incremental` is not specified, the current full monthly export behavior remains.

**Steps for Incremental Export:**

1.  **Read Timestamp:** Retrieve `last_successful_export_sync.txt` from GCS. If it doesn\'t exist (first sync after implementing this feature), prompt the user to either run a full export first or offer to set a default "since" time (e.g., start of the current day, or a user-provided date). For simplicity, initially, it might require a full export to have run once.
2.  **Query Notion:**
    *   Construct a Notion API filter to fetch pages where `last_edited_time` is `on_or_after` the stored `last_successful_export_sync.txt` timestamp.
    *   This fetches all pages created or edited since the last sync.
3.  **Process & Group New/Updated Pages:**
    *   For each page retrieved:
        *   Transform it into the `simple_entry_data` format (as done currently).
        *   Determine the `entry_date` (e.g., "2025-05-16") for the page.
        *   Group these processed entries by their `entry_date`'s month (e.g., "2025-05").
4.  **Update Monthly JSONs in GCS:**
    *   For each month that has new or updated entries:
        a.  **Download Existing File:** Download the corresponding monthly JSON file (e.g., `notion_exports/2025-05.json`) from GCS. If it doesn\'t exist, start with an empty list of entries.
        b.  **Merge Entries:**
            *   Convert the downloaded JSON content to a list of entry dictionaries.
            *   For each new/updated entry for that month:
                *   If an entry with the same `page_id` already exists in the list, replace it.
                *   If it doesn't exist, add it to the list.
        c.  **Upload Modified File:** Upload the updated list of entries back to the monthly JSON file in GCS, overwriting the previous version.
5.  **Update Timestamp:** After all affected monthly files are successfully updated in GCS, write the current timestamp (UTC) to `last_successful_export_sync.txt` in GCS.

### 5.3. Indexing Process (`build_index.py`)

*   `build_index.py` will continue to read *all* JSON files from the `GCS_EXPORT_PREFIX` (e.g., `notion_exports/`).
*   Its existing logic for comparing entry `last_edited_time` against `last_known_entry_versions.json` (or a similar mechanism if implemented) should prevent re-embedding and re-upserting content that hasn't actually changed, even if a monthly file was re-uploaded due to other entries in that month changing.
*   This means `build_index.py` doesn\'t need major changes for incremental data, as long as the GCS JSON files accurately reflect the latest state.

### 5.4. Triggering Mechanisms

**Option A: Manual CLI Commands (Simplest Start)**
*   User manually runs `python cli.py --export --incremental`.
*   User manually runs `python build_index.py` afterwards.
*   **Pros:** Easy to implement initially.
*   **Cons:** Requires manual intervention; user needs to remember two steps.

**Option B: Frontend "Refresh" Button (User-Initiated, Asynchronous)**
*   A "Refresh/Sync My Journal" button in the frontend UI.
*   This button calls a new backend API endpoint (e.g., `/api/sync-now`).
*   The API endpoint would:
    1.  Trigger the incremental export logic (similar to `cli.py --export --incremental`) as a background task. This could involve packaging the script\'s logic into a function callable by the API, or the API invoking the `cli.py` script if the execution environment allows. Using a proper background task queue (e.g., Celery with Redis/RabbitMQ, or platform-specific services like GCP Cloud Tasks) is recommended for robustness to avoid HTTP timeouts.
    2.  Once the export is complete, trigger the `build_index.py` logic, also as a background task.
    3.  The API could return an immediate "Sync initiated" response. Further status updates could be polled or pushed (e.g., via WebSockets).
*   **Pros:** Good user experience, single point of interaction.
*   **Cons:** More complex backend implementation (background tasks, API endpoint, status reporting). Vercel\'s environment needs careful consideration for long-running background tasks (may need to use Vercel Functions with longer timeouts or integrate an external queue/worker system).

**Option C: Automated Scheduled Sync (e.g., Vercel Cron Jobs)**
*   A scheduled job (e.g., daily Vercel Cron Job) that:
    1.  Invokes a serverless function or script that runs the incremental export logic.
    2.  Invokes another serverless function or script that runs the build index logic.
*   **Pros:** Fully automated, "set and forget."
*   **Cons:** User has less direct control over when sync happens. Debugging scheduled jobs can be more challenging. Requires Vercel Cron Job setup.

**Recommendation for Initial Implementation:**
Start with **Option A (Manual CLI)** for simplicity and to validate the core incremental logic. Then, consider **Option C (Vercel Cron Jobs)** for automation if the primary user (yourself) prefers background updates, or **Option B (Frontend Button)** if more interactive control is desired. Option B is the most user-friendly for a general application.

## 6. Alternatives Considered

1.  **Full Re-export, Smart Indexing:** Continue full monthly re-exports from Notion to GCS. Rely solely on `build_index.py`'s intelligence (comparing `last_edited_time` from `last_known_entry_versions.json`) to avoid re-embedding.
    *   **Pros:** Simpler export logic.
    *   **Cons:** Still fetches all data from Notion, potentially hitting rate limits or being slow for the export step. Overwrites GCS files completely each time.
2.  **Notion Webhooks (If/When Mature):** If Notion provides robust and granular webhooks for page changes, these could trigger more real-time updates directly to GCS and potentially Pinecone.
    *   **Pros:** Potentially more real-time.
    *   **Cons:** Depends heavily on Notion's webhook capabilities, can be complex to manage state and ensure reliability for many small updates. Likely overkill for a single-user journal.

## 7. Impact / Risks / Open Questions

*   **State Management:** The reliability of reading/writing `last_successful_export_sync.txt` is critical. What if it gets corrupted or deleted? (Fallback to full sync or user prompt).
*   **Error Handling during Incremental Sync:**
    *   If Notion query fails: Don't update the GCS timestamp. Retry later.
    *   If GCS upload of a monthly file fails: The timestamp shouldn't be updated. How to handle partially updated GCS state? (May require retrying the failed month).
*   **Merge Logic Complexity:** Modifying JSON files by downloading, merging, and re-uploading needs careful implementation to avoid data loss or corruption. Ensure atomicity or a backup mechanism if possible.
*   **Clock Skew:** Ensure timestamps are handled consistently in UTC to avoid issues with `last_edited_time` comparisons.
*   **First Run:** How is the system bootstrapped when `last_successful_export_sync.txt` doesn't exist? (Mandate a full export first, or allow user to specify a "sync since" date).
*   **"Missed" Edits:** If an entry's `last_edited_time` is *before* the `last_successful_export_sync.txt` but its content somehow changed without updating `last_edited_time` (unlikely with Notion but a theoretical edge case), it would be missed. This is a general limitation of timestamp-based sync.
*   **Definition of "Success":** The `last_successful_export_sync.txt` timestamp should only be updated if all steps (fetch from Notion, process, update *all* relevant monthly GCS files) complete without error for the current batch of changes.
*   **Long-Running Processes on Vercel (for Option B/C):** Vercel functions have execution time limits. Long syncs might need to be broken down or run in an environment suited for longer tasks (e.g., external worker, or Vercel background functions if they fit the need).

## 8. (Optional) Implementation Plan / Phases

1.  **Phase 1: Core Logic & Manual Trigger**
    *   Implement `last_successful_export_sync.txt` storage/retrieval in GCS.
    *   Add `--incremental` flag and logic to `cli.py`:
        *   Read timestamp.
        *   Query Notion by `last_edited_time`.
        *   Implement GCS monthly JSON download, merge new/updated entries, re-upload.
        *   Update timestamp in GCS on full success.
    *   Thoroughly test `cli.py --export --incremental` and subsequent `python build_index.py`.
2.  **Phase 2: Automation/User Trigger**
    *   Based on preference, implement either:
        *   Vercel Cron Job to call the incremental export and build index scripts/functions.
        *   Frontend button and backend API endpoint with background task execution for sync.

## 9. (Optional) Future Work / Follow-on

*   More sophisticated status reporting and error logging for sync processes.
*   UI indication of "last synced" time and sync status.
*   Investigate a "force full resync" option that clears the incremental timestamp and runs the full export.
*   Consider if `build_index.py` could be made more targeted if it knew exactly which monthly files were updated by the incremental export, rather than listing all files in the GCS prefix. (Minor optimization, current approach is likely fine).

---
This document outlines the proposed approach to adding an incremental sync feature. Feedback and further discussion are encouraged. 