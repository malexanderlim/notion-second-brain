# Technical Design Document: Web UI (Hyper MVP)

## 1. Metadata

*   **Title:** Notion Second Brain - Web UI (Hyper MVP)
*   **Status:** Proposed
*   **Author(s):** AI Assistant & User
*   **Date:** 2024-07-27
*   **Related Docs:** `design/MVP_RAG_TDD.md`, `TASKS.md`

## 2. Introduction / Motivation / Problem

The current Notion Second Brain RAG system is accessed via a Command Line Interface (`cli.py --query`). While functional, this limits accessibility. A simple, intuitive web interface is needed to allow users to easily query their indexed Notion data without using the terminal. This "Hyper MVP" focuses on delivering the core query/answer functionality quickly.

## 3. Goals (Phase 1 - Hyper MVP)

*   Provide a single-page web application (SPA).
*   Include an input field for the user to type their query.
*   Include a button to submit the query.
*   Display the generated answer received from the backend RAG system.
*   Display clickable links to the original Notion source pages used to generate the answer.
*   Utilize `shadcn/ui` components built on React, TypeScript, Vite, and Tailwind CSS for a modern and elegant look/feel.
*   Implement a simple backend API (using Flask or FastAPI) to receive queries and return results.
*   Ensure the UI provides basic feedback for loading states.

## 4. Non-Goals (Phase 1 - Hyper MVP)

*   User authentication or user-specific data segregation.
*   Ability to *edit*, *add*, or *delete* Notion content via the UI.
*   Displaying context snippets or highlighting relevant sections within source pages.
*   Real-time updates or WebSocket integration.
*   Advanced state management solutions (beyond basic React state).
*   Configuration options within the UI (e.g., changing models, `TOP_K`).
*   Displaying query cost, token counts, or performance metrics.
*   Browsing, searching, or filtering the *entire* index content; interaction is purely query-based.
*   Streaming responses from the LLM.
*   Complex error handling beyond displaying a generic failure message.

## 5. Proposed Design / Technical Solution (Phase 1)

### 5.1. Frontend

*   **Framework/Tooling:** React + Vite + TypeScript + Tailwind CSS
*   **UI Components:** `shadcn/ui` (Leveraging Radix UI primitives)
    *   `Input`: For query entry.
    *   `Button`: To submit the query.
    *   `Card` / `CardContent`: To display the formatted answer.
    *   `Skeleton` or `Loader2` (spinner icon): To indicate loading state.
    *   Basic layout components (e.g., `div`s styled with Tailwind).
    *   Links (`<a>`) for source documents.
*   **State Management:** Basic React `useState` for managing query input, loading status, API response (answer + sources), and error state.
*   **API Interaction:** `fetch` API or a lightweight library like `axios` to communicate with the backend API endpoint.

### 5.2. Backend

*   **Framework:** Python with Flask or FastAPI (FastAPI recommended for built-in data validation and async support, though Flask is simpler initially).
*   **API Endpoint:** A single POST endpoint, e.g., `/api/query`.
    *   **Request Body:** `{ "query": "user query string" }`
    *   **Response Body (Success):** `{ "answer": "LLM generated answer", "sources": [ { "title": "Page Title 1", "url": "notion_url_1" }, ... ] }`
    *   **Response Body (Error):** `{ "error": "Error message" }` (with appropriate HTTP status code, e.g., 500).
*   **Logic:**
    1.  Receive the query from the request body.
    2.  **Refactor Core Logic:** Extract the RAG query processing logic (embedding query, FAISS search, context retrieval, LLM call) from `cli.py` into a reusable function/module.
    3.  Call this refactored function with the user's query.
    4.  Format the LLM response and retrieved source document metadata (title, URL) into the JSON response structure.
    5.  Implement basic error handling (e.g., catching exceptions from the RAG process).

### 5.3. Interaction Flow

1.  User navigates to the web UI page.
2.  User types their query into the `shadcn/ui Input` field.
3.  User clicks the `shadcn/ui Button`.
4.  UI enters a loading state (e.g., displays `Skeleton` or spinner, disables button).
5.  UI sends a POST request to the backend `/api/query` endpoint with the query text.
6.  Backend API receives the request.
7.  Backend executes the RAG logic (embedding, search, context retrieval, LLM call).
8.  Backend formats the result (answer + source links) or an error message.
9.  Backend sends the JSON response back to the UI.
10. UI receives the response.
11. UI exits the loading state.
12. If successful, UI displays the answer in a `Card` and lists the source links below it.
13. If error, UI displays a simple error message.

## 6. Alternatives Considered

*   **Server-Side Rendering (e.g., Jinja2 + Flask/FastAPI):** Simpler backend setup, but results in full page reloads and a less modern/interactive user experience compared to a React SPA. Rejected to prioritize a modern UX feel.
*   **Different UI Frameworks (e.g., Material UI, Bootstrap):** `shadcn/ui` was specifically requested and aligns well with the desire for a modern, elegant, composable UI built on Tailwind CSS.
*   **Keeping CLI Only:** Does not meet the goal of increased accessibility.

## 7. Impact / Risks / Open Questions

*   **Impact:** Significantly improves the accessibility and user experience of querying the Notion knowledge base. Provides a foundation for potential future UI features.
*   **Risks:**
    *   Frontend development setup complexity (Node.js, Vite, React, TS, Tailwind).
    *   Requires refactoring existing RAG logic from `cli.py` into a reusable, API-callable format. Potential for introducing bugs during refactoring.
    *   Managing dependencies and potential conflicts between the Python backend and Node.js frontend environments during development and deployment.
    *   Initial UI might feel slow if RAG processing takes significant time (Non-Goal: Streaming).
*   **Open Questions:**
    *   Final deployment strategy (e.g., Docker containers, serverless functions, simple VM)?
    *   How to best structure the project repository to accommodate both the Python backend and the React frontend (e.g., monorepo vs. separate directories)? A `frontend/` directory seems appropriate.
    *   Specific design details for displaying source links (just links? titles + links?).
    *   How should API errors be presented to the user in the UI (generic message vs. specific details)? Start with generic.

## 8. Implementation Plan / Phases

### Phase 1: Hyper MVP (This Document)

1.  **Backend Setup:**
    *   Choose Flask or FastAPI.
    *   Create basic backend structure (`backend/` or similar directory).
    *   Define the `/api/query` endpoint.
    *   Refactor RAG query logic from `cli.py` into a callable function within the backend codebase. Ensure it handles loading the index, embedding the query, searching, retrieving context, calling the LLM, and extracting source links.
    *   Connect the endpoint to the refactored logic.
    *   Add basic error handling.
    *   Setup `requirements.txt` for the backend (including Flask/FastAPI).
    *   Add CORS middleware to allow requests from the frontend development server.
2.  **Frontend Setup:**
    *   Create `frontend/` directory.
    *   Initialize React project using Vite + TypeScript template (`npm create vite@latest frontend -- --template react-ts`).
    *   Install Tailwind CSS and `shadcn/ui` following their official guides.
    *   Install `axios` or use `fetch`.
3.  **Frontend UI Implementation:**
    *   Create main App component.
    *   Add `shadcn/ui` Input, Button, Card components.
    *   Implement basic state management (`useState`) for query, loading, response data (answer, sources), error.
    *   Implement the API call function triggered by the button click.
    *   Display loading state (`Skeleton`/spinner) while waiting for the API.
    *   Render the answer and source links upon successful response.
    *   Display a basic error message on API failure.
4.  **Integration & Testing:**
    *   Run backend server and frontend dev server concurrently.
    *   Test the end-to-end flow: typing query, submitting, seeing loading state, viewing answer and links.
    *   Test basic error handling (e.g., simulate a backend error).

### Phase 2: Enhancements (Future TDD)

*   Improved UI/UX (styling, layout refinements).
*   More robust error display in the UI.
*   Potentially display context snippets alongside source links.
*   Clear/Reset button.
*   (Optional) Streaming response display if backend supports it.

### Phase 3: Further Features (Future TDD)

*   User configuration options (if backend supports).
*   Displaying query metadata (cost, tokens).
*   Authentication.

## 9. Future Work / Follow-on

*   Deployment strategy implementation.
*   More comprehensive testing (unit, integration).
*   Potential integration with the sync process (e.g., showing index status). 