# Technical Design Document: Google OAuth Integration for Notion Second Brain

**Metadata:**

*   `Title:` Google OAuth Integration for Notion Second Brain
*   `Status:` Proposed
*   `Author(s):` AI Assistant (initial draft)
*   `Date:` 2024-07-30
*   `Related Docs:` `RAG_SYSTEM_OVERVIEW.md`, `README.md`

**1. Introduction / Motivation / Problem:**

The Notion Second Brain application is currently publicly accessible via its Vercel deployment. Given that it processes and displays potentially sensitive personal journal entries, public access is a security risk. This document outlines the design for integrating Google OAuth to restrict access to only authorized users (initially, the primary user/owner).

**2. Goals:**

*   Secure all application frontend and backend API endpoints.
*   Allow users to log in using their Google account.
*   Prevent unauthorized access to the application's data and functionality.
*   Maintain user sessions after successful authentication.
*   Provide a mechanism for users to log out.
*   Ensure the solution works both in local development and on Vercel deployments.

**3. Non-Goals:**

*   Role-based access control (RBAC) beyond a single authenticated user (for now).
*   Support for other OAuth providers (e.g., GitHub, Facebook).
*   User self-registration (access will be implicitly granted if the Google account matches an allowed account, or by being the first user if no allow-list is implemented initially).
*   Advanced session management features like "remember me" or session timeout warnings beyond standard cookie expiration.
*   Two-factor authentication directly within this application (relies on Google's 2FA).

**4. Proposed Design / Technical Solution:**

The solution involves modifications to both the backend (FastAPI) and frontend (React/TypeScript) applications. We will use session-based authentication with HTTPOnly cookies.

**4.1 Backend (FastAPI - `backend/` directory):**

*   **Dependencies:**
    *   `Authlib`: For handling the OAuth 2.0 flow with Google.
    *   `starlette[sessions]`: For session middleware (FastAPI uses Starlette). `itsdangerous` will be needed for signed cookies.
*   **Configuration (Environment Variables):**
    *   `GOOGLE_CLIENT_ID`: Provided by Google Cloud Console.
    *   `GOOGLE_CLIENT_SECRET`: Provided by Google Cloud Console.
    *   `SESSION_SECRET_KEY`: A strong, random secret key for signing session cookies.
    *   `AUTHORIZED_USER_EMAIL` (Highly Recommended): The specific Google email address of the sole authorized user. If not set, or if it's empty, access should be denied to prevent unintended exposure.
*   **Middleware:**
    *   `SessionMiddleware` from Starlette will be added to the FastAPI application to handle session cookies. Cookies should be configured as `HTTPOnly`, `Secure` (in production), and `SameSite=Lax`.
*   **New API Endpoints (under `/api/auth/`):**
    *   `GET /api/auth/login/google`:
        *   Redirects the user's browser to the Google OAuth consent screen.
        *   `Authlib` will be used to generate the authorization URL, including `client_id`, `redirect_uri` (`/api/auth/callback/google`), `scope` (e.g., `openid email profile`), and a `state` parameter for CSRF protection.
    *   `GET /api/auth/callback/google`:
        *   Handles the callback from Google after the user authenticates.
        *   Validates the `state` parameter.
        *   Uses the received `authorization_code` to request an access token and ID token from Google (using `Authlib` and the `GOOGLE_CLIENT_SECRET`).
        *   Fetches the user's profile information (especially email) from Google using the access token or ID token.
        *   **(Authorization Check):**
            *   Retrieve the `AUTHORIZED_USER_EMAIL` environment variable.
            *   If `AUTHORIZED_USER_EMAIL` is not set/empty OR if the authenticated user's email does NOT match `AUTHORIZED_USER_EMAIL`, deny access (e.g., return a 403 Forbidden error or redirect to an error page).
        *   Stores relevant user information (e.g., email, name) in the session if access is granted.
        *   Sets the session cookie.
        *   Redirects the user to the frontend application's main page (e.g., `/`).
    *   `POST /api/auth/logout`:
        *   Clears the user's session data.
        *   Clears the session cookie.
        *   Redirects the user to a frontend login page or a logged-out confirmation page.
    *   `GET /api/auth/me`:
        *   Returns the current authenticated user's information (e.g., email, name) from the session.
        *   Returns a 401/403 if no user is authenticated. This endpoint helps the frontend determine authentication status.
*   **Protecting Existing Endpoints (e.g., `/api/query`, `/api/last-updated`):**
    *   A FastAPI dependency will be created to check for a valid session and authenticated user.
    *   If the user is not authenticated, the dependency will raise an `HTTPException` (e.g., 401 Unauthorized or 403 Forbidden), or potentially redirect to the login URL.
    *   This dependency will be added to all routes that require authentication.

**4.2 Frontend (React/TypeScript - `frontend/` directory):**

*   **Dependencies:**
    *   `react-router-dom`: For handling client-side routing to differentiate between login page and protected application content.
    *   `axios`: For making HTTP requests (already in use).
*   **Authentication State Management:**
    *   A React Context (e.g., `AuthContext` - already created) will be used to manage the authentication state (e.g., `isAuthenticated`, `user`, `isLoading`).
    *   The `AuthProvider` will wrap the application.
    *   On application load (e.g., in the main `App` component or a root layout), the frontend will call the backend's `/api/auth/me` endpoint to check if a session exists and retrieve user data, updating the `AuthContext`.
*   **Routing Structure (`react-router-dom`):**
    *   The main application entry point (`main.tsx`) will set up `BrowserRouter`.
    *   `App.tsx` (or a new root component) will define routes:
        *   A public route for `/login` which renders a new `LoginPage` component.
        *   A protected route for `/` (or other app-specific paths) which renders the main application content (e.g., a new `MainAppLayout` component or the existing `App.tsx` content refactored).
    *   A `ProtectedRoute` utility component might be created to handle the logic of checking `isAuthenticated` from `AuthContext` and redirecting to `/login` if not authenticated or if `isLoading` is true.
*   **UI Components:**
    *   **`LoginPage.tsx`:**
        *   Displayed when the user is not authenticated and tries to access the app, or navigates to `/login`.
        *   Contains a prominent "Login with Google" button.
        *   Clicking this button will navigate the browser (e.g., using `window.location.href`) to the backend's `/api/auth/login/google` endpoint.
    *   **`LogoutButton.tsx` (or integrated into an existing component like a Navbar):**
        *   A button visible when the user is authenticated.
        *   Clicking this button will navigate the browser to the backend's `/api/auth/logout` endpoint.
        *   The `AuthContext` will subsequently reflect the logged-out state after the backend redirect and a call to `/api/auth/me` or manual state clearing.
    *   **Main Application UI (e.g., refactored `App.tsx` or a new `MainAppLayout.tsx`):**
        *   This will contain the current query interface and results display.
        *   It will only be rendered if the user is authenticated (enforced by the protected route).
        *   Will display the `LogoutButton`.
*   **Conditional Rendering:**
    *   The application will use `isLoading` from `AuthContext` to show a global loading indicator while `checkAuthState` is in progress.
    *   Based on `isAuthenticated`, either the `LoginPage` or the main application content (via routing) will be rendered.

**5. Alternatives Considered:**

*   **Token-based Authentication (JWTs):** While JWTs are common, session cookies are simpler for this web application use case, especially with HTTPOnly cookies providing some CSRF protection. JWTs would require more complex frontend storage and handling.
*   **Other OAuth Libraries (e.g., `FastAPI-Login`, `python-social-auth`):** `Authlib` is a comprehensive and well-maintained library for OAuth clients and providers, suitable for this task. Other libraries could also work but `Authlib` provides good flexibility.
*   **Vercel Edge Authentication:** Vercel offers some built-in authentication solutions. However, integrating directly into the FastAPI backend provides more control and is more portable if the hosting platform changes.
*   **Firebase Authentication:** Could be an option but might be overkill if only Google Auth is needed and adds another dependency.

**6. Impact / Risks / Open Questions:**

*   **Impact:**
    *   Existing users (if any beyond the owner) will be required to log in.
    *   Slight increase in latency for authenticated requests due to session validation.
    *   Deployment configuration will need to include new environment variables on Vercel.
*   **Risks:**
    *   **Misconfiguration of OAuth Client:** Incorrect redirect URIs or client secrets can break the login flow.
    *   **Session Fixation/Hijacking:** Ensure `SESSION_SECRET_KEY` is strong and cookies are `HTTPOnly` and `Secure`. The `state` parameter in OAuth helps prevent CSRF during login.
    *   **Open Redirects:** Care must be taken with any redirects, ensuring they only go to trusted URLs.
*   **Open Questions:**
    *   How should the application behave if `AUTHORIZED_USER_EMAIL` is not set or is empty? (Recommendation: Deny access by default to ensure security).
    *   Exact error handling and user feedback on the frontend for login failures, including authorization denial (email not matching `AUTHORIZED_USER_EMAIL`).

**7. (Optional) Implementation Plan / Phases:**

(This will be detailed in `AUTH_TASKS.md` or similar.)
1.  Backend: Setup environment, install dependencies.
2.  Backend: Implement `/api/auth/login/google` and `/api/auth/callback/google`.
3.  Backend: Implement session middleware and user storage in session.
4.  Backend: Implement `/api/auth/me` and `/api/auth/logout`.
5.  Backend: Protect existing API routes.
6.  Frontend: Create AuthContext and login UI.
7.  Frontend: Implement login flow redirection and logout functionality.
8.  Frontend: Protect frontend routes/components.
9.  Testing: Thoroughly test all authentication and authorization scenarios.
10. Vercel: Deploy and test with production environment variables.

**8. (Optional) Future Work / Follow-on:**

*   Allowing multiple authorized users via a configurable list.
*   More granular session timeout management. 