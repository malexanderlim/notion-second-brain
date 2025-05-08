# Technical Design Document: Voice Input Feature

**Metadata:**

*   `Title:` Voice Input for Second Brain Querying
*   `Status:` Proposed
*   `Author(s):` AI Assistant (Initial Draft), <Your Name/Team>
*   `Date:` YYYY-MM-DD
*   `Related Docs:` `design/WEB_UI_TDD.md`

## 1. Introduction / Motivation / Problem

This document outlines the design for adding a voice input feature to the Notion Second Brain web interface. Currently, users can only type their queries. Allowing users to speak their queries will enhance accessibility and provide a more convenient way to interact with the application. The feature will leverage a speech-to-text API (initially OpenAI Whisper) to convert spoken audio into text, which will then be used as the query input.

## 2. Goals

*   Integrate a microphone icon/button into the frontend UI.
*   Allow users to click the microphone icon to start audio recording.
*   Capture audio input from the user's microphone.
*   Send the captured audio to a speech-to-text service (OpenAI Whisper API).
*   Receive the transcribed text from the API.
*   Populate the query input field with the transcribed text.
*   (Optional) Automatically submit the query after transcription or require user confirmation.
*   Provide visual feedback to the user during recording and processing states.

## 3. Non-Goals

*   Real-time streaming transcription (initially, audio will be processed after recording stops).
*   Support for languages other than English in the initial MVP.
*   Advanced voice command capabilities beyond simple query input (e.g., "clear input," "go back").
*   Local/on-device speech-to-text processing.

## 4. Proposed Design / Technical Solution

### 4.1 Frontend (React + Vite + TS - `frontend/src/App.tsx`)

1.  **UI Element:**
    *   Add a microphone `IconButton` (e.g., using a `Mic` icon from `lucide-react`) next to the text input field.
    *   The button's appearance will change to indicate status (idle, recording, processing).

2.  **State Management:**
    *   `isRecording` (boolean): Tracks if audio is currently being captured.
    *   `isTranscribing` (boolean): Tracks if the audio is being sent to/processed by the Whisper API.
    *   `transcriptionError` (string | null): Stores any error messages related to voice input.

3.  **Audio Capture:**
    *   Utilize the browser's `MediaRecorder` API (`navigator.mediaDevices.getUserMedia`) to access the microphone and record audio.
    *   Request microphone permissions from the user.
    *   When the microphone icon is clicked:
        *   If not recording, start recording:
            *   Set `isRecording` to `true`.
            *   Store recorded audio chunks in an array.
        *   If recording, stop recording:
            *   Set `isRecording` to `false`.
            *   Combine audio chunks into a single `Blob`.
            *   Initiate transcription process.

4.  **API Interaction:**
    *   Create a new function `handleVoiceQuery(audioBlob: Blob)`.
    *   This function will:
        *   Set `isTranscribing` to `true`.
        *   Create a `FormData` object and append the `audioBlob` as a file (e.g., `audio.webm` or `audio.mp3`).
        *   Make a POST request to a new backend endpoint (e.g., `/api/transcribe`).
        *   On successful response, update the main query input state with the transcribed text and set `isTranscribing` to `false`.
        *   Optionally, automatically submit the query.
        *   On error, set `transcriptionError` and `isTranscribing` to `false`.

### 4.2 Backend (Python - FastAPI/Flask - `backend/main.py`)

1.  **New API Endpoint:**
    *   Create a new POST endpoint, e.g., `/api/transcribe`.
    *   This endpoint will accept file uploads (the audio blob).

2.  **Speech-to-Text Integration (OpenAI Whisper):**
    *   The endpoint handler will:
        *   Receive the audio file from the request.
        *   Ensure `OPENAI_API_KEY` is configured.
        *   Use the OpenAI Python client to call the Whisper API (e.g., `client.audio.transcriptions.create`).
            *   Specify the model (e.g., `whisper-1`).
            *   Pass the audio file.
        *   Return the transcribed text in the API response (e.g., `{"transcription": "text"}`).
        *   Handle potential errors from the Whisper API.

3.  **Dependencies:**
    *   Add `openai` to `backend/requirements.txt` if not already present (it should be there for RAG).
    *   The backend will need to be able to handle file uploads (FastAPI's `UploadFile` or Flask's `request.files`).

### 4.3 Workflow Summary

1.  User clicks microphone icon in the frontend.
2.  Browser requests microphone permission (if not already granted).
3.  Audio recording starts; icon changes to indicate recording.
4.  User speaks their query.
5.  User clicks microphone icon again (or a stop button) to stop recording.
6.  Frontend sends the recorded audio blob to `/api/transcribe` backend endpoint. Icon changes to indicate processing.
7.  Backend receives audio, calls Whisper API.
8.  Whisper API transcribes audio to text.
9.  Backend returns transcribed text to frontend.
10. Frontend populates the query input field with the text. Icon returns to idle state.
11. (Optional) Frontend automatically submits the query.

## 5. Alternatives Considered

*   **Browser Web Speech API (SpeechRecognition):**
    *   Pros: No separate API calls for transcription, potentially lower latency for some use cases.
    *   Cons: Browser support can be inconsistent, transcription quality might vary, might not be as accurate as dedicated models like Whisper. Less control over the STT model.
*   **Other Third-Party Speech-to-Text Services (e.g., Google Cloud Speech-to-Text, AWS Transcribe):**
    *   Pros: Potentially more features or language support.
    *   Cons: Adds another vendor dependency and API key management. Whisper is chosen due to existing OpenAI integration.
*   **Real-time Streaming:**
    *   Could provide a more interactive experience but adds complexity to frontend and backend. Deferred for initial MVP.

## 6. Impact / Risks / Open Questions

*   **Impact:**
    *   Enhanced user experience and accessibility.
    *   New backend dependency on OpenAI Whisper API (though we already use OpenAI for chat completions).
    *   Requires changes to both frontend and backend.
*   **Risks:**
    *   **API Costs:** Whisper API usage will incur costs. Need to monitor.
    *   **Transcription Accuracy:** Accuracy may vary depending on audio quality, accent, background noise.
    *   **Latency:** Time taken for recording, uploading, and transcription might introduce noticeable delay.
    *   **Browser Compatibility:** `MediaRecorder` API is widely supported but should be tested.
    *   **Error Handling:** Robust error handling needed for microphone access issues, API errors, network problems.
*   **Open Questions:**
    *   Should the query be submitted automatically after transcription, or should the user press "Enter" / click "Submit"? (Initial thought: populate field, user confirms by submitting).
    *   What is the maximum recording duration? (Need to decide; Whisper API has limits).
    *   How to best provide feedback for transcription errors? (e.g., toast notification).
    *   Specific audio format to send to Whisper (e.g., .mp3, .wav, .webm)? The API supports multiple formats. `MediaRecorder` often outputs .webm or .ogg. Need to ensure compatibility or perform conversion if necessary.

## 7. (Optional) Implementation Plan / Phases

1.  **Phase 1: Backend Endpoint for Transcription**
    *   Add `/api/transcribe` endpoint to `backend/main.py`.
    *   Integrate OpenAI Whisper API call.
    *   Test with sample audio files (e.g., using `curl` or Postman).
2.  **Phase 2: Frontend Audio Capture & UI**
    *   Add microphone icon and state management to `frontend/src/App.tsx`.
    *   Implement `MediaRecorder` logic for audio capture.
    *   Implement API call to the new backend endpoint.
    *   Populate query input with transcription.
3.  **Phase 3: Refinements & Error Handling**
    *   Add visual feedback for recording/transcribing states.
    *   Implement comprehensive error handling and user notifications.
    *   Test across different browsers.
4.  **Phase 4: Documentation & Testing**
    *   Update `README.md` and any relevant documentation.
    *   Perform thorough end-to-end testing.

## 8. (Optional) Future Work / Follow-on

*   Support for other languages.
*   Real-time streaming transcription.
*   "Wake word" activation.
*   Option to choose STT provider.
*   Investigate client-side audio compression to reduce upload size. 