import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { useAuth } from '../../contexts/AuthContext'; // Path to AuthContext
import LogoutButton from '../auth/LogoutButton'; // Import the new component
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { LoaderCircle, X, Info, CalendarDays, Link as LinkIcon, WandSparkles, Mic, Square, MicOff, ArrowUpCircle } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

// Define types for the API response
interface SourceDocument {
  title: string;
  url: string;
  date: string; 
  id?: string;   
}

interface QueryResponse {
  answer: string;
  sources: SourceDocument[];
  model_used?: string;
  model_api_id_used?: string; 
  model_provider_used?: string; 
  input_tokens?: number; 
  output_tokens?: number; 
  estimated_cost_usd?: number; 
}

interface LastUpdatedApiResponse {
  last_updated_timestamp: string | null;
  error?: string | null;
}

const RECENT_KEYWORDS = ["recently", "latest", "last time", "most recent"];
const YEAR_REGEX = /\b(19|20)\d{2}\b/;
const SUGGESTION_MESSAGE = "Queries about recent events work best with a timeframe (e.g., '...in 2024', '...last month').";

const formatDate = (dateString: string | undefined): string => {
  if (!dateString) return 'Date not available';
  try {
    const date = new Date(dateString);
    if (isNaN(date.getTime())) {
      console.warn(`formatDate: Received invalid date string: ${dateString}`);
      return dateString;
    }
    const options: Intl.DateTimeFormatOptions = { 
      year: 'numeric', 
      month: 'long', 
      day: 'numeric', 
      hour: 'numeric', 
      minute: '2-digit', 
    };
    return date.toLocaleDateString(undefined, options);
  } catch (e) {
    console.error(`formatDate: Error parsing date string: ${dateString}`, e);
    return dateString;
  }
};

const AVAILABLE_MODELS = [
  { value: "gpt-4o", label: "GPT-4o (Advanced multimodal, fast, 128K context)" },
  { value: "gpt-4-turbo", label: "GPT-4 Turbo (High-capability, 128K context)" },
  { value: "gpt-4o-mini", label: "GPT-4o mini (Fast, affordable, 128K context)" },
  { value: "gpt-3.5-turbo", label: "GPT-3.5 Turbo (Cost-effective, 16K context)" },
  { value: "claude-3-7-sonnet-20250219", label: "Anthropic Claude 3.7 Sonnet (Most intelligent, 200K context)" },
  { value: "claude-3-5-sonnet-20241022", label: "Anthropic Claude 3.5 Sonnet (High intelligence, 200K context)" },
  { value: "claude-3-opus-20240229", label: "Anthropic Claude 3 Opus (Powerful, complex tasks, 200K context)" },
  { value: "claude-3-5-haiku-20241022", label: "Anthropic Claude 3.5 Haiku (Fastest, 200K context)" },
];
const DEFAULT_MODEL = "gpt-4o-mini";

const getAudioFileExtension = (mimeType: string): string => {
  if (!mimeType) return 'webm'; // Default to webm if mimeType is somehow null/empty
  const parts = mimeType.split('/');
  if (parts.length > 1 && parts[0] === 'audio') {
    const subType = parts[1].split(';')[0]; // Remove codecs if present, e.g., 'webm;codecs=opus'
    // Handle common audio types
    if (subType === 'webm') return 'webm';
    if (subType === 'mp4') return 'mp4'; // For m4a, browsers might report audio/mp4
    if (subType === 'mpeg') return 'mp3';
    if (subType === 'ogg') return 'ogg';
    if (subType === 'wav' || subType === 'wave' || subType === 'x-wav') return 'wav';
    if (subType === 'm4a' || subType === 'x-m4a') return 'm4a';
    // Add more mappings if needed
    return subType; // Fallback to subtype itself if no specific mapping
  }
  return 'webm'; // Default fallback
};

const MainAppLayout: React.FC = () => {
  const { user } = useAuth();

  const [query, setQuery] = useState<string>("");
  const [response, setResponse] = useState<QueryResponse | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [suggestion, setSuggestion] = useState<string | null>(null);
  const [selectedModel, setSelectedModel] = useState<string>(DEFAULT_MODEL);
  const [modelUsedInResponse, setModelUsedInResponse] = useState<string | null>(null);
  const [modelApiIdUsed, setModelApiIdUsed] = useState<string | null>(null);
  const [modelProviderUsed, setModelProviderUsed] = useState<string | null>(null);
  const [inputTokens, setInputTokens] = useState<number | null>(null);
  const [outputTokens, setOutputTokens] = useState<number | null>(null);
  const [estimatedCost, setEstimatedCost] = useState<number | null>(null);
  const [lastUpdatedInfo, setLastUpdatedInfo] = useState<{ timestamp: string | null; error: string | null; loading: boolean }>({ timestamp: null, error: null, loading: true });
  const [isRecording, setIsRecording] = useState<boolean>(false);
  const [isTranscribing, setIsTranscribing] = useState<boolean>(false);
  const [transcriptionError, setTranscriptionError] = useState<string | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const [recordingTime, setRecordingTime] = useState<number>(0);
  const timerIntervalRef = useRef<number | null>(null);
  const autoStopRecordTimeoutRef = useRef<number | null>(null);
  const isCancelledRef = useRef<boolean>(false);

  const MAX_RECORDING_MS = 30000;

  const IS_PRODUCTION = import.meta.env.PROD;
  const backendUrl = IS_PRODUCTION 
    ? "" 
    : import.meta.env.VITE_BACKEND_URL || "http://localhost:8000";

  useEffect(() => {
    const fetchLastUpdatedTimestamp = async () => {
      try {
        const result = await axios.get<LastUpdatedApiResponse>(`${backendUrl}/api/last-updated`, { withCredentials: true });
        if (result.data.error) {
          setLastUpdatedInfo({ timestamp: null, error: result.data.error, loading: false });
        } else if (result.data.last_updated_timestamp) {
          setLastUpdatedInfo({ timestamp: result.data.last_updated_timestamp, error: null, loading: false });
        } else {
          setLastUpdatedInfo({ timestamp: null, error: "Timestamp not available.", loading: false });
        }
      } catch (err: any) {
        console.error("Failed to fetch last updated timestamp:", err);
        if (axios.isAxiosError(err) && err.response?.status === 401) {
             setLastUpdatedInfo({ timestamp: null, error: "Authentication required to fetch last updated data.", loading: false });
        } else {
            setLastUpdatedInfo({ timestamp: null, error: "Failed to fetch last updated data.", loading: false });
        }
      }
    };
    fetchLastUpdatedTimestamp();
  }, [backendUrl]);

  const checkAndSetSuggestion = (submittedQuery: string) => {
    const lowerQuery = submittedQuery.toLowerCase();
    const hasRecentKeyword = RECENT_KEYWORDS.some(keyword => lowerQuery.includes(keyword));
    const hasYear = YEAR_REGEX.test(submittedQuery);
    if (hasRecentKeyword && !hasYear) {
      setSuggestion(SUGGESTION_MESSAGE);
    } else {
      setSuggestion(null);
    }
  };

  const handleToggleRecording = async () => {
    setTranscriptionError(null);
    if (isRecording) {
      if (mediaRecorderRef.current && mediaRecorderRef.current.state === "recording") {
        isCancelledRef.current = false;
        mediaRecorderRef.current.stop();
      }
      if (timerIntervalRef.current) {
        clearInterval(timerIntervalRef.current);
        timerIntervalRef.current = null;
      }
      setIsRecording(false);
    } else {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        
        // Attempt to force WebM format with opus codec
        const preferredOptions = {
          mimeType: 'audio/webm;codecs=opus',
          audioBitsPerSecond: 128000 // Optional: set a preferred bitrate
        };
        let recorderOptions = {}; // Default to empty options

        if (MediaRecorder.isTypeSupported(preferredOptions.mimeType)) {
          console.log(`Using preferred MIME type: ${preferredOptions.mimeType}`);
          recorderOptions = preferredOptions;
        } else {
          // Log if specific codecs of webm are not supported, try broader webm
          if (MediaRecorder.isTypeSupported('audio/webm')) {
            console.warn(`${preferredOptions.mimeType} with opus not supported, trying generic audio/webm.`);
            recorderOptions = { mimeType: 'audio/webm' };
          } else {
            console.warn(`${preferredOptions.mimeType} and audio/webm are not supported. Using browser default format.`);
            // No specific options, browser will use its default
          }
        }

        const recorder = new MediaRecorder(stream, recorderOptions);
        mediaRecorderRef.current = recorder;
        audioChunksRef.current = [];
        setRecordingTime(0);
        isCancelledRef.current = false;
        recorder.ondataavailable = (event) => {
          if (event.data.size > 0) {
            audioChunksRef.current.push(event.data);
          }
        };
        recorder.onstop = async () => {
          stream.getTracks().forEach(track => track.stop());
          setIsRecording(false);
          if (timerIntervalRef.current) {
            clearInterval(timerIntervalRef.current);
            timerIntervalRef.current = null;
          }
          if (autoStopRecordTimeoutRef.current) {
            clearTimeout(autoStopRecordTimeoutRef.current);
            autoStopRecordTimeoutRef.current = null;
          }
          if (isCancelledRef.current) {
            console.log("Recording cancelled by user.");
            audioChunksRef.current = [];
            return;
          }
          if (audioChunksRef.current.length === 0) {
            console.warn("No audio chunks recorded.");
            setTranscriptionError("No audio was recorded. Please try again.");
            return;
          }

          // Use the actual mimeType from the recorder instance if available, otherwise fallback
          const actualMimeType = mediaRecorderRef.current?.mimeType || 'audio/webm';
          console.log(`MediaRecorder actual mimeType onstop: ${actualMimeType}`); // Log the actual mimeType

          const audioBlob = new Blob(audioChunksRef.current, { type: actualMimeType });
          console.log(`Audio recording blob created. Type: ${audioBlob.type}, Size: ${audioBlob.size} bytes`);
          
          audioChunksRef.current = [];
          setIsTranscribing(true);
          setTranscriptionError(null);
          const formData = new FormData();
          const fileExtension = getAudioFileExtension(actualMimeType);
          const fileName = `voice_query.${fileExtension}`;
          formData.append("file", audioBlob, fileName);
          try {
            const transcribeResponse = await axios.post<{ transcription: string, error?: string }>(
              `${backendUrl}/api/transcribe`,
              formData,
              {
                headers: { 'Content-Type': 'multipart/form-data' },
                withCredentials: true,
              }
            );
            if (transcribeResponse.data.error) {
              setTranscriptionError(transcribeResponse.data.error);
            } else if (transcribeResponse.data.transcription) {
              const transcribedText = transcribeResponse.data.transcription;
              setQuery(transcribedText);
              if (transcribedText.trim()) {
                performQuerySubmission(transcribedText, selectedModel);
              }
            }
          } catch (err: any) {
            console.error("Transcription API Error:", err);
            let errorMessage = "Failed to transcribe audio.";
            if (axios.isAxiosError(err) && err.response) {
              errorMessage = err.response.data?.detail || err.message || errorMessage;
            }
            setTranscriptionError(errorMessage);
          } finally {
            setIsTranscribing(false);
            audioChunksRef.current = [];
            setRecordingTime(0);
          }
        };
        mediaRecorderRef.current.start();
        setIsRecording(true);
        timerIntervalRef.current = window.setInterval(() => {
          setRecordingTime((prevTime) => prevTime + 1);
        }, 1000);
        autoStopRecordTimeoutRef.current = window.setTimeout(() => {
          if (mediaRecorderRef.current && mediaRecorderRef.current.state === "recording") {
            mediaRecorderRef.current.stop();
          }
        }, MAX_RECORDING_MS);
      } catch (err) {
        console.error("Error accessing microphone:", err);
        setTranscriptionError("Microphone access denied or microphone not found. Please check permissions.");
        setIsRecording(false);
      }
    }
  };

  const handleCancelRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === "recording") {
      isCancelledRef.current = true;
      mediaRecorderRef.current.stop();
    }
    if (autoStopRecordTimeoutRef.current) {
      clearTimeout(autoStopRecordTimeoutRef.current);
      autoStopRecordTimeoutRef.current = null;
    }
    setIsRecording(false);
    setIsTranscribing(false);
    setTranscriptionError(null);
    audioChunksRef.current = [];
    setRecordingTime(0);
    setQuery("");
  };

  const performQuerySubmission = async (queryToSubmit: string, modelForQuery: string) => {
    if (!queryToSubmit.trim()) return;
    setIsLoading(true);
    setError(null);
    setResponse(null);
    setModelUsedInResponse(null);
    setModelApiIdUsed(null);
    setModelProviderUsed(null);
    setInputTokens(null);
    setOutputTokens(null);
    setEstimatedCost(null);
    setIsRecording(false);
    setIsTranscribing(false);
    try {
      const result = await axios.post<QueryResponse>(
        `${backendUrl}/api/query`,
        { 
          query: queryToSubmit,
          model_name: modelForQuery
        },
        {
          headers: { 'Content-Type': 'application/json' },
          withCredentials: true,
        }
      );
      setResponse(result.data);
      if (result.data.model_used) setModelUsedInResponse(result.data.model_used);
      if (result.data.model_api_id_used) setModelApiIdUsed(result.data.model_api_id_used);
      if (result.data.model_provider_used) setModelProviderUsed(result.data.model_provider_used);
      if (result.data.input_tokens !== undefined) setInputTokens(result.data.input_tokens);
      if (result.data.output_tokens !== undefined) setOutputTokens(result.data.output_tokens);
      if (result.data.estimated_cost_usd !== undefined) setEstimatedCost(result.data.estimated_cost_usd);
    } catch (err: any) {
      console.error("API Error:", err);
      let errorMessage = "Failed to fetch response from the backend.";
      if (axios.isAxiosError(err) && err.response) {
        errorMessage = err.response.data?.detail || err.message || errorMessage;
         if (err.response?.status === 401) {
            errorMessage = "Authentication failed or session expired. Please login again.";
        }
      }
      setError(errorMessage);
    } finally {
      setIsLoading(false);
      checkAndSetSuggestion(queryToSubmit);
    }
  };

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (isLoading || isRecording || isTranscribing) return;
    performQuerySubmission(query, selectedModel);
  };

  const handleClear = () => {
    setQuery("");
    setResponse(null);
    setError(null);
    setSuggestion(null);
    setModelUsedInResponse(null);
    setSelectedModel(DEFAULT_MODEL);
    setModelApiIdUsed(null);
    setModelProviderUsed(null);
    setInputTokens(null);
    setOutputTokens(null);
    setEstimatedCost(null);
    setIsRecording(false);
    setIsTranscribing(false);
    setTranscriptionError(null);
  };

  const isClearDisabled = query === "" && response === null && error === null && suggestion === null;

  return (
    <div>
      <header className="p-4 bg-muted/30 flex justify-between items-center sticky top-0 z-50 backdrop-blur-sm">
        <h1 className="text-xl font-semibold">My Second Brain</h1>
        {user && (
          <div className="flex items-center space-x-2 bg-muted p-2 rounded-lg">
            {user.picture && <img src={user.picture} alt="User" className="w-8 h-8 rounded-full" />}
            <span className="text-sm font-medium">{user.name || user.email}</span>
            <LogoutButton className="ml-auto" />
          </div>
        )}
      </header>
      <main className="p-4">
        <div className="min-h-screen bg-background flex items-center justify-center p-4">
          <div className="w-full max-w-2xl space-y-8">
            {!isLoading && !response && (
              <div className="flex justify-center mb-4">
                <img src="/Notion_Second_Brain.png" alt="Notion Second Brain Logo" className="h-12 w-12 md:h-20 md:w-20 object-contain" />
              </div>
            )}
            <div className="text-center text-xs text-muted-foreground">
              {lastUpdatedInfo.loading ? (
                <span>Loading last updated info...</span>
              ) : lastUpdatedInfo.error ? (
                <span className="text-red-500">Error: {lastUpdatedInfo.error}</span>
              ) : lastUpdatedInfo.timestamp ? (
                <span>Last Synced Entry: {formatDate(lastUpdatedInfo.timestamp)}</span>
              ) : (
                <span>Last synced info not available.</span>
              )}
            </div>

            <form onSubmit={handleSubmit} className="flex flex-col gap-4">
              <div className="flex flex-col sm:flex-row gap-2 items-center w-full">
                <Input 
                  type="text"
                  placeholder="Ask your Notion knowledge base..."
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  className="flex-grow"
                  disabled={isLoading || isRecording || isTranscribing}
                />
                {!isRecording && !isTranscribing && (
                  <Button 
                    type="button" 
                    variant="ghost" 
                    size="icon" 
                    onClick={handleToggleRecording}
                    disabled={isLoading} 
                    aria-label="Start voice recording"
                    className={`p-2`}
                  >
                    {transcriptionError ? <MicOff className="h-5 w-5 text-red-500" /> : <Mic className="h-5 w-5" />}
                  </Button>
                )}
                {!isRecording && !isTranscribing && (
                  <>
                    <Button type="submit" disabled={isLoading || !query.trim() || isRecording || isTranscribing}>
                      {isLoading ? (
                        <LoaderCircle className="animate-spin h-5 w-5" /> 
                      ) : (
                        "Query"
                      )}
                    </Button>
                    <Button 
                      type="button" 
                      onClick={handleClear} 
                      variant="outline"
                      disabled={isClearDisabled || isLoading || isRecording || isTranscribing}
                    >
                      <X className="h-5 w-5" /> 
                      <span className="ml-2 sm:inline hidden">Clear</span>
                    </Button>
                  </>
                )}
              </div>

              {isRecording && (
                <div className="flex items-center justify-between gap-2 p-2 border rounded-md bg-muted/40">
                  <Button 
                    type="button"
                    variant="ghost"
                    size="icon"
                    onClick={handleCancelRecording}
                    aria-label="Cancel voice recording"
                    className="p-2 text-red-500"
                  >
                    <X className="h-6 w-6" />
                  </Button>
                  <div className="flex flex-col items-center">
                    <span className="text-xs text-muted-foreground">Recording...</span>
                    <span className="text-lg font-semibold tabular-nums">
                      {Math.floor(recordingTime / 60)}:{String(recordingTime % 60).padStart(2, '0')}
                    </span>
                  </div>
                  <Button 
                    type="button"
                    variant="ghost"
                    size="icon"
                    onClick={handleToggleRecording} 
                    aria-label="Send voice recording"
                    className="p-2 text-green-500"
                  >
                    <ArrowUpCircle className="h-6 w-6" />
                  </Button>
                </div>
              )}

              {transcriptionError && (
                <Alert variant="destructive" className="mt-2">
                  <MicOff className="h-4 w-4" />
                  <AlertTitle>Transcription Error</AlertTitle>
                  <AlertDescription>
                    {transcriptionError}
                  </AlertDescription>
                </Alert>
              )}

              <div className="flex flex-col space-y-1.5">
                <label htmlFor="model-select" className="text-sm font-medium text-foreground">Select Model:</label>
                <Select value={selectedModel} onValueChange={setSelectedModel} disabled={isLoading || isRecording || isTranscribing}>
                  <SelectTrigger id="model-select" className="w-full">
                    <SelectValue placeholder="Select a model" />
                  </SelectTrigger>
                  <SelectContent>
                    {AVAILABLE_MODELS.map(model => (
                      <SelectItem key={model.value} value={model.value}>
                        {model.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </form>

            {suggestion && !isLoading && (
              <Alert>
                <Info className="h-4 w-4" />
                <AlertTitle>Query Suggestion</AlertTitle>
                <AlertDescription>{suggestion}</AlertDescription>
              </Alert>
            )}

            {isLoading && (
              <Card className="w-full">
                <CardHeader>
                  <Skeleton className="h-6 w-3/4 mb-2" />
                  <Skeleton className="h-4 w-1/2" />
                </CardHeader>
                <CardContent className="space-y-4">
                  <Skeleton className="h-4 w-full" />
                  <Skeleton className="h-4 w-full" />
                  <Skeleton className="h-4 w-5/6" />
                  <div className="pt-4">
                    <Skeleton className="h-4 w-1/4 mb-2" />
                    <Skeleton className="h-4 w-full" />
                    <Skeleton className="h-4 w-full" />
                  </div>
                </CardContent>
              </Card>
            )}

            {error && (
              <Alert variant="destructive">
                <AlertTitle>Error</AlertTitle>
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}

            {!isLoading && response && (
              <Card className="w-full">
                <CardHeader>
                  <CardTitle>Answer</CardTitle>
                  {modelUsedInResponse && (
                    <div className="text-xs text-muted-foreground pt-1 flex items-center">
                      <WandSparkles className="h-3 w-3 mr-1.5" /> 
                      Processed by: {AVAILABLE_MODELS.find(m => m.value === modelUsedInResponse)?.label || modelUsedInResponse}
                    </div>
                  )}
                  {modelProviderUsed && (
                    <div className="text-xs text-muted-foreground pt-0.5">Provider: {modelProviderUsed}</div>
                  )}
                  {(inputTokens !== null || outputTokens !== null) && (
                    <div className="text-xs text-muted-foreground pt-0.5">
                      Tokens: {inputTokens ?? 'N/A'} (prompt) / {outputTokens ?? 'N/A'} (completion)
                    </div>
                  )}
                  {estimatedCost !== null && (
                    <div className="text-xs text-muted-foreground pt-0.5">
                      Estimated Cost: ${estimatedCost.toFixed(6)} 
                    </div>
                  )}
                </CardHeader>
                <CardContent>
                  <div className="prose dark:prose-invert max-w-none mb-6">
                    <ReactMarkdown 
                      components={{
                        a: ({node, ...props}) => (
                          <a 
                            {...props} 
                            target="_blank" 
                            rel="noopener noreferrer" 
                            className="text-blue-600 hover:text-blue-700 dark:text-blue-400 dark:hover:text-blue-300 hover:underline focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50 rounded-sm"
                          />
                        ),
                        p: ({node, ...props}) => <p className="mb-3 leading-relaxed" {...props} />
                      }}
                    >
                      {response.answer}
                    </ReactMarkdown>
                  </div>
                  
                  {response.sources && response.sources.length > 0 && (
                    <div className="mt-8 pt-6 border-t border-slate-200 dark:border-slate-700">
                      <h3 className="text-base font-semibold mb-3 text-slate-700 dark:text-slate-300">Sources Used:</h3>
                      <ul className="list-disc list-inside space-y-2 text-sm">
                        {response.sources
                          .slice() 
                          .sort((a, b) => {
                            if (!a.date || !b.date) return 0; 
                            try {
                              return new Date(a.date + 'T00:00:00').getTime() - new Date(b.date + 'T00:00:00').getTime(); 
                            } catch (e) {
                              return 0; 
                            }
                          })
                          .map((source) => (
                          <li key={source.id || source.url} className="text-slate-600 dark:text-slate-400">
                            <a 
                              href={source.url}
                              target="_blank" 
                              rel="noopener noreferrer"
                              className="text-blue-600 hover:underline dark:text-blue-400 visited:text-purple-600 dark:visited:text-purple-400"
                            >
                              {source.title || "Untitled Source"}
                            </a>
                            {source.date && (
                              <span className="ml-2 text-xs text-slate-500 dark:text-slate-500">
                                ({formatDate(source.date)})
                              </span>
                            )}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </CardContent>
              </Card>
            )}
          </div>
        </div>
      </main>
    </div>
  );
};

export default MainAppLayout; 