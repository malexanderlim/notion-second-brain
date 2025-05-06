import { useState } from 'react'
import axios from 'axios' // Import axios
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { LoaderCircle, X, Info, CalendarDays, Link as LinkIcon, WandSparkles } from 'lucide-react'; // Added WandSparkles for model
import ReactMarkdown from 'react-markdown'; // Import ReactMarkdown
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
// Import a suitable loader icon if needed later, e.g., from lucide-react
// import { LoaderCircle } from 'lucide-react';

// Define types for the API response
interface SourceDocument {
  title: string;
  url: string;
  date: string; // Made date non-optional for stricter checking, assuming backend always sends it based on logs.
  id?: string;   
}

// Reverted QueryResponse interface
interface QueryResponse {
  answer: string;
  sources: SourceDocument[];
  model_used?: string;
  model_api_id_used?: string; // NEW
  model_provider_used?: string; // NEW
  input_tokens?: number; // NEW
  output_tokens?: number; // NEW
  estimated_cost_usd?: number; // NEW
}

// Constants for suggestion logic
const RECENT_KEYWORDS = ["recently", "latest", "last time", "most recent"];
const YEAR_REGEX = /\b(19|20)\d{2}\b/; // Matches 4-digit years starting 19 or 20
const SUGGESTION_MESSAGE = "Queries about recent events work best with a timeframe (e.g., '...in 2024', '...last month').";

// --- MODIFICATION START: Date formatting utility ---
const formatDate = (dateString: string | undefined): string => {
  if (!dateString) return 'Date not available';
  try {
    // Attempt to create a valid date object. Handles YYYY-MM-DD.
    const date = new Date(dateString + 'T00:00:00'); // Ensure parsing as local date, not UTC
    if (isNaN(date.getTime())) {
      return dateString; // Return original if invalid
    }
    const options: Intl.DateTimeFormatOptions = { year: 'numeric', month: 'long', day: 'numeric' };
    return date.toLocaleDateString(undefined, options);
  } catch (e) {
    return dateString; // Fallback to original string if date parsing fails
  }
};
// --- MODIFICATION END ---

// NEW: Define available models
// These keys (e.g., "gpt-4o") should match the keys in your backend's MODEL_CONFIG
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
const DEFAULT_MODEL = AVAILABLE_MODELS[0].value; // Default to GPT-4o

function App() {
  const [query, setQuery] = useState<string>("");
  const [response, setResponse] = useState<QueryResponse | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [suggestion, setSuggestion] = useState<string | null>(null); // New state for suggestion
  // NEW: State for model selection
  const [selectedModel, setSelectedModel] = useState<string>(DEFAULT_MODEL);
  // NEW: State to display the model used for the response
  const [modelUsedInResponse, setModelUsedInResponse] = useState<string | null>(null);
  // NEW: State for additional response details
  const [modelApiIdUsed, setModelApiIdUsed] = useState<string | null>(null);
  const [modelProviderUsed, setModelProviderUsed] = useState<string | null>(null);
  const [inputTokens, setInputTokens] = useState<number | null>(null);
  const [outputTokens, setOutputTokens] = useState<number | null>(null);
  const [estimatedCost, setEstimatedCost] = useState<number | null>(null);

  const backendUrl = import.meta.env.VITE_BACKEND_URL || "http://localhost:8000"; // Use env var or default

  // Function to check query and set suggestion if needed
  const checkAndSetSuggestion = (submittedQuery: string) => {
    const lowerQuery = submittedQuery.toLowerCase();
    const hasRecentKeyword = RECENT_KEYWORDS.some(keyword => lowerQuery.includes(keyword));
    const hasYear = YEAR_REGEX.test(submittedQuery);
    
    if (hasRecentKeyword && !hasYear) {
      setSuggestion(SUGGESTION_MESSAGE);
    } else {
      setSuggestion(null); // Clear suggestion if conditions not met
    }
  };

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault(); // Prevent default form submission
    if (!query.trim() || isLoading) return; // Don't submit if empty or already loading

    const submittedQuery = query; // Store query before resetting

    setIsLoading(true);
    setError(null);
    setResponse(null);
    setSuggestion(null); // Reset suggestion on new submission
    setModelUsedInResponse(null); // NEW: Reset model used on new submission
    // NEW: Reset additional details on new submission
    setModelApiIdUsed(null);
    setModelProviderUsed(null);
    setInputTokens(null);
    setOutputTokens(null);
    setEstimatedCost(null);

    try {
      const result = await axios.post<QueryResponse>(
        `${backendUrl}/api/query`,
        { 
          query: submittedQuery,
          model_name: selectedModel // NEW: Send selectedModel as model_name
        },
        {
          headers: { 'Content-Type': 'application/json' }
        }
      );
      // --- MODIFIED DEBUGGING ---
      console.log("Axios raw result.data:", result.data); 
      if (result.data.sources && result.data.sources.length > 0) {
        console.log("First source object from Axios:", result.data.sources[0]);
        console.log("Does first source have 'date' property?", result.data.sources[0].hasOwnProperty('date'));
      }
      // --- END DEBUGGING ---
      setResponse(result.data);
      // NEW: Set the model used from the response
      if (result.data.model_used) {
        setModelUsedInResponse(result.data.model_used);
      }
      // NEW: Set additional details from response
      if (result.data.model_api_id_used) setModelApiIdUsed(result.data.model_api_id_used);
      if (result.data.model_provider_used) setModelProviderUsed(result.data.model_provider_used);
      if (result.data.input_tokens !== undefined) setInputTokens(result.data.input_tokens);
      if (result.data.output_tokens !== undefined) setOutputTokens(result.data.output_tokens);
      if (result.data.estimated_cost_usd !== undefined) setEstimatedCost(result.data.estimated_cost_usd);
    } catch (err: any) {
      console.error("API Error:", err);
      let errorMessage = "Failed to fetch response from the backend.";
      if (axios.isAxiosError(err) && err.response) {
        // Extract error detail from backend if available
        errorMessage = err.response.data?.detail || err.message || errorMessage;
      } else if (err instanceof Error) {
        errorMessage = err.message;
      }
      setError(errorMessage);
    } finally {
      setIsLoading(false);
      checkAndSetSuggestion(submittedQuery); // Check for suggestion after request finishes
    }
  };

  const handleClear = () => {
    setQuery("");
    setResponse(null);
    setError(null);
    setSuggestion(null); // Clear suggestion on manual clear
    setModelUsedInResponse(null); // NEW: Clear model used
    setSelectedModel(DEFAULT_MODEL); // NEW: Reset model to default
    // NEW: Clear additional details
    setModelApiIdUsed(null);
    setModelProviderUsed(null);
    setInputTokens(null);
    setOutputTokens(null);
    setEstimatedCost(null);
  };

  // Determine if the clear button should be disabled
  const isClearDisabled = query === "" && response === null && error === null && suggestion === null;

  return (
    <div className="min-h-screen bg-background flex items-center justify-center p-4">
      <div className="w-full max-w-2xl space-y-8">
        <h1 className="text-3xl font-bold text-center">Query Your Second Brain</h1>
        
        <form onSubmit={handleSubmit} className="flex flex-col gap-4">
          <div className="flex gap-2 items-center w-full">
            <Input 
              type="text"
              placeholder="Ask your Notion knowledge base..."
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              disabled={isLoading}
              className="flex-grow"
            />
            <Button type="submit" disabled={isLoading || !query.trim()}>
              {isLoading ? (
                <LoaderCircle className="animate-spin h-5 w-5" /> 
              ) : (
                "Query"
              )}
            </Button>
            <Button 
              type="button"
              variant="outline" 
              size="icon" 
              onClick={handleClear}
              disabled={isLoading || isClearDisabled} 
              aria-label="Clear query and results"
            >
              <X className="h-5 w-5" />
            </Button>
          </div>

          <div className="flex flex-col space-y-1.5">
            <label htmlFor="model-select" className="text-sm font-medium text-foreground">Select Model:</label>
            <Select value={selectedModel} onValueChange={setSelectedModel} disabled={isLoading}>
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

        {/* Suggestion Display */} 
        {suggestion && !isLoading && (
          <Alert>
            <Info className="h-4 w-4" />
            <AlertTitle>Query Suggestion</AlertTitle>
            <AlertDescription>{suggestion}</AlertDescription>
          </Alert>
        )}

        {/* Loading State */}
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

        {/* Error Display - Using Alert */}
        {error && (
           <Alert variant="destructive">
            <AlertTitle>Error</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {/* Response Display - Using ReactMarkdown */}
        {!isLoading && response && (
          <Card className="w-full">
            <CardHeader>
              <CardTitle>Answer</CardTitle>
              {/* Display model used for the response */}
              {modelUsedInResponse && (
                <div className="text-xs text-muted-foreground pt-1 flex items-center">
                  <WandSparkles className="h-3 w-3 mr-1.5" /> 
                  Processed by: {AVAILABLE_MODELS.find(m => m.value === modelUsedInResponse)?.label || modelUsedInResponse}
                </div>
              )}
              {/* NEW: Display additional details */}
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
              {/* Use ReactMarkdown to render the answer */}
              {/* Added prose class for better markdown styling defaults */}
              <div className="prose dark:prose-invert max-w-none mb-6">
                <ReactMarkdown 
                  components={{
                    // Ensure links open in new tab AND are styled
                    a: ({node, ...props}) => (
                      <a 
                        {...props} 
                        target="_blank" 
                        rel="noopener noreferrer" 
                        // Enhanced styling for inline citation links
                        className="text-blue-600 hover:text-blue-700 dark:text-blue-400 dark:hover:text-blue-300 hover:underline focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50 rounded-sm"
                      />
                    ),
                    p: ({node, ...props}) => <p className="mb-3 leading-relaxed" {...props} /> // Improved paragraph spacing
                  }}
                >
                  {response.answer}
                </ReactMarkdown>
              </div>
              
              {/* Display Sources */}
              {response.sources && response.sources.length > 0 && (
                // --- MODIFICATION START: Revert to Simple Bulleted Sources List --- 
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
                // --- MODIFICATION END --- 
              )}
            </CardContent>
          </Card>
        )}

      </div>
    </div>
  )
}

export default App
