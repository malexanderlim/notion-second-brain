import { useState } from 'react'
import axios from 'axios' // Import axios
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { LoaderCircle, X, Info } from 'lucide-react'; // Import LoaderCircle, X icon, and Info icon
import ReactMarkdown from 'react-markdown'; // Import ReactMarkdown
// Import a suitable loader icon if needed later, e.g., from lucide-react
// import { LoaderCircle } from 'lucide-react';

// Define types for the API response
interface SourceDocument {
  title: string;
  url: string;
}

// Reverted QueryResponse interface
interface QueryResponse {
  answer: string;
  sources: SourceDocument[];
}

// Constants for suggestion logic
const RECENT_KEYWORDS = ["recently", "latest", "last time", "most recent"];
const YEAR_REGEX = /\b(19|20)\d{2}\b/; // Matches 4-digit years starting 19 or 20
const SUGGESTION_MESSAGE = "Queries about recent events work best with a timeframe (e.g., '...in 2024', '...last month').";

function App() {
  const [query, setQuery] = useState<string>("");
  const [response, setResponse] = useState<QueryResponse | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [suggestion, setSuggestion] = useState<string | null>(null); // New state for suggestion

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

    try {
      const result = await axios.post<QueryResponse>(
        `${backendUrl}/api/query`,
        { query: submittedQuery }, // Use the stored query
        {
          headers: { 'Content-Type': 'application/json' }
        }
      );
      setResponse(result.data);
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
  };

  // Determine if the clear button should be disabled
  const isClearDisabled = query === "" && response === null && error === null && suggestion === null;

  return (
    <div className="min-h-screen bg-background flex items-center justify-center p-4">
      <div className="w-full max-w-2xl space-y-8">
        <h1 className="text-3xl font-bold text-center">Query Your Second Brain</h1>
        
        <form onSubmit={handleSubmit} className="flex gap-2 items-center">
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
            type="button" // Important: type="button" prevents form submission
            variant="outline" 
            size="icon" // Use icon size for a smaller button
            onClick={handleClear}
            disabled={isLoading || isClearDisabled} // Disable when loading or nothing to clear
            aria-label="Clear query and results"
          >
            <X className="h-5 w-5" />
          </Button>
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
            </CardHeader>
            <CardContent>
              {/* Use ReactMarkdown to render the answer */}
              {/* Added prose class for better markdown styling defaults */}
              <div className="prose dark:prose-invert max-w-none mb-6">
                <ReactMarkdown 
                  components={{
                    // Ensure links open in new tab AND are styled blue
                    a: ({node, ...props}) => (
                      <a 
                        {...props} 
                        target="_blank" 
                        rel="noopener noreferrer" 
                        className="text-blue-600 hover:underline dark:text-blue-400"
                      />
                    )
                  }}
                >
                  {response.answer}
                </ReactMarkdown>
              </div>
              
              {/* Display Sources */}
              {response.sources && response.sources.length > 0 && (
                <div>
                  <h3 className="font-semibold mb-2 text-sm text-muted-foreground">Sources Used:</h3>
                  <ul className="list-disc list-inside space-y-1 text-sm">
                    {response.sources.map((source, index) => (
                      <li key={index}>
                        <a 
                          href={source.url}
                          target="_blank" 
                          rel="noopener noreferrer"
                          className="text-blue-600 hover:underline dark:text-blue-400"
                        >
                          {source.title || "Untitled Source"}
                        </a>
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
  )
}

export default App
