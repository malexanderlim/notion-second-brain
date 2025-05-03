import { useState } from 'react'
import axios from 'axios' // Import axios
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"
// Import a suitable loader icon if needed later, e.g., from lucide-react
// import { LoaderCircle } from 'lucide-react';

// Define types for the API response
interface SourceDocument {
  title: string;
  url: string;
}

interface QueryResponse {
  answer: string;
  sources: SourceDocument[];
}

function App() {
  const [query, setQuery] = useState<string>("");
  const [response, setResponse] = useState<QueryResponse | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const backendUrl = import.meta.env.VITE_BACKEND_URL || "http://localhost:8000"; // Use env var or default

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault(); // Prevent default form submission
    if (!query.trim() || isLoading) return; // Don't submit if empty or already loading

    setIsLoading(true);
    setError(null);
    setResponse(null);

    try {
      const result = await axios.post<QueryResponse>(
        `${backendUrl}/api/query`,
        { query },
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
    }
  };

  return (
    <div className="container mx-auto p-4 max-w-2xl min-h-screen flex flex-col items-center pt-10">
      <h1 className="text-3xl font-bold mb-6">Query Your Second Brain</h1>
      
      <form onSubmit={handleSubmit} className="w-full flex gap-2 mb-8">
        <Input 
          type="text"
          placeholder="Ask your Notion knowledge base..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          disabled={isLoading}
          className="flex-grow"
        />
        <Button type="submit" disabled={isLoading}>
          {isLoading ? (
             // Placeholder for loading icon
            <span className="animate-pulse">Thinking...</span> 
            // <LoaderCircle className="animate-spin h-5 w-5" /> 
          ) : (
            "Query"
          )}
        </Button>
      </form>

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

      {/* Error Display */}
      {error && (
        <Card className="w-full border-destructive">
          <CardHeader>
            <CardTitle className="text-destructive">Error</CardTitle>
          </CardHeader>
          <CardContent>
            <p>{error}</p>
          </CardContent>
        </Card>
      )}

      {/* Response Display */}
      {!isLoading && response && (
        <Card className="w-full">
          <CardHeader>
            <CardTitle>Answer</CardTitle>
          </CardHeader>
          <CardContent>
            {/* Format the answer - potentially render markdown later */}
            <p className="whitespace-pre-wrap mb-6">{response.answer}</p>
            
            {response.sources && response.sources.length > 0 && (
              <div>
                <h3 className="font-semibold mb-2">Sources:</h3>
                <ul className="list-disc list-inside space-y-1">
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
  )
}

export default App
