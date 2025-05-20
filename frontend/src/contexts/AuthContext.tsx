import React, { createContext, useState, useContext, useEffect, ReactNode } from 'react';

// Define the shape of the user object
interface User {
  email: string;
  name: string;
  picture?: string;
}

// Define the shape of the AuthContext state
interface AuthContextType {
  isAuthenticated: boolean;
  user: User | null;
  isLoading: boolean; // To handle async auth state loading
  login: (userData: User) => void; // Kept simple, actual login flow is via backend redirect
  logout: () => void; // Kept simple, actual logout flow is via backend redirect
  checkAuthState: () => Promise<void>; // Function to call /api/auth/me
}

// Create the AuthContext with a default undefined value to prevent accidental use outside provider
const AuthContext = createContext<AuthContextType | undefined>(undefined);

// Define props for the AuthProvider
interface AuthProviderProps {
  children: ReactNode;
}

// Create the AuthProvider component
export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
  const [isAuthenticated, setIsAuthenticated] = useState<boolean>(false);
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true); // Start with loading true

  const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

  // Function to check authentication state (e.g., by calling /api/auth/me)
  const checkAuthState = React.useCallback(async () => {
    setIsLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/api/auth/me`, {
        method: 'GET',
        credentials: 'include', // Important to send cookies
        headers: {
          'Accept': 'application/json',
        },
      });

      if (response.ok) {
        const data = await response.json();
        if (data.authenticated && data.user) {
          setIsAuthenticated(true);
          setUser(data.user);
        } else {
          setIsAuthenticated(false);
          setUser(null);
        }
      } else {
        // Handle non-ok responses (e.g., 401 if not authenticated)
        setIsAuthenticated(false);
        setUser(null);
        // console.warn('Auth check failed or user not authenticated:', response.status);
      }
    } catch (error) {
      console.error('Error checking auth state:', error);
      setIsAuthenticated(false);
      setUser(null);
    }
    setIsLoading(false);
  }, [API_BASE_URL]);
  
  // Call checkAuthState when the provider mounts
  useEffect(() => {
    checkAuthState();
  }, [checkAuthState]);

  // Placeholder login function - actual login redirects to backend
  // This might be called by the frontend *after* a successful redirect from backend if needed
  // but for now, checkAuthState will handle re-syncing state.
  const login = (userData: User) => {
    setIsAuthenticated(true);
    setUser(userData);
  };

  // Placeholder logout function - actual logout redirects to backend
  // This function would be called to clear frontend state after backend logout is initiated.
  const logout = () => {
    setIsAuthenticated(false);
    setUser(null);
    // Optionally, redirect to login page or trigger backend logout if not already done
    // window.location.href = `${API_BASE_URL}/api/auth/logout`; // Example direct navigation
  };

  const contextValue = {
    isAuthenticated,
    user,
    isLoading,
    login,
    logout,
    checkAuthState
  };

  return (
    <AuthContext.Provider value={contextValue}>
      {children}
    </AuthContext.Provider>
  );
};

// Custom hook to use the AuthContext
export const useAuth = (): AuthContextType => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}; 