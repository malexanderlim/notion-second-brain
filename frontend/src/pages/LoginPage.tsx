import React from 'react';
import { Button } from '@/components/ui/button';
// import { useAuth } from '../contexts/AuthContext'; // Might need later if login status is shown here

const LoginPage: React.FC = () => {
  // const { login } = useAuth(); // Actual login is backend redirect
  const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

  const handleLogin = () => {
    // Redirect to the backend Google login endpoint
    window.location.href = `${API_BASE_URL}/api/auth/login/google`;
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-background p-4">
      <div className="w-full max-w-md space-y-6 text-center">
        <h1 className="text-3xl font-bold">Welcome to Your Second Brain</h1>
        <p className="text-muted-foreground">
          Please log in with your Google account to continue.
        </p>
        <Button onClick={handleLogin} className="w-full" size="lg">
          {/* Consider adding a Google icon here */}
          Login with Google
        </Button>
        {/* You could add a loading indicator here if needed, though redirect is usually fast */}
      </div>
    </div>
  );
};

export default LoginPage; 