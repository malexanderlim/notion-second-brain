import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { useAuth } from './contexts/AuthContext';
import LoginPage from './pages/LoginPage';
import MainAppLayout from './components/layout/MainAppLayout';
import { LoaderCircle } from 'lucide-react'; // For loading indicator

const App: React.FC = () => {
  const { isAuthenticated, isLoading, user } = useAuth();

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <LoaderCircle className="animate-spin h-12 w-12 text-primary" />
        <p className="ml-4 text-lg">Loading application...</p>
      </div>
    );
  }

  return (
    <Routes>
      <Route
        path="/login"
        element={isAuthenticated ? <Navigate to="/" replace /> : <LoginPage />}
      />
      <Route
        path="/*"
        element={isAuthenticated ? <MainAppLayout /> : <Navigate to="/login" replace />}
      />
    </Routes>
  );
};

export default App;
