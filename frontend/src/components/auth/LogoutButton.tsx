import React from 'react';
import axios from 'axios'; // Import axios
import { Button } from "@/components/ui/button"; // Assuming this is your ShadCN Button

interface LogoutButtonProps {
  className?: string;
}

const LogoutButton: React.FC<LogoutButtonProps> = ({ className }) => {
  const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

  const handleLogout = async () => { // Make it async
    try {
      // Use axios.post and expect the backend to handle redirect on success
      await axios.post(`${API_BASE_URL}/api/auth/logout`, {}, { withCredentials: true });
      // The backend will redirect, but as a fallback or if it doesn't, 
      // we can force a reload to a known logged-out page (e.g., login page or root).
      // The backend redirect is preferred.
      // If the backend redirects properly, this next line might not even be hit or might be redundant.
      window.location.href = import.meta.env.VITE_FRONTEND_LOGOUT_URL || '/'; // Or a specific login page URL
    } catch (error) {
      console.error('Logout failed:', error);
      // Handle logout error, maybe show a notification to the user
      // For now, we can still try to redirect to a safe page
      window.location.href = import.meta.env.VITE_FRONTEND_LOGOUT_URL || '/';
    }
  };

  return (
    <Button variant="outline" onClick={handleLogout} className={className}>
      Logout
    </Button>
  );
};

export default LogoutButton; 