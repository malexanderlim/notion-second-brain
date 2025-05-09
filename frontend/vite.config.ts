import path from 'path'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from "@tailwindcss/vite"

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [
    react(), 
    tailwindcss(),
  ],
  resolve: {
    alias: [
      // Hyper-specific alias for the problematic import
      { 
        find: '@/lib/utils', 
        replacement: path.resolve(__dirname, './src/lib/utils.ts') 
      },
      // Keep the general @ alias as a fallback for other imports, but ensure it's processed after the specific one
      { 
        find: '@', 
        replacement: path.resolve(__dirname, './src') 
      },
    ],
  },
})
