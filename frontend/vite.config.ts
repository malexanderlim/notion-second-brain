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
      // Define the @ alias to point to the src directory directly
      // path.resolve(__dirname, './src') ensures it's an absolute path
      { find: '@', replacement: path.resolve(__dirname, './src') },
    ],
  },
})
