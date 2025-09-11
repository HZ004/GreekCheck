import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  preview: {
    host: true,
    port: process.env.PORT ? parseInt(process.env.PORT) : 3000,
    allowedHosts: ['greekcheck.onrender.com'] // Add your Render domain here
  }
})
