import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  optimizeDeps: {
    include: ['react-plotly.js', 'plotly.js'], // Ensure Plotly is pre-bundled
  },
  build: {
    rollupOptions: {
      external: [], // Explicitly say nothing is external
    },
  },
  preview: {
    host: true,
    port: process.env.PORT ? parseInt(process.env.PORT) : 3000,
    allowedHosts: ['greekcheck.onrender.com'], // Your Render domain
  },
})
