import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        // Local dev: forward /api/* to your Cloud Run API (or custom domain)
        target: process.env.VITE_API_PROXY || 'https://api.fwvgoldmindai.com',
        changeOrigin: true,
        secure: true,
        rewrite: p => p.replace(/^\/api/, ''),
      },
    },
  },
})
