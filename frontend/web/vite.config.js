// frontend/web/vite.config.js
import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig(({ mode }) => {
  // Load env so VITE_* from .env files are visible here too
  const env = loadEnv(mode, process.cwd(), '');

  // Dev-only proxy targets (override with env vars if needed)
  const API_TARGET =
    env.VITE_API_PROXY || 'https://api.fwvgoldmindai.com';
  const COMPUTE_TARGET =
    env.VITE_COMPUTE_PROXY || 'https://goldmind-compute-884387776097.us-central1.run.app';

  return {
    plugins: [react()],
    server: {
      // Mirror Netlify redirects in dev:
      //  /api/*     -> API_TARGET/*
      //  /compute/* -> COMPUTE_TARGET/*
      proxy: {
        '/api': {
          target: API_TARGET,
          changeOrigin: true,
          secure: true,
          rewrite: (path) => path.replace(/^\/api/, ''),
        },
        '/compute': {
          target: COMPUTE_TARGET,
          changeOrigin: true,
          secure: true,
          rewrite: (path) => path.replace(/^\/compute/, ''),
        },
      },
    },
  };
});
