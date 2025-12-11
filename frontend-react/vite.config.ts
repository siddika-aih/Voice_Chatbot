import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import _path from 'path'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/ws': {
        target: 'ws://localhost:8000',
        ws: true
      },
      '/api': 'http://localhost:8000'
    }
  },
  build: {
    outDir: '../dist',
    emptyOutDir: true
  },
  base: './'  // Production ke liye
})