import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { fileURLToPath, URL } from 'node:url'
import { defineConfig } from 'vite'

// Bun-сервер API. В дев-режиме Vite проксирует на него /meta, /frame, /set,
// /version, поэтому клиент всегда ходит на относительные пути и в прод-сборке
// (когда Bun сам раздаёт собранный клиент) ничего менять не нужно.
const API = process.env.API_URL ?? 'http://127.0.0.1:8788'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: { '@': fileURLToPath(new URL('./src', import.meta.url)) },
  },
  server: {
    port: 5173,
    proxy: {
      '/meta': API,
      '/frame': API,
      '/set': API,
      '/version': API,
    },
  },
})