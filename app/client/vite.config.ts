import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { fileURLToPath, URL } from 'node:url'
import { defineConfig } from 'vite'

// API считает Python-вьюер (web_viewer.py), клиент — только рисует.
// В дев-режиме Vite проксирует /meta /frame /set /version прямо на него; в проде
// собранный клиент раздаёт тонкий bun-сервер (app/server), который те же ручки
// проксирует на Python — так у страницы и API один origin, без CORS.
const API = process.env.API_URL ?? 'http://127.0.0.1:8765'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: { '@': fileURLToPath(new URL('./src', import.meta.url)) },
  },
  server: {
    port: 5173,
    proxy: {
      '/meta': API,
      '/objects': API,
      '/frame': API,
      '/set': API,
      '/version': API,
      // Разметка: ход записи и руки /label/* (предложение, трассировка, запись).
      '/motion': API,
      '/label': API,
    },
  },
})