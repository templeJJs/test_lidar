// Тонкий слой вьюера: bun раздаёт собранный клиент и проксирует API на Python.
//
// Считает по-прежнему Python (`web_viewer.py` + `bag_reader.py` + `zones.py`) —
// это единственная реализация математики. Дублировать её на TS не нужно:
// браузеру всё равно приходит готовый кадр, а клиент — прослойка для отрисовки.
// Bun тут отвечает только за статику и за то, чтобы у страницы и API был один
// origin (иначе пришлось бы прикручивать CORS к Python-серверу).

import { extname, join, normalize, resolve } from 'node:path'

const API_ROUTES = new Set(['/meta', '/frame', '/set', '/version'])

const CONTENT_TYPES: Record<string, string> = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'application/javascript; charset=utf-8',
  '.mjs': 'application/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.map': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.ico': 'image/x-icon',
  '.woff2': 'font/woff2',
}

function parseArgs(argv: string[]) {
  const out = {
    host: '127.0.0.1',
    port: 8788,
    py: process.env.PY_API ?? 'http://127.0.0.1:8765',
    client: resolve(join(import.meta.dir, '..', '..', 'client', 'dist')),
  }
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i]
    const next = () => argv[++i] ?? ''
    if (a === '--host') out.host = next()
    else if (a === '--port') out.port = Number(next())
    else if (a === '--py') out.py = next()
    else if (a === '--client') out.client = resolve(next())
    else if (a === '--help' || a === '-h') {
      console.log(`bun run src/index.ts [--host 127.0.0.1] [--port 8788]
    [--py ${out.py}] [--client ${out.client}]

Статика: собранный клиент (app/client/dist). API /meta /frame /set /version
проксируются на Python-вьюер, который и считает кадры.`)
      process.exit(0)
    }
  }
  return out
}

const args = parseArgs(Bun.argv.slice(2))

async function staticFile(pathname: string): Promise<Response | null> {
  const rel = pathname === '/' ? 'index.html' : pathname.replace(/^\/+/, '')
  const file = normalize(join(args.client, rel))
  if (!file.startsWith(args.client)) return null          // никаких ../
  const bunFile = Bun.file(file)
  if (!(await bunFile.exists())) return null
  const type = CONTENT_TYPES[extname(file).toLowerCase()] ?? 'application/octet-stream'
  return new Response(bunFile, { headers: { 'Content-Type': type } })
}

function hint(): Response {
  return new Response(
    `Клиент не собран: нет ${args.client}\n\n` +
    `Соберите его:  cd app/client && bun run build\n` +
    `Или запустите дев-режим:  cd app/client && bun run dev  (он сам ходит на ${args.py})\n`,
    { status: 503, headers: { 'Content-Type': 'text/plain; charset=utf-8' } },
  )
}

const server = Bun.serve({
  hostname: args.host,
  port: args.port,
  idleTimeout: 120,
  async fetch(req) {
    const url = new URL(req.url)

    if (API_ROUTES.has(url.pathname)) {
      const target = args.py + url.pathname + url.search
      const body = req.method === 'GET' || req.method === 'HEAD' ? undefined : req.body
      try {
        const upstream = await fetch(target, { method: req.method, body })
        return new Response(upstream.body, {
          status: upstream.status,
          headers: {
            'Content-Type': upstream.headers.get('Content-Type') ?? 'application/octet-stream',
            'Cache-Control': 'no-store',
          },
        })
      } catch {
        return new Response(
          JSON.stringify({ error: `Python-вьюер недоступен на ${args.py}` }),
          { status: 502, headers: { 'Content-Type': 'application/json; charset=utf-8' } },
        )
      }
    }

    if (req.method !== 'GET' && req.method !== 'HEAD') {
      return new Response('not found', { status: 404 })
    }

    const found = await staticFile(url.pathname)
    if (found) return found

    // SPA-фолбэк: путь без расширения — отдаём index.html
    if (!extname(url.pathname)) {
      const index = await staticFile('/index.html')
      if (index) return index
    }
    if (url.pathname === '/' || url.pathname === '/index.html') return hint()
    return new Response('not found', { status: 404 })
  },
})

console.log(`Статика : ${args.client}`)
console.log(`API     : ${args.py}  (/meta /frame /set — проксируются)`)
console.log(`Слушаю  : http://${server.hostname}:${server.port}/   (Ctrl+C — стоп)`)