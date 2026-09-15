/**
 * Bun.serve для вьюера: /meta, /frame, /set, /version + раздача собранного клиента.
 *
 * Замена stdlib-сервера из web_viewer.py. Отличия по делу: вместо статики из web/
 * раздаётся сборка клиента (--client-dir, обычно app/client/dist) -- именно её
 * открывает браузер; если сборки нет, на / отдаётся страница-заглушка с подсказкой
 * запустить Vite, чтобы API можно было проверить без клиента.
 *
 * Запросы обслуживаются параллельно (Bun запускает fetch конкурентно), состояние
 * вьюера после замера только читается -- см. http.ts.
 */

import { existsSync, readdirSync, statSync } from 'node:fs'
import { extname, join, resolve, sep } from 'node:path'

import { findDb3 } from './data/bagReader.ts'
import { COLOR_MODES } from './data/bagReader.ts'
import { parseArgsOrExit, resolveRepoPath } from './cli.ts'
import type { CliArgs } from './cli.ts'
import { Viewer } from './http.ts'

const CONTENT_TYPES: Record<string, string> = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'application/javascript; charset=utf-8',
  '.mjs': 'application/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.map': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.gif': 'image/gif',
  '.webp': 'image/webp',
  '.ico': 'image/x-icon',
  '.woff': 'font/woff',
  '.woff2': 'font/woff2',
  '.ttf': 'font/ttf',
  '.txt': 'text/plain; charset=utf-8',
  '.wasm': 'application/wasm',
}

function contentType(path: string): string {
  return CONTENT_TYPES[extname(path).toLowerCase()] ?? 'application/octet-stream'
}

/** Обычный положительный ответ с запретом кэша (страница сама себя перезагружает). */
function send(body: string | Uint8Array | Blob, type: string, status = 200, headers: Record<string, string> = {}): Response {
  return new Response(body, {
    status,
    headers: { 'Content-Type': type, 'Cache-Control': 'no-store', ...headers },
  })
}

function json(value: unknown, status = 200): Response {
  return send(JSON.stringify(value), 'application/json; charset=utf-8', status)
}

/** Список файлов клиента относительно его каталога (рекурсивно, по алфавиту). */
function clientFiles(clientDir: string): string[] {
  const names: string[] = []
  const walk = (dir: string, prefix: string): void => {
    let entries
    try {
      entries = readdirSync(dir, { withFileTypes: true })
    } catch {
      return
    }
    entries.sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0))
    for (const entry of entries) {
      if (entry.isDirectory()) walk(join(dir, entry.name), `${prefix}${entry.name}/`)
      else names.push(`${prefix}${entry.name}`)
    }
  }
  walk(clientDir, '')
  return names
}

/**
 * Хэш mtime+размера файлов клиента: страница опрашивает /version и сама
 * перезагружается, когда сборка обновилась. Файлов нет -- отдаём стабильную
 * заглушку (как Python с пометкой missing), но не падаем.
 */
export function assetsVersion(clientDir: string): string {
  const names = clientFiles(clientDir)
  const parts = names.length
    ? names.map((name) => {
        try {
          const st = statSync(join(clientDir, name))
          return `${name}:${st.mtimeMs}:${st.size}`
        } catch {
          return `${name}:missing`
        }
      })
    : ['index.html:missing', 'assets:missing']
  return new Bun.CryptoHasher('sha1').update(parts.join('|')).digest('hex').slice(0, 16)
}

/** Страница-заглушка: API живо, клиент не собран. */
function stubPage(clientDir: string, port: number): string {
  return `<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Lidar viewer: клиент не собран</title>
<style>
  body { font: 15px/1.5 system-ui, sans-serif; max-width: 46rem; margin: 4rem auto; padding: 0 1.5rem; color: #e8e8ea; background: #16181d; }
  code { background: #23262e; padding: .1rem .35rem; border-radius: .25rem; font-size: .9em; }
  a { color: #6cc7ff; }
  h1 { font-size: 1.4rem; }
</style>
</head>
<body>
<h1>Клиент не собран</h1>
<p>API работает на порту ${port}, но каталога со сборкой клиента нет: <code>${clientDir}</code>.</p>
<p>Собрать: <code>cd app/client &amp;&amp; bun install &amp;&amp; bun run build</code>, либо запустить
Vite в дев-режиме: <code>cd app/client &amp;&amp; bun run dev</code> — он проксирует
<code>/meta</code>, <code>/frame</code>, <code>/set</code> и <code>/version</code> на этот сервер.</p>
<p>Проверить API руками: <a href="/meta">/meta</a>, <a href="/frame?idx=0&amp;mode=zones">/frame?idx=0&amp;mode=zones</a>,
<a href="/version">/version</a>.</p>
</body>
</html>
`
}

/** Файл из каталога клиента; null -- если файла нет или путь уводит наружу. */
function staticFile(clientDir: string, urlPath: string): Response | null {
  let rel: string
  try {
    rel = decodeURIComponent(urlPath)
  } catch {
    return null
  }
  if (rel === '/' || rel === '') rel = 'index.html'
  rel = rel.replace(/^\/+/, '')
  if (rel === '' || rel.includes('\0')) return null
  const full = resolve(clientDir, rel)
  const root = resolve(clientDir)
  if (full !== root && !full.startsWith(root + sep)) return null
  try {
    if (!statSync(full).isFile()) return null
  } catch {
    return null
  }
  const file = Bun.file(full)
  // Хэшированные ассеты Vite можно кэшировать надолго, html -- никогда.
  const immutable = rel.startsWith(`assets${sep}`) || rel.startsWith('assets/')
  return send(file, contentType(full), 200, immutable ? { 'Cache-Control': 'public, max-age=31536000, immutable' } : {})
}

function openBrowser(url: string): void {
  const cmd = process.platform === 'win32'
    ? ['cmd', '/c', 'start', '', url]
    : process.platform === 'darwin' ? ['open', url] : ['xdg-open', url]
  try {
    Bun.spawn(cmd, { stdout: 'ignore', stderr: 'ignore' })
  } catch {
    console.warn(`не удалось открыть браузер, откройте вручную: ${url}`)
  }
}

/** Целое из query-строки, как int() в Python: не разобралось -- null. */
function intOrNull(raw: string | null): number | null {
  if (raw === null || raw.trim() === '') return null
  if (!/^[+-]?\d+$/.test(raw.trim())) throw new Error(`не целое число: ${JSON.stringify(raw)}`)
  return Number.parseInt(raw, 10)
}

function floatOrNull(raw: string | null): number | null {
  if (raw === null || raw.trim() === '') return null
  const value = Number(raw)
  if (!Number.isFinite(value)) throw new Error(`не число: ${JSON.stringify(raw)}`)
  return value
}

function makeHandler(viewer: Viewer, clientDir: string, args: CliArgs) {
  const stub = stubPage(clientDir, args.port)

  return async function handle(req: Request): Promise<Response> {
    if (req.method !== 'GET' && req.method !== 'HEAD') {
      return send('method not allowed', 'text/plain; charset=utf-8', 405)
    }
    const url = new URL(req.url)
    const path = url.pathname

    switch (path) {
      case '/meta':
        return json(viewer.meta())
      case '/frame': {
        let idx = 0
        try {
          idx = intOrNull(url.searchParams.get('idx')) ?? 0
        } catch {
          idx = 0 // Python: ValueError -> idx = 0
        }
        const requested = url.searchParams.get('mode') ?? viewer.mode
        const mode = (COLOR_MODES as readonly string[]).includes(requested) ? requested : viewer.mode
        try {
          return send(viewer.frameBytes(idx, mode), 'application/octet-stream')
        } catch (err) {
          const message = err instanceof Error ? err.message : String(err)
          return json({ error: message }, 500)
        }
      }
      case '/set': {
        try {
          const params = url.searchParams
          return json(viewer.setParams({
            axis: floatOrNull(params.get('axis')),
            gaugeMm: intOrNull(params.get('gauge')),
            contact: intOrNull(params.get('contact')),
            offset: floatOrNull(params.get('offset')),
            side: intOrNull(params.get('side')),
          }))
        } catch (err) {
          const message = err instanceof Error ? err.message : String(err)
          return json({ error: message }, 400)
        }
      }
      case '/version':
        return json({ version: assetsVersion(clientDir) })
      case '/favicon.ico':
        return send('', 'image/x-icon', 204)
      default:
        break
    }

    if (path === '/' || path === '/index.html') {
      const page = staticFile(clientDir, '/index.html')
      return page ?? send(stub, 'text/html; charset=utf-8', 200)
    }
    return staticFile(clientDir, path) ?? send('not found', 'text/plain; charset=utf-8', 404)
  }
}

/** Сводка при старте -- как печатает web_viewer.py. */
function printSummary(viewer: Viewer, args: CliArgs, url: string): void {
  const p = viewer.params
  const rails = [p.axisX - p.gauge / 2, p.axisX + p.gauge / 2]
  const spans = viewer.spans.map(([a, b]) => `[${a.toFixed(1)}, ${b.toFixed(1)}]`).join(', ')
  const blockedTotal = viewer.spans.reduce((sum, [a, b]) => sum + (b - a), 0)
  console.log(`ZONES floor : z = ${p.floorA.toFixed(4)} + ${p.floorB.toFixed(6)}*y  `
    + `(rms ${viewer.floor.rms.toFixed(3)} m, grade ${(p.floorB * 100).toFixed(2)} %)`)
  console.log(`ZONES walls : x = [${viewer.walls.map((w) => w.toFixed(2)).join(', ')}]`)
  console.log(`ZONES axis  : x = ${p.axisX.toFixed(2)} m, gauge = ${Math.round(p.gauge * 1000)} mm `
    + `(rails at x = ${rails[0]?.toFixed(2)} and ${rails[1]?.toFixed(2)})`)
  console.log(`RAILS blocked (y, m): [${spans}]  total ${blockedTotal.toFixed(1)} m of `
    + `${(args.maxDist + args.behind).toFixed(0)} m`)
  console.log(`Serving ${viewer.total} frames at ${url}   (Ctrl+C to stop)`)
}

function main(): void {
  const args = parseArgsOrExit()
  const dbPath = findDb3(resolveRepoPath(args.bagDir))
  console.log(`Reading: ${dbPath}`)

  const viewer = new Viewer(dbPath, {
    maxDist: args.maxDist,
    behind: args.behind,
    mode: args.mode,
    maxPoints: args.maxPoints,
    pointSize: args.pointSize,
    axisX: args.axisX,
    gaugeMm: args.gaugeMm,
    obstaclePoints: args.obstaclePoints,
    obstacleHeight: args.obstacleHeight,
    railClearance: args.railClearance,
  })

  const clientDir = resolveRepoPath(args.clientDir)
  const url = `http://${args.host}:${args.port}/`

  let server: ReturnType<typeof Bun.serve>
  try {
    server = Bun.serve({
      hostname: args.host,
      port: args.port,
      fetch: makeHandler(viewer, clientDir, args),
      error: (err) => json({ error: err instanceof Error ? err.message : String(err) }, 500),
    })
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err)
    console.error(`не удалось занять ${args.host}:${args.port}: ${message}`)
    viewer.close()
    process.exit(1)
  }

  printSummary(viewer, args, url)
  if (!existsSync(join(clientDir, 'index.html'))) {
    console.warn(`сборки клиента нет (${clientDir}/index.html) -- на / отдаётся заглушка`)
  }

  if (args.open) openBrowser(url)

  const stop = (): void => {
    server.stop(true)
    viewer.close()
    console.log('\nStopped.')
    process.exit(0)
  }
  process.on('SIGINT', stop)
  process.on('SIGTERM', stop)
}

main()