/**
 * Сквозная сверка Bun-сервера с Python-эталоном (web_viewer.py).
 *
 * Поднимает оба сервера на разных портах, гоняет по ним одни и те же запросы
 * (/meta, /frame, /set) и печатает таблицу «параметр | Python | Bun | расхождение».
 * Допуски: float 1e-6, счётчики/метки/число точек -- точно, участки засорённости --
 * 0.5 м (они строятся из бинов, но в норме совпадают и точно).
 *
 * Код возврата 1, если хоть одна проверка вышла за допуск. Дочерние процессы
 * глушатся в любом случае.
 */

import { resolve } from 'node:path'

import { REPO_ROOT, resolveRepoPath } from './cli.ts'

const ROOT = REPO_ROOT
const APP_SERVER = resolve(import.meta.dir, '..')

const FLOAT_TOL = 1e-6
const SPAN_TOL = 0.5
const DEFAULT_PYTHON = 'C:\\Users\\K4ler\\AppData\\Local\\Programs\\Python\\Python312\\python.exe'

/** Порядок меток в кадре: индексы меток указывают в этот список (zones.ZONE_NAMES). */
const ZONE_ORDER = ['rail', 'bed', 'ceiling', 'wall', 'middle']

/** Эталон списка режимов (bag_reader.COLOR_MODES): проверяем, что серверы его не теряют. */
const MODES_REF = ['height', 'intensity', 'range', 'ring', 'zones']

const FRAME_INDICES = [0, 50, 100, 150]
const FRAME_MODES = ['zones', 'height']
const SET_QUERIES = [
  'axis=4.05&gauge=1520',
  'axis=3.33&contact=0',
  'axis=3.9&gauge=1435&contact=1&offset=1.2&side=-1',
]

interface ParityCli {
  bag: string
  pyPort: number
  bunPort: number
  python: string
  timeoutMs: number
}

interface Check {
  name: string
  py: string
  bun: string
  diff: string
  ok: boolean
  /** справка: расхождение печатаем, но в итог не засчитываем */
  note: boolean
}

const checks: Check[] = []

// ---------------------------------------------------------------- разбор флагов

function usage(): string {
  return [
    'Сверка Bun-вьюера с Python-эталоном.',
    '',
    'Использование: bun run src/parity.ts [bag_dir] [флаги]',
    '',
    `  bag_dir             каталог с bag (по умолчанию for_hackathon/doubleT_obstacle)`,
    `  --bag ДИР           то же флагом`,
    `  --py-port N         порт Python-вьюера (8767)`,
    `  --bun-port N        порт Bun-сервера (8788)`,
    `  --python ПУТЬ       интерпретатор Python (${DEFAULT_PYTHON})`,
    `  --timeout МС        сколько ждать готовности каждого сервера (300000)`,
  ].join('\n')
}

function parseCli(argv: readonly string[]): ParityCli {
  const cli: ParityCli = {
    bag: 'for_hackathon/doubleT_obstacle',
    pyPort: 8767,
    bunPort: 8788,
    python: process.env.LIDAR_PYTHON ?? DEFAULT_PYTHON,
    timeoutMs: 300000,
  }
  let positional: string | null = null
  for (let i = 0; i < argv.length; i++) {
    const token = argv[i] as string
    const [name, inline] = token.includes('=') ? [token.slice(0, token.indexOf('=')), token.slice(token.indexOf('=') + 1)] : [token, null]
    const value = (): string => {
      if (inline !== null) return inline
      const next = argv[++i]
      if (next === undefined) throw new Error(`флаг ${name} требует значение`)
      return next
    }
    if (name === '--help' || name === '-h') {
      console.log(usage())
      process.exit(0)
    } else if (name === '--bag') cli.bag = value()
    else if (name === '--py-port') cli.pyPort = Number.parseInt(value(), 10)
    else if (name === '--bun-port') cli.bunPort = Number.parseInt(value(), 10)
    else if (name === '--python') cli.python = value()
    else if (name === '--timeout') cli.timeoutMs = Number.parseInt(value(), 10)
    else if (token.startsWith('-')) throw new Error(`неизвестный флаг: ${name}\n${usage()}`)
    else if (positional === null) positional = token
    else throw new Error(`лишний аргумент: ${token}`)
  }
  if (positional !== null) cli.bag = positional
  return cli
}

// ---------------------------------------------------------------- запись проверок

function text(value: unknown): string {
  if (typeof value === 'string') return value
  if (typeof value === 'number' || typeof value === 'boolean') return String(value)
  return JSON.stringify(value)
}

/** Длинные числовые массивы в таблице не помещаются: печатаем начало и длину. */
function arrText(values: readonly number[]): string {
  const head = values.slice(0, 4).map((v) => String(v)).join(', ')
  return values.length <= 6 ? `[${head}]` : `[${head}, ... всего ${values.length}]`
}

function spansText(spans: ReadonlyArray<readonly [number, number]>): string {
  const head = spans.slice(0, 4).map(([a, b]) => `[${a}, ${b}]`).join(', ')
  return spans.length <= 4 ? `[${head}]` : `[${head}, ... всего ${spans.length}]`
}

function record(name: string, py: unknown, bun: unknown, ok: boolean, diff: string, note = false): void {
  checks.push({ name, py: text(py), bun: text(bun), diff, ok, note })
}

function expectFloat(name: string, py: number, bun: number, tol = FLOAT_TOL, note = false): void {
  const delta = Math.abs(py - bun)
  const ok = Number.isFinite(delta) && delta <= tol
  record(name, py, bun, ok, Number.isFinite(delta) ? delta.toExponential(2) : 'NaN', note)
}

function expectExact(name: string, py: unknown, bun: unknown, note = false): void {
  const ok = JSON.stringify(py) === JSON.stringify(bun)
  record(name, py, bun, ok, ok ? '0' : '≠', note)
}

function expectArray(name: string, py: readonly number[], bun: readonly number[], tol = FLOAT_TOL, note = false): void {
  if (py.length !== bun.length) {
    record(name, arrText(py), arrText(bun), false, `длина ${py.length} vs ${bun.length}`, note)
    return
  }
  let max = 0
  for (let i = 0; i < py.length; i++) max = Math.max(max, Math.abs((py[i] as number) - (bun[i] as number)))
  record(name, arrText(py), arrText(bun), max <= tol, max.toExponential(2), note)
}

function expectSpans(name: string, py: ReadonlyArray<readonly [number, number]>,
                     bun: ReadonlyArray<readonly [number, number]>, tol = SPAN_TOL): void {
  if (py.length !== bun.length) {
    record(name, spansText(py), spansText(bun), false, `участков ${py.length} vs ${bun.length}`)
    return
  }
  let max = 0
  for (let i = 0; i < py.length; i++) {
    const a = py[i] as readonly [number, number]
    const b = bun[i] as readonly [number, number]
    max = Math.max(max, Math.abs(a[0] - b[0]), Math.abs(a[1] - b[1]))
  }
  record(name, spansText(py), spansText(bun), max <= tol, max.toExponential(2))
}

function expectHeader(name: string, headerPy: FrameHeader, headerBun: FrameHeader): void {
  record(name, headerPy.text, headerBun.text, headerPy.text === headerBun.text, headerPy.text === headerBun.text ? '0' : '≠')
}

// ---------------------------------------------------------------- дочерние процессы

function start(cmd: readonly string[], cwd: string) {
  const proc = Bun.spawn([...cmd], { cwd, stdout: 'pipe', stderr: 'pipe' })
  const chunks: string[] = []
  const decoder = new TextDecoder()
  const pump = async (stream: ReadableStream<Uint8Array> | null): Promise<void> => {
    if (!stream) return
    try {
      for await (const chunk of stream) chunks.push(decoder.decode(chunk, { stream: true }))
    } catch {
      // процесс закрылся -- это не ошибка сверки
    }
  }
  void pump(proc.stdout)
  void pump(proc.stderr)
  return {
    proc,
    log: (): string => chunks.join(''),
  }
}

type Child = ReturnType<typeof start>

async function stop(child: Child): Promise<void> {
  try {
    child.proc.kill()
  } catch {
    return
  }
  await Promise.race([child.proc.exited, Bun.sleep(2000)])
}

function tail(message: string, lines = 12): string {
  const rows = message.trim().split(/\r?\n/)
  return rows.slice(-lines).join('\n')
}

async function waitReady(url: string, child: Child, label: string, timeoutMs: number): Promise<void> {
  const deadline = Date.now() + timeoutMs
  while (Date.now() < deadline) {
    if (child.proc.exitCode !== null) {
      throw new Error(`${label} завершился с кодом ${child.proc.exitCode}:\n${tail(child.log())}`)
    }
    try {
      const res = await fetch(url, { signal: AbortSignal.timeout(3000) })
      if (res.ok) {
        await res.arrayBuffer()
        return
      }
    } catch {
      // ещё не поднялся
    }
    await Bun.sleep(300)
  }
  throw new Error(`${label}: не ответил на ${url} за ${timeoutMs} мс:\n${tail(child.log())}`)
}

/** Занят ли порт: отвечает ли кто-то на /meta (чужой вьюер -- сверять по нему нельзя). */
async function portBusy(port: number): Promise<boolean> {
  try {
    const res = await fetch(`http://127.0.0.1:${port}/meta`, { signal: AbortSignal.timeout(1500) })
    return res.ok
  } catch {
    return false
  }
}

/**
 * Убедиться, что отвечает именно НАШ процесс. Без этого легко сравнить числа с
 * чужим вьюером, который уже сидит на порту (его состояние к тому же мог изменить
 * любой /set) -- сверка при этом врёт, не падая.
 */
async function expectOwnServer(child: Child, label: string, marker: string, timeoutMs = 20000): Promise<void> {
  const deadline = Date.now() + timeoutMs
  while (Date.now() < deadline) {
    if (child.log().includes(marker)) return
    if (child.proc.exitCode !== null) {
      throw new Error(`${label} завершился с кодом ${child.proc.exitCode}:\n${tail(child.log())}`)
    }
    await Bun.sleep(200)
  }
  throw new Error(`${label}: в выводе нет «${marker}» -- похоже, на порту отвечает чужой сервер:\n${tail(child.log())}`)
}

// ---------------------------------------------------------------- запросы

async function getJson(url: string): Promise<any> {
  const res = await fetch(url)
  const body = await res.text()
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status} ${body.slice(0, 200)}`)
  return JSON.parse(body)
}

interface FrameHeader {
  text: string
  version: number
  kind: number
  count: number
}

interface FrameView {
  header: FrameHeader
  positionsCrc: number
  labels: number[]
  rgbCrc: number
}

function hex(value: number): string {
  return value.toString(16).padStart(8, '0')
}

async function readFrame(base: string, idx: number, mode: string): Promise<FrameView> {
  const res = await fetch(`${base}/frame?idx=${idx}&mode=${mode}`)
  if (!res.ok) throw new Error(`/frame?idx=${idx}&mode=${mode}: HTTP ${res.status}`)
  const bytes = new Uint8Array(await res.arrayBuffer())
  if (bytes.length < 12) throw new Error(`/frame?idx=${idx}: ответ короче заголовка (${bytes.length} байт)`)
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
  const magic = String.fromCharCode(bytes[0] as number, bytes[1] as number, bytes[2] as number, bytes[3] as number)
  const header: FrameHeader = {
    text: `${magic}/${view.getUint8(4)}/kind=${view.getUint8(5)}/reserved=${view.getUint16(6, true)}`,
    version: view.getUint8(4),
    kind: view.getUint8(5),
    count: view.getUint32(8, true),
  }
  const positionsEnd = 12 + header.count * 12
  if (bytes.length < positionsEnd) {
    throw new Error(`/frame?idx=${idx}: обещано ${header.count} точек, а байт всего ${bytes.length}`)
  }
  const labels: number[] = []
  let rgbCrc = 0
  if (header.kind === 1) {
    labels.push(0, 0, 0, 0, 0)
    for (let i = positionsEnd; i < bytes.length; i++) {
      const label = bytes[i] as number
      labels[label] = (labels[label] ?? 0) + 1
    }
  } else {
    rgbCrc = Bun.hash.crc32(bytes.subarray(positionsEnd))
  }
  return {
    header,
    positionsCrc: Bun.hash.crc32(bytes.subarray(12, positionsEnd)),
    labels,
    rgbCrc,
  }
}

function labelsText(labels: readonly number[]): string {
  return ZONE_ORDER.map((name, i) => `${name}=${labels[i] ?? 0}`).join(' ')
}

// ---------------------------------------------------------------- сами сверки

async function compareMeta(pyBase: string, bunBase: string): Promise<void> {
  const py = await getJson(`${pyBase}/meta`)
  const bun = await getJson(`${bunBase}/meta`)
  expectExact('meta.bag', py.bag, bun.bag)
  expectExact('meta.frames', py.frames, bun.frames)
  expectFloat('meta.duration_s', py.duration_s, bun.duration_s)
  expectFloat('meta.hz', py.hz, bun.hz)
  expectExact('meta.max_dist', py.max_dist, bun.max_dist)
  expectExact('meta.behind', py.behind, bun.behind)
  expectExact('meta.default_mode', py.default_mode, bun.default_mode)
  expectExact('meta.modes (эталон)', MODES_REF, py.modes)
  expectExact('meta.modes', py.modes, bun.modes)
  expectExact('meta.point_size', py.point_size, bun.point_size)
  expectExact('meta.max_points_hint', py.max_points_hint, bun.max_points_hint)
  expectFloat('meta.floor.a', py.floor.a, bun.floor.a)
  expectFloat('meta.floor.b', py.floor.b, bun.floor.b)
  expectFloat('meta.floor.rms', py.floor.rms, bun.floor.rms)
  expectFloat('meta.floor.grade_pct', py.floor.grade_pct, bun.floor.grade_pct)
  expectArray('meta.walls', py.walls, bun.walls)
  expectFloat('meta.axis_x', py.axis_x, bun.axis_x)
  expectExact('meta.gauge_mm', py.gauge_mm, bun.gauge_mm)
  expectArray('meta.rails.x', py.rails.x, bun.rails.x)
  expectArray('meta.rails.heights', py.rails.heights, bun.rails.heights)
  expectArray('meta.rails.y', py.rails.y, bun.rails.y)
  expectArray('meta.rails.z', py.rails.z, bun.rails.z)
  expectArray('meta.rails.blocked', py.rails.blocked, bun.rails.blocked)
  expectSpans('meta.blocked_spans', py.blocked_spans, bun.blocked_spans)
  expectExact('meta.grid', py.grid, bun.grid)
  expectExact('meta.zone_names (эталон)', ZONE_ORDER, py.zone_names)
  expectExact('meta.zone_names', py.zone_names, bun.zone_names)
  expectExact('meta.zone_colors', py.zone_colors, bun.zone_colors)
  expectExact('meta.blocked_color', py.blocked_color, bun.blocked_color)
  expectExact('meta.rail_color', py.rail_color, bun.rail_color)
  expectExact('meta.train_top', py.train_top, bun.train_top)
  expectExact('meta.obstacle_points', py.obstacle_points, bun.obstacle_points)
  expectExact('meta.rail_clearance', py.rail_clearance, bun.rail_clearance)
}

async function compareFrames(pyBase: string, bunBase: string): Promise<void> {
  for (const idx of FRAME_INDICES) {
    for (const mode of FRAME_MODES) {
      const label = `frame idx=${idx} mode=${mode}`
      const py = await readFrame(pyBase, idx, mode)
      const bun = await readFrame(bunBase, idx, mode)
      expectHeader(`${label}: заголовок(magic/version/kind/reserved)`, py.header, bun.header)
      expectExact(`${label}: точек`, py.header.count, bun.header.count)
      expectExact(`${label}: позиции crc32`, hex(py.positionsCrc), hex(bun.positionsCrc))
      if (mode === 'zones') {
        expectExact(`${label}: гистограмма меток`, labelsText(py.labels), labelsText(bun.labels))
      } else {
        expectExact(`${label}: цвета crc32 [справка: палитра bagReader]`, hex(py.rgbCrc), hex(bun.rgbCrc), true)
      }
    }
  }
}

async function compareSets(pyBase: string, bunBase: string): Promise<void> {
  for (const query of SET_QUERIES) {
    const tag = `set?${query}`
    const py = await getJson(`${pyBase}/set?${query}`)
    const bun = await getJson(`${bunBase}/set?${query}`)
    expectFloat(`${tag}: axis_x`, py.axis_x, bun.axis_x)
    expectExact(`${tag}: gauge_mm`, py.gauge_mm, bun.gauge_mm)
    expectExact(`${tag}: contact_rail`, py.contact_rail, bun.contact_rail)
    expectFloat(`${tag}: contact_offset`, py.contact_offset, bun.contact_offset)
    expectExact(`${tag}: contact_side`, py.contact_side, bun.contact_side)
    expectFloat(`${tag}: rail_height`, py.rail_height, bun.rail_height)
    expectFloat(`${tag}: floor.a`, py.floor.a, bun.floor.a)
    expectFloat(`${tag}: floor.b`, py.floor.b, bun.floor.b)
    expectFloat(`${tag}: floor.rms`, py.floor.rms, bun.floor.rms)
    expectFloat(`${tag}: floor.grade_pct`, py.floor.grade_pct, bun.floor.grade_pct)
    expectArray(`${tag}: rails.x`, py.rails.x, bun.rails.x)
    expectArray(`${tag}: rails.heights`, py.rails.heights, bun.rails.heights)
    expectArray(`${tag}: rails.y`, py.rails.y, bun.rails.y)
    expectArray(`${tag}: rails.z`, py.rails.z, bun.rails.z)
    expectArray(`${tag}: rails.blocked`, py.rails.blocked, bun.rails.blocked)
    expectSpans(`${tag}: blocked_spans`, py.blocked_spans, bun.blocked_spans)
  }
}

// ---------------------------------------------------------------- таблица

function pad(value: string, width: number): string {
  return value + ' '.repeat(Math.max(0, width - value.length))
}

function printTable(): void {
  const header = ['параметр', 'Python', 'Bun', 'расхождение']
  const rows = checks.map((c) => [c.name, c.py, c.bun, `${c.ok ? ' ' : c.note ? '~' : '!'} ${c.diff}`])
  const widths = header.map((title, i) => Math.max(title.length, ...rows.map((row) => (row[i] as string).length)))
  const line = (cells: readonly string[]): string =>
    cells.map((cell, i) => pad(cell, (widths[i] as number) + 2)).join('|').trimEnd()
  console.log('')
  console.log(line(header))
  console.log(widths.map((w) => '-'.repeat(w + 2)).join('+'))
  for (const row of rows) console.log(line(row))
  console.log('')
}

async function main(): Promise<number> {
  const cli = parseCli(Bun.argv.slice(2))
  const bagDir = resolveRepoPath(cli.bag)
  const pyBase = `http://127.0.0.1:${cli.pyPort}`
  const bunBase = `http://127.0.0.1:${cli.bunPort}`

  console.log(`Сверка Bun-вьюера с Python: bag ${bagDir}`)
  console.log(`Python : ${cli.python} web_viewer.py <bag> --port ${cli.pyPort}  (cwd ${ROOT})`)
  console.log(`Bun    : bun run src/index.ts <bag> --port ${cli.bunPort}  (cwd ${APP_SERVER})`)

  // Чужие серверы на наших портах сделали бы сверку фиктивной: сначала проверяем,
// что порты свободны, а потом -- что отвечает именно наш процесс.
  if (await portBusy(cli.pyPort)) {
    console.error(`порт ${cli.pyPort} уже занят: на /meta отвечает чужой вьюер. `
      + 'Остановите его или укажите другой порт (--py-port).')
    return 1
  }
  if (await portBusy(cli.bunPort)) {
    console.error(`порт ${cli.bunPort} уже занят: на /meta отвечает чужой вьюер. `
      + 'Остановите его или укажите другой порт (--bun-port).')
    return 1
  }

  // -u: без буферизации, иначе сводка Python не попадает в лог до заполнения буфера.
  const pyChild = start([cli.python, '-u', 'web_viewer.py', bagDir, '--port', String(cli.pyPort)], ROOT)
  const bunChild = start([process.execPath, 'run', 'src/index.ts', bagDir, '--port', String(cli.bunPort)], APP_SERVER)

  let failure: unknown = null
  try {
    await waitReady(`${pyBase}/meta`, pyChild, 'Python', cli.timeoutMs)
    await expectOwnServer(pyChild, 'Python', `frames at http://127.0.0.1:${cli.pyPort}/`)
    console.log('\n--- сводка Python ---')
    console.log(tail(pyChild.log(), 6))
    await waitReady(`${bunBase}/meta`, bunChild, 'Bun', cli.timeoutMs)
    await expectOwnServer(bunChild, 'Bun', `frames at http://127.0.0.1:${cli.bunPort}/`)
    console.log('--- сводка Bun ---')
    console.log(tail(bunChild.log(), 6))

    await compareMeta(pyBase, bunBase)
    await compareFrames(pyBase, bunBase)
    await compareSets(pyBase, bunBase)
  } catch (err) {
    failure = err
  } finally {
    await Promise.all([stop(pyChild), stop(bunChild)])
  }

  printTable()
  const mismatches = checks.filter((c) => !c.ok && !c.note)
  const notes = checks.filter((c) => !c.ok && c.note)
  console.log(`проверок ${checks.length}, расхождений ${mismatches.length}`
    + (notes.length ? ` (плюс ${notes.length} справочных)` : ''))
  if (failure) {
    console.error(`\nсверка не дошла до конца: ${failure instanceof Error ? failure.message : String(failure)}`)
    console.error(`лог Python:\n${tail(pyChild.log())}`)
    console.error(`лог Bun:\n${tail(bunChild.log())}`)
    return 1
  }
  if (mismatches.length) {
    console.error('\nРАСХОЖДЕНИЯ:')
    for (const c of mismatches) console.error(`  ${c.name}: Python ${c.py} vs Bun ${c.bun} (${c.diff})`)
    return 1
  }
  console.log('Bun совпал с Python в пределах допусков.')
  return 0
}

void main().then((code) => process.exit(code))