// Чтение rosbag2 `.db3` (PointCloud2) и раскраски кадров -- порт `bag_reader.py`
// на TypeScript (Bun).
//
// ROS не нужен: `.db3` -- это SQLite (встроенный `bun:sqlite`), CDR разбирается
// вручную (`parsePointcloud2Cdr`). Модуль отдаёт кадры по индексу с кольцевым
// кэшем и предзагрузкой (`BagFrames`), режет облако по дальности (`trim`) и
// красит его скалярным полем (`frameColors`). Семантические зоны ('zones')
// считает `zones.ts` -- здесь только имя режима.
//
// Числа обязаны совпадать с Python. Там, где numpy работает во float32
// (арифметика массива float32 с Python-скаляром -- режимы height и range),
// каждый шаг округляется через f32(): без этого младшие биты t разъезжаются и
// индекс в таблице цветов у части точек отличается на единицу.

import { readdirSync } from 'node:fs'

import { Database } from 'bun:sqlite'

import { f32, interp, linspace, normF32, percentileFromSorted, sortedCopy } from './numeric.ts'

/** Режимы раскраски скалярным полем; порядок как в bag_reader.py. */
export const MODES = ['height', 'intensity', 'range', 'ring'] as const

// 'zones' -- не раскраска скалярного поля, а семантические классы из zones.ts,
// поэтому frameColors() её не знает; в список режимов вьюера она добавлена
// отдельно.
export const COLOR_MODES = [...MODES, 'zones'] as const

/** Диапазоны по умолчанию для doubleT_obstacle (измерено: Z in [-7.0, 2.8]). */
export const Z_RANGE: readonly [number, number] = [-7.0, 3.0]
export const RING_COUNT = 128
export const CACHE_FRAMES = 32
export const PREFETCH_AHEAD = 8

/** Кадр: позиции интерливом (x, y, z), плюс необязательные поля. */
export interface Frame {
  xyz: Float32Array
  intensity: Float32Array | null
  ring: Uint16Array | null
}

// --------------------------------------------------------------------- палитра

/** Аппроксимация Google turbo контрольными точками (линейная интерполяция). */
const TURBO_ANCHORS: readonly (readonly [number, number, number])[] = [
  [0.18995, 0.07176, 0.23217],
  [0.19483, 0.08339, 0.26149],
  [0.25701, 0.13633, 0.33211],
  [0.36294, 0.21263, 0.41467],
  [0.49859, 0.25098, 0.4428],
  [0.65493, 0.26085, 0.41072],
  [0.81076, 0.29928, 0.3178],
  [0.90183, 0.38623, 0.2273],
  [0.92614, 0.51292, 0.1606],
  [0.88749, 0.66573, 0.15924],
  [0.80938, 0.81255, 0.23496],
  [0.74133, 0.90882, 0.4354],
  [0.74527, 0.94931, 0.70633],
  [0.78752, 0.96592, 0.92003],
  [0.8407, 0.96458, 0.98167],
]

const TURBO_X = linspace(0.0, 1.0, TURBO_ANCHORS.length)
const TURBO_CHANNELS: readonly Float64Array[] = [0, 1, 2].map(
  (c) => Float64Array.from(TURBO_ANCHORS, (a) => a[c]!),
)

/** t -> RGB float64 (вход любой длины, значения 0..1, как `_ramp` в Python). */
function ramp(t: ArrayLike<number>): Float64Array {
  const out = new Float64Array(t.length * 3)
  for (let i = 0; i < t.length; i++) {
    const v = t[i]! < 0.0 ? 0.0 : t[i]! > 1.0 ? 1.0 : t[i]!
    for (let c = 0; c < 3; c++) out[i * 3 + c] = interp(v, TURBO_X, TURBO_CHANNELS[c]!)
  }
  return out
}

export const LUT_LEVELS = 256

// Таблица цветов: np.interp по трём каналам на каждой точке стоит ~25 мс на
// 300k точек, индексация в LUT -- ~8 мс, при расхождении цвета ~0.015.
export const TURBO_LUT = ramp(linspace(0.0, 1.0, LUT_LEVELS))

/** Индекс уровня LUT из t: clip(t*(LUT_LEVELS-1), 0, LUT_LEVELS-1) + усечение. */
function lutIndex(t: number): number {
  const v = t * (LUT_LEVELS - 1)
  return Math.trunc(v <= 0.0 ? 0.0 : v >= LUT_LEVELS - 1 ? LUT_LEVELS - 1 : v)
}

// ------------------------------------------------------------------ CDR-разбор

export interface ParsedCloud {
  xyz: Float32Array
  intensity: Float32Array | null
  ring: Uint16Array | null
}

/**
 * Разобрать CDR-сериализованный PointCloud2 -> {xyz, intensity, ring}.
 *
 * `withRing=true` добавляет поле `ring` (uint16), если оно есть в сообщении.
 *
 * Логика перенесена из Python-эталона и проверена на живых данных:
 * point_step=26, x/y/z/intensity -- float32 @ 0/4/8/12, ring uint16 @ 16,
 * timestamp float64 @ 18.
 */
export function parsePointcloud2Cdr(data: Uint8Array, withRing = false): ParsedCloud {
  const dv = new DataView(data.buffer, data.byteOffset, data.byteLength)
  const little = data[1] === 1

  const align4 = (off: number): number => off + ((4 - (off % 4)) % 4)
  const readU32 = (off: number): [number, number] => [dv.getUint32(off, little), off + 4]
  const readU8 = (off: number): [number, number] => [dv.getUint8(off), off + 1]
  const readString = (off: number): [string, number] => {
    const [slen, afterLen] = readU32(off)
    const text = slen > 0 ? utf8Decode(data.subarray(afterLen, afterLen + slen - 1)) : ''
    return [text, afterLen + slen]
  }

  let offset = 4
  offset = readU32(offset)[1] // sec
  offset = readU32(offset)[1] // nsec
  offset = readString(offset)[1] // frame_id

  offset = align4(offset)
  const [height, afterHeight] = readU32(offset)
  offset = afterHeight
  const [width, afterWidth] = readU32(offset)
  offset = afterWidth

  const [numFields, afterNumFields] = readU32(offset)
  offset = afterNumFields
  const fieldOffsets = new Map<string, number>()
  for (let f = 0; f < numFields; f++) {
    const [name, afterName] = readString(offset)
    offset = align4(afterName)
    const [fOffset, afterFOffset] = readU32(offset)
    offset = afterFOffset
    offset = readU8(offset)[1] // datatype
    offset = align4(offset)
    offset = readU32(offset)[1] // count
    fieldOffsets.set(name, fOffset)
  }

  const [isBigendian, afterBe] = readU8(offset)
  offset = align4(afterBe)
  const [pointStep, afterStep] = readU32(offset)
  offset = afterStep
  offset = readU32(offset)[1] // row_step
  const [dataLen, afterDataLen] = readU32(offset)
  offset = afterDataLen

  const nDeclared = height * width
  const nPoints = pointStep > 0 ? Math.min(nDeclared, Math.floor(dataLen / pointStep)) : 0
  if (nPoints <= 0) {
    return { xyz: new Float32Array(0), intensity: null, ring: null }
  }

  const cloudLittle = !isBigendian
  const dataStart = offset
  const xOff = fieldOffsets.get('x')
  if (xOff === undefined) throw new Error('PointCloud2 has no "x" field')
  const yOff = fieldOffsets.get('y')
  const zOff = fieldOffsets.get('z')
  const intOff = fieldOffsets.get('intensity')

  const xyz = new Float32Array(nPoints * 3)
  const intensity = intOff === undefined ? null : new Float32Array(nPoints)
  const ring = withRing && fieldOffsets.has('ring') ? new Uint16Array(nPoints) : null
  const rOff = fieldOffsets.get('ring') ?? 0

  const contiguousQuad =
    yOff === xOff + 4 && zOff === xOff + 8 && intOff === xOff + 12

  // Точка остаётся, если она конечна по всем осям и не нулевая (как
  // np.isfinite(...).all(axis=1) & (xyz != 0).any(axis=1) в Python-эталоне);
  // отбор идёт за один проход, длина результата -- число оставленных точек.
  let kept = 0

  const writePoint = (i: number, x: number, y: number, z: number): boolean => {
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) return false
    if (x === 0 && y === 0 && z === 0) return false
    xyz[kept * 3] = x
    xyz[kept * 3 + 1] = y
    xyz[kept * 3 + 2] = z
    return true
  }

  if (contiguousQuad) {
    // x, y, z и intensity идут подряд: читаем четыре float32 одним проходом.
    for (let i = 0; i < nPoints; i++) {
      const o = dataStart + i * pointStep + xOff
      const x = dv.getFloat32(o, cloudLittle)
      const y = dv.getFloat32(o + 4, cloudLittle)
      const z = dv.getFloat32(o + 8, cloudLittle)
      if (!writePoint(i, x, y, z)) continue
      if (intensity !== null) intensity[kept] = dv.getFloat32(o + 12, cloudLittle)
      if (ring !== null) ring[kept] = dv.getUint16(dataStart + i * pointStep + rOff, cloudLittle)
      kept++
    }
  } else {
    if (yOff === undefined) throw new Error('PointCloud2 has no "y" field')
    if (zOff === undefined) throw new Error('PointCloud2 has no "z" field')
    for (let i = 0; i < nPoints; i++) {
      const base = dataStart + i * pointStep
      const x = dv.getFloat32(base + xOff, cloudLittle)
      const y = dv.getFloat32(base + yOff, cloudLittle)
      const z = dv.getFloat32(base + zOff, cloudLittle)
      if (!writePoint(i, x, y, z)) continue
      if (intensity !== null) intensity[kept] = dv.getFloat32(base + intOff!, cloudLittle)
      if (ring !== null) ring[kept] = dv.getUint16(base + rOff, cloudLittle)
      kept++
    }
  }

  const xyzOut = xyz.subarray(0, kept * 3)
  const intenOut = intensity === null ? null : intensity.subarray(0, kept)
  const ringOut = ring === null ? null : ring.subarray(0, kept)
  return { xyz: xyzOut, intensity: intenOut, ring: ringOut }
}

const utf8Decoder = new TextDecoder('utf-8', { fatal: false })

function utf8Decode(bytes: Uint8Array): string {
  return utf8Decoder.decode(bytes)
}

// ------------------------------------------------------------------- раскраски

/** t по высоте (Z) -- float32, как `(xyz[:,2] - lo) / (hi - lo)` в numpy. */
function heightT(xyz: Float32Array, n: number, zRange: readonly [number, number]): Float64Array {
  const out = new Float64Array(n)
  const lo = f32(zRange[0])
  const span = f32(Math.max(zRange[1] - zRange[0], 1e-9))
  for (let i = 0; i < n; i++) out[i] = f32(f32(xyz[i * 3 + 2]! - lo) / span)
  return out
}

/** t по интенсивности -- float64 (перцентили 1 и 99 считаются в float64). */
function intensityT(intensity: Float32Array | null, n: number): Float64Array {
  if (intensity === null) {
    throw new Error('intensity is None: cannot color by intensity')
  }
  const out = new Float64Array(n)
  for (let i = 0; i < n; i++) out[i] = intensity[i]!
  const sorted = sortedCopy(out)
  const lo = percentileFromSorted(sorted, 1.0)
  const hi = percentileFromSorted(sorted, 99.0)
  const span = Math.max(hi - lo, 1e-9)
  for (let i = 0; i < n; i++) out[i] = (out[i]! - lo) / span
  return out
}

/** t по дальности -- float32: np.linalg.norm по float32 и деление во float32. */
function rangeT(xyz: Float32Array, n: number, maxDist: number): Float64Array {
  const out = new Float64Array(n)
  const divisor = f32(Math.max(maxDist, 1e-9))
  for (let i = 0; i < n; i++) {
    const r = normF32(xyz[i * 3]!, xyz[i * 3 + 1]!, xyz[i * 3 + 2]!)
    out[i] = f32(r / divisor)
  }
  return out
}

/** t по кольцу -- float64: (ring % RING_COUNT) / (RING_COUNT - 1). */
function ringT(ring: Uint16Array | null, n: number): Float64Array {
  if (ring === null) {
    throw new Error('ring is None: cannot color by ring')
  }
  const out = new Float64Array(n)
  for (let i = 0; i < n; i++) out[i] = (ring[i]! % RING_COUNT) / (RING_COUNT - 1)
  return out
}

/**
 * Раскраска кадра -> float64 (n*3) в 0..1. Палитра -- turbo.
 * `mode` из MODES; незнакомый режим (в том числе 'zones') -- ошибка.
 */
export function frameColors(
  xyz: Float32Array,
  intensity: Float32Array | null,
  mode: string,
  maxDist: number,
  ring: Uint16Array | null = null,
  zRange: readonly [number, number] = Z_RANGE,
): Float64Array {
  const n = xyz.length / 3
  const out = new Float64Array(n * 3)
  if (n === 0) return out

  let t: Float64Array
  switch (mode) {
    case 'height':
      t = heightT(xyz, n, zRange)
      break
    case 'intensity':
      t = intensityT(intensity, n)
      break
    case 'range':
      t = rangeT(xyz, n, maxDist)
      break
    case 'ring':
      t = ringT(ring, n)
      break
    default:
      throw new Error(`unknown color mode ${JSON.stringify(mode)}, expected one of ${MODES.join(', ')}`)
  }

  for (let i = 0; i < n; i++) {
    const src = lutIndex(t[i]!) * 3
    out[i * 3] = TURBO_LUT[src]!
    out[i * 3 + 1] = TURBO_LUT[src + 1]!
    out[i * 3 + 2] = TURBO_LUT[src + 2]!
  }
  return out
}

// ---------------------------------------------------------------------- обрезка

/**
 * Маска: оставить коридор по Y. Туннель уходит в -Y, за спиной -- `behind`.
 * Точки -- интерлив (x, y, z), поэтому нужен каждый третий элемент.
 */
export function trimRange(xyz: Float32Array, maxDist = 80.0, behind = 5.0): Uint8Array {
  const n = xyz.length / 3
  const mask = new Uint8Array(n)
  for (let i = 0; i < n; i++) {
    const y = xyz[i * 3 + 1]!
    mask[i] = y >= -maxDist && y <= behind ? 1 : 0
  }
  return mask
}

/** Применить `trimRange` к xyz/intensity/ring; null-аргументы остаются null. */
export function trim(
  xyz: Float32Array,
  intensity: Float32Array | null = null,
  ring: Uint16Array | null = null,
  maxDist = 80.0,
  behind = 5.0,
): [Float32Array, Float32Array | null, Uint16Array | null] {
  const mask = trimRange(xyz, maxDist, behind)
  const n = mask.length
  let kept = 0
  for (let i = 0; i < n; i++) if (mask[i]!) kept++

  const outXyz = new Float32Array(kept * 3)
  const outInten = intensity === null ? null : new Float32Array(kept)
  const outRing = ring === null ? null : new Uint16Array(kept)
  let k = 0
  for (let i = 0; i < n; i++) {
    if (!mask[i]!) continue
    outXyz[k * 3] = xyz[i * 3]!
    outXyz[k * 3 + 1] = xyz[i * 3 + 1]!
    outXyz[k * 3 + 2] = xyz[i * 3 + 2]!
    if (outInten !== null) outInten[k] = intensity![i]!
    if (outRing !== null) outRing[k] = ring![i]!
    k++
  }
  return [outXyz, outInten, outRing]
}

// ---------------------------------------------------------------- кадры из bag

export interface BagFramesOptions {
  cacheSize?: number
  prefetchAhead?: number
}

/** Команда фоновой предзагрузке (`prefetchWorker.ts`). */
export type PrefetchCommand =
  | { kind: 'init'; dbPath: string; rowids: number[]; ahead: number; want: number }
  | { kind: 'want'; index: number }
  | { kind: 'stop' }

/** Готовый кадр из фоновой предзагрузки. */
export interface PrefetchFrameMessage {
  index: number
  xyz: Float32Array
  intensity: Float32Array | null
  ring: Uint16Array | null
}

/**
 * Доступ к кадрам rosbag2 `.db3` по индексу с кольцевым кэшем и предзагрузкой.
 *
 * Своё соединение с SQLite: запросы идут с главного потока, а предзагрузка --
 * в отдельном Worker (там же свой коннект), поэтому `prefetch()` не блокирует.
 */
export class BagFrames {
  readonly dbPath: string
  readonly cacheSize: number
  readonly prefetchAhead: number

  private readonly rowids: Float64Array
  private readonly _timestamps: BigInt64Array
  private readonly _topicNames: string[]
  private readonly cache = new Map<number, Frame>()
  private db: Database | null
  private worker: Worker | null = null
  private want = 0
  private closed = false
  private _parseCount = 0
  private _prefetchParses = 0
  private _hits = 0
  private _misses = 0

  constructor(dbPath: string, opts: BagFramesOptions = {}) {
    this.dbPath = dbPath
    this.cacheSize = opts.cacheSize ?? CACHE_FRAMES
    this.prefetchAhead = opts.prefetchAhead ?? PREFETCH_AHEAD

    const db = new Database(dbPath, { readonly: true, safeIntegers: true })
    this.db = db

    let topicIds: bigint[] = []
    try {
      const rows = db
        .query<{ id: bigint }, []>("SELECT id FROM topics WHERE type LIKE '%PointCloud2%'")
        .all()
      topicIds = rows.map((row) => BigInt(row.id))
    } catch {
      topicIds = []
    }

    // rowid обязательно с псевдонимом: bun:sqlite называет колонку rowid по
    // фактическому PRIMARY KEY (в этих bag-ах -- `id`), и row.rowid был бы
    // undefined. С safeIntegers INTEGER приходит bigint -- метки времени в
    // наносекундах (1.7e18) в number не помещаются без потерь.
    const messageRows =
      topicIds.length > 0
        ? db
            .query<{ rid: bigint; timestamp: bigint }, any[]>(
              `SELECT rowid AS rid, timestamp FROM messages WHERE topic_id IN (${topicIds
                .map(() => '?')
                .join(',')}) ORDER BY timestamp ASC`,
            )
            .all(...topicIds)
        : db
            .query<{ rid: bigint; timestamp: bigint }, []>(
              'SELECT rowid AS rid, timestamp FROM messages ORDER BY timestamp ASC',
            )
            .all()

    this.rowids = Float64Array.from(messageRows, (row) => Number(row.rid))
    this._timestamps = BigInt64Array.from(messageRows, (row) => BigInt(row.timestamp))

    try {
      const rows = db.query<{ name: string }, []>('SELECT name FROM topics').all()
      this._topicNames = rows.map((row) => row.name)
    } catch {
      this._topicNames = []
    }

    if (this.prefetchAhead > 0 && this.rowids.length > 1) this.startWorker()
  }

  get length(): number {
    return this.rowids.length
  }

  /** Метки времени кадров, наносекунды, по возрастанию. */
  get timestamps(): BigInt64Array {
    return this._timestamps
  }

  get topicNames(): string[] {
    return this._topicNames
  }

  /** Сколько кадров реально разобрано (попало в кэш; для тестов кэша). */
  get parseCount(): number {
    return this._parseCount
  }

  /** Из них разобрано фоновой предзагрузкой. */
  get prefetchParses(): number {
    return this._prefetchParses
  }

  get hits(): number {
    return this._hits
  }

  get misses(): number {
    return this._misses
  }

  /** Сколько кадров сейчас в кэше (для тестов вытеснения). */
  get cachedCount(): number {
    return this.cache.size
  }

  /** Кадр по индексу; при промахе разбирается и кладётся в кэш. */
  frame(index: number): Frame {
    if (index < 0 || index >= this.rowids.length) {
      throw new RangeError(`кадр ${index} вне диапазона 0..${this.rowids.length - 1}`)
    }
    const cached = this.cache.get(index)
    if (cached !== undefined) {
      // Освежаем запись в LRU: Map хранит порядок вставки.
      this.cache.delete(index)
      this.cache.set(index, cached)
      this._hits++
      return cached
    }
    this._misses++
    const frame = this.parseAt(this.requireDb(), index)
    this._parseCount++
    this.store(index, frame)
    // Подсказываем предзагрузке, куда двигаться дальше (в Python это делает
    // отдельный вызов want()).
    this.prefetch(index)
    return frame
  }

  /** Сообщить предзагрузке, откуда продолжать. Не блокирует; лучшее усилие. */
  prefetch(index: number): void {
    if (this.closed) return
    this.want = Math.trunc(index)
    this.postToWorker({ kind: 'want', index: this.want })
  }

  close(): void {
    if (this.closed) return
    this.closed = true
    if (this.worker !== null) {
      const worker = this.worker
      this.worker = null
      try {
        worker.postMessage({ kind: 'stop' } satisfies PrefetchCommand)
      } catch {
        // воркер уже мог умереть -- это не ошибка
      }
      worker.terminate()
    }
    this.db?.close()
    this.db = null
  }

  private requireDb(): Database {
    if (this.db === null) throw new Error('BagFrames закрыт')
    return this.db
  }

  private parseAt(db: Database, index: number): Frame {
    const row = db
      .query<{ data: Uint8Array }, [number]>('SELECT data FROM messages WHERE rowid=?')
      .get(this.rowids[index]!)
    if (row === null || row === undefined) throw new RangeError(`кадр ${index} не найден в bag`)
    return parsePointcloud2Cdr(row.data, true)
  }

  private store(index: number, frame: Frame): void {
    this.cache.set(index, frame)
    while (this.cache.size > this.cacheSize) {
      const oldest = this.cache.keys().next()
      if (oldest.done === true) break
      this.cache.delete(oldest.value)
    }
  }

  private onPrefetched(message: PrefetchFrameMessage): void {
    this._prefetchParses++
    if (this.cache.has(message.index)) return // уже есть -- порядок LRU не трогаем
    this._parseCount++
    this.store(message.index, {
      xyz: message.xyz,
      intensity: message.intensity,
      ring: message.ring,
    })
  }

  private postToWorker(command: PrefetchCommand): void {
    if (this.worker === null) return
    try {
      this.worker.postMessage(command)
    } catch {
      // предзагрузка -- лучшее усилие: молча отключаем
      this.worker = null
    }
  }

  private startWorker(): void {
    try {
      const worker = new Worker(new URL('./prefetchWorker.ts', import.meta.url).href, {
        type: 'module',
      })
      worker.onmessage = (event: MessageEvent) => {
        const message = event.data as PrefetchFrameMessage | null
        if (message !== null && typeof message?.index === 'number') this.onPrefetched(message)
      }
      worker.onerror = () => {
        // Предзагрузка не критична: кадры всё равно читаются с главного потока.
        this.worker = null
      }
      this.worker = worker
      worker.postMessage({
        kind: 'init',
        dbPath: this.dbPath,
        rowids: Array.from(this.rowids),
        ahead: this.prefetchAhead,
        want: this.want,
      } satisfies PrefetchCommand)
    } catch {
      this.worker = null
    }
  }
}

/** Первый по алфавиту `.db3` в каталоге bag-а (как `bag_reader.find_db3`). */
export function findDb3(bagDir: string): string {
  const names = readdirSync(bagDir).filter((name) => name.endsWith('.db3'))
  names.sort()
  const first = names[0]
  if (first === undefined) throw new Error(`No .db3 files in ${bagDir}`)
  return `${bagDir.replace(/[/\\]+$/, '')}/${first}`
}