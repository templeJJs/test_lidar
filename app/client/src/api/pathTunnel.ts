// Блоки `path` и `tunnel` из /meta — «Линия хода» и «Туннель безопасности».
//
// Типы лежат отдельным файлом, а не в api/types.ts: тот описывает уже
// сложившийся контракт, а эти два блока сервер может и не отдавать (например,
// на старой сборке без пересчёта линии). Клиент обязан работать без них,
// поэтому чтение мягкое: любые незнакомые/битые поля превращаются в null, а не
// в исключение.

export type PathVariantName = 'corridor' | 'rails' | 'axis'

export const PATH_VARIANT_NAMES: readonly PathVariantName[] = ['corridor', 'rails', 'axis']

/** Русские подписи источников линии хода (для панели). */
export const PATH_VARIANT_LABELS: Record<PathVariantName, string> = {
  corridor: 'коридор',
  rails: 'рельсы',
  axis: 'ось',
}

export interface PathVariantJson {
  y: number[]
  x: number[]
  z?: number[]
  source?: string
  /** Вариант посчитан, но данных нет (например, коридор занят) — «нет данных». */
  unavailable?: boolean
}

export interface PathJson {
  step: number
  z_offset: number
  variants: Record<PathVariantName, PathVariantJson | null>
}

export interface TunnelJson {
  half_width: number
  z_low: number
  z_high: number
  rail_height: number
  exclude_contact: boolean
  spans: Array<[number, number]>
  bins_blocked: number
  bins_total: number
}

function isNum(v: unknown): v is number {
  return typeof v === 'number' && Number.isFinite(v)
}

function num(v: unknown): number | null {
  return isNum(v) ? v : null
}

function asRecord(v: unknown): Record<string, unknown> | null {
  return v !== null && typeof v === 'object' ? (v as Record<string, unknown>) : null
}

function numArray(v: unknown, min = 2): number[] | null {
  if (!Array.isArray(v) || v.length < min) return null
  const out = new Array<number>(v.length)
  for (let i = 0; i < v.length; i++) {
    const n = num(v[i])
    if (n === null) return null
    out[i] = n
  }
  return out
}

function readVariant(raw: unknown): PathVariantJson | null {
  const obj = asRecord(raw)
  if (!obj) return null
  const y = numArray(obj.y)
  const x = numArray(obj.x)
  if (!y || !x || y.length !== x.length) return null
  const z = numArray(obj.z)
  return {
    y,
    x,
    z: z && z.length === y.length ? z : undefined,
    source: typeof obj.source === 'string' ? obj.source : undefined,
    unavailable: obj.unavailable === true,
  }
}

function readSpans(raw: unknown): Array<[number, number]> {
  if (!Array.isArray(raw)) return []
  const out: Array<[number, number]> = []
  for (const item of raw) {
    if (!Array.isArray(item) || item.length < 2) continue
    const a = num(item[0])
    const b = num(item[1])
    if (a === null || b === null) continue
    out.push([a, b])
  }
  return out
}

/** Блок `path` из меты; null — сервер его не отдаёт или он нечитаемый. */
export function readPath(meta: unknown): PathJson | null {
  const root = asRecord(meta)
  if (!root) return null
  const raw = asRecord(root.path)
  if (!raw) return null
  const rawVariants = asRecord(raw.variants)
  if (!rawVariants) return null
  const variants = { corridor: null, rails: null, axis: null } as PathJson['variants']
  let usable = false
  for (const name of PATH_VARIANT_NAMES) {
    const v = readVariant(rawVariants[name])
    variants[name] = v
    if (v && !v.unavailable) usable = true
  }
  if (!usable) return null
  return {
    step: num(raw.step) ?? 1,
    z_offset: num(raw.z_offset) ?? 0.16,
    variants,
  }
}

/** Блок `tunnel` из меты; null — сервер его не отдаёт или он нечитаемый. */
export function readTunnel(meta: unknown): TunnelJson | null {
  const root = asRecord(meta)
  if (!root) return null
  const raw = asRecord(root.tunnel)
  if (!raw) return null
  const half = num(raw.half_width)
  if (half === null) return null
  return {
    half_width: half,
    z_low: num(raw.z_low) ?? 0.1,
    z_high: num(raw.z_high) ?? 3.7,
    rail_height: num(raw.rail_height) ?? 0.16,
    exclude_contact: raw.exclude_contact !== false,
    spans: readSpans(raw.spans),
    bins_blocked: num(raw.bins_blocked) ?? 0,
    bins_total: num(raw.bins_total) ?? 0,
  }
}

/** Суммарная длина занятых участков, м. */
export function spansLength(spans: readonly (readonly [number, number])[]): number {
  let sum = 0
  for (const [a, b] of spans) sum += Math.abs(b - a)
  return sum
}