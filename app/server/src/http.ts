/**
 * Ядро веб-вьюера: замер геометрии пути по выборке кадров и отдача кадров.
 *
 * Это зеркало класса Viewer из web_viewer.py: те же шаги замера в том же порядке
 * (стены -> ось -> пол по полосе вокруг оси -> геометрия рельсов), тот же формат
 * ответов. Числа обязаны совпадать с Python -- это проверяет src/parity.ts.
 *
 * Состояние после замера только читается. Правка параметров (setParams) собирает
 * новый объект параметров целиком и присваивает его одним присваиванием, поэтому
 * параллельные запросы кадров видят либо старый набор, либо новый (в Python за это
 * отвечает GIL, здесь -- атомарность присваивания ссылки).
 */

import { basename, dirname, resolve } from 'node:path'

import { BagFrames, COLOR_MODES, frameColors, trim } from './data/bagReader.ts'
import {
  BLOCKED_COLOR,
  RAIL_HEIGHT,
  TRAIN_TOP,
  ZONE_COLORS,
  ZONE_NAMES,
  blockedSpans,
  classify,
  corridorBlocked,
  detectWalls,
  findAxis,
  fitFloor,
  floorProfile,
  railXPositions,
  zoneParams,
} from './data/zones.ts'
import {
  FRAME_HEADER_BYTES,
  FRAME_KIND_LABEL8,
  FRAME_KIND_RGB8,
  FRAME_MAGIC,
  FRAME_VERSION,
} from './data/types.ts'
import type { MetaJson, ParamsJson, RailsJson, ZoneName, ZoneParams } from './data/types.ts'

/**
 * Ответ /meta. В контракте (и в web_viewer.py, и в типах клиента) у meta нет
 * полей contact_*: они меняются только ручкой /set. MetaJson в data/types.ts
 * объявлен как ParamsJson + метаданные, поэтому лишние поля вычитаем.
 */
export type MetaResponse = Omit<MetaJson, 'contact_rail' | 'contact_offset' | 'contact_side' | 'rail_height'>

/** Ширина полосы вокруг оси, по которой фитим пол (zones.fit_floor(x_half=2.0)). */
const FLOOR_HALF_BAND = 2.0
/** Шаг бинов вдоль пути для геометрии рельсов и засорённости коридора. */
const RAIL_BIN_WIDTH = 1.0

export interface ViewerConfig {
  maxDist: number
  behind: number
  mode: string
  maxPoints: number
  pointSize: number
  axisX: number | null
  gaugeMm: number
  obstaclePoints: number
  obstacleHeight: number
  railClearance: number
}

export interface SetParamsInput {
  axis?: number | null
  gaugeMm?: number | null
  contact?: number | null
  offset?: number | null
  side?: number | null
}

export interface PreparedFrame {
  xyz: Float32Array
  intensity: Float32Array | null
  ring: Uint16Array | null
}

const IS_LITTLE_ENDIAN = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1

/**
 * round() как в Python: десятичное округление, ровные половины -- к чётному, причём
 * по ТОЧНОМУ значению double. Наивное `Math.round(x * 10^n) / 10^n` на ровной
 * границе врёт: 3.9 - 1.435/2 в double равно 3.1825000000000001, умножение на 1000
 * даёт ровно 3182.5, и результат уезжает на 1e-3 от Python (замерено сверкой).
 * Поэтому разбираем double на мантиссу и степень двойки и округляем BigInt-ом.
 */
function roundPy(value: number, digits: number): number {
  if (!Number.isFinite(value)) return value
  if (digits < 0) {
    // Отрицательные знаки в вьюере не нужны, но пусть функция остаётся честной.
    const step = 10 ** -digits
    return roundPy(value / step, 0) * step
  }

  const bits = new DataView(new ArrayBuffer(8))
  bits.setFloat64(0, Math.abs(value))
  const high = bits.getUint32(0)
  const low = bits.getUint32(4)
  const exponent = (high >>> 20) & 0x7ff
  let mantissa = (BigInt(high & 0xfffff) << 32n) | BigInt(low)
  let power2: number
  if (exponent === 0) {
    power2 = -1074 // субнормальное: неявного старшего бита нет
  } else {
    mantissa |= 1n << 52n
    power2 = exponent - 1075
  }

  // |value| * 10^digits = mantissa * 10^digits * 2^power2 -- считаем точно.
  const scaled = mantissa * 10n ** BigInt(digits)
  let rounded: bigint
  if (power2 >= 0) {
    rounded = scaled << BigInt(power2)
  } else {
    const divisor = 1n << BigInt(-power2)
    const quotient = scaled / divisor
    const remainder = scaled - quotient * divisor
    const twice = remainder * 2n
    rounded = twice > divisor || (twice === divisor && quotient % 2n !== 0n) ? quotient + 1n : quotient
  }

  const magnitude = Number(rounded) / 10 ** digits
  return value < 0 ? -magnitude : magnitude
}

/** Цвет 0..1 -> uint8 для палитры: int(round(c * 255)), как в web_viewer.py. */
function roundByte(value: number): number {
  return roundPy(value * 255, 0)
}

/** Цвет 0..1 -> uint8 в payload кадра: np.clip(c * 255, 0, 255).astype(np.uint8) -- усечение. */
function colorByte(value: number): number {
  const scaled = value * 255
  if (scaled <= 0) return 0
  if (scaled >= 255) return 255
  return Math.floor(scaled)
}

function colorBytes(rgb: readonly number[]): number[] {
  return [roundByte(rgb[0] as number), roundByte(rgb[1] as number), roundByte(rgb[2] as number)]
}

/** Каждый step-й элемент массива (numpy a[::step] для одномерного массива). */
function strideFloats(src: Float32Array, step: number): Float32Array<ArrayBuffer> {
  const count = Math.ceil(src.length / step)
  const out = new Float32Array(count)
  for (let i = 0, j = 0; i < src.length; i += step, j++) out[j] = src[i] as number
  return out
}

/** Каждый step-й элемент ring (uint16). */
function strideUint16(src: Uint16Array, step: number): Uint16Array<ArrayBuffer> {
  const count = Math.ceil(src.length / step)
  const out = new Uint16Array(count)
  for (let i = 0, j = 0; i < src.length; i += step, j++) out[j] = src[i] as number
  return out
}

/** Каждая step-я точка interleaved-облака (numpy xyz[::step] для (N, 3)). */
function stridePoints(src: Float32Array, step: number): Float32Array<ArrayBuffer> {
  if (step <= 1) return src as Float32Array<ArrayBuffer>
  const points = src.length / 3
  const count = Math.ceil(points / step)
  const out = new Float32Array(count * 3)
  for (let i = 0, j = 0; i < points; i += step, j += 3) {
    out[j] = src[i * 3] as number
    out[j + 1] = src[i * 3 + 1] as number
    out[j + 2] = src[i * 3 + 2] as number
  }
  return out
}

function writeHeader(out: Uint8Array, kind: number, n: number): void {
  const view = new DataView(out.buffer, out.byteOffset, out.byteLength)
  for (let i = 0; i < FRAME_MAGIC.length; i++) view.setUint8(i, FRAME_MAGIC.charCodeAt(i))
  view.setUint8(4, FRAME_VERSION)
  view.setUint8(5, kind)
  view.setUint16(6, 0, true)
  view.setUint32(8, n, true)
}

function writePositions(out: Uint8Array, offset: number, xyz: Float32Array): void {
  if (IS_LITTLE_ENDIAN && offset % 4 === 0) {
    new Float32Array(out.buffer, out.byteOffset + offset, xyz.length).set(xyz)
    return
  }
  const view = new DataView(out.buffer, out.byteOffset, out.byteLength)
  for (let i = 0; i < xyz.length; i++) view.setFloat32(offset + i * 4, xyz[i] as number, true)
}

/** Источник кадров и метаданных для браузера. */
export class Viewer {
  readonly dbPath: string
  readonly bagName: string
  readonly maxDist: number
  readonly behind: number
  readonly mode: string
  readonly maxPoints: number
  readonly pointSize: number
  readonly obstaclePoints: number
  readonly obstacleHeight: number
  readonly railClearance: number

  private readonly frames: BagFrames
  private readonly axisFromCli: number | null
  /** Замер по выборке кадров: ещё и урезанное облако для оси/пола/рельсов. */
  private zoneCloud: Float32Array<ArrayBuffer> = new Float32Array(0)

  private paramsState: ZoneParams
  private wallList: number[] = []
  private floorRms = 0
  private railY: Float64Array<ArrayBufferLike> = new Float64Array(0)
  private railZ: Float64Array<ArrayBufferLike> = new Float64Array(0)
  private railBlocked: Uint8Array<ArrayBufferLike> = new Uint8Array(0)
  private spanList: Array<[number, number]> = []

  constructor(dbPath: string, cfg: ViewerConfig) {
    this.dbPath = resolve(dbPath)
    this.bagName = basename(dirname(this.dbPath))
    this.frames = new BagFrames(this.dbPath)
    if (this.frames.length === 0) throw new Error('No frames in bag')
    this.maxDist = cfg.maxDist
    this.behind = cfg.behind
    this.mode = cfg.mode
    this.maxPoints = cfg.maxPoints
    this.pointSize = cfg.pointSize
    this.obstaclePoints = cfg.obstaclePoints
    this.obstacleHeight = cfg.obstacleHeight
    this.railClearance = cfg.railClearance
    this.axisFromCli = cfg.axisX

    const params = zoneParams(cfg.gaugeMm ? { gauge: cfg.gaugeMm / 1000.0 } : {})
    // Свой массив стен: zoneParams() делит wallX с DEFAULT_ZONE_PARAMS.
    params.wallX = []
    this.paramsState = params
    this.measure()
  }

  // ---------------------------------------------------------------- чтение

  get total(): number {
    return this.frames.length
  }

  /** Параметры модели: после замера не меняются сами по себе (см. setParams). */
  get params(): Readonly<ZoneParams> {
    return this.paramsState
  }

  get walls(): readonly number[] {
    return this.wallList
  }

  get floor(): { a: number; b: number; rms: number } {
    return { a: this.paramsState.floorA, b: this.paramsState.floorB, rms: this.floorRms }
  }

  get spans(): ReadonlyArray<readonly [number, number]> {
    return this.spanList
  }

  close(): void {
    this.frames.close()
  }

  // ---------------------------------------------------------------- замер

  /**
   * Замер геометрии по выборке кадров: стены, ось, пол, рельсы, засорённость.
   * Порядок шагов -- как в Viewer._measure: он важен, потому что каждый следующий
   * шаг видит результат предыдущего (ось ищется по стенам, пол -- по оси).
   */
  private measure(samples = 6): void {
    const step = Math.max(1, Math.floor(this.total / samples))
    const parts: Float32Array[] = []
    for (let i = 0; i < this.total; i += step) {
      const frame = this.frames.frame(i)
      if (frame.xyz.length) parts.push(frame.xyz)
    }

    let total = 0
    for (const part of parts) total += part.length
    const cloud = new Float32Array(total)
    for (let i = 0, offset = 0; i < parts.length; i++) {
      cloud.set(parts[i] as Float32Array, offset)
      offset += (parts[i] as Float32Array).length
    }
    // Ось, пол и рельсы считаем по каждой третьей точке замера: в Python cloud[::3]
    // режет двумерный (N, 3) по первой оси, то есть берёт каждую третью точку целиком.
    this.zoneCloud = stridePoints(cloud, 3)

    const p = this.paramsState
    const walls = cloud.length ? detectWalls(cloud, p) : []
    this.wallList = walls
    if (walls.length >= 2) p.wallX = walls

    if (this.axisFromCli !== null) {
      p.axisX = this.axisFromCli
    } else if (cloud.length) {
      p.axisX = findAxis(this.zoneCloud, p, -this.maxDist, this.behind, {
        clearance: this.railClearance,
        minPoints: this.obstaclePoints,
      })
    }

    // Пол -- только по полосе вокруг оси пути: основания стен иначе тянут нижнюю
    // огибающую вниз и плоскость выходит кривой.
    if (cloud.length) {
      const fit = fitFloor(cloud, { xCenter: p.axisX, xHalf: FLOOR_HALF_BAND })
      if (Number.isFinite(fit.a) && Number.isFinite(fit.b)) {
        p.floorA = fit.a
        p.floorB = fit.b
      }
      this.floorRms = fit.rms
    } else {
      this.floorRms = 0
    }

    if (this.axisFromCli === null && cloud.length) {
      p.axisX = findAxis(this.zoneCloud, p, -this.maxDist, this.behind, {
        clearance: this.railClearance,
        minPoints: this.obstaclePoints,
      })
    }
    this.refreshRailGeometry()
  }

  /** Геометрия рельсов и засорённость коридора по текущим параметрам. */
  private refreshRailGeometry(): void {
    const p = this.paramsState
    const { ys, blocked } = corridorBlocked(this.zoneCloud, p, -this.maxDist, this.behind, {
      binWidth: RAIL_BIN_WIDTH,
      clearance: this.railClearance,
      minPoints: this.obstaclePoints,
      minHeight: this.obstacleHeight,
    })
    this.railY = ys
    this.railBlocked = blocked
    // Профиль пола линеен, поэтому высота точки рельса -- плоскость + RAIL_HEIGHT.
    const profile = floorProfile(ys, p)
    this.railZ = new Float64Array(ys.length)
    for (let i = 0; i < ys.length; i++) this.railZ[i] = (profile[i] as number) + RAIL_HEIGHT
    this.spanList = blockedSpans(ys, blocked, RAIL_BIN_WIDTH)
  }

  // ---------------------------------------------------------------- кадры

  private prepare(idx: number): PreparedFrame {
    const frame = this.frames.frame(idx)
    let [xyz, intensity, ring] = trim(frame.xyz, frame.intensity, frame.ring, this.maxDist, this.behind)
    const points = xyz.length / 3
    if (this.maxPoints > 0 && this.maxPoints < points) {
      const step = Math.ceil(points / this.maxPoints)
      xyz = stridePoints(xyz, step)
      intensity = intensity === null ? null : strideFloats(intensity, step)
      ring = ring === null ? null : strideUint16(ring, step)
    }
    return { xyz, intensity, ring }
  }

  /**
   * Бинарный кадр: заголовок 12 байт, позиции n*3 float32 LE, дальше метки (kind 1)
   * или цвета (kind 0). Без общего лока: состояние после замера только читается.
   */
  frameBytes(idxRaw: number, mode: string): Uint8Array {
    const idx = Math.max(0, Math.min(Math.trunc(idxRaw), this.total - 1))
    const { xyz, intensity, ring } = this.prepare(idx)
    const points = xyz.length / 3
    const labelMode = mode === 'zones'
    const payloadBytes = labelMode ? points : points * 3
    const out = new Uint8Array(FRAME_HEADER_BYTES + points * 12 + payloadBytes)
    writeHeader(out, labelMode ? FRAME_KIND_LABEL8 : FRAME_KIND_RGB8, points)
    writePositions(out, FRAME_HEADER_BYTES, xyz)

    const payloadOffset = FRAME_HEADER_BYTES + points * 12
    if (labelMode) {
      out.set(classify(xyz, this.paramsState), payloadOffset)
    } else {
      const colors = frameColors(xyz, intensity, mode, this.maxDist, ring)
      for (let i = 0; i < points; i++) {
        out[payloadOffset + i * 3] = colorByte(colors[i * 3] as number)
        out[payloadOffset + i * 3 + 1] = colorByte(colors[i * 3 + 1] as number)
        out[payloadOffset + i * 3 + 2] = colorByte(colors[i * 3 + 2] as number)
      }
    }
    return out
  }

  // ---------------------------------------------------------------- JSON

  private floorJson(): ParamsJson['floor'] {
    const p = this.paramsState
    return {
      a: roundPy(p.floorA, 4),
      b: roundPy(p.floorB, 6),
      rms: roundPy(this.floorRms, 3),
      grade_pct: roundPy(p.floorB * 100.0, 2),
    }
  }

  private railsJson(): RailsJson {
    const rails = railXPositions(this.paramsState)
    const blocked = new Array<number>(this.railBlocked.length)
    for (let i = 0; i < this.railBlocked.length; i++) blocked[i] = this.railBlocked[i] ? 1 : 0
    return {
      x: rails.map((rail) => roundPy(rail[0], 3)),
      heights: rails.map((rail) => roundPy(rail[1], 3)),
      y: Array.from(this.railY, (v) => roundPy(v, 3)),
      z: Array.from(this.railZ, (v) => roundPy(v, 3)),
      blocked,
    }
  }

  private spansJson(): Array<[number, number]> {
    return this.spanList.map(([a, b]) => [roundPy(a, 1), roundPy(b, 1)] as [number, number])
  }

  /** Ответ /set: то, что меняется при правке модели рельсов (_params_json). */
  paramsJson(): ParamsJson {
    const p = this.paramsState
    return {
      axis_x: roundPy(p.axisX, 2),
      gauge_mm: roundPy(p.gauge * 1000, 0),
      contact_rail: p.contactRail,
      contact_offset: roundPy(p.contactOffset, 3),
      contact_side: p.contactSide,
      rail_height: roundPy(RAIL_HEIGHT, 3),
      floor: this.floorJson(),
      rails: this.railsJson(),
      blocked_spans: this.spansJson(),
    }
  }

  /** Ответ /meta: метаданные сессии и легенда клиента. */
  meta(): MetaResponse {
    const p = this.paramsState
    const stamps = this.frames.timestamps
    const span = stamps.length > 1 ? Number((stamps[stamps.length - 1] as bigint) - (stamps[0] as bigint)) / 1e9 : 0
    return {
      bag: this.bagName,
      frames: this.total,
      duration_s: roundPy(span, 2),
      hz: span ? roundPy(this.total / span, 2) : 0.0,
      max_dist: this.maxDist,
      behind: this.behind,
      default_mode: this.mode,
      modes: [...COLOR_MODES],
      point_size: this.pointSize,
      max_points_hint: this.maxPoints > 0 ? this.maxPoints : 1000000,
      floor: this.floorJson(),
      walls: this.wallList.map((w) => roundPy(w, 2)),
      axis_x: roundPy(p.axisX, 2),
      gauge_mm: roundPy(p.gauge * 1000, 0),
      rails: this.railsJson(),
      blocked_spans: this.spansJson(),
      grid: { spacing: 5.0, half_width: 40.0, length: roundPy(this.maxDist + this.behind, 1), z: 0.0 },
      zone_names: [...ZONE_NAMES] as ZoneName[],
      zone_colors: ZONE_NAMES.map((name) => colorBytes(ZONE_COLORS[name] as readonly number[])),
      blocked_color: colorBytes(BLOCKED_COLOR as readonly number[]),
      rail_color: colorBytes(ZONE_COLORS.rail as readonly number[]),
      train_top: TRAIN_TOP,
      obstacle_points: this.obstaclePoints,
      rail_clearance: this.railClearance,
    }
  }

  /**
   * Живая правка модели рельсов из браузера.
   *
   * Пол пересчитывается по полосе вокруг новой оси: он от неё зависит (по всему
   * кадру нижнюю огибающую тянут вниз основания стен). Новый набор параметров
   * собирается целиком и присваивается одним присваиванием -- запросы кадров
   * читают состояние атомарно.
   */
  setParams(input: SetParamsInput): ParamsJson {
    const p: ZoneParams = { ...this.paramsState, wallX: [...this.paramsState.wallX] }
    if (input.axis !== undefined && input.axis !== null) p.axisX = input.axis
    if (input.gaugeMm) p.gauge = input.gaugeMm / 1000.0
    if (input.contact !== undefined && input.contact !== null) p.contactRail = Boolean(input.contact)
    if (input.offset !== undefined && input.offset !== null) p.contactOffset = Math.max(0.05, input.offset)
    if (input.side !== undefined && input.side !== null) p.contactSide = input.side >= 0 ? 1 : -1

    if (this.zoneCloud.length) {
      const fit = fitFloor(this.zoneCloud, { xCenter: p.axisX, xHalf: FLOOR_HALF_BAND })
      if (Number.isFinite(fit.a) && Number.isFinite(fit.b)) {
        p.floorA = fit.a
        p.floorB = fit.b
        this.floorRms = fit.rms
      }
    }

    this.paramsState = p
    this.refreshRailGeometry()
    return this.paramsJson()
  }
}