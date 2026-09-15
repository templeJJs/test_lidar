// Разметка облака точек туннеля по зонам: rail / bed / ceiling / wall / middle.
// Порт `zones.py` (без open3d: `makeRailLines` отдаёт массивы, а не LineSet).
//
// Почему именно так:
// - Пол в туннеле НАКЛОННЫЙ (уклон ~2.16%), поэтому все пороги задаются не
//   абсолютным Z, а высотой НАД плоскостью пола z_floor(y) = floorA + floorB*y.
// - Рельсы НЕ отделяются ни интенсивностью, ни геометрией точек (проверено на
//   for_hackathon/doubleT_obstacle), поэтому это МОДЕЛЬ, привязанная к оси пути
//   (axisX) и ширине колеи (gauge).
// - Приоритет меток: первое совпадение побеждает: rail -> bed -> ceiling -> wall
//   -> middle. Реализовано перезаписью масок от младшего приоритета к старшему.
//
// Точек на кадр 300k+, поэтому циклы однопроходные и без временных объектов.

import {
  arange,
  argMax,
  histogram2dCounts,
  histogramCounts,
  linspace,
  median,
  percentileFromSorted,
  polyfit1,
  searchSortedLeft,
  searchSortedRight,
} from './numeric.ts'
import type { FloorFit, ZoneName, ZoneParams } from './types.ts'

export const ZONE_NAMES = ['rail', 'bed', 'ceiling', 'wall', 'middle'] as const

/** Индекс зоны в таблице меток (порядок ZONE_NAMES). */
export const ZONE_LABEL: Record<ZoneName, number> = {
  rail: 0,
  bed: 1,
  ceiling: 2,
  wall: 3,
  middle: 4,
}

export const ZONE_COLORS: Record<ZoneName, [number, number, number]> = {
  rail: [1.0, 0.55, 0.0], // ярко-оранжевый
  bed: [0.35, 0.45, 0.6], // серо-синий       ("низ")
  ceiling: [0.85, 0.3, 0.9], // маджента      ("потолок")
  wall: [0.0, 0.75, 0.7], // бирюзовый
  middle: [0.25, 0.85, 0.35], // зелёный    ("средний уровень")
}

// Рельс рисовать сквозь вертикальное препятствие нельзя: линия утверждала бы
// "здесь можно ехать". Заблокированный участок красится этим цветом.
export const BLOCKED_COLOR: [number, number, number] = [1.0, 0.08, 0.08]

/** Метры: головка рельса над плоскостью пола. */
export const RAIL_HEIGHT = 0.16

// Габарит по высоте: всё выше ложа и ниже этой отметки внутри коридора считаем
// препятствием (поезд не может проехать сквозь него).
export const TRAIN_TOP = 3.7

// Препятствие должно ИМЕТЬ ВЫСОТУ. Счёт точек без этого условия красил красным
// кромку балласта, кабельный лоток и стенку рельса вдоль края коридора:
// замерено на doubleT_obstacle -- 13-27 м из 46 м, при высоте точек в этих
// бинах 0.0-0.19 м, тогда как у настоящего объекта на y = -38 м высота 1.0 м.
export const MIN_OBSTACLE_HEIGHT = 0.5

const COLOR_TABLE: Float64Array = buildColorTable()

function buildColorTable(): Float64Array {
  const table = new Float64Array(ZONE_NAMES.length * 3)
  ZONE_NAMES.forEach((name, i) => {
    const color = ZONE_COLORS[name]
    table[i * 3] = color[0]
    table[i * 3 + 1] = color[1]
    table[i * 3 + 2] = color[2]
  })
  return table
}

/** Умолчания ZoneParams из zones.py (dataclass ZoneParams). */
export const DEFAULT_ZONE_PARAMS: ZoneParams = {
  floorA: -1.886,
  floorB: 0.021585,
  floorBand: 0.3, // метры над плоскостью пола -> 'bed'
  ceilingZ: 2.0, // абсолютный Z -> 'ceiling'
  axisX: 1.6, // поперечное положение оси пути
  gauge: 1.520, // метры
  railHalfWidth: 0.12, // метры вокруг центра каждого рельса
  railLow: -0.1, // метры относительно плоскости пола
  railHigh: 0.35,
  wallX: [], // найденные положения стен по X
  wallBand: 0.5, // метры
  detectWalls: true,
  contactRail: true, // рисовать/классифицировать контактный рельс
  contactSide: 1, // +1 -> со стороны +x от оси, -1 -> со стороны -x
  contactOffset: 1.46, // метры от ОСИ пути (gauge/2 + 0.70)
  contactHeight: 0.2, // метры над плоскостью пола (головка рельса)
  contactHalfWidth: 0.1, // метры
}

/** Параметры зон с переопределениями (в Python -- `ZoneParams(**kw)`). */
export function zoneParams(overrides: Partial<ZoneParams> = {}): ZoneParams {
  return { ...DEFAULT_ZONE_PARAMS, ...overrides }
}

/**
 * Z плоскости пола: a + b*y. Пол наклонён, поэтому пороги задаются над ним.
 * Работает и для скаляра, и для массива Y.
 */
export function floorProfile(y: number, params: ZoneParams): number
export function floorProfile(y: ArrayLike<number>, params: ZoneParams): Float64Array
export function floorProfile(
  y: number | ArrayLike<number>,
  params: ZoneParams,
): number | Float64Array {
  if (typeof y === 'number') return params.floorA + params.floorB * y
  const out = new Float64Array(y.length)
  for (let i = 0; i < y.length; i++) out[i] = params.floorA + params.floorB * y[i]!
  return out
}

export interface FitFloorOptions {
  yMin?: number
  yMax?: number
  bins?: number
  percentile?: number
  clip?: number
  xCenter?: number
  xHalf?: number
}

/**
 * Робастная подгонка пола: {a, b, rms} для z_floor(y) = a + b*y.
 *
 * Пол -- нижняя огибающая поверхности, поэтому берём низкий перцентиль Z по
 * Y-бинам, затем прямую МНК и ОТБРАКОВКУ выбросов по sigma: одиночные бины с
 * конструкциями или редкими выбросами иначе перекашивают плоскость.
 *
 * Полосу `xCenter +- xHalf` задавать ОБЯЗАТЕЛЬНО: по всему кадру нижнюю
 * огибающую тянут вниз основания стен и плоскость получается кривой
 * (замерено: rms 0.18-0.20 по всему кадру против 0.042 в полосе пути x 2..5).
 */
export function fitFloor(xyz: Float32Array, opts: FitFloorOptions = {}): FloorFit {
  const yMin = opts.yMin ?? -60.0
  const yMax = opts.yMax ?? -2.0
  const bins = opts.bins ?? 30
  const percentile = opts.percentile ?? 5.0
  const clip = opts.clip ?? 2.0
  const xCenter = opts.xCenter
  const xHalf = opts.xHalf

  const n = xyz.length / 3
  if (n === 0) return { a: NaN, b: 0.0, rms: 0.0 }

  const useXBand = xCenter !== undefined && xHalf !== undefined
  // Отбор точек полосы запоминаем маской: дальше по ней идут ещё два прохода
  // (подсчёт бинов и раскладка Z), и дублировать условие отбора не хочется.
  const selectedMask = new Uint8Array(n)
  let selected = 0
  for (let i = 0; i < n; i++) {
    const y = xyz[i * 3 + 1]!
    if (!(y >= yMin && y < yMax)) continue
    if (useXBand && !(Math.abs(xyz[i * 3]! - xCenter) <= xHalf)) continue
    selectedMask[i] = 1
    selected++
  }

  const fallback = (): FloorFit => ({ a: medianAllZ(xyz, n), b: 0.0, rms: 0.0 })
  if (selected === 0) return fallback()

  const edges = linspace(yMin, yMax, bins + 1)
  const centers = new Float64Array(bins)
  for (let b = 0; b < bins; b++) centers[b] = 0.5 * (edges[b]! + edges[b + 1]!)

  // Раскладываем выбранные точки по бинам (порядок в бине не важен: дальше
  // нужны только порядковые статистики) и считаем перцентиль Z в каждом бине.
  const binIndex = new Int32Array(selected)
  const binCount = new Float64Array(bins)
  let k = 0
  for (let i = 0; i < n; i++) {
    if (!selectedMask[i]) continue
    let b = searchSortedRight(edges, xyz[i * 3 + 1]!) - 1
    if (b < 0) b = 0
    else if (b >= bins) b = bins - 1
    binIndex[k] = b
    binCount[b] = binCount[b]! + 1
    k++
  }

  const offset = new Int32Array(bins + 1)
  let acc = 0
  for (let b = 0; b < bins; b++) {
    offset[b] = acc
    acc += binCount[b]!
  }
  offset[bins] = acc

  const binZ = new Float64Array(selected)
  const fill = Int32Array.from(offset.subarray(0, bins))
  k = 0
  for (let i = 0; i < n; i++) {
    if (!selectedMask[i]) continue
    const b = binIndex[k]!
    binZ[fill[b]!] = xyz[i * 3 + 2]!
    fill[b] = fill[b]! + 1
    k++
  }

  const levels = new Float64Array(bins).fill(NaN)
  let goodCount = 0
  for (let b = 0; b < bins; b++) {
    if (binCount[b]! < 20) continue
    const segment = binZ.subarray(offset[b]!, offset[b + 1]!)
    segment.sort()
    levels[b] = percentileFromSorted(segment, percentile)
    goodCount++
  }
  if (goodCount < 2) return fallback()

  let good: number[] = []
  for (let b = 0; b < bins; b++) if (!Number.isNaN(levels[b]!)) good.push(b)

  const fitFor = (idx: number[]): [number, number] => {
    const cx = new Float64Array(idx.length)
    const ly = new Float64Array(idx.length)
    idx.forEach((b, i) => {
      cx[i] = centers[b]!
      ly[i] = levels[b]!
    })
    return polyfit1(cx, ly)
  }

  let [slope, intercept] = fitFor(good)
  for (let iteration = 0; iteration < 3; iteration++) {
    const resid = new Float64Array(good.length)
    for (let i = 0; i < good.length; i++) {
      resid[i] = levels[good[i]!]! - (intercept + slope * centers[good[i]!]!)
    }
    const scale = medianAbs(resid) * 1.4826 || 1e-6
    let keepCount = 0
    const keep = new Array<boolean>(good.length)
    for (let i = 0; i < good.length; i++) {
      keep[i] = Math.abs(resid[i]!) <= clip * scale
      if (keep[i]!) keepCount++
    }
    if (keepCount === good.length || keepCount < 3) break
    good = good.filter((_b, i) => keep[i]!)
    ;[slope, intercept] = fitFor(good)
  }

  let sum = 0.0
  for (const b of good) {
    const resid = intercept + slope * centers[b]! - levels[b]!
    sum += resid * resid
  }
  const rms = Math.sqrt(sum / good.length)
  return { a: intercept, b: slope, rms }
}

/** Медиана абсолютных значений (для отбраковки выбросов в fitFloor). */
function medianAbs(values: Float64Array): number {
  const abs = new Float64Array(values.length)
  for (let i = 0; i < values.length; i++) abs[i] = Math.abs(values[i]!)
  return median(abs)
}

/** Медиана Z по всему кадру (аварийный путь, как в Python-эталоне). */
function medianAllZ(xyz: Float32Array, n: number): number {
  const z = new Float64Array(n)
  for (let i = 0; i < n; i++) z[i] = xyz[i * 3 + 2]!
  return median(z)
}

/**
 * Ищем две доминирующие вертикальные стены по гистограмме X.
 *
 * Учитываем только точки ВЫШЕ пола (z - z_floor(y) >= floorBand): пол и рельсы
 * дают сильные пики по X и мешали бы. Берём самый сильный бин, затем самый
 * сильный бин минимум в minSeparation метрах от него. Возвращаем центры по
 * возрастанию; если второй не найден -- один элемент.
 */
export function detectWalls(
  xyz: Float32Array,
  params: ZoneParams,
  minSeparation = 2.0,
  binWidth = 0.1,
): number[] {
  const n = xyz.length / 3
  if (n === 0) return []

  let above = 0
  let minX = Infinity
  let maxX = -Infinity
  const aboveMask = new Uint8Array(n)
  for (let i = 0; i < n; i++) {
    const x = xyz[i * 3]!
    const zrel = xyz[i * 3 + 2]! - (params.floorA + params.floorB * xyz[i * 3 + 1]!)
    if (zrel < params.floorBand) continue
    aboveMask[i] = 1
    above++
    if (x < minX) minX = x
    if (x > maxX) maxX = x
  }
  if (above === 0) return []

  const x0 = Math.floor(minX / binWidth) * binWidth
  const x1 = Math.ceil(maxX / binWidth) * binWidth
  const edges = arange(x0, x1 + binWidth, binWidth)

  const xs = new Float64Array(above)
  let k = 0
  for (let i = 0; i < n; i++) {
    if (aboveMask[i]) xs[k++] = xyz[i * 3]!
  }

  const counts = histogramCounts(xs, edges)
  if (counts.length === 0) return []

  const centers = new Float64Array(counts.length)
  for (let b = 0; b < counts.length; b++) centers[b] = 0.5 * (edges[b]! + edges[b + 1]!)

  const first = argMax(counts)
  let second = 0
  let secondCount = 0
  for (let b = 0; b < counts.length; b++) {
    if (Math.abs(centers[b]! - centers[first]!) < minSeparation) continue
    if (counts[b]! > secondCount) {
      second = b
      secondCount = counts[b]!
    }
  }
  if (secondCount <= 0) return [centers[first]!]
  return [centers[first]!, centers[second]!].sort((a, b) => a - b)
}

/**
 * [(x, высота_над_полом), ...] для КАЖДОГО рельса: ходовые + контактный.
 *
 * Почему рельсов три, а не два: у электрифицированного пути, кроме двух ходовых
 * рельсов на `axis +- gauge/2`, есть контактный рельс. Он вынесен в сторону от
 * оси на `contactOffset` и стоит на изоляторах, то есть на своей высоте
 * `contactHeight`, а не на RAIL_HEIGHT. Полуширина у него тоже своя
 * (`contactHalfWidth`), иначе классификация "утащила" бы его в полосу ходового
 * рельса. При `contactRail = false` остаются ровно два ходовых.
 */
export function railXPositions(params: ZoneParams): Array<[number, number]> {
  const half = params.gauge / 2.0
  const rails: Array<[number, number]> = [
    [params.axisX - half, RAIL_HEIGHT],
    [params.axisX + half, RAIL_HEIGHT],
  ]
  if (params.contactRail) {
    rails.push([params.axisX + params.contactSide * params.contactOffset, params.contactHeight])
  }
  return rails
}

/**
 * Метки зон (uint8, индексы ZONE_NAMES). Первое совпадение побеждает.
 *
 * Порядок приоритета rail -> bed -> ceiling -> wall -> middle реализован
 * перезаписью метки от младшего приоритета к старшему.
 */
export function classify(xyz: Float32Array, params: ZoneParams): Uint8Array {
  const n = xyz.length / 3
  const labels = new Uint8Array(n).fill(ZONE_LABEL.middle)
  if (n === 0) return labels

  const wallX = params.wallX
  const hasWalls = wallX.length > 0
  const railActive = params.railHalfWidth > 0 || params.contactRail
  const rails = railActive ? railXPositions(params) : null
  const runningBand = params.railLow
  const runningBandHigh = params.railHigh
  const contactLow = params.contactHeight + params.railLow
  const contactHigh = params.contactHeight + params.railHigh

  for (let i = 0; i < n; i++) {
    const x = xyz[i * 3]!
    const z = xyz[i * 3 + 2]!
    const zrel = z - (params.floorA + params.floorB * xyz[i * 3 + 1]!)

    if (hasWalls) {
      for (const wx of wallX) {
        if (Math.abs(x - wx) < params.wallBand) {
          labels[i] = ZONE_LABEL.wall
          break
        }
      }
    }
    if (z >= params.ceilingZ) labels[i] = ZONE_LABEL.ceiling
    if (zrel < params.floorBand) labels[i] = ZONE_LABEL.bed

    if (rails !== null) {
      const inRunningBand = zrel >= runningBand && zrel <= runningBandHigh
      const inContactBand = zrel >= contactLow && zrel <= contactHigh
      let isRail = false
      if (inRunningBand) {
        // Ходовые рельсы: полоса railLow..railHigh ОТ плоскости пола.
        if (Math.abs(x - rails[0]![0]) <= params.railHalfWidth) isRail = true
        else if (Math.abs(x - rails[1]![0]) <= params.railHalfWidth) isRail = true
      }
      // Контактный рельс: своя полуширина и свой коридор вокруг contactHeight.
      if (!isRail && inContactBand && rails.length > 2) {
        if (Math.abs(x - rails[2]![0]) <= params.contactHalfWidth) isRail = true
      }
      if (isRail) labels[i] = ZONE_LABEL.rail
    }
  }
  return labels
}

/** n*3 float64 цветов по метке точки; берём ровно из ZONE_COLORS. */
export function zoneColors(xyz: Float32Array, params: ZoneParams): Float64Array {
  return colorsFromLabels(classify(xyz, params))
}

/** n*3 float64 цветов по уже посчитанным меткам (кадр классифицируется один раз). */
export function colorsFromLabels(labels: Uint8Array): Float64Array {
  const out = new Float64Array(labels.length * 3)
  for (let i = 0; i < labels.length; i++) {
    const src = labels[i]! * 3
    out[i * 3] = COLOR_TABLE[src]!
    out[i * 3 + 1] = COLOR_TABLE[src + 1]!
    out[i * 3 + 2] = COLOR_TABLE[src + 2]!
  }
  return out
}

/** {имя_зоны: количество} по всем пяти зонам (нули не опускаем). */
export function classifyCounts(xyz: Float32Array, params: ZoneParams): Record<ZoneName, number> {
  const labels = classify(xyz, params)
  const counts: Record<ZoneName, number> = { rail: 0, bed: 0, ceiling: 0, wall: 0, middle: 0 }
  for (let i = 0; i < labels.length; i++) counts[ZONE_NAMES[labels[i]!]!]++
  return counts
}

/**
 * Узлы вдоль пути. Общий для коридора и для линий рельсов, чтобы разметка
 * засорённости и геометрия рельсов совпадали точка-в-точку.
 */
export function yNodes(yFrom: number, yTo: number, step: number): Float64Array {
  const ys = arange(yFrom, yTo, step)
  if (ys.length === 0 || ys[ys.length - 1] !== yTo) {
    const out = new Float64Array(ys.length + 1)
    out.set(ys)
    out[ys.length] = yTo
    return out
  }
  return ys
}

/**
 * Высота точек в каждом Y-бине: размах перцентилей 5..95, 0 у пустых бинов.
 *
 * Нужна, чтобы отличать препятствие от тонкой поверхности: у столба, человека
 * или упавшего предмета точки занимают по высоте десятки сантиметров, а у
 * кромки балласта, лотка или стенки рельса -- сантиметры.
 */
export function binSpan(
  values: ArrayLike<number>,
  y: ArrayLike<number>,
  edges: ArrayLike<number>,
): Float64Array {
  const nb = edges.length - 1
  const span = new Float64Array(Math.max(nb, 0))
  if (nb <= 0 || values.length === 0) return span

  // Порядок внутри бина не важен (нужны только порядковые статистики), поэтому
  // вместо argsort+searchsorted раскладываем значения по бинам подсчётом.
  // Бин каждого значения считаем один раз: раскладка по бинам идёт подсчётом,
  // порядок внутри бина не важен (нужны только порядковые статистики), поэтому
  // вместо argsort+searchsorted у Python -- два линейных прохода.
  const binIndex = new Int32Array(y.length)
  const counts = new Int32Array(nb)
  for (let i = 0; i < y.length; i++) {
    let b = searchSortedRight(edges, y[i]!) - 1
    if (b < 0) b = 0
    else if (b >= nb) b = nb - 1
    binIndex[i] = b
    counts[b] = counts[b]! + 1
  }
  const offset = new Int32Array(nb + 1)
  let acc = 0
  for (let b = 0; b < nb; b++) {
    offset[b] = acc
    acc += counts[b]!
  }
  offset[nb] = acc

  const sorted = new Float64Array(values.length)
  const fill = Int32Array.from(offset.subarray(0, nb))
  for (let i = 0; i < y.length; i++) {
    const b = binIndex[i]!
    sorted[fill[b]!] = values[i]!
    fill[b] = fill[b]! + 1
  }

  for (let b = 0; b < nb; b++) {
    if (counts[b]! < 2) continue
    const segment = sorted.subarray(offset[b]!, offset[b + 1]!)
    segment.sort()
    span[b] = percentileFromSorted(segment, 95.0) - percentileFromSorted(segment, 5.0)
  }
  return span
}

export interface CorridorOptions {
  binWidth?: number
  clearance?: number
  trainTop?: number
  minPoints?: number
  minHeight?: number
}

/**
 * Засорённость коридора пути вдоль Y: (узлы, маска "заблокировано").
 *
 * Поезд не может проехать сквозь вертикальное препятствие, поэтому рельс
 * рисуется не сплошной линией, а разорванной там, где коридор занят. В коридор
 * берём точки ВЫШЕ ложа (пол и балласт не мешают) и НИЖЕ габарита поезда
 * (`trainTop`) в полосе `|x - axisX| < gauge/2 + clearance`.
 *
 * Одного счёта точек мало: бин набирает `minPoints` и на тонкой поверхности,
 * тянущейся вдоль края коридора (кромка балласта, лоток, стенка рельса), из-за
 * чего красным становилась половина пути. Поэтому бин занят только если точки в
 * нём занимают по высоте не меньше `minHeight` (binSpan). `minHeight = 0`
 * возвращает прежнее поведение -- «занято по одному счёту точек».
 */
export function corridorBlocked(
  xyz: Float32Array,
  params: ZoneParams,
  yFrom: number,
  yTo: number,
  opts: CorridorOptions = {},
): { ys: Float64Array; blocked: Uint8Array } {
  const binWidth = opts.binWidth ?? 1.0
  const clearance = opts.clearance ?? 0.3
  const trainTop = opts.trainTop ?? TRAIN_TOP
  const minPoints = Math.max(Math.trunc(opts.minPoints ?? 20), 1)
  const minHeight = opts.minHeight ?? MIN_OBSTACLE_HEIGHT

  const ys = yNodes(yFrom, yTo, binWidth)
  const n = ys.length
  if (n === 0) return { ys, blocked: new Uint8Array(0) }

  const blocked = new Uint8Array(n)
  const points = xyz.length / 3
  if (points === 0) return { ys, blocked }

  const half = params.gauge / 2.0 + clearance
  const edges = new Float64Array(n + 1)
  for (let i = 0; i < n; i++) edges[i] = ys[i]! - binWidth / 2.0
  edges[n] = ys[n - 1]! + binWidth / 2.0

  // Точки коридора: выше ложа (пол и балласт не мешают) и ниже габарита
  // поезда, в полосе |x - axisX| < gauge/2 + clearance. Маску запоминаем --
  // по ней идут второй проход (раскладка по бинам) и проверка высоты.
  const insideMask = new Uint8Array(points)
  let inside = 0
  for (let i = 0; i < points; i++) {
    const x = xyz[i * 3]!
    const y = xyz[i * 3 + 1]!
    const zrel = xyz[i * 3 + 2]! - (params.floorA + params.floorB * y)
    if (Math.abs(x - params.axisX) >= half) continue
    if (!(zrel > params.floorBand) || !(zrel < trainTop)) continue
    insideMask[i] = 1
    inside++
  }
  if (inside === 0) return { ys, blocked }

  const yIn = new Float64Array(inside)
  const zIn = new Float64Array(inside)
  let k = 0
  for (let i = 0; i < points; i++) {
    if (!insideMask[i]) continue
    yIn[k] = xyz[i * 3 + 1]!
    zIn[k] = xyz[i * 3 + 2]! - (params.floorA + params.floorB * xyz[i * 3 + 1]!)
    k++
  }

  const counts = histogramCounts(yIn, edges)
  const spans = minHeight > 0.0 ? binSpan(zIn, yIn, edges) : null
  for (let b = 0; b < n; b++) {
    let busy = counts[b]! >= minPoints
    if (busy && spans !== null) busy = spans[b]! >= minHeight
    blocked[b] = busy ? 1 : 0
  }
  return { ys, blocked }
}

export interface FreeCorridorOptions extends CorridorOptions {
  axisStep?: number
  binX?: number
}

/**
 * Для каждого кандидата оси -- сколько Y-бинов свободно: (оси, свободно, всего).
 *
 * Поезд не может проехать сквозь вертикальное препятствие, поэтому ось пути
 * ВЫБИРАЕТСЯ по максимуму свободной длины коридора. Середина между стенами --
 * неверная эвристика: на `doubleT_obstacle` она ставит рельс прямо в столб.
 *
 * Двумерная гистограмма (Y, X) + кумулятивная сумма по X дают окно коридора
 * за O(1) на кандидата, поэтому весь перебор -- векторный.
 */
export function freeCorridorProfile(
  xyz: Float32Array,
  params: ZoneParams,
  yFrom: number,
  yTo: number,
  opts: FreeCorridorOptions = {},
): { axes: Float64Array; free: Int32Array; nBins: number } {
  const binWidth = opts.binWidth ?? 1.0
  const clearance = opts.clearance ?? 0.3
  const trainTop = opts.trainTop ?? TRAIN_TOP
  const minPoints = Math.max(Math.trunc(opts.minPoints ?? 20), 1)
  const axisStep = opts.axisStep ?? 0.05
  const binX = opts.binX ?? 0.05

  const ys = yNodes(yFrom, yTo, binWidth)
  const nBins = ys.length
  if (nBins === 0) return { axes: new Float64Array(0), free: new Int32Array(0), nBins: 0 }

  const half = params.gauge / 2.0 + clearance
  const points = xyz.length / 3

  // Кандидаты -- только ВНУТРИ тоннеля. Вне стен "свободно" по определению, и
  // поиск максимума без этого ограничения уводит ось за стену (замерено: 8.75 м
  // при стенах -2.65 и +6.55). Ограничение берём по найденным стенам.
  let innerLo: number
  let innerHi: number
  if (params.wallX.length >= 2) {
    innerLo = Math.min(...params.wallX) + half
    innerHi = Math.max(...params.wallX) - half
  } else if (points > 0) {
    let minX = Infinity
    let maxX = -Infinity
    for (let i = 0; i < points; i++) {
      const x = xyz[i * 3]!
      if (x < minX) minX = x
      if (x > maxX) maxX = x
    }
    innerLo = minX + half
    innerHi = maxX - half
  } else {
    innerLo = params.axisX - 1.0
    innerHi = params.axisX + 1.0
  }
  if (!(innerHi > innerLo)) {
    const mid = 0.5 * (innerLo + innerHi)
    innerLo = mid - 0.5
    innerHi = mid + 0.5
  }
  const axes = arange(innerLo, innerHi + 1e-9, axisStep)

  const allFree = (): Int32Array => new Int32Array(axes.length).fill(nBins)
  if (points === 0) return { axes, free: allFree(), nBins }

  let inside = 0
  const insideMask = new Uint8Array(points)
  for (let i = 0; i < points; i++) {
    const y = xyz[i * 3 + 1]!
    const zrel = xyz[i * 3 + 2]! - (params.floorA + params.floorB * y)
    if (!(zrel > params.floorBand) || !(zrel < trainTop)) continue
    if (!(y >= yFrom) || !(y < yTo)) continue
    insideMask[i] = 1
    inside++
  }
  if (inside === 0) return { axes, free: allFree(), nBins }

  const yIn = new Float64Array(inside)
  const xIn = new Float64Array(inside)
  let k = 0
  for (let i = 0; i < points; i++) {
    if (!insideMask[i]) continue
    yIn[k] = xyz[i * 3 + 1]!
    xIn[k] = xyz[i * 3]!
    k++
  }

  const yEdges = new Float64Array(nBins + 1)
  for (let i = 0; i < nBins; i++) yEdges[i] = ys[i]! - binWidth / 2.0
  yEdges[nBins] = ys[nBins - 1]! + binWidth / 2.0
  const xEdges = arange(Math.floor((innerLo - half) / binX) * binX, innerHi + half + binX, binX)

  const nx = xEdges.length - 1
  const hist = histogram2dCounts(yIn, xIn, yEdges, xEdges)
  if (nx <= 0) return { axes, free: allFree(), nBins }

  // csum[iy, ix] -- число точек в бинах X < ix; столбец 0 нулевой.
  const ncols = nx + 1
  const csum = new Float64Array(nBins * ncols)
  for (let iy = 0; iy < nBins; iy++) {
    let acc = 0.0
    const row = iy * ncols
    for (let ix = 0; ix < nx; ix++) {
      acc += hist[iy * nx + ix]!
      csum[row + ix + 1] = acc
    }
  }

  const free = new Int32Array(axes.length)
  for (let a = 0; a < axes.length; a++) {
    const axis = axes[a]!
    let lo = searchSortedLeft(xEdges, axis - half)
    if (lo < 0) lo = 0
    else if (lo >= ncols) lo = ncols - 1
    let hi = searchSortedRight(xEdges, axis + half)
    if (hi < 0) hi = 0
    else if (hi >= ncols) hi = ncols - 1
    let count = 0
    for (let iy = 0; iy < nBins; iy++) {
      const window = csum[iy * ncols + hi]! - csum[iy * ncols + lo]!
      if (!(window >= minPoints)) count++
    }
    free[a] = count
  }
  return { axes, free, nBins }
}

/**
 * Ось пути = максимум свободной длины коридора.
 *
 * При равенстве берём ЦЕНТР самого длинного непрерывного плато максимума: край
 * свободной зоны хуже, потому что одно небольшое препятствие сдвинет его в столб.
 */
export function findAxis(
  xyz: Float32Array,
  params: ZoneParams,
  yFrom: number,
  yTo: number,
  opts: FreeCorridorOptions = {},
): number {
  const { axes, free, nBins } = freeCorridorProfile(xyz, params, yFrom, yTo, opts)
  if (axes.length === 0 || nBins === 0) return params.axisX

  let min = Infinity
  let best = 0
  for (let i = 0; i < free.length; i++) {
    const v = free[i]!
    if (v < min) min = v
    if (v > best) best = v
  }
  // Нет ни одного препятствия -- ни одна ось не лучше другой, выбирать не из
  // чего: уважаем заданную, а не подставляем середину стен.
  if (min === nBins) return params.axisX
  if (best === 0) return params.axisX

  let runStart = -1
  let bestStart = 0
  let bestEnd = 0
  let bestLength = -1
  for (let i = 0; i <= free.length; i++) {
    const ok = i < free.length && free[i]! === best
    if (ok && runStart < 0) {
      runStart = i
    } else if (!ok && runStart >= 0) {
      const length = i - 1 - runStart
      if (length > bestLength) {
        bestLength = length
        bestStart = runStart
        bestEnd = i - 1
      }
      runStart = -1
    }
  }
  return axes[(bestStart + bestEnd) >> 1]!
}

/** Склеивает соседние заблокированные бины в диапазоны Y: [[y0, y1], ...]. */
export function blockedSpans(
  ys: ArrayLike<number>,
  blocked: ArrayLike<number>,
  binWidth?: number,
): Array<[number, number]> {
  if (ys.length === 0) return []
  let any = false
  for (let i = 0; i < blocked.length; i++) {
    if (blocked[i]!) {
      any = true
      break
    }
  }
  if (!any) return []

  const width = binWidth ?? (ys.length > 1 ? ys[1]! - ys[0]! : 1.0)
  const spans: Array<[number, number]> = []
  let start: number | null = null
  for (let i = 0; i < blocked.length; i++) {
    if (blocked[i]! && start === null) {
      start = ys[i]! - width / 2.0
    } else if (!blocked[i]! && start !== null) {
      spans.push([start, ys[i - 1]! + width / 2.0])
      start = null
    }
  }
  if (start !== null) spans.push([start, ys[ys.length - 1]! + width / 2.0])
  return spans
}

export interface RailLinesOptions {
  yFrom?: number
  yTo?: number
  step?: number
  blocked?: ArrayLike<number> | null
}

/** Полилинии рельсов, как `o3d.geometry.LineSet` в Python (points/lines/colors). */
export interface RailLines {
  points: Float64Array
  lines: Int32Array
  colors: Float64Array
}

/**
 * Полилинии ВСЕХ рельсов (ходовые + контактный) поверх наклонного пола.
 *
 * Число линий берётся из `railXPositions`, поэтому подпись совместима со старым
 * случаем двух ходовых рельсов: при `contactRail = false` получаются ровно две
 * линии. Контактный рельс рисуется на `contactHeight` -- он стоит на изоляторах
 * и не совпадает по высоте с ходовыми, поэтому высота берётся для каждого своя.
 *
 * Если `blocked` не задан -- длинные линии цветом рельса (профиль пола линеен,
 * поэтому геометрия та же). Если задан (маска на узлах Y) -- линии рвутся по
 * шагам, и шаг, у которого занят ЛЮБОЙ из концов, красится BLOCKED_COLOR:
 * рельс визуально упирается в препятствие, а не проходит сквозь него.
 *
 * В Python это LineSet из open3d; здесь те же точки/индексы/цвета массивами.
 */
export function makeRailLines(params: ZoneParams, opts: RailLinesOptions = {}): RailLines {
  const yFrom = opts.yFrom ?? -80.0
  const yTo = opts.yTo ?? 0.0
  const step = opts.step ?? 1.0
  const blocked = opts.blocked ?? null

  const ys = yNodes(yFrom, yTo, step)
  const n = ys.length
  const rails = railXPositions(params)
  const m = rails.length
  const zFloor = floorProfile(ys, params) as Float64Array

  const points = new Float64Array(m * n * 3)
  for (let r = 0; r < m; r++) {
    const railX = rails[r]![0]
    const height = rails[r]![1]
    for (let i = 0; i < n; i++) {
      const at = (r * n + i) * 3
      points[at] = railX
      points[at + 1] = ys[i]!
      points[at + 2] = zFloor[i]! + height
    }
  }

  const railColor = ZONE_COLORS.rail
  if (blocked === null) {
    const lines = new Int32Array(m * 2)
    const colors = new Float64Array(m * 3)
    for (let r = 0; r < m; r++) {
      lines[r * 2] = r * n
      lines[r * 2 + 1] = r * n + (n - 1)
      colors[r * 3] = railColor[0]
      colors[r * 3 + 1] = railColor[1]
      colors[r * 3 + 2] = railColor[2]
    }
    return { points, lines, colors }
  }

  if (blocked.length !== n) {
    throw new RangeError(`blocked must have ${n} entries, got ${blocked.length}`)
  }
  const perRail = n - 1
  const lines = new Int32Array(m * perRail * 2)
  const colors = new Float64Array(m * perRail * 3)
  for (let r = 0; r < m; r++) {
    for (let i = 0; i < perRail; i++) {
      const seg = r * perRail + i
      lines[seg * 2] = r * n + i
      lines[seg * 2 + 1] = r * n + i + 1
      const stop = blocked[i]! || blocked[i + 1]!
      colors[seg * 3] = stop ? BLOCKED_COLOR[0] : railColor[0]
      colors[seg * 3 + 1] = stop ? BLOCKED_COLOR[1] : railColor[1]
      colors[seg * 3 + 2] = stop ? BLOCKED_COLOR[2] : railColor[2]
    }
  }
  return { points, lines, colors }
}