// «Линия хода» и «Туннель безопасности» — геометрия и клиентская раскраска.
//
// Оси лидара: X — поперёк, Y — вдоль (вперёд это −Y), Z — вверх. Линия хода
// задана узлами (y, x, z), где z — уровень головки рельса (УГР). Из узлов
// строится таблица подстановки x(y), по ней же интерполируется высота УГР:
// её вычитание даёт высоту точки **над УГР**, в которой заданы границы туннеля.
//
// Здесь нет ни three.js, ни React: только массивы. Это позволяет проверять
// раскраску отдельно от сцены и держать пересчёт 340 тыс. точек в пределах
// бюджета (см. спеку: ≤20 мс).

import type { PathVariantJson } from '@/api/pathTunnel'
import {
  CONTACT_HALF_WIDTH,
  PATH_ARROW_LENGTH,
  PATH_ARROW_WIDTH,
  PATH_LUT_STEP,
  RAIL_HALF_WIDTH,
  RAIL_HIGH,
  RAIL_LOW,
  TUNNEL_SECTION_STEP,
} from '@/viewer/constants'

/** Линия хода с готовой таблицей подстановки для быстрой выборки x(y). */
export interface PathLine {
  /** Узлы, отсортированные по y по возрастанию. */
  y: Float32Array
  x: Float32Array
  z: Float32Array
  /** УГР выше плоскости пола, м (path.z_offset). */
  zOffset: number
  yMin: number
  yMax: number
  /** Равномерная таблица значений по y: x и УГР. */
  lutX: Float32Array
  lutZ: Float32Array
}

export interface PathLineOptions {
  zOffset: number
  /** Пол вьюера: нужен только как запасной источник z, если сервер его не дал. */
  floorA: number
  floorB: number
}

/** Значение узловой функции в точке y (y внутри диапазона узлов). */
function sampleNodes(nodeY: Float32Array, nodeV: Float32Array, y: number): number {
  const n = nodeY.length
  let lo = 0
  let hi = n - 1
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1
    if (nodeY[mid] <= y) lo = mid
    else hi = mid
  }
  const y0 = nodeY[lo]
  const y1 = nodeY[hi]
  if (y1 === y0) return nodeV[lo]
  const t = (y - y0) / (y1 - y0)
  return nodeV[lo] + (nodeV[hi] - nodeV[lo]) * t
}

/**
 * Сборка линии из блока варианта. null — узлов меньше двух или они битые:
 * панель покажет «нет данных», вьюер не падает.
 */
export function makePathLine(
  v: PathVariantJson | null,
  opts: PathLineOptions,
): PathLine | null {
  if (!v || v.unavailable) return null
  const n = Math.min(v.y.length, v.x.length)
  if (n < 2) return null
  const idx: number[] = []
  for (let i = 0; i < n; i++) {
    const y = v.y[i]
    const x = v.x[i]
    if (Number.isFinite(y) && Number.isFinite(x)) idx.push(i)
  }
  if (idx.length < 2) return null
  idx.sort((a, b) => v.y[a] - v.y[b])

  const m = idx.length
  const y = new Float32Array(m)
  const x = new Float32Array(m)
  const z = new Float32Array(m)
  const hasZ = !!v.z && v.z.length === v.y.length
  for (let i = 0; i < m; i++) {
    const k = idx[i]
    y[i] = v.y[k]
    x[i] = v.x[k]
    // Сервер отдаёт z на уровне УГР; если ключа нет — считаем сами по полу.
    z[i] = hasZ ? v.z![k] : opts.floorA + opts.floorB * v.y[k] + opts.zOffset
  }

  const line: PathLine = {
    y,
    x,
    z,
    zOffset: opts.zOffset,
    yMin: y[0],
    yMax: y[m - 1],
    lutX: new Float32Array(0),
    lutZ: new Float32Array(0),
  }
  const span = line.yMax - line.yMin
  const count = Math.max(2, Math.ceil(span / PATH_LUT_STEP) + 1)
  const lutX = new Float32Array(count)
  const lutZ = new Float32Array(count)
  for (let i = 0; i < count; i++) {
    const yi = i === count - 1 ? line.yMax : line.yMin + i * PATH_LUT_STEP
    lutX[i] = sampleNodes(y, x, yi)
    lutZ[i] = sampleNodes(y, z, yi)
  }
  line.lutX = lutX
  line.lutZ = lutZ
  return line
}

/** x и УГР линии в точке y; false — y вне узлов (точка не в туннеле). */
export function sampleLine(line: PathLine, y: number, out: [number, number]): boolean {
  const n = line.lutX.length
  const fy = (y - line.yMin) / PATH_LUT_STEP
  if (fy < 0 || fy > n - 1) return false
  const i0 = fy | 0
  if (i0 >= n - 1) {
    out[0] = line.lutX[n - 1]
    out[1] = line.lutZ[n - 1]
    return true
  }
  const t = fy - i0
  const k = i0 + 1
  out[0] = line.lutX[i0] + (line.lutX[k] - line.lutX[i0]) * t
  out[1] = line.lutZ[i0] + (line.lutZ[k] - line.lutZ[i0]) * t
  return true
}

// ------------------------------------------------------------------ туннель

/** Всё, что нужно, чтобы решить «точка внутри туннеля» без обращения к сцене. */
export interface TunnelMask {
  line: PathLine
  half: number
  /** Границы над УГР, м. */
  zLow: number
  zHigh: number
  /** Центры ходовых рельсов по X и их высоты относительно плоскости пола. */
  railXs: readonly number[]
  railZs: readonly number[]
  /** Контактный рельс: X и высота над плоскостью пола; null — не вычитается. */
  contactX: number | null
  contactZ: number
  /** Цвет точек внутри туннеля (обычно meta.blocked_color). */
  color: readonly [number, number, number]
}

/**
 * Красит точки внутри туннеля, возвращает их число.
 *
 * Высотные границы туннеля заданы над УГР, а полосы рельсов в zones.py — над
 * плоскостью пола; разница ровно `z_offset`, поэтому полосы пересчитываются
 * один раз здесь, а не на каждой точке.
 */
export function paintTunnel(
  positions: Float32Array,
  count: number,
  out: Uint8Array,
  mask: TunnelMask,
): number {
  const line = mask.line
  const lutX = line.lutX
  const lutZ = line.lutZ
  const lutN = lutX.length
  if (lutN < 2 || count <= 0) return 0

  const yMin = line.yMin
  const half = mask.half
  const zLow = mask.zLow
  const zHigh = mask.zHigh
  const [cr, cg, cb] = mask.color
  const zOff = line.zOffset

  const railXs = mask.railXs
  const railCount = Math.min(railXs.length, mask.railZs.length)
  const railLo = RAIL_LOW - zOff
  const railHi = RAIL_HIGH - zOff
  const contactX = mask.contactX
  const contactBase = mask.contactZ - zOff
  const contactLo = contactBase + RAIL_LOW
  const contactHi = contactBase + RAIL_HIGH

  let painted = 0
  for (let i = 0, j = 0; i < count; i++, j += 3) {
    const y = positions[j + 1]
    const fy = (y - yMin) / PATH_LUT_STEP
    if (fy < 0 || fy > lutN - 1) continue
    const i0 = fy | 0
    let xp: number
    let zp: number
    if (i0 >= lutN - 1) {
      xp = lutX[lutN - 1]
      zp = lutZ[lutN - 1]
    } else {
      const t = fy - i0
      const k = i0 + 1
      xp = lutX[i0] + (lutX[k] - lutX[i0]) * t
      zp = lutZ[i0] + (lutZ[k] - lutZ[i0]) * t
    }
    const dx = positions[j] - xp
    if (dx > half || dx < -half) continue
    const above = positions[j + 2] - zp // высота над УГР
    if (above < zLow || above > zHigh) continue

    // Известное оборудование помехой не считается: ходовые рельсы всегда,
    // контактный — когда панель просит его исключить.
    const x = positions[j]
    let skip = false
    for (let r = 0; r < railCount; r++) {
      const rdx = x - railXs[r]
      if (rdx > RAIL_HALF_WIDTH || rdx < -RAIL_HALF_WIDTH) continue
      const rz = mask.railZs[r] - zOff
      if (above >= rz + railLo && above <= rz + railHi) {
        skip = true
        break
      }
    }
    if (
      !skip &&
      contactX !== null &&
      above >= contactLo &&
      above <= contactHi &&
      x >= contactX - CONTACT_HALF_WIDTH &&
      x <= contactX + CONTACT_HALF_WIDTH
    ) {
      skip = true
    }
    if (skip) continue

    out[j] = cr
    out[j + 1] = cg
    out[j + 2] = cb
    painted++
  }
  return painted
}

/** Каркас туннеля: рёбра прямоугольных сечений через `step` метров по пути. */
export function buildTunnelFrame(input: {
  line: PathLine
  half: number
  zLow: number
  zHigh: number
  step?: number
}): Float32Array {
  const { line, half, zLow, zHigh } = input
  const step = input.step && input.step > 0 ? input.step : TUNNEL_SECTION_STEP
  const scratch: [number, number] = [0, 0]
  const sections: number[] = [] // xc, y, zb, zt подряд
  for (let y = line.yMax; y >= line.yMin - 1e-6; y -= step) {
    if (!sampleLine(line, y, scratch)) continue
    sections.push(scratch[0], y, scratch[1] + zLow, scratch[1] + zHigh)
  }
  const ns = sections.length / 4
  if (ns === 0) return new Float32Array(0)

  const corner = (s: number, k: number): [number, number, number] => {
    const xc = sections[s * 4]
    const y = sections[s * 4 + 1]
    const zb = sections[s * 4 + 2]
    const zt = sections[s * 4 + 3]
    switch (k) {
      case 0:
        return [xc - half, y, zb]
      case 1:
        return [xc + half, y, zb]
      case 2:
        return [xc + half, y, zt]
      default:
        return [xc - half, y, zt]
    }
  }

  const pts: number[] = []
  for (let s = 0; s < ns; s++) {
    for (let k = 0; k < 4; k++) {
      const a = corner(s, k)
      const b = corner(s, (k + 1) % 4)
      pts.push(a[0], a[1], a[2], b[0], b[1], b[2])
    }
  }
  for (let s = 0; s + 1 < ns; s++) {
    for (let k = 0; k < 4; k++) {
      const a = corner(s, k)
      const b = corner(s + 1, k)
      pts.push(a[0], a[1], a[2], b[0], b[1], b[2])
    }
  }
  return new Float32Array(pts)
}

/**
 * Лента вдоль линии: толщину `LineBasicMaterial` в WebGL игнорирует, поэтому
 * линия рисуется полосой из двух треугольников на узел. Ширина откладывается
 * по горизонтали перпендикулярно направлению — лента повторяет изгибы пути.
 *
 * В том же буфере вершин лежит наконечник (плоский треугольник) на самом
 * дальнем узле — там, куда «едет» путь (вперёд это −Y, узлы отсортированы по
 * y по возрастанию, значит дальний — нулевой). Он ориентирован по последнему
 * сегменту и красится вместе с лентой: материал у меша один.
 */
export function buildRibbon(
  line: PathLine,
  thickness: number,
): { positions: Float32Array; indices: Uint32Array } | null {
  const n = line.y.length
  if (n < 2) return null
  const halfW = Math.max(thickness, 1e-3) / 2
  const positions = new Float32Array((n * 2 + 3) * 3)
  const indices = new Uint32Array((n - 1) * 6 + 3)
  for (let i = 0; i < n; i++) {
    const prev = i > 0 ? i - 1 : i
    const next = i + 1 < n ? i + 1 : i
    let tx = line.x[next] - line.x[prev]
    let ty = line.y[next] - line.y[prev]
    const len = Math.hypot(tx, ty)
    if (len < 1e-6) {
      tx = 1
      ty = 0
    } else {
      tx /= len
      ty /= len
    }
    // Нормаль к направлению в горизонтальной плоскости.
    const nx = -ty * halfW
    const ny = tx * halfW
    const o = i * 6
    positions[o] = line.x[i] + nx
    positions[o + 1] = line.y[i] + ny
    positions[o + 2] = line.z[i]
    positions[o + 3] = line.x[i] - nx
    positions[o + 4] = line.y[i] - ny
    positions[o + 5] = line.z[i]
  }
  for (let i = 0; i + 1 < n; i++) {
    const o = i * 6
    const l0 = i * 2
    const r0 = l0 + 1
    const l1 = l0 + 2
    const r1 = l0 + 3
    indices[o] = l0
    indices[o + 1] = r0
    indices[o + 2] = r1
    indices[o + 3] = l0
    indices[o + 4] = r1
    indices[o + 5] = l1
  }

  const tipX = line.x[0]
  const tipY = line.y[0]
  const tipZ = line.z[0]
  let dx = tipX - line.x[1]
  let dy = tipY - line.y[1]
  const segLen = Math.hypot(dx, dy)
  if (segLen < 1e-6) {
    dx = 0
    dy = -1
  } else {
    dx /= segLen
    dy /= segLen
  }
  const baseX = tipX - dx * PATH_ARROW_LENGTH
  const baseY = tipY - dy * PATH_ARROW_LENGTH
  const perpX = -dy * (PATH_ARROW_WIDTH / 2)
  const perpY = dx * (PATH_ARROW_WIDTH / 2)
  const ao = n * 2 * 3
  positions[ao] = tipX
  positions[ao + 1] = tipY
  positions[ao + 2] = tipZ
  positions[ao + 3] = baseX + perpX
  positions[ao + 4] = baseY + perpY
  positions[ao + 5] = tipZ
  positions[ao + 6] = baseX - perpX
  positions[ao + 7] = baseY - perpY
  positions[ao + 8] = tipZ
  const io = (n - 1) * 6
  indices[io] = n * 2
  indices[io + 1] = n * 2 + 1
  indices[io + 2] = n * 2 + 2

  return { positions, indices }
}