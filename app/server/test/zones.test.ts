// Порт tests/test_zones.py на bun:test: пол, стены, рельсы, зоны, засорённость
// коридора, выбор оси. Эталон -- Python-реализация, числа должны совпадать.

import { describe, expect, test } from 'bun:test'

import { arange, linspace } from '../src/data/numeric'
import {
  BLOCKED_COLOR,
  MIN_OBSTACLE_HEIGHT,
  RAIL_HEIGHT,
  TRAIN_TOP,
  ZONE_COLORS,
  ZONE_NAMES,
  binSpan,
  blockedSpans,
  classify,
  classifyCounts,
  colorsFromLabels,
  corridorBlocked,
  detectWalls,
  findAxis,
  fitFloor,
  floorProfile,
  freeCorridorProfile,
  makeRailLines,
  railXPositions,
  yNodes,
  zoneColors,
  zoneParams,
} from '../src/data/zones'
import type { ZoneName, ZoneParams } from '../src/data/types'

/** Одна изолированная точка как (1, 3) float32 -- как кадр из bag-а. */
function probe(x: number, y: number, z: number): Float32Array {
  return Float32Array.from([x, y, z])
}

/** Последовательность из arange-совместимого диапазона (как np.arange). */
function seq(start: number, stop: number, step: number): Float64Array {
  return arange(start, stop, step)
}

function paramsWithWalls(): ZoneParams {
  return zoneParams({ wallX: [6.6] })
}

/** По одному однозначному зонду на зону (y = 0, значит floor_profile(0) = floorA). */
function zoneCases(p: ZoneParams): Record<ZoneName, [number, number, number]> {
  const floor0 = p.floorA
  return {
    rail: [2.36, 0.0, floor0 + 0.16],
    bed: [5.0, 0.0, floor0 + 0.1],
    ceiling: [5.0, 0.0, 2.5],
    wall: [6.6, 0.0, floor0 + 1.0],
    middle: [5.0, 0.0, floor0 + 1.0],
  }
}

/** Детерминированный ГПСЧ вместо np.random.default_rng: тесты проверяют
 *  свойства (две стены, пол с уклоном), а не конкретные значения эталона. */
function makeRng(seed: number): {
  random(): number
  uniform(lo: number, hi: number): number
  normal(mean: number, sigma: number): number
} {
  let state = seed >>> 0
  const random = (): number => {
    state = (state + 0x6d2b79f5) >>> 0
    let t = state
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
  return {
    random,
    uniform: (lo, hi) => lo + (hi - lo) * random(),
    normal: (mean, sigma) => {
      const u1 = Math.max(random(), 1e-12)
      const u2 = random()
      return mean + sigma * Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2)
    },
  }
}

function cloudFrom(points: Array<[number, number, number]>): Float32Array {
  const xyz = new Float32Array(points.length * 3)
  points.forEach((point, i) => {
    xyz[i * 3] = point[0]
    xyz[i * 3 + 1] = point[1]
    xyz[i * 3 + 2] = point[2]
  })
  return xyz
}

describe('FitFloor', () => {
  test('recovers synthetic slope', () => {
    const rng = makeRng(0)
    const n = 60000
    const xyz = new Float32Array(n * 3)
    for (let i = 0; i < n; i++) {
      const y = rng.uniform(-60.0, -2.0)
      const z = 1.0 - 0.03 * y + rng.uniform(-0.02, 0.02)
      xyz[i * 3] = 0.0
      xyz[i * 3 + 1] = y
      xyz[i * 3 + 2] = z
    }
    const { a, b, rms } = fitFloor(xyz)
    expect(Math.abs(a - 1.0)).toBeLessThanOrEqual(0.06)
    expect(Math.abs(b + 0.03)).toBeLessThanOrEqual(0.005)
    expect(rms).toBeLessThan(0.03)
  })

  test('fewer than two bins falls back to flat', () => {
    const xyz = Float32Array.from([0.0, -300.0, -1.0, 0.0, -301.0, -0.8])
    const { a, b, rms } = fitFloor(xyz, { yMin: -60.0, yMax: -2.0 })
    expect(Math.abs(a - -0.9)).toBeLessThanOrEqual(1e-6)
    expect(b).toBe(0.0)
    expect(rms).toBe(0.0)
  })
})

describe('FloorProfile', () => {
  test('matches a plus b y', () => {
    const p = zoneParams({ floorA: 1.0, floorB: -0.03 })
    const y = Float64Array.from([-60.0, -30.0, -2.0, 0.0])
    const got = floorProfile(y, p)
    expect(Array.from(got)).toEqual(Array.from(y, (value) => 1.0 - 0.03 * value))
  })
})

describe('Classify', () => {
  const p = paramsWithWalls()
  const label = (x: number, y: number, z: number): number => classify(probe(x, y, z), p)[0]!

  test('one probe per zone', () => {
    const cases = zoneCases(p)
    for (const [name, [x, y, z]] of Object.entries(cases)) {
      expect(ZONE_NAMES[label(x, y, z)]).toBe(name as ZoneName)
    }
  })

  test('rail beats bed', () => {
    const x = p.axisX + p.gauge / 2.0
    const z = p.floorA + 0.16
    expect(ZONE_NAMES[label(x, 0.0, z)]).toBe('rail')
  })

  test('just above floor is bed not middle', () => {
    const z = p.floorA + 0.29
    expect(ZONE_NAMES[label(5.0, 0.0, z)]).toBe('bed')
  })

  test('high point is ceiling', () => {
    expect(ZONE_NAMES[label(5.0, 0.0, 2.5)]).toBe('ceiling')
  })

  test('wall point', () => {
    const z = p.floorA + 1.0
    expect(ZONE_NAMES[label(6.6, 0.0, z)]).toBe('wall')
  })

  test('middle point', () => {
    const z = p.floorA + 1.0
    expect(ZONE_NAMES[label(5.0, 0.0, z)]).toBe('middle')
  })

  test('empty returns empty uint8', () => {
    const labels = classify(new Float32Array(0), p)
    expect(labels.length).toBe(0)
  })
})

describe('Colors', () => {
  test('five colors pairwise distinct', () => {
    const table = ZONE_NAMES.map((name) => ZONE_COLORS[name].join(','))
    expect(new Set(table).size).toBe(5)
  })

  test('zone colors reproduces table', () => {
    const p = paramsWithWalls()
    const cases = zoneCases(p)
    const names = Object.keys(cases) as ZoneName[]
    const xyz = cloudFrom(names.map((name) => cases[name]))
    const colors = zoneColors(xyz, p)
    expect(colors.length).toBe(names.length * 3)
    names.forEach((name, i) => {
      expect(Array.from(colors.slice(i * 3, i * 3 + 3))).toEqual(ZONE_COLORS[name])
    })
  })

  test('empty shape and dtype', () => {
    const colors = zoneColors(new Float32Array(0), paramsWithWalls())
    expect(colors.length).toBe(0)
  })
})

describe('Counts', () => {
  test('counts sum and keys', () => {
    const p = paramsWithWalls()
    const cases = zoneCases(p)
    const points = Object.values(cases).map((value) => value as [number, number, number])
    points.push([5.0, 0.0, p.floorA + 1.1])
    const xyz = cloudFrom(points)
    const counts = classifyCounts(xyz, p)
    expect(new Set(Object.keys(counts))).toEqual(new Set(ZONE_NAMES))
    expect(Object.values(counts).reduce((sum, value) => sum + value, 0)).toBe(points.length)
    for (const value of Object.values(counts)) expect(Number.isInteger(value)).toBe(true)
  })

  test('empty counts are all zero', () => {
    const counts = classifyCounts(new Float32Array(0), paramsWithWalls())
    expect(new Set(Object.keys(counts))).toEqual(new Set(ZONE_NAMES))
    expect(new Set(Object.values(counts))).toEqual(new Set([0]))
  })
})

describe('DetectWalls', () => {
  test('two synthetic walls', () => {
    const p = zoneParams()
    const rng = makeRng(1)
    const n = 40000
    const xyz = new Float32Array(n * 3)
    for (let i = 0; i < n; i++) {
      const y = rng.uniform(-50.0, -5.0)
      const zrel = rng.uniform(0.5, 2.0)
      const x = (rng.random() < 0.5 ? -2.7 : 6.6) + rng.normal(0.0, 0.03)
      xyz[i * 3] = x
      xyz[i * 3 + 1] = y
      xyz[i * 3 + 2] = p.floorA + p.floorB * y + zrel
    }
    const walls = detectWalls(xyz, p)
    expect(walls.length).toBe(2)
    expect(Math.abs(walls[0]! - -2.7)).toBeLessThanOrEqual(0.3)
    expect(Math.abs(walls[1]! - 6.6)).toBeLessThanOrEqual(0.3)
  })

  test('no points above bed returns empty', () => {
    const p = zoneParams()
    const y = linspace(-30.0, -5.0, 1000)
    const points: Array<[number, number, number]> = []
    for (const yy of y) points.push([3.0, yy, p.floorA + p.floorB * yy + 0.05])
    expect(detectWalls(cloudFrom(points), p)).toEqual([])
  })
})

describe('RailXPositions', () => {
  test('two running rails when contact disabled', () => {
    const p = zoneParams({ contactRail: false })
    const rails = railXPositions(p)
    expect(rails.length).toBe(2)
    const half = p.gauge / 2.0
    expect(rails[0]![0]).toBeCloseTo(p.axisX - half, 6)
    expect(rails[1]![0]).toBeCloseTo(p.axisX + half, 6)
    for (const rail of rails) expect(rail[1]).toBeCloseTo(RAIL_HEIGHT, 12)
  })

  test('three rails with contact', () => {
    const p = zoneParams({ contactRail: true, contactSide: 1 })
    const rails = railXPositions(p)
    expect(rails.length).toBe(3)
    const half = p.gauge / 2.0
    expect(rails[0]![0]).toBeCloseTo(p.axisX - half, 6)
    expect(rails[1]![0]).toBeCloseTo(p.axisX + half, 6)
    expect(rails[2]![0]).toBeCloseTo(p.axisX + p.contactOffset, 6)
    expect(rails[2]![1]).toBeCloseTo(p.contactHeight, 12)
  })
})

describe('ContactRailClassify', () => {
  // Контактный рельс не должен сливаться с ходовым: при contactOffset по
  // умолчанию (0.70) он геометрически попадает в полосу ходового (0.76 +- 0.12),
  // поэтому для изоляции признака берём вынос за пределы этой полосы.
  const params = (overrides: Partial<ZoneParams> = {}): ZoneParams =>
    zoneParams({ wallX: [6.6], contactOffset: 1.3, ...overrides })

  test('contact point is rail', () => {
    const p = params()
    const x = p.axisX + p.contactOffset
    const z = p.floorA + p.contactHeight
    expect(ZONE_NAMES[classify(probe(x, 0.0, z), p)[0]!]).toBe('rail')
  })

  test('contact point is bed when disabled', () => {
    const p = params({ contactRail: false })
    const x = p.axisX + p.contactOffset
    const z = p.floorA + p.contactHeight // невязка 0.20 < floor_band 0.30
    expect(ZONE_NAMES[classify(probe(x, 0.0, z), p)[0]!]).toBe('bed')
  })

  test('negative side mirrors contact rail', () => {
    const p = params({ contactSide: -1 })
    const rails = railXPositions(p)
    expect(rails[2]![0]).toBeCloseTo(p.axisX - p.contactOffset, 6)
    const x = p.axisX - p.contactOffset
    const z = p.floorA + p.contactHeight
    expect(ZONE_NAMES[classify(probe(x, 0.0, z), p)[0]!]).toBe('rail')
  })
})

describe('RailLines', () => {
  test('two rails geometry and color', () => {
    const p = zoneParams({ contactRail: false })
    const { points, lines, colors } = makeRailLines(p, { yFrom: -80.0, yTo: 0.0, step: 1.0 })
    expect(lines.length).toBe(2 * 2) // ровно две линии

    const n = points.length / 3 / 2
    expect(points.length).toBe(2 * n * 3)
    expect(points[0 * 3 + 1]).toBeCloseTo(-80.0, 12)
    expect(points[(n - 1) * 3 + 1]).toBeCloseTo(0.0, 12)

    const leftX = p.axisX - p.gauge / 2.0
    const rightX = p.axisX + p.gauge / 2.0
    for (let i = 0; i < n; i++) {
      expect(points[i * 3]).toBeCloseTo(leftX, 6)
      expect(points[(n + i) * 3]).toBeCloseTo(rightX, 6)
    }

    for (let block = 0; block < 2; block++) {
      for (let i = 0; i < n; i++) {
        const at = (block * n + i) * 3
        const expectedZ = p.floorA + p.floorB * points[at + 1]! + 0.16
        expect(points[at + 2]).toBeCloseTo(expectedZ, 12)
      }
    }

    expect(colors.length).toBe(2 * 3)
    for (let i = 0; i < colors.length; i += 3) {
      expect(Array.from(colors.slice(i, i + 3))).toEqual(ZONE_COLORS.rail)
    }
  })

  test('three rails geometry includes contact', () => {
    const p = zoneParams()
    const { points } = makeRailLines(p, { yFrom: -20.0, yTo: 0.0, step: 2.0 })
    const n = yNodes(-20.0, 0.0, 2.0).length
    const rails = railXPositions(p)
    expect(points.length / 3).toBe(rails.length * n)

    const xs = new Set<number>()
    for (let i = 0; i < points.length / 3; i++) {
      xs.add(Math.round(points[i * 3]! * 1e6) / 1e6)
    }
    expect(xs.size).toBe(3) // три различных положения рельсов по X

    for (let i = 0; i < n; i++) {
      const at = (2 * n + i) * 3
      expect(points[at]).toBeCloseTo(p.axisX + p.contactOffset, 6)
      expect(points[at + 2]).toBeCloseTo(
        p.floorA + p.floorB * points[at + 1]! + p.contactHeight,
        12,
      )
    }
  })
})

describe('ColorsFromLabels', () => {
  test('matches zone colors and table', () => {
    const points: Array<[number, number, number]> = [
      [1.6, -10.0, -1.7], // около рельсов / полосы ложа
      [1.6, -10.0, 2.5], // высоко -> потолок
      [-2.7, -10.0, 1.0], // стена, если wall_x задан
      [4.0, -20.0, 0.5], // высоко над полом, ниже потолка
    ]
    const params = zoneParams({ floorA: -1.886, floorB: 0.021585, wallX: [-2.7, 6.6] })
    const xyz = cloudFrom(points)
    const labels = classify(xyz, params)
    const viaLabels = colorsFromLabels(labels)
    const viaXyz = zoneColors(xyz, params)
    expect(Array.from(viaLabels)).toEqual(Array.from(viaXyz))
    for (let i = 0; i < labels.length; i++) {
      const expected = ZONE_COLORS[ZONE_NAMES[labels[i]!]!]
      expect(Array.from(viaLabels.slice(i * 3, i * 3 + 3))).toEqual(expected)
    }
  })

  test('empty', () => {
    const out = colorsFromLabels(new Uint8Array(0))
    expect(out.length).toBe(0)
  })
})

describe('CorridorBlocked', () => {
  const params = (overrides: Partial<ZoneParams> = {}): ZoneParams =>
    zoneParams({
      floorA: -1.886,
      floorB: 0.021585,
      axisX: 1.6,
      gauge: 1.52,
      wallX: [-2.7, 6.6],
      ...overrides,
    })

  /** Плоское ложе вдоль y в (-40, 0) плюс необязательная вертикальная структура. */
  const cloud = (obstacleYRange: [number, number] | null = null): Float32Array => {
    const points: Array<[number, number, number]> = []
    for (const y of seq(-40.0, 0.0, 0.25)) {
      const floor = -1.886 + 0.021585 * y
      for (const x of seq(0.9, 2.3, 0.1)) points.push([x, y, floor + 0.05]) // точки ниже floor_band
    }
    if (obstacleYRange !== null) {
      for (const y of seq(obstacleYRange[0], obstacleYRange[1], 0.25)) {
        const floor = -1.886 + 0.021585 * y
        for (const x of seq(1.0, 2.2, 0.05)) {
          for (const zz of seq(0.4, 2.5, 0.2)) points.push([x, y, floor + zz])
        }
      }
    }
    return cloudFrom(points)
  }

  test('bed alone is not blocked', () => {
    const p = params()
    const { ys, blocked } = corridorBlocked(cloud(), p, -40.0, 0.0, { binWidth: 1.0 })
    expect(blocked.length).toBe(ys.length)
    expect(blocked.some((value) => value !== 0)).toBe(false) // ложе ниже floor_band
  })

  test('structure blocks only its own y', () => {
    const p = params()
    const { ys, blocked } = corridorBlocked(cloud([-16.0, -14.0]), p, -40.0, 0.0, {
      binWidth: 1.0,
      minPoints: 20,
    })
    expect(blocked.some((value) => value !== 0)).toBe(true)
    const nearest = (target: number): number => {
      let best = 0
      for (let i = 1; i < ys.length; i++) {
        if (Math.abs(ys[i]! - target) < Math.abs(ys[best]! - target)) best = i
      }
      return best
    }
    expect(blocked[nearest(-15.0)]).toBe(1) // y ~ -15 обязан быть занят
    expect(blocked[nearest(-30.0)]).toBe(0) // y ~ -30 свободен
  })

  test('axis moved away clears the corridor', () => {
    const p = params({ axisX: 4.0 })
    const { blocked } = corridorBlocked(cloud([-16.0, -14.0]), p, -40.0, 0.0, {
      binWidth: 1.0,
      minPoints: 20,
    })
    expect(blocked.some((value) => value !== 0)).toBe(false)
  })

  test('empty cloud is all clear', () => {
    const p = params()
    const { ys, blocked } = corridorBlocked(new Float32Array(0), p, -40.0, 0.0)
    expect(blocked.length).toBe(ys.length)
    expect(blocked.some((value) => value !== 0)).toBe(false)
  })

  test('spans merge consecutive bins', () => {
    const ys = arange(-9.0, 1.0, 1.0)
    const blocked = Uint8Array.from([0, 1, 1, 0, 1, 0, 0, 0, 0, 0])
    const spans = blockedSpans(ys, blocked)
    expect(spans.length).toBe(2)
    expect(spans[0]![0]).toBeCloseTo(-8.5, 12)
    expect(spans[0]![1]).toBeCloseTo(-6.5, 12)
    expect(spans[1]![0]).toBeCloseTo(-5.5, 12)
    expect(spans[1]![1]).toBeCloseTo(-4.5, 12)
  })

  test('spans empty when nothing blocked', () => {
    const ys = arange(-5.0, 1.0, 1.0)
    expect(blockedSpans(ys, new Uint8Array(ys.length))).toEqual([])
  })

  /** Тонкая поверхность вдоль края коридора: 5 см по высоте, много точек.
   *  Так выглядит кромка балласта, кабельный лоток или стенка рельса --
   *  замерено на doubleT_obstacle: 0.0-0.19 м по высоте. */
  const thinSurfaceCloud = (yRange: [number, number] = [-16.0, -14.0]): Float32Array => {
    const points: Array<[number, number, number]> = []
    for (const y of seq(yRange[0], yRange[1], 0.25)) {
      const floor = -1.886 + 0.021585 * y
      for (const x of seq(1.0, 2.2, 0.05)) {
        for (const zz of [0.4, 0.42, 0.45]) points.push([x, y, floor + zz])
      }
    }
    return cloudFrom(points)
  }

  test('thin surface does not block', () => {
    const p = params()
    const surface = thinSurfaceCloud()
    const byCount = corridorBlocked(surface, p, -40.0, 0.0, {
      binWidth: 1.0,
      minPoints: 20,
      minHeight: 0.0,
    }).blocked
    expect(byCount.some((value) => value !== 0)).toBe(true) // без требования высоты занято

    const byHeight = corridorBlocked(surface, p, -40.0, 0.0, {
      binWidth: 1.0,
      minPoints: 20,
    }).blocked
    expect(byHeight.some((value) => value !== 0)).toBe(false) // тонкая поверхность не перекрывает путь
  })

  test('tall structure still blocks', () => {
    const p = params()
    const { ys, blocked } = corridorBlocked(cloud([-16.0, -14.0]), p, -40.0, 0.0, {
      binWidth: 1.0,
      minPoints: 20,
      minHeight: 0.5,
    })
    let nearest = 0
    for (let i = 1; i < ys.length; i++) {
      if (Math.abs(ys[i]! + 15.0) < Math.abs(ys[nearest]! + 15.0)) nearest = i
    }
    expect(blocked[nearest]).toBe(1) // объект высотой 2 м обязан остаться препятствием
  })

  test('bin span separates thin from tall', () => {
    const edges = Float64Array.from([-1.5, -0.5, 0.5, 1.5])
    const thin = [0.4, 0.42, 0.45, 0.41, 0.43, 0.44]
    const tall = [0.4, 0.9, 1.4, 1.9, 2.4, 2.5]
    const values = Float64Array.from([...thin, ...tall])
    const ys = Float64Array.from([...thin.map(() => -1.0), ...tall.map(() => 0.0)])
    const spans = binSpan(values, ys, edges)
    expect(spans[0]!).toBeLessThan(0.1) // тонкая поверхность: размах в сантиметрах
    expect(spans[1]!).toBeGreaterThan(1.0) // высокий объект: размах в метрах
    expect(spans[2]!).toBe(0.0) // пустой бин: размах 0
    expect(binSpan(new Float64Array(0), new Float64Array(0), edges).length).toBe(3)
  })
})

describe('RailBlockedRendering', () => {
  const PARAMS = zoneParams({ axisX: 1.6, gauge: 1.52 })

  test('render is split and uses stop colour', () => {
    const ys = yNodes(-10.0, 0.0, 1.0)
    const blocked = new Uint8Array(ys.length)
    let nearest = 0
    for (let i = 1; i < ys.length; i++) {
      if (Math.abs(ys[i]! + 5.0) < Math.abs(ys[nearest]! + 5.0)) nearest = i
    }
    blocked[nearest] = 1

    const { lines, colors } = makeRailLines(PARAMS, {
      yFrom: -10.0,
      yTo: 0.0,
      step: 1.0,
      blocked,
    })
    const rails = railXPositions(PARAMS)
    expect(rails.length).toBe(3)
    expect(lines.length / 2).toBe(rails.length * (ys.length - 1))
    expect(colors.length / 3).toBe(lines.length / 2)

    let stopRows = 0
    let railRows = 0
    for (let i = 0; i < colors.length; i += 3) {
      const row = [colors[i]!, colors[i + 1]!, colors[i + 2]!]
      if (row.every((value, c) => Math.abs(value - BLOCKED_COLOR[c]!) < 1e-12)) stopRows++
      if (row.every((value, c) => Math.abs(value - ZONE_COLORS.rail[c]!) < 1e-12)) railRows++
    }
    expect(stopRows).toBeGreaterThan(0) // заблокированный сегмент красится BLOCKED_COLOR
    expect(railRows).toBeGreaterThan(0)
  })

  test('mismatched blocked length raises', () => {
    expect(() =>
      makeRailLines(PARAMS, {
        yFrom: -10.0,
        yTo: 0.0,
        step: 1.0,
        blocked: new Uint8Array(3),
      }),
    ).toThrow()
  })

  test('rail line points lie on gauge', () => {
    const p = zoneParams({ axisX: 1.6, gauge: 1.52, contactRail: false })
    const { points } = makeRailLines(p, { yFrom: -20.0, yTo: 0.0, step: 2.0 })
    const xs = new Set<number>()
    for (let i = 0; i < points.length / 3; i++) {
      xs.add(Math.round(points[i * 3]! * 1e6) / 1e6)
    }
    expect(xs.size).toBe(2)
    const sorted = Array.from(xs).sort((a, b) => a - b)
    expect(sorted[0]!).toBeCloseTo(1.6 - 0.76, 6)
    expect(sorted[1]!).toBeCloseTo(1.6 + 0.76, 6)
    for (let i = 0; i < points.length / 3; i++) {
      const at = i * 3
      expect(points[at + 2]).toBeCloseTo(p.floorA + p.floorB * points[at + 1]! + RAIL_HEIGHT, 12)
    }
  })
})

describe('FindAxis', () => {
  const PARAMS = zoneParams({ floorA: -1.886, floorB: 0.021585, wallX: [-2.7, 6.6] })

  const cloudWithPillar = (
    pillarX: [number, number] = [1.3, 1.8],
    ySpan: [number, number] = [-30.0, -10.0],
  ): Float32Array => {
    const points: Array<[number, number, number]> = []
    for (const y of seq(-40.0, 0.0, 0.25)) {
      const floor = -1.886 + 0.021585 * y
      for (const x of seq(-2.0, 6.0, 0.1)) points.push([x, y, floor + 0.05]) // плоское ложе
    }
    for (const y of seq(ySpan[0], ySpan[1], 0.25)) {
      const floor = -1.886 + 0.021585 * y
      for (const x of seq(pillarX[0], pillarX[1], 0.05)) {
        for (const zz of seq(0.4, 2.5, 0.2)) points.push([x, y, floor + zz]) // столб
      }
    }
    return cloudFrom(points)
  }

  test('axis avoids the pillar', () => {
    const cloud = cloudWithPillar()
    const axis = findAxis(cloud, PARAMS, -40.0, 0.0, { binWidth: 1.0 })
    // gauge/2 + clearance = 0.76 + 0.30 = 1.06 -> коридор обязан обойти 1.30..1.80
    expect(axis + 1.06 <= 1.3 || axis - 1.06 >= 1.8).toBe(true)
  })

  test('wall midpoint would fail the constraint', () => {
    // Старая эвристика (середина между стенами) попадает внутрь занятого набора.
    const midpoint = 0.5 * (-2.7 + 6.6)
    const corridor: [number, number] = [midpoint - 1.06, midpoint + 1.06]
    expect(corridor[0] < 1.8 && corridor[1] > 1.3).toBe(true)
  })

  test('free axis has full free length', () => {
    const cloud = cloudWithPillar()
    const { axes, free, nBins } = freeCorridorProfile(cloud, PARAMS, -40.0, 0.0, {
      binWidth: 1.0,
    })
    const axis = findAxis(cloud, PARAMS, -40.0, 0.0, { binWidth: 1.0 })
    let idx = 0
    for (let i = 1; i < axes.length; i++) {
      if (Math.abs(axes[i]! - axis) < Math.abs(axes[idx]! - axis)) idx = i
    }
    expect(free[idx]).toBe(nBins) // выбранная ось свободна по всей длине
  })

  test('empty cloud returns current axis', () => {
    const axis = findAxis(new Float32Array(0), PARAMS, -40.0, 0.0)
    expect(axis).toBeCloseTo(PARAMS.axisX, 6)
  })

  test('returns centre of the free plateau', () => {
    const cloud = cloudWithPillar()
    const { axes, free } = freeCorridorProfile(cloud, PARAMS, -40.0, 0.0)
    const axis = findAxis(cloud, PARAMS, -40.0, 0.0)
    let idx = 0
    for (let i = 1; i < axes.length; i++) {
      if (Math.abs(axes[i]! - axis) < Math.abs(axes[idx]! - axis)) idx = i
    }
    // не должны сидеть на самом краю свободной зоны
    if (idx > 0 && idx < free.length - 1) {
      expect(free[idx - 1] === free[idx] || free[idx + 1] === free[idx]).toBe(true)
    }
  })
})

describe('Constants', () => {
  test('match python values', () => {
    expect(RAIL_HEIGHT).toBe(0.16)
    expect(TRAIN_TOP).toBe(3.7)
    expect(MIN_OBSTACLE_HEIGHT).toBe(0.5)
  })
})