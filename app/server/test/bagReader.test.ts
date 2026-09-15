// Порт tests/test_core.py на bun:test: разбор CDR, кэш кадров, раскраски,
// обрезка по дальности. Эталон -- Python-реализация, числа должны совпадать.

import { describe, expect, test } from 'bun:test'
import { existsSync } from 'node:fs'
import { join } from 'node:path'

import {
  BagFrames,
  MODES,
  Z_RANGE,
  frameColors,
  parsePointcloud2Cdr,
  trim,
  trimRange,
} from '../src/data/bagReader'
import { percentileLinear } from '../src/data/numeric'

const ROOT = join(import.meta.dir, '..', '..', '..')
const BAG = join(ROOT, 'for_hackathon', 'doubleT_obstacle')
const DB = join(BAG, 'doubleT_obstacle_0.db3')
const hasBag = existsSync(DB)

/** Сборка минимального CDR PointCloud2 (little-endian, поля float32) -- как build_cdr. */
function buildCdr(
  points: Array<number[]>,
  fieldNames: readonly string[] = ['x', 'y', 'z', 'intensity'],
): Uint8Array {
  const chunks: Uint8Array[] = []
  const u32 = (value: number): Uint8Array => {
    const out = new Uint8Array(4)
    new DataView(out.buffer).setUint32(0, value, true)
    return out
  }
  const push = (bytes: Uint8Array): void => {
    chunks.push(bytes)
  }
  const size = (): number => chunks.reduce((sum, chunk) => sum + chunk.length, 0)
  const pad4 = (): void => {
    const rem = size() % 4
    if (rem !== 0) push(new Uint8Array(4 - rem))
  }

  push(new Uint8Array([0, 1, 0, 0]))
  push(u32(0))
  push(u32(0))
  const frameId = new TextEncoder().encode('lidar\0')
  push(u32(frameId.length))
  push(frameId)
  pad4()

  push(u32(1)) // height
  push(u32(points.length)) // width
  push(u32(fieldNames.length))
  fieldNames.forEach((name, i) => {
    const bytes = new Uint8Array([...new TextEncoder().encode(name), 0])
    push(u32(bytes.length))
    push(bytes)
    pad4()
    push(u32(i * 4))
    push(new Uint8Array([7])) // datatype = FLOAT32
    pad4()
    push(u32(1))
  })
  push(new Uint8Array([0])) // is_bigendian
  pad4()
  const step = 4 * fieldNames.length
  push(u32(step))
  push(u32(step * points.length))
  push(u32(step * points.length))
  const flat = new Float32Array(points.length * fieldNames.length)
  points.forEach((point, i) => flat.set(point, i * fieldNames.length))
  push(new Uint8Array(flat.buffer, flat.byteOffset, flat.byteLength))

  const total = size()
  const out = new Uint8Array(total)
  let at = 0
  for (const chunk of chunks) {
    out.set(chunk, at)
    at += chunk.length
  }
  return out
}

describe('Parser', () => {
  test('synthetic roundtrip', () => {
    const points = [
      [1.0, 2.0, 3.0, 10.0],
      [-4.5, -60.0, 0.5, 200.0],
      [0.0, 0.0, 0.0, 0.0],
    ]
    const { xyz, intensity } = parsePointcloud2Cdr(buildCdr(points))
    expect(xyz.length).toBe(2 * 3) // нулевую точку отбрасываем
    expect(Array.from(xyz.slice(0, 3))).toEqual([1.0, 2.0, 3.0])
    expect(Array.from(xyz.slice(3, 6))).toEqual([-4.5, -60.0, 0.5])
    expect(intensity).not.toBeNull()
    expect(Array.from(intensity!)).toEqual([10.0, 200.0])
  })

  test('synthetic without intensity field', () => {
    const points = [
      [5.0, -1.0, 0.0],
      [6.0, -2.0, 0.0],
    ]
    const { xyz, intensity } = parsePointcloud2Cdr(buildCdr(points, ['x', 'y', 'z']))
    expect(xyz.length).toBe(2 * 3)
    expect(intensity).toBeNull() // нет поля intensity -> null
  })

  test.skipIf(!hasBag)('real bag matches known facts', () => {
    const frames = new BagFrames(DB, { cacheSize: 1, prefetchAhead: 0 })
    const frame = frames.frame(0)
    frames.close()
    expect(frame.xyz.length / 3).toBe(346808)

    let minX = Infinity
    let maxX = -Infinity
    let minY = Infinity
    let minZ = Infinity
    let maxZ = -Infinity
    let maxIntensity = -Infinity
    for (let i = 0; i < frame.xyz.length / 3; i++) {
      const x = frame.xyz[i * 3]!
      const y = frame.xyz[i * 3 + 1]!
      const z = frame.xyz[i * 3 + 2]!
      if (x < minX) minX = x
      if (x > maxX) maxX = x
      if (y < minY) minY = y
      if (z < minZ) minZ = z
      if (z > maxZ) maxZ = z
      const intensity = frame.intensity![i]!
      if (intensity > maxIntensity) maxIntensity = intensity
    }
    expect(minX).toBeCloseTo(-8.67, 1)
    expect(maxX).toBeCloseTo(7.07, 1)
    expect(minY).toBeLessThan(-200.0)
    expect(minZ).toBeLessThan(-6.0)
    expect(maxZ).toBeLessThan(3.0)
    expect(maxIntensity).toBeCloseTo(255.0, 3)
  })
})

describe('BagFrames', () => {
  test.skipIf(!hasBag)('len and topics', () => {
    const frames = new BagFrames(DB, { cacheSize: 2, prefetchAhead: 0 })
    expect(frames.length).toBe(201)
    expect(frames.topicNames).toContain('/sensing/lidar/hesai128/pointcloud')
    frames.close()
  })

  test.skipIf(!hasBag)('cache avoids reparse', () => {
    const frames = new BagFrames(DB, { cacheSize: 2, prefetchAhead: 0 })
    frames.frame(0)
    const first = frames.parseCount
    frames.frame(0)
    expect(frames.parseCount).toBe(first) // второе обращение -- попадание в кэш
    frames.close()
  })

  test.skipIf(!hasBag)('frame carries ring', () => {
    const frames = new BagFrames(DB, { cacheSize: 2, prefetchAhead: 0 })
    const frame = frames.frame(0)
    expect(frame.xyz.length / 3).toBe(346808)
    expect(frame.ring).not.toBeNull()
    expect(frame.ring!.length).toBe(frame.xyz.length / 3)
    let maxRing = 0
    for (let i = 0; i < frame.ring!.length; i++) if (frame.ring![i]! > maxRing) maxRing = frame.ring![i]!
    expect(maxRing).toBe(127)
    frames.close()
  })

  test.skipIf(!hasBag)('ring cache evicts', () => {
    const frames = new BagFrames(DB, { cacheSize: 2, prefetchAhead: 0 })
    for (let i = 0; i < 4; i++) frames.frame(i)
    expect(frames.cachedCount).toBeLessThanOrEqual(2)
    frames.close()
  })

  test.skipIf(!hasBag)('timestamps sorted ns and span', () => {
    const frames = new BagFrames(DB, { cacheSize: 1, prefetchAhead: 0 })
    const timestamps = frames.timestamps
    expect(timestamps.length).toBe(frames.length)
    for (let i = 1; i < timestamps.length; i++) {
      expect(timestamps[i]! > timestamps[i - 1]!).toBe(true)
    }
    // Разница в наносекундах: без BigInt точность теряется (1.7e18 > 2^53).
    const spanSeconds = Number(timestamps[timestamps.length - 1]! - timestamps[0]!) / 1e9
    expect(Math.abs(spanSeconds - 20.4)).toBeLessThanOrEqual(0.5)
    frames.close()
  })

  test.skipIf(!hasBag)('prefetch fills ahead', async () => {
    const frames = new BagFrames(DB, { cacheSize: 8, prefetchAhead: 3 })
    frames.prefetch(0)
    await Bun.sleep(4000)
    expect(frames.cachedCount).toBeGreaterThanOrEqual(2) // предзагрузка грузит кадры вперёд
    expect(frames.prefetchParses).toBeGreaterThan(0)
    frames.close()
  })

  test.skipIf(!hasBag)('prefetched frame is identical to a direct parse', async () => {
    const prefetching = new BagFrames(DB, { cacheSize: 8, prefetchAhead: 3 })
    prefetching.prefetch(0)
    for (let i = 0; i < 200 && prefetching.cachedCount < 2; i++) await Bun.sleep(50)
    expect(prefetching.cachedCount).toBeGreaterThanOrEqual(2)

    const prefetched = prefetching.frame(1)
    expect(prefetching.hits).toBeGreaterThan(0) // кадр пришёл из фонового разбора

    const directFrames = new BagFrames(DB, { cacheSize: 2, prefetchAhead: 0 })
    const direct = directFrames.frame(1)
    directFrames.close()
    expect(prefetched.xyz.length).toBe(direct.xyz.length)
    expect(prefetched.intensity!.length).toBe(direct.intensity!.length)
    expect(prefetched.ring!.length).toBe(direct.ring!.length)
    let differences = 0
    for (let i = 0; i < direct.xyz.length; i++) {
      if (prefetched.xyz[i] !== direct.xyz[i]) differences++
      if (prefetched.intensity![i] !== direct.intensity![i]) differences++
    }
    for (let i = 0; i < direct.ring!.length; i++) {
      if (prefetched.ring![i] !== direct.ring![i]) differences++
    }
    expect(differences).toBe(0)
    prefetching.close()
  })
})

describe('Colors', () => {
  const xyz = Float32Array.from([
    0.0, 0.0, -7.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 3.0,
    1.0, -40.0, 0.0,
  ])
  const intensity = Float32Array.from([0.0, 0.0, 1.0, 255.0])
  const ring = Uint16Array.from([0, 64, 127, 12])

  test('shape dtype range all modes', () => {
    for (const mode of MODES) {
      const colors = frameColors(xyz, intensity, mode, 80.0, ring)
      expect(colors.length).toBe(4 * 3)
      let min = Infinity
      let max = -Infinity
      for (const value of colors) {
        if (value < min) min = value
        if (value > max) max = value
      }
      expect(min).toBeGreaterThanOrEqual(0.0)
      expect(max).toBeLessThanOrEqual(1.0)
      expect(max).toBeGreaterThan(0.5) // цвета яркие, не 0.05-0.2
    }
  })

  test('height mode distinguishes floor and ceiling', () => {
    const colors = frameColors(xyz, intensity, 'height', 80.0)
    expect(Array.from(colors.slice(0, 3))).not.toEqual(Array.from(colors.slice(6, 9)))
  })

  test('range mode varies with distance', () => {
    const near = Float32Array.from([0.0, -1.0, 0.0])
    const far = Float32Array.from([0.0, -79.0, 0.0])
    const nearColors = frameColors(near, null, 'range', 80.0)
    const farColors = frameColors(far, null, 'range', 80.0)
    expect(Array.from(nearColors)).not.toEqual(Array.from(farColors))
  })

  test('intensity mode uses percentiles not max', () => {
    const values = new Float32Array(1000)
    for (let i = 0; i < 10; i++) values[990 + i] = ((i + 1) * 255) / 10
    const points = new Float32Array(1000 * 3)
    const colors = frameColors(points, values, 'intensity', 80.0)
    let max = 0
    for (const value of colors) if (value > max) max = value
    // Нормировка по перцентилям не должна «съедаться» максимумом 255.
    expect(max).toBeGreaterThan(0.5)
  })

  test('missing field raises', () => {
    expect(() => frameColors(xyz, null, 'intensity', 80.0)).toThrow()
    expect(() => frameColors(xyz, intensity, 'ring', 80.0, null)).toThrow()
  })

  test('unknown mode raises', () => {
    expect(() => frameColors(xyz, intensity, 'nope', 80.0)).toThrow()
    expect(() => frameColors(xyz, intensity, 'zones', 80.0)).toThrow()
  })

  test('empty input', () => {
    const colors = frameColors(new Float32Array(0), null, 'height', 80.0)
    expect(colors.length).toBe(0)
  })

  test('turbo palette matches python reference', () => {
    // Контрольные значения из Python (bag_reader._lut_colors по linspace(0,1,256)):
    // Z_RANGE -> t = 0, 0.7, 1.0 -> уровни LUT 0, 178, 255. Палитра обязана
    // совпадать с эталоном, иначе цвет кадра разъедется.
    const colors = frameColors(
      Float32Array.from([0.0, 0.0, Z_RANGE[0], 0.0, 0.0, 0.0, 0.0, 0.0, Z_RANGE[1]]),
      null,
      'height',
      80.0,
    )
    expect(Array.from(colors.slice(0, 3))).toEqual([0.18995, 0.07176, 0.23217])
    expect(Array.from(colors.slice(3, 6))).toEqual([
      0.8271461960784313, 0.7791556470588237, 0.21773741176470593,
    ])
    expect(Array.from(colors.slice(6, 9))).toEqual([0.8407, 0.96458, 0.98167])
    expect(percentileLinear(Float64Array.from([0, 1, 2, 3]), 50.0)).toBe(1.5)
  })
})

describe('Trim', () => {
  test('trim range keeps corridor only', () => {
    const xyz = Float32Array.from([
      0.0, -250.0, 0.0,
      0.0, -80.0, 0.0,
      0.0, 0.0, 0.0,
      0.0, 5.0, 0.0,
      0.0, 40.0, 0.0,
    ])
    const mask = trimRange(xyz, 80.0, 5.0)
    expect(Array.from(mask)).toEqual([0, 1, 1, 1, 0])
  })

  test('trim applies to intensity and ring', () => {
    const xyz = Float32Array.from([0.0, -100.0, 0.0, 0.0, -1.0, 0.0])
    const intensity = Float32Array.from([1.0, 2.0])
    const ring = Uint16Array.from([7, 8])
    const [outXyz, outIntensity, outRing] = trim(xyz, intensity, ring, 80.0, 5.0)
    expect(outXyz.length / 3).toBe(1)
    expect(Array.from(outIntensity!)).toEqual([2.0])
    expect(Array.from(outRing!)).toEqual([8])
  })
})