// Сборка семейств объектов из payload `/objects` для сцены и панели.
//
// Одна часть payload = одна пара (slug, part); семейство — это все части одного
// slug, склеенные в одну геометрию: тумблер в панели включает семейство, а не
// отдельные треугольники. Цвета заданы для известных семейств и выводятся из
// имени для новых, чтобы новый замер сразу был различим на сцене.

import type { ObjectPart, ObjectsPayload, TrackPayload } from '@/api/objects'

export interface Provenance {
  text: string
  /** Класс плашки: измерено (зелёная) / перенесено-не указано (жёлтая) / нет. */
  kind: 'measured' | 'gost' | ''
}

export interface ObjectFamily {
  key: string
  slug: string
  part: string
  nameRu: string
  kind: string
  source: Provenance
  faces: number
  color: number
  vertices: number[][]
  indices: number[]
  /** Прошла ли спецификация приёмку по данным (иначе -- выключено по умолчанию). */
  accepted: boolean
  acceptanceFailures: number
}

export const OBJECT_COLORS: Record<string, number> = {
  track_base: 0x6b7280,
  contact_rail: 0xffd479,
  contact_rail_supports: 0xb08968,
  cable_tray: 0x7fb069,
  hanging_wires: 0xd98cb3,
  tunnel_shell: 0x8d99ae,
  platform_station: 0x5c8dd6,
  columns_pillars: 0xc0a080,
  wall_equipment: 0x9fd3c7,
  inventory: 0xc94f6d,
}

/** Цвет семейства: известный — из таблицы, иначе устойчиво по имени. */
export function colorFor(slug: string): number {
  const known = OBJECT_COLORS[slug]
  if (known !== undefined) return known
  let hash = 0
  for (const ch of slug) hash = (hash * 31 + ch.charCodeAt(0)) % 360
  // Тот же приём, что в прототипе: HSL по хешу, средняя светлота и насыщенность.
  const c = 0.55
  const x = c * (1 - Math.abs(((hash / 60) % 2) - 1))
  const [r, g, b] =
    hash < 60 ? [c, x, 0] : hash < 120 ? [x, c, 0] : hash < 180 ? [0, c, x]
    : hash < 240 ? [0, x, c] : hash < 300 ? [x, 0, c] : [c, 0, x]
  const m = 0.32
  return ((Math.round((r + m) * 255) << 16)
    | (Math.round((g + m) * 255) << 8)
    | Math.round((b + m) * 255))
}

/**
 * Происхождение объекта. Порядок проверок важен: перенос и «запись не указана»
 * перекрывают `kind`, иначе чужой замер выглядел бы как измеренный здесь.
 */
export function provenance(part: ObjectPart): Provenance {
  // Приёмка -- ПЕРВОЙ: если числа не подтверждены данными, остальное неважно,
  // и объект не должен выглядеть как измеренный.
  if (part.accepted === false) {
    return { text: 'не прошёл приёмку', kind: 'gost' }
  }
  if (part.borrowed_from && part.borrowed_from.length > 0) {
    return { text: `перенесено из: ${part.borrowed_from.join(', ')}`, kind: 'gost' }
  }
  if (part.bag_unspecified && part.bag_unspecified.length > 0) {
    return { text: 'запись не указана', kind: 'gost' }
  }
  if (part.kind === 'not_observable') return { text: 'не наблюдается', kind: 'gost' }
  if (part.kind === 'measured') return { text: 'измерено', kind: 'measured' }
  if (part.kind === 'measured_partial') return { text: 'измерено частично', kind: 'measured' }
  return { text: part.kind ?? '—', kind: '' }
}

export function familyKey(part: ObjectPart): string {
  return `${part.slug ?? '?'}::${part.part ?? '?'}`
}

/** Семейства кадра. Пустой payload (объектов нет) даёт пустой список. */
export function buildFamilies(payload: ObjectsPayload | null): ObjectFamily[] {
  if (!payload) return []
  const grouped = new Map<string, ObjectPart[]>()
  for (const part of payload.parts ?? []) {
    if (!part.vertices?.length || !part.faces?.length) continue
    const key = familyKey(part)
    const list = grouped.get(key)
    if (list) list.push(part)
    else grouped.set(key, [part])
  }
  const families: ObjectFamily[] = []
  for (const [key, parts] of grouped) {
    const first = parts[0]
    const vertices: number[][] = []
    const indices: number[] = []
    for (const part of parts) {
      const base = vertices.length
      for (const v of part.vertices) vertices.push(v)
      for (const tri of part.faces) indices.push(tri[0] + base, tri[1] + base, tri[2] + base)
    }
    const slug = first.slug ?? '?'
    families.push({
      key,
      slug,
      part: first.part,
      nameRu: first.name_ru ?? slug,
      kind: first.kind ?? '',
      source: provenance(first),
      faces: indices.length / 3,
      color: colorFor(slug),
      accepted: first.accepted !== false,
      acceptanceFailures: first.acceptance_failures ?? 0,
      vertices,
      indices,
    })
  }
  families.sort((a, b) => b.faces - a.faces)
  return families
}
// ------------------------------------------------------- меши пути (/track) ----
//
// Рельсы и пол приходят из `track_geometry` уже в осях сцены и в том же
// контракте `{vertices, faces}`. Отдельного признака приёмки у них нет: их
// проверяет `check_track_geometry.py`, поэтому помечаем происхождение словами.

function makeFamily(
  key: string, slug: string, nameRu: string, color: number,
  vertices: number[][], faces: number[][], source: Provenance,
): ObjectFamily {
  const indices: number[] = []
  for (const tri of faces) indices.push(tri[0], tri[1], tri[2])
  return {
    key, slug, part: nameRu, nameRu, kind: 'measured', source,
    faces: indices.length / 3, color, vertices, indices,
    accepted: true, acceptanceFailures: 0,
  }
}

export function trackFamilies(track: TrackPayload | null): ObjectFamily[] {
  if (!track) return []
  const out: ObjectFamily[] = []
  for (const rail of track.rails ?? []) {
    const mesh = rail.mesh
    if (!mesh?.vertices?.length || !mesh?.faces?.length) continue
    const side = rail.side === 'left' ? 'левая нить' : rail.side === 'right' ? 'правая нить' : rail.side
    out.push(makeFamily(
      `track-rail-${rail.side}`, 'track_rail', `ходовой рельс (${side})`, 0xc8d2e0,
      mesh.vertices, mesh.faces,
      { text: 'измерено: сечение головки + ГОСТ тело', kind: 'measured' },
    ))
  }
  const floor = track.floor_mesh
  if (floor?.vertices?.length && floor?.faces?.length) {
    out.push(makeFamily(
      'track-floor', 'track_floor', 'пол пути (профиль с лотком)', 0x59637a,
      floor.vertices, floor.faces,
      { text: 'измерено (профиль пола)', kind: 'measured' },
    ))
  }
  return out
}
