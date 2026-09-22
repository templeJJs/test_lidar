// Объекты из замеров (`objects/specs/*.json`) — маршрут /objects.
//
// Меши приходят уже в осях данных вьюера (X поперёк, Y вдоль, Z вверх, метры):
// у каждой части `vertices` и `faces` — тот же контракт, что у мешей пути
// (`/track`), поэтому клиент их не пересчитывает. Числа и происхождение дают
// спецификации, собранные замерами: `kind` (measured / measured_partial /
// not_observable), `borrowed_from` (спецификация измерена на другой записи и
// перенесена по оси) и `bag_unspecified` (конкретная запись не названа).

export interface ObjectPart {
  slug?: string
  name_ru?: string
  kind?: string
  part: string
  shape?: string
  source?: string
  assembled_from?: string | null
  borrowed_from?: string[]
  bag_unspecified?: string[]
  n_instances?: number
  /** Числа спецификации подтверждены данными (итог objects/check_spec.py). */
  accepted?: boolean
  /** Сколько замечаний у спецификации; 0 -- принята. */
  acceptance_failures?: number
  vertices: number[][]
  faces: number[][]
}

export interface ObjectMeta {
  slug: string
  name_ru?: string
  kind?: string
  datum?: string
  n_parts: number
  n_faces: number
  n_skipped_foreign?: number
  n_ambiguous?: number
  accepted?: boolean
  acceptance_failures?: number
  borrowed_from?: string[] | null
  error?: string
}

export interface ObjectsPayload {
  bag: string
  frame: number
  objects: ObjectMeta[]
  parts: ObjectPart[]
  stats: { n_parts: number; n_vertices: number; n_faces: number }
}

export class ObjectsApiError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'ObjectsApiError'
  }
}

/** Объекты кадра. `idx` — номер кадра, как у `/frame`. */
export async function fetchObjects(idx: number, signal?: AbortSignal): Promise<ObjectsPayload> {
  const url = `/objects?idx=${idx}`
  const res = await fetch(url, { cache: 'no-store', signal })
  if (!res.ok) {
    // 503 = сборщик объектов не поднялся (нет objects/to_viewer.py) — это
    // отдельная фича, а не поломка вьюера: показываем текст, кадр живёт.
    let detail = ''
    try {
      detail = ((await res.json()) as { error?: string }).error ?? ''
    } catch {
      detail = ''
    }
    throw new ObjectsApiError(`GET ${url} → ${res.status}${detail ? `: ${detail}` : ''}`)
  }
  return (await res.json()) as ObjectsPayload
}
// --------------------------------------------------------------- меши пути ----
//
// Ходовые рельсы и пол меряет `track_geometry` (маршрут /track): это отдельный
// источник, и дублировать его замеры нельзя. Формат меша тот же --
// `{vertices, faces}` -- поэтому в сцене они живут рядом с объектами из
// спецификаций и управляются теми же тумблерами.

export interface TrackMesh {
  vertices: number[][]
  faces: number[][]
}

export interface TrackRail {
  side: string
  mesh: TrackMesh
  head?: Record<string, unknown>
  source?: Record<string, string>
}

export interface TrackPayload {
  frame: number
  rails: TrackRail[]
  floor_mesh: TrackMesh
  meta?: Record<string, unknown>
}

export async function fetchTrack(idx: number, signal?: AbortSignal): Promise<TrackPayload> {
  const url = `/track?frame=${idx}`
  const res = await fetch(url, { cache: 'no-store', signal })
  if (!res.ok) throw new ObjectsApiError(`GET ${url} → ${res.status}`)
  return (await res.json()) as TrackPayload
}
