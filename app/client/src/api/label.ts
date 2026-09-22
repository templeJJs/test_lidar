// Руки разметки: ход записи, предложение бокса по клику, предпросмотр трассировки
// и запись в `labels/`. Вся геометрия — в `labeling.py` на сервере, клиент только
// показывает и правит числа; поэтому в ответах те же имена полей, что там.

export type LabelClass = 'person' | 'equipment' | 'other'
export type LabelPose = 'standing' | 'walking' | 'running' | 'lying' | 'fallen' | 'unknown'
/** Статус автоматики: детектор подтвердил / предложил кластер / ставим руками. */
export type Automation = 'detector' | 'cluster' | 'manual'

export const LABEL_CLASSES: ReadonlyArray<readonly [LabelClass, string]> = [
  ['person', 'человек'],
  ['equipment', 'оборудование'],
  ['other', 'прочее'],
]

export const LABEL_POSES: ReadonlyArray<readonly [LabelPose, string]> = [
  ['standing', 'стоит'],
  ['walking', 'идёт'],
  ['running', 'бежит'],
  ['lying', 'лежит'],
  ['fallen', 'упал'],
  ['unknown', 'неизвестно'],
]

export const DEFAULT_POSE: Record<LabelClass, LabelPose> = {
  person: 'standing',
  equipment: 'unknown',
  other: 'unknown',
}

export const AUTOMATION_LABELS: Record<Automation, string> = {
  detector: 'детектор подтвердил',
  cluster: 'предложено кластером',
  manual: 'вручную (не предложено)',
}

/** Цвета боксов по статусу автоматики и точки трассировки. */
export const AUTOMATION_COLORS: Record<Automation, number> = {
  detector: 0x4ade80,
  cluster: 0xfacc15,
  manual: 0xf87171,
}

export const PREVIEW_COLOR = 0x38bdf8

export type Vec3 = [number, number, number]

export interface MotionPair {
  frames: [number, number]
  shift_m: number
  m_per_frame: number
  residual_m: number
  bands: number
}

export interface Motion {
  record: string
  m_per_frame: number
  kmh: number
  residual_m: number | null
  hz: number
  measured: boolean
  residual_ok?: boolean
  pairs: MotionPair[]
  error?: string
}

export interface Proposal {
  ok: boolean
  automation: Automation
  reason: string
  center: Vec3
  size: Vec3
  yaw: number
  pitch: number
  roll: number
  pose_suggest?: LabelClass | null
  cluster?: { n_points: number } | null
  detector?: { n_obstacles: number } | null
  path?: { along_m: number; lateral_m: number; height_m: number } | null
  frame: number
  record: string
}

export interface PreviewResult {
  class: LabelClass
  pose: LabelPose
  center: Vec3
  size: Vec3
  yaw: number
  pitch: number
  roll: number
  primitive_size: number[]
  n_points: number
  counts: { rays: number; rays_hit: number; points_removed: number }
  points: Vec3[]
  reflectance: number[] | null
  confirmed: boolean
  detector?: { n_obstacles: number } | null
  path?: { along_m: number; lateral_m: number; height_m: number } | null
  frame: number
  record: string
}

/** Объект разметки в том виде, в каком его принимает /label/save. */
export interface LabelObjectIn {
  id?: string
  class: LabelClass
  pose: LabelPose
  ref_frame: number
  frames: number[]
  center: Vec3
  size: Vec3
  yaw: number
  pitch: number
  roll: number
  speed_m_per_frame: number
  speed_source: 'measured' | 'manual'
  automation: Automation
}

export interface SaveReport {
  record: string
  dir: string
  index: string
  manifest: string
  readme: string
  written: Array<{ id: string; card: string; npz: string; n_points: number; size_bytes: number }>
  errors: Array<{ object: unknown; error: string }>
  objects_total: number
  counts: unknown
}

/** Ошибка разметки: текст сервера показывается в панели как есть. */
export class LabelError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'LabelError'
  }
}

async function readJson<T>(res: Response, what: string): Promise<T> {
  if (!res.ok) {
    let text = `${what} → ${res.status}`
    try {
      const body = (await res.json()) as { error?: string }
      if (body?.error) text = body.error
    } catch {
      /* тело не JSON — оставляем код */
    }
    throw new LabelError(text)
  }
  return (await res.json()) as T
}

function bagQuery(bag: string | null): string {
  return bag ? `?bag=${encodeURIComponent(bag)}` : ''
}

async function post<T>(path: string, bag: string | null, body: unknown): Promise<T> {
  const url = `${path}${bagQuery(bag)}`
  const res = await fetch(url, {
    method: 'POST',
    cache: 'no-store',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  return readJson<T>(res, `POST ${path}`)
}

/** Ход машины по записи (м/кадр). Замер на сервере кэшируется, можно звать смело. */
export async function fetchMotion(bag: string | null, signal?: AbortSignal): Promise<Motion> {
  const res = await fetch(`/motion${bagQuery(bag)}`, { cache: 'no-store', signal })
  const body = await readJson<{ bag: string; motion: Motion }>(res, 'GET /motion')
  return body.motion
}

export function proposeLabel(bag: string | null, frame: number, click: Vec3): Promise<Proposal> {
  return post<Proposal>('/label/propose', bag, { frame, click })
}

/** Предпросмотр: точки, которые увидел бы лидар, и ответ детектора. */
export function previewLabel(bag: string | null, object: LabelObjectIn): Promise<PreviewResult> {
  return post<PreviewResult>('/label/preview', bag, {
    ref_frame: object.ref_frame,
    class: object.class,
    pose: object.pose,
    center: object.center,
    size: object.size,
    yaw: object.yaw,
    pitch: object.pitch,
    roll: object.roll,
  })
}

export function saveLabels(
  bag: string | null,
  objects: LabelObjectIn[],
  hz: number,
): Promise<SaveReport> {
  // hz отдаём явно: клиент считает км/ч -> м/кадр по частоте из /meta, и без
  // этого сервер записал бы трек по своей (10 Гц), а карточка разошлась бы с
  // числом, которое ввёл человек.
  return post<SaveReport>('/label/save', bag, { objects, hz })
}