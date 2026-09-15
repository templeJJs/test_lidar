// Типы API вьюера на стороне клиента. Держим их синхронными с app/server/src/data/types.ts
// и с форматом ответов web_viewer.py — подробности в app/CONTRACT.md.

export type ZoneName = 'rail' | 'bed' | 'ceiling' | 'wall' | 'middle'

export interface FloorJson {
  a: number
  b: number
  rms: number
  grade_pct: number
}

export interface RailsJson {
  x: number[]
  heights: number[]
  y: number[]
  z: number[]
  blocked: number[]
}

export interface Meta {
  bag: string
  frames: number
  duration_s: number
  hz: number
  max_dist: number
  behind: number
  default_mode: string
  modes: string[]
  point_size: number
  max_points_hint: number
  floor: FloorJson
  walls: number[]
  axis_x: number
  gauge_mm: number
  rails: RailsJson
  blocked_spans: Array<[number, number]>
  grid: { spacing: number; half_width: number; length: number; z: number }
  zone_names: ZoneName[]
  zone_colors: number[][]
  blocked_color: number[]
  rail_color: number[]
  train_top: number
  obstacle_points: number
  rail_clearance: number
}

/** Ответ /set: те же поля, что меняются при правке модели рельсов. */
export interface ParamsResponse {
  axis_x: number
  gauge_mm: number
  contact_rail: boolean
  contact_offset: number
  contact_side: number
  rail_height: number
  floor: FloorJson
  rails: RailsJson
  blocked_spans: Array<[number, number]>
}

export const FRAME_KIND_RGB8 = 0
export const FRAME_KIND_LABEL8 = 1
export const FRAME_MAGIC = 'LZFR'
export const FRAME_HEADER_BYTES = 12

/** Разобранный кадр: позиции всегда, цвета или метки — по kind. */
export interface DecodedFrame {
  kind: number
  count: number
  positions: Float32Array
  colors?: Uint8Array
  labels?: Uint8Array
}