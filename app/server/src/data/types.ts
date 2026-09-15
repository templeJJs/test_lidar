// Общие типы данных вьюера. Смысл полей обязан совпадать с zones.py / bag_reader.py,
// подробности контракта — в app/CONTRACT.md.

export type ZoneName = 'rail' | 'bed' | 'ceiling' | 'wall' | 'middle'

export interface ZoneParams {
  floorA: number
  floorB: number
  /** высота над плоскостью пола, ниже которой точка считается «низом» (bed) */
  floorBand: number
  /** абсолютный Z, выше которого точка считается потолком */
  ceilingZ: number
  /** поперечное положение оси пути */
  axisX: number
  /** колея, метры */
  gauge: number
  railHalfWidth: number
  railLow: number
  railHigh: number
  wallX: number[]
  wallBand: number
  detectWalls: boolean
  contactRail: boolean
  contactSide: 1 | -1
  contactOffset: number
  contactHeight: number
  contactHalfWidth: number
}

export interface FloorFit {
  a: number
  b: number
  rms: number
}

export interface RailsJson {
  x: number[]
  heights: number[]
  y: number[]
  z: number[]
  blocked: number[]
}

/** Общая часть ответов /meta и /set: то, что пересчитывается при правке модели. */
export interface TrackJson {
  floor: { a: number; b: number; rms: number; grade_pct: number }
  rails: RailsJson
  blocked_spans: Array<[number, number]>
}

/** Ответ /set: модель рельсов и всё, что от неё зависит. */
export interface ParamsJson extends TrackJson {
  axis_x: number
  gauge_mm: number
  contact_rail: boolean
  contact_offset: number
  contact_side: number
  rail_height: number
}

/** Ответ /meta. Полей contact_* тут нет — /meta их не отдаёт (как и web_viewer.py). */
export interface MetaJson extends TrackJson {
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
  axis_x: number
  gauge_mm: number
  walls: number[]
  grid: { spacing: number; half_width: number; length: number; z: number }
  zone_names: ZoneName[]
  zone_colors: number[][]
  blocked_color: number[]
  rail_color: number[]
  train_top: number
  obstacle_points: number
  rail_clearance: number
}

/** kind в бинарном кадре: 0 — цвета rgb8, 1 — метки зон label8 */
export const FRAME_KIND_RGB8 = 0
export const FRAME_KIND_LABEL8 = 1
export const FRAME_MAGIC = 'LZFR'
export const FRAME_VERSION = 1
export const FRAME_HEADER_BYTES = 12