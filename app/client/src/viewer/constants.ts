// Общие константы вида: палитра, пресеты камеры, размеры разреза.

import type { Meta } from '@/api/types'

/** Фон сцены — как в старом клиенте (светлый фон съедает точки). */
export const BACKGROUND = 0x05070c

/** Кэш кадров и предзагрузка вперёд, как в web/app.js. */
export const PREFETCH_AHEAD = 2
export const CACHE_LIMIT = 32

/** Задержка отправки правок модели рельсов на /set: не спамим сервер при протяжке. */
export const RAIL_DEBOUNCE_MS = 140

/** Та же задержка для ползунков туннеля безопасности. */
export const TUNNEL_DEBOUNCE_MS = 140

/** Цвета линий хода по источникам (см. спеку «Линия хода»). */
export const PATH_COLORS: Record<'corridor' | 'rails' | 'axis', number> = {
  corridor: 0x38d66c, // зелёный — центр свободного коридора
  rails: 0xffd400, // жёлтый — ось измеренных рельсов
  axis: 0x58c8ff, // голубой — прямая по оси из панели
}

/** Каркас туннеля безопасности. */
export const TUNNEL_COLOR = 0xb98cff
/** Шаг сечений каркаса туннеля, м. */
export const TUNNEL_SECTION_STEP = 2

/** Шаг таблицы подстановки x(y) вдоль линии хода, м. */
export const PATH_LUT_STEP = 0.25

/** Диапазоны ползунков туннеля (по спеке /set). */
export const TUNNEL_HALF_MIN = 0.5
export const TUNNEL_HALF_MAX = 2.5
export const TUNNEL_LOW_MIN = -0.5
export const TUNNEL_LOW_MAX = 1.0
export const TUNNEL_HIGH_MIN = 1.0
export const TUNNEL_HIGH_MAX = 5.0

/** Диапазон ползунка толщины ленты линии хода, м. */
export const PATH_THICK_MIN = 0.04
export const PATH_THICK_MAX = 0.6

/** Наконечник на дальнем конце линии хода: длина и полная ширина основания, м. */
export const PATH_ARROW_LENGTH = 0.6
export const PATH_ARROW_WIDTH = 0.4

/**
 * Геометрия рельса из zones.py: полоса вокруг центра рельса и высотный
 * диапазон относительно плоскости пола. Клиент вычитает её из туннеля, иначе
 * маска покраснела бы на головах рельсов, которые стоят выше низа туннеля.
 */
export const RAIL_HALF_WIDTH = 0.12
export const RAIL_LOW = -0.1
export const RAIL_HIGH = 0.35
/** То же для контактного рельса (`contact_half_width` в zones.py). */
export const CONTACT_HALF_WIDTH = 0.1

/** Разрез X-Z в правом нижнем углу. */
export const SECTION_W = 380
export const SECTION_H = 240
export const SECTION_WINDOW = 12

/** Цвет точки в кадре, если метка зоны вышла за пределы палитры. */
export const FALLBACK_COLOR: [number, number, number] = [255, 0, 255]

export type PresetName = 'lidar' | 'top' | 'side' | 'iso' | 'behind' | 'fit'

export const PRESETS: ReadonlyArray<readonly [PresetName, string]> = [
  ['lidar', 'От лидара'],
  ['top', 'Сверху'],
  ['side', 'Сбоку'],
  ['iso', 'Изометрия'],
  ['behind', 'Сзади'],
  ['fit', 'Fit'],
]

/** Расшифровка имён зон для легенды и счётчиков в HUD. */
export const ZONE_LABELS: Record<string, string> = {
  rail: 'рельс',
  bed: 'низ',
  ceiling: 'потолок',
  wall: 'стены',
  middle: 'средний',
}

/** Цвет зоны по имени: в meta.zone_colors порядок задаёт meta.zone_names. */
export function zoneColor(meta: Meta, name: string): number[] {
  const i = meta.zone_names.indexOf(name as Meta['zone_names'][number])
  return i >= 0 ? meta.zone_colors[i] : FALLBACK_COLOR
}

/** Массив rgb8 -> CSS-цвет. */
export function rgbCss(c: readonly number[] | undefined): string {
  if (!c || c.length < 3) return 'rgb(255,0,255)'
  return `rgb(${c[0]},${c[1]},${c[2]})`
}

/** 0xRRGGBB -> CSS-цвет (для цветов линий хода, заданных константами). */
export function hexCss(hex: number): string {
  return `rgb(${(hex >> 16) & 255},${(hex >> 8) & 255},${hex & 255})`
}