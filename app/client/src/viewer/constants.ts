// Общие константы вида: палитра, пресеты камеры, размеры разреза.

import type { Meta } from '@/api/types'

/** Фон сцены — как в старом клиенте (светлый фон съедает точки). */
export const BACKGROUND = 0x05070c

/** Кэш кадров и предзагрузка вперёд, как в web/app.js. */
export const PREFETCH_AHEAD = 2
export const CACHE_LIMIT = 32

/** Задержка отправки правок модели рельсов на /set: не спамим сервер при протяжке. */
export const RAIL_DEBOUNCE_MS = 140

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