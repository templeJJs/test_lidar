// Легенда: плашки цветов из meta.zone_colors + красный «разрыв коридора», а с
// появлением новых слоёв — цвета линии хода и туннеля безопасности. Строки
// новых слоёв показываются только когда сервер отдал соответствующий блок.

import { PATH_VARIANT_LABELS, type PathVariantName } from '@/api/pathTunnel'
import type { Meta } from '@/api/types'
import { PATH_COLORS, TUNNEL_COLOR, ZONE_LABELS, hexCss, rgbCss } from '@/viewer/constants'

function LegendRow({ color, text }: { color: string; text: string }) {
  return (
    <div className="flex items-center gap-2 text-xs">
      <i
        className="h-3.5 w-3.5 shrink-0 rounded-[3px] border border-black/50"
        style={{ background: color }}
      />
      <span>{text}</span>
    </div>
  )
}

export interface LegendProps {
  meta: Meta
  /** Какие варианты линии хода посчитаны (в мете есть блок path). */
  pathVariants?: Record<PathVariantName, boolean> | null
  /** Сервер отдаёт блок tunnel. */
  tunnel?: boolean
}

export function Legend({ meta, pathVariants, tunnel }: LegendProps) {
  const path = pathVariants
    ? (Object.keys(PATH_COLORS) as PathVariantName[]).filter((name) => pathVariants[name])
    : []
  return (
    <div className="space-y-1">
      {meta.zone_names.map((name, i) => (
        <LegendRow
          key={name}
          color={rgbCss(meta.zone_colors[i])}
          text={`${name} — ${ZONE_LABELS[name] ?? name}`}
        />
      ))}
      <LegendRow color={rgbCss(meta.blocked_color)} text="разрыв — коридор занят препятствием" />
      {path.map((name) => (
        <LegendRow key={name} color={hexCss(PATH_COLORS[name])} text={`линия хода — ${PATH_VARIANT_LABELS[name]}`} />
      ))}
      {tunnel && (
        <>
          <LegendRow color={hexCss(TUNNEL_COLOR)} text="туннель — каркас габарита" />
          <LegendRow color={rgbCss(meta.blocked_color)} text="точка внутри туннеля" />
        </>
      )}
    </div>
  )
}
