// Легенда зон: плашки цветов из meta.zone_colors + красный «разрыв коридора».

import type { Meta } from '@/api/types'
import { ZONE_LABELS, rgbCss } from '@/viewer/constants'

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

export function Legend({ meta }: { meta: Meta }) {
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
    </div>
  )
}