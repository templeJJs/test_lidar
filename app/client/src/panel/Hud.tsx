// HUD: те же строки метрик, что были в web/index.html (#hud), но собранные из
// состояния вьюера.

import type { Meta } from '@/api/types'
import type { ViewerStats } from '@/state/useViewer'

export interface HudProps {
  meta: Meta
  idx: number
  stats: ViewerStats
}

function hudText({ meta, idx, stats }: HudProps): string {
  const t = idx / Math.max(meta.hz, 0.01)
  const lines = [
    `кадр ${idx + 1}/${meta.frames}   t = ${t.toFixed(1)} с`,
    `точек ${stats.points}   ${stats.frameMs.toFixed(1)} мс   ${stats.fps.toFixed(1)} fps`,
    `пол z = ${meta.floor.a} + ${meta.floor.b}·y  (${meta.floor.grade_pct} %)`,
    `ось x = ${meta.axis_x} м   колея ${meta.gauge_mm} мм`,
    `стены x = ${meta.walls.join(', ')}`,
  ]
  const spans = meta.blocked_spans
  lines.push(`разрывов ${spans.length}${spans.length ? ': ' + spans.map((p) => `${p[0]}…${p[1]} м`).join(', ') : ''}`)
  if (stats.truncated) lines.push(`кадр обрезан до ${stats.points} точек`)
  if (stats.counts) {
    const parts = stats.counts
      .map((count, i) => ({ count, name: meta.zone_names[i] ?? String(i) }))
      .filter((part) => part.count > 0)
      .map((part) => `${part.name} ${part.count}`)
    if (parts.length) lines.push(parts.join('   '))
  }
  return lines.join('\n')
}

export function Hud(props: HudProps) {
  return (
    <div className="font-mono text-[12px] leading-[1.45] whitespace-pre-line text-[#7fb2ff]">
      {hudText(props)}
    </div>
  )
}