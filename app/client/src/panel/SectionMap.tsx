// Миникарта-разрез X-Z в правом нижнем углу. Перерисовывается, когда сменился
// кадр (frameSeq) или поменялась модель рельсов/пола (meta).

import { useEffect, useRef } from 'react'
import type { Meta } from '@/api/types'
import type { ViewerScene } from '@/viewer/scene'
import { drawSection } from '@/viewer/section'

export interface SectionMapProps {
  scene: ViewerScene
  meta: Meta
  frameSeq: number
}

export function SectionMap({ scene, meta, frameSeq }: SectionMapProps) {
  const ref = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = ref.current
    if (!canvas) return
    drawSection(canvas, meta, {
      positions: scene.positions,
      colors: scene.colors,
      count: scene.count,
    })
  }, [scene, meta, frameSeq])

  return (
    <canvas
      ref={ref}
      className="pointer-events-none absolute right-3 bottom-3 rounded-[10px] border border-white/15 shadow-[0_8px_28px_rgba(0,0,0,.45)]"
    />
  )
}