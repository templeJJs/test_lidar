// Точка входа приложения вьюера: канва 3D, панель, миникарта-разрез, оверлеи.

import { useEffect, useState } from 'react'
import { Button } from '@/components/ui/button'
import { Panel } from '@/panel/Panel'
import { SectionMap } from '@/panel/SectionMap'
import { useLabeling } from '@/state/useLabeling'
import { useViewer } from '@/state/useViewer'

/** Подсказка по управлению — как #hint в web/index.html. */
function Hint() {
  return (
    <div className="pointer-events-none absolute bottom-3 left-1/2 -translate-x-1/2 rounded-lg border border-white/15 bg-[#0c0f18]/70 px-3 py-1.5 text-[12px] whitespace-nowrap text-muted-foreground backdrop-blur-md">
      ЛКМ — поворот, ПКМ/средняя — сдвиг, колесо — зум · Space — пауза, ←/→ — кадр, Tab — панель
    </div>
  )
}

function ErrorBox({ text, onDismiss }: { text: string; onDismiss: () => void }) {
  return (
    <div className="absolute top-3 right-3 max-w-[46vw] rounded-lg border border-red-400/30 bg-red-950/85 px-3 py-2 font-mono text-[12px] whitespace-pre-wrap text-red-100">
      <Button
        variant="ghost"
        size="xs"
        className="float-right -mt-1 ml-2 text-red-100 hover:bg-red-400/20"
        onClick={onDismiss}
      >
        Закрыть
      </Button>
      {text}
    </div>
  )
}

export default function App() {
  const [canvas, setCanvas] = useState<HTMLCanvasElement | null>(null)
  const [wrap, setWrap] = useState<HTMLDivElement | null>(null)
  const [panelOpen, setPanelOpen] = useState(true)
  const viewer = useViewer(canvas)
  const { meta, scene, error } = viewer
  // Разметка живёт отдельным хуком: у неё своя ручка запроса и своя отрисовка,
  // но кадр/сцена/запись — от вьюера, второго источника кадров не заводим.
  const labeling = useLabeling({ canvas, wrap, meta, idx: viewer.idx, scene })

  // Тёмная тема shadcn: светлые токены съедают точки на фоне сцены.
  useEffect(() => {
    document.documentElement.classList.add('dark')
  }, [])

  useEffect(() => {
    document.title = meta ? `LiDAR — ${meta.bag}` : 'LiDAR — облако точек'
  }, [meta])

  // Tab сворачивает панель — как в старом клиенте, где Tab всегда забирался
  // страницей (панель — единственный источник настроек вьюера).
  useEffect(() => {
    const onKeyDown = (ev: KeyboardEvent) => {
      if (ev.code !== 'Tab') return
      ev.preventDefault()
      setPanelOpen((open) => !open)
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [])

  return (
    <div
      ref={setWrap}
      className="relative h-full w-full overflow-hidden bg-[#05070c] text-[#dde4f0]"
    >
      <canvas ref={setCanvas} className="absolute inset-0 block h-full w-full" />

      {!meta && (
        <div className="pointer-events-none absolute inset-0 grid place-items-center text-sm text-muted-foreground">
          {error ? 'Не удалось загрузить /meta' : 'Загрузка метаданных…'}
        </div>
      )}

      {scene && meta && <SectionMap scene={scene} meta={meta} frameSeq={viewer.frameSeq} />}

      {meta && panelOpen && (
        <Panel viewer={viewer} labeling={labeling} onHide={() => setPanelOpen(false)} />
      )}

      {meta && !panelOpen && (
        <Button
          size="xs"
          variant="outline"
          className="absolute top-3 left-3 bg-[#0c0f18]/70 backdrop-blur-md"
          onClick={() => setPanelOpen(true)}
        >
          Панель · Tab
        </Button>
      )}

      {meta && <Hint />}
      {meta && labeling.on && (
        <div className="pointer-events-none absolute top-3 left-1/2 -translate-x-1/2 rounded-lg border border-emerald-400/30 bg-emerald-950/60 px-3 py-1 text-[12px] whitespace-nowrap text-emerald-100 backdrop-blur-md">
          Режим разметки: ЛКМ по объекту — предложение габарита
          {labeling.dragMode ? ' · правка мышью: тяните бокс' : ''}
        </div>
      )}
      {error && <ErrorBox text={error} onDismiss={viewer.dismissError} />}
    </div>
  )
}