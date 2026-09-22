// Состояние разметки: режим, ход записи, объекты кадра, предпросмотр и запись.
//
// Поток «клик → предложение → правка → сохранение»:
//   клик по облаку (в режиме разметки) → /label/propose по ближайшей к лучу точке;
//   сервер отдаёт бокс и статус автоматики (детектор / кластер / вручную);
//   числа бокса правятся в панели, мышью — перетаскиванием по плоскости пола;
//   каждое изменение едет на /label/preview, точки трассировки рисуются в сцене;
//   «Сохранить» → /label/save: карточки, npz, index.jsonl, manifest, README.
//
// Трек живёт в ПУТЕВОЙ системе: объект стоит, сенсор едет. Сервер знает замеренный
// ход записи, панель даёт его переопределить в км/ч (50 км/ч — синтез быстрого
// подхода); пересчёт в м/кадр — по частоте записи из /meta.

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  AUTOMATION_COLORS,
  fetchMotion,
  previewLabel,
  proposeLabel,
  saveLabels,
  type Automation,
  type LabelClass,
  type LabelObjectIn,
  type LabelPose,
  type Motion,
  type PreviewResult,
  type SaveReport,
  type Vec3,
} from '@/api/label'
import type { Meta } from '@/api/types'
import type { LabelBox, ViewerScene } from '@/viewer/scene'

/** Объект в панели: геометрия + трек + статус автоматики. */
export interface LabelItem {
  /** id присваивает сервер при сохранении; до него объект живёт под `key`. */
  key: string
  id: string | null
  cls: LabelClass
  pose: LabelPose
  refFrame: number
  frames: number[]
  center: Vec3
  size: Vec3
  yaw: number
  pitch: number
  roll: number
  automation: Automation
  reason: string
  /** Скорость трека в км/ч; `null` -- «как замерено» (тогда её даёт /motion). */
  speedKmh: number | null
  speedSource: 'measured' | 'manual'
}

export interface LabelingController {
  on: boolean
  setOn: (on: boolean) => void
  motion: Motion | null
  motionError: string | null
  /** Замеренный ход записи, км/ч. */
  measurementKmh: number
  /** Скорость объекта: своя (правка) или замеренная, км/ч. */
  speedKmhOf: (o: LabelItem) => number
  objects: LabelItem[]
  /** Объекты, у которых текущий кадр входит в frames (список кадра). */
  frameObjects: LabelItem[]
  selectedKey: string | null
  selected: LabelItem | null
  select: (key: string | null) => void
  remove: (key: string) => void
  /** Правка полей выбранного объекта; `frames` можно задать прямо. */
  patch: (key: string, next: Partial<LabelItem>) => void
  /** Ручная постановка: объект в точке клика, статус «вручную». */
  addManual: (point: Vec3) => void
  /** Пересчитать предпросмотр выбранного объекта (обычно сам, после правок). */
  refreshPreview: () => void
  preview: PreviewResult | null
  previewError: string | null
  previewBusy: boolean
  status: string | null
  save: () => Promise<void>
  saveBusy: boolean
  saveReport: SaveReport | null
  /** Режим правки мышью: ЛКМ тянет выбранный объект по полу (камера заморожена). */
  dragMode: boolean
  setDragMode: (on: boolean) => void
}

interface Options {
  canvas: HTMLCanvasElement | null
  /** Обёртка канвы: на ней перехватываем клик в режиме разметки. */
  wrap: HTMLElement | null
  meta: Meta | null
  idx: number
  scene: ViewerScene | null
}

let seq = 0
function nextKey(): string {
  seq += 1
  return `lbl-${seq}`
}

export function useLabeling({ canvas, wrap, meta, idx, scene }: Options): LabelingController {
  const [on, setOn] = useState(false)
  const [motionGot, setMotionGot] = useState<{ bag: string; motion: Motion } | null>(null)
  const [motionErrorGot, setMotionErrorGot] = useState<{ bag: string; error: string } | null>(null)
  const [objects, setObjects] = useState<LabelItem[]>([])
  const [selectedKey, setSelectedKey] = useState<string | null>(null)
  const [preview, setPreview] = useState<PreviewResult | null>(null)
  const [previewError, setPreviewError] = useState<string | null>(null)
  const [previewBusy, setPreviewBusy] = useState(false)
  const [status, setStatus] = useState<string | null>(null)
  const [saveBusy, setSaveBusy] = useState(false)
  const [saveReport, setSaveReport] = useState<SaveReport | null>(null)
  const [dragMode, setDragMode] = useState(false)

  const bag = meta?.bag ?? null
  const hz = meta?.hz && meta.hz > 0 ? meta.hz : 10

  // Ход записи: спрашиваем на смену записи. Замер на сервере кэширован, а ответ
  // держим ВМЕСТЕ с именем записи: иначе после смены записи старый ход ещё жил бы
  // в состоянии и трек считался бы по чужому замеру (сброс через эффект -- лишний
  // рендер и лишняя синхронизация derived-состояния).
  useEffect(() => {
    if (!bag) return
    const ac = new AbortController()
    fetchMotion(bag, ac.signal)
      .then((m) => setMotionGot({ bag, motion: m }))
      .catch((err: unknown) => {
        if (ac.signal.aborted) return
        setMotionErrorGot({ bag, error: err instanceof Error ? err.message : String(err) })
      })
    return () => ac.abort()
  }, [bag])

  const motion = motionGot && motionGot.bag === bag ? motionGot.motion : null
  const motionError =
    motionErrorGot && motionErrorGot.bag === bag ? motionErrorGot.error : null
  const measurementKmh = motion ? motion.kmh : 0
  const speedKmhOf = useCallback(
    (o: LabelItem) => (o.speedSource === 'manual' && o.speedKmh !== null ? o.speedKmh : measurementKmh),
    [measurementKmh],
  )

  const selected = useMemo(
    () => objects.find((o) => o.key === selectedKey) ?? null,
    [objects, selectedKey],
  )

  const frameObjects = useMemo(
    () => objects.filter((o) => o.frames.includes(idx)),
    [objects, idx],
  )

  // ------------------------------------------------------------- предпросмотр

  const previewSeq = useRef(0)
  const refreshPreview = useCallback(() => {
    const item = objects.find((o) => o.key === selectedKey)
    if (!item || !bag) {
      setPreview(null)
      return
    }
    const seqNow = previewSeq.current + 1
    previewSeq.current = seqNow
    setPreviewBusy(true)
    previewLabel(bag, toWire(item))
      .then((res) => {
        if (previewSeq.current !== seqNow) return
        setPreview(res)
        setPreviewError(null)
      })
      .catch((err: unknown) => {
        if (previewSeq.current !== seqNow) return
        setPreview(null)
        setPreviewError(err instanceof Error ? err.message : String(err))
      })
      .finally(() => {
        if (previewSeq.current === seqNow) setPreviewBusy(false)
      })
  }, [bag, objects, selectedKey])

  // Правка чисел/класса/кадра: пересчитываем предпросмотр с дебаунсом, чтобы
  // перетаскивание мышью не занимало сервер на каждый пиксель.
  useEffect(() => {
    if (!on || !selected) return
    const t = window.setTimeout(refreshPreview, 140)
    return () => window.clearTimeout(t)
  }, [
    on,
    selected,
    selected?.center,
    selected?.size,
    selected?.yaw,
    selected?.pitch,
    selected?.roll,
    selected?.cls,
    selected?.pose,
    selected?.refFrame,
    refreshPreview,
  ])

  // ------------------------------------------------------------------- объекты

  const patch = useCallback((key: string, next: Partial<LabelItem>) => {
    setObjects((prev) => prev.map((o) => (o.key === key ? { ...o, ...next } : o)))
  }, [])

  const remove = useCallback((key: string) => {
    setObjects((prev) => prev.filter((o) => o.key !== key))
    setSelectedKey((cur) => (cur === key ? null : cur))
    setPreview(null)
  }, [])

  const addManual = useCallback(
    (point: Vec3) => {
      const key = nextKey()
      const item: LabelItem = {
        key,
        id: null,
        cls: 'equipment',
        pose: 'unknown',
        refFrame: idx,
        frames: [idx],
        center: [point[0], point[1], point[2]],
        size: [0.6, 1.0, 1.0],
        yaw: 0,
        pitch: 0,
        roll: 0,
        automation: 'manual',
        reason: 'поставлено вручную — автоматика объект не предложила',
        speedKmh: null,
        speedSource: 'measured',
      }
      setObjects((prev) => [...prev, item])
      setSelectedKey(key)
      setStatus('объект поставлен вручную: правьте поля или тяните мышью')
    },
    [idx],
  )

  // Клик по облаку: предложение бокса сервером.
  const proposeAt = useCallback(
    async (point: Vec3) => {
      if (!bag) return
      setStatus('предложение по клику…')
      try {
        const got = await proposeLabel(bag, idx, point)
        if (got.automation === 'manual' || !got.ok) {
          // Автоматика не предложила габарит (нет кластера вокруг луча) -- тогда
          // объект ставится РУКАМИ в ту же точку: это и есть серый/красный статус.
          addManual(point)
          setStatus(`автоматика не предложила: ${got.reason}`)
          return
        }
        const key = nextKey()
        const item: LabelItem = {
          key,
          id: null,
          cls: (got.pose_suggest as LabelClass) ?? 'equipment',
          pose: got.pose_suggest === 'person' ? 'standing' : 'unknown',
          refFrame: idx,
          frames: [idx],
          center: [got.center[0], got.center[1], got.center[2]],
          size: [got.size[0], got.size[1], got.size[2]],
          yaw: got.yaw,
          pitch: got.pitch,
          roll: got.roll,
          automation: got.automation,
          reason: got.reason,
          speedKmh: null,
          speedSource: 'measured',
        }
        setObjects((prev) => [...prev, item])
        setSelectedKey(key)
        setStatus(got.reason)
      } catch (err: unknown) {
        setStatus(err instanceof Error ? err.message : String(err))
      }
    },
    [bag, idx, addManual],
  )

  // ------------------------------------------------------- клик и перетаскивание

  const dragRef = useRef<{ key: string; dx: number; dy: number; z: number } | null>(null)
  const downRef = useRef<{ x: number; y: number } | null>(null)

  useEffect(() => {
    if (!on || !scene || !canvas || !wrap) return
    const target = wrap

    const onDown = (ev: PointerEvent) => {
      if (ev.target !== canvas || ev.button !== 0) return
      downRef.current = { x: ev.clientX, y: ev.clientY }
      if (!dragMode || !selected) return
      const hit = scene.pointOnPlane(ev.clientX, ev.clientY, selected.center[2])
      if (!hit) return
      dragRef.current = {
        key: selected.key,
        dx: hit[0] - selected.center[0],
        dy: hit[1] - selected.center[1],
        z: selected.center[2],
      }
      // Камеру на время перетаскивания замораживаем: иначе ЛКМ и вращает вид,
      // и тянет объект. Событие глушим в capture-фазе, чтобы OrbitControls
      // (он слушает канву) его не увидел.
      ev.stopPropagation()
      ev.preventDefault()
    }

    const onMove = (ev: PointerEvent) => {
      const drag = dragRef.current
      if (!drag) return
      ev.stopPropagation()
      const hit = scene.pointOnPlane(ev.clientX, ev.clientY, drag.z)
      if (!hit) return
      patch(drag.key, { center: [hit[0] - drag.dx, hit[1] - drag.dy, drag.z] })
    }

    const onUp = (ev: PointerEvent) => {
      const drag = dragRef.current
      const down = downRef.current
      downRef.current = null
      if (drag) {
        dragRef.current = null
        ev.stopPropagation()
        setStatus('объект перенесён: пересчитываю точки трассировки')
        return
      }
      if (!down || ev.target !== canvas || ev.button !== 0) return
      const moved = Math.hypot(ev.clientX - down.x, ev.clientY - down.y)
      if (moved > 5) return
      const point = scene.pickPoint(ev.clientX, ev.clientY)
      if (!point) {
        setStatus('под курсором нет точки облака (кликните по объекту)')
        return
      }
      void proposeAt(point)
    }

    target.addEventListener('pointerdown', onDown, true)
    target.addEventListener('pointermove', onMove, true)
    target.addEventListener('pointerup', onUp, true)
    return () => {
      target.removeEventListener('pointerdown', onDown, true)
      target.removeEventListener('pointermove', onMove, true)
      target.removeEventListener('pointerup', onUp, true)
    }
  }, [on, scene, canvas, wrap, dragMode, selected, patch, proposeAt])

  // ------------------------------------------------------------------- отрисовка

  const boxes = useMemo<LabelBox[]>(
    () =>
      objects.map((o) => ({
        key: o.key,
        center: o.center,
        size: o.size,
        yaw: o.yaw,
        pitch: o.pitch,
        roll: o.roll,
        color: AUTOMATION_COLORS[o.automation],
        selected: o.key === selectedKey,
      })),
    [objects, selectedKey],
  )

  useEffect(() => {
    if (!scene) return
    if (!on) {
      scene.setLabelVisible(false)
      return
    }
    const pts = preview?.points?.length
      ? new Float32Array(preview.points.flat())
      : null
    scene.setLabelLayer(boxes, pts)
  }, [scene, on, boxes, preview])

  // --------------------------------------------------------------------- запись

  const save = useCallback(async () => {
    if (!bag || objects.length === 0) {
      setStatus('сохранять нечего: поставьте хотя бы один объект')
      return
    }
    setSaveBusy(true)
    setStatus('запись в labels/…')
    try {
      const report = await saveLabels(
        bag,
        objects.map((o) => toSaveWire(o, hz, speedKmhOf(o))),
        hz,
      )
      setSaveReport(report)
      // id присваивает сервер: подставляем их обратно, чтобы повторное
      // сохранение обновляло те же карточки, а не плодило новые. Порядок
      // `written` совпадает с порядком объектов только когда ошибок нет.
      if (report.errors.length === 0) {
        const ids = report.written.map((w) => w.id)
        setObjects((prev) =>
          prev.map((o, i) => (ids[i] ? { ...o, id: ids[i] } : o)),
        )
      }
      setStatus(
        report.errors.length
          ? `сохранено ${report.written.length}, ошибок ${report.errors.length}: ${report.errors[0].error}`
          : `сохранено объектов: ${report.written.length} (всего в записи ${report.objects_total})`,
      )
    } catch (err: unknown) {
      setStatus(err instanceof Error ? err.message : String(err))
    } finally {
      setSaveBusy(false)
    }
  }, [bag, objects, hz, speedKmhOf])

  return {
    on,
    setOn,
    motion,
    motionError,
    measurementKmh,
    speedKmhOf,
    objects,
    frameObjects,
    selectedKey,
    selected,
    select: setSelectedKey,
    remove,
    patch,
    addManual,
    refreshPreview,
    preview,
    previewError,
    previewBusy,
    status,
    save,
    saveBusy,
    saveReport,
    dragMode,
    setDragMode,
  }
}

/** Объект панели -> тело для /label/preview и /label/save. */
export function toWire(o: LabelItem): LabelObjectIn {
  return {
    id: o.id ?? undefined,
    class: o.cls,
    pose: o.pose,
    ref_frame: o.refFrame,
    frames: o.frames.length ? o.frames : [o.refFrame],
    center: o.center,
    size: o.size,
    yaw: o.yaw,
    pitch: o.pitch,
    roll: o.roll,
    speed_m_per_frame: 0,
    speed_source: 'measured',
    automation: o.automation,
  }
}

/** Он же для сохранения: км/ч -> м/кадр по частоте записи. */
export function toSaveWire(o: LabelItem, hz: number, kmh: number): LabelObjectIn {
  const wire = toWire(o)
  wire.speed_m_per_frame = kmh / 3.6 / (hz > 0 ? hz : 10)
  wire.speed_source = o.speedSource
  return wire
}