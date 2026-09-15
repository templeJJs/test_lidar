// Состояние вьюера: метаданные, воспроизведение кадров, параметры отображения.
//
// Логика повторяет web/app.js: кадры берутся с предзагрузкой на 2 вперёд, кэш
// держит промисы (один запрос на кадр), при смене раскраски/модели рельсов кэш
// сбрасывается. Отличие: нет /version-поллинга с авто-перезагрузкой страницы.

import { useCallback, useEffect, useRef, useState } from 'react'
import {
  applyRailParams,
  decodeFrame,
  describeError,
  fetchFrame,
  fetchMeta,
  type RailParams,
} from '@/api/client'
import type { Meta, ParamsResponse } from '@/api/types'
import { CACHE_LIMIT, PREFETCH_AHEAD, RAIL_DEBOUNCE_MS, type PresetName } from '@/viewer/constants'
import { ViewerScene } from '@/viewer/scene'

/** Диапазоны ползунков модели рельсов — как в web/index.html. */
export interface RailRange {
  axisMin: number
  axisMax: number
  gaugeMin: number
  gaugeMax: number
  offsetMin: number
  offsetMax: number
}

export interface ViewerStats {
  points: number
  /** Сколько занял разбор и применение последнего кадра, мс. */
  frameMs: number
  fps: number
  /** Счётчики по зонам (индекс — как в meta.zone_names) в режиме zones. */
  counts: number[] | null
  /** Кадр не влез в буфер и был нарисован частично. */
  truncated: boolean
}

export interface ViewerController {
  meta: Meta | null
  error: string | null
  dismissError: () => void
  loading: boolean
  /** Номер кадра, который сейчас нарисован (с нуля). */
  idx: number
  /** Растёт при каждом применённом кадре: по нему перерисовывается разрез. */
  frameSeq: number
  playing: boolean
  rateHz: number
  realtime: boolean
  /** Период кадров по таймстемпам записи, мс (для режима реального времени). */
  recordingPeriodMs: number
  mode: string
  pointSize: number
  maxDist: number
  grid: boolean
  axes: boolean
  additive: boolean
  rails: RailParams
  railRange: RailRange
  stats: ViewerStats
  preset: PresetName | null
  scene: ViewerScene | null
  togglePlay: () => void
  stepBy: (delta: number) => void
  seek: (idx: number) => void
  setRateHz: (v: number) => void
  setRealtime: (v: boolean) => void
  setMode: (mode: string) => void
  setPointSize: (v: number) => void
  setMaxDist: (v: number) => void
  setGrid: (v: boolean) => void
  setAxes: (v: boolean) => void
  setAdditive: (v: boolean) => void
  setRail: (patch: Partial<RailParams>) => void
  applyPreset: (name: PresetName) => void
}

/** Разделяемый с rAF-циклом снимок состояния: React-рендер тут не нужен. */
interface Playback {
  /** Последний нарисованный кадр. */
  idx: number
  /** Последний запрошенный кадр (шаг подряд идущих нажатий не теряется). */
  requested: number
  playing: boolean
  rateHz: number
  realtime: boolean
  frames: number
  recordPeriodMs: number
  nextTick: number
  pending: boolean
  mode: string
}

const EMPTY_STATS: ViewerStats = { points: 0, frameMs: 0, fps: 0, counts: null, truncated: false }

function round2(v: number): number {
  return Math.round(v * 100) / 100
}

/** Модель рельсов, восстановленная из ответа сервера. */
function railsFromMeta(meta: Meta): RailParams {
  const xs = meta.rails.x
  const contact = xs.length >= 3
  const contactX = contact ? xs[2] : meta.axis_x + meta.gauge_mm / 2000
  const offset = Math.max(0.05, Math.abs(contactX - meta.axis_x))
  const side: 1 | -1 = contactX < meta.axis_x ? -1 : 1
  return { axis: meta.axis_x, gaugeMm: meta.gauge_mm, contact, offset: round2(offset), side }
}

function railRangeFromMeta(meta: Meta): RailRange {
  return {
    axisMin: round2(meta.axis_x - 5),
    axisMax: round2(meta.axis_x + 5),
    gaugeMin: 1000,
    gaugeMax: 1600,
    offsetMin: 0.3,
    offsetMax: 2.5,
  }
}

export function useViewer(canvas: HTMLCanvasElement | null): ViewerController {
  const [meta, setMeta] = useState<Meta | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [idx, setIdx] = useState(0)
  const [frameSeq, setFrameSeq] = useState(0)
  const [playing, setPlaying] = useState(true)
  const [rateHz, setRateHzState] = useState(10)
  const [realtime, setRealtimeState] = useState(false)
  const [mode, setModeState] = useState('zones')
  const [pointSize, setPointSizeState] = useState(3)
  const [maxDist, setMaxDistState] = useState(40)
  const [grid, setGridState] = useState(true)
  const [axes, setAxesState] = useState(true)
  const [additive, setAdditiveState] = useState(false)
  const [rails, setRailsState] = useState<RailParams>({ axis: 0, gaugeMm: 1520, contact: true, offset: 1.46, side: 1 })
  const [railRange, setRailRange] = useState<RailRange>({ axisMin: -5, axisMax: 5, gaugeMin: 1000, gaugeMax: 1600, offsetMin: 0.3, offsetMax: 2.5 })
  const [stats, setStats] = useState<ViewerStats>(EMPTY_STATS)
  const [preset, setPreset] = useState<PresetName | null>(null)
  const [scene, setScene] = useState<ViewerScene | null>(null)
  /** Период кадров записи, мс: в режиме реального времени играем по нему. */
  const [recordingPeriodMs, setRecordingPeriodMs] = useState(100)

  const metaRef = useRef<Meta | null>(null)
  const sceneRef = useRef<ViewerScene | null>(null)
  const paletteRef = useRef<readonly number[][]>([])
  const cacheRef = useRef(new Map<number, Promise<ArrayBuffer>>())
  const tokenRef = useRef(0)
  /** Ключ модели рельсов, которую уже отправили на /set (или получили в meta). */
  const appliedRailKeyRef = useRef<string | null>(null)
  const pb = useRef<Playback>({
    idx: 0,
    requested: 0,
    playing: true,
    rateHz: 10,
    realtime: false,
    frames: 1,
    recordPeriodMs: 100,
    nextTick: 0,
    pending: false,
    mode: 'zones',
  })
  const fpsRef = useRef(0)

  // Снимок состояния для rAF-цикла: сам цикл живёт вне React-рендера.
  useEffect(() => {
    const state = pb.current
    state.playing = playing
    state.rateHz = rateHz
    state.realtime = realtime
    state.frames = meta?.frames ?? 1
    state.mode = mode
  })

  // ------------------------------------------------------------- метаданные

  useEffect(() => {
    const controller = new AbortController()
    fetchMeta(controller.signal)
      .then((m) => {
        metaRef.current = m
        paletteRef.current = m.zone_colors
        const span = m.frames > 1 ? (m.duration_s * 1000) / (m.frames - 1) : m.hz > 0 ? 1000 / m.hz : 100
        pb.current.recordPeriodMs = span
        pb.current.frames = m.frames
        pb.current.mode = m.default_mode
        setRecordingPeriodMs(span)
        const railParams = railsFromMeta(m)
        appliedRailKeyRef.current = railKey(railParams)
        setMeta(m)
        setModeState(m.default_mode)
        setPointSizeState(m.point_size)
        setMaxDistState(m.max_dist)
        setRailsState(railParams)
        setRailRange(railRangeFromMeta(m))
        setRateHzState(Math.min(30, Math.max(1, Math.round(m.hz) || 10)))
      })
      .catch((err: unknown) => {
        if (!controller.signal.aborted) setError(`Не удалось получить /meta: ${describeError(err)}`)
      })
    return () => controller.abort()
  }, [])

  // ------------------------------------------------------- запросы кадров

  const frameRequest = useCallback((frameIdx: number): Promise<ArrayBuffer> => {
    const cache = cacheRef.current
    const hit = cache.get(frameIdx)
    if (hit) return hit
    const promise = fetchFrame(frameIdx, pb.current.mode)
    promise.catch(() => {
      cache.delete(frameIdx)
    })
    cache.set(frameIdx, promise)
    if (cache.size > CACHE_LIMIT) {
      for (const key of cache.keys()) {
        if (cache.size <= CACHE_LIMIT) break
        if (key === frameIdx) continue
        cache.delete(key)
      }
    }
    return promise
  }, [])

  const prefetch = useCallback(
    (frameIdx: number) => {
      const m = metaRef.current
      if (!m || frameIdx < 0 || frameIdx >= m.frames) return
      void frameRequest(frameIdx)
    },
    [frameRequest],
  )

  const load = useCallback(
    async (target: number) => {
      const scene = sceneRef.current
      const m = metaRef.current
      if (!scene || !m) return
      const want = Math.max(0, Math.min(Math.round(target), m.frames - 1))
      const token = ++tokenRef.current
      pb.current.requested = want
      for (let ahead = 1; ahead <= PREFETCH_AHEAD; ahead++) prefetch(want + ahead)
      const started = performance.now()
      setLoading(true)
      try {
        const buffer = await frameRequest(want)
        if (token !== tokenRef.current) return
        cacheRef.current.delete(want)
        const applied = scene.applyFrame(decodeFrame(buffer), paletteRef.current)
        const frameMs = performance.now() - started
        pb.current.idx = want
        setIdx(want)
        setStats({
          points: applied.count,
          frameMs,
          fps: fpsRef.current,
          counts: applied.counts,
          truncated: applied.truncated,
        })
        setFrameSeq((seq) => seq + 1)
        setError(null)
      } catch (err: unknown) {
        if (token === tokenRef.current) setError(`Кадр ${want + 1}: ${describeError(err)}`)
      } finally {
        if (token === tokenRef.current) setLoading(false)
      }
    },
    [frameRequest, prefetch],
  )

  const loadRef = useRef(load)
  useEffect(() => {
    loadRef.current = load
  }, [load])

  // ------------------------------------------------------------------ сцена

  // Сцена создаётся один раз: дальше meta меняется только правками модели
  // рельсов (/set), а сцена живёт своей жизнью.
  const ready = meta !== null
  useEffect(() => {
    const m = metaRef.current
    if (!canvas || !m) return
    const instance = new ViewerScene(
      canvas,
      m.grid,
      m.axis_x,
      {
        capacity: Math.max(1000, m.max_points_hint || 400000),
        pointSize: m.point_size,
        maxDist: m.max_dist,
        gridVisible: true,
        axesVisible: true,
      },
    )
    sceneRef.current = instance
    setScene(instance)
    instance.resize()
    instance.cameraPreset('lidar')
    setPreset('lidar')
    void loadRef.current(0)
    return () => {
      sceneRef.current = null
      setScene(null)
      instance.dispose()
    }
  }, [canvas, ready])

  // размер канвы: панель/разрез не трогают окно, но ресайз и смена DPR — да
  useEffect(() => {
    if (!scene) return
    const onResize = () => scene.resize()
    window.addEventListener('resize', onResize)
    return () => window.removeEventListener('resize', onResize)
  }, [scene])

  // ------------------------------------------------- настройки -> в сцену

  useEffect(() => {
    scene?.setPointSize(pointSize)
  }, [scene, pointSize])

  useEffect(() => {
    scene?.setMaxDist(maxDist)
  }, [scene, maxDist])

  useEffect(() => {
    scene?.setGridVisible(grid)
  }, [scene, grid])

  useEffect(() => {
    scene?.setAxesVisible(axes)
  }, [scene, axes])

  useEffect(() => {
    scene?.setAdditive(additive)
  }, [scene, additive])

  // ------------------------------------------------------------ rAF-цикл

  useEffect(() => {
    let raf = 0
    let last = performance.now()
    const tick = (now: number) => {
      raf = requestAnimationFrame(tick)
      const dt = now - last
      last = now
      if (dt > 0) fpsRef.current = fpsRef.current ? fpsRef.current * 0.9 + (1000 / dt) * 0.1 : 1000 / dt

      const state = pb.current
      if (state.playing && !state.pending && now >= state.nextTick) {
        const period = state.realtime
          ? state.recordPeriodMs
          : 1000 / Math.max(state.rateHz, 0.1)
        state.nextTick = Math.max(now + period, state.nextTick + period)
        state.pending = true
        const next = state.idx + 1 >= state.frames ? 0 : state.idx + 1
        void loadRef.current(next).finally(() => {
          state.pending = false
        })
      }
      sceneRef.current?.render()
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [])

  // fps в HUD обновляем 4 раза в секунду: каждый кадр перерисовывать панель незачем
  useEffect(() => {
    const timer = window.setInterval(() => {
      setStats((prev) => ({ ...prev, fps: fpsRef.current }))
    }, 250)
    return () => window.clearInterval(timer)
  }, [])

  // ------------------------------------------------------ действия панели

  const togglePlay = useCallback(() => {
    setPlaying((prev) => {
      pb.current.nextTick = performance.now()
      return !prev
    })
  }, [])

  const seek = useCallback((target: number) => {
    setPlaying(false)
    void loadRef.current(target)
  }, [])

  const stepBy = useCallback(
    (delta: number) => {
      seek(pb.current.requested + delta)
    },
    [seek],
  )

  const setMode = useCallback(
    (next: string) => {
      pb.current.mode = next
      setModeState(next)
      cacheRef.current.clear()
      void loadRef.current(pb.current.idx)
    },
    [],
  )

  const setRail = useCallback((patch: Partial<RailParams>) => {
    setRailsState((prev) => ({ ...prev, ...patch }))
  }, [])

  const applyPreset = useCallback(
    (name: PresetName) => {
      sceneRef.current?.cameraPreset(name)
      setPreset(name)
    },
    [],
  )

  const dismissError = useCallback(() => setError(null), [])

  // Правки модели рельсов уезжают на /set с дебаунсом; сервер возвращает новые
  // рельсы/пол/разрывы, кадр перекрашивается по новой геометрии.
  //
  // Важно: в зависимостях нет meta. Ответ /set подменяет meta, и если бы эффект
  // висел на meta, каждый ответ запускал бы новый /set — бесконечный цикл.
  // Поэтому помним ключ последней отправленной модели: он же стартовый, чтобы
  // при загрузке не дёргать сервер зря.
  const railKeyValue = railKey(rails)
  useEffect(() => {
    if (!ready || railKeyValue === appliedRailKeyRef.current) return
    const timer = window.setTimeout(() => {
      appliedRailKeyRef.current = railKeyValue
      applyRailParams(rails)
        .then((res: ParamsResponse) => {
          setMeta((prev) =>
            prev
              ? {
                  ...prev,
                  axis_x: res.axis_x,
                  gauge_mm: res.gauge_mm,
                  floor: res.floor,
                  rails: res.rails,
                  blocked_spans: res.blocked_spans,
                }
              : prev,
          )
          cacheRef.current.clear()
          return loadRef.current(pb.current.idx)
        })
        .catch((err: unknown) => setError(`Не удалось применить модель рельсов: ${describeError(err)}`))
    }, RAIL_DEBOUNCE_MS)
    return () => window.clearTimeout(timer)
  }, [ready, rails, railKeyValue])

  // ------------------------------------------------------ горячие клавиши

  useEffect(() => {
    const target = (ev: KeyboardEvent) => ev.target as HTMLElement | null
    const typing = (ev: KeyboardEvent) => {
      const el = target(ev)
      if (!el) return false
      const tag = el.tagName
      return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || el.isContentEditable
    }
    const onKeyDown = (ev: KeyboardEvent) => {
      if (typing(ev)) return
      if (ev.code === 'Space') {
        ev.preventDefault()
        togglePlay()
      } else if (ev.code === 'ArrowLeft') {
        ev.preventDefault()
        stepBy(-1)
      } else if (ev.code === 'ArrowRight') {
        ev.preventDefault()
        stepBy(1)
      }
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [stepBy, togglePlay])

  return {
    meta,
    error,
    dismissError,
    loading,
    idx,
    frameSeq,
    playing,
    rateHz,
    realtime,
    recordingPeriodMs,
    mode,
    pointSize,
    maxDist,
    grid,
    axes,
    additive,
    rails,
    railRange,
    stats,
    preset,
    scene,
    togglePlay,
    stepBy,
    seek,
    setRateHz: setRateHzState,
    setRealtime: setRealtimeState,
    setMode,
    setPointSize: setPointSizeState,
    setMaxDist: setMaxDistState,
    setGrid: setGridState,
    setAxes: setAxesState,
    setAdditive: setAdditiveState,
    setRail,
    applyPreset,
  }
}

function railKey(p: RailParams): string {
  return `${p.axis}|${p.gaugeMm}|${p.contact}|${p.offset}|${p.side}`
}