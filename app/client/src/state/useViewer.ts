// Состояние вьюера: метаданные, воспроизведение кадров, параметры отображения.
//
// Логика повторяет web/app.js: кадры берутся с предзагрузкой на 2 вперёд, кэш
// держит промисы (один запрос на кадр), при смене раскраски/модели рельсов кэш
// сбрасывается. Отличие: нет /version-поллинга с авто-перезагрузкой страницы.

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  applyRailParams,
  applyTunnelParams,
  decodeFrame,
  describeError,
  fetchFrame,
  fetchMeta,
  type RailParams,
} from '@/api/client'
import {
  PATH_VARIANT_NAMES,
  readPath,
  readTunnel,
  spansLength,
  type PathVariantName,
  type TunnelJson,
} from '@/api/pathTunnel'
import { fetchObjects, fetchTrack, type ObjectsPayload, type TrackPayload } from '@/api/objects'
import type { Meta, ParamsResponse } from '@/api/types'
import { buildFamilies, trackFamilies, type ObjectFamily } from '@/viewer/objects'
import {
  CACHE_LIMIT,
  PREFETCH_AHEAD,
  PATH_COLORS,
  RAIL_DEBOUNCE_MS,
  TUNNEL_DEBOUNCE_MS,
  type PresetName,
} from '@/viewer/constants'
import { makePathLine, type PathLine, type TunnelMask } from '@/viewer/path'
import { ViewerScene, type PathLayerEntry } from '@/viewer/scene'

/** Диапазоны ползунков модели рельсов — как в web/index.html. */
export interface RailRange {
  axisMin: number
  axisMax: number
  gaugeMin: number
  gaugeMax: number
  offsetMin: number
  offsetMax: number
}

/** Источник линии хода: три варианта по отдельности или все сразу. */
export type PathSource = PathVariantName | 'compare'

export interface PathSettings {
  visible: boolean
  source: PathSource
  /** Ширина ленты, м (LineBasicMaterial толщину игнорирует — рисуем полосу). */
  thickness: number
}

export interface TunnelSettings {
  visible: boolean
  half: number
  low: number
  high: number
  /** Исключать из туннеля контактный рельс. */
  contact: boolean
}

/** Сводка по туннелю из /meta: занятые участки и число бинов. */
export interface TunnelSummary {
  binsBlocked: number
  binsTotal: number
  /** Суммарная длина занятых участков, м. */
  meters: number
  spans: Array<[number, number]>
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
  /** Точек, покрашенных как «внутри туннеля», и время этой покраски, мс. */
  tunnelPoints: number
  tunnelMs: number
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
  path: PathSettings
  tunnel: TunnelSettings
  /** Сервер отдал блок `path` — иначе группа «Линия хода» неактивна. */
  pathAvailable: boolean
  /** Сервер отдал блок `tunnel` — иначе группа «Туннель» неактивна. */
  tunnelAvailable: boolean
  /** Какие варианты линии вообще пришли в мете. */
  pathVariants: Record<PathVariantName, boolean>
  tunnelSummary: TunnelSummary | null
  /** По какому варианту считаются каркас и маска (важно в режиме сравнения). */
  maskSource: PathVariantName | null
  stats: ViewerStats
  preset: PresetName | null
  scene: ViewerScene | null
  /** Объекты из замеров: семейства кадра, тумблеры и происхождение. */
  objects: {
    families: ObjectFamily[]
    hidden: Record<string, boolean>
    error: string | null
    stats: { n_parts: number; n_vertices: number; n_faces: number } | null
    meta: ObjectsPayload['objects']
    setVisible: (key: string, on: boolean) => void
    setAllVisible: (on: boolean) => void
    fitObjects: () => boolean
    /** Режим «только объекты»: облако скрыто, камера по замерам. */
    only: boolean
    setOnly: (on: boolean) => void
  }
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
  setPath: (patch: Partial<PathSettings>) => void
  setTunnel: (patch: Partial<TunnelSettings>) => void
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

const EMPTY_STATS: ViewerStats = {
  points: 0,
  frameMs: 0,
  fps: 0,
  counts: null,
  truncated: false,
  tunnelPoints: 0,
  tunnelMs: 0,
}

/** Значения линий/туннеля по умолчанию — совпадают с блоком `tunnel` в спеке. */
const PATH_DEFAULTS: PathSettings = { visible: true, source: 'corridor', thickness: 0.12 }
const TUNNEL_DEFAULTS: TunnelSettings = { visible: true, half: 1.36, low: 0.1, high: 3.7, contact: true }

function round2(v: number): number {
  return Math.round(v * 100) / 100
}

/** Настройки туннеля из блока /meta (сервер — источник истины на старте). */
function tunnelFromMeta(t: TunnelJson | null): TunnelSettings {
  if (!t) return TUNNEL_DEFAULTS
  return {
    visible: TUNNEL_DEFAULTS.visible,
    half: t.half_width,
    low: t.z_low,
    high: t.z_high,
    contact: t.exclude_contact,
  }
}

/**
 * Пересечение ответа /set с текущей метой: сервер отдаёт только то, что
 * пересчитал, а ключи `tunnel`/`path` появляются лишь у новой сборки.
 */
function mergeSetResponse(prev: Meta | null, res: ParamsResponse): Meta | null {
  if (!prev) return prev
  const extra = res as ParamsResponse & { tunnel?: unknown; path?: unknown }
  const next: Meta = {
    ...prev,
    axis_x: res.axis_x ?? prev.axis_x,
    gauge_mm: res.gauge_mm ?? prev.gauge_mm,
    floor: res.floor ?? prev.floor,
    rails: res.rails ?? prev.rails,
    blocked_spans: res.blocked_spans ?? prev.blocked_spans,
  }
  if (extra.tunnel !== undefined) {
    ;(next as Meta & { tunnel?: unknown }).tunnel = extra.tunnel
  }
  if (extra.path !== undefined) {
    ;(next as Meta & { path?: unknown }).path = extra.path
  }
  return next
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
  const [path, setPathState] = useState<PathSettings>(PATH_DEFAULTS)
  const [tunnel, setTunnelState] = useState<TunnelSettings>(TUNNEL_DEFAULTS)
  const [stats, setStats] = useState<ViewerStats>(EMPTY_STATS)
  // Объекты из замеров: payload кадра, скрытые семейства и текст сбоя слоя.
  const [objects, setObjects] = useState<ObjectsPayload | null>(null)
  const [objectsError, setObjectsError] = useState<string | null>(null)
  const [objectsHidden, setObjectsHidden] = useState<Record<string, boolean>>({})
  // Режим «только объекты»: облако скрыто, камера по габариту замеров --
  // иначе объекты тонут в точках и непонятно, что смотреть.
  const [objectsOnly, setObjectsOnlyState] = useState(false)
  // Меши пути из /track (ходовые рельсы, пол): отдельный источник, тот же формат.
  const [track, setTrack] = useState<TrackPayload | null>(null)
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
  /** То же для настроек туннеля. */
  const appliedTunnelKeyRef = useRef<string | null>(null)
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
        const tunnelParams = tunnelFromMeta(readTunnel(m))
        appliedTunnelKeyRef.current = tunnelKey(tunnelParams)
        setMeta(m)
        setModeState(m.default_mode)
        setPointSizeState(m.point_size)
        setMaxDistState(m.max_dist)
        setRailsState(railParams)
        setRailRange(railRangeFromMeta(m))
        setTunnelState(tunnelParams)
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
          tunnelPoints: scene.tunnelPainted,
          tunnelMs: scene.tunnelPaintMs,
        })
        setFrameSeq((seq) => seq + 1)
        setError(null)
        // Объекты из замеров -- отдельная ручка /objects: сбой слоя НЕ роняет
        // кадр (облако и меши пути уже применены), поэтому ошибка своя.
        try {
          const payload = await fetchObjects(want)
          if (token === tokenRef.current) {
            setObjects(payload)
            setObjectsError(null)
          }
        } catch (err: unknown) {
          if (token === tokenRef.current) {
            setObjects(null)
            setObjectsError(describeError(err))
          }
        }
        // Меши пути: сбой не роняет кадр -- облако и объекты уже применены.
        try {
          const body = await fetchTrack(want)
          if (token === tokenRef.current) setTrack(body)
        } catch {
          if (token === tokenRef.current) setTrack(null)
        }
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

  // ------------------------------------------------ линия хода и туннель

  const pathJson = useMemo(() => readPath(meta), [meta])
  const tunnelJson = useMemo(() => readTunnel(meta), [meta])

  const pathVariants = useMemo<Record<PathVariantName, boolean>>(() => {
    const map = { corridor: false, rails: false, axis: false }
    for (const name of PATH_VARIANT_NAMES) {
      const v = pathJson?.variants[name]
      map[name] = !!v && !v.unavailable && v.y.length >= 2 && v.x.length >= 2
    }
    return map
  }, [pathJson])

  /** Линии хода из меты, готовые к отрисовке и к выборке x(y). */
  const pathLines = useMemo(() => {
    if (!pathJson || !meta) return null
    const opts = { zOffset: pathJson.z_offset, floorA: meta.floor.a, floorB: meta.floor.b }
    const out: Partial<Record<PathVariantName, PathLine>> = {}
    for (const name of PATH_VARIANT_NAMES) {
      const line = makePathLine(pathJson.variants[name], opts)
      if (line) out[name] = line
    }
    return out
  }, [pathJson, meta])

  /**
   * Вариант, по которому строятся каркас и маска. В режиме сравнения выбранного
   * источника нет — берём коридор, а если его не посчитали, первый доступный.
   * Если выбранный вариант недоступен (например, рельсы не нашлись) — тоже
   * откатываемся, чтобы туннель не пропадал молча.
   */
  const maskSource = useMemo<PathVariantName | null>(() => {
    if (!pathLines) return null
    const preferred: PathVariantName | null = path.source === 'compare' ? 'corridor' : path.source
    if (preferred && pathLines[preferred]) return preferred
    for (const name of PATH_VARIANT_NAMES) if (pathLines[name]) return name
    return null
  }, [pathLines, path.source])

  const maskLine = maskSource && pathLines ? (pathLines[maskSource] ?? null) : null

  const maskColor = useMemo<[number, number, number]>(() => {
    const c = meta?.blocked_color
    return c && c.length >= 3 ? [c[0], c[1], c[2]] : [255, 32, 32]
  }, [meta])

  useEffect(() => {
    if (!scene) return
    const entries: PathLayerEntry[] = []
    if (pathLines) {
      const names: PathVariantName[] =
        path.source === 'compare' ? [...PATH_VARIANT_NAMES] : [path.source]
      for (const name of names) {
        const line = pathLines[name]
        if (line) entries.push({ name, line, color: PATH_COLORS[name] })
      }
    }
    scene.setPathLayer(entries, path.thickness, path.visible && entries.length > 0)
  }, [scene, pathLines, path.visible, path.source, path.thickness])

  /** Семейства объектов кадра: склейка частей одного slug в один меш. */
  const objectFamilies = useMemo(() => buildFamilies(objects), [objects])
  const trackFamiliesList = useMemo(() => trackFamilies(track), [track])
  const allFamilies = useMemo(
    () => [...trackFamiliesList, ...objectFamilies],
    [trackFamiliesList, objectFamilies],
  )

  useEffect(() => {
    if (!scene) return
    const visible = new Set(allFamilies.filter((f) => !objectsHidden[f.key]).map((f) => f.key))
    scene.setObjectsLayer(allFamilies, visible)
    // Смена кадра пересобирает слой, но режим «только объекты» не сбрасывает:
    // облако остаётся скрытым, пока пользователь не выключит режим сам.
    scene.setPointsVisible(!objectsOnly)
  }, [scene, allFamilies, objectsHidden, objectsOnly])

  // Непринятые семейства гасим ОДИН раз на семейство: иначе эффект спорил бы с
  // пользователем, который решил посмотреть такой объект всё равно.
  const autoHiddenRef = useRef<Set<string>>(new Set())

  useEffect(() => {
    setObjectsHidden((prev) => {
      let changed = false
      const next = { ...prev }
      for (const family of allFamilies) {
        if (!family.accepted && !autoHiddenRef.current.has(family.key)) {
          autoHiddenRef.current.add(family.key)
          next[family.key] = true
          changed = true
        }
      }
      return changed ? next : prev
    })
  }, [allFamilies])

  const setObjectVisible = useCallback((key: string, on: boolean) => {
    setObjectsHidden((prev) => {
      const next = { ...prev }
      if (on) delete next[key]
      else next[key] = true
      return next
    })
  }, [])

  const setAllObjectsVisible = useCallback(
    (on: boolean) => {
      setObjectsHidden(on ? {} : Object.fromEntries(allFamilies.map((f) => [f.key, true])))
    },
    [allFamilies],
  )

  /** Режим «только объекты»: скрыть облако и вписать камеру по замерам. */
  const setObjectsOnly = useCallback((on: boolean) => {
    setObjectsOnlyState(on)
    const target = sceneRef.current
    if (!target) return
    target.setPointsVisible(!on)
    if (on) {
      const box = target.objectsBounds()
      if (box) target.frameBox(box)
    }
  }, [])

  /** Вписать камеру по габариту объектов (вид «по объектам»). */
  const fitObjects = useCallback(() => {
    const target = sceneRef.current
    if (!target) return false
    const box = target.objectsBounds()
    if (!box) return false
    target.frameBox(box)
    return true
  }, [])

  /** Всё, что нужно маске «точка внутри туннеля»: считается без обращения к сцене. */
  const tunnelMask = useMemo<TunnelMask | null>(() => {
    if (!meta || !tunnelJson || !maskLine) return null
    const xs = meta.rails.x
    const hs = meta.rails.heights
    const running = Math.max(0, Math.min(2, xs.length, hs.length))
    const withContact = tunnel.contact && rails.contact && xs.length >= 3 && hs.length >= 3
    return {
      line: maskLine,
      half: tunnel.half,
      zLow: tunnel.low,
      zHigh: tunnel.high,
      railXs: xs.slice(0, running),
      railZs: hs.slice(0, running),
      contactX: withContact ? xs[2] : null,
      contactZ: withContact ? hs[2] : 0,
      color: maskColor,
    }
  }, [meta, tunnelJson, maskLine, tunnel, rails.contact, maskColor])

  useEffect(() => {
    scene?.setTunnelMask(tunnel.visible ? tunnelMask : null)
  }, [scene, tunnel.visible, tunnelMask])

  useEffect(() => {
    if (!scene) return
    scene.setTunnelFrame(
      tunnelJson && maskLine
        ? { line: maskLine, half: tunnel.half, zLow: tunnel.low, zHigh: tunnel.high }
        : null,
      tunnel.visible,
    )
  }, [scene, tunnelJson, maskLine, tunnel.visible, tunnel.half, tunnel.low, tunnel.high])

  const tunnelSummary = useMemo<TunnelSummary | null>(() => {
    if (!tunnelJson) return null
    return {
      binsBlocked: tunnelJson.bins_blocked,
      binsTotal: tunnelJson.bins_total,
      meters: spansLength(tunnelJson.spans),
      spans: tunnelJson.spans,
    }
  }, [tunnelJson])

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

  // fps в HUD обновляем 4 раза в секунду: каждый кадр перерисовывать панель незачем.
  // Заодно читаем счётчики туннеля: их пишет сцена, React о них не знает.
  useEffect(() => {
    const timer = window.setInterval(() => {
      const scene = sceneRef.current
      setStats((prev) => ({
        ...prev,
        fps: fpsRef.current,
        tunnelPoints: scene?.tunnelPainted ?? 0,
        tunnelMs: scene?.tunnelPaintMs ?? 0,
      }))
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

  const setPath = useCallback((patch: Partial<PathSettings>) => {
    setPathState((prev) => ({ ...prev, ...patch }))
  }, [])

  const setTunnel = useCallback((patch: Partial<TunnelSettings>) => {
    setTunnelState((prev) => ({ ...prev, ...patch }))
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
          setMeta((prev) => mergeSetResponse(prev, res))
          cacheRef.current.clear()
          return loadRef.current(pb.current.idx)
        })
        .catch((err: unknown) => setError(`Не удалось применить модель рельсов: ${describeError(err)}`))
    }, RAIL_DEBOUNCE_MS)
    return () => window.clearTimeout(timer)
  }, [ready, rails, railKeyValue])

  // Ползунки туннеля уезжают на /set тем же дебаунсом. Кадр при этом не
  // перезапрашивается: покраска «внутри туннеля» — клиентская, её пересчитывает
  // эффект по tunnelMask. Если сервер ключей ещё не знает (нет блока `tunnel` в
  // мете), запрос не отправляем вовсе — группа в панели неактивна.
  const tunnelKeyValue = tunnelKey(tunnel)
  useEffect(() => {
    if (!ready || !tunnelJson || tunnelKeyValue === appliedTunnelKeyRef.current) return
    const timer = window.setTimeout(() => {
      appliedTunnelKeyRef.current = tunnelKeyValue
      applyTunnelParams(tunnel)
        .then((res: ParamsResponse) => setMeta((prev) => mergeSetResponse(prev, res)))
        .catch((err: unknown) => setError(`Не удалось применить туннель: ${describeError(err)}`))
    }, TUNNEL_DEBOUNCE_MS)
    return () => window.clearTimeout(timer)
  }, [ready, tunnel, tunnelKeyValue, tunnelJson])

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
    path,
    tunnel,
    pathAvailable: pathJson !== null,
    tunnelAvailable: tunnelJson !== null,
    pathVariants,
    tunnelSummary,
    maskSource,
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
    setPath,
    setTunnel,
    applyPreset,
    objects: {
      families: allFamilies,
      hidden: objectsHidden,
      error: objectsError,
      stats: objects?.stats ?? null,
      meta: objects?.objects ?? [],
      setVisible: setObjectVisible,
      setAllVisible: setAllObjectsVisible,
      fitObjects,
      only: objectsOnly,
      setOnly: setObjectsOnly,
    },
  }
}

function railKey(p: RailParams): string {
  return `${p.axis}|${p.gaugeMm}|${p.contact}|${p.offset}|${p.side}`
}

function tunnelKey(t: TunnelSettings): string {
  return `${t.half}|${t.low}|${t.high}|${t.contact}`
}