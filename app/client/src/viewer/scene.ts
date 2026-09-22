// Сцена three.js вьюера: облако точек, статические оверлеи, камера.
//
// Поведение перенесено из web/app.js: тот же шейдер точек, те же пресеты камеры,
// та же сетка/оси. Отличие одно: линии рельсов в 3D не рисуются — модель рельсов
// влияет только на раскраску зон.
//
// Здесь же живут два новых оверлея: линия хода (плоские ленты) и каркас туннеля
// безопасности, а также клиентская покраска точек внутри туннеля. Базовые цвета
// кадра хранятся отдельно (`baseColors`), иначе после сдвига ползунка туннеля
// прошлые красные точки остались бы красными.

import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { FRAME_KIND_LABEL8, FRAME_KIND_RGB8, type DecodedFrame, type Meta } from '@/api/types'
import { BACKGROUND, FALLBACK_COLOR, TUNNEL_COLOR, type PresetName } from '@/viewer/constants'
import {
  buildRibbon,
  buildTunnelFrame,
  paintTunnel,
  type PathLine,
  type TunnelMask,
} from '@/viewer/path'
import type { ObjectFamily } from '@/viewer/objects'

// Оси лидара: X -- поперёк, Y -- вдоль (вперёд это -Y), Z -- вверх. Поэтому
// "верх" мира задаём по Z, иначе OrbitControls крутит вокруг Y.
THREE.Object3D.DEFAULT_UP.set(0, 0, 1)

const VERTEX_SHADER = /* glsl */ `
  attribute vec3 aColor;
  uniform float uSize;
  uniform float uPixelRatio;
  uniform float uMaxDist;
  varying vec3 vColor;
  varying float vKeep;
  void main() {
    vColor = aColor;
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * mv;
    vKeep = (-position.y <= uMaxDist) ? 1.0 : 0.0;
    float d = max(-mv.z, 0.5);
    gl_PointSize = clamp(uSize * uPixelRatio * (14.0 / d), 1.0, 90.0);
  }`

const FRAGMENT_SHADER = /* glsl */ `
  precision mediump float;
  varying vec3 vColor;
  varying float vKeep;
  void main() {
    if (vKeep < 0.5) discard;
    gl_FragColor = vec4(vColor, 1.0);
  }`

/** Точки трассировки разметки: голубые, чтобы не путать с цветами зон. */
const LABEL_POINT_COLOR = 0x38bdf8

/** R = Rx(roll)·Ry(pitch)·Rz(yaw) — матрица поворота из `labeling._rotation`. */
function labelQuaternion(yaw: number, pitch: number, roll: number): THREE.Quaternion {
  const d = Math.PI / 180
  const q = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), roll * d)
  q.multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), pitch * d))
  q.multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), yaw * d))
  return q
}

/** Результат применения кадра: сколько точек нарисовано и что с метками. */
export interface AppliedFrame {
  count: number
  truncated: boolean
  /** Счётчики по зонам (только для kind 1), индекс — как в meta.zone_names. */
  counts: number[] | null
}

export interface ViewerSceneOptions {
  capacity: number
  pointSize: number
  maxDist: number
  gridVisible: boolean
  axesVisible: boolean
}

/** Один вариант линии хода для отрисовки. */
export interface PathLayerEntry {
  name: string
  line: PathLine
  color: number
}

/** Бокс разметки для отрисовки: оси бокса — X поперёк, Y вдоль, Z вверх. */
export interface LabelBox {
  key: string
  center: [number, number, number]
  /** (ширина поперёк, высота, длина вдоль) — как `size` в labeling.py. */
  size: [number, number, number]
  yaw: number
  pitch: number
  roll: number
  color: number
  selected: boolean
}

/**
 * Слой разметки: каркас бокса на каждый объект + точки трассировки выбранного.
 * Геометрия каркаса одна на всех (единичный куб), её масштабируют под `size`:
 * боксов в кадре десятки, пересобирать буферы на каждый рендер незачем.
 */
export class ViewerScene {
  /** Буферы, которые читает миникарта-разрез. */
  readonly positions: Float32Array
  readonly colors: Uint8Array
  /** Сколько точек кадра сейчас в буферах. */
  count = 0
  /** Сколько точек покрашено как «внутри туннеля» на последнем пересчёте. */
  tunnelPainted = 0
  /** Время последней покраски туннеля, мс (бюджет — 20 мс на 340 тыс. точек). */
  tunnelPaintMs = 0

  private readonly canvas: HTMLCanvasElement
  private readonly capacity: number
  private readonly renderer: THREE.WebGLRenderer
  private readonly scene: THREE.Scene
  private readonly camera: THREE.PerspectiveCamera
  private readonly controls: OrbitControls
  private readonly geometry: THREE.BufferGeometry
  private readonly positionAttr: THREE.BufferAttribute
  private readonly colorAttr: THREE.BufferAttribute
  private readonly material: THREE.ShaderMaterial
  private readonly points: THREE.Points
  /** Базовые цвета кадра без покраски туннеля. */
  private readonly baseColors: Uint8Array
  private gridMesh: THREE.LineSegments
  private gridLength: number
  private readonly axesMesh: THREE.LineSegments
  private readonly pathGroup: THREE.Group
  private readonly objectsGroup: THREE.Group
  private readonly labelGroup: THREE.Group
  private readonly tunnelMesh: THREE.LineSegments
  private tunnelMask: TunnelMask | null = null
  private readonly grid: Meta['grid']
  private axisX: number
  private dist: number
  private disposed = false

  constructor(canvas: HTMLCanvasElement, grid: Meta['grid'], axisX: number, opts: ViewerSceneOptions) {
    this.canvas = canvas
    this.capacity = Math.max(1000, opts.capacity)
    this.grid = grid
    this.axisX = axisX
    this.dist = opts.maxDist
    this.gridLength = -1

    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true })
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2))
    this.renderer.setSize(canvas.clientWidth || window.innerWidth, canvas.clientHeight || window.innerHeight, false)
    this.renderer.setClearColor(BACKGROUND, 1)

    this.scene = new THREE.Scene()
    this.camera = new THREE.PerspectiveCamera(
      60,
      (canvas.clientWidth || window.innerWidth) / (canvas.clientHeight || window.innerHeight),
      0.1,
      4000,
    )
    this.camera.up.set(0, 0, 1)
    this.camera.position.set(axisX, 2, 1.6)

    this.controls = new OrbitControls(this.camera, canvas)
    this.controls.enableDamping = true
    this.controls.dampingFactor = 0.12
    this.controls.screenSpacePanning = false
    this.controls.maxDistance = 1500
    this.controls.target.set(axisX, -20, -0.5)
    this.controls.update()

    this.positions = new Float32Array(this.capacity * 3)
    this.colors = new Uint8Array(this.capacity * 3)
    this.baseColors = new Uint8Array(this.capacity * 3)
    this.geometry = new THREE.BufferGeometry()
    this.positionAttr = new THREE.BufferAttribute(this.positions, 3)
    this.positionAttr.setUsage(THREE.DynamicDrawUsage)
    this.colorAttr = new THREE.BufferAttribute(this.colors, 3, true) // normalized uint8
    this.colorAttr.setUsage(THREE.DynamicDrawUsage)
    this.geometry.setAttribute('position', this.positionAttr)
    this.geometry.setAttribute('aColor', this.colorAttr)
    this.geometry.setDrawRange(0, 0)

    this.material = new THREE.ShaderMaterial({
      uniforms: {
        uSize: { value: opts.pointSize },
        uPixelRatio: { value: Math.min(window.devicePixelRatio || 1, 2) },
        uMaxDist: { value: opts.maxDist },
      },
      vertexShader: VERTEX_SHADER,
      fragmentShader: FRAGMENT_SHADER,
      depthTest: true,
      depthWrite: true,
    })
    this.points = new THREE.Points(this.geometry, this.material)
    this.points.frustumCulled = false
    this.points.renderOrder = 1
    this.scene.add(this.points)

    // Свет для ОБЪЕКТОВ ИЗ ЗАМЕРОВ: они нарисованы MeshStandardMaterial, а без
    // источников света такой материал чёрный (проверено: до этой правки объекты
    // в сцене выглядели чёрными пятнами). Облако точек и линии хода света не
    // касаются -- у точек свой шейдер, у ленты MeshBasicMaterial.
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x223044, 1.6))
    const sun = new THREE.DirectionalLight(0xffffff, 1.9)
    sun.position.set(30, 25, 40)
    this.scene.add(sun)
    const fill = new THREE.DirectionalLight(0x88aaff, 0.7)
    fill.position.set(-25, -15, 20)
    this.scene.add(fill)

    this.gridMesh = this.makeGrid()
    this.gridMesh.visible = opts.gridVisible
    this.scene.add(this.gridMesh)

    this.axesMesh = this.makeAxes()
    this.axesMesh.visible = opts.axesVisible
    this.scene.add(this.axesMesh)

    this.pathGroup = new THREE.Group()
    this.scene.add(this.pathGroup)

    // Объекты из замеров (маршрут /objects): своя группа, чтобы тумблеры
    // семейств и «чистое поле» не трогали меши пути.
    this.objectsGroup = new THREE.Group()
    this.objectsGroup.visible = false
    this.scene.add(this.objectsGroup)

    // Слой разметки: каркасы боксов и точки трассировки. Отдельная группа и своя
    // прозрачность, чтобы режим разметки не зависел от тумблеров объектов/пути.
    this.labelGroup = new THREE.Group()
    this.labelGroup.visible = false
    this.scene.add(this.labelGroup)

    this.tunnelMesh = new THREE.LineSegments(
      new THREE.BufferGeometry(),
      new THREE.LineBasicMaterial({
        color: TUNNEL_COLOR,
        transparent: true,
        opacity: 0.45,
        depthWrite: false,
      }),
    )
    this.tunnelMesh.renderOrder = 3
    this.tunnelMesh.visible = false
    this.scene.add(this.tunnelMesh)
  }

  // ------------------------------------------------------------------ кадры

  /**
   * Копирует разобранный кадр в буферы сцены. Цвета для kind 1 берём из палитры
   * зон: сервер присылает метки, а не цвета. Базовые цвета идут в отдельный
   * буфер, а в цвета сцены попадают через покраску туннеля — она и выставляет
   * `needsUpdate`.
   */
  applyFrame(frame: DecodedFrame, zoneColors: readonly number[][]): AppliedFrame {
    const fits = Math.min(frame.count, this.capacity)
    this.positions.set(frame.positions.subarray(0, fits * 3))
    let counts: number[] | null = null

    if (frame.kind === FRAME_KIND_RGB8) {
      const src = frame.colors
      if (!src) throw new Error('кадр rgb8 без цветовой полезной нагрузки')
      this.baseColors.set(src.subarray(0, fits * 3))
    } else if (frame.kind === FRAME_KIND_LABEL8) {
      const labels = frame.labels
      if (!labels) throw new Error('кадр label8 без меток')
      for (let i = 0, j = 0; i < fits; i++) {
        const c = zoneColors[labels[i]] ?? FALLBACK_COLOR
        this.baseColors[j++] = c[0]
        this.baseColors[j++] = c[1]
        this.baseColors[j++] = c[2]
      }
      counts = new Array<number>(zoneColors.length).fill(0)
      for (let i = 0; i < frame.count; i++) counts[labels[i]]++
    } else {
      throw new Error(`неизвестный тип раскраски ${frame.kind}`)
    }

    this.count = fits
    this.positionAttr.needsUpdate = true
    this.repaintTunnel()
    this.geometry.setDrawRange(0, fits)
    return { count: fits, truncated: frame.count > this.capacity, counts }
  }

  // ------------------------------------------------------------ отображение

  setPointSize(size: number): void {
    this.material.uniforms.uSize.value = size
  }

  /** Дальность обрезки: вершинный шейдер отбрасывает точки дальше uMaxDist, а
   * сетка пересобирается по новой длине (как buildGrid в старом клиенте). */
  setMaxDist(dist: number): void {
    this.dist = dist
    this.material.uniforms.uMaxDist.value = dist
    this.rebuildGrid()
  }

  setGridVisible(visible: boolean): void {
    this.gridMesh.visible = visible
  }

  setAxesVisible(visible: boolean): void {
    this.axesMesh.visible = visible
  }

  /** "Плотная заливка": сложение цветов без записи в глубину. */
  setAdditive(additive: boolean): void {
    this.material.blending = additive ? THREE.AdditiveBlending : THREE.NormalBlending
    this.material.depthWrite = !additive
    this.material.needsUpdate = true
  }

  // ----------------------------------------------------- линия хода, туннель

  /**
   * Линия хода лентами. Геометрия пересобирается целиком: узлов у линии
   * десятки, это дешевле, чем отслеживать изменения по частям.
   */
  setPathLayer(entries: readonly PathLayerEntry[], thickness: number, visible: boolean): void {
    for (const child of this.pathGroup.children) {
      const mesh = child as THREE.Mesh
      mesh.geometry.dispose()
      ;(mesh.material as THREE.Material).dispose()
    }
    this.pathGroup.clear()
    this.pathGroup.visible = visible
    if (!visible) return
    for (const entry of entries) {
      const ribbon = buildRibbon(entry.line, thickness)
      if (!ribbon) continue
      const geometry = new THREE.BufferGeometry()
      geometry.setAttribute('position', new THREE.BufferAttribute(ribbon.positions, 3))
      geometry.setIndex(new THREE.BufferAttribute(ribbon.indices, 1))
      // depthTest выключен: маршрут должен читаться поверх облака точек, иначе
      // он тонет в головах рельсов и балласте.
      const material = new THREE.MeshBasicMaterial({
        color: entry.color,
        side: THREE.DoubleSide,
        depthTest: false,
        depthWrite: false,
        transparent: false,
      })
      const mesh = new THREE.Mesh(geometry, material)
      mesh.name = `path-${entry.name}`
      mesh.frustumCulled = false
      mesh.renderOrder = 4
      this.pathGroup.add(mesh)
    }
  }

  setObjectsLayer(families: readonly ObjectFamily[], visible: ReadonlySet<string>): void {
    for (const child of this.objectsGroup.children) {
      const mesh = child as THREE.Mesh
      mesh.geometry.dispose()
      ;(mesh.material as THREE.Material).dispose()
    }
    this.objectsGroup.clear()
    let shown = 0
    for (const family of families) {
      if (!visible.has(family.key)) continue
      const geometry = new THREE.BufferGeometry()
      geometry.setAttribute('position', new THREE.Float32BufferAttribute(family.vertices.flat(), 3))
      geometry.setIndex(family.indices)
      geometry.computeVertexNormals()
      // DoubleSide обязателен: часть замеров -- ПОВЕРХНОСТИ (пол, основание,
      // сечение обделки), а не замкнутые тела, и с односторонним материалом они
      // исчезают при взгляде снизу.
      const material = new THREE.MeshStandardMaterial({
        color: family.color,
        side: THREE.DoubleSide,
        metalness: 0.1,
        roughness: 0.65,
        // Слабое свечение: объект стоит у пола, куда свет почти не достаёт, и
        // без него замер читается как тёмное пятно в облаке.
        emissive: new THREE.Color(family.color).multiplyScalar(0.22),
      })
      const mesh = new THREE.Mesh(geometry, material)
      mesh.name = `objects-${family.key}`
      mesh.renderOrder = 3
      mesh.frustumCulled = false
      this.objectsGroup.add(mesh)
      shown += 1
    }
    this.objectsGroup.visible = shown > 0
  }

  // ------------------------------------------------------------ разметка

  /**
   * Каркасы боксов разметки и точки трассировки выбранного объекта.
   * Пересобирается целиком на каждое изменение: боксов десятки, а поворот
   * бокса задаётся кватернионом Rx(roll)·Ry(pitch)·Rz(yaw) — та же матрица, что
   * в `labeling._rotation`, иначе каркас разошёлся бы с мешем трассировки.
   */
  setLabelLayer(boxes: readonly LabelBox[], points: Float32Array | null): void {
    for (const child of this.labelGroup.children) {
      const mesh = child as THREE.Mesh
      mesh.geometry.dispose()
      if (Array.isArray(mesh.material)) mesh.material.forEach((m) => m.dispose())
      else (mesh.material as THREE.Material).dispose()
      mesh.clear()
    }
    this.labelGroup.clear()
    const cube = new THREE.EdgesGeometry(new THREE.BoxGeometry(1, 1, 1))
    for (const box of boxes) {
      const material = new THREE.LineBasicMaterial({
        color: box.color,
        transparent: true,
        opacity: box.selected ? 1 : 0.8,
        depthTest: true,
      })
      const mesh = new THREE.LineSegments(cube, material)
      mesh.position.set(box.center[0], box.center[1], box.center[2])
      // size = (поперёк, высота, вдоль) -> куб по осям (X, Z, Y).
      mesh.scale.set(
        Math.max(box.size[0], 0.05),
        Math.max(box.size[2], 0.05),
        Math.max(box.size[1], 0.05),
      )
      mesh.quaternion.copy(labelQuaternion(box.yaw, box.pitch, box.roll))
      mesh.renderOrder = 5
      mesh.frustumCulled = false
      this.labelGroup.add(mesh)
      if (box.selected) {
        const marker = new THREE.Mesh(
          new THREE.SphereGeometry(0.14, 12, 8),
          new THREE.MeshBasicMaterial({ color: box.color }),
        )
        marker.position.copy(mesh.position)
        marker.renderOrder = 6
        marker.frustumCulled = false
        this.labelGroup.add(marker)
      }
    }
    if (points && points.length >= 3) {
      const geometry = new THREE.BufferGeometry()
      geometry.setAttribute('position', new THREE.BufferAttribute(points, 3))
      const material = new THREE.PointsMaterial({
        color: LABEL_POINT_COLOR,
        size: 0.12,
        sizeAttenuation: true,
        depthTest: true,
      })
      const cloud = new THREE.Points(geometry, material)
      cloud.renderOrder = 5
      cloud.frustumCulled = false
      this.labelGroup.add(cloud)
    }
    this.labelGroup.visible = true
  }

  setLabelVisible(visible: boolean): void {
    this.labelGroup.visible = visible
  }

  /**
   * Точка облака под курсором: ближайшая к клику в ПИКСЕЛЯХ (как её видит
   * камера). Перебираем позиции кадра вручную, без Raycaster: у `Points`
   * порог задаётся в мире и не совпадает с экранным размером точки, а проекция
   * 340 тыс. точек занимает единицы миллисекунд.
   */
  pickPoint(clientX: number, clientY: number, radiusPx = 14): [number, number, number] | null {
    const rect = this.canvas.getBoundingClientRect()
    if (rect.width <= 0 || rect.height <= 0 || this.count === 0) return null
    const ndcX = ((clientX - rect.left) / rect.width) * 2 - 1
    const ndcY = -((clientY - rect.top) / rect.height) * 2 + 1
    this.camera.updateMatrixWorld()
    const m = new THREE.Matrix4().multiplyMatrices(
      this.camera.projectionMatrix,
      this.camera.matrixWorldInverse,
    )
    const e = m.elements
    const halfW = rect.width / 2
    const halfH = rect.height / 2
    let best = -1
    let bestD = radiusPx * radiusPx
    for (let i = 0; i < this.count; i++) {
      const x = this.positions[i * 3]
      const y = this.positions[i * 3 + 1]
      const z = this.positions[i * 3 + 2]
      if (-y > this.dist) continue // такие точки шейдер не рисует
      const w = e[3] * x + e[7] * y + e[11] * z + e[15]
      if (w <= 1e-6) continue
      const cx = (e[0] * x + e[4] * y + e[8] * z + e[12]) / w
      const cy = (e[1] * x + e[5] * y + e[9] * z + e[13]) / w
      const dx = (cx - ndcX) * halfW
      const dy = (cy - ndcY) * halfH
      const d = dx * dx + dy * dy
      if (d < bestD) {
        bestD = d
        best = i
      }
    }
    if (best < 0) return null
    return [
      this.positions[best * 3],
      this.positions[best * 3 + 1],
      this.positions[best * 3 + 2],
    ]
  }

  /** Пересечение луча курсора с горизонтальной плоскостью Z = z (правка мышью). */
  pointOnPlane(clientX: number, clientY: number, z: number): [number, number] | null {
    const rect = this.canvas.getBoundingClientRect()
    const ndc = new THREE.Vector2(
      ((clientX - rect.left) / rect.width) * 2 - 1,
      -((clientY - rect.top) / rect.height) * 2 + 1,
    )
    const ray = new THREE.Raycaster()
    ray.setFromCamera(ndc, this.camera)
    const plane = new THREE.Plane(new THREE.Vector3(0, 0, 1), -z)
    const hit = new THREE.Vector3()
    if (!ray.ray.intersectPlane(plane, hit)) return null
    return [hit.x, hit.y]
  }

  // ------------------------------------------------------------ разметка (конец)

  /** Показать/скрыть облако точек: режим «только объекты» (смотреть замеры). */
  setPointsVisible(visible: boolean): void {
    this.points.visible = visible
  }

  /** Есть ли что показывать: панель по этому решает, включать ли тумблеры. */
  get objectsShown(): number {
    return this.objectsGroup.visible ? this.objectsGroup.children.length : 0
  }

  /** Габарит показанных объектов -- для вида «по объектам». */
  objectsBounds(): THREE.Box3 | null {
    if (!this.objectsGroup.visible) return null
    const box = new THREE.Box3()
    for (const child of this.objectsGroup.children) box.expandByObject(child)
    return box.isEmpty() ? null : box
  }

  /** Вписать камеру по габариту (вид «по объектам»): тот же приём, что у «fit». */
  frameBox(box: THREE.Box3): void {
    if (box.isEmpty()) return
    const c = box.getCenter(new THREE.Vector3())
    const r = Math.max(box.getSize(new THREE.Vector3()).length() * 0.55, 3)
    this.camera.position.set(c.x + r, c.y + r * 0.8, c.z + r * 0.7)
    this.controls.target.copy(c)
    this.controls.update()
  }

  /** Каркас туннеля: рёбра сечений каждые 2 м. */
  setTunnelFrame(
    input: { line: PathLine; half: number; zLow: number; zHigh: number } | null,
    visible: boolean,
  ): void {
    if (!input || !visible) {
      this.tunnelMesh.visible = false
      return
    }
    const positions = buildTunnelFrame(input)
    this.tunnelMesh.geometry.dispose()
    const geometry = new THREE.BufferGeometry()
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))
    this.tunnelMesh.geometry = geometry
    this.tunnelMesh.visible = positions.length >= 6
  }

  /** Маска «внутри туннеля»; null — не красить (выключено или нет данных). */
  setTunnelMask(mask: TunnelMask | null): void {
    this.tunnelMask = mask
    this.repaintTunnel()
  }

  private repaintTunnel(): void {
    const n = this.count
    const started = performance.now()
    this.colors.set(this.baseColors.subarray(0, n * 3))
    this.tunnelPainted =
      this.tunnelMask && n > 0
        ? paintTunnel(this.positions, n, this.colors, this.tunnelMask)
        : 0
    this.colorAttr.needsUpdate = true
    this.tunnelPaintMs = performance.now() - started
  }

  // ----------------------------------------------------------------- камера

  cameraPreset(name: PresetName): void {
    const cx = this.axisX
    const ymid = -this.dist / 2
    const preset: Record<Exclude<PresetName, 'fit'>, [number[], number[]]> = {
      lidar: [[cx, 2.0, 1.6], [cx, -25, -0.5]],
      top: [[cx, ymid, 70], [cx, ymid, 0]],
      side: [[38, ymid, 0], [cx, ymid, 0]],
      iso: [[30, 22, 22], [cx, ymid, 0]],
      behind: [[cx, 12, 4], [cx, -25, -0.5]],
    }
    if (name === 'fit') {
      const n = this.count || 1
      const box = new THREE.Box3()
      const v = new THREE.Vector3()
      for (let i = 0; i < n; i += Math.max(1, Math.floor(n / 4000))) {
        v.set(this.positions[i * 3], this.positions[i * 3 + 1], this.positions[i * 3 + 2])
        box.expandByPoint(v)
      }
      if (box.isEmpty()) return
      const c = box.getCenter(new THREE.Vector3())
      const r = Math.max(box.getSize(new THREE.Vector3()).length() * 0.55, 3)
      this.camera.position.set(c.x + r, c.y + r * 0.8, c.z + r * 0.7)
      this.controls.target.copy(c)
    } else {
      const [pos, target] = preset[name]
      this.camera.position.set(pos[0], pos[1], pos[2])
      this.controls.target.set(target[0], target[1], target[2])
    }
    this.controls.update()
  }

  // ----------------------------------------------------------- жизнеобеспечение

  resize(): void {
    const w = this.canvas.clientWidth || window.innerWidth
    const h = this.canvas.clientHeight || window.innerHeight
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2))
    this.renderer.setSize(w, h, false)
    this.camera.aspect = w / Math.max(h, 1)
    this.camera.updateProjectionMatrix()
    this.material.uniforms.uPixelRatio.value = Math.min(window.devicePixelRatio || 1, 2)
  }

  render(): void {
    if (this.disposed) return
    this.controls.update()
    this.renderer.render(this.scene, this.camera)
  }

  dispose(): void {
    if (this.disposed) return
    this.disposed = true
    this.controls.dispose()
    this.geometry.dispose()
    this.material.dispose()
    this.gridMesh.geometry.dispose()
    ;(this.gridMesh.material as THREE.Material).dispose()
    this.axesMesh.geometry.dispose()
    ;(this.axesMesh.material as THREE.Material).dispose()
    this.setPathLayer([], 0, false)
    for (const child of this.labelGroup.children) {
      const mesh = child as THREE.Mesh
      mesh.geometry.dispose()
      ;(mesh.material as THREE.Material).dispose()
    }
    this.labelGroup.clear()
    this.tunnelMesh.geometry.dispose()
    ;(this.tunnelMesh.material as THREE.Material).dispose()
    this.renderer.dispose()
    this.renderer.forceContextLoss()
  }

  // ---------------------------------------------------------------- оверлеи

  private makeGrid(): THREE.LineSegments {
    const mesh = new THREE.LineSegments(
      new THREE.BufferGeometry(),
      new THREE.LineBasicMaterial({ color: 0x4a5568, transparent: true, opacity: 0.55 }),
    )
    mesh.renderOrder = 0
    return mesh
  }

  private makeAxes(): THREE.LineSegments {
    const geometry = new THREE.BufferGeometry()
    geometry.setAttribute(
      'position',
      new THREE.Float32BufferAttribute([0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 6], 3),
    )
    geometry.setAttribute(
      'color',
      new THREE.Float32BufferAttribute(
        [1, 0.25, 0.25, 1, 0.25, 0.25, 0.3, 1, 0.35, 0.3, 1, 0.35, 0.4, 0.6, 1, 0.4, 0.6, 1],
        3,
      ),
    )
    return new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({ vertexColors: true }))
  }

  /** Сетка 5 м на плоскости Z = grid.z: ±half_width по X, 0..-length по Y. */
  private rebuildGrid(): void {
    const g = this.grid
    const length = Math.min(this.dist, g.length)
    const halfWidth = Math.min(g.half_width, 40)
    if (length === this.gridLength) return
    this.gridLength = length
    const pts: number[] = []
    for (let x = -halfWidth; x <= halfWidth + 1e-6; x += g.spacing) pts.push(x, -length, g.z, x, 0, g.z)
    for (let y = -length; y <= 1e-6; y += g.spacing) pts.push(-halfWidth, y, g.z, halfWidth, y, g.z)
    this.gridMesh.geometry.dispose()
    const geometry = new THREE.BufferGeometry()
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(pts, 3))
    this.gridMesh.geometry = geometry
  }
}