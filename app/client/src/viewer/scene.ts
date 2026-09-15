// Сцена three.js вьюера: облако точек, статические оверлеи, камера.
//
// Поведение перенесено из web/app.js: тот же шейдер точек, те же пресеты камеры,
// та же сетка/оси. Отличие одно: линии рельсов в 3D не рисуются (оверлей лент
// удалён по требованию заказчика) — модель рельсов влияет только на раскраску зон.

import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { FRAME_KIND_LABEL8, FRAME_KIND_RGB8, type DecodedFrame, type Meta } from '@/api/types'
import { BACKGROUND, FALLBACK_COLOR, type PresetName } from '@/viewer/constants'

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

export class ViewerScene {
  /** Буферы, которые читает миникарта-разрез. */
  readonly positions: Float32Array
  readonly colors: Uint8Array
  /** Сколько точек кадра сейчас в буферах. */
  count = 0

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
  private gridMesh: THREE.LineSegments
  private gridLength: number
  private readonly axesMesh: THREE.LineSegments
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

    this.gridMesh = this.makeGrid()
    this.gridMesh.visible = opts.gridVisible
    this.scene.add(this.gridMesh)

    this.axesMesh = this.makeAxes()
    this.axesMesh.visible = opts.axesVisible
    this.scene.add(this.axesMesh)
  }

  // ------------------------------------------------------------------ кадры

  /**
   * Копирует разобранный кадр в буферы сцены. Цвета для kind 1 берём из палитры
   * зон: сервер присылает метки, а не цвета.
   */
  applyFrame(frame: DecodedFrame, zoneColors: readonly number[][]): AppliedFrame {
    const fits = Math.min(frame.count, this.capacity)
    this.positions.set(frame.positions.subarray(0, fits * 3))
    let counts: number[] | null = null

    if (frame.kind === FRAME_KIND_RGB8) {
      const src = frame.colors
      if (!src) throw new Error('кадр rgb8 без цветовой полезной нагрузки')
      this.colors.set(src.subarray(0, fits * 3))
    } else if (frame.kind === FRAME_KIND_LABEL8) {
      const labels = frame.labels
      if (!labels) throw new Error('кадр label8 без меток')
      for (let i = 0, j = 0; i < fits; i++) {
        const c = zoneColors[labels[i]] ?? FALLBACK_COLOR
        this.colors[j++] = c[0]
        this.colors[j++] = c[1]
        this.colors[j++] = c[2]
      }
      counts = new Array<number>(zoneColors.length).fill(0)
      for (let i = 0; i < frame.count; i++) counts[labels[i]]++
    } else {
      throw new Error(`неизвестный тип раскраски ${frame.kind}`)
    }

    this.count = fits
    this.positionAttr.needsUpdate = true
    this.colorAttr.needsUpdate = true
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