// Миникарта: разрез X-Z по точкам первых 12 м вперёд, правый нижний угол.
// Рисуется из буферов сцены (без дополнительных запросов к серверу) и только
// когда сменился кадр — как в web/app.js.

import type { Meta } from '@/api/types'
import { SECTION_H, SECTION_W, SECTION_WINDOW, rgbCss, zoneColor } from '@/viewer/constants'

/** Источник точек: буферы текущего кадра из ViewerScene. */
export interface SectionSource {
  positions: Float32Array
  colors: Uint8Array
  count: number
}

function roundRect(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number) {
  ctx.beginPath()
  ctx.moveTo(x + r, y)
  ctx.arcTo(x + w, y, x + w, y + h, r)
  ctx.arcTo(x + w, y + h, x, y + h, r)
  ctx.arcTo(x, y + h, x, y, r)
  ctx.arcTo(x, y, x + w, y, r)
  ctx.closePath()
}

export function drawSection(canvas: HTMLCanvasElement, meta: Meta, src: SectionSource): void {
  const dpr = Math.min(window.devicePixelRatio || 1, 2)
  const iw = Math.round(SECTION_W * dpr)
  const ih = Math.round(SECTION_H * dpr)
  if (canvas.width !== iw || canvas.height !== ih) {
    canvas.width = iw
    canvas.height = ih
  }
  canvas.style.width = `${SECTION_W}px`
  canvas.style.height = `${SECTION_H}px`

  const ctx = canvas.getContext('2d')
  if (!ctx) return

  const W = SECTION_W
  const H = SECTION_H
  const padL = 36
  const padR = 10
  const padT = 34
  const padB = 22
  const plotX = padL
  const plotY = padT
  const plotW = W - padL - padR
  const plotH = H - padT - padB
  const yWin = -SECTION_WINDOW
  const yc = -SECTION_WINDOW / 2
  const wallColor = rgbCss(zoneColor(meta, 'wall'))
  const floorZ = meta.floor.a + meta.floor.b * yc

  // 1) диапазон X/Z: точки окна + все оверлеи, чтобы ничего не обрезалось
  const { positions, colors, count } = src
  let xmin = Infinity
  let xmax = -Infinity
  let zmin = Infinity
  let zmax = -Infinity
  for (let i = 0; i < count; i++) {
    const j = i * 3
    if (positions[j + 1] < yWin) continue
    const x = positions[j]
    const z = positions[j + 2]
    if (x < xmin) xmin = x
    if (x > xmax) xmax = x
    if (z < zmin) zmin = z
    if (z > zmax) zmax = z
  }
  const eatX = (v: number) => {
    if (!Number.isFinite(v)) return
    if (v < xmin) xmin = v
    if (v > xmax) xmax = v
  }
  eatX(meta.axis_x)
  for (const x of meta.walls) eatX(x)
  for (const x of meta.rails.x) eatX(x)
  if (Number.isFinite(floorZ)) {
    if (floorZ < zmin) zmin = floorZ
    if (floorZ > zmax) zmax = floorZ
  }
  if (!(xmin <= xmax)) {
    xmin = -2
    xmax = 2
  }
  if (!(zmin <= zmax)) {
    zmin = -1
    zmax = 1
  }
  const mrg = 0.6
  xmin -= mrg
  xmax += mrg
  zmin -= mrg
  zmax += mrg
  let dx = xmax - xmin
  let dz = zmax - zmin
  if (!(dx > 0)) dx = 1
  if (!(dz > 0)) dz = 1
  const sc = Math.min(plotW / dx, plotH / dz) // равный масштаб по X и Z
  const xc = (xmin + xmax) / 2
  const zc = (zmin + zmax) / 2
  const mapX = (x: number) => plotX + plotW / 2 + (x - xc) * sc
  const mapY = (z: number) => plotY + plotH / 2 - (z - zc) * sc

  // 2) точки: пишем пиксели через ImageData, без строк на каждую точку
  const img = ctx.createImageData(iw, ih)
  const buf32 = new Uint32Array(img.data.buffer)
  buf32.fill(0xff180f0c) // фон панели (ABGR)
  for (let i = 0; i < count; i++) {
    const j = i * 3
    if (positions[j + 1] < yWin) continue
    const px = Math.round(mapX(positions[j]) * dpr)
    const py = Math.round(mapY(positions[j + 2]) * dpr)
    if (px < 0 || py < 0 || px >= iw || py >= ih) continue
    buf32[py * iw + px] = 0xff000000 | (colors[j + 2] << 16) | (colors[j + 1] << 8) | colors[j]
  }
  ctx.putImageData(img, 0, 0)
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)

  // 3) светлая сетка 1 м и подписи в метрах
  const step = sc >= 20 ? 1 : 2
  ctx.lineWidth = 1
  ctx.strokeStyle = 'rgba(120,140,170,0.18)'
  ctx.beginPath()
  for (let x = Math.ceil(xmin / step) * step; x <= xmax; x += step) {
    const px = Math.round(mapX(x)) + 0.5
    ctx.moveTo(px, plotY)
    ctx.lineTo(px, plotY + plotH)
  }
  for (let z = Math.ceil(zmin / step) * step; z <= zmax; z += step) {
    const py = Math.round(mapY(z)) + 0.5
    ctx.moveTo(plotX, py)
    ctx.lineTo(plotX + plotW, py)
  }
  ctx.stroke()

  ctx.fillStyle = 'rgba(138,151,176,0.9)'
  ctx.font = '9px "Segoe UI", system-ui, sans-serif'
  ctx.textAlign = 'center'
  ctx.textBaseline = 'top'
  for (let x = Math.ceil(xmin / step) * step; x <= xmax; x += step) {
    ctx.fillText(String(x), mapX(x), plotY + plotH + 3)
  }
  ctx.textAlign = 'right'
  ctx.textBaseline = 'middle'
  for (let z = Math.ceil(zmin / step) * step; z <= zmax; z += step) {
    ctx.fillText(String(z), plotX - 4, mapY(z))
  }

  // 4) оверлеи
  const fy = mapY(floorZ)
  ctx.setLineDash([])
  ctx.strokeStyle = 'rgb(130,150,180)' // пол: z = a + b*y в центре окна
  ctx.lineWidth = 1.5
  ctx.beginPath()
  ctx.moveTo(plotX, fy)
  ctx.lineTo(plotX + plotW, fy)
  ctx.stroke()

  ctx.strokeStyle = wallColor // стены
  ctx.setLineDash([5, 4])
  ctx.lineWidth = 1.25
  ctx.beginPath()
  for (const x of meta.walls) {
    const px = mapX(x)
    if (px < plotX || px > plotX + plotW) continue
    ctx.moveTo(px, plotY)
    ctx.lineTo(px, plotY + plotH)
  }
  ctx.stroke()

  ctx.strokeStyle = '#7fb2ff' // ось пути
  ctx.setLineDash([2, 3])
  ctx.lineWidth = 1.25
  ctx.beginPath()
  const axp = mapX(meta.axis_x)
  if (axp >= plotX && axp <= plotX + plotW) {
    ctx.moveTo(axp, plotY)
    ctx.lineTo(axp, plotY + plotH)
  }
  ctx.stroke()
  ctx.setLineDash([])

  ctx.strokeStyle = rgbCss(meta.rail_color) // рельсы: засечки у линии пола
  ctx.lineWidth = 2.5
  ctx.beginPath()
  for (const x of meta.rails.x) {
    const px = mapX(x)
    if (px < plotX || px > plotX + plotW) continue
    ctx.moveTo(px, mapY(floorZ + 0.7))
    ctx.lineTo(px, mapY(floorZ - 0.15))
  }
  ctx.stroke()

  // 5) заголовок и сводка
  ctx.textAlign = 'left'
  ctx.textBaseline = 'alphabetic'
  ctx.fillStyle = '#dde4f0'
  ctx.font = '600 12px "Segoe UI", system-ui, sans-serif'
  ctx.fillText(`Разрез X-Z (первые ${SECTION_WINDOW} м)`, 10, 15)
  ctx.fillStyle = 'rgba(138,151,176,0.95)'
  ctx.font = '10px "Segoe UI", system-ui, sans-serif'
  ctx.fillText(
    `пол z = ${meta.floor.a} + ${meta.floor.b}·y , ось x = ${meta.axis_x} м , колея ${meta.gauge_mm} мм`,
    10,
    28,
  )

  // 6) мини-легенда внутри панели
  const lx = plotX + plotW - 104
  const ly = plotY + 6
  ctx.fillStyle = 'rgba(5,7,12,0.72)'
  ctx.strokeStyle = 'rgba(140,160,200,0.25)'
  ctx.lineWidth = 1
  roundRect(ctx, lx, ly, 98, 60, 6)
  ctx.fill()
  ctx.stroke()
  ctx.font = '10px "Segoe UI", system-ui, sans-serif'
  ctx.textAlign = 'left'
  ctx.textBaseline = 'middle'
  const legendRow = (row: number, color: string, dash: number[], text: string) => {
    const y = ly + 12 + row * 12
    ctx.strokeStyle = color
    ctx.lineWidth = 2
    ctx.setLineDash(dash)
    ctx.beginPath()
    ctx.moveTo(lx + 8, y)
    ctx.lineTo(lx + 26, y)
    ctx.stroke()
    ctx.setLineDash([])
    ctx.fillStyle = '#dde4f0'
    ctx.fillText(text, lx + 32, y)
  }
  legendRow(0, 'rgb(130,150,180)', [], 'пол')
  legendRow(1, rgbCss(meta.rail_color), [], 'рельсы')
  legendRow(2, wallColor, [4, 3], 'стены')
  legendRow(3, '#7fb2ff', [2, 3], 'ось')
}