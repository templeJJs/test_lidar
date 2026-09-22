// Клиент HTTP-ручек вьюера. Формат ответов описан в app/CONTRACT.md и совпадает
// с web_viewer.py, поэтому клиент ходит на относительные пути: в дев-режиме их
// проксирует Vite, в собранном виде отдаёт Python-сервер.

import {
  FRAME_HEADER_BYTES,
  FRAME_KIND_LABEL8,
  FRAME_KIND_RGB8,
  type DecodedFrame,
  type Meta,
  type ParamsResponse,
} from '@/api/types'

const MAGIC = [0x4c, 0x5a, 0x46, 0x52] // 'LZFR'
const FRAME_VERSION = 1

/** Ошибка обращения к вьюеру: текст показывается в панели ошибок. */
export class ApiError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'ApiError'
  }
}

async function getJson<T>(url: string, signal?: AbortSignal): Promise<T> {
  const res = await fetch(url, { cache: 'no-store', signal })
  if (!res.ok) throw new ApiError(`GET ${url} → ${res.status}`)
  return (await res.json()) as T
}

export function fetchMeta(signal?: AbortSignal): Promise<Meta> {
  return getJson<Meta>('/meta', signal)
}

export function fetchFrame(idx: number, mode: string, signal?: AbortSignal): Promise<ArrayBuffer> {
  const url = `/frame?idx=${idx}&mode=${encodeURIComponent(mode)}`
  return fetch(url, { cache: 'no-store', signal }).then((res) => {
    if (!res.ok) throw new ApiError(`GET ${url} → ${res.status}`)
    return res.arrayBuffer()
  })
}

/** Правки модели рельсов — те же параметры, что писал старый клиент. */
export interface RailParams {
  axis: number
  gaugeMm: number
  contact: boolean
  offset: number
  side: 1 | -1
}

export function railQuery(p: RailParams): string {
  return new URLSearchParams({
    axis: String(p.axis),
    gauge: String(p.gaugeMm),
    contact: p.contact ? '1' : '0',
    offset: String(p.offset),
    side: p.side < 0 ? '-1' : '1',
  }).toString()
}

export function applyRailParams(p: RailParams): Promise<ParamsResponse> {
  return getJson<ParamsResponse>(`/set?${railQuery(p)}`)
}

/**
 * Правки туннеля безопасности. Сервер, который этих ключей ещё не знает,
 * молча их игнорирует: в ответе не будет `tunnel`/`path`, и клиент оставит
 * прежние значения из /meta (пересечение делает `mergeSetResponse`).
 */
export interface TunnelParams {
  half: number
  low: number
  high: number
  contact: boolean
}

export function tunnelQuery(p: TunnelParams): string {
  return new URLSearchParams({
    tunnel_half: String(p.half),
    tunnel_low: String(p.low),
    tunnel_high: String(p.high),
    tunnel_contact: p.contact ? '1' : '0',
  }).toString()
}

export function applyTunnelParams(p: TunnelParams): Promise<ParamsResponse> {
  return getJson<ParamsResponse>(`/set?${tunnelQuery(p)}`)
}

/**
 * Разбор бинарного кадра: заголовок + позиции + полезная нагрузка.
 * Массивы — представления поверх того же буфера, без копирования: дальше
 * сцена копирует их в свои буферы один раз.
 */
export function decodeFrame(buffer: ArrayBuffer): DecodedFrame {
  if (buffer.byteLength < FRAME_HEADER_BYTES) throw new ApiError('кадр короче заголовка')
  const bytes = new Uint8Array(buffer)
  for (let i = 0; i < MAGIC.length; i++) {
    if (bytes[i] !== MAGIC[i]) throw new ApiError('неверная сигнатура кадра')
  }
  const view = new DataView(buffer)
  const version = view.getUint8(4)
  if (version !== FRAME_VERSION) throw new ApiError(`версия кадра ${version}, ожидалась ${FRAME_VERSION}`)
  const kind = view.getUint8(5)
  const count = view.getUint32(8, true)
  const payloadOffset = FRAME_HEADER_BYTES + count * 12
  const payloadBytes = kind === FRAME_KIND_LABEL8 ? count : count * 3
  if (buffer.byteLength < payloadOffset + payloadBytes) {
    throw new ApiError(`кадр обрезан: ${count} точек, ${buffer.byteLength} байт`)
  }
  const positions = new Float32Array(buffer, FRAME_HEADER_BYTES, count * 3)
  if (kind === FRAME_KIND_RGB8) {
    return { kind, count, positions, colors: new Uint8Array(buffer, payloadOffset, count * 3) }
  }
  if (kind === FRAME_KIND_LABEL8) {
    return { kind, count, positions, labels: new Uint8Array(buffer, payloadOffset, count) }
  }
  throw new ApiError(`неизвестный тип полезной нагрузки ${kind}`)
}

/** Текст ошибки для показа пользователю. */
export function describeError(err: unknown): string {
  if (err instanceof Error) return err.message
  return String(err)
}