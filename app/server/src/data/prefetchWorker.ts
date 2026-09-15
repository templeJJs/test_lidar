// Фоновая предзагрузка кадров для `BagFrames` (порт потока `_prefetch_loop`
// из bag_reader.py).
//
// Живёт в отдельном Worker: sqlite и разбор CDR синхронные, поэтому на главном
// потоке они блокировали бы HTTP-запросы. Здесь своё соединение с базой, а
// готовые кадры уезжают наверх передачей буферов (без копирования).

import { Database } from 'bun:sqlite'

import { parsePointcloud2Cdr } from './bagReader.ts'
import type { PrefetchCommand, PrefetchFrameMessage } from './bagReader.ts'

type WorkerContext = {
  onmessage: ((event: MessageEvent) => void) | null
  // Transferable из DOM-библиотеки тут недоступен (lib: ESNext), а передаём мы
  // ровно буферы типизированных массивов.
  postMessage(message: unknown, transfer?: ArrayBuffer[]): void
}

const ctx = globalThis as unknown as WorkerContext

let dbPath = ''
let rowids: number[] = []
let ahead = 0
let want = 0
let stopped = false
let running = false

// Уже отправленные кадры: свой кэш воркер не видит, а переразбирать одни и те
// же кадры по кругу -- лишняя работа и лишние сообщения.
const sent = new Set<number>()

ctx.onmessage = (event: MessageEvent) => {
  const command = event.data as PrefetchCommand | null
  if (command === null || typeof command !== 'object') return
  switch (command.kind) {
    case 'stop':
      stopped = true
      return
    case 'want':
      want = Math.trunc(command.index)
      return
    case 'init':
      dbPath = command.dbPath
      rowids = command.rowids
      ahead = command.ahead
      want = command.want
      if (!running) {
        running = true
        void loop()
      }
      return
  }
}

async function loop(): Promise<void> {
  const db = new Database(dbPath, { readonly: true, safeIntegers: true })
  while (!stopped) {
    const start = want
    const end = Math.min(start + 1 + ahead, rowids.length)
    for (let index = start + 1; index < end; index++) {
      if (stopped) return
      if (sent.has(index)) continue
      let frame: ReturnType<typeof parsePointcloud2Cdr>
      try {
        const row = db
          .query<{ data: Uint8Array }, [number]>('SELECT data FROM messages WHERE rowid=?')
          .get(rowids[index]!)
        if (row === null || row === undefined) return
        frame = parsePointcloud2Cdr(row.data, true)
      } catch {
        return
      }
      sent.add(index)
      const message: PrefetchFrameMessage = {
        index,
        xyz: frame.xyz,
        intensity: frame.intensity,
        ring: frame.ring,
      }
      const transfer: ArrayBuffer[] = [frame.xyz.buffer as ArrayBuffer]
      if (frame.intensity !== null) transfer.push(frame.intensity.buffer as ArrayBuffer)
      if (frame.ring !== null) transfer.push(frame.ring.buffer as ArrayBuffer)
      ctx.postMessage(message, transfer)
    }
    await Bun.sleep(20)
  }
}