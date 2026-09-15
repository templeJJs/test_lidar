/**
 * Разбор флагов командной строки Bun-вьюера: зеркало main() из web_viewer.py.
 *
 * Свой разбор вместо парсер-зависимости: флагов немного, а имена, значения по
 * умолчанию и поведение при ошибке должны совпадать с argparse-версией
 * (позиционный bag_dir, значение у каждого флага, --flag=value тоже принимаем).
 */

import { existsSync } from 'node:fs'
import { resolve } from 'node:path'

import { MIN_OBSTACLE_HEIGHT } from './data/zones.ts'

export const DEFAULT_BAG_DIR = 'for_hackathon/doubleT_obstacle'
export const DEFAULT_CLIENT_DIR = 'app/client/dist'

/** Корень репозитория: app/server/src -> три уровня вверх. */
export const REPO_ROOT = resolve(import.meta.dir, '../../..')

/**
 * Путь из флага: пробуем несколько баз, пока файл не найдётся.
 *
 * Значения по умолчанию (bag_dir, --client-dir) заданы от корня репозитория, как у
 * web_viewer.py, а bun-скрипт обычно запускают из app/server -- первый кандидат
 * покрывает обычные программы, второй -- дефолты. Третий нужен для документированного
 * вызова `bun run src/parity.ts --bag ../for_hackathon/doubleT_obstacle`: из
 * app/server лишний ".." уводит в app/for_hackathon, поэтому пробуем и без него.
 */
export function resolveRepoPath(pathOrDir: string): string {
  const candidates = [
    resolve(pathOrDir),
    resolve(REPO_ROOT, pathOrDir),
    resolve(REPO_ROOT, pathOrDir.replace(/^([.][.][/\\])+/, '')),
  ]
  for (const candidate of candidates) {
    if (existsSync(candidate)) return candidate
  }
  return candidates[0] as string
}

export interface CliArgs {
  bagDir: string
  host: string
  port: number
  mode: string
  maxDist: number
  behind: number
  maxPoints: number
  pointSize: number
  /** null -- ось ищем сами (как args.axis_x is None в Python) */
  axisX: number | null
  gaugeMm: number
  obstaclePoints: number
  obstacleHeight: number
  railClearance: number
  open: boolean
  clientDir: string
}

type FlagKind = 'str' | 'int' | 'float' | 'floatOrNull' | 'bool'

interface FlagSpec {
  flag: string
  key: keyof CliArgs
  kind: FlagKind
  help: string
}

/** Флаги в порядке, в котором их печатает --help. */
const FLAGS: readonly FlagSpec[] = [
  { flag: '--host', key: 'host', kind: 'str', help: 'адрес прослушивания' },
  { flag: '--port', key: 'port', kind: 'int', help: 'порт HTTP' },
  { flag: '--mode', key: 'mode', kind: 'str', help: 'начальная раскраска' },
  { flag: '--max-dist', key: 'maxDist', kind: 'float', help: 'дальность вперёд по -Y, м' },
  { flag: '--behind', key: 'behind', kind: 'float', help: 'обрезка позади, м' },
  { flag: '--max-points', key: 'maxPoints', kind: 'int', help: 'предел точек на кадр (размер ответа)' },
  { flag: '--point-size', key: 'pointSize', kind: 'float', help: 'размер точки в браузере' },
  { flag: '--axis-x', key: 'axisX', kind: 'floatOrNull', help: 'ось пути, м; не задана -- ищем по кадрам' },
  { flag: '--gauge-mm', key: 'gaugeMm', kind: 'float', help: 'колея, мм (0 -- из замера)' },
  { flag: '--obstacle-points', key: 'obstaclePoints', kind: 'int', help: 'минимум точек в бине засорённости' },
  { flag: '--obstacle-height', key: 'obstacleHeight', kind: 'float', help: 'минимальная высота структуры в бине, м; 0 -- считать по одним точкам' },
  { flag: '--rail-clearance', key: 'railClearance', kind: 'float', help: 'зазор коридора за габарит поезда, м' },
  { flag: '--open', key: 'open', kind: 'bool', help: 'открыть страницу в браузере' },
  { flag: '--client-dir', key: 'clientDir', kind: 'str', help: 'каталог собранного клиента (index.html + assets/)' },
]

const BY_FLAG = new Map(FLAGS.map((spec) => [spec.flag, spec]))

export class CliError extends Error {}

export function defaultArgs(): CliArgs {
  return {
    bagDir: DEFAULT_BAG_DIR,
    host: '127.0.0.1',
    port: 8788,
    mode: 'zones',
    maxDist: 40,
    behind: 5,
    maxPoints: 400000,
    pointSize: 3,
    axisX: null,
    gaugeMm: 0,
    obstaclePoints: 20,
    obstacleHeight: MIN_OBSTACLE_HEIGHT,
    railClearance: 0.3,
    open: false,
    clientDir: DEFAULT_CLIENT_DIR,
  }
}

export function usage(): string {
  const rows = FLAGS.map((spec) => ({
    spec,
    arg: spec.flag + (spec.kind === 'bool' ? '' : ' ЗНАЧЕНИЕ'),
  }))
  const width = Math.max(...rows.map((row) => row.arg.length), '-h, --help'.length) + 2
  const lines = [
    'three.js web viewer for rosbag2 .db3 point clouds',
    '',
    'Использование: bun run src/index.ts [bag_dir] [флаги]',
    '',
    'Позиционные аргументы:',
    `  ${'bag_dir'.padEnd(width)}каталог с bag (по умолчанию ${DEFAULT_BAG_DIR})`,
    '',
    'Флаги:',
  ]
  const defaults = defaultArgs() as unknown as Record<string, unknown>
  for (const { spec, arg } of rows) {
    const def = defaults[spec.key]
    const shown = def === null ? 'не задано' : String(def)
    lines.push(`  ${arg.padEnd(width)}${spec.help} (по умолчанию: ${shown})`)
  }
  lines.push(`  ${'-h, --help'.padEnd(width)}этот текст`)
  return lines.join('\n')
}

function floatValue(flag: string, raw: string): number {
  const value = Number(raw)
  if (raw.trim() === '' || !Number.isFinite(value)) {
    throw new CliError(`флаг ${flag}: ожидалось число, получено ${JSON.stringify(raw)}`)
  }
  return value
}

function intValue(flag: string, raw: string): number {
  const value = floatValue(flag, raw)
  if (!Number.isInteger(value)) {
    throw new CliError(`флаг ${flag}: ожидалось целое число, получено ${JSON.stringify(raw)}`)
  }
  return value
}

/** Разбор argv без имени рантайма и скрипта. Бросает CliError на плохой ввод. */
export function parseArgs(argv: readonly string[]): CliArgs {
  const args = defaultArgs()
  const record = args as unknown as Record<string, string | number | boolean | null>
  let bagDir: string | null = null

  for (let i = 0; i < argv.length; i++) {
    const token = argv[i] as string
    if (!token.startsWith('-') || token === '-') {
      if (bagDir !== null) throw new CliError(`лишний позиционный аргумент: ${JSON.stringify(token)}`)
      bagDir = token
      continue
    }
    if (token === '-h' || token === '--help') throw new CliError('__help__')

    const eq = token.indexOf('=')
    const name = eq >= 0 ? token.slice(0, eq) : token
    const spec = BY_FLAG.get(name)
    if (spec === undefined) throw new CliError(`неизвестный флаг: ${name}`)

    if (spec.kind === 'bool') {
      if (eq >= 0) throw new CliError(`флаг ${name} не принимает значение`)
      record[spec.key] = true
      continue
    }

    let raw: string
    if (eq >= 0) {
      raw = token.slice(eq + 1)
    } else {
      const next = argv[++i]
      if (next === undefined) throw new CliError(`флаг ${name} требует значение`)
      raw = next
    }

    switch (spec.kind) {
      case 'str':
        record[spec.key] = raw
        break
      case 'int':
        record[spec.key] = intValue(name, raw)
        break
      case 'float':
      case 'floatOrNull':
        record[spec.key] = floatValue(name, raw)
        break
      default:
        break
    }
  }

  if (bagDir !== null) args.bagDir = bagDir
  return args
}

/** То же, но с печатью подсказки и выходом: код 0 на --help, 2 на ошибке (как argparse). */
export function parseArgsOrExit(argv: readonly string[] = Bun.argv.slice(2)): CliArgs {
  try {
    return parseArgs(argv)
  } catch (err) {
    if (err instanceof CliError && err.message === '__help__') {
      console.log(usage())
      process.exit(0)
    }
    const message = err instanceof Error ? err.message : String(err)
    console.error(`${message}\n`)
    console.error(usage())
    process.exit(2)
  }
}