# Контракт: переписывание вьюера на Bun + TS

Переписываем сервер (сейчас `web_viewer.py` на stdlib) на Bun, а пользовательскую
часть (сейчас `web/app.js` без сборки) — на React + TypeScript + Tailwind + shadcn.
Python-версия остаётся в репозитории как эталон, пока новый код не совпадёт с ней
по числам; удалять её на этой итерации нельзя.

## Раскладка и владение файлами

```
app/
├── CONTRACT.md            # этот файл; агентам только читать
├── server/                # bun-пакет, рантайм bun, зависимостей нет (bun:sqlite встроен)
│   ├── bunfig.toml        # [install] backend = "copyfile" — иначе bun падает с EPERM
│   ├── package.json
│   ├── src/data/**        # ← агент A: порт bag_reader.py + zones.py
│   ├── src/http.ts        # ← агент B: маршруты
│   ├── src/index.ts       # ← агент B: Bun.serve + раздача собранного клиента
│   ├── src/cli.ts         # ← агент B: флаги командной строки
│   ├── src/parity.ts      # ← агент B: сверка с Python-сервером
│   └── test/**            # ← агент A: bun:test (порт tests/)
└── client/                # Vite + React + TS + Tailwind v4 + shadcn
    ├── vite.config.ts     # не трогать: проксирует /meta /frame /set /version на API_URL
    ├── src/index.css      # не трогать: тема shadcn
    ├── src/components/ui/**  # не трогать: сгенерировано shadcn
    ├── src/api/types.ts   # замороженные типы API; агент C может импортировать, не менять смысл
    └── src/**             # ← агент C: вьюер, панель, состояние
```

Эталон, который читаем и с которым сверяемся (не менять!):

- `bag_reader.py` — чтение `.db3`, CDR, кадры, обрезка, раскраски;
- `zones.py` — пол, стены, рельсы, зоны, засорённость коридора;
- `web_viewer.py` — HTTP-сервер и формат ответов;
- `tests/` — юнит-тесты (unittest) для `bag_reader.py` и `zones.py`;
- `web/app.js`, `web/index.html` — текущий клиент (three.js, шейдеры, панель).

## API (тот же самый, что у `web_viewer.py`)

`GET /meta` → JSON:

```
bag, frames, duration_s, hz, max_dist, behind, default_mode, modes, point_size,
max_points_hint, floor: {a, b, rms, grade_pct}, walls: number[], axis_x, gauge_mm,
rails: {x: number[], heights: number[], y: number[], z: number[], blocked: number[]},
blocked_spans: [number, number][], grid: {spacing, half_width, length, z},
zone_names: string[], zone_colors: number[][], blocked_color: number[], rail_color: number[],
train_top, obstacle_points, rail_clearance
```

`GET /frame?idx=N&mode=M` → бинарь, little-endian:

```
смещение 0:  magic 'LZFR'(4 байта) | version u8 (=1) | kind u8 | reserved u16 | n u32   = 12 байт
смещение 12: позиции n*3 float32 (x, y, z — оси лидара: X поперёк, Y вдоль, вперёд −Y, Z вверх)
дальше:      kind 0 (rgb8)  -> n*3 uint8 цвета 0..255
             kind 1 (label8)-> n uint8 метки (индексы в zone_names)
```

`M` из `modes` (по умолчанию `zones`). Для `zones` сервер классифицирует кадр
(`kind 1`), для остальных — красит скалярным полем (`kind 0`).

`GET /set?axis=&gauge=&contact=&offset=&side=` → JSON как `_params_json()` в
`web_viewer.py`: `axis_x, gauge_mm, contact_rail, contact_offset, contact_side,
rail_height, floor: {a, b, rms, grade_pct}, rails: {...}, blocked_spans`. Пол
пересчитывается по полосе вокруг новой оси, засорённость — тоже.

`GET /version` → `{version: string}` — хэш mtime+размера файлов клиента (для
авто-перезагрузки страницы). В дев-режиме клиент отдаёт Vite, так что ручка нужна
только для собранного клиента; если файлов клиента рядом нет — вернуть заглушку,
но не падать.

Коды ответов: 404 на неизвестный путь, 400 + `{error}` на плохие параметры.

## Замороженный интерфейс данных (`app/server/src/data/**`)

Точки везде — интерлив `Float32Array` длиной `3n` (x,y,z), как приходит из bag.
Числа считаем в `number` (float64), как в numpy-версии.

```ts
// data/types.ts — уже создан, менять только с обновлением этого файла
export type ZoneName = 'rail' | 'bed' | 'ceiling' | 'wall' | 'middle'
export interface ZoneParams { floorA: number; floorB: number; floorBand: number;
  ceilingZ: number; axisX: number; gauge: number; railHalfWidth: number;
  railLow: number; railHigh: number; wallX: number[]; wallBand: number;
  detectWalls: boolean; contactRail: boolean; contactSide: 1 | -1;
  contactOffset: number; contactHeight: number; contactHalfWidth: number }

// data/bagReader.ts
export interface Frame { xyz: Float32Array; intensity: Float32Array | null; ring: Uint16Array | null }
export class BagFrames {
  constructor(dbPath: string, opts?: { cacheSize?: number; prefetchAhead?: number })
  get length(): number
  get timestamps(): BigInt64Array      // наносекунды, по возрастанию
  get topicNames(): string[]
  get parseCount(): number             // сколько кадров реально разобрано (для тестов кэша)
  frame(i: number): Frame
  prefetch(i: number): void            // не блокирует; лучшее усилие
  close(): void
}
export function findDb3(bagDir: string): string
export function trimRange(xyz: Float32Array, maxDist: number, behind: number): Uint8Array  // bool маска
export function trim(xyz: Float32Array, intensity: Float32Array | null,
                     ring: Uint16Array | null, maxDist: number, behind: number): [Float32Array, Float32Array | null, Uint16Array | null]
export function frameColors(xyz: Float32Array, intensity: Float32Array | null,
                            mode: string, maxDist: number, ring?: Uint16Array | null): Float64Array  // n*3, 0..1
export function parsePointcloud2Cdr(data: Uint8Array, withRing?: boolean):
  { xyz: Float32Array; intensity: Float32Array | null; ring: Uint16Array | null }

// data/zones.ts
export const ZONE_NAMES: readonly ZoneName[]
export const ZONE_COLORS: Record<ZoneName, [number, number, number]>   // 0..1, как в zones.py
export const BLOCKED_COLOR: [number, number, number]
export const RAIL_HEIGHT: number          // 0.16
export const TRAIN_TOP: number            // 3.7
export const MIN_OBSTACLE_HEIGHT: number  // 0.5
export function fitFloor(xyz: Float32Array, opts?: { yMin?: number; yMax?: number; bins?: number;
  percentile?: number; clip?: number; xCenter?: number; xHalf?: number }): { a: number; b: number; rms: number }
export function detectWalls(xyz: Float32Array, p: ZoneParams, minSeparation?: number, binWidth?: number): number[]
export function railXPositions(p: ZoneParams): Array<[number, number]>   // [x, высота над полом]
export function classify(xyz: Float32Array, p: ZoneParams): Uint8Array
export function classifyCounts(xyz: Float32Array, p: ZoneParams): Record<ZoneName, number>
export function corridorBlocked(xyz: Float32Array, p: ZoneParams, yFrom: number, yTo: number,
  opts?: { binWidth?: number; clearance?: number; trainTop?: number; minPoints?: number;
           minHeight?: number }): { ys: Float64Array; blocked: Uint8Array }
export function blockedSpans(ys: Float64Array, blocked: Uint8Array, binWidth?: number): Array<[number, number]>
export function findAxis(xyz: Float32Array, p: ZoneParams, yFrom: number, yTo: number,
  opts?: { clearance?: number; minPoints?: number; binWidth?: number; axisStep?: number; binX?: number }): number
```

Поведение обязано совпадать с Python, включая тонкости:

- `classify` — приоритет `rail → bed → ceiling → wall → middle` (старшие маски
  затирают младшие), пороги считаются от плоскости пола `floorA + floorB*y`;
- `corridorBlocked` — бин занят, если точек ≥ `minPoints` **и** их размах по высоте
  (перцентили 5..95) ≥ `minHeight` (по умолчанию `MIN_OBSTACLE_HEIGHT`); при
  `minHeight = 0` — прежнее поведение по одному счёту;
- `fitFloor` — 5-й перцентиль по Y-бинам, МНК, отбраковка выбросов по sigma (3 итерации);
- `findAxis` — максимум свободной длины коридора, при равенстве центр самого
  длинного плато; кандидаты только внутри найденных стен.

## Проверяемость (то, что должно быть зелёным)

1. `cd app/server && bun test` — портированные юнит-тесты (`tests/test_core.py`,
   `tests/test_zones.py`) проходят на TS-реализации.
2. `cd app/server && bun run src/parity.ts --bag ../for_hackathon/doubleT_obstacle`
   — сверка с Python: `meta` (floor a/b, walls, axis_x, blocked_spans), кадры
   (число точек, гистограмма меток, контрольная сумма позиций) совпадают в
   пределах 1e-6 для float и точно для счётчиков/участков.
3. `cd app/client && bun run build` — сборка без ошибок TS.
4. Сквозная: Bun-сервер поднят, страница в headless Chromium грузится без ошибок
   в консоли, HUD показывает то же число точек, что Python-сервер.

## Окружение (важно)

- bun 1.3.14, Windows. `bun install` иногда падает с `EPERM: NtSetInformationFile`
  — лечится повтором и `bunfig.toml` с `[install] backend = "copyfile"` (уже создан
  в `app/client`, добавь такой же в `app/server`, если будешь что-то ставить).
- Python для сверки: `C:\Users\K4ler\AppData\Local\Programs\Python\Python312\python.exe`
  (там numpy + open3d). Тесты Python: `python -m unittest discover -s tests -t .` из
  корня репозитория.
- Данные: `for_hackathon/doubleT_obstacle/doubleT_obstacle_0.db3` — 201 кадр,
  346 808 точек/кадр, ~10 Гц, 20.4 с. Есть ещё 5 bag-ов рядом.
- Порты: Python-вьюер 8765/8766 (может быть поднят или нет), Bun API — 8788,
  Vite — 5173. Клиент ходит на относительные пути, в дев-режиме Vite проксирует
  их на API (`API_URL` в `vite.config.ts`); можно направить на Python-сервер той
  же ручкой и разрабатывать клиент без Bun-сервера.
- Оси лидара: X — поперёк, Y — вдоль (вперёд это −Y), Z — вверх. Цвета на экране
  должны совпадать с палитрой (клиент пишет цвета как есть, без тонмаппинга).
- Язык: комментарии и тексты интерфейса — русские, как в существующем коде.

## Чего не делать

- Не удалять и не править Python-файлы, `tests/`, `web/` — они эталон.
- Не менять `app/CONTRACT.md`, `app/client/vite.config.ts`, `app/client/tsconfig*.json`,
  `app/client/src/index.css`, `app/client/components.json`, `app/client/src/components/ui/**`,
  `app/client/src/lib/utils.ts`.
- Не добавлять зависимости без нужды; сервер обходится встроенными модулями Bun.