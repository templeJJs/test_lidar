# Как устроен `app/`

Разделение ровно такое: **Python считает, браузер рисует, bun раздаёт.**

```
Python: web_viewer.py + bag_reader.py + zones.py      ← единственная реализация вычислений
   читает .db3 (SQLite), разбирает CDR PointCloud2, считает пол/стены/зону/ось,
   засорённость коридора; отдаёт /meta (JSON) и /frame (бинарный кадр)
        ▲
        │ /meta, /frame, /set, /version
        │
bun: app/server/src/index.ts                          ← тонкий слой, кода ~120 строк
   раздаёт собранный клиент (app/client/dist) и проксирует те же ручки на Python,
   чтобы у страницы и API был один origin (иначе нужен CORS в Python)
        ▲
        │ относительные пути /meta, /frame, /set
        │
React: app/client                                     ← прослойка для отрисовки
   three.js: декодирует бинарный кадр, метки → цвета палитры, WebGL;
   камера, панель на shadcn, легенда, разрез X–Z. Никакой математики.
```

Вычисления намеренно **не** переписаны на TypeScript: дублировать проверенный
numpy-код на другом языке — это работа без пользы и риск разойтись с эталоном.
Формат `/meta` и `/frame` описан в докстроке `web_viewer.py` и в
`app/client/src/api/types.ts`; клиент от Python-версии не зависит — ему важен
только этот формат.

## Запуск

```bash
# 1) сервер с вычислениями (Python 3.12: нужен numpy; open3d уже не нужен)
python web_viewer.py for_hackathon/doubleT_obstacle --port 8765

# 2a) дев-режим клиента: Vite с HMR, проксирует API на Python
cd app/client && bun run dev            # http://127.0.0.1:5173

# 2b) прод: собрать клиент и раздать через bun (проксирует API на Python)
cd app/client && bun run build
cd app/server && bun run start          # http://127.0.0.1:8788
```

Флаги bun-сервера: `--host`, `--port` (8788), `--py` (адрес Python-вьюера,
по умолчанию `http://127.0.0.1:8765`), `--client` (каталог собранного клиента,
по умолчанию `app/client/dist`).

## Что в `app/client`

- `src/viewer/` — three.js: шейдер точек (размер от расстояния, обрезка по
  дальности, цвета как есть), сетка и оси, пресеты камеры, разрез X–Z;
- `src/state/` — метаданные, воспроизведение с предзагрузкой, правка модели
  рельсов через `/set` (дебаунс), горячие клавиши;
- `src/panel/` — панель и HUD на shadcn (`src/components/ui/**` сгенерированы
  CLI shadcn, руками не правятся);
- `src/api/` — запросы и разбор бинарного кадра, типы ответов.

Линий рельсов в 3D нет: модель рельсов влияет только на раскраску зон и на
засорённость коридора.

## Окружение

- bun 1.3.14, Windows. `bun install` иногда падает с `EPERM: NtSetInformationFile` —
  помогает повтор и `[install] backend = "copyfile"` в `bunfig.toml` (уже прописан).
- Порты по умолчанию: Python 8765, bun 8788, Vite 5173.