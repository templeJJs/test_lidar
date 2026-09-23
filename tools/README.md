# tools/ — пайплайн ассетов для разметки

Скачивание 3D-ассетов с фиксацией лицензий, проверка «а чем реально читается» и
запекание позы из GLB/FBX в массив вершин, который читает Python.

Живые числа по каждому ассету — в `assets/manifest.json` (генерируется
`check_assets.py --write-manifest`), лицензии — в `assets/LICENSES.md`.
Бинарники ассетов лежат в `assets/` и **в git не попадают** (`.gitignore`).

## Быстрый старт

```bash
PY=/c/Users/K4ler/AppData/Local/Programs/Python/Python312/python.exe   # там open3d 0.19

# 1. скачать и распаковать (идемпотентно; при повторном запуске ничего не качает)
PYTHONIOENCODING=utf-8 $PY tools/fetch_assets.py
PYTHONIOENCODING=utf-8 $PY tools/fetch_assets.py --check      # только проверить sha256
PYTHONIOENCODING=utf-8 $PY tools/fetch_assets.py --files      # + таблица файл/размер/sha256

# 2. состояние пайплайна одной командой (файлы, лицензии, кто читает, запекание)
PYTHONIOENCODING=utf-8 $PY tools/check_assets.py
PYTHONIOENCODING=utf-8 $PY tools/check_assets.py --detail     # + клипы, единицы, ключевые меши

# 3. запечь позу
node tools/bake_pose.js assets/cesium_man/CesiumMan.glb --list
node tools/bake_pose.js assets/cesium_man/CesiumMan.glb \
     --clip animation_0 --times 0,0.5,1,1.5 --outdir assets/_poses
```

## Что чем читается

| ридер | что умеет | замечания |
|-------|-----------|-----------|
| `open3d` 0.19 (`io.read_triangle_mesh`) | glb/gltf/obj/fbx/ply/stl — через встроенный assimp | скиннинг не применяет (отдаёт bind-pose), DRACO не умеет, свои варнинги печатает в **stdout** |
| three 0.186 в Node (`tools/inspect_mesh.mjs`) | glb/gltf/fbx/obj/stl/ply | скиннинг + морфы + клипы, DRACO недоступен (нет `Worker`/`importScripts`), текстуры подменяются заглушкой (нет декодера картинок) |
| браузерный клиент (`app/client`) | те же three-лоадеры + DRACOLoader | единственный, кто читает `kira.glb` (DRACO) |

three лежит в `app/client/node_modules` и из корня репозитория не резолвится,
поэтому `tools/*.mjs` импортируют его по абсолютному пути
(`tools/three_env.mjs`, переопределяется через `THREE_ROOT`).

## `tools/bake_pose.js`

```
node tools/bake_pose.js <asset> --list                       # список клипов с длительностями
node tools/bake_pose.js <asset> --clip <имя> --time 0.5 --out pose.npz
node tools/bake_pose.js <asset> --clip <имя> --times 0,0.5,1 --outdir assets/_poses
node tools/bake_pose.js <asset> --scale 0.01 --clip <имя> --time 0.5 --out pose_scaled.npz
node tools/bake_pose.js <asset> --clip <имя> --times 0,1 --loop once --outdir assets/_poses
```

* без `--clip` печётся поза покоя (bind pose + текущий скелет);
* `--times` — несколько моментов в одном процессе: состояние сцены
  восстанавливается между запеканиями, поэтому порядок вызовов на результат не влияет;
* `--scale S` домножает вершины и пишет множитель в метаданные (масштаб в метры —
  см. `manifest.json → units.calibrated_scale_to_meters`).

**`--loop once` — обязателен, если печётся момент ровно на `t == длительность клипа`.**
По умолчанию (`repeat`) three сворачивает действие, дошедшее до длительности, обратно
на `t = 0`, поэтому запечённый «последний» кадр молча оказывается ПЕРВЫМ. Измерено на
`Human Armature|Death` (t = 3.5 s): `repeat` даёт позу `t = 0` (высота 1.8134 m, стоя),
`once` — настоящий конец (высота 0.3740 m, лежит), разница 2.63 m. Ниже длительности
клипа оба режима дают одно и то же (проверено: 0.000000 m).
Файл момента пишется как `t<время>` с тремя знаками (`t3.5`), поэтому два близких
времени (3.499999 и 3.5) — это одно имя файла: второй вызов затрёт первый.

Рядом с `pose.npz` кладётся `pose.json`: клип, время, длительность, режим цикла
(`clip_loop_mode`), число вершин и треугольников, bbox, таблица мешей (со смещениями),
время запекания, sha256 исходника.

### Формат `.npz`

Пишется без зависимостей — это zip из `.npy` (ZIP method 0 + CRC32), ровно то,
что делает `numpy.savez`; читается напрямую:

```python
import numpy as np
z = np.load("CesiumMan_animation_0_t0.5.npz")
V = z["vertices"]   # (N, 3) float32 — мировые оси ассета (Y-up), скиннинг применён
F = z["faces"]      # (M, 3) int32  — индексы в V, несколько мешей уже сшиты со сдвигом
```

`vertices` — то, что рисует вьюер в этот момент (`SkinnedMesh.getVertexPosition`,
умноженный на `matrixWorld`, потому что three считает скиннинг в локальных осях
меша). Смещения мешей — в `pose.json → meshes[].vertex_offset / face_offset`.

## `tools/bake_poses.py` — дерево запечённых поз и контракт

Готовит сразу все клипы человекоподобных ассетов и сводный манифест:

```bash
PYTHONIOENCODING=utf-8 $PY tools/bake_poses.py            # запечь всё, что ещё не запечено
PYTHONIOENCODING=utf-8 $PY tools/bake_poses.py --reuse    # не печь: только пересобрать манифест
PYTHONIOENCODING=utf-8 $PY tools/bake_poses.py --only rigged_animated_humanoid
PYTHONIOENCODING=utf-8 $PY tools/check_poses.py --independent   # проверка + сверка перепечкой
```

Раскладка (она же контракт):

```
assets/_poses/manifest.json                 # сводка: ассеты, клипы, классы разметки, правила игры
assets/_poses/<asset>/<clip>/*.npz          # один момент = один файл (+ .json рядом)
assets/_poses/<asset>/<clip>/frames.npz     # весь клип одним файлом: vertices (T,N,3) f32,
                                            # faces (M,3) i32, times (T,) f32
assets/_poses/<asset>/<clip>/clip.json      # метаданные клипа (длительность, цикл, моменты, габариты)
```

* **Вершины в `.npz` уже в метрах** (запекание идёт с `--scale <calibrated_scale_to_meters>`).
  `scale_applied` в пофайловых `.json` — это уже применённый множитель: умножать на него
  ещё раз НЕ надо (пример ниже — для «сырого» запекания без `--scale`).
* Оси: меш Y-up (как в glTF/FBX), кадр — Z-up, переход `(x, y, z) -> (x, -z, y)`.
* Сколько моментов: 24 на клип короче 0.75 s, 18 до 2 s, 14 до 5 s, 12 длиннее; клип из
  одного кадра (<= 0.075 s) — 1 момент. Шаг выходит 26–55 ms у коротких, 250–830 ms у длинных.
* Цикл определяется ИЗМЕРЕНИЕМ (сравнение позы в конце с позой в начале через `--loop once`),
  а не по имени: у всех клипов `rigged_animated_humanoid` расхождение ровно 0.0000 m, поэтому
  они циклические, а `Human Armature|Death` — 2.6342 m (ratio 0.99), поэтому нет.
  Циклические кладутся моментами `[0, d)` без правого конца (склейка без дубля кадра),
  нециклические — `[0, d]`. У циклов с расхождением > 2 cm стоит `loop_needs_crossfade`.
* `class_map` в манифесте — соответствие `standing/walking/running/lying/fallen/unknown`
  на `asset` + `clip` + номера моментов, с обоснованием числами.

## Как это подхватывает разметка

```python
import json, numpy as np, open3d as o3d

pose = np.load("assets/_poses/CesiumMan_animation_0_t0.5.npz")
meta = json.load(open("assets/_poses/CesiumMan_animation_0_t0.5.json"))
V = pose["vertices"].astype(np.float64) * meta["scale_applied"]
F = pose["faces"].astype(np.uint32)

# 1. ассет (Y-up, его собственные оси) -> мировые оси трека
V_world = (V * SCALE_TO_METERS) @ R.T + t          # R: (3,3), t: (3,)

# 2. BVH строится ОДИН раз на траекторию: меш уже в мировых осях
scene = o3d.t.geometry.RaycastingScene()
scene.add_triangles(o3d.t.geometry.TriangleMesh(
    o3d.core.Tensor(V_world.astype(np.float32)),
    o3d.core.Tensor(np.ascontiguousarray(F.astype(np.uint32)))))

# 3. трассировка: rays = (M, 6) [origin(3), direction(3)]; origin — позиция сенсора,
#    direction — уже повёрнутое направление луча
ans = scene.cast_rays(o3d.core.Tensor(rays.astype(np.float32)))
t_hit = ans["t_hit"].numpy()
```

Замеры (повторяются на этой машине): каст всех 346 тыс. лучей — ≈5.3 мс,
по подмножеству кандидатов — ≈0.1 мс, число треугольников на время каста не влияет.
Поэтому «много поз» = много BVH: на каждую позу в траектории свой `add_triangles`
(4–13 мс на позу, см. `bake_ms` в JSON), а лучи по ним — уже дёшево.

`SCALE_TO_METERS` для каждого ассета — `manifest.json → units.calibrated_scale_to_meters`
(например: Kenney 1.0, CesiumMan 1.0, Fox 0.01, Quaternius FBX 0.00333,
Rigged Humanoid 0.00872). Обоснование каждой цифры — в `units.evidence`.

Важно про форматы: **FBX в этих паках отдаёт сантиметры**, GLB/OBJ — метры.
Проверено на одной и той же модели Kenney (`light-square`): GLB/OBJ
`0.05 x 0.60 x 0.2375` м, а FBX тех же 60 треугольников — `5 x 60 x 23.75`, то есть
×100. Поэтому один и тот же ассет, взятый в GLB или в FBX, требует разных
множителей (см. `units.asset_units_to_meters`).

## Файлы

| файл | роль |
|------|------|
| `fetch_assets.py` | реестр ассетов, скачивание, sha256, распаковка (только stdlib) |
| `three_env.mjs` | three в Node: DOM-заглушки, загрузка ассета, `bakePose`, статистика |
| `inspect_mesh.mjs` | CLI: JSON-отчёт по файлу (вершины/треугольники/bbox/клипы) |
| `npz.mjs` | писатель `.npz`/`.npy` без зависимостей |
| `bake_pose.js` | CLI запекания позы в `.npz` + `.json` |
| `bake_poses.py` | все клипы человекоподобных ассетов -> `assets/_poses/**` + `manifest.json` (контракт) |
| `check_poses.py` | проверка запечённого: монотонность моментов, смещения, габариты, свежесть перепечкой |
| `check_assets.py` | проверка всего пайплайна; `--write-manifest` пересобирает манифест |

## `tools/build_dataset.py` — сборка обучающего датасета

Синтезирует разметку на записях `for_hackathon` без ручных кликов (раскладка
сидом, запись — `labeling.propose/preview/save`), кладёт её в ОТДЕЛЬНЫЙ каталог
`dataset/` (в `.gitignore`; ручная разметка `labels/` не трогается):
`dataset/labels/` -- синтетика (люди в позах и дальностях, часть с
траекториями; оборудование примитивами), `dataset/anchor_person/` -- РЕАЛЬНЫЙ
человек из `doubleT_obstacle` кадрами 13..63 (GT -- измеренные точки кластера,
`points_source=measured`), `dataset/manifest.json` -- статистика, негативы
(кадры без разметки, где детектор молчит), сплит train/val ПО ЗАПИСАМ,
seed и ревизии.

```bash
PYTHONIOENCODING=utf-8 $PY tools/build_dataset.py                     # все 6 записей
PYTHONIOENCODING=utf-8 $PY tools/build_dataset.py --workers 3         # быстрее (память: ~2 ГБ/воркер)
PYTHONIOENCODING=utf-8 $PY tools/build_dataset.py --records doubleT_obstacle roundT_doubleT
```

Идемпотентно: повторный запуск с тем же сидом пропускает готовые записи
(индекс пересобирается по карточкам). Оценка времени: ход-профиль ~0.12 с/кадр
записи, трассировка ~0.1-0.2 с/кадр объекта; все 6 записей -- ~10 мин одним
воркером. Проверки -- `tests/test_build_dataset.py`.