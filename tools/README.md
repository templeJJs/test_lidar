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
```

* без `--clip` печётся поза покоя (bind pose + текущий скелет);
* `--times` — несколько моментов в одном процессе: состояние сцены
  восстанавливается между запеканиями, поэтому порядок вызовов на результат не влияет;
* `--scale S` домножает вершины и пишет множитель в метаданные (масштаб в метры —
  см. `manifest.json → units.calibrated_scale_to_meters`).

Рядом с `pose.npz` кладётся `pose.json`: клип, время, длительность, число вершин и
треугольников, bbox, таблица мешей (со смещениями), время запекания, sha256 исходника.

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
| `check_assets.py` | проверка всего пайплайна; `--write-manifest` пересобирает манифест |