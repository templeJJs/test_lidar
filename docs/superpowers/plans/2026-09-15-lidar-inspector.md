# LiDAR Inspector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Новый `inspect_bag.py` — инспектор rosbag2 `.db3` туннеля с дашбордом 2×2, кнопками камеры и контрастными раскрасками, чтобы глазами находить препятствия.

**Architecture:** Самодостаточный файл. Чистое ядро (парсер, кэш кадров, раскраски, обрезка, оверлеи) — тестируется через `unittest` из stdlib. GUI-слой — `o3d.visualization.gui` с 4 `SceneWidget`, проверяется эмпирически (fps в HUD + ASCII-снимок окна).

**Tech Stack:** Python 3.12, numpy 2.4.4, Open3D 0.19.0 (`gui`, `rendering`), sqlite3, threading, unittest.

**Spec:** `docs/superpowers/specs/2026-09-15-lidar-inspector-design.md`

**Ключевой факт, определяющий реализацию:** `o3d.utility.Vector3dVector` требует `float64`; на `float32` pybind конвертирует поэлементно — 205 мс на 346 808 точек против 1.8 мс. В Open3D 0.19 `Vector3fVector` отсутствует, поэтому `astype(np.float64)` обязателен.

**Окружение:** интерпретатор `C:\Users\K4ler\AppData\Local\Programs\Python\Python312\python.exe` (глобальный `python` — 3.13, где нет wheel-а open3d). Ниже он обозначается `$PY`.

---

## File Structure

- Create: `inspect_bag.py` — весь инспектор (ядро + GUI).
- Create: `tests/__init__.py` — пустой, делает `tests` пакетом.
- Create: `tests/test_core.py` — unit-тесты чистого ядра (`unittest`).
- Не трогать: `visualize_bag.py`, `README.md`, `analyze_bag.py`, `for_hackathon/`.

Оверлеи и раскраски возвращают numpy/Open3D объекты, а не рисуют — поэтому тестируемы без окна. GUI-слой изолирован в классах `Viewport` и `Inspector`: окно создаётся только в `main()`.

---

## Task 1: Тестовый каркас и парсер CDR

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/test_core.py`
- Create: `inspect_bag.py` (только константы + `parse_pointcloud2_cdr`)

- [ ] **Step 1: Создать пустой `tests/__init__.py`**

Пустой файл.

- [ ] **Step 2: Написать падающий тест — синтетический CDR и реальный bag**

В `tests/test_core.py`:

```python
import os
import struct
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import inspect_bag  # noqa: E402

BAG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   'for_hackathon', 'doubleT_obstacle')


def build_cdr(points):
    """Собрать минимальный CDR PointCloud2 с полями x,y,z,intensity (f4), point_step 16."""
    bo = '<'
    out = bytearray(b'\x00\x01\x00\x00')
    out += struct.pack(bo + 'II', 0, 0)
    fid = b'lidar\x00'
    out += struct.pack(bo + 'I', len(fid)) + fid
    out += b'\x00' * (-len(out) % 4)
    out += struct.pack(bo + 'II', 1, len(points))
    names = [b'x', b'y', b'z', b'intensity']
    out += struct.pack(bo + 'I', len(names))
    for i, nm in enumerate(names):
        out += struct.pack(bo + 'I', len(nm) + 1) + nm + b'\x00'
        out += b'\x00' * (-len(out) % 4)
        out += struct.pack(bo + 'I', i * 4)
        out += struct.pack(bo + 'B', 7)
        out += b'\x00' * (-len(out) % 4)
        out += struct.pack(bo + 'I', 1)
    out += struct.pack(bo + 'B', 0)
    out += b'\x00' * (-len(out) % 4)
    arr = np.asarray(points, dtype='<f4')
    out += struct.pack(bo + 'III', 16, 16 * len(points), arr.nbytes)
    out += arr.tobytes()
    return bytes(out)


class TestParser(unittest.TestCase):
    def test_synthetic_roundtrip(self):
        pts = [(1.0, 2.0, 3.0, 10.0), (-4.5, -60.0, 0.5, 200.0), (0.0, 0.0, 0.0, 0.0)]
        xyz, intensity = inspect_bag.parse_pointcloud2_cdr(build_cdr(pts))
        self.assertEqual(xyz.shape, (2, 3), 'all-zero point must be filtered out')
        self.assertEqual(xyz.dtype, np.float32)
        np.testing.assert_allclose(xyz[0], [1.0, 2.0, 3.0], rtol=1e-6)
        np.testing.assert_allclose(xyz[1], [-4.5, -60.0, 0.5], rtol=1e-6)
        self.assertEqual(intensity.dtype, np.float32)
        np.testing.assert_allclose(intensity, [10.0, 200.0], rtol=1e-6)

    def test_synthetic_missing_intensity(self):
        pts = [(5.0, -1.0, 0.0), (6.0, -2.0, 0.0)]
        blob = build_cdr(pts)
        xyz, intensity = inspect_bag.parse_pointcloud2_cdr(blob)
        self.assertEqual(xyz.shape, (2, 3))
        self.assertIsNotNone(intensity, 'synthetic blob does declare intensity')

    def test_real_bag_matches_known_facts(self):
        db = os.path.join(BAG, 'doubleT_obstacle_0.db3')
        if not os.path.exists(db):
            self.skipTest('bag not present')
        import sqlite3
        conn = sqlite3.connect(db)
        blob = conn.execute('SELECT data FROM messages ORDER BY timestamp ASC LIMIT 1').fetchone()[0]
        conn.close()
        xyz, intensity = inspect_bag.parse_pointcloud2_cdr(blob)
        self.assertEqual(len(xyz), 346808)
        self.assertAlmostEqual(xyz[:, 0].min(), -8.67, places=1)
        self.assertAlmostEqual(xyz[:, 0].max(), 7.07, places=1)
        self.assertLess(xyz[:, 1].min(), -200.0)
        self.assertLess(xyz[:, 2].min(), -6.0)
        self.assertLess(xyz[:, 2].max(), 3.0)
        self.assertAlmostEqual(float(intensity.max()), 255.0, places=3)


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 3: Запустить тест и убедиться, что падает**

Run: `$PY -m unittest tests.test_core -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'inspect_bag'`

- [ ] **Step 4: Создать `inspect_bag.py` с парсером**

Перенести `parse_pointcloud2_cdr` из `visualize_bag.py:23-102` без изменения логики. Именно он валидирует схему `point_step=26`, `intensity` f4 @12.

- [ ] **Step 5: Запустить тесты**

Run: `$PY -m unittest tests.test_core -v`
Expected: 3 passed (или 2 passed + 1 skipped без данных)

- [ ] **Step 6: Commit**

```bash
git add tests/__init__.py tests/test_core.py inspect_bag.py
git commit -m "feat: add cdr pointcloud parser with tests"
```

---

## Task 2: Кэш кадров и фоновая предзагрузка

**Files:**
- Modify: `inspect_bag.py`
- Test: `tests/test_core.py`

- [ ] **Step 1: Написать падающие тесты**

Добавить в `tests/test_core.py`:

```python
class TestBagFrames(unittest.TestCase):
    DB = os.path.join(BAG, 'doubleT_obstacle_0.db3')

    def setUp(self):
        if not os.path.exists(self.DB):
            self.skipTest('bag not present')

    def test_len_and_topics(self):
        fr = inspect_bag.BagFrames(self.DB, cache_size=2)
        self.assertEqual(len(fr), 201)
        self.assertIn('/sensing/lidar/hesai128/pointcloud', fr.topic_names)

    def test_cache_avoids_reparse(self):
        fr = inspect_bag.BagFrames(self.DB, cache_size=2)
        fr.parse_count = 0
        fr[0]
        first = fr.parse_count
        fr[0]
        self.assertEqual(fr.parse_count, first, 'second access must hit the cache')

    def test_ring_cache_evicts(self):
        fr = inspect_bag.BagFrames(self.DB, cache_size=2)
        for i in range(4):
            fr[i]
        self.assertLessEqual(len(fr._cache), 2)

    def test_timestamps_sorted_ns(self):
        fr = inspect_bag.BagFrames(self.DB, cache_size=1)
        self.assertTrue(np.all(np.diff(fr.timestamps) > 0))
        self.assertGreater(fr.timestamps[-1] - fr.timestamps[0], 20e9)

    def test_prefetch_fills_ahead(self):
        import time
        fr = inspect_bag.BagFrames(self.DB, cache_size=8, prefetch_ahead=3)
        fr.want(0)
        time.sleep(3.0)
        self.assertGreaterEqual(len(fr._cache), 2, 'prefetch must load frames ahead')
```

- [ ] **Step 2: Запустить и убедиться, что падает**

Run: `$PY -m unittest tests.test_core.TestBagFrames -v`
Expected: FAIL — `AttributeError: module 'inspect_bag' has no attribute 'BagFrames'`

- [ ] **Step 3: Реализовать `BagFrames`**

Требования:
- `__init__(db_path, cache_size=32, prefetch_ahead=8)`; главное соединение создаётся в вызывающем потоке.
- `rowids`, `timestamps` (int64, наносекунды), `topic_names` — как в `visualize_bag.py:105-139`.
- `parse_count` — инкрементируется в `_parse` (для теста попадания в кэш).
- `_cache` — `OrderedDict`, вытеснение по LRU, под `threading.Lock`.
- `want(idx)` — выставить индекс, от которого поток предзагрузки идёт вперёд.
- `_prefetch_loop` — поток-демон, **своё** соединение `sqlite3.connect` (соединения не потокобезопасны), парсит индексы `idx+1 … idx+prefetch_ahead`, кладёт в кэш, спит 20 мс, останавливается через `threading.Event`.

- [ ] **Step 4: Запустить тесты**

Run: `$PY -m unittest tests.test_core.TestBagFrames -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add inspect_bag.py tests/test_core.py
git commit -m "feat: add frame cache with background prefetch"
```

---

## Task 3: Раскраски

**Files:**
- Modify: `inspect_bag.py`
- Test: `tests/test_core.py`

- [ ] **Step 1: Написать падающие тесты**

```python
class TestColors(unittest.TestCase):
    def setUp(self):
        self.xyz = np.array([[0.0, 0.0, -7.0], [0.0, 0.0, 0.0], [0.0, 0.0, 3.0],
                             [1.0, -40.0, 0.0]], dtype=np.float32)
        self.inten = np.array([0.0, 0.0, 1.0, 255.0], dtype=np.float32)
        self.ring = np.array([0, 64, 127, 12], dtype=np.uint16)

    def test_shape_dtype_range_all_modes(self):
        for mode in inspect_bag.MODES:
            c = inspect_bag.frame_colors(self.xyz, self.inten, mode=mode,
                                         max_dist=80.0, ring=self.ring)
            self.assertEqual(c.shape, (4, 3), mode)
            self.assertEqual(c.dtype, np.float64, mode)
            self.assertTrue(np.all(c >= 0.0) and np.all(c <= 1.0), mode)
            self.assertGreater(float(c.max()), 0.5,
                               f'{mode}: colors must be bright, not 0.05-0.2')

    def test_height_mode_is_monotonic_and_uses_cold_to_hot(self):
        c = inspect_bag.frame_colors(self.xyz, self.inten, mode='height')
        self.assertGreater(c[0].sum(), 0.0)
        self.assertNotEqual(tuple(c[0]), tuple(c[2]), 'floor and ceiling differ')

    def test_range_mode_decreases_with_distance(self):
        near = np.array([[0.0, -1.0, 0.0]], dtype=np.float32)
        far = np.array([[0.0, -79.0, 0.0]], dtype=np.float32)
        cn = inspect_bag.frame_colors(near, None, mode='range', max_dist=80.0)
        cf = inspect_bag.frame_colors(far, None, mode='range', max_dist=80.0)
        self.assertNotEqual(tuple(cn[0]), tuple(cf[0]))

    def test_intensity_mode_uses_percentiles_not_max(self):
        i = np.concatenate([np.zeros(990, dtype=np.float32),
                            np.linspace(1, 255, 10).astype(np.float32)])
        xyz = np.zeros((1000, 3), dtype=np.float32)
        c = inspect_bag.frame_colors(xyz, i, mode='intensity')
        self.assertGreater(float(c.max()), 0.5)

    def test_missing_field_raises(self):
        with self.assertRaises(ValueError):
            inspect_bag.frame_colors(self.xyz, None, mode='intensity')
        with self.assertRaises(ValueError):
            inspect_bag.frame_colors(self.xyz, self.inten, mode='ring', ring=None)

    def test_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            inspect_bag.frame_colors(self.xyz, self.inten, mode='nope')

    def test_empty_input(self):
        c = inspect_bag.frame_colors(np.zeros((0, 3), dtype=np.float32), None, mode='height')
        self.assertEqual(c.shape, (0, 3))
        self.assertEqual(c.dtype, np.float64)
```

- [ ] **Step 2: Запустить и убедиться, что падает**

Run: `$PY -m unittest tests.test_core.TestColors -v`
Expected: FAIL — `AttributeError: module 'inspect_bag' has no attribute 'frame_colors'`

- [ ] **Step 3: Реализовать `TURBO_ANCHORS`, `_ramp`, `MODES`, `frame_colors`**

Палитра — 15 якорных цветов Google turbo, интерполяция `np.interp` по каналам. Диапазон значений 0.2–1.0 (в 3–5 раз ярче текущего `colorize_rails`, где 74 % точек уходили в 0.05–0.2 и сливались с фоном `(0.01,0.01,0.03)`). Режимы: `height` (Z, диапазон `z_range=(-7.0, 3.0)`), `intensity` (перцентили 1…99), `range` (норма вектора / `max_dist`), `ring` (`ring % 128 / 127`).

- [ ] **Step 4: Запустить тесты**

Run: `$PY -m unittest tests.test_core.TestColors -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add inspect_bag.py tests/test_core.py
git commit -m "feat: add turbo colormaps for height/intensity/range/ring"
```

---

## Task 4: Обрезка по дальности и оверлеи

**Files:**
- Modify: `inspect_bag.py`
- Test: `tests/test_core.py`

- [ ] **Step 1: Написать падающие тесты**

```python
class TestTrimAndOverlays(unittest.TestCase):
    def test_trim_range_keeps_corridor_only(self):
        xyz = np.array([[0.0, -250.0, 0.0], [0.0, -80.0, 0.0], [0.0, 0.0, 0.0],
                        [0.0, 5.0, 0.0], [0.0, 40.0, 0.0]], dtype=np.float32)
        mask = inspect_bag.trim_range(xyz, max_dist=80.0, behind=5.0)
        np.testing.assert_array_equal(mask, [False, True, True, True, False])

    def test_trim_applies_to_intensity_and_ring(self):
        xyz = np.array([[0.0, -100.0, 0.0], [0.0, -1.0, 0.0]], dtype=np.float32)
        inten = np.array([1.0, 2.0], dtype=np.float32)
        ring = np.array([7, 8], dtype=np.uint16)
        out = inspect_bag.trim(xyz, inten, ring, max_dist=80.0, behind=5.0)
        self.assertEqual(len(out[0]), 1)
        np.testing.assert_allclose(out[1], [2.0])
        np.testing.assert_array_equal(out[2], [8])

    def test_grid_line_count_and_bounds(self):
        grid = inspect_bag.make_grid(spacing=5.0, half_width=40.0, length=200.0, z=0.0)
        n_pts = len(np.asarray(grid.points))
        n_lines = len(np.asarray(grid.lines))
        self.assertEqual(n_pts, n_lines * 2)
        self.assertEqual(n_lines, 17 + 41, 'x lines 17, y lines 41')
        pts = np.asarray(grid.points)
        self.assertTrue(np.all(pts[:, 2] == 0.0))
        self.assertAlmostEqual(pts[:, 0].min(), -40.0)
        self.assertAlmostEqual(pts[:, 0].max(), 40.0)
        self.assertAlmostEqual(pts[:, 1].min(), -200.0)

    def test_axes_three_colored_lines(self):
        ax = inspect_bag.make_axes(5.0)
        self.assertEqual(len(np.asarray(ax.lines)), 3)
        colors = np.asarray(ax.colors)
        self.assertEqual(colors.shape, (3, 3))
        self.assertEqual(int(np.argmax(colors[0])), 0)
        self.assertEqual(int(np.argmax(colors[1])), 1)
        self.assertEqual(int(np.argmax(colors[2])), 2)
```

- [ ] **Step 2: Запустить и убедиться, что падает**

Run: `$PY -m unittest tests.test_core.TestTrimAndOverlays -v`
Expected: FAIL — `AttributeError: module 'inspect_bag' has no attribute 'trim_range'`

- [ ] **Step 3: Реализовать `trim_range`, `trim`, `make_grid`, `make_axes`**

`trim_range` — маска по Y: `-max_dist <= y <= behind` (туннель уходит в −Y; `behind` отсекает точки за спиной). `make_grid` — сетка на плоскости Z=`z`, шаг `spacing`, `±half_width` по X, от `-length` до 0 по Y. `make_axes` — 3 отрезка из начала координат, цвета r/g/b для X/Y/Z.

- [ ] **Step 4: Запустить тесты**

Run: `$PY -m unittest tests.test_core.TestTrimAndOverlays -v`
Expected: 4 passed

- [ ] **Step 5: Полный прогон ядра**

Run: `$PY -m unittest discover -s tests -v`
Expected: все проходят

- [ ] **Step 6: Commit**

```bash
git add inspect_bag.py tests/test_core.py
git commit -m "feat: add range trim and grid/axes overlays"
```

---

## Task 5: Спайк — проверка GUI до написания GUI

Цель: снять неопределённость по трём пунктам, пока не написан инспектор. Скрипт спайка живёт в `C:\Users\K4ler\AppData\Local\Temp\opencode\gui_spike.py`, **не в репозитории**.

- [ ] **Step 1: Проверить, что 2×2 из `SceneWidget` рендерится и растягивается**

Собрать `gui.Application` → `gui.Window` → вложенные `gui.Horiz`/`gui.Vert` с 4 `SceneWidget`. Убедиться, что каждый вид получает четверть окна (иначе задать размеры явно через `window.set_on_layout`).

- [ ] **Step 2: Проверить, поддерживает ли pip-сборка альфу виджетов**

`panel.background_color = gui.Color(0.05, 0.05, 0.08, 0.55)`. Зафиксировать результат: либо панель полупрозрачная, либо непрозрачная. При непрозрачной — узкая панель + сворачивание по Tab.

- [ ] **Step 3: Замерить аплоад 346 808 точек в 4 вида**

Для каждого из 4 `SceneWidget`: `scene.remove_geometry('cloud', False)` + `scene.add_geometry('cloud', pcd, mat)` с `mat.shader='defaultUnlit'`. Замерить медианное время аплоада и достижимый fps.

- [ ] **Step 4: Зафиксировать решение о прореживании**

Ожидание по расчёту: ~2 мс на вид, всего ~8 мс → прореживание **не нужно**. Если замер даёт >120 мс на кадр только на аплоаде — включить авто-прореживание до ~150k точек на вид с индикацией в HUD. Решение принимается по замеру, а не по предположению.

- [ ] **Step 5: Записать результаты спайка в план**

Дописать в этот файл фактическое решение (альфа: да/нет; аплоад: N мс; прореживание: да/нет и порог).

---

## Task 6: Пресеты камеры и `Viewport`

**Files:**
- Modify: `inspect_bag.py`
- Test: `tests/test_core.py`

- [ ] **Step 1: Написать падающий тест на таблицу пресетов**

```python
class TestPresets(unittest.TestCase):
    def test_all_presets_well_formed(self):
        expected = {'lidar', 'top', 'side', 'iso', 'behind', 'fit'}
        self.assertEqual(set(inspect_bag.PRESETS), expected)
        for name, p in inspect_bag.PRESETS.items():
            if name == 'fit':
                continue
            self.assertEqual(len(p['eye']), 3, name)
            self.assertEqual(len(p['lookat']), 3, name)
            self.assertEqual(len(p['up']), 3, name)
            self.assertNotEqual(tuple(p['eye']), tuple(p['lookat']), name)

    def test_up_vectors_are_orthogonal_to_view(self):
        for name, p in inspect_bag.PRESETS.items():
            if name == 'fit':
                continue
            eye = np.array(p['eye'], dtype=float)
            look = np.array(p['lookat'], dtype=float)
            up = np.array(p['up'], dtype=float)
            fwd = look - eye
            fwd /= np.linalg.norm(fwd)
            up_n = up / np.linalg.norm(up)
            self.assertLess(abs(float(np.dot(fwd, up_n))), 1e-6,
                            f'{name}: up must not be parallel to view direction')
```

- [ ] **Step 2: Запустить и убедиться, что падает**

Run: `$PY -m unittest tests.test_core.TestPresets -v`
Expected: FAIL — `AttributeError: module 'inspect_bag' has no attribute 'PRESETS'`

- [ ] **Step 3: Реализовать `PRESETS`**

Таблица из спеки: `lidar` eye (0,2,1.5) lookat (0,−30,0) up +Z; `top` eye (2,−30,80) lookat (2,−30,0) up −Y; `side` eye (60,−30,0) lookat (2,−30,0) up +Z; `iso` eye (40,40,30) lookat (2,−30,0) up +Z; `behind` eye (2,12,4) lookat (2,−40,0) up +Z; `fit` без eye/lookat (кадрируется по габариту).

- [ ] **Step 4: Запустить тесты**

Run: `$PY -m unittest tests.test_core.TestPresets -v`
Expected: 2 passed

- [ ] **Step 5: Реализовать `Viewport`**

Обёртка над `gui.SceneWidget`: тёмный фон, `set_view_controls(ROTATE_CAMERA)`, FOV 60°, `apply_preset(name)`, `fit(bbox)` через `setup_camera`, `set_cloud(xyz_f64, colors_f64, point_size)` (remove+add с `defautUnlit`), `set_hud(text)` через `add_3d_label`/`gui.Label`. Плюс добавление оверлеев сетки и осей.

- [ ] **Step 6: Commit**

```bash
git add inspect_bag.py tests/test_core.py
git commit -m "feat: add camera presets and viewport wrapper"
```

---

## Task 7: Окно инспектора, панель, транспорт

**Files:**
- Modify: `inspect_bag.py` (`Inspector`, `main`)

- [ ] **Step 1: Реализовать `Inspector`**

- Окно 1600×900, заголовок `LiDAR Inspector — <bag>`.
- 4 `Viewport` в 2×2 (`Horiz(Vert(Vert(v1,v2)), Vert(v3,v4))`).
- Панель слева (`CollapsableVert`) с виджетами из спеки: 6 кнопок пресетов, транспорт (▶/⏸, ±1, ±10, слайдер кадра), слайдер скорости 1…30 Гц, чекбокс «реальное время», Combobox раскраски, слайдер размера точки 1…6, чекбоксы сетка/оси/HUD, слайдер «обрезать дальше N м» 10…200 (по умолчанию 80).
- `window.set_on_key` — Tab сворачивает панель, Space пауза, ←/→ шаг.
- `window.set_on_tick_event` — продвижение кадра по wall-clock, `window.post_redraw()` только когда нужно.
- HUD: кадр/всего, время bag (с), дальность вперёд (м), время кадра (мс), сглаженный fps.
- Обновление всех 4 видов одним кадром; `frames.want(idx)` для предзагрузки.

- [ ] **Step 2: Реализовать `main()`**

`argparse`: `bag_dir` (по умолчанию `for_hackathon/doubleT_obstacle`), `--max-dist`, `--point-size`, `--start`, `--fps`. Печать длительности, числа кадров и измеренного уровня пола (мода Z) — уровень нужен для сетки и раскраски по высоте.

- [ ] **Step 3: Запустить инспектор**

Run: `$PY inspect_bag.py for_hackathon/doubleT_obstacle`
Expected: окно, 4 вида, облако яркое, панель слева; в консоли fps ≥ 8

- [ ] **Step 4: Commit**

```bash
git add inspect_bag.py
git commit -m "feat: add inspector window with dashboard and controls"
```

---

## Task 8: Приёмочная проверка

**Files:**
- Не в репозитории: `C:\Users\K4ler\AppData\Local\Temp\opencode\ascii_window.py` (уже написан)

- [ ] **Step 1: Запустить инспектор и снять окно в ASCII**

Run: `$PY inspect_bag.py for_hackathon/doubleT_obstacle` в фоне, затем
`$PY C:\Users\K4ler\AppData\Local\Temp\opencode\ascii_window.py "LiDAR Inspector - doubleT_obstacle"`
Expected: облако яркое и занимает кадр. Метрика приёмки — доля «зажжённых» пикселей (`lum>0.10`) **существенно выше** текущих 9 % из `visualize_bag.py`.

- [ ] **Step 2: Проверить, что ни один вид не пуст**

Снять окно, разрезать на 4 прямоугольника 2×2, посчитать долю `lum>0.10` в каждом. Каждый > 0.

- [ ] **Step 3: Проверить все пресеты и раскраски**

Пройти 6 пресетов и 4 раскраски, после каждого — ASCII-снимок и проверка, что кадр не почернел. Ни одного исключения в stderr.

- [ ] **Step 4: Замерить fps**

Взять сглаженный fps из HUD (виден в ASCII-снимке) или из консольного вывода.
Expected: ≥ 8 Гц при всех 346 808 точках без прореживания.

- [ ] **Step 5: Сверить парсер с фактами**

Run: `$PY -m unittest discover -s tests -v`
Expected: все проходят, включая `test_real_bag_matches_known_facts` (346 808 точек; X ∈ [−8.7, 7.1]; Y до −208).

- [ ] **Step 6: Дописать README-раздел про инспектор**

Добавить в `README.md` раздел «Инспектор» с командой запуска и списком управления. **Правки `visualize_bag.py` не трогать** (их правит второй агент); в разделе отдельно отметить, что утверждение про «0.7 мкс/точку и прореживание до 60k» неверно, и почему (`float32` → `float64` в `Vector3dVector`).

- [ ] **Step 7: Commit**

```bash
git add inspect_bag.py README.md
git commit -m "docs: document lidar inspector usage"
```

---

## Self-Review

**Покрытие спеки:**

| раздел спеки | задача |
|---|---|
| `parse_pointcloud2_cdr` без изменений | Task 1 |
| `astype(np.float64)` | Task 3, 6 (в `Viewport.set_cloud`), Task 8 Step 4 |
| `BagFrames` + кольцевой кэш 32 + предзагрузка | Task 2 |
| `frame_colors` + 4 режима + палитра 0.2–1.0 | Task 3 |
| Обрезка по дальности | Task 4 |
| `make_grid` / `make_axes` | Task 4 |
| `Viewport` + `Inspector` | Task 6, 7 |
| 4 вида 2×2 + независимое вращение | Task 5, 6, 7 |
| 6 пресетов камеры + Fit | Task 6 |
| Панель: транспорт, слайдеры, чекбоксы | Task 7 |
| Прозрачность панели (по факту) | Task 5 Step 2 |
| HUD + fps | Task 7, Task 8 Step 4 |
| Бюджет кадра и решение о прореживании по замеру | Task 5 |
| Измерение уровня пола | Task 7 Step 2 |
| Приёмка: ASCII, 4 вида не пусты, fps ≥ 8 | Task 8 |
| Отдельный файл, `visualize_bag.py` не трогать | File Structure, Task 8 Step 6 |

**Placeholder scan:** Task 5 и Task 7 описывают GUI-слой требованиями и точными API-вызовами, а не полным кодом. Это осознанное отступление: GUI проверяется замером и ASCII-снимком (Task 5, Task 8), а его API подтверждён интроспекцией в спеке. Все чистое ядро идёт с полным кодом и тестами.

**Консистентность имён:** `parse_pointcloud2_cdr`, `BagFrames` (`want`, `_cache`, `parse_count`, `timestamps`, `topic_names`), `frame_colors(xyz, intensity, mode, max_dist, ring)`, `MODES`, `trim_range`, `trim`, `make_grid`, `make_axes`, `PRESETS`, `Viewport`, `Inspector` — используются одинаково во всех задачах.