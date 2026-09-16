# Симулятор туннеля и лидарных данных — план реализации (MVP)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** генератор синтетических записей метро-туннеля в формате наших `.db3`, с честной моделью сенсора, точной разметкой объектов и проверяемым сходством с реальными данными.

**Architecture:** пакет `sim/` на Python 3.12. Модель сенсора извлекается из настоящей записи (`sensor.py`), сцена строится параметрически и отдаётся мешами (`scene.py`), объекты — параметрические меши с материалом (`objects.py`), сканирование — трассировка лучей через `open3d.t.geometry.RaycastingScene` (`scanner.py`), запись — CDR-энкодер + SQLite в схеме rosbag2 (`writer.py`), разметка — манифест (`manifest.py`), вход — CLI (`cli.py`), сверка с реальностью — `check_sim.py`. Наш существующий конвейер (`bag_reader`, `zones`, `web_viewer`) на синтетике работает без правок — это и есть главный критерий приёмки.

**Tech Stack:** Python 3.12 (`C:\Users\K4ler\AppData\Local\Programs\Python\Python312\python.exe`), numpy, open3d (RaycastingScene, TriangleMesh), sqlite3, unittest (тесты — как в проекте, не pytest). Образец формата CDR — `bag_reader.parse_pointcloud2_cdr`, ГОСТ-профили рельса — `track_geometry.rail_profile_mm`.

**Спека:** `docs/superpowers/specs/2026-09-15-tunnel-simulator-design.md`.

---

## Файловая структура

| файл | ответственность |
|---|---|
| `sim/__init__.py` | пустой, пакет |
| `sim/sensor.py` | `SensorModel`: таблица углов места, число колонок, шум, статистика интенсивности; `from_bag`, `save`, `load` |
| `sim/scene.py` | `SceneParams`, `build_scene` → `Scene` (меши туннеля, рельсов, шпал, балласта; поза сенсора; путь) |
| `sim/objects.py` | `ObjectSpec`, `OBJECT_LIBRARY` (person, suitcase), `place_objects` → список объектов с боксами и флагом «в габарите» |
| `sim/scanner.py` | `DirFrame`, `scan(scene, sensor, frame_idx, rng)` → `Frame(xyz, intensity, ring, timestamp)` |
| `sim/writer.py` | `encode_pointcloud2_cdr`, `write_bag(frames, manifest, out_dir)` |
| `sim/manifest.py` | `Manifest`, `SceneInfo`, `ObjectInfo`; `to_json`/`from_json` |
| `sim/cli.py` | `python -m sim.cli sensor …` и `python -m sim.cli dataset …` |
| `sim/check_sim.py` | метрики «сим похож на реал» против указанной настоящей записи |
| `tests/test_sim_sensor.py` | тесты модели сенсора |
| `tests/test_sim_scene.py` | тесты геометрии сцены |
| `tests/test_sim_objects.py` | тесты объектов и габарита |
| `tests/test_sim_scanner.py` | тесты сканирования |
| `tests/test_sim_writer.py` | round-trip записи через `bag_reader` |
| `tests/test_sim_pipeline.py` | сквозные: наш сервер на синтетике, семантика объекта в коридоре |

Тесты запускаются из корня репозитория:

```bash
"C:/Users/K4ler/AppData/Local/Programs/Python/Python312/python.exe" -m unittest tests.test_sim_sensor -v
```

---

## Task 1: Модель сенсора из настоящей записи

**Files:**
- Create: `sim/__init__.py`, `sim/sensor.py`
- Test: `tests/test_sim_sensor.py`

- [ ] **Step 1: Написать падающий тест**

```python
"""Тесты модели сенсора: она извлекается из настоящей записи и воспроизводима."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bag_reader  # noqa: E402
from sim import sensor  # noqa: E402

BAG = 'for_hackathon/doubleT_obstacle'


class TestSensorFromBag(unittest.TestCase):
    def setUp(self):
        if not os.path.exists(BAG):
            self.skipTest('bag не распакован')
        self.model = sensor.SensorModel.from_bag(bag_reader.find_db3(BAG))

    def test_has_128_rings_with_measured_elevations(self):
        self.assertEqual(self.model.rings, 128)
        self.assertAlmostEqual(self.model.elevations_deg[0], 14.40, delta=0.15)
        self.assertAlmostEqual(self.model.elevations_deg[64], -3.02, delta=0.15)
        self.assertAlmostEqual(self.model.elevations_deg[127], -25.12, delta=0.15)

    def test_vertical_step_is_non_uniform_and_kept_from_data(self):
        step = float(np.median(np.abs(np.diff(self.model.elevations_deg))))
        self.assertAlmostEqual(step, 0.168, delta=0.02)
        spread = float(np.max(np.abs(np.diff(self.model.elevations_deg))))
        self.assertGreater(spread, 0.5, 'шаг по вертикали неравномерный, это надо сохранить')

    def test_azimuth_columns_match_the_reference_frame(self):
        self.assertAlmostEqual(self.model.azimuth_columns, 2709, delta=60)

    def test_intensity_stats_and_noise_are_measured(self):
        self.assertAlmostEqual(self.model.intensity['median'], 11.0, delta=2.0)
        self.assertGreater(self.model.range_sigma_m, 0.005)
        self.assertLess(self.model.range_sigma_m, 0.10)

    def test_roundtrip_through_json(self):
        d = self.model.to_dict()
        again = sensor.SensorModel.from_dict(d)
        np.testing.assert_allclose(again.elevations_deg, self.model.elevations_deg)
        self.assertEqual(again.azimuth_columns, self.model.azimuth_columns)


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2: Запустить тест — должен упасть**

Run: `... -m unittest tests.test_sim_sensor -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sim'`

- [ ] **Step 3: Реализовать `sim/sensor.py`**

```python
"""Модель лидара, извлечённая из настоящей записи (не из даташита).

Почему так: у нас конкретный сенсор, и воспроизводить надо его. Замеры по всем
шести записям показали одинаковые 128 колец с углами +14.40..-25.12 и медианным
шагом 0.168, но разное число азимутальных колонок на кадр (2709 против ~1400),
поэтому колонки -- параметр модели, а не константа.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import bag_reader


@dataclass
class SensorModel:
    rings: int = 128
    elevations_deg: np.ndarray = field(default_factory=lambda: np.zeros(0))
    azimuth_columns: int = 2709
    range_median_m: float = 6.9
    range_p95_m: float = 28.2
    range_max_m: float = 208.7
    range_sigma_m: float = 0.02
    intensity: dict = field(default_factory=dict)     # median, p90, share_gt25
    dropout: float = 0.0

    @classmethod
    def from_bag(cls, db_path: str, frame: int = 0) -> 'SensorModel':
        """Снять модель с одного кадра настоящей записи."""
        frames = bag_reader.BagFrames(db_path, cache_size=2, prefetch_ahead=0)
        try:
            f = frames[frame]
            xyz = f.xyz.astype(np.float64)
            ring = f.ring
            inten = f.intensity
        finally:
            frames.close()
        if ring is None:
            raise ValueError('в записи нет поля ring -- модель сенсора не снять')

        r = np.hypot(xyz[:, 0], xyz[:, 1])
        elev = np.degrees(np.arctan2(xyz[:, 2], r))
        rings = int(ring.max()) + 1
        elevations = np.full(rings, np.nan)
        for k in range(rings):
            m = ring == k
            if int(np.count_nonzero(m)) >= 10:
                elevations[k] = float(np.median(elev[m]))
        # пустые кольца заполняем линейной интерполяцией: сенсор должен иметь все лучи
        good = ~np.isnan(elevations)
        if not bool(good.all()):
            idx = np.arange(rings)
            elevations = np.interp(idx, idx[good], elevations[good])

        # шум дальности оцениваем по остатку плоскости пола: точки пола лежат на
        # плоскости, любой разброс вокруг неё -- шум сенсора
        import zones
        a, b, rms = zones.fit_floor(xyz, y_min=-40.0, y_max=-2.0)
        sigma = float(rms) if np.isfinite(rms) and rms > 0 else 0.02

        stats = {}
        if inten is not None:
            stats = {'median': float(np.median(inten)),
                     'p90': float(np.percentile(inten, 90)),
                     'share_gt25': float(np.mean(inten > 25))}
        columns = int(round(len(xyz) / float(rings)))
        return cls(rings=rings, elevations_deg=elevations, azimuth_columns=columns,
                   range_median_m=float(np.median(r)), range_p95_m=float(np.percentile(r, 95)),
                   range_max_m=float(r.max()), range_sigma_m=max(sigma, 0.005),
                   intensity=stats)

    def to_dict(self) -> dict:
        return {'rings': self.rings,
                'elevations_deg': [float(v) for v in self.elevations_deg],
                'azimuth_columns': self.azimuth_columns,
                'range_median_m': self.range_median_m, 'range_p95_m': self.range_p95_m,
                'range_max_m': self.range_max_m, 'range_sigma_m': self.range_sigma_m,
                'intensity': self.intensity, 'dropout': self.dropout}

    @classmethod
    def from_dict(cls, d: dict) -> 'SensorModel':
        d = dict(d)
        d['elevations_deg'] = np.asarray(d['elevations_deg'], dtype=np.float64)
        return cls(**d)

    def save(self, path: str) -> None:
        import json
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(self.to_dict(), fh, ensure_ascii=False, indent=1)

    @classmethod
    def load(cls, path: str) -> 'SensorModel':
        import json
        with open(path, encoding='utf-8') as fh:
            return cls.from_dict(json.load(fh))

    def azimuths_deg(self) -> np.ndarray:
        """Азимуты кадра: полный оборот, ровно `azimuth_columns` замеров."""
        return np.linspace(0.0, 360.0, self.azimuth_columns, endpoint=False)
```

- [ ] **Step 4: Запустить тест — должен пройти**

Run: `... -m unittest tests.test_sim_sensor -v`
Expected: OK (5 тестов)

- [ ] **Step 5: Коммит**

```bash
git add sim/__init__.py sim/sensor.py tests/test_sim_sensor.py
git commit -m "feat(sim): модель сенсора из настоящей записи"
```

---

## Task 2: Сцена — туннель, пол, сенсор

**Files:**
- Create: `sim/scene.py`
- Create: `sim/track_geometry_adapter.py` — в этой задаче **заглушка** (пустой `add_track`), меши рельсов и шпал наполнит Task 3
- Test: `tests/test_sim_scene.py`

- [ ] **Step 1: Написать падающий тест**

```python
"""Тесты геометрии сцены: стены/пол на измеренных местах, трассировка попадает."""
import os
import sys
import unittest

import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim import scene  # noqa: E402


class TestStraightTunnel(unittest.TestCase):
    def setUp(self):
        self.p = scene.SceneParams(seed=1, kind='straight', section='wide',
                                   length_m=60.0, grade_pct=2.34, tracks=2)
        self.s = scene.build_scene(self.p)

    def test_mesh_is_built(self):
        self.assertIsInstance(self.s.mesh, o3d.t.geometry.TriangleMesh)
        self.assertGreater(int(self.s.mesh.vertex.positions.shape[0]), 100)

    def test_casts_a_ray_into_the_floor_at_expected_height(self):
        rc = self.s.raycasting()
        origin = np.array([[0.0, -10.0, 2.0]], dtype=np.float32)
        direction = np.array([[0.0, 0.0, -1.0]], dtype=np.float32)
        rays = np.concatenate([origin, direction], axis=1)
        res = rc.cast_rays(o3d.core.Tensor(rays))
        hit_z = float(res['t_hit'].numpy()[0]) * -1.0 + 2.0
        # пол: a + b*y, при a=-1.82 (сенсор 1.82 м над полом) в y=-10 уровень ~-2.05
        self.assertAlmostEqual(hit_z, aux_floor(self.p, -10.0), delta=0.05)

    def test_side_rays_hit_the_walls_at_measured_x(self):
        rc = self.s.raycasting()
        for sign, expect in ((-1, -2.62), (1, 6.62)):
            rays = np.array([[0.0, -10.0, 0.5, float(sign), 0.0, 0.0]], dtype=np.float32)
            res = rc.cast_rays(o3d.core.Tensor(rays))
            hit = float(res['t_hit'].numpy()[0]) * sign
            self.assertAlmostEqual(hit, expect, delta=0.15)

    def test_sensor_pose_follows_the_section(self):
        self.assertAlmostEqual(self.s.sensor_pose['z'], 1.82, delta=0.05)
        narrow = scene.build_scene(scene.SceneParams(seed=1, kind='straight',
                                                     section='narrow', length_m=60.0))
        self.assertLess(narrow.sensor_pose['z'], 1.82)


def aux_floor(p: scene.SceneParams, y: float) -> float:
    return scene.floor_z(p, y)


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2: Запустить тест — упадёт**

Run: `... -m unittest tests.test_sim_scene -v`
Expected: FAIL — нет модуля `sim.scene`

- [ ] **Step 3: Реализовать `sim/scene.py` и заглушку адаптера**

Файл `sim/track_geometry_adapter.py` на этом шаге — минимальная заглушка, чтобы
`scene.py` импортировался и тесты Task 2 проходили:

```python
"""Заглушка адаптера рельсов: Task 3 заменит её настоящими мешами R65 и шпал."""


def add_track(parts: list, p) -> dict:
    from sim.scene import GAUGE_M
    axis = p.track_axis_x
    return {'axis_x': axis, 'gauge': GAUGE_M,
            'rails_x': [axis - GAUGE_M / 2.0, axis + GAUGE_M / 2.0]}
```

```python
"""Параметрическая сцена туннеля: меши, путь, поза сенсора.

Числа взяты из замеров: широкий двухпутный туннель -- стены -2.62/+6.62 (9.24 м),
пол при сенсоре -1.82 и уклон 2.34 %; узкий однопутный -- ширина ~4.3 м, пол ~-1.32.
Разрез (параметры рельсов, шпал, балласта) -- в `build_track_geometry`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import open3d as o3d

# Замеры по записям: (ширина туннеля, пол под сенсором, уклон %)
SECTIONS = {
    'wide':   {'width_m': 9.24, 'sensor_z': 1.82, 'grade_pct': 2.34, 'tracks': 2},
    'narrow': {'width_m': 4.30, 'sensor_z': 1.32, 'grade_pct': 1.00, 'tracks': 1},
}
GAUGE_M = 1.520
TUNNEL_HEIGHT_M = 4.10          # от пола до потолка (потолок в зонах начинается с z=2.0 при полe -1.8)


@dataclass
class SceneParams:
    seed: int
    kind: str = 'straight'          # straight | curve | turnout (MVP -- только straight)
    section: str = 'wide'
    length_m: float = 200.0
    grade_pct: float = None         # None -> из SECTIONS
    tracks: int = None              # None -> из SECTIONS
    track_axis_x: float = None      # поперёк, где стоит наш путь; None -> 0.0 (сенсор над ним)
    platform: bool = False
    gate: bool = False

    def resolved(self) -> 'SceneParams':
        s = SECTIONS[self.section]
        p = SceneParams(**{**self.__dict__})
        if p.grade_pct is None:
            p.grade_pct = s['grade_pct']
        if p.tracks is None:
            p.tracks = s['tracks']
        if p.track_axis_x is None:
            p.track_axis_x = 0.0
        return p


def floor_z(p: SceneParams, y):
    """Уровень пола в системе сенсора: сенсор на высоте `sensor_z` над полом."""
    s = SECTIONS[p.section]
    return -float(s['sensor_z']) + p.grade_pct / 100.0 * np.asarray(y, dtype=np.float64)


@dataclass
class Scene:
    params: SceneParams
    mesh: o3d.t.geometry.TriangleMesh
    track: dict                       # {'axis_x': float, 'gauge': float, 'rails_x': [l, r]}
    sensor_pose: dict                 # {'x','y','z','pitch_deg'}
    objects: list = field(default_factory=list)

    def raycasting(self) -> o3d.t.geometry.RaycastingScene:
        rc = o3d.t.geometry.RaycastingScene()
        rc.add_triangles(self.mesh)
        return rc


def _box(cx, cy, cz, sx, sy, sz) -> o3d.geometry.TriangleMesh:
    b = o3d.geometry.TriangleMesh.create_box(width=sx, height=sy, depth=sz)
    b.translate((cx - sx / 2.0, cy - sy / 2.0, cz - sz / 2.0))
    return b


def build_scene(params: SceneParams) -> Scene:
    """Собрать сцену: пол, потолок, две стены, рельсы и шпалы нашего пути."""
    p = params.resolved()
    s = SECTIONS[p.section]
    width = s['width_m']
    y = np.array([-p.length_m, p.length_m])
    z_floor = floor_z(p, y)

    parts = []
    # пол -- наклонная плита
    seg = 4
    ys = np.linspace(-p.length_m, p.length_m, seg + 1)
    for i in range(seg):
        y0, y1 = ys[i], ys[i + 1]
        z0, z1 = float(floor_z(p, y0)), float(floor_z(p, y1))
        parts.append(_box(0.0, (y0 + y1) / 2.0, (z0 + z1) / 2.0 - 0.1,
                          width, y1 - y0, 0.2))
    # стены и потолок
    for x_wall in (-2.62, 6.62):
        parts.append(_box(x_wall, 0.0, float(np.mean(z_floor)) + TUNNEL_HEIGHT_M / 2.0,
                          0.2, 2 * p.length_m, TUNNEL_HEIGHT_M))
    parts.append(_box(2.0, 0.0, float(np.mean(z_floor)) + TUNNEL_HEIGHT_M,
                      width, 2 * p.length_m, 0.2))

    from sim.track_geometry_adapter import add_track        # Task 3 наполнит модуль
    track = add_track(parts, p)

    mesh = o3d.geometry.TriangleMesh()
    for part in parts:
        mesh += part
    mesh.compute_vertex_normals()
    return Scene(params=p, mesh=mesh.to_tensor(), track=track,
                 sensor_pose={'x': 0.0, 'y': 0.0, 'z': 0.0, 'pitch_deg': 0.0})
```

- [ ] **Step 4: Запустить тест — должен пройти**

Run: `... -m unittest tests.test_sim_scene -v`
(на этом шаге `add_track` ещё заглушка — Task 3 её наполнит; тест стен и пола проходит)

- [ ] **Step 5: Коммит**

```bash
git add sim/scene.py tests/test_sim_scene.py
git commit -m "feat(sim): параметрическая сцена туннеля и поза сенсора"
```

---

## Task 3: Рельсы, шпалы, балласт (ГОСТ-профили из `track_geometry`)

**Files:**
- Create: `sim/track_geometry_adapter.py`
- Test: `tests/test_sim_scene.py` (дополнить классом)

- [ ] **Step 1: Написать падающий тест**

```python
class TestTrack(unittest.TestCase):
    def setUp(self):
        self.p = scene.SceneParams(seed=1, kind='straight', section='wide', length_m=40.0)
        self.s = scene.build_scene(self.p)
        self.rc = self.s.raycasting()

    def test_two_running_rails_at_gauge(self):
        lx, rx = self.s.track['rails_x']
        self.assertAlmostEqual(rx - lx, 1.520, delta=0.01)

    def test_rail_head_is_16_cm_above_the_floor(self):
        y = -10.0
        x = self.s.track['rails_x'][0]
        rays = np.array([[x, y, 1.0, 0.0, 0.0, -1.0]], dtype=np.float32)
        res = self.rc.cast_rays(o3d.core.Tensor(rays))
        hit_z = 1.0 - float(res['t_hit'].numpy()[0])
        self.assertAlmostEqual(hit_z, scene.floor_z(self.p, y) + 0.16, delta=0.03)

    def test_profile_comes_from_track_geometry_module(self):
        import track_geometry
        verts, sources = track_geometry.rail_profile_mm(73.0)
        self.assertGreater(len(verts), 8)
        self.assertIn('gost R65', sources)
```

- [ ] **Step 2: Запустить — упадёт** (`track_geometry_adapter` отсутствует)

- [ ] **Step 3: Реализовать `sim/track_geometry_adapter.py`**

```python
"""Рельсы и шпалы для сцены: контур R65 берём у `track_geometry`, не дублируем.

`track_geometry.rail_profile_mm(head_top_width_mm)` отдаёт замкнутый контур в
локальной системе (u наружу, v вверх от низа подошвы, мм). Здесь контур
выдавливается вдоль пути и садится в мир так, чтобы верх головки оказался на
`floor + RAIL_HEIGHT` (0.16 м) -- как в наших зонах.
"""
from __future__ import annotations

import numpy as np
import open3d as o3d

import track_geometry
import zones

RAIL_HEAD_TOP_WIDTH_MM = 73.0
SLEEPER_SPACING_M = 0.55


def rail_mesh(x_center: float, y_from: float, y_to: float, z_at, out_sign: int,
              step_m: float = 2.0) -> o3d.geometry.TriangleMesh:
    verts_mm, _sources = track_geometry.rail_profile_mm(RAIL_HEAD_TOP_WIDTH_MM)
    v_top = max(v for _u, v in verts_mm)
    ys = np.arange(y_from, y_to + 1e-9, step_m)
    sections = []
    for y in ys:
        z_base = z_at(y) + zones.RAIL_HEIGHT - v_top / 1000.0
        ring = []
        for u_mm, v_mm in verts_mm:
            x = x_center + out_sign * (u_mm / 1000.0)
            z = z_base + v_mm / 1000.0
            ring.append((x, y, z))
        sections.append(ring)
    verts, tris = [], []
    m = len(sections[0])
    for i, ring in enumerate(sections):
        verts.extend(ring)
        if i:
            for k in range(m):
                a = (i - 1) * m + k
                b = (i - 1) * m + (k + 1) % m
                c = i * m + k
                d = i * m + (k + 1) % m
                tris.append((a, b, c))
                tris.append((b, d, c))
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(verts))
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(tris))
    mesh.compute_vertex_normals()
    return mesh


def sleeper_mesh(x_center: float, y: float, z: float) -> o3d.geometry.TriangleMesh:
    """Шпала по ГОСТ 78-2004 (тип I): 2750 x 250 x 150 мм, утоплена в балласт."""
    box = o3d.geometry.TriangleMesh.create_box(2.75, 0.25, 0.15)
    box.translate((x_center - 1.375, y - 0.125, z - 0.10))
    return box


def add_track(parts: list, p) -> dict:
    """Дописать в сцену балласт, шпалы и два ходовых рельса нашего пути."""
    from sim.scene import GAUGE_M, floor_z
    axis = p.track_axis_x
    left_x, right_x = axis - GAUGE_M / 2.0, axis + GAUGE_M / 2.0
    z_at = lambda y: floor_z(p, y)  # noqa: E731
    y_from, y_to = -p.length_m, p.length_m

    for y in np.arange(y_from, y_to, SLEEPER_SPACING_M):
        parts.append(sleeper_mesh(axis, float(y), float(z_at(y))))
    parts.append(rail_mesh(left_x, y_from, y_to, z_at, out_sign=-1))
    parts.append(rail_mesh(right_x, y_from, y_to, z_at, out_sign=+1))
    return {'axis_x': axis, 'gauge': GAUGE_M, 'rails_x': [left_x, right_x]}
```

- [ ] **Step 4: Запустить тесты сцены** — Expected: OK

- [ ] **Step 5: Коммит**

```bash
git add sim/track_geometry_adapter.py tests/test_sim_scene.py
git commit -m "feat(sim): рельсы R65 и шпалы по ГОСТ в сцене"
```

---

## Task 4: Объекты и габарит

**Files:**
- Create: `sim/objects.py`
- Test: `tests/test_sim_objects.py`

- [ ] **Step 1: Написать падающий тест**

```python
"""Тесты библиотеки объектов: меши, материалы, флаг «в габарите»."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim import objects, scene  # noqa: E402


class TestObjectLibrary(unittest.TestCase):
    def test_person_and_suitcase_are_parameterized(self):
        self.assertIn('person', objects.OBJECT_LIBRARY)
        self.assertIn('suitcase', objects.OBJECT_LIBRARY)
        person = objects.OBJECT_LIBRARY['person'](height_m=1.75)
        self.assertAlmostEqual(person.size_m[2], 1.75, delta=0.01)
        self.assertIsNotNone(person.mesh)
        self.assertGreater(person.reflectivity, 0.0)

    def test_placement_on_the_rail_and_in_the_gauge(self):
        sp = scene.SceneParams(seed=7, kind='straight', section='wide', length_m=60.0)
        s = scene.build_scene(sp)
        specs = [objects.Placement('person', y_along_m=-30.0, lateral_m=0.0,
                                   placement='in_gauge')]
        objs = objects.place_objects(sp, s.track, specs)
        self.assertEqual(len(objs), 1)
        o = objs[0]
        self.assertAlmostEqual(o['y_along_m'], -30.0, delta=0.01)
        self.assertTrue(o['in_gabarit'], 'человек в колее -- внутри габарита')

    def test_object_outside_the_gabarit_is_flagged(self):
        sp = scene.SceneParams(seed=7, kind='straight', section='wide', length_m=60.0)
        s = scene.build_scene(sp)
        specs = [objects.Placement('suitcase', y_along_m=-20.0, lateral_m=3.2,
                                   placement='by_wall')]
        objs = objects.place_objects(sp, s.track, specs)
        self.assertFalse(objs[0]['in_gabarit'], 'чемодан в 3.2 м от пути -- вне габарита')


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2: Запустить — упадёт**

- [ ] **Step 3: Реализовать `sim/objects.py`**

```python
"""Библиотека объектов на пути: параметрические меши, материал, размещение.

Габарит для флага `in_gabarit` -- тот же, что в детекторе: полуширина 1.36 м
(полуширина самого широкого вагона метро) и высота 0.10..3.70 м над головкой
рельса (габарит М, ГОСТ 23961-80).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import open3d as o3d

import zones
from sim.scene import GAUGE_M, floor_z

GABARIT_HALF_WIDTH_M = 1.36
GABARIT_Z_LOW_M, GABARIT_Z_HIGH_M = 0.10, 3.70


@dataclass
class ObjectMesh:
    mesh: o3d.geometry.TriangleMesh
    size_m: tuple
    reflectivity: float


def _person(height_m: float = 1.75, width_m: float = 0.46) -> ObjectMesh:
    """Человек: цилиндр туловища + сфера головы. Отражательная способность -- одежда."""
    body_h = height_m * 0.82
    body = o3d.geometry.TriangleMesh.create_cylinder(radius=width_m / 2.0, height=body_h)
    body.translate((0.0, 0.0, body_h / 2.0))
    head = o3d.geometry.TriangleMesh.create_sphere(radius=width_m / 2.2)
    head.translate((0.0, 0.0, body_h + width_m / 2.2))
    mesh = body + head
    mesh.compute_vertex_normals()
    return ObjectMesh(mesh=mesh, size_m=(width_m, width_m, height_m), reflectivity=0.35)


def _suitcase(length_m: float = 0.55, height_m: float = 0.70, depth_m: float = 0.25) -> ObjectMesh:
    box = o3d.geometry.TriangleMesh.create_box(length_m, depth_m, height_m)
    box.translate((-length_m / 2.0, -depth_m / 2.0, 0.0))
    box.compute_vertex_normals()
    return ObjectMesh(mesh=box, size_m=(length_m, depth_m, height_m), reflectivity=0.20)


OBJECT_LIBRARY = {'person': _person, 'suitcase': _suitcase}


@dataclass
class Placement:
    cls: str
    y_along_m: float
    lateral_m: float = 0.0            # смещение от оси пути, м
    placement: str = 'in_gauge'       # in_gauge | on_rail | by_wall
    yaw_deg: float = 0.0


def _in_gabarit(lateral_from_axis: float, z_rel_rail_head: float, height_m: float) -> bool:
    """Пересекается ли бокс объекта с габаритом поезда."""
    if abs(lateral_from_axis) - 0.0 > GABARIT_HALF_WIDTH_M:
        return False
    top = z_rel_rail_head + height_m
    return top > GABARIT_Z_LOW_M and z_rel_rail_head < GABARIT_Z_HIGH_M


def place_objects(params, track: dict, specs: list) -> list:
    """Разместить объекты по правилам и отдать их описание с боксами."""
    out = []
    for i, spec in enumerate(specs, start=1):
        maker = OBJECT_LIBRARY[spec.cls]
        om = maker()
        x = track['axis_x'] + spec.lateral_m
        if spec.placement == 'on_rail':
            x = track['rails_x'][0] if spec.lateral_m <= 0 else track['rails_x'][1]
        z_floor = float(floor_z(params, spec.y_along_m))
        z = z_floor + zones.RAIL_HEIGHT          # объект стоит на уровне головки рельса
        om.mesh.translate((x, spec.y_along_m, z))
        if spec.yaw_deg:
            om.mesh.rotate(o3d.geometry.get_rotation_matrix_from_xyz(
                (0.0, 0.0, np.radians(spec.yaw_deg))), center=(x, spec.y_along_m, z))
        sx, sy, sz = om.size_m
        half = (sx / 2.0, sy / 2.0, sz / 2.0)
        out.append({
            'id': i, 'class': spec.cls,
            'bbox_xyz': [[x - half[0], spec.y_along_m - half[1], z],
                         [x + half[0], spec.y_along_m + half[1], z + sz]],
            'y_along_m': float(spec.y_along_m),
            'offset_from_axis_m': float(x - track['axis_x']),
            'in_gabarit': bool(_in_gabarit(x - track['axis_x'],
                                           z - z_floor - zones.RAIL_HEIGHT, sz)),
            'reflectivity': float(om.reflectivity),
            'frames_visible': [],
        })
    return out
```

- [ ] **Step 4: Запустить — Expected: OK**

- [ ] **Step 5: Коммит**

```bash
git add sim/objects.py tests/test_sim_objects.py
git commit -m "feat(sim): библиотека объектов на пути и флаг габарита"
```

---

## Task 5: Сканирование — лучи, окклюзия, интенсивность, шум

**Files:**
- Create: `sim/scanner.py`
- Test: `tests/test_sim_scanner.py`

- [ ] **Step 1: Написать падающий тест**

```python
"""Тесты сканирования: облако похоже на настоящее по геометрии и статистике."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim import objects, scanner, scene  # noqa: E402
from sim.sensor import SensorModel  # noqa: E402


def make(seed=3, obstacles=()):
    p = scene.SceneParams(seed=seed, kind='straight', section='wide', length_m=60.0)
    s = scene.build_scene(p)
    if obstacles:
        s.objects = objects.place_objects(p, s.track, list(obstacles))
        for o in s.objects:
            for part in scanner.object_meshes(o):
                s.mesh = s.mesh + part.to_tensor()
    sensor = SensorModel(azimuth_columns=200, range_sigma_m=0.01)  # быстрый тест
    return p, s, sensor


class TestScanner(unittest.TestCase):
    def test_point_count_is_rings_times_columns_minus_dropouts(self):
        p, s, sensor = make()
        f = scanner.scan(s, sensor, frame_idx=0, rng=np.random.default_rng(0))
        self.assertEqual(f.ring.shape[0], f.xyz.shape[0])
        self.assertLessEqual(f.xyz.shape[0], 128 * 200)
        self.assertGreater(f.xyz.shape[0], 128 * 200 * 0.90)

    def test_rings_span_the_measured_elevation_range(self):
        p, s, sensor = make()
        f = scanner.scan(s, sensor, frame_idx=0, rng=np.random.default_rng(0))
        r = np.hypot(f.xyz[:, 0], f.xyz[:, 1])
        elev = np.degrees(np.arctan2(f.xyz[:, 2], r))
        self.assertAlmostEqual(float(elev.max()), 14.40, delta=0.5)
        self.assertAlmostEqual(float(elev.min()), -25.12, delta=0.5)

    def test_an_object_in_front_changes_the_cloud(self):
        p, s, sensor = make()
        empty = scanner.scan(s, sensor, 0, np.random.default_rng(1))
        p2, s2, sensor2 = make(obstacles=[objects.Placement('person', -20.0, 0.0, 'in_gauge')])
        with_obj = scanner.scan(s2, sensor2, 0, np.random.default_rng(1))
        near = with_obj.xyz[np.hypot(with_obj.xyz[:, 0], with_obj.xyz[:, 1]) < 22.0]
        self.assertGreater(near.shape[0], 100, 'объект в 20 м должен дать возвраты')
        self.assertNotEqual(empty.xyz.shape[0], with_obj.xyz.shape[0])

    def test_intensity_is_in_byte_range_and_has_high_tail(self):
        p, s, sensor = make()
        f = scanner.scan(s, sensor, 0, np.random.default_rng(2))
        self.assertGreaterEqual(int(f.intensity.min()), 0)
        self.assertLessEqual(int(f.intensity.max()), 255)
        self.assertGreater(float(np.percentile(f.intensity, 90)), 5.0)


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2: Запустить — упадёт**

- [ ] **Step 3: Реализовать `sim/scanner.py`**

```python
"""Сканирование сцены: лучи по таблице сенсора, окклюзия трассировкой.

Трассировка -- `o3d.t.geometry.RaycastingScene`: затенение считается честно,
точки за препятствием в облако не попадают. Интенсивность -- модель от
отражательной способности, угла падения и дальности, параметры подобраны так,
чтобы статистика попадала в измеренный диапазон (медиана ~11, доля >25 -- 1..11 %).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d


@dataclass
class Frame:
    xyz: np.ndarray          # (N, 3) float32
    intensity: np.ndarray    # (N,) float32
    ring: np.ndarray         # (N,) uint16
    timestamp: float         # секунды от начала записи


def object_meshes(obj: dict) -> list:
    """Меши объекта по его описанию (нужны сцене для добавления в трассировку)."""
    return obj.get('_meshes', [])


def _directions(sensor) -> np.ndarray:
    """Единичные векторы всех лучей кадра: (rings*columns, 3)."""
    elev = np.radians(np.asarray(sensor.elevations_deg, dtype=np.float64))
    azim = np.radians(sensor.azimuths_deg())
    e, a = np.meshgrid(elev, azim, indexing='ij')
    ce, se = np.cos(e), np.sin(e)
    x = ce * np.sin(a)
    y = -ce * np.cos(a)          # вперёд это -Y, как в наших данных
    z = se
    return np.stack([x, y, z], axis=-1).reshape(-1, 3)


def scan(scene, sensor, frame_idx: int, rng: np.random.Generator,
         max_range_m: float = 210.0) -> Frame:
    rc = scene.raycasting()
    dirs = _directions(sensor)
    origin = np.zeros_like(dirs, dtype=np.float32)
    rays = np.concatenate([origin, dirs.astype(np.float32)], axis=1)
    res = rc.cast_rays(o3d.core.Tensor(rays))
    t = res['t_hit'].numpy().astype(np.float64)

    hit = np.isfinite(t) & (t > 0.5) & (t < max_range_m)
    # шум дальности (сенсор) + случайные пропуски возврата
    t = t + rng.normal(0.0, sensor.range_sigma_m, size=t.shape)
    hit &= rng.random(t.shape) > sensor.dropout

    dirs_h = dirs[hit]
    rng_m = t[hit]
    xyz = (dirs_h * rng_m[:, None]).astype(np.float32)
    ring = (np.nonzero(hit.reshape(sensor.rings, -1))[0]).astype(np.uint16)
    inten = _intensity(xyz, dirs_h, rng_m, sensor, rng)
    return Frame(xyz=xyz, intensity=inten, ring=ring,
                 timestamp=frame_idx / 10.0)


def _intensity(xyz, dirs, ranges, sensor, rng) -> np.ndarray:
    """Модель интенсивности: база сцены × отражательность × косинус угла падения.

    База подобрана по замерам: медиана сырых значений у наших записей 6..11,
    у 90-го процентиля 26, доля >25 -- 1..11 % (зависит от сцены).
    """
    base = 14.0
    k_range = np.exp(-ranges / 120.0)
    grazing = np.clip(np.abs(dirs[:, 2]), 0.08, 1.0) ** 0.5
    # отражательность сцены: стены/пол/рельсы -- разные коэффициенты
    refl = np.full(ranges.shape, 0.55)
    refl[np.abs(xyz[:, 0] - (-2.62)) < 0.4] = 0.75
    refl[np.abs(xyz[:, 0] - 6.62) < 0.4] = 0.75
    refl[xyz[:, 2] < -1.0] = 0.35               # пол/балласт темнее
    value = base * refl * grazing * k_range
    value *= rng.lognormal(mean=0.0, sigma=0.6, size=value.shape)   # разброс поверхностей
    return np.clip(value, 0.0, 255.0).astype(np.float32)
```

- [ ] **Step 4: Запустить — Expected: OK**

- [ ] **Step 5: Коммит**

```bash
git add sim/scanner.py tests/test_sim_scanner.py
git commit -m "feat(sim): сканирование сцены лучами с окклюзией и интенсивностью"
```

---

## Task 6: Запись `.db3` и round-trip через `bag_reader`

**Files:**
- Create: `sim/writer.py`
- Test: `tests/test_sim_writer.py`

- [ ] **Step 1: Написать падающий тест**

```python
"""Round-trip: записанный bag читается нашим же bag_reader без расхождений."""
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bag_reader  # noqa: E402
from sim import writer  # noqa: E402

TOPIC = '/sensing/lidar/hesai128/pointcloud'
TYPE = 'sensor_msgs/msg/PointCloud2'


def frame(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    return writer.FrameOut(
        xyz=rng.normal(0.0, 5.0, size=(n, 3)).astype(np.float32),
        intensity=rng.integers(0, 256, size=n).astype(np.float32),
        ring=rng.integers(0, 128, size=n).astype(np.uint16),
        timestamp=0.1)


class TestWriter(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix='simtest_')
        self.db = os.path.join(self.dir, 'sim_0.db3')

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_roundtrip_through_bag_reader(self):
        frames = [frame(seed=i) for i in range(3)]
        writer.write_bag(frames, self.db, topic=TOPIC, msg_type=TYPE, hz=10.0)
        fr = bag_reader.BagFrames(self.db, cache_size=2, prefetch_ahead=0)
        try:
            self.assertEqual(len(fr), 3)
            self.assertIn(TOPIC, fr.topic_names)
            back = fr[1]
            np.testing.assert_allclose(back.xyz, frames[1].xyz, rtol=1e-6)
            np.testing.assert_allclose(back.intensity, frames[1].intensity, rtol=1e-6)
            np.testing.assert_array_equal(back.ring, frames[1].ring)
        finally:
            fr.close()

    def test_timestamps_ascend_from_a_fixed_start(self):
        frames = [frame(seed=i) for i in range(4)]
        writer.write_bag(frames, self.db, topic=TOPIC, msg_type=TYPE, hz=10.0,
                         start_ns=1_700_000_000_000_000_000)
        fr = bag_reader.BagFrames(self.db, cache_size=1, prefetch_ahead=0)
        try:
            ts = fr.timestamps
            self.assertTrue(bool(np.all(np.diff(ts) > 0)))
            self.assertEqual(int(ts[0]), 1_700_000_000_000_000_000)
        finally:
            fr.close()

    def test_same_seed_gives_identical_file(self):
        a = os.path.join(self.dir, 'a.db3')
        b = os.path.join(self.dir, 'b.db3')
        writer.write_bag([frame(seed=5)], a, topic=TOPIC, msg_type=TYPE, hz=10.0,
                         start_ns=1)
        writer.write_bag([frame(seed=5)], b, topic=TOPIC, msg_type=TYPE, hz=10.0,
                         start_ns=1)
        with open(a, 'rb') as fa, open(b, 'rb') as fb:
            self.assertEqual(fa.read(), fb.read())


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2: Запустить — упадёт**

- [ ] **Step 3: Реализовать `sim/writer.py`**

```python
"""Запись синтетических кадров в rosbag2 `.db3` ровно в том виде, в каком их
читает `bag_reader`: CDR PointCloud2, point_step 26, поля x,y,z,intensity,ring,timestamp.

Схема SQLite -- как у rosbag2: `topics(id, name, type, serialization_format, ...)`
и `messages(id, topic_id, timestamp, data)`. Идентификатор темы 1.
"""
from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass

import numpy as np

POINT_STEP = 26
FIELDS = (('x', 0), ('y', 4), ('z', 8), ('intensity', 12), ('ring', 16), ('timestamp', 18))


@dataclass
class FrameOut:
    xyz: np.ndarray
    intensity: np.ndarray
    ring: np.ndarray
    timestamp: float


def _align4(n: int) -> int:
    return n + (-n % 4)


def encode_pointcloud2_cdr(frame: FrameOut, frame_id: str = 'lidar') -> bytes:
    """CDR little-endian PointCloud2: заголовок, поля, данные по 26 байт на точку."""
    n = int(frame.xyz.shape[0])
    body = bytearray()
    body += bytes([0x00, 0x01, 0x00, 0x00])                     # encapsulation
    body += b'\x00' * 8                                         # stamp (sec, nsec) -- по кадру
    name = frame_id.encode() + b'\x00'
    body += len(name).to_bytes(4, 'little') + name
    body += b'\x00' * (-len(body) % 4)
    body += (1).to_bytes(4, 'little')                           # height
    body += n.to_bytes(4, 'little')                             # width
    body += len(FIELDS).to_bytes(4, 'little')
    for i, (fname, off) in enumerate(FIELDS):
        nm = fname.encode() + b'\x00'
        body += len(nm).to_bytes(4, 'little') + nm
        body += b'\x00' * (-len(body) % 4)
        body += off.to_bytes(4, 'little')                       # offset
        body += (7 if i < 4 else (4 if i == 4 else 8)).to_bytes(1, 'little')  # datatype
        body += b'\x00' * (-len(body) % 4)
        body += (1).to_bytes(4, 'little')                       # count
    body += bytes([0])                                          # is_bigendian
    body += b'\x00' * (-len(body) % 4)
    body += POINT_STEP.to_bytes(4, 'little')
    body += (POINT_STEP * n).to_bytes(4, 'little')              # row_step
    body += (POINT_STEP * n).to_bytes(4, 'little')              # data_len

    cloud = np.zeros((n, POINT_STEP), dtype=np.uint8)
    cloud[:, 0:12] = frame.xyz.astype('<f4').view(np.uint8).reshape(n, 12)
    cloud[:, 12:16] = frame.intensity.astype('<f4').view(np.uint8).reshape(n, 4)
    cloud[:, 16:18] = frame.ring.astype('<u2').view(np.uint8).reshape(n, 2)
    cloud[:, 18:26] = np.float64(frame.timestamp).astype('<f8').view(np.uint8).reshape(1, 8)
    body += cloud.tobytes()
    return bytes(body)


def write_bag(frames: list, db_path: str, topic: str, msg_type: str, hz: float = 10.0,
              start_ns: int = 1_700_000_000_000_000_000, frame_id: str = 'lidar') -> str:
    """Записать кадры в `.db3`. Время отсчитывается от фиксированного начала."""
    if os.path.exists(db_path):
        os.remove(db_path)
    conn = sqlite3.connect(db_path)
    conn.executescript(
        'CREATE TABLE topics (id INTEGER PRIMARY KEY, name TEXT, type TEXT, '
        'serialization_format TEXT, offered_qos_profiles TEXT);'
        'CREATE TABLE messages (id INTEGER PRIMARY KEY, topic_id INTEGER, '
        'timestamp INTEGER, data BLOB);')
    conn.execute('INSERT INTO topics VALUES (1, ?, ?, ?, ?)', (topic, msg_type, 'cdr', ''))
    step_ns = int(round(1e9 / hz))
    for i, f in enumerate(frames):
        ts = int(start_ns) + i * step_ns
        conn.execute('INSERT INTO messages (topic_id, timestamp, data) VALUES (1, ?, ?)',
                     (ts, sqlite3.Binary(encode_pointcloud2_cdr(f, frame_id))))
    conn.commit()
    conn.close()
    return db_path
```

- [ ] **Step 4: Запустить — Expected: OK (3 теста)**

- [ ] **Step 5: Коммит**

```bash
git add sim/writer.py tests/test_sim_writer.py
git commit -m "feat(sim): запись синтетических кадров в .db3 (round-trip с bag_reader)"
```

---

## Task 7: Манифест со сценой и разметкой

**Files:**
- Create: `sim/manifest.py`
- Test: `tests/test_sim_writer.py` (дополнить)

- [ ] **Step 1: Написать падающий тест**

```python
class TestManifest(unittest.TestCase):
    def test_manifest_roundtrip_and_required_fields(self):
        from sim.manifest import Manifest, SceneInfo, ObjectInfo
        m = Manifest(seed=42, frames=5, hz=10.0,
                     scene=SceneInfo(kind='straight', section='wide', length_m=60.0,
                                     grade_pct=2.34, tracks=2,
                                     sensor_pose={'x': 0.0, 'y': 0.0, 'z': 1.82}),
                     objects=[ObjectInfo(id=1, cls='person',
                                         bbox_xyz=[[-0.2, -20.3, -1.6], [0.2, -19.7, 0.15]],
                                         y_along_m=-20.0, offset_from_axis_m=0.0,
                                         in_gabarit=True, frames_visible=[0, 1])])
        d = m.to_dict()
        for key in ('seed', 'frames', 'hz', 'scene', 'objects'):
            self.assertIn(key, d)
        self.assertEqual(d['objects'][0]['class'], 'person')
        self.assertTrue(d['objects'][0]['in_gabarit'])
        again = Manifest.from_dict(d)
        self.assertEqual(again.objects[0].y_along_m, -20.0)

    def test_manifest_is_written_next_to_the_bag(self):
        import json
        import tempfile
        from sim.manifest import Manifest, SceneInfo
        d = tempfile.mkdtemp(prefix='simtest_')
        path = Manifest(seed=1, frames=1, hz=10.0,
                        scene=SceneInfo(kind='straight', section='narrow', length_m=20.0,
                                        grade_pct=1.0, tracks=1, sensor_pose={'z': 1.32}),
                        objects=[]).save(d)
        with open(path, encoding='utf-8') as fh:
            data = json.load(fh)
        self.assertEqual(data['scene']['section'], 'narrow')
```

- [ ] **Step 2: Запустить — упадёт**

- [ ] **Step 3: Реализовать `sim/manifest.py`**

```python
"""Манифест датасета: сцена, сенсор, объекты с разметкой, кадры.

Разметка -- смысл всего упражнения: без неё синтетику нельзя использовать для
оценки детектора (precision/recall), только «посмотреть глазами».
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field


@dataclass
class SceneInfo:
    kind: str = 'straight'
    section: str = 'wide'
    length_m: float = 200.0
    grade_pct: float = 2.34
    tracks: int = 2
    sensor_pose: dict = field(default_factory=dict)
    tunnel: dict = field(default_factory=dict)


@dataclass
class ObjectInfo:
    id: int
    cls: str
    bbox_xyz: list
    y_along_m: float
    offset_from_axis_m: float
    in_gabarit: bool
    frames_visible: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d['class'] = d.pop('cls')
        return d

    @classmethod
    def from_dict(cls, d: dict) -> 'ObjectInfo':
        d = dict(d)
        d['cls'] = d.pop('class')
        return cls(**d)


@dataclass
class Manifest:
    seed: int
    frames: int
    hz: float
    scene: SceneInfo
    objects: list = field(default_factory=list)
    sensor: str = 'sensor_hesai128.json'

    def to_dict(self) -> dict:
        return {'seed': self.seed, 'frames': self.frames, 'hz': self.hz,
                'scene': asdict(self.scene),
                'objects': [o.to_dict() for o in self.objects],
                'sensor': self.sensor}

    @classmethod
    def from_dict(cls, d: dict) -> 'Manifest':
        return cls(seed=d['seed'], frames=d['frames'], hz=d['hz'],
                   scene=SceneInfo(**d['scene']),
                   objects=[ObjectInfo.from_dict(o) for o in d.get('objects', [])],
                   sensor=d.get('sensor', 'sensor_hesai128.json'))

    def save(self, out_dir: str, name: str = 'manifest.json') -> str:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, name)
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(self.to_dict(), fh, ensure_ascii=False, indent=1)
        return path
```

- [ ] **Step 4: Запустить — Expected: OK**

- [ ] **Step 5: Коммит**

```bash
git add sim/manifest.py tests/test_sim_writer.py
git commit -m "feat(sim): манифест датасета с разметкой объектов"
```

---

## Task 8: CLI — сборка датасета

**Files:**
- Create: `sim/cli.py`
- Test: `tests/test_sim_pipeline.py` (первый тест)

- [ ] **Step 1: Написать падающий тест**

```python
"""Сквозные тесты: CLI собирает датасет, наш конвейер его читает."""
import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bag_reader  # noqa: E402
from sim import cli  # noqa: E402


class TestCliDataset(unittest.TestCase):
    def setUp(self):
        self.out = tempfile.mkdtemp(prefix='simds_')

    def tearDown(self):
        shutil.rmtree(self.out, ignore_errors=True)

    def test_dataset_has_bag_manifest_and_obstacle(self):
        path = cli.make_dataset(out_dir=self.out, seed=42, bags=1, frames=3,
                                section='wide', columns=200, obstacles=[
                                    cli.ObstacleSpec('person', -20.0, 0.0)])
        bag_dir = path['bags'][0]
        self.assertTrue(os.path.isdir(bag_dir))
        manifest = json.load(open(os.path.join(bag_dir, 'manifest.json'), encoding='utf-8'))
        self.assertEqual(manifest['seed'], 42)
        self.assertEqual(manifest['frames'], 3)
        self.assertEqual(manifest['objects'][0]['class'], 'person')
        self.assertAlmostEqual(manifest['objects'][0]['y_along_m'], -20.0, delta=0.01,
                               msg='разметка должна совпадать с местом установки')
        fr = bag_reader.BagFrames(bag_reader.find_db3(bag_dir), cache_size=2, prefetch_ahead=0)
        try:
            self.assertEqual(len(fr), 3)
            self.assertIsNotNone(fr[0].ring)
        finally:
            fr.close()
```

- [ ] **Step 2: Запустить — упадёт**

- [ ] **Step 3: Реализовать `sim/cli.py`**

```python
"""Сборка датасета: сенсор из настоящей записи + сцены + объекты + запись.

Примеры:
  python -m sim.cli sensor --bag for_hackathon/doubleT_obstacle --out sim/sensor_hesai128.json
  python -m sim.cli dataset --out .scratch/sim_out --seed 42 --bags 2 --frames 5 \
      --section wide --obstacle person:-20:0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass

import numpy as np

import bag_reader
from sim import objects, scene, writer
from sim.manifest import Manifest, ObjectInfo, SceneInfo
from sim.scanner import scan
from sim.sensor import SensorModel


@dataclass
class ObstacleSpec:
    cls: str
    y_along_m: float
    lateral_m: float = 0.0
    placement: str = 'in_gauge'


def parse_obstacle(text: str) -> ObstacleSpec:
    cls, y, lat = (text.split(':') + ['0'])[:3]
    return ObstacleSpec(cls, float(y), float(lat))


def make_dataset(out_dir: str, seed: int, bags: int, frames: int, section: str = 'wide',
                 columns: int | None = None, obstacles: list | None = None,
                 length_m: float = 60.0) -> dict:
    """Сгенерировать `bags` записей по `frames` кадров. Возвращает пути к багам."""
    obstacles = obstacles or []
    sensor_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'sensor_hesai128.json')
    model = (SensorModel.load(sensor_path) if os.path.exists(sensor_path)
             else SensorModel(azimuth_columns=2709))
    if columns:
        model.azimuth_columns = int(columns)

    out_bags = []
    for b in range(bags):
        rng = np.random.default_rng(seed + b)
        params = scene.SceneParams(seed=seed + b, kind='straight', section=section,
                                  length_m=length_m)
        s = scene.build_scene(params)
        placed = objects.place_objects(params, s.track, obstacles) if obstacles else []
        for o in placed:
            for part in o.pop('_meshes', []):
                s.mesh = s.mesh + part.to_tensor()
        s.objects = placed

        frames_out, visible = [], {o['id']: [] for o in placed}
        for i in range(frames):
            f = scan(s, model, i, np.random.default_rng(seed + b * 1000 + i))
            frames_out.append(writer.FrameOut(f.xyz, f.intensity, f.ring, f.timestamp))
            for o in placed:
                lo, hi = o['bbox_xyz']
                inside = ((f.xyz[:, 1] >= lo[1] - 0.2) & (f.xyz[:, 1] <= hi[1] + 0.2)
                          & (f.xyz[:, 2] >= lo[2] - 0.2) & (f.xyz[:, 2] <= hi[2] + 0.2))
                if int(np.count_nonzero(inside)) >= 5:
                    visible[o['id']].append(i)

        bag_dir = os.path.join(out_dir, f'sim_{section}_{seed + b:04d}')
        os.makedirs(bag_dir, exist_ok=True)
        db = os.path.join(bag_dir, os.path.basename(bag_dir) + '_0.db3')
        writer.write_bag(frames_out, db, topic='/sensing/lidar/hesai128/pointcloud',
                         msg_type='sensor_msgs/msg/PointCloud2', hz=10.0)
        infos = [ObjectInfo(id=o['id'], cls=o['class'], bbox_xyz=o['bbox_xyz'],
                            y_along_m=o['y_along_m'], offset_from_axis_m=o['offset_from_axis_m'],
                            in_gabarit=o['in_gabarit'],
                            frames_visible=visible[o['id']]) for o in placed]
        Manifest(seed=seed + b, frames=frames, hz=10.0,
                 scene=SceneInfo(kind=params.kind, section=params.section,
                                 length_m=length_m, grade_pct=float(params.grade_pct or 0.0),
                                 tracks=int(params.tracks or 1),
                                 sensor_pose=scene.SECTIONS[section] | {'x': 0.0, 'y': 0.0},
                                 tunnel={'width_m': scene.SECTIONS[section]['width_m']}),
                 objects=infos).save(bag_dir)
        out_bags.append(bag_dir)
    return {'bags': out_bags, 'sensor': model.to_dict()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='симулятор туннеля: сенсор и датасеты')
    sub = ap.add_subparsers(dest='cmd', required=True)

    p_s = sub.add_parser('sensor', help='снять модель сенсора с настоящей записи')
    p_s.add_argument('--bag', required=True)
    p_s.add_argument('--out', default='sim/sensor_hesai128.json')

    p_d = sub.add_parser('dataset', help='собрать датасет синтетических записей')
    p_d.add_argument('--out', required=True)
    p_d.add_argument('--seed', type=int, default=1)
    p_d.add_argument('--bags', type=int, default=1)
    p_d.add_argument('--frames', type=int, default=5)
    p_d.add_argument('--section', choices=sorted(scene.SECTIONS), default='wide')
    p_d.add_argument('--columns', type=int, default=None)
    p_d.add_argument('--length', type=float, default=60.0)
    p_d.add_argument('--obstacle', action='append', default=[],
                     help='класс:y:смещение, например person:-20:0')

    args = ap.parse_args(argv)
    if args.cmd == 'sensor':
        model = SensorModel.from_bag(bag_reader.find_db3(args.bag))
        model.save(args.out)
        print(f'сенсор: {model.rings} колец, {model.azimuth_columns} колонок, '
              f'шум {model.range_sigma_m:.3f} м -> {args.out}')
        return 0

    res = make_dataset(out_dir=args.out, seed=args.seed, bags=args.bags, frames=args.frames,
                       section=args.section, columns=args.columns,
                       obstacles=[parse_obstacle(t) for t in args.obstacle],
                       length_m=args.length)
    print(json.dumps(res, ensure_ascii=False, default=str)[:400])
    return 0


if __name__ == '__main__':
    sys.exit(main())
```

- [ ] **Step 4: Запустить — Expected: OK**

- [ ] **Step 5: Коммит**

```bash
git add sim/cli.py tests/test_sim_pipeline.py
git commit -m "feat(sim): CLI сборки датасета и модель сенсора как артефакт"
```

---

## Task 9: Сверка «сим похож на реал»

**Files:**
- Create: `sim/check_sim.py`
- Test: `tests/test_sim_pipeline.py` (дополнить)

- [ ] **Step 1: Написать падающий тест**

```python
class TestCheckSim(unittest.TestCase):
    def test_metrics_match_reference_within_tolerance(self):
        from sim import check_sim
        out = tempfile.mkdtemp(prefix='simchk_')
        try:
            cli.make_dataset(out_dir=out, seed=7, bags=1, frames=3, section='wide', columns=400)
            bag_dir = os.path.join(out, 'sim_wide_0007')
            rep = check_sim.compare(sim_bag=bag_dir, real_bag='for_hackathon/doubleT_obstacle',
                                    frames=3)
            self.assertIn('point_count', rep)
            self.assertLess(rep['floor_a_delta_m'], 0.05, 'пол должен совпасть по уровню')
            self.assertLess(rep['wall_x_delta_m'], 0.15, 'стены должны совпасть по X')
            self.assertIn('elevation_max_delta_deg', rep)
        finally:
            shutil.rmtree(out, ignore_errors=True)
```

- [ ] **Step 2: Запустить — упадёт**

- [ ] **Step 3: Реализовать `sim/check_sim.py`**

```python
"""Сверка синтетики с реальностью: без неё симулятор бесполезен.

Считаем: число точек на кадр, долю точек в полосе рельсов, положение пола и стен,
крайние углы места, покрытие азимута, статистику интенсивности. Числам из спеки
(«Приёмка», критерий 3) соответствуют допуски в `TOLERANCES`.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

import bag_reader
import zones

TOLERANCES = {'floor_a_delta_m': 0.05, 'wall_x_delta_m': 0.10,
              'intensity_median_rel': 0.20, 'point_count_rel': 0.25}


def _stats(db_path: str, frames: int = 5) -> dict:
    fr = bag_reader.BagFrames(db_path, cache_size=2, prefetch_ahead=0)
    try:
        idxs = np.linspace(0, len(fr) - 1, min(frames, len(fr))).astype(int)
        parts, rings, inten = [], [], []
        for i in idxs:
            f = fr[i]
            parts.append(f.xyz[:, :3].astype(np.float64))
            if f.ring is not None:
                rings.append(f.ring)
            if f.intensity is not None:
                inten.append(f.intensity)
    finally:
        fr.close()
    c = np.concatenate(parts)
    x, y, z = c[:, 0], c[:, 1], c[:, 2]
    hist, edges = np.histogram(x, bins=np.arange(-12, 14, 0.25))
    wall = float(0.5 * (edges[int(np.argmax(hist))] + edges[int(np.argmax(hist)) + 1]))
    a, b, rms = zones.fit_floor(c, y_min=-40.0, y_max=-2.0)
    r = np.hypot(x, y)
    elev = np.degrees(np.arctan2(z, r))
    out = {'points_per_frame': float(len(c) / len(parts)), 'wall_x': wall,
           'floor_a': float(a), 'floor_b': float(b), 'rms': float(rms),
           'elev_max_deg': float(np.percentile(elev, 99.5)),
           'elev_min_deg': float(np.percentile(elev, 0.5)),
           'range_p95_m': float(np.percentile(r, 95))}
    if inten:
        ii = np.concatenate(inten)
        out['intensity_median'] = float(np.median(ii))
        out['intensity_p90'] = float(np.percentile(ii, 90))
        out['intensity_share_gt25'] = float(np.mean(ii > 25))
    return out


def compare(sim_bag: str, real_bag: str, frames: int = 5) -> dict:
    sim = _stats(bag_reader.find_db3(sim_bag), frames)
    real = _stats(bag_reader.find_db3(real_bag), frames)
    rep = {f'sim_{k}': v for k, v in sim.items()}
    rep.update({f'real_{k}': v for k, v in real.items()})
    rep['floor_a_delta_m'] = abs(sim['floor_a'] - real['floor_a'])
    rep['wall_x_delta_m'] = abs(sim['wall_x'] - real['wall_x'])
    rep['elevation_max_delta_deg'] = abs(sim['elev_max_deg'] - real['elev_max_deg'])
    rep['point_count_rel'] = abs(sim['points_per_frame'] - real['points_per_frame']) \
        / max(real['points_per_frame'], 1.0)
    ok = {k: (rep[k] <= v) for k, v in TOLERANCES.items()}
    rep['tolerances_ok'] = ok
    rep['passed'] = all(ok.values())
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='сверка синтетики с реальной записью')
    ap.add_argument('--sim', required=True)
    ap.add_argument('--real', required=True)
    ap.add_argument('--frames', type=int, default=5)
    args = ap.parse_args(argv)
    rep = compare(args.sim, args.real, args.frames)
    for k in sorted(rep):
        print(f'  {k:26s} {rep[k]}')
    return 0 if rep['passed'] else 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
```

- [ ] **Step 4: Запустить — Expected: OK** (или отчёт с расхождениями — тогда правим сцену/шум, а не тест)

- [ ] **Step 5: Коммит**

```bash
git add sim/check_sim.py tests/test_sim_pipeline.py
git commit -m "feat(sim): сверка синтетики с реальной записью по метрикам"
```

- [ ] **Step 6: Прогнать сверку по обеим семьям записей (критерий 3 спеки)**

Широкая семья и узкая — разные сцены, поэтому сверяем с двумя эталонами:

```bash
"C:/Users/K4ler/AppData/Local/Programs/Python/Python312/python.exe" -m sim.check_sim \
  --sim .scratch/sim_out/sim_wide_0042 --real for_hackathon/doubleT_obstacle --frames 5
"C:/Users/K4ler/AppData/Local/Programs/Python/Python312/python.exe" -m sim.check_sim \
  --sim .scratch/sim_out/sim_narrow_0042 --real for_hackathon/roundT_doubleT --frames 5
```

Expected: в обоих отчётах `passed: True`; расхождения по полу < 0.05 м, по стенам < 0.1 м,
по интенсивности — медиана в пределах ±20 %. Если расходится — правим сцену/модель
интенсивности, а не допуски.

---

## Task 10: Наш конвейер работает на синтетике

**Files:**
- Test: `tests/test_sim_pipeline.py` (дополнить)

- [ ] **Step 1: Написать падающий тест**

```python
class TestOurPipelineOnSynthetic(unittest.TestCase):
    def test_zones_find_floor_and_walls_in_the_synthetic_bag(self):
        out = tempfile.mkdtemp(prefix='simp_')
        try:
            cli.make_dataset(out_dir=out, seed=11, bags=1, frames=3, section='wide',
                             columns=400)
            bag = bag_reader.find_db3(os.path.join(out, 'sim_wide_0011'))
            fr = bag_reader.BagFrames(bag, cache_size=2, prefetch_ahead=0)
            try:
                c = np.concatenate([fr[i].xyz[:, :3].astype(np.float64) for i in range(3)])
            finally:
                fr.close()
            walls = zones.detect_walls(c, zones.ZoneParams(floor_a=-1.82, floor_b=0.0234))
            self.assertGreaterEqual(len(walls), 2)
            self.assertAlmostEqual(min(walls), -2.62, delta=0.2)
            self.assertAlmostEqual(max(walls), 6.62, delta=0.2)
            labels = zones.classify(c, zones.ZoneParams(floor_a=-1.82, floor_b=0.0234,
                                                        axis_x=0.0, wall_x=tuple(walls)))
            counts = np.bincount(labels, minlength=len(zones.ZONE_NAMES))
            self.assertGreater(counts[zones.ZONE_NAMES.index('wall')], 0)
            self.assertGreater(counts[zones.ZONE_NAMES.index('bed')], 0)
        finally:
            shutil.rmtree(out, ignore_errors=True)
```

- [ ] **Step 2: Запустить — упадёт** (пока сцена не даст стены/пол in range)

- [ ] **Step 3: Починить сцену/сканер, пока тест не пройдёт**

Диагностика (печатает, что реально попало в облако и куда сели стены):

```bash
"C:/Users/K4ler/AppData/Local/Programs/Python/Python312/python.exe" - <<'PY'
import numpy as np, bag_reader, zones
fr = bag_reader.BagFrames(bag_reader.find_db3('.scratch/sim_out/sim_wide_0011'), cache_size=2, prefetch_ahead=0)
c = np.concatenate([fr[i].xyz[:, :3].astype(np.float64) for i in range(len(fr))])
fr.close()
h, e = np.histogram(c[:, 0], bins=np.arange(-12, 14, 0.25))
top = np.argsort(h)[-4:][::-1]
print('точек', len(c), '| плотные полосы X:', [round(0.5*(e[i]+e[i+1]), 2) for i in top])
print('Z: min %.2f max %.2f | Y: %.1f..%.1f' % (c[:,2].min(), c[:,2].max(), c[:,1].min(), c[:,1].max()))
print('стены детектором:', zones.detect_walls(c, zones.ZoneParams(floor_a=-1.82, floor_b=0.0234)))
PY
```

Типичные причины провала: стены вне `max_dist` (обрезать сцену или сдвинуть сенсор),
потолок перекрывает верхние лучи (опустить потолок), шум больше допуска (уменьшить
`range_sigma_m`), рельсы не дают возвратов (проверить `add_track`).

- [ ] **Step 4: Запустить — Expected: OK**

- [ ] **Step 5: Коммит**

```bash
git add tests/test_sim_pipeline.py sim/
git commit -m "test(sim): наш конвейер (зоны, стены, пол) работает на синтетике"
```

---

## Task 11: Семантика — объект ловится на своём месте

**Files:**
- Test: `tests/test_sim_pipeline.py` (дополнить)

- [ ] **Step 1: Написать падающий тест**

```python
class TestObstacleSemantics(unittest.TestCase):
    def test_object_is_found_at_its_own_y(self):
        out = tempfile.mkdtemp(prefix='sims_')
        try:
            cli.make_dataset(out_dir=out, seed=5, bags=1, frames=3, section='wide',
                             columns=400,
                             obstacles=[cli.ObstacleSpec('person', -20.0, 0.0)])
            bag_dir = os.path.join(out, 'sim_wide_0005')
            bag = bag_reader.find_db3(bag_dir)
            fr = bag_reader.BagFrames(bag, cache_size=2, prefetch_ahead=0)
            try:
                c = np.concatenate([fr[i].xyz[:, :3].astype(np.float64) for i in range(3)])
            finally:
                fr.close()
            p = zones.ZoneParams(floor_a=-1.82, floor_b=0.0234, axis_x=0.0,
                                 wall_x=(-2.62, 6.62))
            t = zones.tunnel_blocked(c, p, -40.0, 40.0)
            spans = t['spans']
            hit = [s for s in spans if s[0] <= -20.0 <= s[1]]
            self.assertTrue(hit, f'человек на y=-20 обязан попасть в коридор, участки {spans}')
            length = sum(b - a for a, b in spans)
            self.assertLess(length, 6.0, 'на пустом туннеле коридор должен быть почти свободен')
        finally:
            shutil.rmtree(out, ignore_errors=True)
```

- [ ] **Step 2: Запустить — упадёт или покажет, что детектор видит не то**

- [ ] **Step 3: Довести сцену/сканер/детектор до согласия** (если объект не ловится:
  проверить высоту объекта, пороги `min_points`/`min_height`, что объект вообще дал
  возвраты — `frames_visible` в манифесте)

- [ ] **Step 4: Запустить — Expected: OK**

- [ ] **Step 5: Коммит**

```bash
git add tests/test_sim_pipeline.py sim/
git commit -m "test(sim): объект на известном y детектируется коридором"
```

---

## Task 12: Производительность полного прогона (критерий 6 спеки)

**Files:**
- Modify: `sim/cli.py` (добавить вывод времени на кадр)

- [ ] **Step 1: Замерить полный датасет**

```bash
"C:/Users/K4ler/AppData/Local/Programs/Python/Python312/python.exe" - <<'PY'
import time, tempfile, shutil
from sim import cli
out = tempfile.mkdtemp(prefix='simperf_')
t0 = time.time()
res = cli.make_dataset(out_dir=out, seed=99, bags=1, frames=201, section='wide',
                       obstacles=[cli.ObstacleSpec('person', -20.0, 0.0)])
dt = time.time() - t0
print(f'201 кадр + запись: {dt:.1f} с ({dt/201*1000:.0f} мс на кадр)')
print('даг:', res['bags'][0])
shutil.rmtree(out, ignore_errors=True)
PY
```

- [ ] **Step 2: Сравнить с целью из спеки**

Цель — **не дольше 5 минут (300 с) на 201 кадр**. Если дольше, варианты по порядку:
уменьшить число колонок на кадр (тогда это фиксируется в манифесте и в сверке),
распараллелить кадры по процессам, упростить меши (меньше шпал в трассировке).

- [ ] **Step 3: Записать замер в README симулятора**

Создать `sim/README.md` с фактическими числами: сколько идёт кадр, сколько запись,
какой командой воспроизводится датасет и как проверяется сходство с реальностью.

- [ ] **Step 4: Коммит**

```bash
git add sim/README.md sim/cli.py
git commit -m "perf(sim): замер полного прогона 201 кадра и README симулятора"
```

---

## Что вне этого плана

- кривые и стрелки (фаза 2 спеки) — отдельный план после MVP;
- платформа и ворота (фаза 3);
- остальные классы объектов и движение сенсора (фаза 4);
- `score.py` — скорборд детектора (фаза 5);
- запись поточечных меток в `frames/frame_%04d.npz` — включается, когда понадобится
  обучение сегментации.

## Риски, за которыми следим по ходу

1. **Интенсивность** (Task 5): если распределение не попадёт в измеренный диапазон
   (медиана 6…11, доля >25 — 0.8…10.7 %), правим модель, а не подгоняем тест.
2. **Трассировка** (Task 5): 128 × 2709 лучей на кадр — если 201 кадр не укладывается
   в 5 минут, уменьшаем колонки в тестах и фиксируем это в манифесте.
3. **Стены и пол** (Task 10): синтетика обязана проходить наш же детект стен и фит пола
   с теми же допусками, что на реальных данных.