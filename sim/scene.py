"""Параметрическая сцена туннеля: меши, ось пути, поза сенсора.

Числа взяты из замеров по нашим шести записям (`sim/dataset_facts.json`):
широкий двухпутный туннель 9.2 м (стены −2.65/+6.55), узкие однопутные 3.6…5.0 м,
пол под сенсором 1.36…1.86 м, уклон 0.48…1.94 %, дрейф оси 0.15…0.9 м на видимом
участке (это и есть амплитуда поворотов).

Сечение строится «трубой» по оси туннеля: на каждой станции `y` берётся контур
(пол -- две точки с боковым наклоном, стены, потолок) и соседние контуры сшиваются
в ленту. Благодаря этому туннель умеет поворачивать (`kind='curve'`), а поверхности
остаются точными: при `kind='straight'` и нулевом боковом наклоне пол совпадает с
`floor_z` ровно, стены стоят ровно на замеренных `x`.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np
import open3d as o3d

# Замеры по записям: ширина туннеля, пол под сенсором, уклон %
SECTIONS = {
    'wide':   {'width_m': 9.20, 'sensor_z': 1.82, 'grade_pct': 2.34, 'tracks': 2},
    'narrow': {'width_m': 4.30, 'sensor_z': 1.32, 'grade_pct': 1.00, 'tracks': 1},
    # 'facts' -- профиль туннеля берётся случайно из замеров по всем записям
    # (см. `fact_profiles`), а не из двух усреднённых семей
    'facts':  {'width_m': 5.00, 'sensor_z': 1.50, 'grade_pct': 1.00, 'tracks': 1},
}

FACTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset_facts.json')
GAUGE_M = 1.520
TUNNEL_HEIGHT_M = 4.10          # от пола до потолка
WALL_THICKNESS_M = 0.2          # оставлено для совместимости: стены строятся поверхностью

# Внутренние плоскости стен (x поперёк) для «эталонного» прямого туннеля.
WALL_X_M = {'wide': (-2.62, 6.62), 'narrow': (-2.15, 2.15)}

# Диапазоны для случайного туннеля -- всё из замеров, ничего выдуманного
GRADE_PCT_RANGE = (0.48, 1.94)
SENSOR_Z_RANGE = (1.36, 1.86)
WIDTH_RANGE = (3.60, 9.20)
CENTRE_DRIFT_RANGE = (0.15, 0.90)     # дрейф оси на видимом участке ~30 м
VISIBLE_LENGTH_M = 30.0               # на этой длине замерен дрейф
CANT_MM_RANGE = (0.0, 25.0)           # боковой наклон поверхности (возвышение), мм
FLOOR_ROUGHNESS_M = 0.060            # шероховатость ложа: у реальных записей rms 0.04-0.10
SIM_MEDIAN_AT_SCALE_1 = 9.5          # медиана интенсивности синтетики при масштабе 1 (замерено)
STATION_STEP_M = 2.0                  # шаг станций трубы: поверхности остаются точными


def fact_profiles(path: str = FACTS_PATH) -> list:
    """Профили туннелей, замеренные по ВСЕМ записям (sim/dataset_facts.json).

    Нужны, чтобы случайная сцена опиралась на все датасеты, а не на две
    усреднённые семьи: у нас шесть разных туннелей -- от двухпутного 9.2 м до
    однопутного 3.6 м, с разными уклонами и высотой подвеса сенсора.
    """
    try:
        with open(path, encoding='utf-8') as fh:
            data = json.load(fh)
    except OSError:
        return []
    out = []
    for name, f in (data.get('bags') or {}).items():
        walls = f.get('walls_x') or []
        if len(walls) < 2 or 'floor' not in f:
            continue
        out.append({'bag': name,
                    'walls_x': (float(walls[0]), float(walls[-1])),
                    'grade_pct': float(f['floor']['grade_pct']),
                    'sensor_z': round(-float(f['floor']['a']), 3),
                    'columns': f.get('azimuth_columns'),
                    # поле зрения прибора: у широкой записи 247°, у узкой 101°;
                    # без него сцена светит в хвостовой сектор, которого в записи нет
                    'azimuth_span_deg': f.get('azimuth_span_deg'),
                    # интенсивность у записей разная (медиана 6…12, доля >25 0.7…10.8 %):
                    # по ней сцена подбирает свой масштаб отражения
                    'intensity_median': (f.get('intensity') or {}).get('median'),
                    'share_gt25': (f.get('intensity') or {}).get('share_gt25')})
    return out


@dataclass
class SceneParams:
    seed: int
    kind: str = 'straight'          # straight | curve (случайный поворот)
    section: str = 'wide'           # wide | narrow | facts
    length_m: float = 200.0
    grade_pct: float = None         # None -> из SECTIONS (straight) или случайно (curve)
    tracks: int = None
    track_axis_x: float = None
    platform: bool = False
    gate: bool = False
    # поля случайного туннеля; None -> подставляются в resolved()
    sensor_z: float = None
    walls_x: tuple = None
    curve_radius_m: float = None
    curve_sign: int = None
    cant_mm: float = None
    # масштаб отражения: подбирается под профиль записи (медиана интенсивности
    # у записей 6…12, а синтетика при масштабе 1 даёт ~9.5)
    intensity_scale: float = None
    # и разброс: у записей хвост ярких точек разной толщины (доля >25 от 0.7 до
    # 10.8 %) -- медиана и хвост у них даже антикоррелированы
    intensity_sigma: float = None
    fact_bag: str = None            # запись, профиль которой повторяет сцена

    def resolved(self) -> 'SceneParams':
        s = SECTIONS[self.section]
        p = SceneParams(**{**self.__dict__})
        rng = np.random.default_rng(int(self.seed))
        randomize = self.kind != 'straight'
        # профиль можно задать явно (`fact_bag`): тогда сцена воспроизводит ИМЕННО
        # эту запись, без дрожания стен и уклона -- иначе сверять сцену с записью
        # нечем (стены уходят на 0.28 м при допуске 0.10)
        pinned = bool(self.fact_bag) and self.section == 'facts'

        # профиль из замеров по всем записям: в случайном режиме берём любую запись
        # и слегка сдвигаем её параметры, чтобы сцены были разными, но похожими
        prof = None
        if self.section == 'facts' and (pinned or randomize):
            profiles = fact_profiles()
            if profiles:
                if pinned:
                    prof = next((q for q in profiles if q['bag'] == self.fact_bag), None)
                    if prof is None:
                        raise ValueError(f'в замерах нет записи {self.fact_bag!r}; есть: '
                                         + ', '.join(q['bag'] for q in profiles))
                else:
                    prof = profiles[int(rng.integers(0, len(profiles)))]
                p.fact_bag = prof['bag']
                median = prof.get('intensity_median')
                if median:
                    # SIM_MEDIAN при масштабе 1 замерена сканером на широкой сцене
                    p.intensity_scale = min(max(float(median) / SIM_MEDIAN_AT_SCALE_1,
                                                0.35), 2.5)
                share = prof.get('share_gt25')
                if share is not None:
                    # хвост: доля точек выше 25 у записей 0.7…10.8 %, и она даже
                    # антикоррелирована с медианой -- поэтому подбираем отдельно
                    p.intensity_sigma = min(max(0.25 + 4.5 * float(share), 0.30), 0.95)

        if p.grade_pct is None:
            if pinned:
                p.grade_pct = prof['grade_pct']
            elif prof is not None:
                p.grade_pct = prof['grade_pct'] + float(rng.normal(0.0, 0.15))
                if rng.random() < 0.5:
                    p.grade_pct = -p.grade_pct
            elif randomize:
                p.grade_pct = (float(rng.uniform(*GRADE_PCT_RANGE))
                               * (1 if rng.random() < 0.5 else -1))
            else:
                p.grade_pct = s['grade_pct']
        if p.tracks is None:
            p.tracks = s['tracks']
        if p.track_axis_x is None:
            p.track_axis_x = 0.0
        if p.sensor_z is None:
            if pinned:
                p.sensor_z = prof['sensor_z']
            elif prof is not None:
                p.sensor_z = prof['sensor_z'] + float(rng.normal(0.0, 0.05))
            elif randomize:
                p.sensor_z = float(rng.uniform(*SENSOR_Z_RANGE))
            else:
                p.sensor_z = s['sensor_z']
        if p.walls_x is None:
            if pinned:
                p.walls_x = tuple(prof['walls_x'])
            elif prof is not None:
                wl, wr = prof['walls_x']
                # дрожание небольшое: сцена должна оставаться ТЕМ ЖЕ туннелем
                # (иначе стены расходятся с записью сильнее допуска 0.1 м),
                # разнообразие дают уклон, радиус поворота, наклон и яркость
                width = (wr - wl) * float(rng.uniform(0.97, 1.03))
                centre = 0.5 * (wl + wr) + float(rng.normal(0.0, 0.04))
                p.walls_x = (centre - width / 2.0, centre + width / 2.0)
            elif randomize:
                width = float(rng.uniform(*WIDTH_RANGE))
                centre = float(rng.uniform(-0.2, 0.2))
                p.walls_x = (centre - width / 2.0, centre + width / 2.0)
            else:
                p.walls_x = WALL_X_M[self.section]
        if p.curve_radius_m is None:
            if randomize:
                drift = float(rng.uniform(*CENTRE_DRIFT_RANGE))
                p.curve_radius_m = (VISIBLE_LENGTH_M ** 2) / (2.0 * drift)
                p.curve_sign = 1 if rng.random() < 0.5 else -1
            else:
                p.curve_radius_m = float('inf')
                p.curve_sign = 1
        if p.curve_sign is None:
            p.curve_sign = 1
        if p.cant_mm is None:
            if randomize:
                p.cant_mm = float(rng.uniform(*CANT_MM_RANGE)) * (1 if rng.random() < 0.5 else -1)
            else:
                p.cant_mm = 0.0
        return p


def _resolved(p: SceneParams) -> SceneParams:
    """Параметры с подставленными случайными полями.

    Нужно потому, что сцену принято описывать коротко (`SceneParams(seed=…)`), а
    `floor_z`/`centre_x` вызывают и те, кто не звал `resolved()`: у неразрешённых
    параметров случайные поля равны None, и без подстановки они падают. Вызов
    детерминирован: генератор создаётся заново от того же зерна.
    """
    if (p.grade_pct is None or p.sensor_z is None or p.walls_x is None
            or p.curve_radius_m is None or p.cant_mm is None):
        return p.resolved()
    return p


def floor_z(p: SceneParams, y):
    """Уровень пола ПО ОСИ пути в системе сенсора: сенсор на `sensor_z` над полом."""
    p = _resolved(p)
    return -float(p.sensor_z) + p.grade_pct / 100.0 * np.asarray(y, dtype=np.float64)


def centre_x(p: SceneParams, y):
    """Поперечное положение оси туннеля: прямая или дуга окружности.

    Дрейф подобран так, чтобы на `VISIBLE_LENGTH_M` он попадал в замеренный
    диапазон 0.1…0.9 м -- то есть радиусы получаются 500…4500 м, как в метро.
    """
    p = _resolved(p)
    y_arr = np.asarray(y, dtype=np.float64)
    if p.kind == 'straight' or not np.isfinite(p.curve_radius_m):
        return np.full_like(y_arr, float(p.track_axis_x))
    return float(p.track_axis_x) + p.curve_sign * (y_arr ** 2) / (2.0 * p.curve_radius_m)


def cant_slope(p: SceneParams) -> float:
    """Боковой наклон поверхности (возвышение, мм на половину колеи)."""
    p = _resolved(p)
    return float(p.cant_mm) / 1000.0 / max(GAUGE_M / 2.0, 1e-6)


def floor_z_xy(p: SceneParams, x, y):
    """Уровень пола в точке (x, y): плоскость по оси плюс боковой наклон."""
    p = _resolved(p)
    y_arr = np.asarray(y, dtype=np.float64)
    x_arr = np.asarray(x, dtype=np.float64)
    return floor_z(p, y_arr) + cant_slope(p) * (x_arr - centre_x(p, y_arr))


@dataclass
class Scene:
    params: SceneParams
    mesh: o3d.t.geometry.TriangleMesh
    track: dict                       # {'axis_x', 'gauge', 'rails_x'}
    sensor_pose: dict                 # {'x','y','z','pitch_deg'}
    objects: list = field(default_factory=list)

    def raycasting(self) -> o3d.t.geometry.RaycastingScene:
        rc = o3d.t.geometry.RaycastingScene()
        rc.add_triangles(self.mesh)
        return rc


def to_tensor_mesh(part: o3d.geometry.TriangleMesh) -> o3d.t.geometry.TriangleMesh:
    """Меш в тензорный вид: только такие принимает `RaycastingScene`.

    У legacy-меша в установленном open3d 0.19 нет `.to_tensor()`.
    """
    return o3d.t.geometry.TriangleMesh.from_legacy(part)


def stations(p: SceneParams) -> np.ndarray:
    """Станции трубы вдоль оси: шаг мелкий, чтобы поверхности были точными."""
    n = max(3, int(round(2.0 * p.length_m / STATION_STEP_M)) + 1)
    return np.linspace(-p.length_m, p.length_m, n)


def tube_mesh(p: SceneParams) -> o3d.geometry.TriangleMesh:
    """Труба туннеля: контуры на станциях, сшитые в ленту.

    Контур станции -- четыре точки: пол слева, пол справа, потолок справа,
    потолок слева. Пол берётся из `floor_z_xy` (то есть с боковым наклоном),
    поэтому при нулевом наклоне он ровно совпадает с `floor_z`.
    """
    xl0, xr0 = p.walls_x
    dx0 = float(centre_x(p, 0.0))
    verts, tris = [], []
    ys = stations(p)
    # шероховатость ложа: балласт лежит НА плоскости пола, а не колеблется вокруг
    # неё. Односторонним разбросом воспроизводится нижняя огибающая: наш фит пола
    # (`zones.fit_floor` -- 5-й перцентиль по бинам) у записей попадает ровно на
    # уровень пола, а симметричный шум тянул его на 0.05 м вниз (замерено:
    # floor_a −1.874 против −1.821 у doubleT_obstacle при допуске 0.05).
    rng = np.random.default_rng(int(p.seed) + 7919)
    for y in ys:
        dx = float(centre_x(p, y)) - dx0
        xl, xr = xl0 + dx, xr0 + dx
        zl = float(floor_z_xy(p, xl, y)) + float(rng.uniform(0.0, FLOOR_ROUGHNESS_M))
        zr = float(floor_z_xy(p, xr, y)) + float(rng.uniform(0.0, FLOOR_ROUGHNESS_M))
        ztop_l = zl + TUNNEL_HEIGHT_M
        ztop_r = zr + TUNNEL_HEIGHT_M
        ring = [(xl, y, zl), (xr, y, zr), (xr, y, ztop_r), (xl, y, ztop_l)]
        verts.extend(ring)
    m = 4
    for i in range(len(ys) - 1):
        for k in range(m):
            a = i * m + k
            b = i * m + (k + 1) % m
            c = (i + 1) * m + k
            d = (i + 1) * m + (k + 1) % m
            tris.append((a, c, b))
            tris.append((b, c, d))
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(verts, dtype=np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(tris, dtype=np.int32))
    return mesh


def build_scene(params: SceneParams) -> Scene:
    """Собрать сцену: труба туннеля, рельсы и шпалы нашего пути."""
    p = params.resolved()
    parts = [tube_mesh(p)]

    from sim.track_geometry_adapter import add_track
    track = add_track(parts, p)

    mesh = o3d.geometry.TriangleMesh()
    for part in parts:
        mesh += part
    mesh.compute_vertex_normals()
    return Scene(params=p, mesh=to_tensor_mesh(mesh), track=track,
                 sensor_pose={'x': 0.0, 'y': 0.0, 'z': float(p.sensor_z),
                              'pitch_deg': 0.0})