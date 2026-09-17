"""Параметрическая сцена туннеля: меши, путь, поза сенсора.

Числа взяты из замеров: широкий двухпутный туннель -- стены -2.62/+6.62 (9.24 м),
пол при сенсоре -1.82 и уклон 2.34 %; узкий однопутный -- ширина ~4.3 м, пол ~-1.32.
Разрез (параметры рельсов, шпал, балласта) -- в `build_track_geometry`.

Сечение строится прямым в локальной системе (пол z=0, потолок `TUNNEL_HEIGHT_M`)
и наклоняется целиком на уклон: уклон в записи -- это наклон вагона относительно
сцены, то есть в системе сенсора наклонено всё сечение, а не только пол. Пол
отдельно взят сеткой прямо по `floor_z`, поэтому трассировка попадает в него точно,
без ступенек между сегментами плиты.
"""
from __future__ import annotations

import math
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

# Внутренние плоскости стен (x поперёк). Для широкого сечения это замер (-2.62/+6.62);
# по узким записям снята только ширина (4.3...5.0 м), положение стен в них не измерено,
# поэтому ставим замеренную ширину симметрично оси пути -- сенсор в узких записях над ним.
WALL_X_M = {'wide': (-2.62, 6.62), 'narrow': (-2.15, 2.15)}
WALL_THICKNESS_M = 0.2
WALL_BELOW_FLOOR_M = 0.4        # стена заходит под пол, чтобы не было щели на стыке
FLOOR_GRID_X, FLOOR_GRID_Y = 4, 60   # сетка плиты пола: 5x61 вершин (плоскость сама точная)


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
    """Уровень пола в системе сенсора: сенсор на высоте `sensor_z` над полом.

    Неразрешённые параметры (`grade_pct=None`) берут уклон из `SECTIONS`: сцену
    принято описывать коротко, а пол нужен и тем, кто строит объекты/рельсы.
    """
    s = SECTIONS[p.section]
    grade_pct = s['grade_pct'] if p.grade_pct is None else p.grade_pct
    return -float(s['sensor_z']) + grade_pct / 100.0 * np.asarray(y, dtype=np.float64)


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


def to_tensor_mesh(part: o3d.geometry.TriangleMesh) -> o3d.t.geometry.TriangleMesh:
    """Меш в тензорный вид: только такие принимает `RaycastingScene`.

    У legacy-меша в установленном open3d 0.19 нет `.to_tensor()` -- конверсия
    делается фабрикой `from_legacy`.
    """
    return o3d.t.geometry.TriangleMesh.from_legacy(part)


def _box(cx, cy, cz, sx, sy, sz) -> o3d.geometry.TriangleMesh:
    b = o3d.geometry.TriangleMesh.create_box(width=sx, height=sy, depth=sz)
    b.translate((cx - sx / 2.0, cy - sy / 2.0, cz - sz / 2.0))
    return b


def _floor_mesh(p: SceneParams, x_left: float, x_right: float) -> o3d.geometry.TriangleMesh:
    """Плита пола: сетка, у которой z взят из `floor_z`, то есть ровно замеренная плоскость.

    Плоскость задаётся точно, поэтому разрешение сетки на геометрию не влияет --
    оно определяет только число треугольников (`FLOOR_GRID_X`x`FLOOR_GRID_Y`).
    """
    xs = np.linspace(x_left, x_right, FLOOR_GRID_X + 1)
    ys = np.linspace(-p.length_m, p.length_m, FLOOR_GRID_Y + 1)
    xx, yy = np.meshgrid(xs, ys, indexing='ij')
    positions = np.column_stack([xx.ravel(), yy.ravel(),
                                 np.asarray(floor_z(p, yy.ravel()), dtype=np.float64)])
    grid = np.arange(positions.shape[0], dtype=np.int32).reshape(xx.shape)
    quads = np.column_stack([grid[:-1, :-1].ravel(), grid[1:, :-1].ravel(),
                             grid[1:, 1:].ravel(), grid[:-1, 1:].ravel()])
    triangles = np.vstack([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]]).astype(np.int32)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(positions)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    return mesh


def _section_parts(p: SceneParams, x_left: float, x_right: float,
                   z0: float) -> list:
    """Стены и потолок: прямые плиты в локальной системе, потом наклон целиком."""
    axis_x = 0.5 * (x_left + x_right)
    width = x_right - x_left
    angle = math.atan(p.grade_pct / 100.0)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    # локальная длина больше: после наклона сцена должна остаться ровно length_m
    half_y = p.length_m / cos_a
    rotation = np.array([[1.0, 0.0, 0.0],
                         [0.0, cos_a, -sin_a],
                         [0.0, sin_a, cos_a]])
    wall_center_z = (TUNNEL_HEIGHT_M + WALL_BELOW_FLOOR_M) / 2.0
    parts = [
        # стена ставится наружу от замеренной плоскости: лидар видит ровно x стены
        _box(x_left - WALL_THICKNESS_M / 2.0, 0.0, wall_center_z,
             WALL_THICKNESS_M, 2.0 * half_y, TUNNEL_HEIGHT_M + WALL_BELOW_FLOOR_M),
        _box(x_right + WALL_THICKNESS_M / 2.0, 0.0, wall_center_z,
             WALL_THICKNESS_M, 2.0 * half_y, TUNNEL_HEIGHT_M + WALL_BELOW_FLOOR_M),
        _box(axis_x, 0.0, TUNNEL_HEIGHT_M + WALL_THICKNESS_M / 2.0,
             width, 2.0 * half_y, WALL_THICKNESS_M),
    ]
    for part in parts:
        part.rotate(rotation, center=(0.0, 0.0, 0.0))
        part.translate((0.0, 0.0, z0))
    return parts


def build_scene(params: SceneParams) -> Scene:
    """Собрать сцену: пол, потолок, две стены, рельсы и шпалы нашего пути."""
    p = params.resolved()
    s = SECTIONS[p.section]
    x_left, x_right = WALL_X_M[p.section]
    z0 = float(floor_z(p, 0.0))

    parts = [_floor_mesh(p, x_left, x_right)]
    parts += _section_parts(p, x_left, x_right, z0)

    from sim.track_geometry_adapter import add_track        # Task 3 наполнит модуль
    track = add_track(parts, p)

    mesh = o3d.geometry.TriangleMesh()
    for part in parts:
        mesh += part
    mesh.compute_vertex_normals()
    return Scene(params=p, mesh=to_tensor_mesh(mesh), track=track,
                 sensor_pose={'x': 0.0, 'y': 0.0, 'z': float(s['sensor_z']),
                              'pitch_deg': 0.0})