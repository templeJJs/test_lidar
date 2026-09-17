"""Рельсы и шпалы для сцены: контур R65 берём у `track_geometry`, не дублируем.

`track_geometry.rail_profile_mm(head_top_width_mm)` отдаёт замкнутый контур в
локальной системе (u наружу, v вверх от низа подошвы, мм). Здесь контур
выдавливается вдоль пути и садится в мир так, чтобы верх головки оказался на
`floor + RAIL_HEIGHT` (0.16 м) -- как в наших зонах.

Путь берётся у сцены, а не из константы: поперёк -- `centre_x` (туннель умеет
поворачивать), по высоте -- `floor_z_xy` (пол с боковым наклоном). Поэтому рельс
лежит на полу и в прямом, и в повёрнутом туннеле; при `straight` и нулевом
наклоне всё сводится к замеренной прямой.
"""
from __future__ import annotations

import numpy as np
import open3d as o3d

import track_geometry
import zones

RAIL_HEAD_TOP_WIDTH_MM = 73.0
RAIL_STEP_M = 2.0                       # шаг колец вдоль пути (ровно шаг станций сцены)
SLEEPER_SPACING_M = 0.55
SLEEPER_ABOVE_FLOOR_M = 0.03            # сколько верха шпалы видно над полом
SLEEPER_WIDTH_M, SLEEPER_LENGTH_M, SLEEPER_HEIGHT_M = 2.75, 0.25, 0.15   # ГОСТ 78-2004, тип I


def _plane_slopes(z_at, x_center: float, y: float) -> tuple:
    """Наклоны пола под шпалой (dz/dx, dz/dy) -- пол в сцене локально плоский."""
    dz_dx = float(z_at(x_center + 1.0, y) - z_at(x_center, y))
    dz_dy = float(z_at(x_center, y + 1.0) - z_at(x_center, y))
    return dz_dx, dz_dy


def _plane_rotation(dz_dx: float, dz_dy: float) -> np.ndarray:
    """Поворот, укладывающий верх шпалы в плоскость пола `z = a*x + b*y + c`.

    Ортонормированный базис из касательных пола: он поворачивает бокс как жёсткое
    тело, поэтому шпала остаётся ГОСТ-овской, а не скошенной.
    """
    e1 = np.array([1.0, 0.0, dz_dx])
    e1 /= np.linalg.norm(e1)
    tangent_y = np.array([0.0, 1.0, dz_dy])
    e2 = tangent_y - e1 * float(tangent_y @ e1)
    e2 /= np.linalg.norm(e2)
    normal = np.cross(e1, e2)
    if normal[2] < 0.0:
        normal = -normal
    return np.column_stack([e1, e2, normal])


def rail_mesh(x_at, y_from: float, y_to: float, z_at, out_sign: int,
              step_m: float = RAIL_STEP_M) -> o3d.geometry.TriangleMesh:
    """Один ходовой рельс: профиль Р65 из `track_geometry`, выдавленный вдоль пути.

    `x_at(y)` -- ось рельса, `z_at(x, y)` -- пол под ним. Кольца профиля
    склеиваются по кругу (`(k+1) % m`), иначе вместо трубы получилась бы открытая
    лента. Пол -- плоскость с постоянным уклоном, шаг 2 м её не огрубляет.
    """
    verts_mm, _sources = track_geometry.rail_profile_mm(RAIL_HEAD_TOP_WIDTH_MM)
    v_top = max(v for _u, v in verts_mm)
    ys = np.arange(y_from, y_to + 1e-9, step_m)
    sections = []
    for y in ys:
        x_center = float(x_at(y))
        z_base = float(z_at(x_center, y)) + zones.RAIL_HEIGHT - v_top / 1000.0
        ring = []
        for u_mm, v_mm in verts_mm:
            x = x_center + out_sign * (u_mm / 1000.0)
            z = z_base + v_mm / 1000.0
            ring.append((x, float(y), z))
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


def sleeper_mesh(x_center: float, y: float, z_top: float,
                 dz_dx: float, dz_dy: float) -> o3d.geometry.TriangleMesh:
    """Шпала по ГОСТ 78-2004 (тип I): 2750 x 250 x 150 мм, утоплена в балласт.

    Верх шпалы выставляется в плоскость пола на 3 см (12 см уходят ниже): шпала
    лежит в балласте, но её верх виден лидару, а не тонет в плите. Проступание
    держим меньше допуска теста на пол -- луч вниз по оси пути тогда упирается в
    шпалу лишь на 3 см выше замеренной плоскости.
    """
    box = o3d.geometry.TriangleMesh.create_box(SLEEPER_WIDTH_M, SLEEPER_LENGTH_M,
                                               SLEEPER_HEIGHT_M)
    box.translate((-SLEEPER_WIDTH_M / 2.0, -SLEEPER_LENGTH_M / 2.0, -SLEEPER_HEIGHT_M))
    box.rotate(_plane_rotation(dz_dx, dz_dy), center=(0.0, 0.0, 0.0))
    box.translate((x_center, y, z_top))
    return box


def add_track(parts: list, p) -> dict:
    """Дописать в сцену шпалы и два ходовых рельса нашего пути."""
    from sim.scene import GAUGE_M, centre_x, floor_z_xy
    axis = float(p.track_axis_x)
    half = GAUGE_M / 2.0
    z_at = lambda x, y: float(floor_z_xy(p, x, y))                # noqa: E731
    axis_at = lambda y: float(centre_x(p, y))                     # noqa: E731
    x_at = lambda y, side: axis_at(y) + side * half               # noqa: E731
    y_from, y_to = -p.length_m, p.length_m

    for y in np.arange(y_from, y_to, SLEEPER_SPACING_M):
        y = float(y)
        x_center = axis_at(y)
        dz_dx, dz_dy = _plane_slopes(z_at, x_center, y)
        parts.append(sleeper_mesh(x_center, y, z_at(x_center, y) + SLEEPER_ABOVE_FLOOR_M,
                                  dz_dx, dz_dy))
    parts.append(rail_mesh(lambda y: x_at(y, -1), y_from, y_to, z_at, out_sign=-1))
    parts.append(rail_mesh(lambda y: x_at(y, +1), y_from, y_to, z_at, out_sign=+1))
    rails_x = [x_at(0.0, -1), x_at(0.0, +1)]
    return {'axis_x': axis, 'gauge': GAUGE_M, 'rails_x': rails_x}