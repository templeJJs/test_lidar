"""Библиотека объектов на пути: параметрические меши, материал, размещение.

Габарит для флага `in_gabarit` берётся из `zones` -- это те же числа, по которым
размечает детектор, и дублировать их нельзя: полуширина `zones.TUNNEL_HALF_WIDTH`
(1.36 м -- полуширина самого широкого вагона метро) и высота
`zones.TUNNEL_Z_LOW..zones.TUNNEL_Z_HIGH` (0.10..3.70 м над головкой рельса,
габарит М, ГОСТ 23961-80). Объект стоит на уровне головки рельса, то есть на
`floor_z_xy(params, x, y) + zones.RAIL_HEIGHT`.

Поперёк всё отсчитывается от ОСИ ПУТИ на этом `y` (`scene.centre_x`), а не от
константы: туннель умеет поворачивать, и в повороте ось уходит вбок. В прямом
туннеле без бокового наклона это ровно `floor_z(params, y)` и `params.track_axis_x`
-- то есть замеренная прямая воспроизводится точка в точку.

Меши -- legacy `o3d.geometry.TriangleMesh`: в open3d 0.19 у них нет `.to_tensor()`,
в тензорный вид конвертирует `scene.to_tensor_mesh`. Каждое описание объекта несёт
служебный список `_meshes`, по которому сцена добавляет объект в трассировку
(`scanner.object_meshes`), а CLI вырезает ключ перед записью манифеста.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import open3d as o3d

import zones
from sim.scene import centre_x, floor_z_xy

GABARIT_HALF_WIDTH_M = zones.TUNNEL_HALF_WIDTH   # 1.36 м от оси пути
GABARIT_Z_LOW_M = zones.TUNNEL_Z_LOW             # 0.10 м над уровнем головки рельса
GABARIT_Z_HIGH_M = zones.TUNNEL_Z_HIGH           # 3.70 м над уровнем головки рельса


@dataclass
class ObjectMesh:
    mesh: o3d.geometry.TriangleMesh
    size_m: tuple
    reflectivity: float


def _person(height_m: float = 1.75, width_m: float = 0.46) -> ObjectMesh:
    """Человек: цилиндр туловища + сфера головы. Отражательная способность -- одежда.

    Голова упирается в объявленный верх, поэтому меш ровно `height_m` в высоту и
    `size_m` -- честный размер: по нему считается бокс в разметке.
    """
    head_r = width_m / 2.2
    body_h = height_m - 2.0 * head_r
    body = o3d.geometry.TriangleMesh.create_cylinder(radius=width_m / 2.0, height=body_h)
    body.translate((0.0, 0.0, body_h / 2.0))
    head = o3d.geometry.TriangleMesh.create_sphere(radius=head_r)
    head.translate((0.0, 0.0, height_m - head_r))
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


def _in_gabarit(lateral_from_axis: float, z_rel_rail_head: float, height_m: float,
                half_width_m: float = 0.0) -> bool:
    """Пересекается ли бокс объекта с габаритом поезда.

    `half_width_m` -- полуширина самого объекта: габарит это ПРОСТРАНСТВО, поэтому
    объект, стоящий центром у самой кромки, в габарит всё равно заходит. Центр без
    полуширины дал бы неверную разметку -- `in_gabarit=False` там, где детектор
    честно видит помеху в коридоре.
    """
    if abs(lateral_from_axis) - half_width_m > GABARIT_HALF_WIDTH_M:
        return False
    top = z_rel_rail_head + height_m
    return top > GABARIT_Z_LOW_M and z_rel_rail_head < GABARIT_Z_HIGH_M


def _half_extent_xy(sx: float, sy: float, yaw_deg: float) -> tuple:
    """Полуразмеры бокса поперёк (x) и вдоль (y) пути.

    Поворот вокруг Z разворачивает бокс в плане, поэтому AABB от него растёт:
    `bbox_xyz` -- разметка, и по ней решают, виден ли объект в кадре.
    """
    if not yaw_deg:
        return sx / 2.0, sy / 2.0
    a = math.radians(yaw_deg)
    cos_a, sin_a = abs(math.cos(a)), abs(math.sin(a))
    return ((cos_a * sx + sin_a * sy) / 2.0,
            (sin_a * sx + cos_a * sy) / 2.0)


def place_objects(params, track: dict, specs: list) -> list:
    """Разместить объекты по правилам и отдать их описание с боксами.

    Правила (`Placement.placement`): `in_gauge` -- от оси пути на `lateral_m`;
    `on_rail` -- по рельсу, выбранному знаком `lateral_m` (<=0 -- левый);
    `by_wall` -- тоже `lateral_m` от оси, но у стены (положение стен -- `scene.WALL_X_M`).
    """
    out = []
    for i, spec in enumerate(specs, start=1):
        om = OBJECT_LIBRARY[spec.cls]()
        axis_x = float(centre_x(params, spec.y_along_m))
        x = axis_x + spec.lateral_m
        if spec.placement == 'on_rail':
            half_gauge = track['gauge'] / 2.0
            x = axis_x - half_gauge if spec.lateral_m <= 0 else axis_x + half_gauge
        z_floor = float(floor_z_xy(params, x, spec.y_along_m))
        z = z_floor + zones.RAIL_HEIGHT          # объект стоит на уровне головки рельса
        om.mesh.translate((x, spec.y_along_m, z))
        if spec.yaw_deg:
            om.mesh.rotate(o3d.geometry.get_rotation_matrix_from_xyz(
                (0.0, 0.0, np.radians(spec.yaw_deg))), center=(x, spec.y_along_m, z))
        sx, sy, sz = om.size_m
        half_x, half_y = _half_extent_xy(sx, sy, spec.yaw_deg)
        offset = x - axis_x
        z_rel_rail_head = z - z_floor - zones.RAIL_HEIGHT
        out.append({
            'id': i, 'class': spec.cls,
            'bbox_xyz': [[x - half_x, spec.y_along_m - half_y, z],
                         [x + half_x, spec.y_along_m + half_y, z + sz]],
            'y_along_m': float(spec.y_along_m),
            'offset_from_axis_m': float(offset),
            'in_gabarit': bool(_in_gabarit(offset, z_rel_rail_head, sz, half_x)),
            'reflectivity': float(om.reflectivity),
            'frames_visible': [],
            '_meshes': [om.mesh],
        })
    return out