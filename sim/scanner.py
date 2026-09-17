"""Сканирование сцены: лучи по таблице сенсора, окклюзия трассировкой, интенсивность.

Затенение честное -- `o3d.t.geometry.RaycastingScene`: возврат есть только у того
луча, который дошёл до поверхности первым, точки за препятствием в облако не
попадают. Лучи берутся из модели сенсора, а не выдуманы: `rings` углов места из
её таблицы (та снята с настоящей записи) на `azimuth_columns` азимутов полного
оборота. Система координат -- как в наших записях: X поперёк, Y вдоль, Z вверх,
вперёд это -Y.

Интенсивность -- модель от отражательной способности поверхности, косинуса угла
падения и дальности. Коэффициенты берутся у сцены: стены -- из `s.params.walls_x`,
пол -- по `floor_z_xy`, объекты -- из их `reflectivity`. Показатели степеней
подобраны по замерам настоящих записей: там интенсивность почти не зависит от
кольца (медиана 10..13 на всех углах места) и мягко падает с дальностью (медиана
17 на 2 м, 12 на 7 м, 8 на 30 м, 5 на 145 м, то есть примерно `exp(-r/120)`),
поэтому отражение и угол падения входят с малыми степенями. Итог даёт медиану
6..11, p90 20..30 и долю >25 в 0.7..10.8 %, как у шести наших записей.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d

from sim.scene import TUNNEL_HEIGHT_M, floor_z_xy, to_tensor_mesh

MAX_RANGE_M = 210.0             # замеренный максимум записей -- 208.7 м
MIN_RANGE_M = 1e-3              # отсечка самосовпадения луча с началом
FRAME_HZ = 10.0                 # частота кадров, как у `sim.writer.write_bag`

BASE_INTENSITY = 15.0
WALL_REFLECTIVITY = 0.75        # облицовка стены и свод
FLOOR_REFLECTIVITY = 0.35       # балласт и шпалы
TRACK_REFLECTIVITY = 0.55       # всё остальное в колее
RAIL_REFLECTIVITY = 0.90        # стальная головка рельса

FLOOR_BAND_M = 0.10             # допуск плоскости пола: шпалы +3 см, головка рельса +16 см
WALL_BAND_M = 0.35
RAIL_BAND_M = 0.15
CEILING_BAND_M = 0.60
OBJECT_BAND_M = 0.05

INCIDENCE_MIN = 0.08            # предел скользящего падения: дальше луч видит только муть
INCIDENCE_POWER = 0.3           # отклик сенсора сжат: в записях яркость почти не зависит от угла
REFLECTIVITY_POWER = 0.6
RANGE_FALLOFF_M = 120.0
INTENSITY_SIGMA = 0.62          # длинный хвост ярких точек: p90 в 2.4 раза больше медианы


@dataclass
class Frame:
    xyz: np.ndarray          # (N, 3) float32
    intensity: np.ndarray    # (N,) float32
    ring: np.ndarray         # (N,) uint16
    timestamp: float         # секунды от начала записи


def object_meshes(obj: dict) -> list:
    """Меши объекта по его описанию (нужны сцене для добавления в трассировку)."""
    return obj.get('_meshes', [])


def add_objects(scene_obj) -> None:
    """Дописать меши всех объектов сцены в её меш -- иначе трассировка их не увидит.

    В open3d 0.19 тензорные меши не складываются (`+` не определён), а объекты
    отдают legacy-меши, поэтому склейка идёт в legacy-виде, и только готовая
    сборка переводится в тензорный вид (`scene.to_tensor_mesh`).
    """
    parts = [part for obj in (scene_obj.objects or ()) for part in object_meshes(obj)]
    if not parts:
        return
    merged = o3d.t.geometry.TriangleMesh.to_legacy(scene_obj.mesh)
    for part in parts:
        merged += part
    scene_obj.mesh = to_tensor_mesh(merged)


def _directions(sensor) -> np.ndarray:
    """Единичные векторы всех лучей кадра: (rings*columns, 3), порядок -- по кольцам.

    Порядок важен: `ring` в облаке это номер строки, то есть номер угла места в
    таблице сенсора, и облако остаётся разложимым по кольцам.
    """
    elev = np.radians(np.asarray(sensor.elevations_deg, dtype=np.float64))
    if elev.shape[0] != int(sensor.rings):
        raise ValueError(f'в модели сенсора {elev.shape[0]} углов места вместо {sensor.rings}; '
                         'возьмите модель из артефакта (sim/sensor_hesai128.json)')
    azim = np.radians(sensor.azimuths_deg())
    e, a = np.meshgrid(elev, azim, indexing='ij')
    ce, se = np.cos(e), np.sin(e)
    x = ce * np.sin(a)
    y = -ce * np.cos(a)          # вперёд это -Y, как в наших записях
    z = se
    return np.stack([x, y, z], axis=-1).reshape(-1, 3)


def scan(scene_obj, sensor, frame_idx: int, rng: np.random.Generator,
         max_range_m: float = MAX_RANGE_M) -> Frame:
    """Отсканировать сцену одним кадром: возвраты, интенсивность, кольца.

    Возвраты дальше `max_range_m` игнорируются -- у наших записей предел 208.7 м.
    К попаданиям добавляется шум дальности сенсора (`range_sigma_m`) и случайные
    пропуски возврата (`dropout`), поэтому число точек в кадре не равно
    `rings * azimuth_columns` в точности.
    """
    dirs = _directions(sensor)
    rc = scene_obj.raycasting()
    rays = np.concatenate([np.zeros(dirs.shape, dtype=np.float32),
                           dirs.astype(np.float32)], axis=1)
    res = rc.cast_rays(o3d.core.Tensor(rays))

    t = res['t_hit'].numpy().astype(np.float64)
    hit = np.isfinite(t) & (t > MIN_RANGE_M) & (t <= max_range_m)
    t = t + rng.normal(0.0, sensor.range_sigma_m, size=t.shape)
    hit &= rng.random(t.shape) > sensor.dropout

    dirs_h = dirs[hit]
    ranges = t[hit]
    xyz = (dirs_h * ranges[:, None]).astype(np.float32)
    ring = np.nonzero(hit.reshape(int(sensor.rings), -1))[0].astype(np.uint16)
    intensity = _intensity(xyz, dirs_h, ranges,
                           res['primitive_normals'].numpy()[hit], scene_obj, rng)
    return Frame(xyz=xyz, intensity=intensity, ring=ring, timestamp=frame_idx / FRAME_HZ)


def _cos_incidence(dirs: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """Косинус угла падения луча на поверхность: |скалярное произведение| / (1e-9 + |n|).

    Нормаль примитива у попавшего луча всегда есть (у промаха она нулевая), а
    знак не важен: поверхность для симулятора двусторонняя. Скользящее падение
    ограничено снизу -- иначе точка на предельном угле падения получила бы ноль.
    """
    n = np.asarray(normals, dtype=np.float64)
    norm = np.linalg.norm(n, axis=1) + 1e-9
    cos_inc = np.abs(np.einsum('ij,ij->i', dirs, n)) / norm
    return np.clip(cos_inc, INCIDENCE_MIN, 1.0)


def _reflectivity(xyz: np.ndarray, scene_obj) -> np.ndarray:
    """Отражательная способность поверхности под каждой точкой -- из геометрии сцены.

    Пол и балласт -- ниже плоскости `floor_z_xy` плюс допуск на шум и на шпалу
    (она лежит на 3 см выше пола): головка рельса выше на 16 см, поэтому допуск
    10 см разделяет их честно. Стены -- по `s.params.walls_x`, свод -- по высоте
    трубы. Объекты -- их собственная отражательность из библиотеки.
    """
    params = scene_obj.params
    if params.walls_x is None or params.sensor_z is None:
        params = params.resolved()
    x = xyz[:, 0].astype(np.float64)
    y = xyz[:, 1].astype(np.float64)
    z = xyz[:, 2].astype(np.float64)
    above_floor = z - np.asarray(floor_z_xy(params, x, y), dtype=np.float64)

    refl = np.full(x.shape, TRACK_REFLECTIVITY, dtype=np.float64)
    refl[above_floor <= FLOOR_BAND_M] = FLOOR_REFLECTIVITY
    for wall_x in params.walls_x:
        refl[np.abs(x - float(wall_x)) <= WALL_BAND_M] = WALL_REFLECTIVITY
    refl[above_floor >= TUNNEL_HEIGHT_M - CEILING_BAND_M] = WALL_REFLECTIVITY
    for rail_x in scene_obj.track.get('rails_x', ()):
        rail = (np.abs(x - float(rail_x)) <= RAIL_BAND_M) & (above_floor > FLOOR_BAND_M)
        refl[rail] = RAIL_REFLECTIVITY
    for obj in (scene_obj.objects or ()):
        lo, hi = obj['bbox_xyz']
        inside = ((x >= lo[0] - OBJECT_BAND_M) & (x <= hi[0] + OBJECT_BAND_M)
                  & (y >= lo[1] - OBJECT_BAND_M) & (y <= hi[1] + OBJECT_BAND_M)
                  & (z >= lo[2] - OBJECT_BAND_M) & (z <= hi[2] + OBJECT_BAND_M))
        refl[inside] = float(obj['reflectivity'])
    return refl


def _intensity(xyz: np.ndarray, dirs: np.ndarray, ranges: np.ndarray,
               normals: np.ndarray, scene_obj, rng: np.random.Generator) -> np.ndarray:
    """Интенсивность возврата: база × отражение × косинус угла падения × дальность × разброс.

    Разброс логнормальный: в записях хвост ярких точек длинный (p90 в 2.4 раза
    больше медианы), а сам разброс -- от неоднородности поверхностей, а не от
    одной лишь геометрии.
    """
    reflectivity = _reflectivity(xyz, scene_obj)
    cos_inc = _cos_incidence(dirs, normals)
    value = (BASE_INTENSITY * reflectivity ** REFLECTIVITY_POWER
             * cos_inc ** INCIDENCE_POWER * np.exp(-ranges / RANGE_FALLOFF_M)
             * rng.lognormal(mean=0.0, sigma=INTENSITY_SIGMA, size=ranges.shape))
    return np.clip(value, 0.0, 255.0).astype(np.float32)