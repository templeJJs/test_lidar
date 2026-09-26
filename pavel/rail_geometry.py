"""Адаптер: детекция рельсов из test/track_geometry.py в контракте rail_detection.

Зачем
-----
`track_geometry.build_track` ведёт нить ПОЛИЛИНИЕЙ — узлами через 0.5 м, каждый
следующий ищется в окне от предсказания по предыдущим, с отбраковкой по MAD и
остановкой об стену. `rail_detection.find_rail_lines` фитит колею прямой или
параболой по парам пиков. На кривых полилиния не режет дугу хордой и не
«перескакивает» на стену, потому что коридор поиска едет за нитью, а не стоит
прямым в системе лидара.

Этот модуль отдаёт результат полилинии в том виде, какого ждут существующие
потребители (`track_map.py`, `rail_mapper.py`, `visualize_bag.py`), не трогая
ни `track_geometry.py`, ни `rail_detection.py`.

Что отдаётся
------------
`find_rail_lines(points, intensity, ...)` -> те же `(rail_lines, ground_plane)`
или `(rail_lines, ground_plane, pairs)` при `return_pairs=True`:

  * rail_lines — по нити dict {'knots_y', 'knots_x'}: между узлами np.interp.
    Узлы берутся ПРЯМО из полилинии, поэтому форма пути сохраняется как есть,
    без фита. Формат узлов понимает `rail_x` ЭТОГО модуля; текущий
    `rail_detection.rail_x` умеет только 2- и 3-кортежи (knots описаны в README
    и реализованы в old/rail_detection.py, но в актуальный файл не вернулись),
    поэтому для сторонних потребителей есть `patch_rail_detection()`;
  * ground_plane — (a, b, c) плоскости пола из build_track['floor'];
  * pairs — (N, 5) как (y, x_left, x_right, z_left, z_right), собранные из
    узлов двух нитей на общей сетке Y. Первые три столбца — те же «сырые
    измерения колеи на срезе», что и у прежней детекции (она отдаёт (N, 3)),
    поэтому сшивка в `track_map.py` работает без правок. Два последних —
    ИЗМЕРЕННАЯ высота верха головки в узле; прежняя детекция её не давала,
    и потребителю оставалась только плоскость земли.

ВАЖНО, колея по граням против колеи по осям
--------------------------------------------
Узлы полилинии track_geometry — это ОСИ нитей, между ними ~1587 мм
(gauge_axis, номинал 1594.6 = 1520 + ширина головки). Прежняя детекция давала
пики ГОЛОВОК, и `track_map.py` отбраковывает пары по `GAUGE = 1.52` с допуском
0.12 м — пары по осям он выбросил бы все до одной. Поэтому по умолчанию
(`pairs_inner=True`) пары сдвигаются на полширины головки внутрь колеи, к
ИЗМЕРЕННОЙ внутренней грани: x_left += w/2, x_right -= w/2, где w —
meta['head_width_mm'] этого кадра. Ширина головки измерена по облаку, не взята
из ГОСТа. `pairs_inner=False` отдаёт оси нитей как есть.

Несовпадение входа
------------------
`build_track` принимает (db_path, frame_index), а прежний контракт — массив
точек. Здесь это снято подменой `track_geometry.BagFrames` на заглушку, которая
отдаёт переданное облако (см. `_frames_from_points`): облако не перечитывается
с диска, а db_path/frame_index остаются ключами ВНУТРЕННИХ кэшей
track_geometry — тёплой сборки (затравка от предыдущего кадра) и связи решений
между кадрами. Поэтому для работы этих механизмов в `find_rail_lines` стоит
передавать `bag_key` и `frame_index`; без них каждый кадр считается холодным.
"""

from __future__ import annotations

import os
import sys
import threading

import numpy as np

_TEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test')
if _TEST_DIR not in sys.path:
    sys.path.insert(0, _TEST_DIR)

import track_geometry as _tg  # noqa: E402

# Шаг сетки Y, на которой узлы двух нитей сводятся в пары. Совпадает с шагом
# узлов полилинии (POLY_BIN_M = 0.5 м): мельче сетку делать нечего, узлов
# между ними нет.
PAIR_STEP_M = _tg.POLY_BIN_M

# Узлы, помеченные как продолжение за пределы данных, в пары НЕ идут: это
# модель, а не измерение, и в карту пути такие узлы класть нельзя — они бы
# выглядели там как наблюдения. Оставляем то, что опирается на точки.
_MEASURED_ORIGINS = frozenset({'measured', 'measured_pair', 'sparse', 'dense'})

_LOCK = threading.Lock()
_FRAME_SEQ = [0]

# Решение последнего кадра: build_track стоит 0.16-0.3 с, а маска, колея и меши
# нужны по одному и тому же кадру сразу после детекции. Держим только последнее
# (в visualize_bag кадры идут строго по одному).
_LAST = {'track': None}

# Контекст текущего кадра для вызовов БЕЗ явных bag_key/frame_index.
# Прежний контракт find_rail_lines их не предусматривает (принимает только
# облако), а тёплой сборке и связи решений нужны «какая запись» и «какой кадр».
# Потребители (visualize_bag и др.) ставят контекст через set_frame_context,
# и подменённая функция берёт ключи отсюда. Без контекста каждый кадр идёт
# холодным проходом — результат тот же, просто вдвое медленнее и без связи.
_CTX = threading.local()


def set_frame_context(bag_key, frame_index):
    """Сказать адаптеру, какой кадр какой записи считается сейчас.

    Нужно для тёплой сборки (0.16 с против 0.3 с) и связи решений соседних
    кадров. Вызывать перед каждой детекцией в покадровом проходе.
    """
    _CTX.bag_key = None if bag_key is None else str(bag_key)
    _CTX.frame_index = None if frame_index is None else int(frame_index)


def clear_frame_context():
    _CTX.bag_key = None
    _CTX.frame_index = None


def _ctx_keys():
    return getattr(_CTX, 'bag_key', None), getattr(_CTX, 'frame_index', None)


class _StubFrames:
    """Заглушка BagFrames: отдаёт заранее переданное облако вместо чтения .db3.

    track_geometry берёт из кадра только xyz (`frames[i][0]`), а intensity и
    ring в геометрии не участвуют — см. комментарий в build_track.
    """

    def __init__(self, xyz):
        self._xyz = xyz

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return (self._xyz, None, None)

    def timestamp(self, index):
        return 0


def _frames_from_points(points):
    """Контекст, в котором track_geometry.BagFrames отдаёт наше облако."""
    xyz = np.asarray(points, dtype=np.float64)[:, :3]

    class _Ctx:
        def __enter__(self):
            _LOCK.acquire()
            self.saved = _tg.BagFrames
            _tg.BagFrames = lambda *a, **kw: _StubFrames(xyz)
            return self

        def __exit__(self, *exc):
            _tg.BagFrames = self.saved
            _LOCK.release()
            return False

    return _Ctx()


def rail_x(line, y):
    """X нити в точке Y. Понимает knots-dict, 2-кортеж (прямая) и 3-й (парабола).

    Текущий `rail_detection.rail_x` knots не понимает (см. докстроку модуля),
    поэтому потребителям линий ОТСЮДА нужен этот вариант либо
    `patch_rail_detection()`.
    """
    if isinstance(line, dict):
        return np.interp(y, line['knots_y'], line['knots_x'])
    if len(line) == 3:
        return line[0] * y * y + line[1] * y + line[2]
    return line[0] * y + line[1]


def patch_rail_detection():
    """Научить `rail_detection.rail_x` понимать knots-линии этого модуля.

    Нужно, когда линии отсюда уходят в код, который зовёт rail_x из
    rail_detection (apply_rail_lines, measure_gauge, _find_own_track). Патч
    аддитивный: 2- и 3-кортежи считаются ровно как раньше.
    """
    import rail_detection as _rd

    if getattr(_rd.rail_x, '_knots_aware', False):
        return _rd.rail_x

    rail_x._knots_aware = True
    _rd.rail_x = rail_x
    # rail_x уже связан по имени внутри своего модуля, поэтому переприсвоения
    # атрибута достаточно: apply_rail_lines/measure_gauge/_find_own_track
    # разрешают имя через глобали rail_detection на каждом вызове.
    return rail_x


def enable(also_visualize=True, full=True):
    """Переключить существующий тракт на детекцию полилинией.

    Подменяет `rail_detection.find_rail_lines` и, при also_visualize, уже
    импортированное имя в `visualize_bag` (модуль тянет функцию к себе через
    `from rail_detection import ...`, поэтому правка только в rail_detection
    до него не дойдёт). Заодно ставит knots-совместимый rail_x.

    full=True (по умолчанию) заменяет ВЕСЬ тракт: маска, колея и меши тоже
    считаются по решению track_geometry, и из rail_detection не вызывается
    ничего. full=False оставляет ему маску/колею/меши, подменяя только поиск
    линий, — режим сравнения: видно, что даёт новая геометрия, а что старая
    обвязка вокруг неё.

    Вызывается из visualize_bag/track_map по флагу --polyline; отдельно нужен,
    только если гоняешь детекцию из своего скрипта.
    """
    import rail_detection as _rd

    patch_rail_detection()
    _rd.find_rail_lines = find_rail_lines
    if full:
        _rd.apply_rail_lines = rail_mask
        _rd.measure_gauge = gauge
        _rd.build_rail_meshes = rail_meshes

    # Имя уже втянуто к себе каждым потребителем через `from rail_detection
    # import find_rail_lines`, поэтому правки в rail_detection мало: надо
    # переставить его и в модулях-потребителях. '__main__' в списке обязателен —
    # запущенный как скрипт track_map.py лежит в sys.modules ИМЕННО под ним, а
    # не под своим именем, и без этого прогон молча шёл бы старой детекцией.
    names = ['__main__', 'track_map', 'rail_mapper']
    if also_visualize:
        names.append('visualize_bag')

    subs = {'find_rail_lines': find_rail_lines}
    if full:
        subs.update({'apply_rail_lines': rail_mask,
                     'measure_gauge': gauge,
                     'build_rail_meshes': rail_meshes})
    for _name in names:
        mod = sys.modules.get(_name)
        if mod is None:
            continue
        for attr, fn in subs.items():
            if hasattr(mod, attr):
                setattr(mod, attr, fn)
    return find_rail_lines


def _nodes_of(rail_meta):
    """Узлы нити как (y, x, z) float-массив + их происхождение."""
    nodes = rail_meta.get('nodes') or []
    if not nodes:
        return np.empty((0, 3)), []
    arr = np.asarray(nodes, dtype=np.float64)
    origin = rail_meta.get('node_origin') or []
    if len(origin) != len(arr):
        # происхождение не размечено — считаем все узлы измеренными
        origin = ['measured'] * len(arr)
    return arr, list(origin)


def _line_from_nodes(ys, xs):
    """Нить как knots-структура, понятная rail_detection.rail_x."""
    order = np.argsort(ys)
    return {'knots_y': ys[order], 'knots_x': xs[order]}


def build_frame(points, bag_key=None, frame_index=None, warm=True, **kwargs):
    """Сырой результат track_geometry.build_track по облаку точек.

    bag_key / frame_index — ключи внутренних кэшей track_geometry (тёплая
    сборка и связь кадров). Передавай их при покадровом проходе по записи,
    иначе каждый кадр считается холодным (0.3 с против 0.16 с) и решения
    соседних кадров не связываются.
    """
    if bag_key is None or frame_index is None:
        # уникальный ключ: кэши не сработают, кадр считается холодным
        with _LOCK:
            _FRAME_SEQ[0] += 1
            bag_key = '<points:%d>' % _FRAME_SEQ[0]
        frame_index = 0
        warm = False

    with _frames_from_points(points):
        return _tg.build_track(str(bag_key), int(frame_index), warm=warm, **kwargs)


def rail_lines_from_track(track, pairs_inner=True):
    """(rail_lines, ground_plane, pairs) из результата build_track.

    pairs_inner — сдвинуть пары с осей нитей на внутренние грани головок
    (см. докстроку модуля). rail_lines остаются ОСЯМИ нитей в любом случае:
    именно по ним строится меш и именно их ждёт `apply_rail_lines`.
    """
    empty_pairs = np.empty((0, 5))

    floor = track.get('floor') or {}
    ground = None
    if all(k in floor for k in 'abc'):
        ground = (float(floor['a']), float(floor['b']), float(floor['c']))

    rails_meta = ((track.get('meta') or {}).get('rail_mesh') or {}).get('rails') or {}

    lines = []
    per_side = {}
    for side in ('left', 'right'):
        arr, origin = _nodes_of(rails_meta.get(side) or {})
        if len(arr) < 2:
            continue
        ys, xs, zs = arr[:, 0], arr[:, 1], arr[:, 2]
        lines.append(_line_from_nodes(ys, xs))
        keep = np.array([o in _MEASURED_ORIGINS for o in origin], dtype=bool)
        per_side[side] = (ys[keep], xs[keep], zs[keep])

    if len(per_side) < 2:
        return lines, ground, empty_pairs

    # Пары (y, x_left, x_right): узлы обеих нитей стоят на одной сетке 0.5 м,
    # поэтому общий диапазон берётся пересечением, а X — интерполяцией по узлам
    # своей нити (между узлами нить и так задана np.interp).
    yl, xl, zl = per_side['left']
    yr, xr, zr = per_side['right']
    if len(yl) < 2 or len(yr) < 2:
        return lines, ground, empty_pairs

    y_lo = max(yl.min(), yr.min())
    y_hi = min(yl.max(), yr.max())
    if not (y_hi > y_lo):
        return lines, ground, empty_pairs

    grid = np.arange(y_lo, y_hi + 0.5 * PAIR_STEP_M, PAIR_STEP_M)
    ol, orr = np.argsort(yl), np.argsort(yr)
    x_left = np.interp(grid, yl[ol], xl[ol])
    x_right = np.interp(grid, yr[orr], xr[orr])
    # Высота верха головки в узле — ИЗМЕРЕНА, а не выведена из плоскости земли.
    # Без неё потребителю остаётся только z = a*x + b*y + c, а плоскость одна на
    # кадр и на дальних дистанциях уходит от настоящей головки (замер на
    # roundT_doubleT f60: +118 мм на 0-10 м против -166 мм на 30-45 м).
    z_left = np.interp(grid, yl[ol], zl[ol])
    z_right = np.interp(grid, yr[orr], zr[orr])

    if pairs_inner:
        # оси нитей -> внутренние грани головок: полширины внутрь с каждой
        # стороны. Ширина ИЗМЕРЕНА по этому кадру, а не взята из ГОСТа.
        half_w = float((track.get('meta') or {}).get('head_width_mm') or 0.0) / 2000.0
        x_left = x_left + half_w
        x_right = x_right - half_w

    pairs = np.column_stack([grid, x_left, x_right, z_left, z_right])
    return lines, ground, pairs


def find_rail_lines(points, intensity=None, return_pairs=False,
                    bag_key=None, frame_index=None, warm=True,
                    max_y_distance=None, pairs_inner=True, **_ignored):
    """Детекция рельсов полилинией в контракте rail_detection.find_rail_lines.

    intensity не используется (геометрия строится по xyz), принимается ради
    совместимости вызовов. Лишние параметры прежней детекции (gauge, gauge_tol,
    own_x_range, curved, fit_range) принимаются и игнорируются: их роль в
    полилинии играют константы POLY_* / GAUGE_* внутри track_geometry.

    max_y_distance — обрезка по дальности ДО детекции, как в прежнем контракте.
    """
    if bag_key is None and frame_index is None:
        bag_key, frame_index = _ctx_keys()

    pts_full = np.asarray(points, dtype=np.float64)
    pts = pts_full
    if max_y_distance is not None and max_y_distance > 0:
        pts = pts_full[np.abs(pts_full[:, 1]) <= max_y_distance]

    empty_pairs = np.empty((0, 5))
    if len(pts) < 100:
        return ([], None, empty_pairs) if return_pairs else ([], None)

    # ПЛОСКОСТЬ ПОЛА СЧИТАЕТСЯ ПО ПОЛНОМУ ОБЛАКУ, а нити — по обрезанному.
    # `_floor_plane` набирает ячейки в полосе y в [-60, -2]; при обрезке до
    # ближней зоны (track_map зовёт с max_y_distance=15) ячеек остаётся вчетверо
    # меньше, и край платформы на |x| = 3..5 м, который на 0.06-0.80 м выше пути,
    # перекашивает МНК: поперечный уклон a уходил до +0.38 против -0.017 по
    # полному облаку (замерено на doubleT_platform f64 и f111; |a| > 0.20 на 9%
    # кадров). Плоскость идёт в track_map как `planes` и дальше в маску рельсов,
    # поэтому завал плоскости смещает и dz, по которому ищется головка.
    floor_ab = None
    if pts is not pts_full and len(pts_full) >= 2000:
        try:
            floor_ab = _tg._floor_plane(pts_full[:, 0], pts_full[:, 1], pts_full[:, 2])
        except (ValueError, IndexError, np.linalg.LinAlgError):
            floor_ab = None

    try:
        track = build_frame(pts, bag_key=bag_key, frame_index=frame_index,
                            warm=warm, floor_ab=floor_ab)
    except (ValueError, IndexError, RuntimeError):
        # кадр без пути (пустой, у платформы, вне тоннеля) — как прежний отказ
        return ([], None, empty_pairs) if return_pairs else ([], None)

    _LAST['track'] = track
    lines, ground, pairs = rail_lines_from_track(track, pairs_inner=pairs_inner)
    return (lines, ground, pairs) if return_pairs else (lines, ground)


# --------------------------------------------------------------------------
# ПОЛНАЯ замена rail_detection: маска, колея и меши тоже из track_geometry.
#
# find_rail_lines выше отдаёт линии в прежнем контракте, и их достаточно, чтобы
# старые apply_rail_lines/measure_gauge/build_rail_meshes работали. Функции ниже
# идут дальше: они берут готовое решение build_track целиком, поэтому маска
# ложится на ИЗМЕРЕННУЮ полилинию, колея приходит из meta (а не переизмеряется
# по маске), а меш — из измеренного сечения с подуклонкой вместо прямоугольника
# 75x150 мм. Ставятся через enable(full=True).
# --------------------------------------------------------------------------

# Полуширина полосы отбора точек вокруг оси нити, м. Головка Р65 шириной 72-75 мм,
# плюс запас на поперечный наклон сечения и на разброс узлов.
MASK_RADIUS_M = 0.075
# Окно по высоте над УЗЛОМ нити (узел стоит на верху головки), м. Вниз — на
# видимую глубину боковой грани (35-40 мм замерено в track_geometry), вверх —
# на шум дальномера.
MASK_DZ = (-0.045, 0.020)


def last_track():
    """Решение build_track по последнему кадру (или None)."""
    return _LAST['track']


def rail_mask(points, rail_lines=None, ground_plane=None, track=None,
              radius=MASK_RADIUS_M, dz=MASK_DZ, **_ignored):
    """Маска точек рельсов по УЗЛАМ полилинии — замена apply_rail_lines.

    Отличие от apply_rail_lines: тот держит ось линией x(y) и подправляет её
    медианой точек в окнах по 2 м, а здесь ось уже полилиния, и у каждого узла
    своя высота верха головки. Поэтому окно по высоте берётся от УЗЛА
    (±MASK_DZ), а не от плоскости земли: на вираже нити разнесены по высоте, и
    общий порог dz над землёй одну из них срезает.

    ground_plane/rail_lines принимаются ради совместимости сигнатуры и не нужны:
    всё берётся из решения кадра.
    """
    pts = np.asarray(points)
    n = len(pts)
    mask = np.zeros(n, dtype=bool)
    if n == 0:
        return mask

    track = track if track is not None else _LAST['track']
    if track is None:
        return mask

    rails_meta = ((track.get('meta') or {}).get('rail_mesh') or {}).get('rails') or {}
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

    for side in ('left', 'right'):
        arr, _ = _nodes_of(rails_meta.get(side) or {})
        if len(arr) < 2:
            continue
        order = np.argsort(arr[:, 0])
        ys, xs, zs = arr[order, 0], arr[order, 1], arr[order, 2]
        inside = (y >= ys[0]) & (y <= ys[-1])
        if not inside.any():
            continue
        x_at = np.interp(y, ys, xs)
        z_at = np.interp(y, ys, zs)
        d = z - z_at
        mask |= inside & (np.abs(x - x_at) < radius) & (d > dz[0]) & (d < dz[1])

    return mask


def gauge(points=None, rail_mask=None, rail_lines=None, ground_plane=None,
          track=None, **_ignored):
    """Колея из meta решения — замена measure_gauge.

    Возвращает тот же dict ('axes'/'inner'/'reason'), что measure_gauge, чтобы
    печать в visualize_bag не менялась. Разница в происхождении чисел: здесь
    колея ИЗМЕРЕНА по головкам при построении геометрии (inner — на 13 мм ниже
    верха, уровень ПТЭ), а не пересчитана по точкам маски срез за срезом.
    Поэтому 'values' — один замер на кадр, а не набор срезов.
    """
    res = {'pair_index': 0, 'axes': None, 'inner': None, 'reason': None}
    track = track if track is not None else _LAST['track']
    if track is None:
        res['reason'] = 'нет решения кадра'
        return res

    meta = track.get('meta') or {}
    for key, name in (('gauge_axis_mm', 'axes'), ('gauge_inner_mm', 'inner')):
        v = meta.get(key)
        if v is None:
            continue
        m = float(v) / 1000.0
        res[name] = {'values': np.array([m]), 'mean': m, 'std': 0.0,
                     'min': m, 'max': m, 'n': 1}
    if res['axes'] is None and res['inner'] is None:
        res['reason'] = 'колея в решении не измерена'
    return res


def rail_meshes(points=None, rail_mask=None, rail_lines=None, ground_plane=None,
                track=None, color=(0.85, 0.15, 0.15), **_ignored):
    """Меши рельсов из ИЗМЕРЕННОГО сечения — замена build_rail_meshes.

    build_rail_meshes ставит прямоугольник 75x150 мм вдоль усреднённой линии.
    Здесь меш приходит готовым из track_geometry: сечение измерено по облаку
    (ширина по перцентилям, профиль из 5 узлов), развёрнуто вдоль полилинии
    перпендикулярно локальной касательной, с подуклонкой 1:20. Шейка и подошва
    не рисуются — лидар их не видит (source: web/foot 'not observed').
    """
    import open3d as o3d

    out = []
    track = track if track is not None else _LAST['track']
    if track is None:
        return out

    for rail in track.get('rails') or []:
        m = rail.get('mesh') or {}
        verts, faces = m.get('vertices'), m.get('faces')
        if not verts or not faces:
            continue
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(np.asarray(verts, dtype=np.float64))
        mesh.triangles = o3d.utility.Vector3iVector(np.asarray(faces, dtype=np.int32))
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color(list(color))
        out.append(mesh)
    return out
