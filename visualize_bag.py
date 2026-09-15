"""
Animated visualization of PointCloud2 from ROS 2 bag (.db3).
No ROS needed — reads SQLite directly, renders with Open3D.

Frames are parsed on demand (vectorized numpy, no per-point Python loops),
so memory stays flat regardless of the bag length.
"""

import argparse
import os
import sqlite3
import struct
import sys
import time
from collections import OrderedDict

import numpy as np
import open3d as o3d

CACHE_FRAMES = 8


def parse_pointcloud2_cdr(data: bytes):
    """Parse CDR-serialized PointCloud2, return (xyz, intensity) as numpy arrays.

    xyz is float32 (N, 3); intensity is float32 (N,) or None when absent.
    """
    bo = '<' if data[1] == 1 else '>'

    def align4(off):
        return off + (-off % 4)

    def read_u32(off):
        return struct.unpack_from(f'{bo}I', data, off)[0], off + 4

    def read_u8(off):
        return data[off], off + 1

    def read_string(off):
        slen, off = read_u32(off)
        s = data[off:off + slen - 1].decode('utf-8')
        return s, off + slen

    offset = 4
    _sec, offset = read_u32(offset)
    _nsec, offset = read_u32(offset)
    _frame_id, offset = read_string(offset)

    offset = align4(offset)
    height, offset = read_u32(offset)
    width, offset = read_u32(offset)

    num_fields, offset = read_u32(offset)
    field_offsets = {}
    for _ in range(num_fields):
        fname, offset = read_string(offset)
        offset = align4(offset)
        f_offset, offset = read_u32(offset)
        _f_datatype, offset = read_u8(offset)
        offset = align4(offset)
        _f_count, offset = read_u32(offset)
        field_offsets[fname] = f_offset

    is_bigendian, offset = read_u8(offset)
    offset = align4(offset)
    point_step, offset = read_u32(offset)
    _row_step, offset = read_u32(offset)
    data_len, offset = read_u32(offset)

    n_declared = height * width
    n_points = min(n_declared, data_len // point_step) if point_step else 0
    cloud = np.frombuffer(data, dtype=np.uint8, offset=offset, count=data_len)
    cloud_bo = '>' if is_bigendian else '<'

    if n_points == 0:
        return np.zeros((0, 3), dtype=np.float32), None

    for axis in ('x', 'y', 'z'):
        if axis not in field_offsets:
            raise ValueError(f'PointCloud2 has no "{axis}" field')

    # Структурный dtype: поля читаются страйдом прямо из буфера, промежуточных копий нет.
    names = [name for name in ('x', 'y', 'z', 'intensity') if name in field_offsets]
    record = np.frombuffer(cloud, dtype=np.dtype({
        'names': names,
        'formats': [f'{cloud_bo}f4'] * len(names),
        'offsets': [field_offsets[name] for name in names],
        'itemsize': point_step,
    }), count=n_points)

    xyz = np.empty((n_points, 3), dtype=np.float32)
    for axis_i, axis in enumerate(('x', 'y', 'z')):
        xyz[:, axis_i] = record[axis]

    intensity = record['intensity'].copy() if 'intensity' in field_offsets else None

    keep = np.isfinite(xyz).all(axis=1) & (xyz != 0).any(axis=1)
    xyz = xyz[keep]
    if intensity is not None:
        intensity = intensity[keep]

    return xyz, intensity


class BagFrames:
    """Random access to parsed PointCloud2 frames of a rosbag2 .db3 file."""

    def __init__(self, db_path, cache_size=CACHE_FRAMES):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        cur = self.conn.cursor()

        try:
            cur.execute("SELECT id FROM topics WHERE type LIKE '%PointCloud2%'")
            topic_ids = [row[0] for row in cur.fetchall()]
        except sqlite3.Error:
            topic_ids = []

        if topic_ids:
            placeholders = ','.join('?' * len(topic_ids))
            cur.execute(
                f"SELECT rowid, timestamp FROM messages WHERE topic_id IN ({placeholders}) "
                "ORDER BY timestamp ASC", topic_ids)
        else:
            cur.execute("SELECT rowid, timestamp FROM messages ORDER BY timestamp ASC")
        rows = cur.fetchall()
        self.rowids = [row[0] for row in rows]
        self.timestamps = np.array([row[1] for row in rows], dtype=np.int64)  # наносекунды
        self.topic_names = self._topic_names(cur)

        self.cache_size = cache_size
        self._cache = OrderedDict()

    def _topic_names(self, cur):
        try:
            cur.execute("SELECT name FROM topics")
            return [row[0] for row in cur.fetchall()]
        except sqlite3.Error:
            return []

    def __len__(self):
        return len(self.rowids)

    def __getitem__(self, index):
        """Возвращает (xyz, intensity, timestamp) — как исходный список кадров в репозитории."""
        if index in self._cache:
            self._cache.move_to_end(index)
            return self._cache[index]

        cur = self.conn.cursor()
        cur.execute("SELECT data, timestamp FROM messages WHERE rowid=?", (self.rowids[index],))
        row = cur.fetchone()
        if row is None:
            raise IndexError(index)

        xyz, intensity = parse_pointcloud2_cdr(row[0])
        frame = (xyz, intensity, row[1])
        self._cache[index] = frame
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return frame


def _estimate_ground_plane(points, cell_size=1.0):
    """
    Робастная оценка плоскости земли через grid-based подход.

    Делит облако на ячейки cell_size × cell_size, берёт 5-й перцентиль Z
    в каждой ячейке (≈ уровень земли), затем RANSAC по сетке.
    """
    from sklearn.linear_model import RANSACRegressor

    z = points[:, 2]
    x_cells = np.arange(points[:, 0].min() - 0.5, points[:, 0].max() + 0.5, cell_size)
    y_cells = np.arange(points[:, 1].min() - 0.5, points[:, 1].max() + 0.5, cell_size)

    gx, gy, gz = [], [], []
    for xi in range(len(x_cells) - 1):
        xmask = (points[:, 0] >= x_cells[xi]) & (points[:, 0] < x_cells[xi + 1])
        for yi in range(len(y_cells) - 1):
            mask = xmask & (points[:, 1] >= y_cells[yi]) & (points[:, 1] < y_cells[yi + 1])
            if mask.sum() > 10:
                gx.append((x_cells[xi] + x_cells[xi + 1]) / 2)
                gy.append((y_cells[yi] + y_cells[yi + 1]) / 2)
                gz.append(np.percentile(z[mask], 5))

    if len(gx) < 10:
        return None

    XY = np.column_stack([gx, gy])
    # random_state фиксирован: без него посев менял и подгонку плоскости земли,
    # и (через неё) состав точек рельсовой маски — 1516…1549 мм на одном кадре.
    ransac = RANSACRegressor(residual_threshold=0.15, max_trials=1000, random_state=0)
    ransac.fit(XY, np.array(gz))
    a, b = ransac.estimator_.coef_
    c = ransac.estimator_.intercept_

    if abs(a) > 0.2 or abs(b) > 0.2:
        return None

    return (a, b, c)


def find_rail_lines(points, intensity, gauge=1.52, gauge_tol=0.15):
    """
    Находит рельсы через поперечные Z-профили + ограничение по колее.

    Алгоритм:
    1. Grid-based RANSAC → плоскость земли
    2. Поперечные срезы по Y → Z-профиль → пики 3-25 см над землёй = головки рельсов
    3. Пары пиков с расстоянием gauge ± gauge_tol → рельсовая колея
    4. Кластеризация пар → отдельные пути
    5. Фит линий x(y) для каждого рельса

    Возвращает:
        rail_lines: list of (slope, intercept) — линии x = slope*y + intercept
        ground_plane: (a, b, c) — коэффициенты плоскости z = a*x + b*y + c
    """
    from sklearn.cluster import DBSCAN

    if len(points) < 100:
        return [], None

    # 1. Robust ground plane via grid percentiles
    ground_plane = _estimate_ground_plane(points)
    if ground_plane is None:
        return [], None

    a, b, c = ground_plane

    z_ground = a * points[:, 0] + b * points[:, 1] + c
    dz = points[:, 2] - z_ground

    # 2. Scan cross-sections, find height peaks, match gauge pairs
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    all_pairs = []  # (y, x_left, x_right)

    for y_center in np.arange(y_min + 0.5, y_max - 0.5, 0.5):
        band = (points[:, 1] > y_center - 0.25) & (points[:, 1] < y_center + 0.25)
        if band.sum() < 30:
            continue

        bp = points[band]
        bdz = dz[band]

        x_bins = np.arange(bp[:, 0].min(), bp[:, 0].max(), 0.05)
        if len(x_bins) < 3:
            continue

        # Z-profile: 80th percentile per X-bin (catches rail head sticking up)
        x_centers = []
        z_profile = []
        for xi in range(len(x_bins) - 1):
            xmask = (bp[:, 0] >= x_bins[xi]) & (bp[:, 0] < x_bins[xi + 1])
            if xmask.sum() > 0:
                x_centers.append((x_bins[xi] + x_bins[xi + 1]) / 2)
                z_profile.append(np.percentile(bdz[xmask], 80))

        if len(x_centers) < 3:
            continue

        x_centers = np.array(x_centers)
        z_profile = np.array(z_profile)

        # Rail peaks: 3-25cm above ground (head height)
        rail_height = (z_profile > 0.03) & (z_profile < 0.25)
        if not rail_height.any():
            continue

        rx = x_centers[rail_height]
        rz = z_profile[rail_height]

        # Group adjacent bins into peaks
        groups = [[0]]
        for j in range(1, len(rx)):
            if rx[j] - rx[j - 1] < 0.15:
                groups[-1].append(j)
            else:
                groups.append([j])

        peaks = [(rx[g].mean(), rz[g].max()) for g in groups]

        # Find pairs at gauge distance
        for ai in range(len(peaks)):
            for bi in range(ai + 1, len(peaks)):
                dist = abs(peaks[bi][0] - peaks[ai][0])
                if abs(dist - gauge) < gauge_tol:
                    all_pairs.append((y_center, peaks[ai][0], peaks[bi][0]))

    if len(all_pairs) < 3:
        return [], ground_plane

    pairs = np.array(all_pairs)
    midpoints = (pairs[:, 1] + pairs[:, 2]) / 2

    # 3. Cluster pairs by track (midpoint X)
    db = DBSCAN(eps=0.5, min_samples=3).fit(midpoints.reshape(-1, 1))
    labels = db.labels_

    # 4. Fit rail lines for each track
    rail_lines = []
    for lbl in sorted(set(labels) - {-1}):
        mask = labels == lbl
        p = pairs[mask]
        y_vals = p[:, 0]

        for rail_idx in [1, 2]:  # left and right rail
            x_vals = p[:, rail_idx]
            if y_vals.max() - y_vals.min() < 1.0:
                rail_lines.append((0.0, x_vals.mean()))
            else:
                coeffs = np.polyfit(y_vals, x_vals, 1)
                rail_lines.append((coeffs[0], coeffs[1]))

    return rail_lines, ground_plane


def _find_own_track(rail_lines):
    """Возвращает индекс пары (ti, ti+1) ближайшей к лидару (X=0)."""
    best_ti = None
    best_dist = float('inf')
    for ti in range(0, len(rail_lines), 2):
        if ti + 1 >= len(rail_lines):
            break
        mid_x = abs((rail_lines[ti][1] + rail_lines[ti + 1][1]) / 2)
        if mid_x < best_dist:
            best_dist = mid_x
            best_ti = ti
    return best_ti


def _robust_median(values, iters=3, mad_k=3.0):
    """Медиана с итеративной отбраковкой выбросов по MAD (по умолчанию 3 итерации)."""
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return None
    center = float(np.median(v))
    for _ in range(iters):
        mad = float(np.median(np.abs(v - center)))
        if mad <= 0:
            break
        keep = np.abs(v - center) <= mad_k * 1.4826 * mad
        if keep.all() or not keep.any():
            break
        v = v[keep]
        center = float(np.median(v))
    return center


def _slice_center(rail_pts, y, slice_half=0.5, center_window=None, center_iters=3,
                  min_points=0):
    """Центр нити в срезе y по оси X.

    center_window=None — прежнее поведение: медиана X внутри среза ±slice_half.
    center_window=W    — устойчивая медиана по Y-окну ±W с отбраковкой по MAD;
                         выбросы в окне (соседние объекты, шум) не тянут центр.
    min_points         — минимум точек в окне (0 = без порога, как в прежнем коде);
                         на редких дальних срезах медиана по 1-2 точкам бессмысленна.
    """
    if center_window is None:
        band = (rail_pts[:, 1] > y - slice_half) & (rail_pts[:, 1] < y + slice_half)
        if not band.any() or band.sum() < min_points:
            return None
        # np.float32 без приведения к float64 — чтобы прежнее число совпало побитово.
        return np.median(rail_pts[band, 0])
    band = np.abs(rail_pts[:, 1] - y) < center_window
    if not band.any() or band.sum() < min_points:
        return None
    return _robust_median(rail_pts[band, 0], iters=center_iters)


def _gauge_stats(values):
    """Сводка по набору срезов (метры → метры, печать в мм делает вызывающий)."""
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return None
    return {'values': v, 'mean': float(v.mean()), 'std': float(v.std()),
            'min': float(v.min()), 'max': float(v.max()), 'n': int(v.size)}


def measure_gauge(points, rail_mask, rail_lines, ground_plane, pair_index=None,
                  slice_step=1.0, slice_half=0.5, center_window=1.5,
                  center_iters=3, center_min_points=0, own_offset=0.1,
                  head_half_width=0.15, head_depth=0.013, head_top_pct=98.0,
                  head_slice_half=1.5, head_min_points=25, head_pct=95.0):
    """Колея своей пары нитей в двух базах отсчёта.

    axes  — расстояние между центрами нитей (прежняя база). Центр среза считается
            через _slice_center: при center_window=W — устойчивая медиана по
            Y-окну ±W, при center_window=None — медиана в ±slice_half (как было).
            «Среднее по срезам» не меняется.
    inner — нормативная база: расстояние между внутренними рабочими гранями
            головок на уровне head_depth ниже поверхности катания (по умолчанию
            13 мм, ПТЭ п.45). Грани берутся как head_pct/100-head_pct перцентили
            футпринта головки (точки в ±head_half_width от линии, полоса
            head_depth под верхом головки). Левая нить пары — с меньшим X, её
            внутренняя грань обращена в +X, правая — в −X. Y-диапазон тот же,
            что у базы по осям (пересечение нитей маски).

    Ничего не меняет в rail_lines/rail_mask: только читает points и маску.
    Возвращает dict: 'pair_index', 'axes', 'inner' (сводки _gauge_stats) и
    'reason' — почему база не посчитана (None, если посчитаны обе).
    """
    result = {'pair_index': pair_index, 'axes': None, 'inner': None, 'reason': None}

    if not rail_lines or ground_plane is None or len(points) == 0:
        result['reason'] = 'нет линий/плоскости/точек'
        return result

    if pair_index is None:
        pair_index = _find_own_track(rail_lines)
    if pair_index is None or pair_index + 1 >= len(rail_lines):
        result['reason'] = 'своя пара нитей не найдена'
        return result
    result['pair_index'] = pair_index

    s1, i1 = rail_lines[pair_index]        # нить с меньшим X
    s2, i2 = rail_lines[pair_index + 1]    # нить с большим X

    rail_pts = points[rail_mask]
    if len(rail_pts) == 0:
        result['reason'] = 'пустая рельсовая маска'
        return result

    r1 = rail_pts[np.abs(rail_pts[:, 0] - (s1 * rail_pts[:, 1] + i1)) < own_offset]
    r2 = rail_pts[np.abs(rail_pts[:, 0] - (s2 * rail_pts[:, 1] + i2)) < own_offset]
    if len(r1) <= 5 or len(r2) <= 5:
        result['reason'] = 'мало точек в нитях своей пары'
        return result

    y_lo = max(r1[:, 1].min(), r2[:, 1].min())
    y_hi = min(r1[:, 1].max(), r2[:, 1].max())

    dists = []
    for y in np.arange(y_lo + slice_half, y_hi - slice_half, slice_step):
        b1 = (r1[:, 1] > y - slice_half) & (r1[:, 1] < y + slice_half)
        b2 = (r2[:, 1] > y - slice_half) & (r2[:, 1] < y + slice_half)
        if b1.sum() > 0 and b2.sum() > 0:
            c1 = _slice_center(r1, y, slice_half, center_window, center_iters,
                               center_min_points)
            c2 = _slice_center(r2, y, slice_half, center_window, center_iters,
                               center_min_points)
            if c1 is not None and c2 is not None:
                dists.append(abs(c2 - c1))
    result['axes'] = _gauge_stats(dists)

    a, b, c = ground_plane
    dz = points[:, 2] - (a * points[:, 0] + b * points[:, 1] + c)

    def footprint(slope, intercept):
        off = points[:, 0] - (slope * points[:, 1] + intercept)
        head = (np.abs(off) < head_half_width) & (dz > 0.03) & (dz < 0.40)
        if head.sum() < 30:
            return None
        top = float(np.percentile(dz[head], head_top_pct))  # поверхность катания
        band = head & (dz >= top - head_depth) & (dz <= top)
        if band.sum() < 30:
            return None
        return off[band], points[band, 1]

    fp1 = footprint(s1, i1)
    fp2 = footprint(s2, i2)
    if fp1 is None or fp2 is None:
        result['reason'] = ('футпринт головки не найден, база по осям тоже не посчитана'
                            if result['axes'] is None else 'футпринт головки не найден')
        return result

    o1, y1 = fp1
    o2, y2 = fp2
    inner = []
    for y in np.arange(y_lo + slice_half, y_hi - slice_half, slice_step):
        b1 = np.abs(y1 - y) < head_slice_half
        b2 = np.abs(y2 - y) < head_slice_half
        if b1.sum() < head_min_points or b2.sum() < head_min_points:
            continue
        left_inner = (s1 * y + i1) + np.percentile(o1[b1], head_pct)
        right_inner = (s2 * y + i2) + np.percentile(o2[b2], 100.0 - head_pct)
        inner.append(right_inner - left_inner)
    result['inner'] = _gauge_stats(inner)

    if result['axes'] is None and result['inner'] is None:
        result['reason'] = 'ни одна база не посчитана'
    return result


def apply_rail_lines(points, rail_lines, ground_plane, rail_radius=0.04,
                     own_track_only=False):
    """
    Быстрое применение найденных рельсовых линий к новому кадру.
    own_track_only=True — красить только путь поезда (ближайший к лидару).
    Возвращает булеву маску (True = рельс).
    """
    n = len(points)
    rail_mask = np.zeros(n, dtype=bool)

    if not rail_lines or ground_plane is None or n == 0:
        return rail_mask

    if own_track_only:
        best_ti = _find_own_track(rail_lines)
        if best_ti is None:
            return rail_mask
        lines_to_use = [rail_lines[best_ti], rail_lines[best_ti + 1]]
    else:
        lines_to_use = rail_lines

    a, b, c = ground_plane
    z_ground = a * points[:, 0] + b * points[:, 1] + c
    dz = points[:, 2] - z_ground
    near_ground = (dz > -0.02) & (dz < 0.12)
    near_ground_idx = np.where(near_ground)[0]

    if len(near_ground_idx) == 0:
        return rail_mask

    ng_y = points[near_ground_idx, 1]
    ng_x = points[near_ground_idx, 0]

    for slope, intercept in lines_to_use:
        x_rail = slope * ng_y + intercept
        close = np.abs(ng_x - x_rail) < rail_radius
        rail_mask[near_ground_idx[close]] = True

    return rail_mask


def detect_rails(points, intensity, rail_lines=None, ground_plane=None,
                 rail_radius=0.04):
    """
    Обёртка: если rail_lines заданы — быстрое применение,
    иначе — полная детекция (find + apply).
    """
    if rail_lines is not None:
        return apply_rail_lines(points, rail_lines, ground_plane, rail_radius)

    lines, gplane = find_rail_lines(points, intensity)
    return apply_rail_lines(points, lines, gplane, rail_radius)


def build_rail_meshes(points, rail_mask, rail_lines, ground_plane,
                      rail_width=0.075, rail_height=0.15, step=0.5):
    """
    Строит 3D-меши рельсов по реальным точкам (повторяет кривые и стрелки).

    Для каждого рельса: берёт точки rail_mask, усредняет X в полосах по Y,
    строит прямоугольный профиль вдоль получившейся кривой.

    Параметры:
        rail_width: ширина профиля (по умолчанию 75 мм — головка Р65)
        rail_height: высота профиля (по умолчанию 150 мм)
        step: шаг по Y (м)
    """
    meshes = []
    if not rail_lines or ground_plane is None:
        return meshes

    a, b, c = ground_plane
    hw = rail_width / 2

    rail_pts = points[rail_mask]
    if len(rail_pts) == 0:
        return meshes

    for slope, intercept in rail_lines:
        # Выбрать точки, принадлежащие этому рельсу (ближайшие к линии)
        x_expected = slope * rail_pts[:, 1] + intercept
        dx = np.abs(rail_pts[:, 0] - x_expected)
        belong = dx < 0.1
        rp = rail_pts[belong]

        if len(rp) < 5:
            continue

        # Усреднить X в полосах по Y → кривая рельса
        y_min, y_max = rp[:, 1].min(), rp[:, 1].max()
        y_vals = np.arange(y_min, y_max, step)
        if len(y_vals) < 2:
            continue

        centerline = []  # (x, y, z_base)
        for y in y_vals:
            band = (rp[:, 1] >= y - step / 2) & (rp[:, 1] < y + step / 2)
            if band.sum() > 0:
                x_center = np.median(rp[band, 0])
            else:
                x_center = slope * y + intercept
            z_base = a * x_center + b * y + c
            centerline.append((x_center, y, z_base))

        # Построить меш вдоль centerline
        verts = []
        tris = []
        for yi, (xc, y, zb) in enumerate(centerline):
            zt = zb + rail_height
            verts.append([xc - hw, y, zb])
            verts.append([xc + hw, y, zb])
            verts.append([xc + hw, y, zt])
            verts.append([xc - hw, y, zt])

            if yi > 0:
                prev = (yi - 1) * 4
                cur = yi * 4
                for face in range(4):
                    nf = (face + 1) % 4
                    tris.append([prev + face, cur + face, cur + nf])
                    tris.append([prev + face, cur + nf, prev + nf])

        if len(centerline) > 1:
            tris.append([0, 1, 2])
            tris.append([0, 2, 3])
            last = (len(centerline) - 1) * 4
            tris.append([last, last + 2, last + 1])
            tris.append([last, last + 3, last + 2])

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(np.array(verts))
        mesh.triangles = o3d.utility.Vector3iVector(np.array(tris))
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([0.85, 0.15, 0.15])
        meshes.append(mesh)

    return meshes


def colorize_rails(points, intensity, rail_mask=None):
    """
    Окраска облака точек:
    - Базовый слой: градиент по высоте (синий → голубой → зелёный → оливковый → фиолетовый)
    - Intensity усиливает яркость
    - Рельсы (из rail_mask) — ярко-красный
    - Конструкции выше земли — бирюзовый
    - Отражатели — жёлтый
    """
    n = len(points)
    colors = np.zeros((n, 3))

    if n == 0:
        return colors

    z = points[:, 2]
    z_min, z_max = np.percentile(z, 1), np.percentile(z, 99)
    z_norm = np.clip((z - z_min) / (z_max - z_min + 1e-8), 0, 1)

    # --- Базовый слой: градиент по высоте (без красного — он для рельсов) ---
    cmap = np.array([
        [0.0,  0.45, 0.35, 0.25],  # тёплый коричневый (земля)
        [0.25, 0.40, 0.50, 0.30],  # оливковый
        [0.50, 0.20, 0.65, 0.45],  # зелёный
        [0.75, 0.60, 0.70, 0.30],  # жёлто-зелёный
        [1.0,  0.55, 0.30, 0.70],  # фиолетовый
    ])
    for ch in range(3):
        colors[:, ch] = np.interp(z_norm, cmap[:, 0], cmap[:, ch + 1])

    # --- Intensity модулирует яркость ---
    if intensity is not None:
        i_norm = np.clip(intensity / (np.percentile(intensity, 98) + 1e-8), 0, 1)
        brightness = 0.35 + 0.65 * i_norm
        colors *= brightness[:, np.newaxis]

        z_med = np.median(z)
        i = intensity
        ground_mask = (z > z_med - 1.0) & (z < z_med + 0.5)

        # --- Конструкции выше земли — бирюзовый ---
        struct_mask = ~ground_mask & (i > 30)
        s_t = np.clip((i[struct_mask] - 30) / 100.0, 0, 1)
        colors[struct_mask, 0] = 0.1 + 0.2 * s_t
        colors[struct_mask, 1] = 0.6 + 0.3 * s_t
        colors[struct_mask, 2] = 0.7 + 0.25 * s_t

        # --- Отражатели/знаки — ярко-жёлтый ---
        hot = i > 150
        colors[hot, 0] = 1.0
        colors[hot, 1] = 0.9
        colors[hot, 2] = 0.2

        ultra = i > 230
        colors[ultra, 0] = 1.0
        colors[ultra, 1] = 1.0
        colors[ultra, 2] = 0.6

    # --- Рельсы — ярко-красный (поверх всего) ---
    if rail_mask is not None and rail_mask.any():
        colors[rail_mask, 0] = 1.0
        colors[rail_mask, 1] = 0.05
        colors[rail_mask, 2] = 0.05

    return colors


def find_db3(bag_dir):
    db3_files = sorted(f for f in os.listdir(bag_dir) if f.endswith('.db3'))
    if not db3_files:
        raise FileNotFoundError(f'No .db3 files in {bag_dir}')
    return os.path.join(bag_dir, db3_files[0])


def main():
    parser = argparse.ArgumentParser(description='Animated LiDAR PointCloud2 viewer for rosbag2 .db3')
    parser.add_argument('bag_dir', nargs='?', default='for_hackathon/doubleT_platform',
                        help='папка с bag-файлом (по умолчанию for_hackathon/doubleT_platform)')
    parser.add_argument('--fps', type=float, default=10.0, help='скорость воспроизведения (по умолчанию 10)')
    parser.add_argument('--start', type=int, default=0, help='с какого кадра начать')
    parser.add_argument('--point-size', type=float, default=3.0,
                        help='размер точки (по умолчанию 3.0 — как в их настройке под белый фон)')
    parser.add_argument('--max-points', type=int, default=0,
                        help='прореживание для отображения (0 = все точки, по умолчанию). '
                             'Скорость теперь не зависит от числа точек, флаг нужен только на слабом GPU')
    parser.add_argument('--real-time', action='store_true',
                        help='темп по таймстемпам bag (записи сняты на ~10 Гц), а не по --fps')
    parser.add_argument('--no-loop', action='store_true', help='остановиться на последнем кадре')
    parser.add_argument('--center-window', type=float, default=1.5,
                        help='полуширина Y-окна устойчивого центра нити, м (по умолчанию 1.5). '
                             '0 = прежняя медиана в ±0.5 м внутри среза')
    parser.add_argument('--center-iters', type=int, default=3,
                        help='итераций отбраковки по MAD в окне центра нити (по умолчанию 3)')
    parser.add_argument('--center-min-points', type=int, default=0,
                        help='минимум точек в Y-окне центра нити (0 = без порога, по умолчанию; '
                             'порог 3+ срезает редкие дальние срезы: на doubleT_platform разброс '
                             'по кадрам падает втрое, но среднее растёт на ~9 мм')
    parser.add_argument('--head-depth', type=float, default=0.013,
                        help='на какой глубине под поверхностью катания мерить внутренние грани '
                             'головок, м (по умолчанию 0.013 = 13 мм, ПТЭ п.45)')
    args = parser.parse_args()
    center_window = args.center_window if args.center_window > 0 else None

    def prepare(pts, intensity, rail_mask=None):
        """Векторы и цвета для Open3D, с учётом --max-points.

        astype(float64) обязателен: Vector3dVector ждёт float64, а на float32 pybind
        конвертирует поэлементно — 112 мс на 185k точек и 208 мс на 346k против 1 мс
        после приведения (замерено). update_geometry при этом меньше 1 мс.
        """
        if 0 < args.max_points < len(pts):
            step = -(-len(pts) // args.max_points)
            pts = pts[::step]
            if intensity is not None:
                intensity = intensity[::step]
            if rail_mask is not None:
                rail_mask = rail_mask[::step]
        return (o3d.utility.Vector3dVector(pts.astype(np.float64)),
                o3d.utility.Vector3dVector(colorize_rails(pts, intensity, rail_mask)))

    db_path = find_db3(args.bag_dir)
    print(f'Reading: {db_path}')

    frames = BagFrames(db_path)
    total = len(frames)
    if total == 0:
        print('No frames found in bag')
        return 1
    if frames.topic_names:
        print(f'Topics: {", ".join(frames.topic_names)}')
    duration = (frames.timestamps[-1] - frames.timestamps[0]) / 1e9
    print(f'Total frames: {total}, длительность {duration:.1f} с, {total / duration:.1f} Гц')

    idx = max(0, min(args.start, total - 1))
    xyz, intensity, _ = frames[idx]
    shown_points = min(len(xyz), args.max_points) if 0 < args.max_points else len(xyz)
    print(f'Frame {idx}: {len(xyz)} валидных точек, в рендер идёт {shown_points}')

    # Диагностика: где облако точек?
    pts0 = xyz
    print(f"=== Статистика стартового кадра ({len(pts0)} точек) ===")
    print(f"  X: min={pts0[:,0].min():.1f}  max={pts0[:,0].max():.1f}  mean={pts0[:,0].mean():.1f}")
    print(f"  Y: min={pts0[:,1].min():.1f}  max={pts0[:,1].max():.1f}  mean={pts0[:,1].mean():.1f}")
    print(f"  Z: min={pts0[:,2].min():.1f}  max={pts0[:,2].max():.1f}  mean={pts0[:,2].mean():.1f}")
    # Где больше всего точек — ближе к 0 или дальше?
    dist = np.sqrt(pts0[:,0]**2 + pts0[:,1]**2 + pts0[:,2]**2)
    print(f"  Расстояние от (0,0,0): min={dist.min():.1f}  max={dist.max():.1f}  median={np.median(dist):.1f}")
    print()

    # Setup Open3D animated viewer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f'LiDAR - {os.path.basename(args.bag_dir)}', width=1280, height=720)

    # Detect rail lines on the starting frame (heavy, ~250ms)
    pts = xyz
    print("Detecting rail lines...")
    rail_lines, ground_plane = find_rail_lines(pts, intensity)
    print(f"Found {len(rail_lines)} rail lines")

    rail_mask = apply_rail_lines(pts, rail_lines, ground_plane, own_track_only=True)
    print(f"Rails in frame 0: {rail_mask.sum()} points")

    # Колея ближайшего пути (по которому едет поезд — mid X ≈ 0):
    # две базы отсчёта — по осям головок (старое число) и по внутренним граням.
    gauge = measure_gauge(pts, rail_mask, rail_lines, ground_plane,
                          center_window=center_window,
                          center_iters=args.center_iters,
                          center_min_points=args.center_min_points,
                          head_depth=args.head_depth)
    if gauge['axes'] is not None:
        d = gauge['axes']['values']
        print(f"\n=== Колея (путь поезда) ===")
        if gauge['inner'] is not None:
            print(f"  Колея (внутренние грани, {args.head_depth * 1000:.0f} мм под верхом головки): "
                  f"{gauge['inner']['mean'] * 1000:.0f} мм")
            print(f"    по срезам: СКО {gauge['inner']['std'] * 1000:.1f}, размах "
                  f"{(gauge['inner']['max'] - gauge['inner']['min']) * 1000:.1f}, "
                  f"срезов {gauge['inner']['n']}")
        else:
            print(f"  Колея (внутренние грани): не измерена — {gauge['reason']}")
        print(f"  Колея (оси головок, старое):                    {d.mean() * 1000:.0f} мм")
        print(f"  Среднее: {d.mean() * 1000:.0f} мм")
        print(f"  Мин:     {d.min() * 1000:.0f} мм")
        print(f"  Макс:    {d.max() * 1000:.0f} мм")
        print(f"  Норма:   1520 мм")

    pcd = o3d.geometry.PointCloud()
    pcd.points, pcd.colors = prepare(pts, intensity, rail_mask)
    vis.add_geometry(pcd)

    opt = vis.get_render_option()
    opt.point_size = args.point_size
    opt.background_color = np.array([1.0, 1.0, 1.0])

    # Камера: ближний план пути — тут видны обе рельсовые нити, по которым меряется колея.
    # front — это НАПРАВЛЕНИЕ ВЗГЛЯДА (проверено: при front=(0,-1,0) зонды вдоль -Y
    # попадают в кадр, то есть камера смотрит вдаль; (0,0.95,0.10) — в сторону сенсора).
    # set_up в Open3D 0.19 на результат не влияет, но пусть остаётся для читаемости.
    ctr = vis.get_view_control()
    ctr.set_lookat([0.0, -7.0, -1.0])   # центр основного облака
    ctr.set_front([0.0, 0.95, 0.10])   # почти горизонтально
    ctr.set_up([0.0, 0.0, 1.0])        # Z вверх
    ctr.set_zoom(0.04)

    print(f'Playing animation at {args.fps:g} Hz ({total} frames)')
    print('  мышь — вращение/зум/сдвиг, Q или закрыть окно — выход')

    frame_period = 1.0 / args.fps if args.fps > 0 else 0.0
    slow_frames = 0
    shown = 0
    t_frame = time.time()
    t_start = t_frame

    while True:
        frame_start = time.time()
        pts, intensity, _ = frames[idx]
        rail_mask = apply_rail_lines(pts, rail_lines, ground_plane, own_track_only=True)

        pcd.points, pcd.colors = prepare(pts, intensity, rail_mask)
        shown += 1

        vis.update_geometry(pcd)
        if not vis.poll_events():
            break
        vis.update_renderer()

        if time.time() - frame_start > 2 * frame_period and frame_period > 0 and slow_frames < 3:
            slow_frames += 1
            print(f'  frame {idx}: {time.time() - frame_start:.3f}s per frame '
                  f'(target {frame_period:.3f}s) — снизьте --fps или --max-points, если playback рвётся')

        period = frame_period
        if idx + 1 < total:
            if args.real_time:
                period = min((frames.timestamps[idx + 1] - frames.timestamps[idx]) / 1e9, 1.0)
            idx += 1
        elif args.no_loop:
            continue
        else:
            idx = 0
            if args.real_time:
                period = 0.0  # зацикливание — без паузы

        if period > 0:
            elapsed = time.time() - t_frame
            if elapsed < period:
                time.sleep(period - elapsed)
                t_frame += period
            else:
                t_frame = time.time()

    vis.destroy_window()
    wall = time.time() - t_start
    print(f'Done: {shown} кадров за {wall:.1f} с ({shown / wall if wall else 0:.1f} кадр/с).')
    return 0


if __name__ == '__main__':
    sys.exit(main())