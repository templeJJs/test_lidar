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

from rail_detection import (
    find_rail_lines,
    apply_rail_lines,
    measure_gauge,
    detect_rails,
    build_rail_meshes,
)

CACHE_FRAMES = 8


def _compute_pair_dy(frames, i_from, i_to, step, min_intensity, _cache):
    """Сдвиг вдоль пути между соседними кадрами, м (со знаком). С кешем.

    Основной источник — SE(2)-одометрия: она решает ту же задачу полностью
    (dx, dy, dtheta по Кабшу с отбраковкой выбросов), тогда как оценка по
    медиане dY — её вырожденный частный случай. На повороте разница
    принципиальная: замер на roundT_doubleT, кадры 115-143, дал у одометрии
    ровный -1.70 м/кадр при rms 0.01-0.08, а у оценки по dY — мусор
    (-0.118, +0.116, -0.760, +1.694 на соседних кадрах).

    speed_estimator остаётся фолбэком: там, где отражателей меньше трёх пар,
    Кабш не решается, а консенсуса по dY ещё хватает.

    Знак приводится к соглашению аккумулятора: dy — это сдвиг СЦЕНЫ между
    кадрами, то есть насколько надо подвинуть старый кадр по Y, чтобы он лёг
    в систему следующего. Одометрия отдаёт движение СЕНСОРА, поэтому знак
    инвертируется.
    """
    if i_from in _cache:
        return _cache[i_from]

    total = len(frames)
    pts_a, int_a, ts_a = frames[i_from]
    pts_b, int_b, ts_b = frames[i_to]
    dt = (ts_b - ts_a) / 1e9

    if dt <= 0 or dt > 1.0:
        _cache[i_from] = 0.0
        return 0.0

    dy = float('nan')

    # 1. Одометрия по соседней паре — база, а не дальняя: за 0.3 с сцена
    # успевает повернуться, XZ-подпись отражателя уезжает и матчинг ловит
    # не тот объект.
    from odometry import estimate_motion_from_frames

    m = estimate_motion_from_frames(pts_a, int_a, pts_b, int_b,
                                    min_intensity=min_intensity)
    if m is not None:
        dy = -float(m['dy'])

    # 2. Фолбэк — консенсус по dY. Прежде дальняя база (step) бралась первой
    # и безусловно: она возвращала не None, а уверенный мусор (на кадре 120
    # roundT_doubleT — 27 км/ч с consensus 4/4 против реальных 61), и её
    # speed_ms умножался на dt СОСЕДНЕЙ пары, хотя мерился за dt_far.
    if not np.isfinite(dy):
        from speed_estimator import estimate_speed_from_frames

        result = estimate_speed_from_frames(
            pts_a, int_a, pts_b, int_b, dt,
            min_intensity=min_intensity,
            min_speed_kmh=0.0, min_consensus=2)

        if result is not None:
            dy = result['dy_per_dt'] * dt
        elif step > 1 and i_from + step < total:
            # Дальняя база — последняя надежда, и её результат приводится к
            # шагу одного кадра по СВОЕМУ dt_far, а не по чужому dt.
            pts_far, int_far, ts_far = frames[min(i_from + step, total - 1)]
            dt_far = (ts_far - ts_a) / 1e9
            if dt_far > 0:
                far = estimate_speed_from_frames(
                    pts_a, int_a, pts_far, int_far, dt_far,
                    min_intensity=min_intensity,
                    min_speed_kmh=0.0, min_consensus=2)
                if far is not None:
                    dy = far['dy_per_dt'] * dt

    # nan, а не 0.0: сдвиг 0 означал бы «кадры совпадают», и кадр клался
    # в аккумулят несмещённым — рельсы от этого размазываются сильнее,
    # чем если кадр просто пропустить. Вызывающий отбрасывает nan.
    _cache[i_from] = dy
    return dy


def _pair_dthetas(frames, indices, min_intensity, _cache=None):
    """Поворот (рад) между соседними кадрами списка indices.

    Берётся из SE(2)-одометрии по отражателям — той же, что строит траекторию.
    Если оценки нет, возвращается nan, и вызывающий считает поворот нулевым:
    пропустить поворот хуже, чем ничего, но лучше, чем повернуть наугад.
    """
    from odometry import estimate_motion_from_frames

    if _cache is None:
        _cache = {}

    out = []
    for k in range(len(indices) - 1):
        i_from, i_to = indices[k], indices[k + 1]
        if i_from in _cache:
            out.append(_cache[i_from])
            continue
        pts_a, int_a, _ = frames[i_from]
        pts_b, int_b, _ = frames[i_to]
        m = estimate_motion_from_frames(pts_a, int_a, pts_b, int_b,
                                        min_intensity=min_intensity)
        th = float(m['dtheta']) if m is not None else float('nan')
        _cache[i_from] = th
        out.append(th)
    return np.asarray(out, dtype=float)


def accumulate_with_speed(frames, center_idx, n_accumulate, min_intensity=50, step=3,
                          _dy_cache=None, _dtheta_cache=None):
    """Аккумуляция кадров со сдвигом по Y на основе скорости (отражатели).

    Быстрее ICP, достаточно для прямых тоннелей.
    Возвращает (merged_points, merged_intensity, speed_kmh).

    _dy_cache: optional dict — кеш dy между парами кадров (i_from -> dy).
    """
    total = len(frames)
    start = max(0, center_idx - n_accumulate + 1)
    indices = list(range(start, min(center_idx + 1, total)))

    if len(indices) < 2:
        pts, intensity, _ = frames[center_idx]
        return pts, intensity, 0.0

    if _dy_cache is None:
        _dy_cache = {}

    # Вычисляем dy между соседними парами (с кешем)
    dys = []
    for k in range(len(indices) - 1):
        i_from, i_to = indices[k], indices[k + 1]
        dy = _compute_pair_dy(frames, i_from, i_to, step, min_intensity, _dy_cache)
        dys.append(dy)

    # Отбраковка выбросов: поезд не меняет скорость рывком за 0.1 с, поэтому
    # оценка, выпадающая из ряда соседних, — это развал матчинга отражателей,
    # а не реальное торможение. Такой dy сдвигает кадр не туда и размазывает
    # рельсы: на roundT_pressureGate_roundT кадр 250 давал 27 км/ч вместо 55,
    # и число найденных пар падало с 33 до 4.
    # Порог чисто относительный (0.5*|med|) схлопывался, когда сама медиана
    # приходила битой: на roundT_doubleT окно [-0.118, 1.738, 0.116, -0.760]
    # давало med=-0.001, и под нож шли ВСЕ четыре кадра — аккумуляция
    # обнулялась до одного кадра, отсюда мигание картинки. Абсолютный пол в
    # 0.25 м (примерно 9 км/ч при 10 Гц) не даёт порогу выродиться.
    dys = np.asarray(dys, dtype=float)
    finite = np.isfinite(dys)
    if finite.sum() >= 2:
        med = np.median(dys[finite])
        tol = max(0.5 * abs(med), 0.25)
        dys[finite & (np.abs(dys - med) > tol)] = np.nan

    # Поворот между кадрами. Сдвига по Y недостаточно: на кривой кадр
    # повёрнут относительно центрального, и совмещение только по Y
    # размазывает рельсы тем сильнее, чем дальше от лидара. При 0.2°/кадр
    # (реальный темп на roundT_pressureGate_roundT) за 5 кадров набегает ~1°,
    # что на 30 м даёт полметра — число найденных пар падало с 38 до 6.
    dthetas = _pair_dthetas(frames, indices, min_intensity, _dtheta_cache)

    # Кумулятивное преобразование: каждый кадр приводим к позе center_idx.
    # Кадр с неизвестным сдвигом (nan) и всё, что за ним, использовать нельзя —
    # цепочка через него теряет смысл.
    n = len(indices)
    cumulative_dy = np.zeros(n)
    cumulative_th = np.zeros(n)
    usable = np.ones(n, dtype=bool)
    for k in range(n - 2, -1, -1):
        if not np.isfinite(dys[k]) or not usable[k + 1]:
            usable[k] = False
            continue
        cumulative_dy[k] = cumulative_dy[k + 1] + dys[k]
        th = dthetas[k]
        cumulative_th[k] = cumulative_th[k + 1] + (th if np.isfinite(th) else 0.0)

    # Собираем облака с адаптивной фильтрацией по дальности
    all_pts = []
    all_int = []
    for k, fi in enumerate(indices):
        if not usable[k]:
            continue
        pts, intensity, _ = frames[fi]
        th = cumulative_th[k]
        if abs(cumulative_dy[k]) > 0.001 or abs(th) > 1e-4:
            pts = pts.copy()
            if abs(th) > 1e-4:
                # Поворот вокруг лидара, затем сдвиг вдоль пути — тем же
                # порядком, каким движение накапливалось от кадра к кадру.
                c, s = np.cos(th), np.sin(th)
                x0, y0 = pts[:, 0].copy(), pts[:, 1].copy()
                pts[:, 0] = c * x0 - s * y0
                pts[:, 1] = s * x0 + c * y0
            pts[:, 1] += cumulative_dy[k]
            min_y = abs(cumulative_dy[k]) * 2.0
            far_mask = np.abs(pts[:, 1]) >= min_y
            pts = pts[far_mask]
            if intensity is not None:
                intensity = intensity[far_mask]
        all_pts.append(pts)
        if intensity is not None:
            all_int.append(intensity)

    if not all_pts:
        pts, intensity, _ = frames[center_idx]
        return pts, intensity, 0.0

    merged_pts = np.vstack(all_pts)
    merged_int = np.concatenate(all_int) if all_int else None

    # Средняя скорость — по фактически использованному диапазону, а не по
    # запрошенному: отброшенные кадры иначе занижали бы её в разы.
    first = int(np.argmax(usable))
    total_dy = abs(cumulative_dy[first])
    total_dt = (frames.timestamps[indices[-1]] - frames.timestamps[indices[first]]) / 1e9
    speed_kmh = (total_dy / total_dt * 3.6) if total_dt > 0 else 0.0

    return merged_pts, merged_int, speed_kmh


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


def _rail_map_mask(pts, frame_idx, rail_map_data, radius, ground_plane):
    """Маска рельсовых точек через глобальные кривые.

    Для каждой точки кадра:
    1. global_y = local_y + cumulative_dy[frame_idx]
    2. left_x = interp(global_y, left_knots)
    3. right_x = interp(global_y, right_knots)
    4. Если |point_x - left_x| < radius или |point_x - right_x| < radius → рельс
    5. Плюс фильтр по высоте: только точки 0-40см над землёй
    """
    cum_dy = rail_map_data['cumulative_dy']
    left_ky = rail_map_data['left_knot_ys']
    left_kx = rail_map_data['left_knot_xs']
    right_ky = rail_map_data['right_knot_ys']
    right_kx = rail_map_data['right_knot_xs']
    ground_planes = rail_map_data['ground_planes']

    n = len(pts)
    mask = np.zeros(n, dtype=bool)
    if n == 0:
        return mask

    # Global Y для каждой точки
    dy = cum_dy[frame_idx] if frame_idx < len(cum_dy) else cum_dy[-1]
    global_y = pts[:, 1] + dy

    # Интерполяция: X рельса как функция global_y
    left_x = np.interp(global_y, left_ky, left_kx,
                        left=left_kx[0], right=left_kx[-1])
    right_x = np.interp(global_y, right_ky, right_kx,
                         left=right_kx[0], right=right_kx[-1])

    # Близость к рельсу
    near_left = np.abs(pts[:, 0] - left_x) < radius
    near_right = np.abs(pts[:, 0] - right_x) < radius

    # Фильтр по высоте над землёй (рельсы = 0-40см над землёй)
    if ground_plane is not None:
        a, b, c = ground_plane
        z_ground = a * pts[:, 0] + b * pts[:, 1] + c
        dz = pts[:, 2] - z_ground
        height_ok = (dz > -0.02) & (dz < 0.40)
    elif frame_idx < len(ground_planes):
        gp = ground_planes[frame_idx]
        z_ground = gp[0] * pts[:, 0] + gp[1] * pts[:, 1] + gp[2]
        dz = pts[:, 2] - z_ground
        height_ok = (dz > -0.02) & (dz < 0.40)
    else:
        height_ok = np.ones(n, dtype=bool)

    mask = (near_left | near_right) & height_ok
    return mask


# Высота головки рельса над плоскостью земли. Гистограмма dz в полосе ±8 см
# вокруг нити (5 кадров, 14.8k точек) бимодальна: шпалы/балласт дают широкий
# горб с пиком на 0.06, головка — узкий пик 0.20-0.24 с обрывом на 0.27.
# Провал между модами на 0.16-0.18, туда и ставится нижний порог.
RAIL_HEAD_LO = 0.16
RAIL_HEAD_HI = 0.30


def compute_node_coverage(frames, track, poses, planes, near=20.0,
                          radius=0.12, head_lo=RAIL_HEAD_LO, head_hi=RAIL_HEAD_HI,
                          min_hits=2, verbose=True):
    """Доля кадров, подтвердивших каждый узел карты РЕАЛЬНЫМИ точками головки.

    Зачем: линия карты совпадает с точками головки до 2 см на 0-20 м, но на
    40-60 м расходится на 10 см по медиане и 55 см по p90 — накопленный дрейф
    одометрии разворачивает дальний конец. Поэтому «закрашивать» надо не то,
    что проекция показывает сейчас на 50 м, а то, что было подтверждено, когда
    этот участок проезжали вблизи.

    Узел считается подтверждённым на кадре, если у ожидаемой нити в полосе
    ±0.75 м по дальности нашлось >= min_hits точек на высоте головки.
    Смотрим только ближнюю зону (near), где проекция точна.

    Возвращает (coverage, seen): доля подтверждений и сколько кадров видели узел.
    """
    from track_map import map_to_frame

    s_all = track['s']
    conf = np.zeros(len(s_all))
    seen = np.zeros(len(s_all))
    t0 = time.time()

    for f in range(len(poses)):
        gp = planes[f] if planes is not None else None
        if gp is None or not np.all(np.isfinite(gp)):
            continue
        pts, _, _ = frames[f]
        a, b, c = gp
        dz = pts[:, 2] - (a * pts[:, 0] + b * pts[:, 1] + c)
        head = (dz > head_lo) & (dz < head_hi)
        if head.sum() == 0:
            continue
        hx, hf = pts[head, 0], -pts[head, 1]

        p = map_to_frame(track, poses, f, max_range=near)
        for j, ss in enumerate(p['s']):
            k = int(np.searchsorted(s_all, ss))
            if k >= len(s_all):
                continue
            seen[k] += 1
            band = np.abs(hf - p['fwd'][j]) < 0.75
            if not band.any():
                continue
            near_rail = ((np.abs(hx[band] - p['left_x'][j]) < radius)
                         | (np.abs(hx[band] - p['right_x'][j]) < radius))
            if near_rail.sum() >= min_hits:
                conf[k] += 1

        if verbose and (f + 1) % 50 == 0:
            print(f'  покрытие {f + 1}/{len(poses)}  {time.time() - t0:.1f}s')

    return conf / np.maximum(seen, 1), seen


def _track_map_project(frame_idx, track_map_data, max_range):
    """Проекция карты пути в систему кадра — обёртка над track_map.map_to_frame."""
    from track_map import map_to_frame
    return map_to_frame(track_map_data['track'], track_map_data['poses'],
                        frame_idx, max_range=max_range)


def _track_map_mask(pts, proj, ground_plane, radius=0.12,
                    head_lo=RAIL_HEAD_LO, head_hi=RAIL_HEAD_HI):
    """Маска точек у рельсов по спроецированной карте пути.

    Карта задана как X рельса в зависимости от дальности вперёд, поэтому
    для каждой точки интерполируем ожидаемый X на её дальности и сравниваем.

    Высотное окно — по ГОЛОВКЕ рельса, а не «всё от земли до 40 см». Прежний
    допуск (-0.05..0.40) пропускал шпалы и балласт: головка лежит на dz≈0.22,
    шпала на dz≈0.0, и окно принимало обе. Замер по 20 кадрам
    roundT_pressureGate_roundT: 71% точек маски были шпалами и только 22% —
    головкой. Гистограмма dz в узкой полосе у рельса бимодальна с чистым
    провалом на 0.16-0.18, поэтому порог ставится туда.
    """
    n = len(pts)
    if n == 0 or len(proj['fwd']) < 2:
        return np.zeros(n, dtype=bool)

    order = np.argsort(proj['fwd'])
    fwd_nodes = proj['fwd'][order]
    left_nodes = proj['left_x'][order]
    right_nodes = proj['right_x'][order]

    fwd = -pts[:, 1]          # ось Y лидара смотрит назад
    in_range = (fwd >= fwd_nodes[0]) & (fwd <= fwd_nodes[-1])

    left_x = np.interp(fwd, fwd_nodes, left_nodes)
    right_x = np.interp(fwd, fwd_nodes, right_nodes)
    near = ((np.abs(pts[:, 0] - left_x) < radius) |
            (np.abs(pts[:, 0] - right_x) < radius))

    if ground_plane is not None:
        a, b, c = ground_plane
        dz = pts[:, 2] - (a * pts[:, 0] + b * pts[:, 1] + c)
        height_ok = (dz > head_lo) & (dz < head_hi)
    else:
        # Без плоскости земли отделить головку от шпалы нечем: dz не посчитать.
        # Раньше маска в этом случае бралась целиком — то есть заведомо грязной.
        return np.zeros(n, dtype=bool)

    return in_range & near & height_ok


# Open3D не умеет задавать толщину линии (line_width в рендере игнорируется
# на большинстве бэкендов), поэтому «толстая» нить рисуется пучком смещённых
# копий вокруг осевой.
# Верх головки над плоскостью земли, м. Измерено в test/zones.py
# (RAIL_HEIGHT = 0.16, уровень головки рельса); прежде здесь было 0.15.
RAIL_HEIGHT_OVER_GROUND = 0.16

_THICK_OFFSETS = [(0.0, 0.0), (0.04, 0.0), (-0.04, 0.0),
                  (0.0, 0.04), (0.0, -0.04), (0.03, 0.03), (-0.03, -0.03)]


def _measured_rail_z(y, side):
    """Высота верха головки по УЗЛАМ полилинии этого кадра, или None.

    Узел полилинии стоит на верху головки по факту измерения, и у каждого своя
    высота. Плоскость земли такой высоты не знает: она одна на кадр, поэтому
    z = a*x + b*y + c экстраполируется линейно и на дальних дистанциях уходит
    от настоящей головки тем сильнее, чем круче профиль пути (линия «спадает
    с рельса»). Там, где узлы есть, берём измеренное.

    Возвращает z только В ПРЕДЕЛАХ узлов: за ними np.interp дал бы константу
    крайнего узла, а это та же выдумка, что и плоскость, только хуже.
    """
    rg = sys.modules.get('rail_geometry')
    if rg is None:
        return None
    track = rg.last_track()
    if track is None:
        return None

    rails = ((track.get('meta') or {}).get('rail_mesh') or {}).get('rails') or {}
    arr, _ = rg._nodes_of(rails.get(side) or {})
    if len(arr) < 2:
        return None

    order = np.argsort(arr[:, 0])
    ys, zs = arr[order, 0], arr[order, 2]
    z = np.full(len(y), np.nan)
    inside = (y >= ys[0]) & (y <= ys[-1])
    if not inside.any():
        return None
    z[inside] = np.interp(y[inside], ys, zs)
    return z


def _build_track_lineset(proj, ground_plane, color=(0.9, 0.1, 0.1), thick=True):
    """LineSet двух рельсов по спроецированной карте пути.

    Высота берётся из измеренных узлов полилинии, где они есть, и достраивается
    плоскостью земли только там, где узлов нет (дальше края полилинии).
    """
    if len(proj['fwd']) < 2:
        return None

    order = np.argsort(proj['fwd'])
    fwd = proj['fwd'][order]
    lx, rx = proj['left_x'][order], proj['right_x'][order]
    ly, ry = proj['left_y'][order], proj['right_y'][order]

    a, b, c = ground_plane if ground_plane is not None else (0.0, 0.0, -1.5)

    # Превышение головки над полом: сшитое в карте, если есть, иначе константа.
    # Карта знает его на всю длину, включая дальние узлы, где узлов полилинии
    # текущего кадра уже нет.
    if 'rail_dz' in proj:
        rail_z = np.asarray(proj['rail_dz'])[order]
        rail_z = np.where(np.isnan(rail_z), RAIL_HEIGHT_OVER_GROUND, rail_z)
    else:
        rail_z = RAIL_HEIGHT_OVER_GROUND

    # Плоскость — запасной вариант; измеренные узлы имеют приоритет.
    zl = a * lx + b * ly + c + rail_z
    zr = a * rx + b * ry + c + rail_z
    for z_plane, y_side, side in ((zl, ly, 'left'), (zr, ry, 'right')):
        z_meas = _measured_rail_z(y_side, side)
        if z_meas is not None:
            ok = ~np.isnan(z_meas)
            z_plane[ok] = z_meas[ok]

    n = len(fwd)
    offsets = _THICK_OFFSETS if thick else [(0.0, 0.0)]

    all_pts = []
    lines = []
    base = 0
    for ox, oz in offsets:
        left_pts = np.column_stack([lx + ox, ly, zl + oz])
        right_pts = np.column_stack([rx + ox, ry, zr + oz])
        all_pts.append(left_pts)
        all_pts.append(right_pts)
        lines += [[base + k, base + k + 1] for k in range(n - 1)]
        lines += [[base + n + k, base + n + k + 1] for k in range(n - 1)]
        base += 2 * n

    pts_arr = np.vstack(all_pts)

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts_arr.astype(np.float64))
    ls.lines = o3d.utility.Vector2iVector(np.array(lines))
    ls.colors = o3d.utility.Vector3dVector(
        np.tile(np.asarray(color, dtype=float), (len(lines), 1)))
    return ls


def _build_rail_lineset(frame_idx, rail_map_data, ground_plane, max_y=150.0,
                        step=0.5):
    """Строит Open3D LineSet — две сплошные линии рельсов в СК кадра.

    Генерирует точки каждые step метров по Y кадра, интерполирует X из кривых,
    Z берёт с плоскости земли + 0.15м (верх головки).
    """
    cum_dy = rail_map_data['cumulative_dy']
    left_ky = rail_map_data['left_knot_ys']
    left_kx = rail_map_data['left_knot_xs']
    right_ky = rail_map_data['right_knot_ys']
    right_kx = rail_map_data['right_knot_xs']

    dy = cum_dy[frame_idx] if frame_idx < len(cum_dy) else cum_dy[-1]

    # Локальные Y (лидар смотрит в -Y)
    local_ys = np.arange(-max_y, -0.5, step)
    global_ys = local_ys + dy

    left_xs = np.interp(global_ys, left_ky, left_kx,
                         left=left_kx[0], right=left_kx[-1])
    right_xs = np.interp(global_ys, right_ky, right_kx,
                          left=right_kx[0], right=right_kx[-1])

    # Z: плоскость земли + высота головки
    rail_z_offset = 0.15  # верх головки ≈ 15см над землёй
    if ground_plane is not None:
        a, b, c = ground_plane
    else:
        a, b, c = 0, 0, -1.5

    n = len(local_ys)
    # Левый рельс
    left_pts = np.zeros((n, 3))
    left_pts[:, 0] = left_xs
    left_pts[:, 1] = local_ys
    left_pts[:, 2] = a * left_xs + b * local_ys + c + rail_z_offset

    # Правый рельс
    right_pts = np.zeros((n, 3))
    right_pts[:, 0] = right_xs
    right_pts[:, 1] = local_ys
    right_pts[:, 2] = a * right_xs + b * local_ys + c + rail_z_offset

    # Объединяем: [left_0, left_1, ..., left_n-1, right_0, ..., right_n-1]
    all_pts = np.vstack([left_pts, right_pts])

    # Линии: left_0-left_1, left_1-left_2, ..., right_0-right_1, ...
    lines = []
    for k in range(n - 1):
        lines.append([k, k + 1])           # левый рельс
        lines.append([n + k, n + k + 1])   # правый рельс

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(all_pts.astype(np.float64))
    ls.lines = o3d.utility.Vector2iVector(np.array(lines))

    # Зелёный цвет
    colors = np.tile([0.0, 1.0, 0.0], (len(lines), 1))
    ls.colors = o3d.utility.Vector3dVector(colors)

    return ls


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
    parser.add_argument('--auto-close', action='store_true',
                        help='закрыть окно после последнего кадра (для batch-запуска)')
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
    parser.add_argument('--own-x-range', type=float, default=1.5,
                        help='макс. смещение центра колеи от лидара по X, м (по умолчанию 1.5). '
                             'На прямой midpoint ≈ 0, на повороте смещается, но не дальше этого. '
                             'Пары рельсов с midpoint за пределами диапазона отбрасываются')
    parser.add_argument('--max-y-distance', type=float, default=40.0,
                        help='макс. дальность по Y от лидара, м (по умолчанию 40). '
                             'Дальше облако разреженное и рельсы уходят косо')
    parser.add_argument('--redetect-every', type=int, default=10,
                        help='пересчитывать рельсы каждые N кадров (по умолчанию 10). '
                             '0 = только на стартовом кадре')
    parser.add_argument('--polyline', action='store_true',
                        help='ВЕСЬ тракт рельсов из test/track_geometry.py вместо '
                             'rail_detection.py: поиск нитей, маска точек, колея и меши. '
                             'Нить ведётся полилинией — узлами через 0.5 м, каждый '
                             'следующий ищется в окне от предсказания, поэтому на кривых '
                             'линия не режет дугу хордой и не перескакивает на стену. '
                             'Маска берёт полосу у ИЗМЕРЕННОГО верха головки (у прежней '
                             '43%% точек были шпалы и балласт), колея измерена по головкам '
                             'на уровне ПТЭ, меш — развёртка измеренного сечения с '
                             'подуклонкой 1:20 вместо прямоугольника 75x150 мм. '
                             '--own-x-range не действует: коридор поиска едет за нитью, '
                             'а не стоит прямым в системе лидара')
    parser.add_argument('--accumulate-speed', type=int, default=0,
                        help='накопление N кадров по скорости (отражатели). Быстрее ICP, для прямых тоннелей. '
                             '15-20 безопасно даже на поворотах')
    parser.add_argument('--rail-labels', action='store_true',
                        help='использовать предрассчитанную разметку рельсов из rail_labels/ '
                             '(создаётся через python -m rail_mapping.run)')
    parser.add_argument('--rail-map', action='store_true',
                        help='использовать 3D-карту рельсов из rail_map_3d.npy + cumulative_dy.npy '
                             '(создаётся через python rail_mapper.py <bag_dir>)')
    parser.add_argument('--track-map', action='store_true',
                        help='карта пути из track_map.npz — сшивка ближних наблюдений '
                             'колеи через SE(2)-одометрию (python track_map.py <bag_dir>)')
    parser.add_argument('--no-rail-points', action='store_true',
                        help='не красить точки облака в цвет рельсов — показывать '
                             'только линию пути (чище видно продолжение вдаль)')
    parser.add_argument('--rail-line-color', default='red',
                        choices=['red', 'green'],
                        help='цвет линии пути (по умолчанию red)')
    parser.add_argument('--thin-rail-line', action='store_true',
                        help='тонкая линия пути вместо толстой')
    parser.add_argument('--track-map-radius', type=float, default=0.12,
                        help='полуширина окна вокруг рельса для маски, м (по умолчанию 0.12; '
                             'головка рельса ~75 мм, прежние 0.20 захватывали шпалу рядом)')
    parser.add_argument('--rail-head-lo', type=float, default=RAIL_HEAD_LO,
                        help=f'нижняя граница головки рельса над землёй, м '
                             f'(по умолчанию {RAIL_HEAD_LO}; ниже — шпалы и балласт)')
    parser.add_argument('--rail-head-hi', type=float, default=RAIL_HEAD_HI,
                        help=f'верхняя граница головки рельса над землёй, м '
                             f'(по умолчанию {RAIL_HEAD_HI})')
    parser.add_argument('--track-map-coverage', type=float, default=0.0,
                        metavar='DOLYA',
                        help='рисовать только узлы, подтверждённые реальными точками '
                             'головки не реже DOLYA (напр. 0.5) на кадрах, где узел '
                             'был в ближней зоне. Снимает дрожание дальнего конца: '
                             'линия на 50 м строится по тому, что видели вблизи, '
                             'а не по текущей проекции. Считается один раз при старте')
    parser.add_argument('--track-map-coverage-near', type=float, default=20.0,
                        help='ближняя зона для подсчёта покрытия, м (по умолчанию 20)')
    parser.add_argument('--guard', action='store_true',
                        help='показать габаритный коридор guard/: зелёный — подтверждён '
                             'измерением стен, жёлтый — ведётся предсказанием '
                             '("не наблюдается", не "свободно")')
    parser.add_argument('--guard-no-walls', action='store_true',
                        help='не рисовать серые линии стен, только габарит')
    parser.add_argument('--guard-quiet', action='store_true',
                        help='не печатать строку статуса коридора на каждый кадр')
    parser.add_argument('--guard-accum', action='store_true',
                        help='накапливать коридор по кадрам через одометрию '
                             '(нужен trajectory.npz): поднимает худший кадр')
    parser.add_argument('--rail-map-radius', type=float, default=0.10,
                        help='радиус KDTree-запроса для проекции карты, м (по умолчанию 0.10)')
    args = parser.parse_args()
    center_window = args.center_window if args.center_window > 0 else None
    _line_color = {'red': (0.9, 0.1, 0.1), 'green': (0.0, 0.8, 0.0)}[args.rail_line_color]
    _line_thick = not args.thin_rail_line

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
        if args.no_rail_points:
            rail_mask = None
        return (o3d.utility.Vector3dVector(pts.astype(np.float64)),
                o3d.utility.Vector3dVector(colorize_rails(pts, intensity, rail_mask)))

    db_path = find_db3(args.bag_dir)
    print(f'Reading: {db_path}')

    if args.polyline:
        # Подменяет find_rail_lines здесь и в rail_detection, плюс учит rail_x
        # понимать knots-линии (их отдаёт полилинейная детекция).
        import rail_geometry
        rail_geometry.enable()
        globals()['find_rail_lines'] = rail_geometry.find_rail_lines
        print('Детекция: ПОЛИЛИНИЯ (test/track_geometry.py)')

    frames = BagFrames(db_path)
    total = len(frames)
    if total == 0:
        print('No frames found in bag')
        return 1

    # Карта пути (--track-map)
    track_map_data = None
    if args.track_map:
        tm_path = os.path.join(args.bag_dir, 'track_map.npz')
        traj_path = os.path.join(args.bag_dir, 'trajectory.npz')
        if os.path.exists(tm_path) and os.path.exists(traj_path):
            tm = dict(np.load(tm_path))
            traj = dict(np.load(traj_path))
            track_map_data = {'track': tm, 'poses': traj['poses']}
            print(f'Карта пути: {len(tm["s"])} узлов, '
                  f'{tm["s"][0]:.0f}..{tm["s"][-1]:.0f} м')
            if args.track_map_coverage > 0:
                cov, seen = compute_node_coverage(
                    frames, tm, traj['poses'], tm.get('planes'),
                    near=args.track_map_coverage_near,
                    radius=args.track_map_radius,
                    head_lo=args.rail_head_lo, head_hi=args.rail_head_hi)
                keep = (cov >= args.track_map_coverage) & (seen > 0)
                print(f'  покрытие: med {np.median(cov):.2f}, '
                      f'узлов с покрытием >= {args.track_map_coverage:g}: '
                      f'{keep.sum()} из {len(keep)}')
                if keep.sum() >= 2:
                    tm = {k: (v[keep] if isinstance(v, np.ndarray)
                              and v.shape[:1] == keep.shape else v)
                          for k, v in tm.items()}
                    track_map_data['track'] = tm
                else:
                    print('  слишком мало подтверждённых узлов — фильтр не применён')
        else:
            print(f'track_map.npz не найден в {args.bag_dir}, '
                  f'запустите: python track_map.py {args.bag_dir}')
            args.track_map = False

    # Глобальные кривые рельсов (--rail-map)
    rail_map_data = None
    if args.rail_map:
        polyfit_path = os.path.join(args.bag_dir, 'rail_polyfit.npz')
        if os.path.exists(polyfit_path):
            rail_map_data = dict(np.load(polyfit_path))
            n_left = len(rail_map_data['left_knot_ys'])
            n_right = len(rail_map_data['right_knot_ys'])
            gy_range = (rail_map_data['left_knot_ys'].min(),
                        rail_map_data['left_knot_ys'].max())
            print(f'Глобальные кривые рельсов: {n_left}+{n_right} узлов, '
                  f'Y={gy_range[0]:.0f}..{gy_range[1]:.0f}м')
        else:
            print(f'rail_polyfit.npz не найден в {args.bag_dir}, '
                  f'запустите: python rail_mapper.py {args.bag_dir}')
            args.rail_map = False

    # Предрассчитанные маски рельсов из rail_mapping
    label_dir = os.path.join(args.bag_dir, 'rail_labels') if args.rail_labels else None
    if label_dir and not os.path.isdir(label_dir):
        print(f'rail_labels/ не найден в {args.bag_dir}, запустите: python -m rail_mapping.run {args.bag_dir}')
        label_dir = None
    elif label_dir:
        print(f'Используются предрассчитанные маски из {label_dir}/')
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

    # Detect rail lines on the starting frame
    dy_cache = {}  # shared cache for pairwise dy
    pts = xyz
    rail_lines = []
    ground_plane = None

    if args.track_map and track_map_data is not None:
        gp_arr = track_map_data['track'].get('planes')
        if gp_arr is not None and idx < len(gp_arr) and not np.isnan(gp_arr[idx]).any():
            ground_plane = tuple(gp_arr[idx])
        proj = _track_map_project(idx, track_map_data, args.max_y_distance)
        rail_mask = _track_map_mask(pts, proj, ground_plane, args.track_map_radius,
                                    args.rail_head_lo, args.rail_head_hi)
        print(f"Rails in frame {idx}: {rail_mask.sum()} points "
              f"(карта пути, до {proj['fwd'].max() if len(proj['fwd']) else 0:.0f} м)")
    elif args.rail_map and rail_map_data is not None:
        # Берём ground_plane из карты, детекцию пропускаем
        gp_arr = rail_map_data['ground_planes']
        if idx < len(gp_arr):
            ground_plane = tuple(gp_arr[idx])
        rail_mask = _rail_map_mask(pts, idx, rail_map_data, args.rail_map_radius,
                                    ground_plane)
        print(f"Rails in frame {idx}: {rail_mask.sum()} points (из rail_polyfit)")
    else:
        # Детекция рельсов (только если не --rail-map)
        if args.accumulate_speed > 1:
            detect_pts, detect_int, init_speed = accumulate_with_speed(
                frames, idx, args.accumulate_speed, _dy_cache=dy_cache)
            print(f'Накопление (скорость): {len(pts)} -> {len(detect_pts)} точек '
                  f'({args.accumulate_speed} кадров, {init_speed:.1f} km/h)')
        else:
            detect_pts = pts
            detect_int = intensity
        print("Detecting rail lines...")
        if args.polyline:
            # какой кадр какой записи — для тёплой сборки и связи решений
            rail_geometry.set_frame_context(db_path, idx)
        rail_lines, ground_plane = find_rail_lines(detect_pts, detect_int if args.accumulate_speed <= 1 else None,
                                                    own_x_range=args.own_x_range,
                                                    max_y_distance=args.max_y_distance)
        print(f"Found {len(rail_lines)} rail lines")

        if label_dir:
            label_path = os.path.join(label_dir, f'{idx:04d}.npy')
            if os.path.exists(label_path):
                rail_mask = np.load(label_path)
                print(f"Rails in frame {idx}: {rail_mask.sum()} points (из rail_labels/)")
            else:
                rail_mask = apply_rail_lines(pts, rail_lines, ground_plane, own_track_only=True,
                                             max_y_distance=args.max_y_distance)
                print(f"Rails in frame {idx}: {rail_mask.sum()} points")
        else:
            rail_mask = apply_rail_lines(pts, rail_lines, ground_plane, own_track_only=True,
                                             max_y_distance=args.max_y_distance)
            print(f"Rails in frame {idx}: {rail_mask.sum()} points")

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

    # LineSet габаритного коридора (--guard)
    guard_lineset = None
    _guard_acc = None
    if args.guard:
        from guard.run import analyze as _guard_analyze
        from guard.draw import corridor_lineset, corridor_bar, corridor_line
        if args.guard_accum:
            from guard.accum import CorridorAccumulator, load_poses
            _poses = load_poses(args.bag_dir)
            if _poses is None:
                print(f'  --guard-accum: нет trajectory.npz в {args.bag_dir}, '
                      f'накопление выключено (python odometry.py {args.bag_dir} --save)')
            else:
                _guard_acc = CorridorAccumulator(_poses)
        # Габарит считается ВСЕГДА по чистому кадру, даже при
        # --accumulate-speed: склейка по скорости размазывает головки рельсов
        # на повороте (замер roundT_pressureGate_roundT f140: ось скачет
        # -0.48 -> +0.12 м, колея 1398 -> 1322 мм), а движущийся объект
        # растягивает в полосу вдоль пути — то есть ровно то, что детектор
        # препятствий должен видеть как объект. Накопление габарита делает
        # --guard-accum, и оно копит результаты, а не точки.
        _g_axis, _g_wall = _guard_analyze(frames[idx][0])
        if _guard_acc is not None:
            _g_wall = _guard_acc.update(idx, _g_wall)
        guard_lineset = corridor_lineset(_g_wall, _g_axis,
                                         show_walls=not args.guard_no_walls)
        if guard_lineset is not None:
            vis.add_geometry(guard_lineset)
        if not args.guard_quiet:
            print(f'  [{idx:4d}] {corridor_bar(_g_wall)} '
                  f'{corridor_line(_g_wall, _g_axis)}')

    # LineSet для рельсов (--rail-map)
    rail_lineset = None
    if args.track_map and track_map_data is not None:
        rail_lineset = _build_track_lineset(
            _track_map_project(idx, track_map_data, args.max_y_distance), ground_plane,
            color=_line_color, thick=_line_thick)
        if rail_lineset is not None:
            vis.add_geometry(rail_lineset)
    elif args.rail_map and rail_map_data is not None:
        rail_lineset = _build_rail_lineset(idx, rail_map_data, ground_plane,
                                            max_y=args.max_y_distance)
        vis.add_geometry(rail_lineset)

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

    redetect_every = 0 if (label_dir or args.rail_map or args.track_map) else args.redetect_every
    last_detect_idx = idx  # стартовый кадр уже посчитан

    while True:
        frame_start = time.time()
        _cur_speed_kmh = 0.0
        if args.accumulate_speed > 1:
            pts, intensity, _cur_speed_kmh = accumulate_with_speed(
                frames, idx, args.accumulate_speed, _dy_cache=dy_cache)
        else:
            pts, intensity, _ = frames[idx]

        # Пересчёт рельсов каждые N кадров с проверкой консистентности
        if redetect_every > 0 and abs(idx - last_detect_idx) >= redetect_every:
            if args.accumulate_speed > 1:
                detect_pts = pts  # уже аккумулированные
                detect_int = None
            else:
                detect_pts = pts
                detect_int = intensity
            if args.polyline:
                rail_geometry.set_frame_context(db_path, idx)
            rail_lines_new, ground_plane_new = find_rail_lines(
                detect_pts, detect_int,
                own_x_range=args.own_x_range,
                max_y_distance=args.max_y_distance)
            if rail_lines_new and ground_plane_new is not None:
                # Проверка: новые рельсы не должны сильно отличаться от предыдущих
                from rail_detection import _find_own_track, rail_x
                old_ti = _find_own_track(rail_lines)
                new_ti = _find_own_track(rail_lines_new)
                accept = True
                if old_ti is not None and new_ti is not None:
                    # Сдвиг осей рельсов не больше 0.15м. Берём rail_x(·, 0),
                    # а не line[1]: у параболы второй элемент — наклон.
                    old_mid = (rail_x(rail_lines[old_ti], 0.0)
                               + rail_x(rail_lines[old_ti + 1], 0.0)) / 2
                    new_mid = (rail_x(rail_lines_new[new_ti], 0.0)
                               + rail_x(rail_lines_new[new_ti + 1], 0.0)) / 2
                    if abs(new_mid - old_mid) > 0.15:
                        accept = False
                    # Колея не должна скакать больше чем на 100мм
                    old_gauge = abs(rail_x(rail_lines[old_ti + 1], 0.0)
                                    - rail_x(rail_lines[old_ti], 0.0))
                    new_gauge = abs(rail_x(rail_lines_new[new_ti + 1], 0.0)
                                    - rail_x(rail_lines_new[new_ti], 0.0))
                    if abs(new_gauge - old_gauge) > 0.1:
                        accept = False
                    # Плоскость земли: высота (intercept c) не должна прыгать > 0.1м
                    if abs(ground_plane_new[2] - ground_plane[2]) > 0.1:
                        accept = False
                if accept:
                    rail_lines = rail_lines_new
                    ground_plane = ground_plane_new
            last_detect_idx = idx

        if args.track_map and track_map_data is not None:
            gp_arr = track_map_data['track'].get('planes')
            if gp_arr is not None and idx < len(gp_arr) and not np.isnan(gp_arr[idx]).any():
                ground_plane = tuple(gp_arr[idx])
            _proj = _track_map_project(idx, track_map_data, args.max_y_distance)
            rail_mask = _track_map_mask(pts, _proj, ground_plane, args.track_map_radius,
                                        args.rail_head_lo, args.rail_head_hi)
        elif args.rail_map and rail_map_data is not None:
            rail_mask = _rail_map_mask(pts, idx, rail_map_data, args.rail_map_radius,
                                        ground_plane)
        elif label_dir:
            label_path = os.path.join(label_dir, f'{idx:04d}.npy')
            if os.path.exists(label_path):
                rail_mask = np.load(label_path)
            else:
                rail_mask = apply_rail_lines(pts, rail_lines, ground_plane, own_track_only=True,
                                             max_y_distance=args.max_y_distance)
        else:
            rail_mask = apply_rail_lines(pts, rail_lines, ground_plane, own_track_only=True,
                                             max_y_distance=args.max_y_distance)

        # Вывод: скорость + длина рельсов
        rail_y = np.abs(pts[rail_mask, 1]) if rail_mask.any() else np.array([0.0])
        rail_len = float(rail_y.max())
        _n_rail = int(rail_mask.sum())
        print(f'\r  [{idx:4d}] {_cur_speed_kmh:5.1f} km/h  рельсы: {rail_len:5.1f}м  ({_n_rail} точек)', end='', flush=True)

        pcd.points, pcd.colors = prepare(pts, intensity, rail_mask)
        shown += 1

        # Обновляем линии рельсов
        if rail_lineset is not None and args.track_map:
            new_ls = _build_track_lineset(_proj, ground_plane,
                                          color=_line_color, thick=_line_thick)
            if new_ls is not None:
                rail_lineset.points = new_ls.points
                rail_lineset.lines = new_ls.lines
                rail_lineset.colors = new_ls.colors
                vis.update_geometry(rail_lineset)
        elif rail_lineset is not None:
            new_ls = _build_rail_lineset(idx, rail_map_data, ground_plane,
                                          max_y=args.max_y_distance)
            rail_lineset.points = new_ls.points
            rail_lineset.lines = new_ls.lines
            rail_lineset.colors = new_ls.colors
            vis.update_geometry(rail_lineset)

        if args.guard:
            _g_axis, _g_wall = _guard_analyze(frames[idx][0])
            if _guard_acc is not None:
                _g_wall = _guard_acc.update(idx, _g_wall)
            new_g = corridor_lineset(_g_wall, _g_axis,
                                     show_walls=not args.guard_no_walls)
            if new_g is not None:
                if guard_lineset is None:
                    guard_lineset = new_g
                    vis.add_geometry(guard_lineset)
                else:
                    guard_lineset.points = new_g.points
                    guard_lineset.lines = new_g.lines
                    guard_lineset.colors = new_g.colors
                    vis.update_geometry(guard_lineset)
            if not args.guard_quiet:
                print(f'  [{idx:4d}] {corridor_bar(_g_wall)} '
                      f'{corridor_line(_g_wall, _g_axis)}')

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
        elif args.auto_close:
            break
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