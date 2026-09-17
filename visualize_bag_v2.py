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
    parser.add_argument('--max-y-distance', type=float, default=20.0,
                        help='макс. дальность по Y от лидара, м (по умолчанию 20). '
                             'Дальше облако разреженное и рельсы уходят косо')
    parser.add_argument('--redetect-every', type=int, default=1,
                        help='пересчитывать рельсы каждые N кадров (по умолчанию 1). '
                             '0 = только на стартовом кадре')
    parser.add_argument('--rail-labels', action='store_true',
                        help='использовать предрассчитанную разметку рельсов из rail_labels/ '
                             '(создаётся через python -m rail_mapping.run)')
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

    # Detect rail lines on the starting frame (heavy, ~250ms)
    pts = xyz
    print("Detecting rail lines...")
    rail_lines, ground_plane = find_rail_lines(pts, intensity,
                                                own_x_range=args.own_x_range,
                                                max_y_distance=args.max_y_distance)
    print(f"Found {len(rail_lines)} rail lines")

    if label_dir:
        label_path = os.path.join(label_dir, f'{idx:03d}.npy')
        if os.path.exists(label_path):
            rail_mask = np.load(label_path)
            print(f"Rails in frame {idx}: {rail_mask.sum()} points (из rail_labels/)")
        else:
            rail_mask = apply_rail_lines(pts, rail_lines, ground_plane, own_track_only=True)
            print(f"Rails in frame {idx}: {rail_mask.sum()} points")
    else:
        rail_mask = apply_rail_lines(pts, rail_lines, ground_plane, own_track_only=True)
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

    redetect_every = args.redetect_every
    last_detect_idx = idx  # стартовый кадр уже посчитан

    while True:
        frame_start = time.time()
        pts, intensity, _ = frames[idx]

        # Пересчёт рельсов каждые N кадров с проверкой консистентности
        if redetect_every > 0 and abs(idx - last_detect_idx) >= redetect_every:
            rail_lines_new, ground_plane_new = find_rail_lines(
                pts, intensity,
                own_x_range=args.own_x_range,
                max_y_distance=args.max_y_distance)
            if rail_lines_new and ground_plane_new is not None:
                # Проверка: новые рельсы не должны сильно отличаться от предыдущих
                from rail_detection import _find_own_track
                old_ti = _find_own_track(rail_lines)
                new_ti = _find_own_track(rail_lines_new)
                accept = True
                if old_ti is not None and new_ti is not None:
                    # Сдвиг осей рельсов (intercept) не больше 0.15м
                    old_mid = (rail_lines[old_ti][1] + rail_lines[old_ti + 1][1]) / 2
                    new_mid = (rail_lines_new[new_ti][1] + rail_lines_new[new_ti + 1][1]) / 2
                    if abs(new_mid - old_mid) > 0.15:
                        accept = False
                    # Колея не должна скакать больше чем на 100мм
                    old_gauge = abs(rail_lines[old_ti + 1][1] - rail_lines[old_ti][1])
                    new_gauge = abs(rail_lines_new[new_ti + 1][1] - rail_lines_new[new_ti][1])
                    if abs(new_gauge - old_gauge) > 0.1:
                        accept = False
                    # Плоскость земли: высота (intercept c) не должна прыгать > 0.1м
                    if abs(ground_plane_new[2] - ground_plane[2]) > 0.1:
                        accept = False
                if accept:
                    rail_lines = rail_lines_new
                    ground_plane = ground_plane_new
                ti = _find_own_track(rail_lines)
                if ti is not None:
                    g = abs(rail_lines[ti+1][1] - rail_lines[ti][1]) * 1000
                    mid = (rail_lines[ti][1] + rail_lines[ti+1][1]) / 2 * 1000
                    print(f'  [{idx:4d}] колея {g:.0f} мм  mid {mid:+.0f} мм  lines {len(rail_lines)}')
                else:
                    print(f'  [{idx:4d}] своя пара не найдена, lines {len(rail_lines)}')
            else:
                print(f'  [{idx:4d}] рельсы не найдены')
            last_detect_idx = idx

        if label_dir:
            label_path = os.path.join(label_dir, f'{idx:03d}.npy')
            if os.path.exists(label_path):
                rail_mask = np.load(label_path)
            else:
                rail_mask = apply_rail_lines(pts, rail_lines, ground_plane, own_track_only=True)
        else:
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