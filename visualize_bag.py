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
        if index in self._cache:
            self._cache.move_to_end(index)
            return self._cache[index]

        cur = self.conn.cursor()
        cur.execute("SELECT data FROM messages WHERE rowid=?", (self.rowids[index],))
        row = cur.fetchone()
        if row is None:
            raise IndexError(index)

        frame = parse_pointcloud2_cdr(row[0])
        self._cache[index] = frame
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return frame


def colorize_rails(points, intensity):
    """
    Окраска с ярким выделением рельсов (оранжевый/жёлтый).
    Остальное — приглушённые холодные тона.
    """
    n = len(points)
    colors = np.zeros((n, 3))

    if intensity is None or n == 0:
        return colors + 0.3

    z = points[:, 2]
    z_med = np.median(z)
    i = intensity

    # --- Базовый слой: тёмные холодные тона по высоте ---
    z_norm = np.clip((z - z.min()) / (z.max() - z.min() + 1e-8), 0, 1)
    colors[:, 0] = 0.06 + 0.08 * z_norm
    colors[:, 1] = 0.06 + 0.10 * z_norm
    colors[:, 2] = 0.12 + 0.15 * z_norm

    # --- Intensity: слабые отражения — голубоватый ---
    i_log = np.log1p(i) / np.log1p(255)

    weak = i_log < 0.3
    colors[weak, 0] = 0.05 + 0.08 * i_log[weak]
    colors[weak, 1] = 0.07 + 0.10 * i_log[weak]
    colors[weak, 2] = 0.15 + 0.12 * i_log[weak]

    # Средние — серо-голубой
    mid = (i_log >= 0.3) & (i_log < 0.5)
    t = (i_log[mid] - 0.3) / 0.2
    colors[mid, 0] = 0.10 + 0.10 * t
    colors[mid, 1] = 0.15 + 0.15 * t
    colors[mid, 2] = 0.20 + 0.10 * t

    # --- Рельсы: на уровне земли, intensity > 20 → ЯРКО-ОРАНЖЕВЫЙ ---
    ground_mask = (z > z_med - 1.0) & (z < z_med + 0.5)

    rail = ground_mask & (i > 20)
    rail_t = np.clip((i[rail] - 20) / 80.0, 0, 1)
    colors[rail, 0] = 1.0
    colors[rail, 1] = 0.5 + 0.3 * rail_t
    colors[rail, 2] = 0.0

    # Очень яркие рельсы — ярко-жёлтый
    hot_rail = ground_mask & (i > 80)
    colors[hot_rail, 0] = 1.0
    colors[hot_rail, 1] = 0.9
    colors[hot_rail, 2] = 0.1

    # --- Конструкции выше земли — бирюзовый ---
    struct_mask = ~ground_mask & (i > 30)
    s_t = np.clip((i[struct_mask] - 30) / 100.0, 0, 1)
    colors[struct_mask, 0] = 0.0 + 0.1 * s_t
    colors[struct_mask, 1] = 0.5 + 0.3 * s_t
    colors[struct_mask, 2] = 0.6 + 0.2 * s_t

    # --- Отражатели/знаки (intensity > 150) — ярко-красный ---
    hot = i > 150
    colors[hot, 0] = 1.0
    colors[hot, 1] = 0.1
    colors[hot, 2] = 0.1

    # Сверх-яркие — белый
    ultra = i > 230
    colors[ultra, 0] = 1.0
    colors[ultra, 1] = 1.0
    colors[ultra, 2] = 1.0

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
    parser.add_argument('--point-size', type=float, default=2.0, help='размер точки (по умолчанию 2.0)')
    parser.add_argument('--max-points', type=int, default=0,
                        help='прореживание для отображения (0 = все точки, по умолчанию). '
                             'Скорость теперь не зависит от числа точек, флаг нужен только на слабом GPU')
    parser.add_argument('--real-time', action='store_true',
                        help='темп по таймстемпам bag (записи сняты на ~10 Гц), а не по --fps')
    parser.add_argument('--no-loop', action='store_true', help='остановиться на последнем кадре')
    args = parser.parse_args()

    def prepare(xyz, intensity):
        """Цвета + векторы для Open3D.

        astype(float64) здесь обязателен: Vector3dVector ждёт float64, и на float32
        pybind конвертирует поэлементно — 112 мс на 185k точек и 208 мс на 346k
        против 1 мс после приведения (замерено). update_geometry при этом <1 мс.
        """
        if 0 < args.max_points < len(xyz):
            step = -(-len(xyz) // args.max_points)
            xyz = xyz[::step]
            if intensity is not None:
                intensity = intensity[::step]
        return (o3d.utility.Vector3dVector(xyz.astype(np.float64)),
                o3d.utility.Vector3dVector(colorize_rails(xyz, intensity)))

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
    xyz, intensity = frames[idx]
    shown_points = min(len(xyz), args.max_points) if 0 < args.max_points else len(xyz)
    print(f'Frame 0: {len(xyz)} валидных точек, в рендер идёт {shown_points}')

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f'LiDAR - {os.path.basename(args.bag_dir)}', width=1280, height=720)

    pcd = o3d.geometry.PointCloud()
    pcd.points, pcd.colors = prepare(xyz, intensity)
    vis.add_geometry(pcd)

    opt = vis.get_render_option()
    opt.point_size = args.point_size
    opt.background_color = np.array([0.01, 0.01, 0.03])

    # Камера в коридоре пути, взгляд вперёд по пути (-Y), Z вверх.
    # front — это НАПРАВЛЕНИЕ ВЗГЛЯДА: (0,1,0) уводит камеру назад, в сторону от пути.
    # set_up в Open3D 0.19 на результат не влияет (проверено зондами на экране) — при
    # front=(0,-1,0) вверх и так получается +Z.
    ctr = vis.get_view_control()
    ctr.set_lookat([0.0, -30.0, 0.0])
    ctr.set_front([0.0, -1.0, 0.0])
    ctr.set_zoom(0.015)

    print(f'Playing animation at {args.fps:g} Hz ({total} frames)')
    print('  мышь — вращение/зум/сдвиг, Q или закрыть окно — выход')

    frame_period = 1.0 / args.fps if args.fps > 0 else 0.0
    slow_frames = 0
    shown = 0
    t_frame = time.time()
    t_start = t_frame

    while True:
        frame_start = time.time()
        xyz, intensity = frames[idx]
        shown += 1

        pcd.points, pcd.colors = prepare(xyz, intensity)

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