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

try:
    import open3d as o3d
except ImportError:
    o3d = None

from rail_detection import (
    find_rail_lines,
    apply_rail_lines,
    measure_gauge,
    detect_rails,
    build_rail_meshes,
    rail_x,
)

CACHE_FRAMES = 8


def _compute_pair_dy(frames, i_from, i_to, step, min_intensity, _cache):
    """Compute dy between two adjacent frames, with cache lookup.

    Returns dy in meters (signed). Uses cache[i_from] if available.
    Logic is identical to the original per-pair code in accumulate_with_speed.
    """
    if i_from in _cache:
        return _cache[i_from]

    from speed_estimator import estimate_speed_from_frames

    total = len(frames)
    pts_a, int_a, ts_a = frames[i_from]
    pts_b, int_b, ts_b = frames[i_to]
    dt = (ts_b - ts_a) / 1e9

    if dt <= 0 or dt > 1.0:
        _cache[i_from] = 0.0
        return 0.0

    # Пробуем с текущим step, если не получается — step=1
    result = None
    if step > 1 and i_from + step < total:
        pts_far, int_far, ts_far = frames[min(i_from + step, total - 1)]
        dt_far = (ts_far - ts_a) / 1e9
        if dt_far > 0:
            result = estimate_speed_from_frames(
                pts_a, int_a, pts_far, int_far, dt_far,
                min_intensity=min_intensity,
                min_speed_kmh=0.0, min_consensus=2)

    if result is None:
        result = estimate_speed_from_frames(
            pts_a, int_a, pts_b, int_b, dt,
            min_intensity=min_intensity,
            min_speed_kmh=0.0, min_consensus=2)

    if result is not None:
        dy = result['speed_ms'] * result['direction'] * dt
    else:
        dy = 0.0

    _cache[i_from] = dy
    return dy


def accumulate_with_speed(frames, center_idx, n_accumulate, min_intensity=50, step=3,
                          _dy_cache=None):
    """Аккумуляция кадров со сдвигом по Y на основе скорости (отражатели).

    Быстрее ICP, достаточно для прямых тоннелей.
    Возвращает (merged_points, merged_intensity, speed_kmh).

    _dy_cache: optional dict — кеш dy между парами кадров (i_from -> dy).
               При последовательных вызовах с скользящим окном избегает
               повторного вычисления скорости для уже посчитанных пар.
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

    # Кумулятивный сдвиг: каждый кадр сдвигаем к позиции center_idx
    # cumulative_dy[k] = сумма dy от кадра k до center_idx
    n = len(indices)
    cumulative_dy = np.zeros(n)
    for k in range(n - 2, -1, -1):
        cumulative_dy[k] = cumulative_dy[k + 1] + dys[k]

    # Собираем облака с адаптивной фильтрацией по дальности:
    # из старых кадров берём только дальние точки (ближние и так плотные,
    # а на повороте ближние "уплывают" по X сильнее дальних).
    # min_y_for_old = |cumulative_dy| * 2 — чем старше кадр, тем дальше порог.
    all_pts = []
    all_int = []
    for k, fi in enumerate(indices):
        pts, intensity, _ = frames[fi]
        if abs(cumulative_dy[k]) > 0.001:
            pts = pts.copy()
            pts[:, 1] += cumulative_dy[k]
            # Из старого кадра берём только точки дальше порога
            min_y = abs(cumulative_dy[k]) * 2.0
            far_mask = np.abs(pts[:, 1]) >= min_y
            pts = pts[far_mask]
            if intensity is not None:
                intensity = intensity[far_mask]
        all_pts.append(pts)
        if intensity is not None:
            all_int.append(intensity)

    merged_pts = np.vstack(all_pts)
    merged_int = np.concatenate(all_int) if all_int else None

    # Средняя скорость
    total_dy = abs(cumulative_dy[0])
    total_dt = (frames.timestamps[indices[-1]] - frames.timestamps[indices[0]]) / 1e9
    speed_kmh = (total_dy / total_dt * 3.6) if total_dt > 0 else 0.0

    return merged_pts, merged_int, speed_kmh


def _accumulate_icp(frames, center_idx, n_accumulate, voxel=0.08, threshold=0.5):
    """Совмещает n_accumulate кадров вокруг center_idx через ICP.

    Возвращает объединённое облако точек в СК центрального кадра.
    """
    total = len(frames)
    # Берём кадры ДО текущего (уже проехали — они стабильнее)
    indices = list(range(max(0, center_idx - n_accumulate + 1), center_idx + 1))
    if len(indices) < 2:
        pts0, _, _ = frames[center_idx]
        return pts0

    base_idx = indices[-1]  # текущий кадр — базовый
    pts_base, _, _ = frames[base_idx]
    accumulated = [pts_base]

    prev_pts = pts_base
    prev_step_T = np.eye(4)

    # Идём от текущего назад
    for i in range(len(indices) - 2, -1, -1):
        fi = indices[i]
        pts_i, _, _ = frames[fi]

        src_pcd = o3d.geometry.PointCloud()
        src_pcd.points = o3d.utility.Vector3dVector(pts_i)
        src_pcd = src_pcd.voxel_down_sample(voxel)
        src_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=30))

        tgt_pcd = o3d.geometry.PointCloud()
        tgt_pcd.points = o3d.utility.Vector3dVector(prev_pts)
        tgt_pcd = tgt_pcd.voxel_down_sample(voxel)
        tgt_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=30))

        result = o3d.pipelines.registration.registration_icp(
            src_pcd, tgt_pcd,
            max_correspondence_distance=threshold,
            init=prev_step_T,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
        )

        if result.fitness < 0.3:
            prev_pts = pts_i
            continue

        T = result.transformation
        prev_step_T = T.copy()

        pts_hom = np.hstack([pts_i, np.ones((len(pts_i), 1))])
        pts_transformed = (T @ pts_hom.T).T[:, :3]
        accumulated.append(pts_transformed)
        prev_pts = pts_i

    return np.vstack(accumulated)


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


def colorize_rails(points, intensity, rail_mask=None, ground_plane=None,
                    rail_lines=None):
    """
    Окраска облака точек:
    - Базовый слой: градиент по высоте (синий → голубой → зелёный → оливковый → фиолетовый)
    - Intensity усиливает яркость
    - Полотно пути (между рельсами, низко над землёй) — рыжий
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

    # --- Полотно пути — рыжий (между рельсами) ---
    if ground_plane is not None and rail_lines is not None and len(rail_lines) >= 2:
        a, b, c = ground_plane
        z_ground = a * points[:, 0] + b * points[:, 1] + c
        dz = points[:, 2] - z_ground
        low = (dz > -0.05) & (dz < 0.25)
        x_left = rail_x(rail_lines[0], points[:, 1])
        x_right = rail_x(rail_lines[1], points[:, 1])
        between = (points[:, 0] > x_left) & (points[:, 0] < x_right)
        trackbed_mask = low & between
        if rail_mask is not None:
            trackbed_mask &= ~rail_mask
        colors[trackbed_mask, 0] = 1.0
        colors[trackbed_mask, 1] = 0.45
        colors[trackbed_mask, 2] = 0.0

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


def _build_circle_offsets(radius):
    """Pre-compute (dy, dx) offsets for a filled circle of given radius."""
    offsets = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy <= radius * radius:
                offsets.append((dy, dx))
    return np.array(offsets, dtype=np.int32)  # (K, 2)


# Cache circle offsets per radius to avoid recomputing every frame
_circle_offset_cache = {}


def _render_frame_project(pts, colors, width=1280, height=720, point_radius=2,
                          bg_color=(13, 13, 26)):
    """Render point cloud by projecting 3D->2D with the same camera as Open3D viewer.

    No matplotlib, no Open3D — pure numpy. Fast and works everywhere.
    Camera: lookat=(0,-7,-1), front=(0,0.95,0.10), up=(0,0,1), zoom=0.04.
    """
    n = len(pts)
    if n == 0:
        img = np.full((height, width, 3), bg_color, dtype=np.uint8)
        return img

    # Camera setup matching Open3D
    lookat = np.array([0.0, -7.0, -1.0])
    front = np.array([0.0, 0.95, 0.10])
    front = front / np.linalg.norm(front)
    up = np.array([0.0, 0.0, 1.0])

    # Camera basis (right-hand: right, up, -front)
    right = np.cross(front, up)
    right = right / np.linalg.norm(right)
    cam_up = np.cross(right, front)
    cam_up = cam_up / np.linalg.norm(cam_up)

    # Camera position: behind lookat along front direction
    # zoom=0.04 in Open3D ~ distance. Tuned to match visual.
    cam_dist = 12.0
    cam_pos = lookat + front * cam_dist

    # Project points to camera space
    rel = pts - cam_pos  # (N, 3)
    # Camera coords: x=right, y=cam_up, z=-front (into screen)
    cx = rel @ right
    cy = rel @ cam_up
    cz = -(rel @ front)  # depth (positive = in front of camera)

    # Only keep points in front of camera
    valid = cz > 0.5
    cx = cx[valid]
    cy = cy[valid]
    cz = cz[valid]
    c = colors[valid]

    if len(cx) == 0:
        img = np.full((height, width, 3), bg_color, dtype=np.uint8)
        return img

    # Perspective projection
    fov_scale = 800.0  # focal length in pixels (tuned to match Open3D zoom=0.04)
    px = (cx / cz * fov_scale + width / 2).astype(np.int32)
    py = (-cy / cz * fov_scale + height / 2).astype(np.int32)  # flip Y

    # Clip to image bounds (with margin for point radius)
    margin = max(point_radius, 1)
    in_bounds = (px >= -margin) & (px < width + margin) & (py >= -margin) & (py < height + margin)
    px = px[in_bounds]
    py = py[in_bounds]
    cz_vis = cz[in_bounds]
    c = c[in_bounds]

    if len(px) == 0:
        img = np.full((height, width, 3), bg_color, dtype=np.uint8)
        return img

    # Sort by depth (far first, so near points overwrite)
    order = np.argsort(-cz_vis)
    px = px[order]
    py = py[order]
    c = c[order]

    # Draw into image
    img = np.full((height, width, 3), bg_color, dtype=np.uint8)
    c_uint8 = np.clip(c * 255, 0, 255).astype(np.uint8)

    if point_radius <= 1:
        # Clip to exact bounds for radius=1
        exact = (px >= 0) & (px < width) & (py >= 0) & (py < height)
        img[py[exact], px[exact]] = c_uint8[exact]
    else:
        # Vectorized: expand all points by circle offsets in one shot
        if point_radius not in _circle_offset_cache:
            _circle_offset_cache[point_radius] = _build_circle_offsets(point_radius)
        offsets = _circle_offset_cache[point_radius]  # (K, 2) — (dy, dx)
        K = len(offsets)
        N = len(px)

        # Repeat each point K times, add each offset
        # px_exp, py_exp: (N*K,)
        px_exp = np.repeat(px, K) + np.tile(offsets[:, 1], N)
        py_exp = np.repeat(py, K) + np.tile(offsets[:, 0], N)
        c_exp = np.repeat(c_uint8, K, axis=0)

        # Clip to image bounds
        valid_exp = (px_exp >= 0) & (px_exp < width) & (py_exp >= 0) & (py_exp < height)
        img[py_exp[valid_exp], px_exp[valid_exp]] = c_exp[valid_exp]

    return img


def _can_use_open3d_offscreen():
    """Check if Open3D offscreen rendering actually produces non-empty images.

    Open3D visible=False creates a window but capture_screen_float_buffer
    often returns an all-white/black buffer on headless/macOS/Colab.
    """
    if o3d is None:
        return False
    try:
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=64, height=64)
        # Add a visible point cloud to test actual rendering
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64))
        pcd.colors = o3d.utility.Vector3dVector(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64))
        vis.add_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        buf = vis.capture_screen_float_buffer(do_render=True)
        img = np.asarray(buf)
        vis.destroy_window()
        # If all pixels are the same color, rendering is broken
        return img.std() > 0.01
    except Exception:
        return False


def _record_mp4(args, frames, total, start_idx, xyz, intensity):
    """Offscreen rendering to MP4. Pipes raw RGB frames directly into ffmpeg (no temp PNGs)."""
    import subprocess
    import shutil

    W, H = 1280, 720

    print(f'Recording to {args.record}')

    end = total if args.record_end < 0 else min(args.record_end + 1, total)
    frame_indices = list(range(start_idx, end, args.record_every))
    if args.record_max_frames > 0:
        frame_indices = frame_indices[:args.record_max_frames]
    print(f'Frames {start_idx}..{end - 1} every {args.record_every} = {len(frame_indices)} frames')

    # Cache for pairwise dy — shared across all accumulate_with_speed calls
    dy_cache = {}

    # Initial rail detection
    n_acc = args.accumulate_speed or args.accumulate
    if args.accumulate_speed > 1:
        detect_pts, detect_int, init_speed = accumulate_with_speed(
            frames, start_idx, args.accumulate_speed, _dy_cache=dy_cache)
        print(f'Accumulation: {len(xyz)} -> {len(detect_pts)} pts '
              f'({args.accumulate_speed} frames, {init_speed:.1f} km/h)')
    else:
        detect_pts, detect_int = xyz, intensity

    rail_lines, ground_plane = find_rail_lines(
        detect_pts, detect_int if n_acc <= 1 else None,
        own_x_range=args.own_x_range, max_y_distance=args.max_y_distance)
    print(f'Rail lines: {len(rail_lines)}')

    # Pick encoder: try h264_nvenc (Colab GPU), fall back to libx264
    ffmpeg_path = shutil.which('ffmpeg')
    if not ffmpeg_path:
        print('ERROR: ffmpeg not found in PATH')
        return 1

    encoder = 'libx264'
    try:
        probe = subprocess.run(
            [ffmpeg_path, '-hide_banner', '-encoders'],
            capture_output=True, text=True, timeout=5)
        if 'h264_nvenc' in probe.stdout:
            # Actually test that nvenc works (Colab lists it but GPU may not support it)
            test_cmd = [
                ffmpeg_path, '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
                '-s', '64x64', '-pix_fmt', 'rgb24', '-r', '1',
                '-i', 'pipe:0',
                '-frames:v', '1', '-c:v', 'h264_nvenc', '-f', 'null', '-',
            ]
            test_proc = subprocess.run(
                test_cmd, input=b'\x00' * (64 * 64 * 3),
                capture_output=True, timeout=10)
            if test_proc.returncode == 0:
                encoder = 'h264_nvenc'
    except Exception:
        pass

    print(f'Renderer: numpy projection (same camera as Open3D), encoder: {encoder}')

    # Start ffmpeg pipe — raw RGB input, no temp files
    ffmpeg_cmd = [
        ffmpeg_path, '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{W}x{H}',
        '-pix_fmt', 'rgb24',
        '-r', str(int(args.fps)),
        '-i', '-',
    ]
    if encoder == 'h264_nvenc':
        ffmpeg_cmd += ['-c:v', 'h264_nvenc', '-preset', 'p4', '-cq', '23']
    else:
        ffmpeg_cmd += ['-c:v', 'libx264', '-preset', 'fast', '-crf', '23']
    ffmpeg_cmd += ['-pix_fmt', 'yuv420p', args.record]

    ffproc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE,
                              stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    t_start = time.time()
    last_detect_idx = start_idx
    redetect_every = args.redetect_every

    try:
        for i, idx in enumerate(frame_indices):
            # Get frame
            _cur_speed_kmh = 0.0
            if args.accumulate_speed > 1:
                pts, ints, _cur_speed_kmh = accumulate_with_speed(frames, idx, args.accumulate_speed,
                                                     _dy_cache=dy_cache)
            else:
                pts, ints, _ = frames[idx]

            # Re-detect rails
            if redetect_every > 0 and abs(idx - last_detect_idx) >= redetect_every:
                if args.accumulate_speed > 1:
                    det_pts, det_int, _ = accumulate_with_speed(
                        frames, idx, args.accumulate_speed, _dy_cache=dy_cache)
                    det_int = None
                else:
                    det_pts, det_int = pts, ints
                rl_new, gp_new = find_rail_lines(
                    det_pts, det_int, own_x_range=args.own_x_range,
                    max_y_distance=args.max_y_distance,
                    prev_rail_lines=rail_lines if rail_lines else None)
                if rl_new and gp_new is not None:
                    rail_lines = rl_new
                    ground_plane = gp_new
                last_detect_idx = idx

            rail_mask = apply_rail_lines(
                pts, rail_lines, ground_plane,
                own_track_only=True, max_y_distance=args.max_y_distance)
            colors = colorize_rails(pts, ints, rail_mask, ground_plane=ground_plane,
                                       rail_lines=rail_lines)

            # Subsample
            if 0 < args.max_points < len(pts):
                step = -(-len(pts) // args.max_points)
                pts = pts[::step]
                colors = colors[::step]

            img = _render_frame_project(pts, colors, width=W, height=H,
                                        point_radius=int(args.point_size))

            # Write raw RGB directly to ffmpeg stdin
            try:
                ffproc.stdin.write(img.tobytes())
            except BrokenPipeError:
                ffproc.wait()
                stderr = ffproc.stderr.read().decode(errors='replace')
                print(f'ffmpeg died at frame {idx} (rc={ffproc.returncode}):\n{stderr[-800:]}')
                return 1

            elapsed = time.time() - t_start
            speed_fps = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(frame_indices) - i - 1) / speed_fps if speed_fps > 0 else 0
            rail_y = np.abs(pts[rail_mask, 1]) if rail_mask.any() else np.array([0.0])
            rail_len = float(rail_y.max())
            if (i + 1) % 10 == 0 or i == 0:
                print(f'  [{i+1}/{len(frame_indices)}] frame {idx}, '
                      f'{_cur_speed_kmh:.0f} km/h, рельсы {rail_len:.0f}м ({int(rail_mask.sum())} точек), '
                      f'{speed_fps:.1f} fps, ETA {eta:.0f}s')
    finally:
        ffproc.stdin.close()
        ffproc.wait()

    if ffproc.returncode != 0:
        stderr = ffproc.stderr.read().decode(errors='replace')
        print(f'ffmpeg error (rc={ffproc.returncode}):\n{stderr[-800:]}')
        return 1

    wall = time.time() - t_start
    size_mb = os.path.getsize(args.record) / 1024 / 1024
    print(f'Done: {len(frame_indices)} frames in {wall:.1f}s ({len(frame_indices)/wall:.1f} fps)')
    print(f'Output: {args.record} ({size_mb:.1f} MB)')
    return 0


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
    parser.add_argument('--max-y-distance', type=float, default=25.0,
                        help='макс. дальность по Y от лидара, м (по умолчанию 20). '
                             'Дальше облако разреженное и рельсы уходят косо')
    parser.add_argument('--redetect-every', type=int, default=1,
                        help='пересчитывать рельсы каждые N кадров (по умолчанию 1). '
                             '0 = только на стартовом кадре')
    parser.add_argument('--accumulate', type=int, default=0,
                        help='накопление N кадров через ICP перед детекцией (0 = выкл, 3-5 рекомендуется)')
    parser.add_argument('--accumulate-speed', type=int, default=0,
                        help='накопление N кадров по скорости (отражатели). Быстрее ICP, для прямых тоннелей')
    parser.add_argument('--rail-labels', action='store_true',
                        help='использовать предрассчитанную разметку рельсов из rail_labels/ '
                             '(создаётся через python -m rail_mapping.run)')
    parser.add_argument('--record', type=str, default=None,
                        help='записать в MP4 файл (offscreen, без окна). Пример: --record out.mp4')
    parser.add_argument('--record-end', type=int, default=-1,
                        help='последний кадр для записи (-1 = все)')
    parser.add_argument('--record-every', type=int, default=1,
                        help='записывать каждый N-й кадр')
    parser.add_argument('--record-max-frames', type=int, default=-1,
                        help='макс. кадров для записи (для теста, -1 = все)')
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
                o3d.utility.Vector3dVector(colorize_rails(pts, intensity, rail_mask,
                                                          ground_plane=ground_plane,
                                                          rail_lines=rail_lines)))

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

    # --record: offscreen rendering to MP4
    if args.record:
        return _record_mp4(args, frames, total, idx, xyz, intensity)

    # Setup Open3D animated viewer
    if o3d is None:
        print('Open3D not available. Use --record to render to MP4 (matplotlib fallback).')
        return 1
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f'LiDAR - {os.path.basename(args.bag_dir)}', width=1280, height=720)

    # Detect rail lines on the starting frame
    dy_cache = {}  # shared cache for pairwise dy
    pts = xyz
    n_acc = args.accumulate_speed or args.accumulate
    if args.accumulate_speed > 1:
        detect_pts, detect_int, init_speed = accumulate_with_speed(
            frames, idx, args.accumulate_speed, _dy_cache=dy_cache)
        print(f'Накопление (скорость): {len(pts)} → {len(detect_pts)} точек '
              f'({args.accumulate_speed} кадров, {init_speed:.1f} km/h)')
    elif args.accumulate > 1:
        detect_pts = _accumulate_icp(frames, idx, args.accumulate)
        detect_int = None
        print(f'Накопление (ICP): {len(pts)} → {len(detect_pts)} точек ({args.accumulate} кадров)')
    else:
        detect_pts = pts
        detect_int = intensity
    rail_lines, ground_plane = find_rail_lines(detect_pts, detect_int if n_acc <= 1 else None,
                                                own_x_range=args.own_x_range,
                                                max_y_distance=args.max_y_distance)

    if label_dir:
        label_path = os.path.join(label_dir, f'{idx:03d}.npy')
        if os.path.exists(label_path):
            rail_mask = np.load(label_path)
        else:
            rail_mask = apply_rail_lines(pts, rail_lines, ground_plane, own_track_only=True, max_y_distance=args.max_y_distance)
    else:
        rail_mask = apply_rail_lines(pts, rail_lines, ground_plane, own_track_only=True, max_y_distance=args.max_y_distance)

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
        _cur_speed_kmh = 0.0
        if args.accumulate_speed > 1:
            pts, intensity, _cur_speed_kmh = accumulate_with_speed(
                frames, idx, args.accumulate_speed, _dy_cache=dy_cache)
        else:
            pts, intensity, _ = frames[idx]

        # Пересчёт рельсов каждые N кадров с проверкой консистентности
        if redetect_every > 0 and abs(idx - last_detect_idx) >= redetect_every:
            if args.accumulate_speed > 1:
                detect_pts = pts  # уже аккумулированные выше
                detect_int = None
            elif args.accumulate > 1:
                detect_pts = _accumulate_icp(frames, idx, args.accumulate)
                detect_int = None
            else:
                detect_pts = pts
                detect_int = intensity
            rail_lines_new, ground_plane_new = find_rail_lines(
                detect_pts, detect_int,
                own_x_range=args.own_x_range,
                max_y_distance=args.max_y_distance,
                prev_rail_lines=rail_lines if rail_lines else None)
            if rail_lines_new and ground_plane_new is not None:
                # Проверка: новые рельсы не должны сильно отличаться от предыдущих
                from rail_detection import _find_own_track, rail_x
                old_ti = _find_own_track(rail_lines)
                new_ti = _find_own_track(rail_lines_new)
                accept = True
                if old_ti is not None and new_ti is not None:
                    # Сдвиг осей рельсов не больше 0.15м (на Y=0)
                    old_mid = (rail_x(rail_lines[old_ti], 0) + rail_x(rail_lines[old_ti + 1], 0)) / 2
                    new_mid = (rail_x(rail_lines_new[new_ti], 0) + rail_x(rail_lines_new[new_ti + 1], 0)) / 2
                    if abs(new_mid - old_mid) > 0.15:
                        accept = False
                    # Колея не должна скакать больше чем на 100мм
                    old_gauge = abs(rail_x(rail_lines[old_ti + 1], 0) - rail_x(rail_lines[old_ti], 0))
                    new_gauge = abs(rail_x(rail_lines_new[new_ti + 1], 0) - rail_x(rail_lines_new[new_ti], 0))
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
                    g = abs(rail_x(rail_lines[ti+1], 0) - rail_x(rail_lines[ti], 0)) * 1000
                    mid = (rail_x(rail_lines[ti], 0) + rail_x(rail_lines[ti+1], 0)) / 2 * 1000
                    pass  # print(f'  [{idx:4d}] колея {g:.0f} мм  mid {mid:+.0f} мм  lines {len(rail_lines)}')
                else:
                    pass  # print(f'  [{idx:4d}] своя пара не найдена, lines {len(rail_lines)}')
            else:
                pass  # print(f'  [{idx:4d}] рельсы не найдены')
            last_detect_idx = idx

        if label_dir:
            label_path = os.path.join(label_dir, f'{idx:03d}.npy')
            if os.path.exists(label_path):
                rail_mask = np.load(label_path)
            else:
                rail_mask = apply_rail_lines(pts, rail_lines, ground_plane, own_track_only=True, max_y_distance=args.max_y_distance)
        else:
            rail_mask = apply_rail_lines(pts, rail_lines, ground_plane, own_track_only=True, max_y_distance=args.max_y_distance)

        # Вывод: скорость + длина рельсов
        rail_y = np.abs(pts[rail_mask, 1]) if rail_mask.any() else np.array([0.0])
        rail_len = float(rail_y.max())
        _n_rail = int(rail_mask.sum())
        print(f'\r  [{idx:4d}] {_cur_speed_kmh:5.1f} km/h  рельсы: {rail_len:5.1f}м  ({_n_rail} точек)', end='', flush=True)

        pcd.points, pcd.colors = prepare(pts, intensity, rail_mask)
        shown += 1

        vis.update_geometry(pcd)
        if not vis.poll_events():
            break
        vis.update_renderer()

        if time.time() - frame_start > 2 * frame_period and frame_period > 0 and slow_frames < 3:
            slow_frames += 1
            # print(f'  frame {idx}: {time.time() - frame_start:.3f}s per frame '
            #       f'(target {frame_period:.3f}s) — снизьте --fps или --max-points, если playback рвётся')

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