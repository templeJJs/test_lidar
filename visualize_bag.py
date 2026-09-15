"""
Animated visualization of PointCloud2 from ROS 2 bag (.db3).
No ROS needed — reads SQLite directly, renders with Open3D.
"""

import sqlite3
import struct
import numpy as np
import open3d as o3d
import sys
import os
import time


def parse_pointcloud2_cdr(data: bytes):
    """Parse CDR-serialized PointCloud2, return (xyz, intensity) as numpy arrays."""
    encapsulation = data[:4]
    little_endian = encapsulation[1] == 1
    bo = '<' if little_endian else '>'
    offset = 4

    def read_u32(off):
        return struct.unpack_from(f'{bo}I', data, off)[0], off + 4

    def read_u8(off):
        return data[off], off + 1

    def read_string(off):
        slen, off = read_u32(off)
        s = data[off:off + slen - 1].decode('utf-8')
        off += slen
        return s, off

    sec, offset = read_u32(offset)
    nsec, offset = read_u32(offset)
    frame_id, offset = read_string(offset)

    if offset % 4 != 0:
        offset += 4 - (offset % 4)
    height, offset = read_u32(offset)
    width, offset = read_u32(offset)

    num_fields, offset = read_u32(offset)
    fields = []
    for _ in range(num_fields):
        fname, offset = read_string(offset)
        if offset % 4 != 0:
            offset += 4 - (offset % 4)
        f_offset, offset = read_u32(offset)
        f_datatype, offset = read_u8(offset)
        if offset % 4 != 0:
            offset += 4 - (offset % 4)
        f_count, offset = read_u32(offset)
        fields.append({'name': fname, 'offset': f_offset})

    is_bigendian, offset = read_u8(offset)
    if offset % 4 != 0:
        offset += 4 - (offset % 4)
    point_step, offset = read_u32(offset)
    row_step, offset = read_u32(offset)
    data_len, offset = read_u32(offset)
    cloud_data = np.frombuffer(data, dtype=np.uint8, offset=offset, count=data_len)

    field_map = {f['name']: f['offset'] for f in fields}
    cloud_bo = '>' if is_bigendian else '<'
    dt = np.dtype(f'{cloud_bo}f4')

    n_points = height * width

    # Extract x, y, z using stride tricks
    points = np.zeros((n_points, 3), dtype=np.float32)
    for axis_i, axis in enumerate(('x', 'y', 'z')):
        off = field_map[axis]
        # View each float at strided positions
        for i in range(0, n_points, 10000):
            end = min(i + 10000, n_points)
            for j in range(i, end):
                base = j * point_step + off
                points[j, axis_i] = np.frombuffer(cloud_data[base:base+4], dtype=dt)[0]

    # Extract intensity if available
    intensity = None
    if 'intensity' in field_map:
        intensity = np.zeros(n_points, dtype=np.float32)
        ioff = field_map['intensity']
        for j in range(n_points):
            base = j * point_step + ioff
            intensity[j] = np.frombuffer(cloud_data[base:base+4], dtype=dt)[0]

    # Filter invalid
    valid = np.isfinite(points).all(axis=1) & np.any(points != 0, axis=1)
    points = points[valid]
    if intensity is not None:
        intensity = intensity[valid]

    return points, intensity


def find_rail_lines(points, intensity, ground_height_range=(-0.02, 0.12),
                    intensity_threshold=12):
    """
    Находит линии рельсов через RANSAC земли + гистограммный анализ.

    Возвращает:
        rail_lines: list of (slope, intercept) — линии x = slope*y + intercept
        ground_plane: (a, b, c) — коэффициенты плоскости z = a*x + b*y + c

    Тяжёлая функция — вызывать один раз на первом кадре.
    """
    from sklearn.linear_model import RANSACRegressor

    if len(points) < 100 or intensity is None:
        return [], None

    z = points[:, 2]
    z_med = np.median(z)

    # 1. RANSAC ground plane: z = a*x + b*y + c
    ground_cand = (z > z_med - 1.5) & (z < z_med + 0.5)
    if ground_cand.sum() < 50:
        return [], None

    ransac = RANSACRegressor(residual_threshold=0.1, max_trials=500)
    ransac.fit(points[ground_cand, :2], z[ground_cand])
    a, b = ransac.estimator_.coef_
    c = ransac.estimator_.intercept_
    ground_plane = (a, b, c)

    z_ground = ransac.predict(points[:, :2])
    dz = z - z_ground

    # 2. Rail candidates: near ground + brighter than average
    near_ground = (dz > ground_height_range[0]) & (dz < ground_height_range[1])
    bright = intensity > intensity_threshold
    candidates = near_ground & bright

    if candidates.sum() < 20:
        return [], ground_plane

    cpts = points[candidates]

    # 3. Scan Y in overlapping strips, build X histogram, find peaks
    y_min, y_max = cpts[:, 1].min(), cpts[:, 1].max()
    y_range = y_max - y_min
    if y_range < 1.0:
        return [], ground_plane

    strip_width = max(2.0, y_range / 15)
    strip_step = strip_width / 2
    x_bin_size = 0.1

    votes = []
    y_start = y_min
    while y_start < y_max:
        strip = (cpts[:, 1] >= y_start) & (cpts[:, 1] < y_start + strip_width)
        if strip.sum() > 5:
            x_vals = cpts[strip, 0]
            bins = np.arange(x_vals.min() - 0.1, x_vals.max() + 0.2, x_bin_size)
            if len(bins) > 2:
                hist, edges = np.histogram(x_vals, bins=bins)
                centers = (edges[:-1] + edges[1:]) / 2
                threshold = max(3, hist.max() * 0.15)
                for j in range(len(hist)):
                    if hist[j] >= threshold:
                        left = hist[j - 1] if j > 0 else 0
                        right = hist[j + 1] if j < len(hist) - 1 else 0
                        if hist[j] >= left and hist[j] >= right:
                            votes.append([centers[j], y_start + strip_width / 2])
        y_start += strip_step

    if len(votes) < 3:
        return [], ground_plane

    votes = np.array(votes)

    # 4. Cluster votes by X position to find rail lines
    from sklearn.cluster import DBSCAN
    db = DBSCAN(eps=0.3, min_samples=2).fit(votes[:, 0].reshape(-1, 1))
    labels = db.labels_
    unique_labels = set(labels) - {-1}

    rail_lines = []
    for lbl in unique_labels:
        lbl_mask = labels == lbl
        v = votes[lbl_mask]
        if len(v) < 2:
            continue
        y_v = v[:, 1]
        x_v = v[:, 0]
        if y_v.max() - y_v.min() < 1.0:
            rail_lines.append((0.0, x_v.mean()))
        else:
            coeffs = np.polyfit(y_v, x_v, 1)
            rail_lines.append((coeffs[0], coeffs[1]))

    return rail_lines, ground_plane


def apply_rail_lines(points, rail_lines, ground_plane, rail_radius=0.04):
    """
    Быстрое применение найденных рельсовых линий к новому кадру.
    Возвращает булеву маску (True = рельс).
    """
    n = len(points)
    rail_mask = np.zeros(n, dtype=bool)

    if not rail_lines or ground_plane is None or n == 0:
        return rail_mask

    a, b, c = ground_plane
    z_ground = a * points[:, 0] + b * points[:, 1] + c
    dz = points[:, 2] - z_ground
    near_ground = (dz > -0.02) & (dz < 0.12)
    near_ground_idx = np.where(near_ground)[0]

    if len(near_ground_idx) == 0:
        return rail_mask

    ng_y = points[near_ground_idx, 1]
    ng_x = points[near_ground_idx, 0]

    for slope, intercept in rail_lines:
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


def colorize_rails(points, intensity, rail_mask=None):
    """
    Яркая окраска облака точек:
    - Базовый слой: градиент по высоте (синий внизу → голубой → зелёный → жёлтый → красный вверху)
    - Intensity усиливает яркость
    - Рельсы (из rail_mask) — ярко-зелёный
    - Конструкции выше земли — бирюзовый
    - Отражатели — ярко-белый
    """
    n = len(points)
    colors = np.zeros((n, 3))

    if n == 0:
        return colors

    z = points[:, 2]
    z_min, z_max = np.percentile(z, 1), np.percentile(z, 99)
    z_norm = np.clip((z - z_min) / (z_max - z_min + 1e-8), 0, 1)

    # --- Базовый слой: turbo-подобная цветовая карта по высоте ---
    cmap = np.array([
        [0.0,  0.15, 0.15, 0.80],  # тёмно-синий
        [0.25, 0.10, 0.55, 0.90],  # голубой
        [0.45, 0.10, 0.80, 0.40],  # зелёный
        [0.70, 0.90, 0.85, 0.15],  # жёлтый
        [1.0,  0.95, 0.20, 0.10],  # красный
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

        # --- Конструкции выше земли — яркий бирюзовый ---
        struct_mask = ~ground_mask & (i > 30)
        s_t = np.clip((i[struct_mask] - 30) / 100.0, 0, 1)
        colors[struct_mask, 0] = 0.1 + 0.2 * s_t
        colors[struct_mask, 1] = 0.6 + 0.3 * s_t
        colors[struct_mask, 2] = 0.7 + 0.25 * s_t

        # --- Отражатели/знаки — ярко-белый/розовый ---
        hot = i > 150
        colors[hot, 0] = 1.0
        colors[hot, 1] = 0.6
        colors[hot, 2] = 0.6

        ultra = i > 230
        colors[ultra] = 1.0

    # --- Рельсы — ярко-зелёный (поверх всего) ---
    if rail_mask is not None and rail_mask.any():
        colors[rail_mask, 0] = 0.1
        colors[rail_mask, 1] = 0.9
        colors[rail_mask, 2] = 0.1

    return colors


def main():
    bag_dir = sys.argv[1] if len(sys.argv) > 1 else 'for_hackathon/doubleT_platform'

    db3_files = [f for f in os.listdir(bag_dir) if f.endswith('.db3')]
    if not db3_files:
        print(f"No .db3 files found in {bag_dir}")
        return

    db_path = os.path.join(bag_dir, db3_files[0])
    print(f"Reading: {db_path}")

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM messages")
    total = cur.fetchone()[0]
    print(f"Total frames: {total}")

    # Preload all frames
    print("Loading all frames into memory...")
    frames = []
    cur.execute("SELECT data, timestamp FROM messages ORDER BY timestamp ASC")
    for i, (blob, ts) in enumerate(cur):
        pts, intensity = parse_pointcloud2_cdr(blob)
        frames.append((pts, intensity, ts))
        if (i + 1) % 50 == 0:
            print(f"  Loaded {i+1}/{total}")
    conn.close()
    print(f"Loaded {len(frames)} frames\n")

    # Диагностика: где облако точек?
    pts0 = frames[0][0]
    print(f"=== Статистика первого кадра ({len(pts0)} точек) ===")
    print(f"  X: min={pts0[:,0].min():.1f}  max={pts0[:,0].max():.1f}  mean={pts0[:,0].mean():.1f}")
    print(f"  Y: min={pts0[:,1].min():.1f}  max={pts0[:,1].max():.1f}  mean={pts0[:,1].mean():.1f}")
    print(f"  Z: min={pts0[:,2].min():.1f}  max={pts0[:,2].max():.1f}  mean={pts0[:,2].mean():.1f}")
    # Где больше всего точек — ближе к 0 или дальше?
    dist = np.sqrt(pts0[:,0]**2 + pts0[:,1]**2 + pts0[:,2]**2)
    print(f"  Расстояние от (0,0,0): min={dist.min():.1f}  max={dist.max():.1f}  median={np.median(dist):.1f}")
    print()

    # Setup Open3D animated viewer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"LiDAR - {os.path.basename(bag_dir)}", width=1280, height=720)

    # Detect rail lines on the first frame (heavy, ~250ms)
    pts, intensity, _ = frames[0]
    print("Detecting rail lines...")
    rail_lines, ground_plane = find_rail_lines(pts, intensity)
    print(f"Found {len(rail_lines)} rail lines")

    rail_mask = apply_rail_lines(pts, rail_lines, ground_plane)
    print(f"Rails in frame 0: {rail_mask.sum()} points")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colorize_rails(pts, intensity, rail_mask))
    vis.add_geometry(pcd)

    # Set render options
    opt = vis.get_render_option()
    opt.point_size = 3.0
    opt.background_color = np.array([1.0, 1.0, 1.0])

    # Камера из позиции лидара (0,0,0), смотрим вдоль пути (-Y)
    ctr = vis.get_view_control()
    ctr.set_lookat([0.0, -7.0, -1.0])   # центр основного облака
    ctr.set_front([0.0, 0.95, 0.10])   # вдоль пути, почти горизонтально
    ctr.set_up([0.0, 0.0, 1.0])        # Z вверх
    ctr.set_zoom(0.04)

    print("Playing animation...")
    print("  Close window to stop")

    frame_idx = 0
    last_time = time.time()
    fps_target = 10  # ~10 Hz playback

    while True:
        pts, intensity, ts = frames[frame_idx]
        rail_mask = apply_rail_lines(pts, rail_lines, ground_plane)

        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(colorize_rails(pts, intensity, rail_mask))

        vis.update_geometry(pcd)
        if not vis.poll_events():
            break
        vis.update_renderer()

        frame_idx = (frame_idx + 1) % len(frames)

        # Pace playback
        elapsed = time.time() - last_time
        sleep_time = 1.0 / fps_target - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
        last_time = time.time()

    vis.destroy_window()
    print("Done.")


if __name__ == '__main__':
    main()
