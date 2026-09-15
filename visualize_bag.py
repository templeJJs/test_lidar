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

    # Setup Open3D animated viewer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"LiDAR - {os.path.basename(bag_dir)}", width=1280, height=720)

    pcd = o3d.geometry.PointCloud()
    pts, intensity, _ = frames[0]
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colorize_rails(pts, intensity))
    vis.add_geometry(pcd)

    # Set render options
    opt = vis.get_render_option()
    opt.point_size = 2.0
    opt.background_color = np.array([0.01, 0.01, 0.03])

    # Камера из позиции лидара (0,0,0), смотрим вдоль пути (-Y)
    ctr = vis.get_view_control()
    ctr.set_lookat([0.0, -30.0, 0.0])  # смотрим на точку 30м вперёд по пути
    ctr.set_front([0.0, 1.0, 0.0])     # направление взгляда вдоль -Y
    ctr.set_up([0.0, 0.0, 1.0])        # Z вверх
    ctr.set_zoom(0.015)

    print("Playing animation...")
    print("  Close window to stop")

    frame_idx = 0
    last_time = time.time()
    fps_target = 10  # ~10 Hz playback

    while True:
        pts, intensity, ts = frames[frame_idx]

        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(colorize_rails(pts, intensity))

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
