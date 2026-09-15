"""
Анализ данных из bag-файла: что меняется между кадрами, что даёт intensity.
"""
import sqlite3
import struct
import numpy as np
import sys
import os


def parse_frame(data: bytes):
    """Parse CDR PointCloud2, return (xyz, intensity) arrays."""
    bo = '<' if data[1] == 1 else '>'
    offset = 4

    def read_u32(off):
        return struct.unpack_from(f'{bo}I', data, off)[0], off + 4
    def read_u8(off):
        return data[off], off + 1
    def read_string(off):
        slen, off = read_u32(off)
        s = data[off:off + slen - 1].decode('utf-8')
        return s, off + slen

    sec, offset = read_u32(offset)
    nsec, offset = read_u32(offset)
    frame_id, offset = read_string(offset)
    if offset % 4: offset += 4 - offset % 4
    height, offset = read_u32(offset)
    width, offset = read_u32(offset)

    num_fields, offset = read_u32(offset)
    fields = {}
    for _ in range(num_fields):
        fname, offset = read_string(offset)
        if offset % 4: offset += 4 - offset % 4
        f_offset, offset = read_u32(offset)
        f_datatype, offset = read_u8(offset)
        if offset % 4: offset += 4 - offset % 4
        f_count, offset = read_u32(offset)
        fields[fname] = f_offset

    is_bigendian, offset = read_u8(offset)
    if offset % 4: offset += 4 - offset % 4
    point_step, offset = read_u32(offset)
    row_step, offset = read_u32(offset)
    data_len, offset = read_u32(offset)

    cloud = np.frombuffer(data, dtype=np.uint8, offset=offset, count=data_len)
    n = height * width
    cbo = '>' if is_bigendian else '<'
    dt = np.dtype(f'{cbo}f4')

    xyz = np.zeros((n, 3), dtype=np.float32)
    for i, ax in enumerate(('x', 'y', 'z')):
        off = fields[ax]
        for j in range(n):
            base = j * point_step + off
            xyz[j, i] = np.frombuffer(cloud[base:base+4], dtype=dt)[0]

    intensity = np.zeros(n, dtype=np.float32)
    ioff = fields.get('intensity')
    if ioff is not None:
        for j in range(n):
            base = j * point_step + ioff
            intensity[j] = np.frombuffer(cloud[base:base+4], dtype=dt)[0]

    # Ring channel
    ring = None
    roff = fields.get('ring')
    if roff is not None:
        rdt = np.dtype(f'{cbo}u2')  # uint16
        ring = np.zeros(n, dtype=np.uint16)
        for j in range(n):
            base = j * point_step + roff
            ring[j] = np.frombuffer(cloud[base:base+2], dtype=rdt)[0]

    return xyz, intensity, ring, sec + nsec * 1e-9


def main():
    bag_dir = sys.argv[1] if len(sys.argv) > 1 else 'for_hackathon/doubleT_platform'
    db3 = [f for f in os.listdir(bag_dir) if f.endswith('.db3')][0]
    db_path = os.path.join(bag_dir, db3)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Берём первые 5 кадров + последний
    cur.execute("SELECT data, timestamp FROM messages ORDER BY timestamp ASC LIMIT 5")
    rows = cur.fetchall()
    cur.execute("SELECT data, timestamp FROM messages ORDER BY timestamp DESC LIMIT 1")
    last = cur.fetchone()
    conn.close()

    print(f"=== Анализ: {bag_dir} ===\n")

    frames = []
    for i, (blob, ts) in enumerate(rows):
        xyz, intensity, ring, t = parse_frame(blob)
        valid = np.isfinite(xyz).all(axis=1)
        nonzero = np.any(xyz != 0, axis=1)
        mask = valid & nonzero

        print(f"--- Кадр {i} (t={t:.3f}) ---")
        print(f"  Всего точек: {len(xyz)}")
        print(f"  Валидных (не NaN, не 0): {mask.sum()}")
        print(f"  Нулевых точек (x=y=z=0): {(~nonzero & valid).sum()}")
        print(f"  X: [{xyz[mask,0].min():.2f}, {xyz[mask,0].max():.2f}]")
        print(f"  Y: [{xyz[mask,1].min():.2f}, {xyz[mask,1].max():.2f}]")
        print(f"  Z: [{xyz[mask,2].min():.2f}, {xyz[mask,2].max():.2f}]")
        print(f"  Intensity: min={intensity[mask].min():.1f}, max={intensity[mask].max():.1f}, "
              f"mean={intensity[mask].mean():.1f}, median={np.median(intensity[mask]):.1f}")
        if ring is not None:
            print(f"  Ring: min={ring[mask].min()}, max={ring[mask].max()}, unique={len(np.unique(ring[mask]))}")

        # Гистограмма intensity
        hist, edges = np.histogram(intensity[mask], bins=10)
        print(f"  Intensity гистограмма:")
        for h, e1, e2 in zip(hist, edges[:-1], edges[1:]):
            bar = '#' * min(int(h / max(hist) * 40), 40)
            print(f"    [{e1:7.1f} - {e2:7.1f}]: {h:6d} {bar}")

        frames.append((xyz[mask], intensity[mask]))
        print()

    # Сравнение кадров 0 и 1
    print("=== Сравнение кадра 0 и кадра 1 ===")
    p0, i0 = frames[0]
    p1, i1 = frames[1]

    # Bounding box сдвиг
    for ax, name in enumerate(['X', 'Y', 'Z']):
        d = p1[:, ax].mean() - p0[:, ax].mean()
        print(f"  Сдвиг среднего {name}: {d:.4f} м")

    # Последний кадр
    xyz_last, int_last, ring_last, t_last = parse_frame(last[0])
    mask_last = np.isfinite(xyz_last).all(axis=1) & np.any(xyz_last != 0, axis=1)
    print(f"\n=== Последний кадр (t={t_last:.3f}) ===")
    print(f"  Валидных точек: {mask_last.sum()}")
    print(f"  X: [{xyz_last[mask_last,0].min():.2f}, {xyz_last[mask_last,0].max():.2f}]")
    print(f"  Y: [{xyz_last[mask_last,1].min():.2f}, {xyz_last[mask_last,1].max():.2f}]")
    print(f"  Z: [{xyz_last[mask_last,2].min():.2f}, {xyz_last[mask_last,2].max():.2f}]")

    d_total = xyz_last[mask_last].mean(axis=0) - p0.mean(axis=0)
    print(f"\n  Общий сдвиг среднего от кадра 0:")
    print(f"    X: {d_total[0]:.4f} м")
    print(f"    Y: {d_total[1]:.4f} м")
    print(f"    Z: {d_total[2]:.4f} м")


if __name__ == '__main__':
    main()
