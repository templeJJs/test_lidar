"""Разбор PointCloud2 без open3d и без ROS — для прод-ноды и офлайн-чтения.

Два входа, один выход (xyz float32 (N,3), intensity float32 (N,) или None):

    parse_pointcloud2_cdr(data)  — СЫРЫЕ CDR-байты сообщения (поле data в
                                   sqlite-бэге rosbag2). Каноническая реализация;
                                   visualize_bag.parse_pointcloud2_cdr —
                                   тонкая обёртка над ней.
    points_from_msg(msg)         — уже десериализованный sensor_msgs/PointCloud2
                                   (подписка в ROS-ноде).

Фильтрация одинаковая: отбрасываются NaN/Inf и нулевые (0,0,0) точки.
"""

import struct

import numpy as np


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


def points_from_msg(msg):
    """(xyz float32 (N,3), intensity float32 (N,) или None) из сообщения.

    Та же страйдовая техника, что в parse_pointcloud2_cdr: колонки читаются
    по смещениям из msg.fields, поэтому разбор не зависит от порядка полей.
    Форматы полей, отличные от float32, не поддерживаются (на записях ТЗ и
    у Hesai/Livox координаты — float32).
    """
    point_step = int(msg.point_step)
    n_declared = int(msg.width) * int(msg.height)
    if n_declared == 0 or point_step == 0:
        return np.zeros((0, 3), dtype=np.float32), None

    field_offsets = {}
    for f in msg.fields:
        field_offsets[f.name] = (int(f.offset), int(f.datatype))
    FLOAT32 = 7                  # sensor_msgs/PointField.FLOAT32
    for axis in ('x', 'y', 'z'):
        if axis not in field_offsets:
            raise ValueError(f'PointCloud2 has no "{axis}" field')
        if field_offsets[axis][1] != FLOAT32:
            raise ValueError(f'PointCloud2 field "{axis}" is not float32')
    if ('intensity' in field_offsets
            and field_offsets['intensity'][1] != FLOAT32):
        raise ValueError('PointCloud2 field "intensity" is not float32')
    field_offsets = {name: off for name, (off, _dt) in field_offsets.items()}

    raw = np.frombuffer(msg.data, dtype=np.uint8)
    n_points = min(n_declared, raw.size // point_step)
    if n_points == 0:
        return np.zeros((0, 3), dtype=np.float32), None

    cloud_bo = '>' if msg.is_bigendian else '<'
    names = [name for name in ('x', 'y', 'z', 'intensity') if name in field_offsets]
    record = np.frombuffer(raw, dtype=np.dtype({
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
