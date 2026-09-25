"""Чтение rosbag2 `.db3` (PointCloud2) и раскраски кадров -- без GUI.

ROS не нужен: `.db3` -- это SQLite, CDR разбирается вручную
(`parse_pointcloud2_cdr`). Модуль отдаёт кадры по индексу с кольцевым кэшем и
предзагрузкой (`BagFrames`), режет облако по дальности (`trim`) и красит его
скалярным полем (`frame_colors`). Семантические зоны ('zones' в `COLOR_MODES`)
считает `zones.py` -- тут только имя режима.

Кто пользуется: `web_viewer.py` (отдаёт кадры в браузер), `tests/test_core.py`.
"""

import os
import re
import sqlite3
import struct
import threading
from collections import OrderedDict, namedtuple

import numpy as np


MODES = ('height', 'intensity', 'range', 'ring')
# 'zones' -- не раскраска скалярного поля, а семантические классы из zones.py, поэтому
# frame_colors() её не знает; в список режимов вьюера она добавлена отдельно.
COLOR_MODES = MODES + ('zones',)

# Диапазоны по умолчанию для doubleT_obstacle (измерено: Z in [-7.0, 2.8]).
Z_RANGE = (-7.0, 3.0)
RING_COUNT = 128
CACHE_FRAMES = 32
PREFETCH_AHEAD = 8

# Аппроксимация Google turbo контрольными точками (линейная интерполяция).
TURBO_ANCHORS = np.array([
    [0.18995, 0.07176, 0.23217],
    [0.19483, 0.08339, 0.26149],
    [0.25701, 0.13633, 0.33211],
    [0.36294, 0.21263, 0.41467],
    [0.49859, 0.25098, 0.44280],
    [0.65493, 0.26085, 0.41072],
    [0.81076, 0.29928, 0.31780],
    [0.90183, 0.38623, 0.22730],
    [0.92614, 0.51292, 0.16060],
    [0.88749, 0.66573, 0.15924],
    [0.80938, 0.81255, 0.23496],
    [0.74133, 0.90882, 0.43540],
    [0.74527, 0.94931, 0.70633],
    [0.78752, 0.96592, 0.92003],
    [0.84070, 0.96458, 0.98167],
], dtype=np.float64)

_TURBO_X = np.linspace(0.0, 1.0, len(TURBO_ANCHORS))


def _ramp(t):
    """t (любой формы) -> RGB float64 той же формы + ось 3, значения 0..1."""
    t = np.clip(np.asarray(t, dtype=np.float64), 0.0, 1.0)
    return np.stack([np.interp(t, _TURBO_X, TURBO_ANCHORS[:, c]) for c in range(3)], axis=-1)


_LUT_LEVELS = 256
# Таблица цветов: np.interp по трём каналам на каждой точке стоит ~25 мс на 300k
# точек, индексация в LUT -- ~8 мс, при расхождении цвета ~0.015 (незаметно).
_TURBO_LUT = _ramp(np.linspace(0.0, 1.0, _LUT_LEVELS))


def _lut_colors(t):
    idx = np.clip(np.asarray(t, dtype=np.float64) * (_LUT_LEVELS - 1),
                  0.0, _LUT_LEVELS - 1).astype(np.uint8)
    return _TURBO_LUT[idx]


def parse_pointcloud2_cdr(data: bytes, with_ring=False):
    """Разобрать CDR-сериализованный PointCloud2 -> (xyz, intensity[, ring]).

    with_ring=True добавляет третьим элементом поле `ring` (uint16), если оно есть.

    Логика перенесена из существующего рабочего парсера и проверена на живых
    данных: point_step=26, x/y/z/intensity -- float32 @ 0/4/8/12,
    ring uint16 @ 16, timestamp float64 @ 18.
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
    if n_points == 0:
        empty = np.zeros((0, 3), dtype=np.float32)
        if with_ring:
            return empty, None, None
        return empty, None

    cloud = np.frombuffer(data, dtype=np.uint8, offset=offset, count=data_len)
    cloud_bo = '>' if is_bigendian else '<'
    dt = np.dtype(f'{cloud_bo}f4')

    # (n_points, point_step): одна строка на точку, поле -- обычный срез колонки.
    raw = cloud[:n_points * point_step].reshape(n_points, point_step)

    x_off = field_offsets.get('x')
    if x_off is None:
        raise ValueError('PointCloud2 has no "x" field')

    contiguous_quad = (
        field_offsets.get('y') == x_off + 4
        and field_offsets.get('z') == x_off + 8
        and field_offsets.get('intensity') == x_off + 12
    )
    if contiguous_quad:
        # x, y, z и intensity идут подряд -> одно непрерывное копирование
        # 16 байт на точку вместо четырёх отдельных копий колонок (замерено: 58 -> 40 мс).
        quad = np.ascontiguousarray(
            raw[:, x_off:x_off + 16]).view(dt).reshape(n_points, 4)
        xyz_source = quad[:, :3]
        intensity_source = quad[:, 3]
    else:
        def float_field(name):
            f_off = field_offsets.get(name)
            if f_off is None:
                return None
            return np.ascontiguousarray(raw[:, f_off:f_off + 4]).view(dt).reshape(n_points)

        xyz_source = np.empty((n_points, 3), dtype=np.float32)
        for axis_i, axis in enumerate(('x', 'y', 'z')):
            column = float_field(axis)
            if column is None:
                raise ValueError(f'PointCloud2 has no "{axis}" field')
            xyz_source[:, axis_i] = column
        intensity_source = float_field('intensity')

    ring = None
    if with_ring:
        r_off = field_offsets.get('ring')
        if r_off is not None:
            ring = np.ascontiguousarray(
                raw[:, r_off:r_off + 2]).view(f'{cloud_bo}u2').reshape(n_points)

    keep = (np.isfinite(xyz_source).all(axis=1)
            & (xyz_source != 0).any(axis=1))
    xyz = np.ascontiguousarray(xyz_source[keep])
    intensity = (None if intensity_source is None
                 else np.ascontiguousarray(intensity_source[keep]))
    if ring is not None:
        ring = np.ascontiguousarray(ring[keep])

    if with_ring:
        return xyz, intensity, ring
    return xyz, intensity


def frame_colors(xyz, intensity, mode='height', max_dist=80.0, ring=None,
                 z_range=Z_RANGE):
    """Раскраска кадра -> float64 (N,3) в 0..1. Палитра -- turbo."""
    xyz = np.asarray(xyz)
    n = len(xyz)
    out = np.empty((n, 3), dtype=np.float64)
    if n == 0:
        return out

    if mode == 'height':
        lo, hi = z_range
        t = (xyz[:, 2] - lo) / max(hi - lo, 1e-9)
    elif mode == 'intensity':
        if intensity is None:
            raise ValueError('intensity is None: cannot color by intensity')
        lo, hi = np.percentile(np.asarray(intensity, dtype=np.float64), [1, 99])
        t = (np.asarray(intensity, dtype=np.float64) - lo) / max(hi - lo, 1e-9)
    elif mode == 'range':
        t = np.linalg.norm(xyz, axis=1) / max(max_dist, 1e-9)
    elif mode == 'ring':
        if ring is None:
            raise ValueError('ring is None: cannot color by ring')
        t = (np.asarray(ring) % RING_COUNT) / float(RING_COUNT - 1)
    else:
        raise ValueError(f'unknown color mode {mode!r}, expected one of {MODES}')

    return _lut_colors(t)


def trim_range(xyz, max_dist=80.0, behind=5.0):
    """Маска: оставить коридор по Y. Туннель уходит в -Y, за спиной - `behind`."""
    y = np.asarray(xyz)[:, 1]
    return (y >= -max_dist) & (y <= behind)


def trim(xyz, intensity=None, ring=None, max_dist=80.0, behind=5.0):
    """Применить `trim_range` к xyz/intensity/ring. None-аргументы возвращаются как None."""
    mask = trim_range(xyz, max_dist=max_dist, behind=behind)
    out_xyz = np.asarray(xyz)[mask]
    out_inten = None if intensity is None else np.asarray(intensity)[mask]
    out_ring = None if ring is None else np.asarray(ring)[mask]
    return out_xyz, out_inten, out_ring


Frame = namedtuple('Frame', 'xyz intensity ring')


class BagFrames:
    """Доступ к кадрам rosbag2 `.db3` по индексу с кольцевым кэшем и предзагрузкой.

    Разбор идёт в фоновом потоке со своим соединением: объекты sqlite3.Connection
    не потокобезопасны.
    """

    def __init__(self, db_path, cache_size=CACHE_FRAMES, prefetch_ahead=PREFETCH_AHEAD):
        # Многофайловый bag: rosbag2 режет длинную запись на куски
        # <имя>_0.db3, <имя>_1.db3, ... Потребители передают ОДИН db_path (файл
        # или каталог записи) и не должны знать, один там файл или сто -- по
        # любому файлу записи подхватываем все её куски, порядок по
        # metadata.yaml (иначе по номеру суффикса), кадры склеиваются по
        # timestamp. Список путей тоже принимается.
        if isinstance(db_path, (list, tuple)):
            files = [str(p) for p in db_path]
        else:
            p = str(db_path)
            if os.path.isdir(p):
                files = find_db3s(p)
            else:
                d = os.path.dirname(p) or '.'
                same = [f for f in os.listdir(d) if f.endswith('.db3')]
                files = find_db3s(d) if len(same) > 1 else [p]
        if not files:
            raise FileNotFoundError(f'нет .db3: {db_path!r}')
        self._files = files
        self.db_path = files[0]      # совместимость: первый файл записи
        self.cache_size = cache_size
        self.prefetch_ahead = prefetch_ahead
        self.parse_count = 0
        self.hits = 0
        self.misses = 0
        self.prefetch_parses = 0
        self._lock = threading.Lock()
        # Соединение на поток: sqlite3.Connection нельзя использовать из другого
        # потока, а веб-вьюер обслуживает каждый HTTP-запрос в своём потоке.
        # Многофайловый bag -- соединение на (поток, файл).
        self._local = threading.local()
        self._readers = []
        self._readers_lock = threading.Lock()
        self._cache = OrderedDict()

        # Индекс: (timestamp, файл, rowid) по всем кускам, общий порядок по
        # timestamp -- он же порядок воспроизведения rosbag2.
        self._conn = sqlite3.connect(files[0])
        entries = []
        for fi, path in enumerate(files):
            conn = self._conn if fi == 0 else sqlite3.connect(path)
            try:
                cur = conn.cursor()
                try:
                    cur.execute("SELECT id FROM topics WHERE type LIKE '%PointCloud2%'")
                    topic_ids = [row[0] for row in cur.fetchall()]
                except sqlite3.Error:
                    topic_ids = []

                if topic_ids:
                    placeholders = ','.join('?' * len(topic_ids))
                    cur.execute(
                        f'SELECT rowid, timestamp FROM messages WHERE topic_id IN ({placeholders}) '
                        'ORDER BY timestamp ASC', topic_ids)
                else:
                    cur.execute('SELECT rowid, timestamp FROM messages ORDER BY timestamp ASC')
                rows = cur.fetchall()
                entries.extend((r[1], fi, r[0]) for r in rows)
                if fi == 0:
                    try:
                        self.topic_names = [row[0] for row in cur.execute('SELECT name FROM topics')]
                    except sqlite3.Error:
                        self.topic_names = []
            finally:
                if fi != 0:
                    conn.close()
        entries.sort()
        self.rowids = [(fi, rid) for _, fi, rid in entries]
        self.timestamps = np.array([e[0] for e in entries], dtype=np.int64)

        self._want = 0
        self._stop = threading.Event()
        self._thread = None
        if prefetch_ahead > 0 and len(self.rowids) > 1:
            self._thread = threading.Thread(target=self._prefetch_loop, daemon=True)
            self._thread.start()

    def _conn_for_thread(self, file_idx=0):
        """Соединение для текущего потока и нужного куска (sqlite3 не потокобезопасен)."""
        conns = getattr(self._local, 'conns', None)
        if conns is None:
            conns = {}
            self._local.conns = conns
        conn = conns.get(file_idx)
        if conn is None:
            conn = sqlite3.connect(self._files[file_idx])
            conns[file_idx] = conn
            with self._readers_lock:
                self._readers.append(conn)
        return conn

    def __len__(self):
        return len(self.rowids)

    def timestamp(self, index):
        """Метка времени кадра (наносекунды) — как третий элемент у других парсеров."""
        return int(self.timestamps[index])

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        with self._readers_lock:
            readers, self._readers = self._readers, []
        for conn in readers:
            try:
                conn.close()
            except sqlite3.Error:
                pass
        try:
            self._conn.close()
        except sqlite3.Error:
            pass

    def want(self, index):
        """Сообщить потоку предзагрузки, откуда продолжать."""
        self._want = int(index)

    def _parse_with(self, conn, index):
        file_idx, rowid = self.rowids[index]
        row = conn.execute('SELECT data FROM messages WHERE rowid=?',
                           (rowid,)).fetchone()
        if row is None:
            raise IndexError(index)
        with self._lock:
            self.parse_count += 1
            if threading.current_thread() is not threading.main_thread():
                self.prefetch_parses += 1
        return Frame(*parse_pointcloud2_cdr(row[0], with_ring=True))

    def _get_cached(self, index):
        with self._lock:
            if index in self._cache:
                self._cache.move_to_end(index)
                return self._cache[index]
        return None

    def _store(self, index, frame):
        with self._lock:
            self._cache[index] = frame
            self._cache.move_to_end(index)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)

    def __getitem__(self, index):
        if index < 0 or index >= len(self.rowids):
            raise IndexError(index)
        cached = self._get_cached(index)
        if cached is not None:
            with self._lock:
                self.hits += 1
            return cached
        with self._lock:
            self.misses += 1
        file_idx, _ = self.rowids[index]
        frame = self._parse_with(self._conn_for_thread(file_idx), index)
        self._store(index, frame)
        return frame

    def _prefetch_loop(self):
        conns = {}
        try:
            while not self._stop.is_set():
                start = self._want
                end = min(start + 1 + self.prefetch_ahead, len(self.rowids))
                for idx in range(start + 1, end):
                    if self._stop.is_set():
                        return
                    if self._get_cached(idx) is not None:
                        continue
                    try:
                        fi, _ = self.rowids[idx]
                        conn = conns.get(fi)
                        if conn is None:
                            conn = sqlite3.connect(self._files[fi])
                            conns[fi] = conn
                        self._store(idx, self._parse_with(conn, idx))
                    except (sqlite3.Error, IndexError):
                        return
                self._stop.wait(0.02)
        finally:
            for conn in conns.values():
                try:
                    conn.close()
                except sqlite3.Error:
                    pass


def _db3_order(bag_dir, names):
    """Порядок кусков многофайлового bag.

    Источник -- metadata.yaml (`relative_file_paths`, порядок записи rosbag2);
    порядок кусков в самом архиве/каталоге СЛУЧАЙНЫЙ (замерено на new_data).
    Без metadata -- числовая сортировка по суффиксу _N.
    """
    meta = os.path.join(bag_dir, 'metadata.yaml')
    if os.path.isfile(meta):
        try:
            txt = open(meta, encoding='utf-8').read()
            listed = [os.path.basename(p) for p in re.findall(r'-\s*([\w./\\-]+\.db3)', txt)]
            if listed and set(listed) == set(names):
                return listed
        except OSError:
            pass

    def key(n):
        m = re.search(r'_(\d+)\.db3$', n)
        return (0, int(m.group(1)), n) if m else (1, 0, n)

    return sorted(names, key=key)


def find_db3s(bag_dir):
    """Все .db3 записи в правильном порядке (многофайловые bag'и)."""
    db3_files = [f for f in os.listdir(bag_dir) if f.endswith('.db3')]
    if not db3_files:
        raise FileNotFoundError(f'No .db3 files in {bag_dir}')
    ordered = _db3_order(bag_dir, db3_files) if len(db3_files) > 1 else db3_files
    return [os.path.join(bag_dir, n) for n in ordered]


def find_db3(bag_dir):
    db3_files = sorted(f for f in os.listdir(bag_dir) if f.endswith('.db3'))
    if not db3_files:
        raise FileNotFoundError(f'No .db3 files in {bag_dir}')
    return os.path.join(bag_dir, db3_files[0])


