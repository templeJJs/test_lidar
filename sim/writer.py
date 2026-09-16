"""Запись синтетических кадров в rosbag2 `.db3` ровно в том виде, в каком их
читает `bag_reader`: CDR PointCloud2, point_step 26, поля x,y,z,intensity,ring,timestamp.

Схема SQLite -- как у rosbag2: `topics(id, name, type, serialization_format, ...)`
и `messages(id, topic_id, timestamp, data)`. Идентификатор темы 1.
`BagFrames` ищет темы по типу PointCloud2 и берёт сообщения по возрастанию
`timestamp`, поэтому важен только он -- целые наносекунды от фиксированного начала.
"""
from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass

import numpy as np

POINT_STEP = 26
FIELDS = (('x', 0), ('y', 4), ('z', 8), ('intensity', 12), ('ring', 16), ('timestamp', 18))
# ROS PointConstants: 7 -- FLOAT32, 4 -- UINT16, 8 -- FLOAT64
_DATATYPES = (7, 7, 7, 7, 4, 8)


@dataclass
class FrameOut:
    xyz: np.ndarray
    intensity: np.ndarray
    ring: np.ndarray
    timestamp: float


def _align4(n: int) -> int:
    return n + (-n % 4)


def encode_pointcloud2_cdr(frame: FrameOut, frame_id: str = 'lidar') -> bytes:
    """CDR little-endian PointCloud2: заголовок, поля, данные по 26 байт на точку.

    Порядок и выравнивание те же, что разбирает `bag_reader.parse_pointcloud2_cdr`:
    строки (frame_id, имена полей) с 4-байтовой длиной и добивкой до кратного 4,
    перед `point_step` -- байт is_bigendian с добивкой.
    """
    n = int(frame.xyz.shape[0])
    body = bytearray()
    body += bytes([0x00, 0x01, 0x00, 0x00])                     # encapsulation
    body += b'\x00' * 8                                         # stamp (sec, nsec) -- по кадру
    name = frame_id.encode() + b'\x00'
    body += len(name).to_bytes(4, 'little') + name
    body += b'\x00' * (-len(body) % 4)
    body += (1).to_bytes(4, 'little')                           # height
    body += n.to_bytes(4, 'little')                             # width
    body += len(FIELDS).to_bytes(4, 'little')
    for i, (fname, off) in enumerate(FIELDS):
        nm = fname.encode() + b'\x00'
        body += len(nm).to_bytes(4, 'little') + nm
        body += b'\x00' * (-len(body) % 4)
        body += off.to_bytes(4, 'little')                       # offset
        body += _DATATYPES[i].to_bytes(1, 'little')             # datatype
        body += b'\x00' * (-len(body) % 4)
        body += (1).to_bytes(4, 'little')                       # count
    body += bytes([0])                                          # is_bigendian
    body += b'\x00' * (-len(body) % 4)
    body += POINT_STEP.to_bytes(4, 'little')
    body += (POINT_STEP * n).to_bytes(4, 'little')              # row_step
    body += (POINT_STEP * n).to_bytes(4, 'little')              # data_len

    cloud = np.zeros((n, POINT_STEP), dtype=np.uint8)
    cloud[:, 0:12] = frame.xyz.astype('<f4').view(np.uint8).reshape(n, 12)
    cloud[:, 12:16] = frame.intensity.astype('<f4').view(np.uint8).reshape(n, 4)
    cloud[:, 16:18] = frame.ring.astype('<u2').view(np.uint8).reshape(n, 2)
    cloud[:, 18:26] = np.asarray([[frame.timestamp]], dtype='<f8').view(np.uint8)
    body += cloud.tobytes()
    return bytes(body)


def write_bag(frames: list, db_path: str, topic: str, msg_type: str, hz: float = 10.0,
              start_ns: int = 1_700_000_000_000_000_000, frame_id: str = 'lidar') -> str:
    """Записать кадры в `.db3`. Время отсчитывается от фиксированного начала.

    Одинаковый вход даёт побайтово одинаковый файл: теме всегда id=1, таймстемпы --
    ровно `start_ns + i * step_ns`, база каждый раз создаётся с нуля.
    """
    if os.path.exists(db_path):
        os.remove(db_path)
    conn = sqlite3.connect(db_path)
    conn.executescript(
        'CREATE TABLE topics (id INTEGER PRIMARY KEY, name TEXT, type TEXT, '
        'serialization_format TEXT, offered_qos_profiles TEXT);'
        'CREATE TABLE messages (id INTEGER PRIMARY KEY, topic_id INTEGER, '
        'timestamp INTEGER, data BLOB);')
    conn.execute('INSERT INTO topics VALUES (1, ?, ?, ?, ?)', (topic, msg_type, 'cdr', ''))
    step_ns = int(round(1e9 / hz))
    for i, f in enumerate(frames):
        ts = int(start_ns) + i * step_ns
        conn.execute('INSERT INTO messages (topic_id, timestamp, data) VALUES (1, ?, ?)',
                     (ts, sqlite3.Binary(encode_pointcloud2_cdr(f, frame_id))))
    conn.commit()
    conn.close()
    return db_path