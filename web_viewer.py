"""Веб-вьюер облака точек на three.js: stdlib-сервер, без новых зависимостей.

Почему так: камера в Open3D (`look_at` + `setup_camera` + ROTATE_CAMERA) ведёт себя
непредсказуемо, а прозрачные кнопки, цветные плашки легенды и русский текст там
недостижимы (атлас шрифта без кириллицы, `gui.Label` не рисует background_color).
Браузер закрывает всё это сразу: OrbitControls, HTML/CSS-оверлей, нормальные шрифты
и точная передача цветов без тонмаппинга Filament.

Сервер отдаёт:
  GET /                        страница
  GET /app.js, /three.module.min.js, /OrbitControls.js   статика из web/
  GET /meta                    JSON: кадры, замеры пола/стен/оси, рельсы, легенда
  GET /track?bag=&frame=       JSON 3D-геометрии пути (рельсы/шпалы/пол) из
        track_geometry.build_track; результат кэшируется по (bag, frame)
        (пересчёт 1-2.6 с/кадр), при отсутствии track_geometry -- 503
  GET /frame?idx=N&mode=zones  бинарный кадр:
        magic 'LZFR'(4) | version u8 | kind u8 | reserved u16 | n u32   = 12 байт
        затем позиции n*3 float32 (LE), затем полезная нагрузка:
          kind 0 -> цвета n*3 uint8 (rgb)
          kind 1 -> метки зон n*1 uint8 (индексы в zone_names)
Кадр считается в Python (переиспользуются проверенные bag_reader + zones), браузер
только рисует. Позиции начинаются со смещения 12, кратны 4 -- годятся для
Float32Array-вью без копирования.
"""

import argparse
import dataclasses
import hashlib
import json
import math
import os
import struct
import sys
import threading
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

import bag_reader
import zones

WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'web')
MAGIC = b'LZFR'
VERSION = 1
KIND_RGB8 = 0
KIND_LABEL8 = 1
HEADER = '<4sBBHI'

# Ключи контракта build_track, которые ждёт страница (web/track3d.js).
TRACK_KEYS = ('frame', 'axis', 'floor', 'rails', 'sleepers', 'floor_mesh', 'meta')
# Сколько кадров геометрии держать в памяти: JSON кадра ~1 МБ, машина в притирку.
TRACK_CACHE_FRAMES = 5


def jsonable(obj):
    """Привести результат build_track к строго валидному JSON.

    Нужно потому, что build_track может вернуть numpy-скаляры/массивы (json их не
    знает) и NaN/Inf (json пишет их как `NaN`/`Infinity`, а `JSON.parse` в браузере
    на этом падает). NaN/Inf заменяются на null.
    """
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        return value if math.isfinite(value) else None
    return obj


class TrackGeometry:
    """3D-геометрия пути из `track_geometry.build_track` -- ленивый импорт и кэш.

    build_track считается 1.1-2.6 с на кадр и держит numpy-массивы, поэтому:
      * результат кэшируется по (bag, frame) уже готовым JSON-байтом -- протяжка
        слайдера не пересчитывает один и тот же кадр и не блокирует сервер;
      * сборка идёт по одной за раз (build_lock): на 2 ГБ свободной памяти два
        параллельных пересчёта могут не ужиться;
      * попадание в кэш не ждёт чужую сборку -- отдельный cache_lock.

    Модуль импортируется при первом запросе: он может появиться уже после старта
    сервера. Если его нет -- ImportError, обработчик отдаёт 503 с понятным текстом.
    """

    def __init__(self, bag_name, cache_frames=TRACK_CACHE_FRAMES):
        self.bag_name = bag_name
        self.max_cache = max(1, int(cache_frames))
        self._cache = OrderedDict()          # (bag, frame) -> bytes JSON
        self._cache_lock = threading.Lock()
        self._build_lock = threading.Lock()
        self._build = None
        self._error = None

    def resolve(self):
        """Ленивый импорт build_track; ошибка запоминается, чтобы не искать снова."""
        if self._build is not None:
            return self._build
        if self._error is not None:
            raise ImportError(self._error)
        try:
            from track_geometry import build_track
        except ImportError as exc:
            self._error = (f'track_geometry недоступен: {exc}. '
                           f'Меши пути не строятся, облако точек работает.')
            raise ImportError(self._error) from exc
        self._build = build_track
        return build_track

    def _cached(self, key):
        with self._cache_lock:
            body = self._cache.get(key)
            if body is not None:
                self._cache.move_to_end(key)
            return body

    def _store(self, key, body):
        with self._cache_lock:
            self._cache[key] = body
            self._cache.move_to_end(key)
            while len(self._cache) > self.max_cache:
                self._cache.popitem(last=False)     # вытесняем самый старый кадр

    def json_bytes(self, bag, frame, db_path):
        key = (str(bag), int(frame))
        body = self._cached(key)
        if body is not None:
            return body
        build = self.resolve()                      # ImportError -- до лока
        with self._build_lock:
            body = self._cached(key)                # пока ждали лок, мог посчитать сосед
            if body is not None:
                return body
            data = jsonable(build(db_path, key[1]))
            missing = [k for k in TRACK_KEYS if k not in (data or {})]
            if missing:
                print(f'  ! build_track вернул не все ключи контракта: нет {missing}',
                      flush=True)
            body = json.dumps(data, ensure_ascii=False).encode('utf-8')
            self._store(key, body)
            return body

    def status(self):
        try:
            self.resolve()
        except ImportError as exc:
            return {'available': False, 'error': str(exc)}
        return {'available': True, 'error': None}


class Viewer:
    """Источник кадров и метаданных для браузера."""

    def __init__(self, db_path, args):
        self.db_path = db_path
        self.bag_name = os.path.basename(os.path.dirname(os.path.abspath(db_path)))
        self.track = TrackGeometry(self.bag_name)
        self.frames = bag_reader.BagFrames(db_path)
        self.total = len(self.frames)
        if self.total == 0:
            raise SystemExit('No frames in bag')
        self.max_dist = float(args.max_dist)
        # Геометрия (коридор, ось, профиль рельсов) считается по своей дальности:
        # она раньше совпадала с передаваемой, но видимый диапазон задаёт ползунок
        # в клиенте, и если обрезать данные по нему, ползунок выше 40 м бесполезен
        # (замер: 44 м и 190 м давали одну и ту же картинку).
        self.analysis_dist = float(args.analysis_dist)
        self.behind = float(args.behind)
        self.mode = args.mode
        self.max_points = int(args.max_points)
        self.point_size = float(args.point_size)
        self.zone_params = zones.ZoneParams()
        self.rail_clearance = float(args.rail_clearance)
        self.obstacle_points = int(args.obstacle_points)
        self.obstacle_height = float(args.obstacle_height)
        if args.gauge_mm:
            self.zone_params.gauge = float(args.gauge_mm) / 1000.0
        self._lock = threading.Lock()
        self._measure()

    # ---------------------------------------------------------------- measure

    def _measure(self, samples=6):
        step = max(1, self.total // samples)
        parts = []
        for i in range(0, self.total, step):
            xyz = self.frames[i].xyz
            if len(xyz):
                parts.append(np.asarray(xyz[:, :3], dtype=np.float64))
        cloud = np.concatenate(parts) if parts else np.zeros((0, 3))
        self._zone_cloud = np.ascontiguousarray(cloud[::3])

        self.walls = tuple(zones.detect_walls(cloud, self.zone_params)) if len(cloud) else ()
        if len(self.walls) >= 2:
            self.zone_params.wall_x = self.walls

        if self.args_axis is not None:
            self.zone_params.axis_x = float(self.args_axis)
        elif len(cloud):
            self.zone_params.axis_x = zones.find_axis(
                self._zone_cloud, self.zone_params, -self.analysis_dist, self.behind,
                clearance=self.rail_clearance, min_points=self.obstacle_points)

        # Пол -- только по полосе вокруг оси пути (основания стен иначе тянут
        # нижнюю огибающую вниз и плоскость выходит кривой).
        a, b, rms = (zones.fit_floor(cloud, x_center=self.zone_params.axis_x, x_half=2.0)
                     if len(cloud) else (float('nan'), 0.0, 0.0))
        if np.isfinite(a) and np.isfinite(b):
            self.zone_params.floor_a = float(a)
            self.zone_params.floor_b = float(b)
        self.floor_rms = float(rms)

        if self.args_axis is None and len(cloud):
            self.zone_params.axis_x = zones.find_axis(
                self._zone_cloud, self.zone_params, -self.analysis_dist, self.behind,
                clearance=self.rail_clearance, min_points=self.obstacle_points)
        self._refresh_rail_geometry()

    args_axis = None

    def _refresh_rail_geometry(self):
        ys, blocked = zones.corridor_blocked(
            self._zone_cloud, self.zone_params, -self.analysis_dist, self.behind,
            bin_width=1.0, clearance=self.rail_clearance,
            min_points=self.obstacle_points, min_height=self.obstacle_height)
        self.rail_y = ys
        self.rail_blocked = blocked
        self.rail_z = zones.floor_profile(ys, self.zone_params) + zones.RAIL_HEIGHT
        self.blocked_spans = zones.blocked_spans(ys, blocked, 1.0)

    # ------------------------------------------------------------------ frame

    def _prepare(self, idx, mode):
        frame = self.frames[idx]
        xyz, inten, ring = bag_reader.trim(
            frame.xyz, frame.intensity, frame.ring,
            max_dist=self.max_dist, behind=self.behind)
        if 0 < self.max_points < len(xyz):
            step = -(-len(xyz) // self.max_points)
            xyz = xyz[::step]
            inten = None if inten is None else inten[::step]
            ring = None if ring is None else ring[::step]
        return xyz, inten, ring, mode

    def frame_bytes(self, idx, mode):
        # Без общего лока: запросы идут параллельно (предзагрузка + воспроизведение),
        # а состояние после _measure только читается.
        idx = max(0, min(int(idx), self.total - 1))
        xyz, inten, ring, mode = self._prepare(idx, mode)
        positions = np.ascontiguousarray(xyz, dtype='<f4').tobytes()
        if mode == 'zones':
            labels = zones.classify(xyz, self.zone_params)
            payload = labels.astype(np.uint8).tobytes()
            kind = KIND_LABEL8
        else:
            colors = bag_reader.frame_colors(
                xyz, inten, mode=mode, max_dist=self.analysis_dist, ring=ring)
            rgb = np.clip(np.asarray(colors) * 255.0, 0, 255).astype(np.uint8)
            payload = np.ascontiguousarray(rgb).tobytes()
            kind = KIND_RGB8
        header = struct.pack(HEADER, MAGIC, VERSION, kind, 0, len(xyz))
        return header + positions + payload

    def _rails_json(self):
        return {
            'x': [round(float(x), 3) for x, _h in zones.rail_x_positions(self.zone_params)],
            'heights': [round(float(h), 3) for _x, h in zones.rail_x_positions(self.zone_params)],
            'y': [round(float(v), 3) for v in self.rail_y],
            'z': [round(float(v), 3) for v in self.rail_z],
            'blocked': [int(bool(v)) for v in self.rail_blocked],
        }

    def _floor_json(self):
        return {
            'a': round(float(self.zone_params.floor_a), 4),
            'b': round(float(self.zone_params.floor_b), 6),
            'rms': round(self.floor_rms, 3),
            'grade_pct': round(float(self.zone_params.floor_b) * 100.0, 2),
        }

    def _params_json(self):
        p = self.zone_params
        return {
            'axis_x': round(float(p.axis_x), 2),
            'gauge_mm': int(round(p.gauge * 1000)),
            'contact_rail': bool(p.contact_rail),
            'contact_offset': round(float(p.contact_offset), 3),
            'contact_side': int(p.contact_side),
            'rail_height': round(float(zones.RAIL_HEIGHT), 3),
            'floor': self._floor_json(),
            'rails': self._rails_json(),
            'blocked_spans': [[round(float(a), 1), round(float(b), 1)]
                              for a, b in self.blocked_spans],
        }

    def set_params(self, axis=None, gauge_mm=None, contact=None, offset=None,
                   side=None):
        """Живая правка модели рельсов из браузера.

        Пол пересчитывается по полосе вокруг новой оси: он от неё зависит
        (по всему кадру нижнюю огибающую тянут вниз основания стен).
        """
        p = dataclasses.replace(self.zone_params)
        if axis is not None:
            p.axis_x = float(axis)
        if gauge_mm:
            p.gauge = float(gauge_mm) / 1000.0
        if contact is not None:
            p.contact_rail = bool(contact)
        if offset is not None:
            p.contact_offset = max(0.05, float(offset))
        if side is not None:
            p.contact_side = 1 if int(side) >= 0 else -1
        if self._zone_cloud is not None and len(self._zone_cloud):
            a, b, rms = zones.fit_floor(self._zone_cloud, x_center=p.axis_x,
                                        x_half=2.0)
            if np.isfinite(a) and np.isfinite(b):
                p.floor_a = float(a)
                p.floor_b = float(b)
                self.floor_rms = float(rms)
        self.zone_params = p          # подмена целиком: запросы кадров читают атомарно
        self._refresh_rail_geometry()
        return self._params_json()

    def meta(self):
        span = (self.frames.timestamps[-1] - self.frames.timestamps[0]) / 1e9
        zone_colors = [[int(round(c * 255)) for c in zones.ZONE_COLORS[n]]
                       for n in zones.ZONE_NAMES]
        blocked = [int(bool(v)) for v in self.rail_blocked]
        return {
            'bag': self.bag_name,
            'frames': self.total,
            'duration_s': round(float(span), 2),
            'hz': round(float(self.total / span), 2) if span else 0.0,
            'max_dist': self.max_dist,
            'behind': self.behind,
            'default_mode': self.mode,
            'modes': list(bag_reader.COLOR_MODES),
            'point_size': self.point_size,
            'max_points_hint': self.max_points if self.max_points > 0 else 1000000,
            'floor': self._floor_json(),
            'walls': [round(float(w), 2) for w in self.walls],
            'axis_x': round(float(self.zone_params.axis_x), 2),
            'gauge_mm': int(round(self.zone_params.gauge * 1000)),
            'rails': self._rails_json(),
            'blocked_spans': [[round(float(a), 1), round(float(b), 1)]
                              for a, b in self.blocked_spans],
            'grid': {'spacing': 5.0, 'half_width': 40.0,
                     'length': round(self.max_dist + self.behind, 1), 'z': 0.0},
            'zone_names': list(zones.ZONE_NAMES),
            'zone_colors': zone_colors,
            'blocked_color': [int(round(c * 255)) for c in zones.BLOCKED_COLOR],
            'rail_color': [int(round(c * 255)) for c in zones.ZONE_COLORS['rail']],
            'train_top': float(zones.TRAIN_TOP),
            'obstacle_points': self.obstacle_points,
            'rail_clearance': self.rail_clearance,
        }


WEB_ASSETS = ('index.html', 'app.js', 'track3d.js', 'OrbitControls.js',
              'three.module.min.js')

# Собранный React-клиент (app/client/dist) раздаёт тот же Python-сервер: страница
# и API живут на одном origin, поэтому второй процесс и CORS не нужны. Если
# каталога нет — отдаём старый клиент из web/ (он по-прежнему доступен на /legacy).
CLIENT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'app', 'client', 'dist')

ASSET_TYPES = {
    '.html': 'text/html; charset=utf-8',
    '.js': 'application/javascript; charset=utf-8',
    '.mjs': 'application/javascript; charset=utf-8',
    '.css': 'text/css; charset=utf-8',
    '.json': 'application/json; charset=utf-8',
    '.map': 'application/json; charset=utf-8',
    '.svg': 'image/svg+xml',
    '.png': 'image/png',
    '.ico': 'image/x-icon',
    '.woff2': 'font/woff2',
}


def assets_version():
    """Хэш mtime+размера файлов фронтенда: страница опрашивает его и сама
    перезагружается, когда я правлю HTML/JS (автообновление вместо F5)."""
    parts = []
    for name in WEB_ASSETS:
        try:
            st = os.stat(os.path.join(WEB_DIR, name))
            parts.append(f'{name}:{st.st_mtime_ns}:{st.st_size}')
        except OSError:
            parts.append(f'{name}:missing')
    return hashlib.sha1('|'.join(parts).encode()).hexdigest()[:16]


def make_handler(viewer, client_dir=CLIENT_DIR):
    class Handler(BaseHTTPRequestHandler):
        server_version = 'lidar-web'

        def log_message(self, *args):
            pass

        def _send(self, body, ctype, status=200):
            self.send_response(status)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _error_json(self, message):
            """Тело ошибки для /track: страница показывает текст как есть."""
            return json.dumps({'error': message}, ensure_ascii=False).encode('utf-8')

        def _static(self, name, ctype):
            path = os.path.join(WEB_DIR, name)
            if not os.path.exists(path):
                self._send(b'not found', 'text/plain; charset=utf-8', 404)
                return
            with open(path, 'rb') as fh:
                self._send(fh.read(), ctype)

        def _client(self, rel):
            """Отдать файл собранного React-клиента. None, если файла нет.

            Путь проверяем на выход за каталог клиента: без этого /assets/../..
            читало бы что угодно с диска.
            """
            root = os.path.normpath(client_dir)
            full = os.path.normpath(os.path.join(root, rel.lstrip('/')))
            if not full.startswith(root + os.sep) or not os.path.isfile(full):
                return None
            ctype = ASSET_TYPES.get(os.path.splitext(full)[1].lower(),
                                    'application/octet-stream')
            with open(full, 'rb') as fh:
                self._send(fh.read(), ctype)
            return True

        def do_GET(self):
            url = urlparse(self.path)
            path = url.path
            if path in ('/', '/index.html'):
                if self._client('index.html'):
                    return
                return self._static('index.html', 'text/html; charset=utf-8')
            if path == '/legacy':
                return self._static('index.html', 'text/html; charset=utf-8')
            if path.startswith('/assets/'):
                if self._client(path):
                    return
                return self._send(b'not found', 'text/plain; charset=utf-8', 404)
            if path == '/app.js':
                return self._static('app.js', 'application/javascript; charset=utf-8')
            if path == '/track3d.js':
                return self._static('track3d.js',
                                    'application/javascript; charset=utf-8')
            if path == '/OrbitControls.js':
                return self._static('OrbitControls.js',
                                    'application/javascript; charset=utf-8')
            if path == '/three.module.min.js':
                return self._static('three.module.min.js',
                                    'application/javascript; charset=utf-8')
            if path == '/set':
                q = parse_qs(url.query)

                def num(key, cast=float):
                    v = q.get(key, [None])[0]
                    return None if v in (None, '') else cast(v)

                try:
                    body = json.dumps(viewer.set_params(
                        axis=num('axis'), gauge_mm=num('gauge', int),
                        contact=num('contact', int), offset=num('offset'),
                        side=num('side', int))).encode('utf-8')
                    return self._send(body, 'application/json; charset=utf-8')
                except Exception as exc:  # noqa: BLE001
                    return self._send(json.dumps({'error': str(exc)}).encode('utf-8'),
                                      'application/json; charset=utf-8', 400)
            if path == '/version':
                body = json.dumps({'version': assets_version()}).encode('utf-8')
                return self._send(body, 'application/json; charset=utf-8')
            if path == '/meta':
                body = json.dumps(viewer.meta()).encode('utf-8')
                return self._send(body, 'application/json; charset=utf-8')
            if path == '/track':
                q = parse_qs(url.query)
                bag = q.get('bag', [viewer.bag_name])[0] or viewer.bag_name
                if bag != viewer.bag_name:
                    return self._send(self._error_json(
                        f'unknown bag {bag!r}: этот сервер отдаёт только '
                        f'{viewer.bag_name!r}'), 'application/json; charset=utf-8', 404)
                try:
                    frame = int(q.get('frame', ['0'])[0])
                except ValueError:
                    frame = 0
                frame = max(0, min(frame, viewer.total - 1))
                try:
                    body = viewer.track.json_bytes(bag, frame, viewer.db_path)
                except ImportError as exc:        # модуль не найден -- фича, не падение
                    return self._send(self._error_json(str(exc)),
                                      'application/json; charset=utf-8', 503)
                except Exception as exc:  # noqa: BLE001
                    return self._send(self._error_json(f'build_track: {exc!r}'),
                                      'application/json; charset=utf-8', 500)
                return self._send(body, 'application/json; charset=utf-8')
            if path == '/frame':
                q = parse_qs(url.query)
                try:
                    idx = int(q.get('idx', ['0'])[0])
                except ValueError:
                    idx = 0
                mode = q.get('mode', [viewer.mode])[0]
                if mode not in bag_reader.COLOR_MODES:
                    mode = viewer.mode
                return self._send(viewer.frame_bytes(idx, mode),
                                  'application/octet-stream')
            if path == '/favicon.ico':
                return self._send(b'', 'image/x-icon', 204)
            return self._send(b'not found', 'text/plain; charset=utf-8', 404)

    return Handler


def main():
    parser = argparse.ArgumentParser(
        description='three.js web viewer for rosbag2 .db3 point clouds')
    parser.add_argument('bag_dir', nargs='?', default='for_hackathon/doubleT_obstacle',
                        help='bag folder')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--mode', default='zones', help='initial coloring')
    parser.add_argument('--max-dist', type=float, default=200.0,
                        help='дальность, которую сервер отдаёт клиенту, м; '
                             'видимый диапазон внутри неё задаёт ползунок '
                             '"Дальность, м" в панели (5..200)')
    parser.add_argument('--analysis-dist', type=float, default=40.0,
                        help='дальность для геометрии: коридор, ось, профиль '
                             'рельсов, засорённость (по умолчанию 40)')
    parser.add_argument('--behind', type=float, default=5.0)
    parser.add_argument('--max-points', type=int, default=400000,
                        help='cap on points per frame (payload size)')
    parser.add_argument('--point-size', type=float, default=3.0)
    parser.add_argument('--axis-x', type=float, default=None)
    parser.add_argument('--gauge-mm', type=float, default=0.0)
    parser.add_argument('--obstacle-points', type=int, default=20)
    parser.add_argument('--obstacle-height', type=float,
                        default=zones.MIN_OBSTACLE_HEIGHT,
                        help='минимальная высота структуры в бине, м; '
                             '0 -- считать занятость по одним точкам')
    parser.add_argument('--rail-clearance', type=float, default=0.30)
    parser.add_argument('--client-dir', default=CLIENT_DIR,
                        help='каталог собранного React-клиента (app/client/dist); '
                             'если его нет -- отдаётся старый клиент из web/')
    parser.add_argument('--open', action='store_true',
                        help='open the page in the default browser')
    args = parser.parse_args()

    db_path = bag_reader.find_db3(args.bag_dir)
    print(f'Reading: {db_path}')

    Viewer.args_axis = args.axis_x
    viewer = Viewer(db_path, args)

    url = f'http://{args.host}:{args.port}/'
    print(f'ZONES floor : z = {viewer.zone_params.floor_a:.4f} + '
          f'{viewer.zone_params.floor_b:.6f}*y  (rms {viewer.floor_rms:.3f} m, '
          f'grade {viewer.zone_params.floor_b * 100:.2f} %)')
    print(f'ZONES walls : x = {[round(w, 2) for w in viewer.walls]}')
    print(f'ZONES axis  : x = {viewer.zone_params.axis_x:.2f} m, '
          f'gauge = {viewer.zone_params.gauge * 1000:.0f} mm '
          f'(rails at x = {viewer.zone_params.axis_x - viewer.zone_params.gauge / 2:.2f} '
          f'and {viewer.zone_params.axis_x + viewer.zone_params.gauge / 2:.2f})')
    print(f'RAILS blocked (y, m): {[[round(a, 1), round(b, 1)] for a, b in viewer.blocked_spans]}'
          f'  total {sum(b - a for a, b in viewer.blocked_spans):.1f} m of '
          f'{viewer.max_dist + viewer.behind:.0f} m')
    track_status = viewer.track.status()
    print('TRACK       : ' + (
        f'track_geometry найден, /track отдаёт геометрию пути '
        f'(кэш {viewer.track.max_cache} кадров)'
        if track_status['available']
        else f'track_geometry недоступен ({track_status["error"]}) -- /track ответит 503'))
    has_client = os.path.isfile(os.path.join(args.client_dir, 'index.html'))
    print(f'CLIENT      : {"React (app/client/dist)" if has_client else "старый web/"}'
          f'{"" if has_client else "  -- собери: cd app/client && bun run build"}')
    print(f'Serving {viewer.total} frames at {url}   (Ctrl+C to stop)')

    if args.open:
        import webbrowser
        webbrowser.open(url)

    httpd = ThreadingHTTPServer((args.host, args.port),
                                make_handler(viewer, args.client_dir))
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print('\nStopped.')
    finally:
        viewer.frames.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())