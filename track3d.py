"""Отдельный вьюер 3D-объектов пути (рельсы / шпалы / пол) плюс облако точек.

Зачем отдельный сервер: `web_viewer.py` и `web/*` -- файлы параллельного агента,
их нельзя ни менять, ни копировать. Здесь свой stdlib-сервер и своя страница
(`web3d/`), а из `web/` раздаётся только three.js r160 и OrbitControls (read-only).

Геометрию пути считает `track_geometry.build_track(db_path, frame_index)` (модуль
параллельного агента). Сервер её не пересчитывает: отдаёт как есть, но приводит
к строго валидному JSON (numpy-скаляры/массивы -> числа и списки, NaN/Inf -> null).

Оси данных: X поперёк, Y вдоль (вперёд -- -Y), Z вверх.

Маршруты:
  GET /                       web3d/index.html
  GET /app.js                 web3d/app.js
  GET /three.module.min.js    web/three.module.min.js   (только чтение)
  GET /OrbitControls.js       web/OrbitControls.js      (только чтение)
  GET /meta                   каталог bag'ов, число кадров, доступность track_geometry
  GET /track?bag=&frame=      JSON из build_track, без правок
  GET /points?bag=&frame=     облако точек кадра (странице опционально), бинарь:
        magic 'T3DP'(4) | version u8 | kind u8 | reserved u16 | n u32  = 12 байт
        далее позиции n*3 float32 LE, далее цвета n*3 uint8
      Смещение 12 кратно 4, поэтому позиции читаются Float32Array-вью без копии.
"""

import argparse
import json
import math
import os
import sqlite3
import struct
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

import bag_reader


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(ROOT_DIR, 'web')       # three.js параллельного агента
PAGE_DIR = os.path.join(ROOT_DIR, 'web3d')    # моя страница

MAGIC = b'T3DP'
VERSION = 1
KIND_RGB8 = 0
POINT_HEADER = '<4sBBHI'
POINT_MODES = ('height', 'intensity', 'range', 'ring')
POINT_MAX_POINTS = 250000
POINT_MAX_DIST = 40.0
POINT_BEHIND = 5.0

# Ключи, которые страница ждёт по контракту build_track.
CONTRACT_KEYS = ('frame', 'axis', 'floor', 'rails', 'sleepers', 'floor_mesh', 'meta')


# --------------------------------------------------------------------- helpers

def jsonable(obj):
    """Привести результат build_track к строго валидному JSON.

    Нужно, потому что модуль пути может вернуть numpy-скаляры/массивы (json их не
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


def count_frames(db_path):
    """Число кадров PointCloud2 в .db3. sqlite COUNT -- без выгрузки rowid в память."""
    conn = sqlite3.connect(db_path)
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
                f'SELECT COUNT(*) FROM messages WHERE topic_id IN ({placeholders})',
                topic_ids)
        else:
            cur.execute('SELECT COUNT(*) FROM messages')
        return int(cur.fetchone()[0])
    finally:
        conn.close()


class Catalog:
    """Bag'ы в --root: имя -> путь к .db3 и число кадров (считается один раз)."""

    def __init__(self, root):
        self.root = root
        self.names = []          # в порядке сортировки
        self.paths = {}
        self._frames = {}
        if os.path.isdir(root):
            for name in sorted(os.listdir(root)):
                bag_dir = os.path.join(root, name)
                if not os.path.isdir(bag_dir):
                    continue
                try:
                    self.paths[name] = bag_reader.find_db3(bag_dir)
                except (FileNotFoundError, OSError):
                    continue
                self.names.append(name)

    def has(self, name):
        return name in self.paths

    def frames(self, name):
        if name not in self._frames:
            self._frames[name] = count_frames(self.paths[name])
        return self._frames[name]

    def as_json(self):
        return [{'name': n, 'db': os.path.basename(self.paths[n]),
                 'frames': self.frames(n)} for n in self.names]


class BagStore:
    """Ровно один открытый bag за раз: память машины в притирку (~2 ГБ).

    Кэш кадров внутри BagFrames ограничен (cache_size), предзагрузка выключена --
    облако нужно только для запрошенного кадра.
    """

    def __init__(self, cache_frames=2):
        self.cache_frames = int(cache_frames)
        self._name = None
        self._frames = None

    def frames_for(self, name, db_path):
        if self._name == name and self._frames is not None:
            return self._frames
        self.close()
        self._frames = bag_reader.BagFrames(
            db_path, cache_size=self.cache_frames, prefetch_ahead=0)
        self._name = name
        return self._frames

    def close(self):
        if self._frames is not None:
            try:
                self._frames.close()
            except Exception:  # noqa: BLE001 -- закрытие не должно ронять сервер
                pass
        self._frames = None
        self._name = None


def resolve_bag(root, name):
    """`name` -- имя bag'а в root или путь к каталогу/.db3."""
    if name:
        candidate = os.path.join(root, name)
        if os.path.isdir(candidate):
            return bag_reader.find_db3(candidate)
        if os.path.isfile(candidate) and candidate.endswith('.db3'):
            return candidate
        if os.path.isfile(name) and name.endswith('.db3'):
            return name
    raise FileNotFoundError(f'bag {name!r} not found in {root!r}')


class App:
    """Состояние сервера: каталог bag'ов, одно облако, ленивый импорт track_geometry."""

    def __init__(self, root, default_bag):
        self.catalog = Catalog(root)
        self.bags = BagStore()
        self.default_bag = default_bag if self.catalog.has(default_bag) else (
            self.catalog.names[0] if self.catalog.names else default_bag)
        # build_track тяжёлый и держит numpy-массивы: считаем по одному запросу за раз,
        # иначе на 2 ГБ памяти два параллельных запроса могут не ужиться.
        self.lock = threading.Lock()
        self._build_track = None
        self._module_error = None

    # ------------------------------------------------------------- track module

    def build_track(self):
        """Ленивый импорт: модуль может появиться уже после старта сервера."""
        if self._build_track is not None:
            return self._build_track
        if self._module_error is not None:
            raise ImportError(self._module_error)
        try:
            from track_geometry import build_track
        except ImportError as exc:
            self._module_error = (f'track_geometry недоступен: {exc}. '
                                  f'Меши пути не строятся, облако точек работает.')
            raise ImportError(self._module_error) from exc
        self._build_track = build_track
        return build_track

    def module_status(self):
        try:
            self.build_track()
        except ImportError as exc:
            return {'available': False, 'error': str(exc)}
        return {'available': True, 'error': None}

    # ------------------------------------------------------------------ data

    def track(self, bag, frame):
        if not self.catalog.has(bag):
            raise KeyError(f'unknown bag {bag!r}')
        frames = max(1, self.catalog.frames(bag))
        index = max(0, min(int(frame), frames - 1))
        db_path = self.catalog.paths[bag]
        build_track = self.build_track()
        with self.lock:
            data = build_track(db_path, index)
        data = jsonable(data)
        missing = [k for k in CONTRACT_KEYS if k not in (data or {})]
        if missing:
            print(f'  ! build_track вернул не все ключи контракта: нет {missing}',
                  flush=True)
        return data

    def points(self, bag, frame, mode):
        if not self.catalog.has(bag):
            raise KeyError(f'unknown bag {bag!r}')
        if mode not in POINT_MODES:
            mode = POINT_MODES[0]
        frames = max(1, self.catalog.frames(bag))
        index = max(0, min(int(frame), frames - 1))
        with self.lock:
            frames_obj = self.bags.frames_for(bag, self.catalog.paths[bag])
            raw = frames_obj[index]
            xyz, intensity, ring = bag_reader.trim(
                raw.xyz, raw.intensity, raw.ring,
                max_dist=POINT_MAX_DIST, behind=POINT_BEHIND)
            if 0 < POINT_MAX_POINTS < len(xyz):
                step = -(-len(xyz) // POINT_MAX_POINTS)
                xyz = xyz[::step]
                intensity = None if intensity is None else intensity[::step]
                ring = None if ring is None else ring[::step]
            colors = bag_reader.frame_colors(
                xyz, intensity, mode=mode, max_dist=POINT_MAX_DIST, ring=ring)
            rgb = np.clip(np.asarray(colors) * 255.0, 0, 255).astype(np.uint8)
            positions = np.ascontiguousarray(xyz, dtype='<f4').tobytes()
            payload = np.ascontiguousarray(rgb).tobytes()
        header = struct.pack(POINT_HEADER, MAGIC, VERSION, KIND_RGB8, 0, len(xyz))
        return header + positions + payload

    def close(self):
        self.bags.close()

    def meta(self):
        return {
            'bags': self.catalog.as_json(),
            'default_bag': self.default_bag,
            'default_frame': 0,
            'track_geometry': self.module_status(),
            'contract_keys': list(CONTRACT_KEYS),
            'point_modes': list(POINT_MODES),
            'points': {'max_dist': POINT_MAX_DIST, 'behind': POINT_BEHIND,
                       'max_points': POINT_MAX_POINTS},
            'clean_background': 0xe9ecf1,
            'units': 'метры; X поперёк, Y вдоль (вперёд -Y), Z вверх',
        }


# ---------------------------------------------------------------------- server

CTYPES = {
    'html': 'text/html; charset=utf-8',
    'js': 'application/javascript; charset=utf-8',
    'json': 'application/json; charset=utf-8',
    'bin': 'application/octet-stream',
    'text': 'text/plain; charset=utf-8',
}


def make_handler(app):
    class Handler(BaseHTTPRequestHandler):
        server_version = 'track3d'

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

        def _json(self, obj, status=200):
            self._send(json.dumps(obj, ensure_ascii=False).encode('utf-8'),
                       CTYPES['json'], status)

        def _error(self, status, message):
            self._json({'error': message}, status)

        def _file(self, path, kind, missing='not found'):
            if not os.path.isfile(path):
                return self._error(404, f'{missing}: {path}')
            with open(path, 'rb') as fh:
                self._send(fh.read(), CTYPES[kind])

        def _query(self):
            return parse_qs(urlparse(self.path).query)

        def _bag_frame(self, query):
            bag = query.get('bag', [app.default_bag])[0] or app.default_bag
            try:
                frame = int(query.get('frame', ['0'])[0])
            except ValueError:
                frame = 0
            return bag, frame

        def do_GET(self):
            path = urlparse(self.path).path
            try:
                if path in ('/', '/index.html'):
                    return self._file(os.path.join(PAGE_DIR, 'index.html'), 'html')
                if path == '/app.js':
                    return self._file(os.path.join(PAGE_DIR, 'app.js'), 'js')
                if path == '/three.module.min.js':
                    return self._file(os.path.join(WEB_DIR, 'three.module.min.js'), 'js')
                if path == '/OrbitControls.js':
                    return self._file(os.path.join(WEB_DIR, 'OrbitControls.js'), 'js')
                if path == '/favicon.ico':
                    return self._send(b'', 'image/x-icon', 204)
                if path == '/meta':
                    return self._json(app.meta())
                if path == '/track':
                    query = self._query()
                    bag, frame = self._bag_frame(query)
                    if not app.catalog.has(bag):
                        return self._error(404, f'unknown bag {bag!r}')
                    try:
                        data = app.track(bag, frame)
                    except ImportError as exc:
                        return self._error(503, str(exc))
                    except (KeyError, FileNotFoundError) as exc:
                        return self._error(404, str(exc))
                    except Exception as exc:  # noqa: BLE001
                        return self._error(500, f'build_track: {exc!r}')
                    return self._json(data)
                if path == '/points':
                    query = self._query()
                    bag, frame = self._bag_frame(query)
                    if not app.catalog.has(bag):
                        return self._error(404, f'unknown bag {bag!r}')
                    mode = query.get('mode', [POINT_MODES[0]])[0]
                    try:
                        body = app.points(bag, frame, mode)
                    except Exception as exc:  # noqa: BLE001
                        return self._error(500, f'points: {exc!r}')
                    return self._send(body, CTYPES['bin'])
                return self._error(404, f'unknown route {path!r}')
            except Exception as exc:  # noqa: BLE001
                try:
                    self._error(500, f'{exc!r}')
                except Exception:  # noqa: BLE001
                    pass

    return Handler


def main():
    parser = argparse.ArgumentParser(
        description='3D-вьюер объектов пути (рельсы/шпалы/пол) + облако точек')
    parser.add_argument('--root', default=os.path.join(ROOT_DIR, 'for_hackathon'),
                        help='каталог с bag-папками')
    parser.add_argument('--bag', default='doubleT_platform',
                        help='bag по умолчанию (имя папки в --root или путь к .db3)')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8770)
    parser.add_argument('--cache-frames', type=int, default=2,
                        help='сколько кадров облака держать в памяти')
    args = parser.parse_args()

    app = App(args.root, args.bag)
    if not app.catalog.names:
        print(f'В {args.root} нет ни одного .db3 -- облако точек будет недоступно.',
              flush=True)

    status = app.module_status()
    url = f'http://{args.host}:{args.port}/'

    print('Track3D -- 3D-объекты пути (рельсы, шпалы, пол) + облако точек', flush=True)
    print(f'  bag по умолчанию : {app.default_bag}', flush=True)
    print(f'  bag-ов доступно  : {len(app.catalog.names)} '
          f'({", ".join(app.catalog.names) or "нет"})', flush=True)
    if status['available']:
        print('  track_geometry   : найден, /track отдаёт реальную геометрию пути',
              flush=True)
    else:
        print(f'  track_geometry   : НЕ найден ({status["error"]})', flush=True)
        print('                     /track ответит 503, страница покажет только облако',
              flush=True)
    print('', flush=True)
    print(f'  Открыть в браузере: {url}', flush=True)
    print('  На странице: селектор bag, слайдер кадра, тумблеры точек / рельсов / '
          'шпал / пола', flush=True)
    print('    и кнопка «Чистое поле» -- прячет облако, ставит нейтральный фон и '
          'вписывает камеру', flush=True)
    print('    по габариту объектов; повторное нажатие возвращает облако и камеру.',
          flush=True)
    print(f'  Проверка API: {url}meta  и  {url}track?bag={app.default_bag}&frame=0',
          flush=True)
    print('  Ctrl+C -- остановить.', flush=True)

    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print('\nStopped.', flush=True)
    finally:
        httpd.server_close()
        app.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())