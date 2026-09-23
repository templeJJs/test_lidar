"""Веб-вьюер облака точек на three.js: stdlib-сервер, без новых зависимостей.

Почему так: камера в Open3D (`look_at` + `setup_camera` + ROTATE_CAMERA) ведёт себя
непредсказуемо, а прозрачные кнопки, цветные плашки легенды и русский текст там
недостижимы (атлас шрифта без кириллицы, `gui.Label` не рисует background_color).
Браузер закрывает всё это сразу: OrbitControls, HTML/CSS-оверлей, нормальные шрифты
и точная передача цветов без тонмаппинга Filament.

Сервер отдаёт:
  GET /                        страница
  GET /app.js, /three.module.min.js, /OrbitControls.js   статика из web/
  GET /meta[?bag=]             JSON: кадры, замеры пола/стен/оси, рельсы, легенда,
        линия хода (блок `path`: три варианта) и туннель безопасности (`tunnel`);
        в режиме корня добавляется список записей (`bags`)
  GET /bags                    диагностика каталога: живые записи (LRU), счётчики
        открытий/вытеснений и по геометрии пути -- сколько кадров отдано из кэша
        памяти, из хранилища precompute и сколько собрано на месте (`track`)
  GET /profile?bag=            сохранённый профиль вида этой записи (или null)
  POST /profile?bag=           сохранить профиль вида (JSON в теле) в
        web/view_profiles.json -- профили переживают перезапуск сервера
  GET /track?bag=&frame=       JSON 3D-геометрии пути (рельсы/шпалы/пол):
        сначала готовый кадр из хранилища precompute (см. TrackStore и
        precompute_track.py), иначе сборка track_geometry.build_track на месте;
        результат кэшируется по (bag, frame); при отсутствии track_geometry
        сборка невозможна -- 503 на кадр, которого нет в хранилище
  GET /frame?bag=&idx=N&mode=zones  бинарный кадр:
        magic 'LZFR'(4) | version u8 | kind u8 | reserved u16 | n u32   = 12 байт
        затем позиции n*3 float32 (LE), затем полезная нагрузка:
          kind 0 -> цвета n*3 uint8 (rgb)
          kind 1 -> метки зон n*1 uint8 (индексы в zone_names)
        В режиме zones метка rail считается по оси ЭТОГО кадра: узлы берутся из
        кэша /track, из хранилища precompute (TrackStore) или -- последним
        запасом -- сборкой на месте (allow_build, см. RailZone). В `reserved` --
        номер кадра, по оси которого посчитана метка (axis_frame): != idx только
        в запасном пути, когда геометрию взять не удалось вовсе (нет ни
        track_geometry, ни кадра в хранилище).
  GET /labels?bag=&idx=N      ТОЛЬКО метки зон кадра (kind 1) без позиций:
        тот же заголовок 12 байт + n байт uint8. Уточнение раскраски к уже
        нарисованному кадру (метка rail -- по оси своего кадра, см. RailZone); в
        `reserved` -- axis_frame, как у /frame.
  GET /set?bag=&axis=&gauge=... живые правки модели рельсов и туннеля

Позиционный аргумент -- либо папка одной записи (внутри *.db3), либо корень с
папками записей (например for_hackathon): тогда в одном процессе живут все
записи, а `?bag=<имя>` выбирает нужную (без параметра -- запись по умолчанию).
Состояние на запись тяжёлое, поэтому экземпляры поднимаются лениво, и живых
держим не больше MAX_LIVE_BAGS (LRU): при открытии новой самая давняя закрывается.

Кадр считается в Python (переиспользуются проверенные bag_reader + zones), браузер
только рисует. Позиции начинаются со смещения 12, кратны 4 -- годятся для
Float32Array-вью без копирования.
"""

import argparse
import base64
import copy
import dataclasses
import gzip
import hashlib
import json
import math
import os
import sqlite3
import struct
import sys
import threading
import time
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
# Сколько кадров геометрии держать в памяти: JSON кадра 0.47-1.04 МБ (замер по
# for_hackathon, 12 кадров -- ~10 МБ на запись; живых записей максимум 2). Было 5:
# при протяжке слайдером туда-сюда один и тот же кадр пересобирался заново.
TRACK_CACHE_FRAMES = 12
# Хранилище заранее посчитанной геометрии пути (пишет precompute_track.py):
# <root>/<запись>/<кадр>.json.gz. Путь от каталога проекта, а не от cwd: сервер
# запускают и из корня, и из подпапки. Переопределяется `--track-cache DIR`.
TRACK_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               '.cache', 'track')
# Сколько записей (датасетов) держать открытыми одновременно: экземпляр на запись
# несёт облако анализа, BagFrames со своим кэшем кадров и кэш геометрии пути.
MAX_LIVE_BAGS = 2
# Профили вида по записям: файл рядом с клиентом, в WEB_ASSETS не входит (иначе
# сохранение вида меняло бы /version и страница сама перезагружалась бы).
VIEW_PROFILES = os.path.join(WEB_DIR, 'view_profiles.json')
# Пресеты камеры, которые можно записать в профиль (совпадают с web/app.js).
PROFILE_PRESETS = ('lidar', 'track', 'profile', 'top', 'side', 'iso', 'behind', 'fit')
# Числовые поля профиля: то, что реально двигает вид и раскраску.
PROFILE_NUM_KEYS = ('axis_x', 'gauge_mm', 'max_dist', 'analysis_dist')


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


def safe_store_name(name):
    """Имя записи, годное как имя каталога в хранилище.

    Имена записей -- имена папок (`roundT_doubleT`), они же приходят из `?bag=`.
    Разделители пути вырезаем: иначе `?bag=../..` читало бы файлы вне хранилища.
    Сервер и precompute_track.py зовут эту функцию одинаково, поэтому пути
    совпадают.
    """
    name = str(name).strip() or '_'
    for ch in ('/', '\\', ':', '*', '?', '"', '<', '>', '|'):
        name = name.replace(ch, '_')
    return '_' if name in ('.', '..') else name


class TrackStore:
    """Хранилище ГОТОВОЙ геометрии пути: <root>/<запись>/<кадр>.json.gz.

    Пишет `precompute_track.py`, читает сервер -- и читает первым делом, до
    всякой сборки. Зачем: сборка кадра стоит 0.2-0.6 с, а воспроизведение просит
    10 кадров в секунду (бюджет 0.1 с); с готовым кадром ответ -- чтение gzip
    (~10 мс), и ядро больше не занято сборкой постоянно.

    Формат: gzip от РОВНО тех байт, которые вернул бы /track при сборке на месте
    (`json.dumps(jsonable(build_track(...)), ensure_ascii=False)` в utf-8), то
    есть распакованное содержимое отдаётся как есть, без повторной сериализации.
    Поэтому хранилище не меняет данные, а только способ их достать.

    Порядок кадров внутри записи важен: у build_track есть тёплая (инкрементальная)
    сборка и связь решений соседних кадров (см. track_geometry: `warm`, LINK_*),
    поэтому precompute_track.py идёт по кадрам 0, 1, 2, ... -- как воспроизведение.
    """

    def __init__(self, root=None):
        self.root = os.path.abspath(root) if root else TRACK_CACHE_DIR

    def path(self, bag, frame):
        """Путь файла кадра."""
        return os.path.join(self.root, safe_store_name(bag), f'{int(frame)}.json.gz')

    def has(self, bag, frame):
        return os.path.isfile(self.path(bag, frame))

    def get(self, bag, frame):
        """Тело кадра из хранилища или None (нет файла / файл не прочитан)."""
        path = self.path(bag, frame)
        try:
            with gzip.open(path, 'rb') as fh:
                return fh.read()
        except FileNotFoundError:
            return None
        except (OSError, EOFError, gzip.BadGzipFile) as exc:
            # Битый файл хранилища не должен ронять сервер: это промах, кадр
            # соберётся на месте (и предупреждение о неполном хранилище скажет).
            print(f'  ! хранилище геометрии: {path} не прочитан ({exc!r}) -- '
                  f'кадр будет посчитан на месте', flush=True)
            return None

    def put(self, bag, frame, body):
        """Записать кадр атомарно (.tmp + os.replace) -- для precompute_track.py.

        `mtime=0` в заголовке gzip: байты файла зависят только от содержимого, а не
        от времени сборки. Иначе gzip пишет в заголовок время, и повторный проход с
        теми же данными менял бы все файлы (сверка по sha1 бесполезна).
        """
        path = self.path(bag, frame)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + '.tmp'
        try:
            with open(tmp, 'wb') as raw:
                with gzip.GzipFile(filename='', fileobj=raw, mode='wb',
                                   compresslevel=6, mtime=0) as fh:
                    fh.write(body)
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        return path

    def coverage(self, bag):
        """Сколько кадров этой записи уже в хранилище."""
        directory = os.path.join(self.root, safe_store_name(bag))
        try:
            names = os.listdir(directory)
        except OSError:
            return 0
        return sum(1 for n in names if n.endswith('.json.gz') and n[:-8].isdigit())

    def counts(self):
        """(кадров всего, записей с кадрами) -- для лога и диагностики."""
        frames = bags = 0
        try:
            entries = sorted(os.listdir(self.root))
        except OSError:
            return 0, 0
        for name in entries:
            directory = os.path.join(self.root, name)
            if not os.path.isdir(directory):
                continue
            try:
                names = os.listdir(directory)
            except OSError:
                continue
            n = sum(1 for f in names if f.endswith('.json.gz') and f[:-8].isdigit())
            if n:
                bags += 1
                frames += n
        return frames, bags

    def size(self):
        """Сколько хранилище занимает на диске, байт."""
        total = 0
        for root, _dirs, files in os.walk(self.root):
            for name in files:
                try:
                    total += os.path.getsize(os.path.join(root, name))
                except OSError:
                    pass
        return total


# Предупреждение о промахе хранилища печатается один раз за сессию: иначе строка
# про сборку на месте забила бы лог при каждом кадре (а это норма, если
# precompute_track.py ещё не запускали).
_STORE_MISS_WARNED = False
_STORE_MISS_LOCK = threading.Lock()
# Отличить «каталог хранилища не задан -- взять стандартный» от «хранилище
# выключено» (`--track-cache none`): в обоих случаях значение None, а поведение
# разное.
_STORE_DEFAULT = object()
# Значения `--track-cache`, которыми хранилище выключают (A/B и отладка).
STORE_OFF = ('', 'none', 'off', 'нет')


def track_store(cache_dir):
    """TrackStore по значению `--track-cache`; None -- хранилище выключено.

    None (аргумент не задан) -- стандартный каталог `.cache/track`; пустая строка
    и 'none'/'off' -- выключено: сервер собирает геометрию на месте, как раньше.
    """
    if cache_dir is None:
        return TrackStore()
    if str(cache_dir).strip().lower() in STORE_OFF:
        return None
    return TrackStore(str(cache_dir))


class TrackGeometry:
    """3D-геометрия пути из `track_geometry.build_track` -- ленивый импорт и кэш.

    Порядок источников кадра: кэш в памяти (12 кадров) -> хранилище precompute
    (TrackStore, чтение gzip ~10 мс) -> сборка на месте build_track (0.2-0.6 с,
    последний запас). Сборка осталась рабочей: она нужна и без precompute, и для
    кадра, которого в хранилище нет (тогда один раз за сессию печатается
    предупреждение -- видно, что хранилище неполное).

    build_track считается 1.1-2.6 с на кадр и держит numpy-массивы, поэтому:
      * результат кэшируется по (bag, frame) уже готовым JSON-байтом -- протяжка
        слайдера не пересчитывает один и тот же кадр и не блокирует сервер;
      * сборка идёт по одной за раз (build_lock): на 2 ГБ свободной памяти два
        параллельных пересчёта могут не ужиться;
      * попадание в кэш или хранилище не ждёт чужую сборку -- отдельный cache_lock.

    Модуль импортируется при первом запросе: он может появиться уже после старта
    сервера. Если его нет -- ImportError, обработчик отдаёт 503 с понятным текстом
    (запрос кадра, который есть в хранилище, до сборки не доходит и работает).
    """

    def __init__(self, bag_name, cache_frames=TRACK_CACHE_FRAMES, store=_STORE_DEFAULT):
        self.bag_name = bag_name
        self.max_cache = max(1, int(cache_frames))
        # store=None (явно) -- хранилище выключено: только сборка на месте.
        self.store = TrackStore() if store is _STORE_DEFAULT else store
        self._cache = OrderedDict()          # (bag, frame) -> bytes JSON
        self._cache_lock = threading.Lock()
        self._build_lock = threading.Lock()
        self._build = None
        self._error = None
        # Счётчики для диагностики (/bags): сколько кадров отдано из памяти, из
        # хранилища precompute и сколько СОБРАНО на месте. Полное хранилище --
        # сборок 0; сборки видны и по времени ответа, но счётчик точнее.
        self.cache_hits = 0
        self.store_hits = 0
        self.builds = 0

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
                self.cache_hits += 1
            return body

    def peek(self, bag, frame):
        """Уже посчитанный JSON кадра или None -- БЕЗ сборки.

        Нужно метке зоны rail: полоса строится по нитям того же кадра, если /track
        для него уже в кэше (клиент просит его параллельно с кадром), и сборку
        никто не ждёт. Хранилище precompute -- тот же источник БЕЗ сборки: при
        полном хранилище у /frame?mode=zones и /labels сборки нет вовсе.
        """
        key = (str(bag), int(frame))
        body = self._cached(key)
        if body is None:
            body = self._stored(key)
        return body

    def _store(self, key, body):
        with self._cache_lock:
            self._cache[key] = body
            self._cache.move_to_end(key)
            while len(self._cache) > self.max_cache:
                self._cache.popitem(last=False)     # вытесняем самый старый кадр

    def _stored(self, key):
        """Готовое тело кадра из хранилища precompute (и сразу в кэш памяти).

        Возвращает ровно те байты, что записал precompute_track.py, -- пересборки
        и повторной сериализации нет, поэтому тело /track не меняется. None --
        хранилище выключено или кадра в нём нет (тогда вызывающий собирает).
        """
        if self.store is None:
            return None
        body = self.store.get(*key)
        if body is not None:
            self.store_hits += 1
            self._store(key, body)
        return body

    def _warn_store_miss(self, key):
        """Один раз за сессию: кадр посчитан на месте, хранилище его не покрыло."""
        global _STORE_MISS_WARNED
        with _STORE_MISS_LOCK:
            if _STORE_MISS_WARNED:
                return
            _STORE_MISS_WARNED = True
        print(f'  ! геометрия кадра посчитана на месте: precompute не покрывает '
              f'закадр ({key[0]}, кадр {key[1]}) -- в хранилище '
              f'{(self.store.root if self.store else "нет каталога")} его нет; '
              f'собрать заранее: python precompute_track.py '
              f'[--bag {key[0]}]', flush=True)

    def compute_bytes(self, bag, frame, db_path):
        """JSON кадра СБОРКОЙ на месте -- без хранилища и без кэша в памяти.

        Это ровно то, что /track делал всегда, и этим же путём пользуется
        precompute_track.py: байты хранилища и сборки совпадают по построению.
        Блокировку на время сборки берёт вызывающий (json_bytes); precompute
        зовёт последовательно в одном процессе на запись.
        """
        key = (str(bag), int(frame))
        data = jsonable(self.resolve()(db_path, key[1]))
        missing = [k for k in TRACK_KEYS if k not in (data or {})]
        if missing:
            print(f'  ! build_track вернул не все ключи контракта: нет {missing}',
                  flush=True)
        body = json.dumps(data, ensure_ascii=False).encode('utf-8')
        self._store(key, body)
        self.builds += 1
        return body

    def json_bytes(self, bag, frame, db_path):
        """Тело /track: кэш памяти -> хранилище precompute -> сборка на месте."""
        key = (str(bag), int(frame))
        body = self._cached(key)
        if body is not None:
            return body
        body = self._stored(key)
        if body is not None:
            return body
        build = self.resolve()                      # ImportError -- до лока
        with self._build_lock:
            body = self._cached(key)                # пока ждали лок, мог посчитать сосед
            if body is None:
                body = self._stored(key)
            if body is not None:
                return body
            self._warn_store_miss(key)
            return self.compute_bytes(bag, frame, db_path)

    def status(self):
        try:
            self.resolve()
        except ImportError as exc:
            return {'available': False, 'error': str(exc), 'store': self.store_info()}
        return {'available': True, 'error': None, 'store': self.store_info()}

    def store_info(self):
        """Сколько хранилище покрывает: для стартового лога, None -- если выключено."""
        if self.store is None:
            return None
        frames, bags = self.store.counts()
        return {'dir': self.store.root, 'frames': frames, 'bags': bags,
                'own_frames': self.store.coverage(self.bag_name)}


def count_frames(db_path):
    """Число кадров PointCloud2 в .db3 -- только COUNT.

    Каталогу записей нужны только числа: BagFrames читает весь список rowid с
    таймстемпами, и держать шесть таких объектов ради счётчика незачем.
    """
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        try:
            cur.execute("SELECT id FROM topics WHERE type LIKE '%PointCloud2%'")
            topic_ids = [row[0] for row in cur.fetchall()]
        except sqlite3.Error:
            topic_ids = []
        if topic_ids:
            marks = ','.join('?' * len(topic_ids))
            cur.execute(f'SELECT COUNT(*) FROM messages WHERE topic_id IN ({marks})',
                        topic_ids)
        else:
            cur.execute('SELECT COUNT(*) FROM messages')
        return int(cur.fetchone()[0])
    finally:
        conn.close()


class ViewProfiles:
    """Профили вида по записям в web/view_profiles.json (словарь по имени записи).

    Профиль хранит только то, что реально влияет на вид и раскраску:
      preset        -- пресет камеры (lidar / track / profile / top / side / iso /
                       behind / fit), необязательный;
      position/target -- явные координаты камеры и цели, необязательные;
      axis_x, gauge_mm -- модель рельсов (ось и колея), те же, что у ползунков;
      max_dist      -- дальность, с которой начинается ползунок «Дальность, м»;
      analysis_dist -- дальность серверного анализа (коридор, ось, профиль рельсов)
                       и нормировка раскраски: применяется при создании экземпляра
                       записи, то есть при её первом открытии после старта.

    Файл читается один раз при старте и целиком перезаписывается при сохранении
    (атомарно, через .tmp + os.replace) -- профили переживают перезапуск сервера.
    """

    def __init__(self, path=VIEW_PROFILES):
        self.path = path
        self._lock = threading.Lock()
        self._data = {}
        self._load()

    def _load(self):
        try:
            with open(self.path, encoding='utf-8') as fh:
                data = json.load(fh)
            self._data = ({str(k): v for k, v in data.items() if isinstance(v, dict)}
                          if isinstance(data, dict) else {})
        except FileNotFoundError:
            self._data = {}
        except (OSError, ValueError) as exc:  # noqa: BLE001
            print(f'  ! {self.path}: профили не прочитаны ({exc}); беру пустой набор',
                  flush=True)
            self._data = {}

    def get(self, bag):
        with self._lock:
            return dict(self._data.get(str(bag)) or {})

    def has(self, bag):
        with self._lock:
            return str(bag) in self._data

    def all(self):
        with self._lock:
            return dict(self._data)

    def save(self, bag, profile, extra=None):
        """Слить присланный профиль с сохранённым и записать файл.

        Значения фильтруются: неизвестный пресет отбрасывается, координаты и
        числа -- только конечные, `analysis_dist` сервер дописывает сам (клиент
        его не знает) через `extra`.
        """
        clean = {}
        preset = profile.get('preset')
        if preset in PROFILE_PRESETS:
            clean['preset'] = preset
        for key in ('position', 'target'):
            value = profile.get(key)
            if isinstance(value, (list, tuple)) and len(value) == 3:
                try:
                    clean[key] = [float(v) for v in value]
                except (TypeError, ValueError):
                    pass
        for key in PROFILE_NUM_KEYS:
            value = profile.get(key)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                clean[key] = float(value)
        for key, value in (extra or {}).items():
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                clean[key] = float(value)
        clean['saved_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
        with self._lock:
            self._data[str(bag)] = clean
            tmp = self.path + '.tmp'
            with open(tmp, 'w', encoding='utf-8') as fh:
                json.dump(self._data, fh, ensure_ascii=False, indent=2, sort_keys=True)
                fh.write('\n')
            os.replace(tmp, self.path)      # читатель не увидит половину файла
        return clean


class _BagLease:
    """Запись, взятая на время запроса: пока запрос идёт, её не вытесняют."""

    def __init__(self, catalog, name, viewer):
        self.catalog = catalog
        self.name = name
        self.viewer = viewer

    def __enter__(self):
        return self.viewer

    def __exit__(self, *exc):
        self.catalog.release(self.name)
        return False


class BagCatalog:
    """Записи (датасеты) в одном процессе: ленивое открытие + LRU.

    Позиционный аргумент -- либо папка одной записи (внутри *.db3; поведение как
    раньше), либо корень с папками записей. Экземпляр Viewer поднимается при первом
    обращении к записи (анализ занимает ~1-3 с), а живых держим не больше max_live:
    при открытии новой самая давняя закрывается (frames.close() + отпускаем облако
    анализа). Запись, взятая запросом (checkout), не вытесняется.
    """

    def __init__(self, root, args, max_live=MAX_LIVE_BAGS):
        self.root = root
        self.args = args
        self.max_live = max(1, int(max_live))
        self.profiles = ViewProfiles()
        self.paths = OrderedDict()      # имя записи -> путь к .db3
        self._frames = {}               # имя -> число кадров (только COUNT)
        self._live = OrderedDict()      # имя -> Viewer, порядок = LRU
        self._inuse = {}                # имя -> сколько запросов держат запись
        self._lock = threading.RLock()
        self._boot_lock = threading.Lock()
        self.opened = 0
        self.evicted = 0
        self._scan()

    def _scan(self):
        root = self.root
        if os.path.isfile(root) and root.endswith('.db3'):
            self.paths[os.path.basename(os.path.dirname(os.path.abspath(root)))] = root
            return
        if not os.path.isdir(root):
            raise FileNotFoundError(f'{root}: ни папки, ни .db3')
        try:
            # Папка записи: внутри есть *.db3 -- ведём себя как раньше (одна запись).
            self.paths[os.path.basename(os.path.abspath(root))] = bag_reader.find_db3(root)
            return
        except FileNotFoundError:
            pass
        for name in sorted(os.listdir(root)):
            bag_dir = os.path.join(root, name)
            if not os.path.isdir(bag_dir):
                continue
            try:
                self.paths[name] = bag_reader.find_db3(bag_dir)
            except (FileNotFoundError, OSError):
                continue
        if not self.paths:
            raise FileNotFoundError(f'{root}: ни .db3, ни подпапок с .db3')

    @property
    def names(self):
        return list(self.paths)

    @property
    def multi(self):
        return len(self.paths) > 1

    @property
    def default_name(self):
        wanted = getattr(self.args, 'bag', None)
        if wanted and wanted in self.paths:
            return wanted
        return self.names[0]

    def has(self, name):
        return name in self.paths

    def frames(self, name):
        with self._lock:
            count = self._frames.get(name)
        if count is None:
            count = count_frames(self.paths[name])
            with self._lock:
                self._frames[name] = count
        return count

    def as_json(self):
        """Список записей для клиента: имя, кадры, есть ли профиль вида."""
        return [{'name': name, 'frames': self.frames(name),
                 'has_profile': self.profiles.has(name)} for name in self.names]

    # ---------------------------------------------------------------- live ----

    def _args_for(self, name):
        """Аргументы запуска для записи: значения профиля вида перекрывают CLI.

        `--axis-x` из командной строки считается явным выбором и профиль не
        перебивает: иначе сохранённый вид незаметно отменял бы флаг.
        """
        profile = self.profiles.get(name)
        overrides = {key: float(profile[key]) for key in PROFILE_NUM_KEYS
                     if isinstance(profile.get(key), (int, float))}
        if self.args.axis_x is not None:
            overrides.pop('axis_x', None)
        if not overrides:
            return self.args
        args = copy.copy(self.args)         # у каждой записи свои значения
        for key, value in overrides.items():
            setattr(args, key, value)
        return args

    def _open(self, name):
        with self._lock:
            viewer = self._live.get(name)
            if viewer is not None:
                self._live.move_to_end(name)
                return viewer
        with self._boot_lock:               # анализ тяжёлый: открываем по одной
            with self._lock:
                viewer = self._live.get(name)
                if viewer is not None:
                    self._live.move_to_end(name)
                    return viewer
            print(f'  bag {name}: открываю (живых не больше {self.max_live})', flush=True)
            viewer = Viewer(self.paths[name], self._args_for(name))
            with self._lock:
                self._live[name] = viewer
                self._live.move_to_end(name)
                self.opened += 1
                while len(self._live) > self.max_live:
                    victim = next((n for n in self._live
                                   if n != name and n not in self._inuse), None)
                    if victim is None:
                        print(f'  ! все живые записи заняты запросами: держу '
                              f'{len(self._live)} вместо {self.max_live}', flush=True)
                        break
                    self._close(victim, self._live.pop(victim))
                    self.evicted += 1
            return viewer

    def _close(self, name, viewer):
        try:
            viewer.frames.close()
        except Exception:  # noqa: BLE001 -- закрытие не должно ронять сервер
            pass
        viewer._zone_cloud = None           # облако анализа: самая тяжёлая часть
        print(f'  bag {name}: вытеснен из памяти (LRU {self.max_live})', flush=True)

    def checkout(self, name=None):
        """Взять запись на время запроса (контекстный менеджер).

        Счётчик занятости поднимается ДО открытия: пока запись открывается, её
        тоже нельзя вытеснять.
        """
        key = name or self.default_name
        if key not in self.paths:
            raise KeyError(f'unknown bag {name!r}: доступны {", ".join(self.names)}')
        with self._lock:
            self._inuse[key] = self._inuse.get(key, 0) + 1
        try:
            viewer = self._open(key)
        except Exception:
            self.release(key)
            raise
        return _BagLease(self, key, viewer)

    def release(self, name):
        with self._lock:
            left = self._inuse.get(name, 1) - 1
            if left > 0:
                self._inuse[name] = left
            else:
                self._inuse.pop(name, None)

    def open_default(self):
        """Прогреть запись по умолчанию -- как раньше делал main()."""
        return self._open(self.default_name)

    def live_viewer(self, name):
        """Живой экземпляр или None (ничего не открывает)."""
        with self._lock:
            return self._live.get(name) if self.has(name) else None

    def status(self):
        with self._lock:
            live = list(self._live)
            track = {name: {'cache_hits': viewer.track.cache_hits,
                            'store_hits': viewer.track.store_hits,
                            'builds': viewer.track.builds}
                     for name, viewer in self._live.items()}
        return {
            'root': self.root,
            'default': self.default_name,
            'max_live': self.max_live,
            'bags': [{'name': name, 'frames': self.frames(name), 'live': name in live,
                      'has_profile': self.profiles.has(name)} for name in self.names],
            'live': live,
            'inuse': {k: v for k, v in self._inuse.items() if v},
            'opened': self.opened,
            'evicted': self.evicted,
            # Геометрия пути у живых записей: откуда взялись кадры. `builds` -- сколько
            # собрано НА МЕСТЕ (хранилище precompute не покрыло); при полном
            # хранилище во время воспроизведения тут ноль.
            'track': track,
        }

    def close(self):
        with self._lock:
            for viewer in list(self._live.values()):
                try:
                    viewer.frames.close()
                except Exception:  # noqa: BLE001
                    pass
            self._live.clear()


class RailZone:
    """Метка зоны `rail` -- по ОСИ КАДРА (полилинии узлов), а не по прямой у пола.

    Два дефекта, которые это лечит.

    1) Полоса рельса в `zones.ZoneParams` задана относительно ПЛОСКОСТИ ПОЛА
    (`rail_low = -0.10`), то есть заходит на 10 см под неё, и классификация
    перезаписывает метку `bed` меткой `rail`. На записях, где плоскость пола
    «плавает» (rms 0.128 м), это красило точки под полом как рельс: замер на
    `doubleT_obstacle` -- 86 точек из 2108 (4.1 %) ниже пола, до -0.073 м; на
    кадре 100 -- 87 (до -0.084 м).

    2) Ось считалась ПРЯМОЙ (`line`: x = s*y + i, `top_line`: z = s*y + i), а
    геометрия пути давно строит её полилинией (узлы каждые 0.5 м, у каждого своя
    касательная и своя измеренная коронка). Чем круче поворот, тем сильнее прямая
    расходится с рельсом: замер полосы по узлам против прямой маски (for_hackathon,
    /frame?mode=zones) -- `roundT_doubleT` кадр 0 левая нить: 677 точек в полосе,
    прямая теряет 165 (24 %), ложных 6; кадр 126 правая: 685, теряет 75 (11 %);
    `doubleT_obstacle` кадр 0: 420/394, теряет 35/38 (8-10 %);
    `squareT_platform_squareT_switch` кадр 0 левая: 619, теряет 66 (11 %).
    Потерянные точки уходили в зону middle (зелёную), и рельс читался как
    «оранжевый + зелёный».

    Ось кадра -- узлы (y, x_axis, z_top): полоса строится по |u| <= u_select_m
    (0.06 м) и -head_below <= dz <= +head_above, где u = x - x_axis(y),
    dz = z - z_top(y), а x_axis(y)/z_top(y) -- ЛИНЕЙНАЯ ИНТЕРПОЛЯЦИЯ между
    соседними узлами по y (по ближайшему узлу шаг 0.5 м давал бы ступеньки).
    Источник узлов, по порядку: `meta.rail_mesh.rails.<side>.nodes`
    ([[y, x_axis, z_top], ...], сортировка по y) -> `mesh_measured.vertices`
    -> `mesh.vertices`. У меша вершин на узел не всегда `nodes_per_bin` (у ТЕЛА
    рельса 20, а не 5), поэтому узлы ищутся разбиением по разрывам y, а не делением
    длины на 5; деление по `nodes_per_bin` осталось последним запасом.

    JSON /track кэширован по (bag, frame) в TrackGeometry, поэтому узлы берутся из
    кэша (peek), а полоса по ним считается векторно (~несколько мс на кадр
    180-350 тыс. точек). Источники узлов по порядку: кэш памяти, хранилище
    precompute (то же `peek` -- чтение gzip ~10 мс, сборки нет) и только последним
    -- сборка по запросу (`allow_build=True` у /frame и /labels; сборка
    переиспользуется обоими маршрутами, кэш по (bag, frame)). Полоса тогда всегда по СОБСТВЕННОЙ оси
    кадра: у соседа при отставании в один кадр она теряет 1.9 % рельса, при 20 --
    1.6 %, при 50 -- 89.7 % (замер `roundT_doubleT` кадр 150: узлы кадров 149, 130 и
    100 против полосы своего кадра), а потерянные точки уходят в зелёную middle.
    Дополнительно у /frame был замер 12 точек rail из 1365 в полосе при оси кадра
    0 и 1375 из 1395 при оси кадра 150 -- то есть даже сосед в один кадр даёт
    зелёные точки на теле рельса, поэтому чужая ось не годится ни при каком
    отставании.

    Узлы БЛИЖАЙШЕГО уже разобранного кадра (полуширина вдвое больше) остались
    последним запасом: он работает, только когда собрать нечего (нет
    track_geometry, сборка упала) -- тогда это лучше, чем ничего.

    У самой метки есть и маршрут /labels (см. `label_bytes`): он считает полосу по
    оси ИМЕННО своего кадра и без позиций. Клиент зовёт его после `/track` этого
    кадра и перекрашивает уже нарисованные точки -- на случай, когда ось кадра
    взять не удалось (запасной путь) или геометрия подъехала позже кадра.

    Если узлов нет ни у одной нити -- прямая `line`/`top_line` (как было раньше).
    Если нет и её (нет track_geometry, модуль упал, в JSON нет rail_mesh) --
    падение запрещено: `labels()` возвращает полосу zones с `rail_low = 0.0`,
    то есть НЕ ниже плоскости пола, и помечает причину в `reason` (лог и /meta).

    Гарантия «не ниже пола»: в конце точки с `zrel < 0` никогда не остаются
    меткой rail -- они перекрашиваются в `bed`. Это работает во всех путях.
    """

    # Полоса вокруг оси: X -- по узлам оси, Z -- от измеренной коронки головки.
    # Значения по умолчанию -- правило самого детектора (|u| <= 60 мм от оси,
    # глубина до 40 мм под верхом головки, как в apply_rail_lines); если в JSON
    # /track есть u_select_m/depth_select_m, берём их (они у записи свои).
    # Колея/ось пары тут не участвуют: нить -- измеренная линия, а не модель.
    HALF_X = 0.06            # метры, по умолчанию (u_select_m из /track, если есть)
    HEAD_BELOW = 0.04        # метры вниз от коронки (0..40 мм) = depth_select_m[1]
    # Вверх от коронки -- 20 мм: коронка это СГЛАЖЕННАЯ медиана сечения (и max z по
    # группе вершин), реальные точки головки ложатся до ~20 мм выше неё. Замер: с
    # 0 мм полоса по узлам давала 369 точек на `roundT_doubleT` f0 (левая нить)
    # вместо 677-678 при 20 мм.
    HEAD_ABOVE = 0.02
    NODE_JUMP_M = 0.05       # разрыв y между узлами меша (соседние узлы ~0.5 м)

    def __init__(self, track, bag_name, db_path=None):
        self.track = track
        self.bag_name = bag_name
        self.db_path = db_path      # нужен, чтобы по требованию собрать /track кадра
        self.ref_frame = 0          # опорный кадр нитей (Viewer.path_frame)
        self._by_frame = {}         # frame -> [{'side','line','nodes','node_source'}]
        self._reason = None         # почему нет нитей
        self.half_x = self.HALF_X           # из u_select_m записи
        self.head_below = self.HEAD_BELOW   # из depth_select_m записи
        self.head_above = self.HEAD_ABOVE
        self.source = 'unknown'     # 'lines' | 'band'
        self.lines_frame = None     # из какого кадра взяты нити
        self.model = None           # 'polyline' | 'mixed' | 'straight' | 'band'
        self.node_source = None     # 'nodes' | 'mesh_measured' | 'mesh' | 'per_bin' | 'line'
        self.nodes_last = 0         # узлов оси в последней классификации
        self.nodes_y_range = None   # (y_min, y_max) узлов последней классификации
        self.last_ms = 0.0
        self.last_points = 0
        self._last_frame = None     # кадр последней классификации (для /meta)
        # Кадр, по оси которого посчитана полоса В ЭТОМ запросе (thread-local):
        # /frame, /labels и предзагрузка идут параллельно (ThreadingHTTPServer), а
        # `lines_frame` -- общее поле для /meta, и чужой запрос успевает его
        # переписать между `lines()` и чтением. Из-за гонки полоса молча бралась
        # вдвое шире (`<-- НЕ СВОЯ ОСЬ`), а /labels отдавал чужой axis_frame:
        # замер на живой странице -- предупреждения «ось кадра 8, показан кадр 7» и
        # «ось кадра 1, показан кадр 0» при полностью исправной геометрии.
        self._tls = threading.local()

    MAX_FRAMES = 8                  # нити по кадрам: столько же, сколько /track

    @staticmethod
    def _axis_from_verts(verts, per_bin, thr=None):
        """Узлы оси (n, 3: y, x_axis, z_top) из вершин меша; None -- не получилось.

        Меш строится секциями вдоль y: вершины одной секции (узла) идут подряд, а
        между секциями y прыгает на шаг бина (~0.5 м) против миллиметров внутри.
        Поэтому секции ищутся по РАЗРЫВАМ y, а не делением длины на `nodes_per_bin`:
        у полоски измеренной коронки вершин на узел ровно `nodes_per_bin` (5), а у
        ТЕЛА рельса (GOST + коронка) их 20, и деление на 5 дало бы мусорные узлы.

        Из секции берутся `per_bin` самых ВЫСОКИХ вершин (коронка головки): ось --
        среднее по ним, коронка -- max z. Для полоски это все её вершины, для тела --
        те же 5 узлов коронки (тело уходит только вниз).
        """
        try:
            arr = np.asarray(verts, float)
        except (TypeError, ValueError):
            return None
        if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] < 4:
            return None
        if not np.isfinite(arr).all():
            return None
        n_per = max(2, int(per_bin))
        jump = RailZone.NODE_JUMP_M if thr is None else float(thr)
        cut = np.r_[0, np.nonzero(np.abs(np.diff(arr[:, 1])) > jump)[0] + 1, arr.shape[0]]
        sizes = np.diff(cut)
        if sizes.size >= 2 and sizes.min() >= 2 and sizes.max() <= 4 * n_per:
            # Секции найдены по разрывам y. Размеры секций могут и отличаться (у
            # меша с зашитыми разрывами) -- важно лишь, что секция не вырождена:
            # вырожденная (одна вершина) дала бы ось по краю головки, а не по центру.
            k = min(n_per, int(sizes.min()))
            rows = []
            for a, b in zip(cut[:-1], cut[1:]):
                sec = arr[a:b]
                top = sec[np.argsort(-sec[:, 2])[:k]]
                rows.append([top[:, 1].mean(), top[:, 0].mean(), top[:, 2].max()])
            nodes = np.asarray(rows, float)
        elif sizes.size == 1 and arr.shape[0] % n_per == 0:
            # Разрывов y нет вовсе: вершины одного узла лежат на одном y (так устроена
            # развёртка ВДОЛЬ ПРЯМОЙ оси) -- делим ровно по `nodes_per_bin`.
            grouped = arr.reshape(arr.shape[0] // n_per, n_per, 3)
            nodes = np.c_[grouped[:, :, 1].mean(axis=1), grouped[:, :, 0].mean(axis=1),
                          grouped[:, :, 2].max(axis=1)]
        else:
            return None
        order = np.argsort(nodes[:, 0], kind='stable')
        nodes = nodes[order]
        nodes = nodes[np.r_[True, np.diff(nodes[:, 0]) > 1e-6]]  # np.interp: рост y
        return nodes if nodes.shape[0] >= 2 else None

    @staticmethod
    def _axis_from_key(raw):
        """Узлы из готового ключа `nodes` = [[y, x_axis, z_top], ...]; None -- нет."""
        if not isinstance(raw, (list, tuple)) or len(raw) < 2:
            return None
        rows = []
        for item in raw:
            if not isinstance(item, (list, tuple)) or len(item) < 3:
                return None
            try:
                rows.append([float(item[0]), float(item[1]), float(item[2])])
            except (TypeError, ValueError):
                return None
        arr = np.asarray(rows, float)
        if not np.isfinite(arr).all():
            return None
        arr = arr[np.argsort(arr[:, 0], kind='stable')]
        arr = arr[np.r_[True, np.diff(arr[:, 0]) > 1e-6]]
        return arr if arr.shape[0] >= 2 else None

    @classmethod
    def _side_axis(cls, mesh_item, list_item, per_bin):
        """Узлы оси одной нити и имя источника; (None, None) -- если не нашлись.

        Порядок: ключ `nodes` -> `mesh_measured` (полоска коронки) -> `mesh` (тело
        рельса). Меш берём и из meta, и из верхнего списка `rails` (вершины есть
        только там).
        """
        nodes = cls._axis_from_key((mesh_item or {}).get('nodes'))
        if nodes is not None:
            return nodes, 'nodes'
        for src in ('mesh_measured', 'mesh'):
            for item in (list_item, mesh_item):
                verts = ((item or {}).get(src) or {}).get('vertices')
                if not verts:
                    continue
                nodes = cls._axis_from_verts(verts, per_bin)
                if nodes is not None:
                    return nodes, src
        return None, None

    def _nearest_frame(self, key):
        """Ближайший по номеру кадр с уже разобранной геометрией (или None).

        Нужен как заменитель своего кадра: геометрия СОСЕДНИХ кадров почти совпадает
        (замер roundT_doubleT, кадр 126 против полосы своего кадра: сосед 124 теряет
        3.6 % полосы, сосед 125 -- 6.5 %, опорный кадр 0 -- 98.7 %).
        """
        best = best_d = None
        for k, items in self._by_frame.items():
            if not items:
                continue
            d = abs(int(k) - int(key))
            if best_d is None or d < best_d:
                best, best_d = k, d
        return best

    def lines(self, frame, allow_build=False):
        """Нити кадра: [{'side','line','nodes','node_source'}]; [] -- нет (см. reason).

        Сначала пробуем JSON ИМЕННО этого кадра (peek из кэша /track, БЕЗ сборки:
        попадание в кэш не должно ждать чужую сборку). Облако смещается кадр к
        кадру, поэтому узлы своего кадра точнее; если их в кэше ещё нет, а сборка
        разрешена -- собираем; если нет -- берём ближайший УЖЕ разобранный кадр, а
        если и того нет -- опорный (path_frame), он посчитан при анализе записи.

        `allow_build=True` (маршруты /frame и /labels) включает сборку: нужна ось
        ИМЕННО этого кадра, иначе полоса уезжает (сосед в один кадр теряет 1.9 %
        рельса, в 50 кадров -- 89.7 %), а потерянные точки красит зелёная middle.
        Клиент просит /track того же кадра параллельно, поэтому обычно сборки нет.
        Падение запрещено: не собралось -- уходим в прежний fallback.
        """
        key = int(frame)
        self._tls.used = None       # ось ЭТОГО запроса: пока не определена
        cached = self._by_frame.get(key)
        if cached is not None:
            self._use_frame(key)
            return cached
        raw = self.track.peek(self.bag_name, key)
        used = key
        build_note = None
        if raw is None and allow_build and self.db_path is not None:
            # Две попытки: сборка кадра тяжёлая (память, чтение .db3, общий кэш
            # моделей), и разовый сбой на ней уже видели на живом сервере -- при
            # загрузке страницы сборка кадра 6 roundT_doubleT упала, метка rail
            # уехала на ось кадра 5 и рельс покрасился зелёным. Повтор даёт свою
            # ось; при устойчивом сбое поведение прежнее -- запасной путь ниже.
            for _attempt in (1, 2):
                try:
                    raw = self.track.json_bytes(self.bag_name, key, self.db_path)
                    build_note = None
                    break
                except Exception as exc:  # noqa: BLE001 -- фича, не падение
                    build_note = f'геометрия кадра {key} не собрана ({exc!r})'
                    raw = None
        if raw is None:
            # Ни своего кадра, ни сборки (запасной путь: сборка не разрешена или
            # упала -- нет track_geometry): берём ближайший разобранный кадр, иначе
            # опорный. НЕ кэшируем чужие нити под своим ключом -- иначе кадр,
            # появившийся в кэше позже, остался бы без своих.
            near = self._nearest_frame(key)
            if near is not None:
                self._use_frame(near)
                return self._by_frame[near]
            used = self.ref_frame
            raw = self.track.peek(self.bag_name, used)
        if raw is None:
            self._reason = build_note or (
                f'в кэше /track нет ни кадра {key}, ни ближайшего '
                f'разобранного, ни опорного {used} (сборка кадра '
                f'не удалась)')
            return []
        try:
            data = json.loads(raw.decode('utf-8'))
        except Exception as exc:  # noqa: BLE001 -- фича, не падение
            self._reason = f'JSON /track не разобран: {exc!r}'
            return []
        rail_mesh = ((data or {}).get('meta') or {}).get('rail_mesh') or {}
        rails = rail_mesh.get('rails')
        # Верхний список `rails` (не meta) -- единственное место с ВЕРШИНАМИ мешей:
        # meta.rail_mesh.rails.<side> несёт только n_verts/n_faces и коэффициенты.
        by_side = {}
        if isinstance(rails, dict):
            for side, item in rails.items():
                if isinstance(item, dict):
                    by_side.setdefault(str(side), {})['meta'] = item
        for item in (data.get('rails') or []):
            if isinstance(item, dict) and item.get('side') is not None:
                by_side.setdefault(str(item['side']), {})['list'] = item
        if not by_side:
            self._reason = '/track вернул JSON без meta.rail_mesh.rails (нитей нет)'
            return []
        # Правило отбора точек у самого детектора: те же границы, что у меша.
        try:
            self.half_x = float(rail_mesh.get('u_select_m') or self.HALF_X)
        except (TypeError, ValueError):
            self.half_x = self.HALF_X
        depth = rail_mesh.get('depth_select_m')
        if isinstance(depth, (list, tuple)) and len(depth) == 2:
            try:
                self.head_below = float(depth[1])
                # Вверх -- не меньше 20 мм: коронка это сглаженная медиана сечения,
                # и точки головки ложатся выше неё (см. HEAD_ABOVE).
                self.head_above = max(self.HEAD_ABOVE, -float(depth[0]))
            except (TypeError, ValueError):
                pass
        try:
            per_bin = int(rail_mesh.get('nodes_per_bin') or 5)
        except (TypeError, ValueError):
            per_bin = 5
        found = []
        for side, pair in sorted(by_side.items()):
            meta_item, list_item = pair.get('meta'), pair.get('list')
            straight = None
            line = meta_item.get('line') if isinstance(meta_item, dict) else None
            top = meta_item.get('top_line') if isinstance(meta_item, dict) else None
            if isinstance(line, dict) and isinstance(top, dict):
                try:
                    straight = (float(line['s']), float(line['i']),
                                float(top['s']), float(top['i']))
                except (KeyError, TypeError, ValueError):
                    straight = None
            nodes, node_src = self._side_axis(meta_item, list_item, per_bin)
            if straight is None and nodes is None:
                continue
            found.append({'side': side, 'line': straight, 'nodes': nodes,
                          'node_source': node_src})
        if not found:
            self._reason = ('в rail_mesh нет ни узлов оси (nodes/mesh), ни линий '
                            'нитей (line/top_line)')
        while len(self._by_frame) >= self.MAX_FRAMES:
            self._by_frame.pop(next(iter(self._by_frame)))    # LRU: самый старый кадр
        self._by_frame[used] = found      # под ключом ТОГО кадра, чей JSON разобран
        self._use_frame(used)
        return found

    def _use_frame(self, used):
        """Запомнить, по чьей оси посчитана полоса: общее поле (для /meta) и своё
        для текущего запроса (`used_frame()` -- на нём строятся решения о полуширине
        и поле axis_frame в ответах, см. `_tls` в __init__)."""
        self.lines_frame = used
        self._tls.used = used

    def used_frame(self):
        """Кадр оси ЭТОГО запроса (None -- полосы не было)."""
        return getattr(self._tls, 'used', None)

    def mask(self, xyz, frame, allow_build=False):
        """Булева маска «точка на рельсе» по оси кадра (None -- оси нет).

        Узел оси даёт x_axis(y) и z_top(y) линейной интерполяцией по y: у нити своя
        касательная (повороты в плане) и своя измеренная коронка (подъёмы/спуски).
        За пределами узлов интерполяция держит крайние значения -- это продолжение по
        прямой, как было раньше у линии.

        Если узлы взяты не из своего кадра (запасной путь: собрать геометрию не
        удалось), полосу по X берём вдвое шире: облако смещается кадр к кадру, и
        чужие узлы на узкой полосе теряют рельс. Ниже плоскости пола это не
        заводит -- ограничение стоит в `labels()`.
        """
        lines = self.lines(frame, allow_build=allow_build)
        if not lines:
            return None
        # Полуширина -- по оси ЭТОГО запроса (`used_frame()`, а не общее поле):
        # чужой запрос успевает переписать его между `lines()` и этой строкой, и
        # полоса молча становилась вдвое шире -- рельс красил пол рядом с собой.
        half = self.half_x if self.used_frame() == int(frame) else self.half_x * 2.0
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        out = np.zeros(len(x), dtype=bool)
        nodes_total = 0
        span = None
        models = set()
        for item in lines:
            nodes = item.get('nodes')
            if nodes is not None:
                x_axis = np.interp(y, nodes[:, 0], nodes[:, 1])
                z_top = np.interp(y, nodes[:, 0], nodes[:, 2])
                nodes_total += int(nodes.shape[0])
                y_lo, y_hi = float(nodes[0, 0]), float(nodes[-1, 0])
                span = (y_lo, y_hi) if span is None else (min(span[0], y_lo),
                                                          max(span[1], y_hi))
                models.add('polyline')
            elif item.get('line') is not None:
                sx, ix, sz, iz = item['line']
                x_axis = sx * y + ix
                z_top = sz * y + iz
                models.add('straight')
            else:
                continue
            out |= ((np.abs(x - x_axis) <= half)
                    & (z >= z_top - self.head_below) & (z <= z_top + self.head_above))
        self.nodes_last = nodes_total
        self.nodes_y_range = span
        self.node_source = next((i['node_source'] for i in lines
                                 if i.get('node_source')), None)
        self.model = ('mixed' if len(models) > 1 else
                      (next(iter(models)) if models else 'straight'))
        return out

    def labels(self, xyz, params, frame, allow_build=False):
        """Метки зон кадра: обычная классификация + метка rail из оси кадра.

        Классификацию зовём БЕЗ полосы рельсов (`rail_half_width = 0`,
        `contact_rail = False`), иначе старая полоса снова затирала бы `bed` под
        полом. Дальше метка rail ставится ровно точкам маски, и в конце -- жёсткая
        страховка: `zrel < 0` не может быть рельсом.

        `allow_build=True` -- ось кадра обязана быть своей, и если геометрии кадра
        в кэше нет, она считается (см. `lines`). Так делают оба маршрута метки --
        /frame (mode=zones) и /labels: иначе рельс красится двумя цветами.
        """
        t0 = time.perf_counter()
        self._last_frame = int(frame)
        base = dataclasses.replace(params, rail_half_width=0.0, contact_rail=False)
        mask = self.mask(xyz, frame, allow_build=allow_build)
        if mask is None:
            # Запасной путь: полоса zones, но НЕ ниже плоскости пола.
            base = dataclasses.replace(params, rail_low=0.0)
            self.source = 'band'
            self.model = 'band'
        else:
            self.source = 'lines'
        labels = zones.classify(xyz, base)
        if mask is not None:
            labels[mask] = zones.ZONE_NAMES.index('rail')
        zrel = xyz[:, 2] - zones.floor_profile(xyz[:, 1], params)
        cut = (labels == zones.ZONE_NAMES.index('rail')) & (zrel < 0)
        if cut.any():                      # страховка: ниже пола рельса не бывает
            labels[cut] = zones.ZONE_NAMES.index('bed')
        self.last_ms = (time.perf_counter() - t0) * 1000
        self.last_points = int((labels == zones.ZONE_NAMES.index('rail')).sum())
        return labels

    def as_json(self):
        used = self._by_frame.get(self.lines_frame) if self.lines_frame is not None else None
        return {
            'source': self.source,
            'reason': self._reason,
            # Ось кадра: 'polyline' -- по узлам (касательная + коронка у каждого),
            # 'straight' -- по прямой линии нити, 'mixed' -- и то, и другое, 'band' --
            # геометрии нет вовсе (полоса zones, не ниже пола).
            'model': self.model,
            'node_source': self.node_source,
            'nodes': self.nodes_last,
            'nodes_y_range': (None if self.nodes_y_range is None
                              else [round(self.nodes_y_range[0], 3),
                                    round(self.nodes_y_range[1], 3)]),
            'lines': len(used or []),
            'lines_frame': self.lines_frame,
            # Коэффициенты нитей (x = s*y + i) -- чтобы клиент мог локально красить
            # полосу вокруг рельса (например зелёную middle, которая иначе читается
            # как «второй рельс»), не заводя второй источник геометрии.
            'lines_xy': [[round(float(i['line'][0]), 6), round(float(i['line'][1]), 4)]
                         for i in (used or []) if i.get('line') is not None],
            'lines_from_this_frame': (self.lines_frame is not None
                                      and self.lines_frame == self._last_frame),
            'half_x_used': round(self.half_x * (1.0 if (self.lines_frame is not None
                                 and self.lines_frame == self._last_frame) else 2.0), 4),
            'half_x': round(self.half_x, 4),
            'head_below': round(self.head_below, 4),
            'head_above': round(self.head_above, 4),
            'last_ms': round(self.last_ms, 2),
            'last_points': self.last_points,
            'floor_guard': 'zrel < 0 -> bed (метка rail ниже пола невозможна)',
        }


class Viewer:
    """Источник кадров и метаданных для браузера."""

    def __init__(self, db_path, args):
        self.db_path = db_path
        self.bag_name = os.path.basename(os.path.dirname(os.path.abspath(db_path)))
        self.track = TrackGeometry(self.bag_name,
                                   store=track_store(getattr(args, 'track_cache', None)))
        self.rail_zone = RailZone(self.track, self.bag_name, db_path)
        self.frames = bag_reader.BagFrames(db_path)
        self.total = len(self.frames)
        if self.total == 0:
            raise SystemExit('No frames in bag')
        # Ось берём из своих args, а не из атрибута класса Viewer.args_axis: в
        # режиме нескольких записей у каждой записи свой профиль вида (main()
        # по-прежнему ставит классовый атрибут для совместимости).
        self.args_axis = args.axis_x
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
        # Контактного рельса в модели больше нет: он был РУЧНОЙ опцией (ползунки
        # «контактный рельс сбоку» + смещение 1.45 м из ПТЭ), а не измерением.
        # Замеры по for_hackathon: детектор (track_geometry) видит ровно ДВЕ нити
        # во всех шести записях; полоса на месте нормативного контактного рельса
        # заполнена диффузно (доля точек в ±0.05 м от медианы 49-58 % -- как у
        # равномерного заполнения, у настоящих рельсов 52-73 % при узкой головке),
        # то есть это платформа/стена, а не рельс.
        # Для анализа это ничего не меняет: полоса контактного рельса лежит ВНЕ
        # полуширины коридора (смещение 1.45 м > полуширина 1.36 м), поэтому его
        # исключение из туннеля ни разу не срабатывало -- занятость совпадает
        # бит в бит с исключением и без него (проверено: 3/46, 31/46, 0/46 бинов
        # и те же span'ы на трёх записях). Роль «известного оборудования вне
        # ходовых рельсов» играет автоматическое исключение платформы
        # (exclude_platform), а его положение берётся из облака.
        # Ось и колея -- из детектора через профиль записи (см. _args_for), не из UI.
        # Флаг --contact-rail возвращает третью нить для отладки/совместимости.
        self.zone_params.contact_rail = bool(getattr(args, 'contact_rail', False))
        # Туннель безопасности: состояние правок из панели (см. set_params).
        self.tunnel = {
            'half_width': zones.TUNNEL_HALF_WIDTH,
            'z_low': zones.TUNNEL_Z_LOW,
            'z_high': zones.TUNNEL_Z_HIGH,
            'contact': True,
            # Вдоль какой линии заметается объём: 'rails' (измеренные головки рельсов --
            # так и просили: коридор вдоль рельс), 'corridor' (центр свободного места)
            # или 'axis' (прямая по оси записи).
            'variant': 'rails',
        }
        # Ось рельсов из track_geometry берётся с одного кадра: модель пути
        # статична по bag'у, а build_track стоит ~0.4 с -- кэшируем по кадру
        # (кадр 0 при полном хранилище приходит готовым, без сборки).
        self.path_frame = 0
        self.rail_zone.ref_frame = self.path_frame   # нити по опорному кадру
        self._rail_axis_cache = {}
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
        self._refresh_path_and_tunnel()

    def _rail_axis(self, db_path, frame):
        """Ось пути из `track_geometry` для варианта `rails` (кэш по кадру).

        Ошибки наружу не глушим: их ловит `zones.path_variants` и помечает вариант
        unavailable -- вьюер из-за недоступного (или падающего) track_geometry
        падать не должен. JSON /track переиспользуется, поэтому за сборку кадра
        платят один раз и /meta, и /track.
        """
        key = int(frame)
        if key in self._rail_axis_cache:
            return self._rail_axis_cache[key]
        data = json.loads(self.track.json_bytes(self.bag_name, key, db_path))
        axis = (data or {}).get('axis')
        if not isinstance(axis, dict) or 's' not in axis or 'i' not in axis:
            raise ValueError('build_track вернул не модель оси (нет axis.s/axis.i)')
        self._rail_axis_cache[key] = axis
        return axis

    def _refresh_path_and_tunnel(self):
        """Линия хода (три варианта) и туннель безопасности вдоль выбранной линии.

        Объём заметается вдоль той линии, которую выбрал пользователь в панели
        (`/set?path_variant=corridor|rails|axis`), а не всегда вдоль оси: именно
        из-за этого габарит лез в столбы, когда ось была задана неверно.
        Вариант без данных (или 'axis') -- тот же объём по `params.axis_x`.
        """
        self.path = zones.path_variants(
            self._zone_cloud, self.zone_params, -self.analysis_dist, self.behind,
            step=zones.PATH_STEP,
            min_points=self.obstacle_points, min_height=self.obstacle_height,
            rail_model=self._rail_axis, db_path=self.db_path,
            frame=self.path_frame)
        variant = str(self.tunnel.get('variant') or 'axis')
        line = (self.path.get('variants', {}) or {}).get(variant) or {}
        path_y = path_x = None
        line_y = line.get('y')   # может быть numpy-массивом: 'or []' на нём невалиден
        if variant != 'axis' and line.get('available') and line_y is not None and len(line_y):
            path_y = np.asarray(line['y'], dtype=np.float64)
            path_x = np.asarray(line['x'], dtype=np.float64)
        self.tunnel_result = zones.tunnel_blocked(
            self._zone_cloud, self.zone_params, -self.analysis_dist, self.behind,
            path_y=path_y, path_x=path_x,
            half_width=self.tunnel['half_width'], z_low=self.tunnel['z_low'],
            z_high=self.tunnel['z_high'],
            exclude_contact=bool(self.tunnel['contact']),
            min_points=self.obstacle_points, min_height=self.obstacle_height)

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
        axis_frame = 0
        if mode == 'zones':
            # Метка rail -- по оси ЭТОГО кадра (allow_build=True), а не из полосы у
            # пола. Геометрии кадра в кэше памяти нет -- узлы берутся из хранилища
            # precompute, и только если кадра нет и там, он собирается здесь же
            # (сборка переиспользуется /track того же кадра, кэш по (bag, frame)),
            # поэтому работа не удваивается. Иначе кадр уехал бы на ось ближайшего
            # разобранного кадра, а на повороте рельс покрасился бы ЗЕЛЁНЫМ
            # (middle): замер roundT_doubleT f150 при оси кадра 0 -- 12 точек rail
            # из 1365 в полосе рельса, f151 при оси 150 -- 1375 из 1395 (20
            # зелёных на теле рельса). Ниже плоскости пола метки rail не бывает.
            labels = self.rail_zone.labels(xyz, self.zone_params, idx, allow_build=True)
            payload = labels.astype(np.uint8).tobytes()
            kind = KIND_LABEL8
            # Ось ЭТОГО запроса, а не общее поле `lines_frame`: параллельный
            # запрос (предзагрузка соседнего кадра) успевает его переписать.
            used = self.rail_zone.used_frame()
            axis_frame = idx if used is None else int(used)
        else:
            colors = bag_reader.frame_colors(
                xyz, inten, mode=mode, max_dist=self.analysis_dist, ring=ring)
            rgb = np.clip(np.asarray(colors) * 255.0, 0, 255).astype(np.uint8)
            payload = np.ascontiguousarray(rgb).tobytes()
            kind = KIND_RGB8
        # `reserved` у кадра зон -- номер кадра, по оси которого посчитана метка
        # rail (axis_frame), как у /labels: клиент видит, своя ось или запасной
        # путь (track_geometry недоступен). У цветных режимов поле остаётся нулём.
        header = struct.pack(HEADER, MAGIC, VERSION, kind,
                             max(0, min(axis_frame, 0xFFFF)), len(xyz))
        return header + positions + payload

    def label_bytes(self, idx):
        """Только метки зон кадра: 12 байт заголовка + n байт uint8, без позиций.

        Позиции не отдаём: они уже приехали из /frame того же кадра (их там
        n*12 байт, на кадре 190 тыс. точек это 2.3 МБ). Маршрут -- уточнение
        раскраски: он ставит метку rail по оси ИМЕННО своего кадра, а /frame мог
        уехать на ось соседнего (запасной путь: геометрию собрать не удалось).

        `allow_build=True`: ось обязана быть осью ЭТОГО кадра. Геометрия обычно уже
        в кэше (клиент зовёт /labels после /track, а /frame кадра собирает её сам),
        иначе считается здесь.
        В поле `reserved` заголовка кладём номер кадра, по оси которого посчитана
        метка (axis_frame) -- клиент видит, своя ось или чужая (fallback). То же
        поле несёт и /frame в режиме zones.
        """
        idx = max(0, min(int(idx), self.total - 1))
        xyz, _inten, _ring, _mode = self._prepare(idx, 'zones')
        labels = self.rail_zone.labels(xyz, self.zone_params, idx, allow_build=True)
        used = self.rail_zone.used_frame()      # ось ЭТОГО запроса (см. _use_frame)
        axis_frame = idx if used is None else int(used)
        header = struct.pack(HEADER, MAGIC, VERSION, KIND_LABEL8,
                             max(0, min(axis_frame, 0xFFFF)), len(xyz))
        return header + labels.astype(np.uint8).tobytes()

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

    def _path_json(self):
        """Блок `path` из спеки: шаг, смещение по высоте и три варианта линии."""
        block = self.path
        variants = {}
        for name, variant in block['variants'].items():
            item = {
                'y': [round(float(v), 3) for v in variant['y']],
                'x': [round(float(v), 3) for v in variant['x']],
                'z': [round(float(v), 3) for v in variant['z']],
                'source': variant['source'],
                'available': bool(variant['available']),
            }
            if 'error' in variant:
                item['error'] = variant['error']
            variants[name] = item
        return {
            'step': round(float(block['step']), 3),
            'z_offset': round(float(block['z_offset']), 3),
            'variants': variants,
        }

    def _tunnel_json(self):
        """Блок `tunnel` из спеки (+ y/x/blocked/platform для каркаса и HUD)."""
        t = self.tunnel_result
        return {
            'half_width': round(float(t['half_width']), 3),
            'z_low': round(float(t['z_low']), 3),
            'z_high': round(float(t['z_high']), 3),
            'rail_height': round(float(t['rail_height']), 3),
            'exclude_contact': bool(t['exclude_contact']),
            'spans': [[round(float(a), 1), round(float(b), 1)] for a, b in t['spans']],
            'bins_blocked': int(t['bins_blocked']),
            'bins_total': int(t['bins_total']),
            'y': [round(float(v), 3) for v in t['y']],
            'x': [round(float(v), 3) for v in t['x']],
            'blocked': [int(bool(v)) for v in t['blocked']],
            'platform': [int(bool(v)) for v in t['platform']],
        }

    def _params_json(self):
        p = self.zone_params
        return {
            'axis_x': round(float(p.axis_x), 2),
            'gauge_mm': int(round(p.gauge * 1000)),
            'contact_rail': bool(p.contact_rail),
            'contact_offset': round(float(p.contact_offset), 3),
            'contact_side': int(p.contact_side),
            'contact_height': round(float(p.contact_height), 3),
            'rail_height': round(float(zones.RAIL_HEIGHT), 3),
            'floor': self._floor_json(),
            'rails': self._rails_json(),
            'blocked_spans': [[round(float(a), 1), round(float(b), 1)]
                              for a, b in self.blocked_spans],
            'path': self._path_json(),
            'tunnel': self._tunnel_json(),
        }

    def set_params(self, axis=None, gauge_mm=None, contact=None, offset=None,
                   side=None, tunnel_half=None, tunnel_low=None, tunnel_high=None,
                   tunnel_contact=None):
        """Живая правка модели рельсов и туннеля из браузера.

        Пол пересчитывается по полосе вокруг новой оси: он от неё зависит
        (по всему кадру нижнюю огибающую тянут вниз основания стен). Границы
        туннеля -- из спеки: полуширина 0.5..2.5, низ -0.5..1.0, верх 1.0..5.0.
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
        if tunnel_half is not None:
            self.tunnel['half_width'] = float(min(max(float(tunnel_half), 0.5), 2.5))
        if tunnel_low is not None:
            self.tunnel['z_low'] = float(min(max(float(tunnel_low), -0.5), 1.0))
        if tunnel_high is not None:
            self.tunnel['z_high'] = float(min(max(float(tunnel_high), 1.0), 5.0))
        if tunnel_contact is not None:
            self.tunnel['contact'] = bool(int(tunnel_contact))
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
            'path': self._path_json(),
            'tunnel': self._tunnel_json(),
            'grid': {'spacing': 5.0, 'half_width': 40.0,
                     'length': round(self.max_dist + self.behind, 1), 'z': 0.0},
            'zone_names': list(zones.ZONE_NAMES),
            'zone_colors': zone_colors,
            'blocked_color': [int(round(c * 255)) for c in zones.BLOCKED_COLOR],
            'rail_color': [int(round(c * 255)) for c in zones.ZONE_COLORS['rail']],
            'train_top': float(zones.TRAIN_TOP),
            'obstacle_points': self.obstacle_points,
            'rail_clearance': self.rail_clearance,
            # Откуда берётся метка зоны rail (полоса вокруг найденных нитей или
            # запасная полоса zones с rail_low = 0) + замер последнего кадра.
            'rail_zone': self.rail_zone.as_json(),
        }


# safetylayer.js и objects.js подключаются динамическим import() из app.js, но в
# версии обязаны быть: без них правка этих слоёв не меняла /version и страница не
# перезагружалась (auto-reload молча пропускал половину фронтенда).
WEB_ASSETS = ('index.html', 'app.js', 'track3d.js', 'labellayer.js', 'safetylayer.js',
              'objects.js', 'OrbitControls.js', 'three.module.min.js')

# Собранный клиент (app/client/dist) раздаёт тот же Python-сервер: страница и API
# живут на одном origin, поэтому второй процесс и CORS не нужны. Это ЕДИНСТВЕННЫЙ
# интерфейс: прежней html-страницы из web/ как страницы нет (там остаётся слой
# объектов пути web/track3d.js, но страницы, которая бы его подключала, больше нет).
# Каталога сборки нет — на `/` отдаётся короткая страница «клиент не собран».
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


class ViewerServer(ThreadingHTTPServer):
    """Обрыв связи -- не ошибка сервера.

    `_send` ловит ConnectionReset/Aborted/BrokenPipe сам, но исключение может
    вылезти и мимо него (чтение тела POST, заголовки), а socketserver печатает
    на любое исключение полный трейсбек -- за сессию лог пух на сотни килобайт.
    """

    def handle_error(self, request, client_address):
        if isinstance(sys.exc_info()[1], ConnectionError):
            return
        super().handle_error(request, client_address)


def make_handler(bags, client_dir=CLIENT_DIR):
    class Handler(BaseHTTPRequestHandler):
        server_version = 'lidar-web'

        def log_message(self, *args):
            pass

        def _send(self, body, ctype, status=200):
            self.send_response(status)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            # Оборванная связь -- НЕ ошибка сервера: страница перезагружается
            # (автообновление по /version) и бросает большой запрос на полпути,
            # браузер закрывает сокет, и запись заголовков/тела летит с
            # ConnectionReset/Aborted/BrokenPipe. Раньше это вылезало трейсбеком
            # в лог, а такой же трейсбек от /frame или /objects читается как
            # «500 на маршруте». Ловим всю отправку, включая заголовки.
            try:
                self.end_headers()
                self.wfile.write(body)
            except (ConnectionError, BrokenPipeError, OSError):
                pass

        def _json(self, obj, status=200, ensure_ascii=False):
            self._send(json.dumps(obj, ensure_ascii=ensure_ascii).encode('utf-8'),
                       'application/json; charset=utf-8', status)

        def _error_json(self, message, status=400):
            """Тело ошибки: страница показывает текст как есть.

            KeyError приходит от каталога записей, и str() от него даёт лишние
            кавычки -- разворачиваем.
            """
            if isinstance(message, KeyError) and message.args:
                message = message.args[0]
            self._json({'error': str(message)}, status)

        def _bag(self, url):
            """Запись по `?bag=` (пусто -- запись по умолчанию).

            Возвращает контекстный менеджер: пока запрос идёт, запись не вытеснят.
            Неизвестное имя -- KeyError, обработчик отдаёт 404.
            """
            return bags.checkout(parse_qs(url.query).get('bag', [None])[0])

        def _route_bag(self, url, handler):
            """Выполнить handler(viewer) для выбранной записи, ошибки -- в JSON."""
            try:
                with self._bag(url) as viewer:
                    return handler(viewer)
            except KeyError as exc:
                return self._error_json(exc, 404)
            except (ValueError, TypeError) as exc:
                # Кривое тело запроса (не-число в frame/click, битый JSON) -- вина
                # клиента: 400 с обычным текстом, а не 500 с repr внутренностей.
                return self._error_json(str(exc), 400)
            except Exception as exc:  # noqa: BLE001
                return self._error_json(f'{exc!r}', 500)

        def _body(self):
            """Тело POST как объект JSON (пустое тело -- пустой объект)."""
            length = int(self.headers.get('Content-Length') or 0)
            raw = self.rfile.read(length) if length > 0 else b''
            if not raw.strip():
                return {}
            payload = json.loads(raw.decode('utf-8'))
            if not isinstance(payload, dict):
                raise ValueError('тело запроса должно быть объектом JSON')
            return payload

        def _labeling(self):
            try:
                import labeling
            except ImportError as exc:      # модуль не найден -- фича, не падение
                raise RuntimeError(f'labeling недоступен: {exc!r}') from exc
            return labeling

        def _xyz_at(self, viewer):
            """Источник кадра для разметки: запись уже открыта вьюером."""
            def xyz_at(frame):
                i = max(0, min(int(frame), viewer.total - 1))
                fr = viewer.frames[i]
                return fr.xyz, fr.intensity, fr.ring

            return xyz_at

        def _hidden_indices(self, viewer, frame, removed_mask):
            """Полные индексы кадра -> индексы массива, который РИСУЕТ страница.

            Облако приходит странице через /frame, а это `_prepare`: обрезка
            коридора (`max_dist`/`behind`) и, при `max_points`, прореживание шагом.
            Разметка считает маску по ПОЛНОМУ кадру (`viewer.frames[i].xyz`), и её
            индексы нельзя отдавать клиенту как есть -- после обрезки все номера
            сдвигаются. Здесь маска переводится в нумерацию `_prepare`: страница
            гасит ровно те точки, которые заслонил объект.

            Длина ответа равна `points_removed` ЗА ВЫЧЕТОМ точек, которых у
            страницы нет вовсе: заслонённый луч может иметь второй возврат дальше
            `max_dist` (в /frame он обрезан, и гасить его нечего). Их число --
            `counts.points_removed_drawn`; разница видна в панели.
            """
            i = max(0, min(int(frame), viewer.total - 1))
            mask = np.asarray(removed_mask, dtype=bool)
            xyz = np.asarray(viewer.frames[i].xyz)
            if mask.size != xyz.shape[0]:
                return []
            keep = np.flatnonzero(bag_reader.trim_range(
                xyz, max_dist=viewer.max_dist, behind=viewer.behind))
            step = 1
            if 0 < viewer.max_points < keep.size:
                step = -(-keep.size // viewer.max_points)
            drawn = keep[::step]
            hidden = np.isin(drawn, np.flatnonzero(mask))
            return [int(v) for v in np.flatnonzero(hidden)]

        def _hz_of(self, viewer):
            """Частота записи, как в /meta: по длительности всех кадров.

            Нужна для км/ч: клиент переводит км/ч обратно в м/кадр по той же
            частоте, и если сервер считал бы её по-своему (10 Гц по умолчанию),
            записанный ход отличался бы от введённого человеком.
            """
            ts = viewer.frames.timestamps
            span = (ts[-1] - ts[0]) / 1e9 if len(ts) > 1 else 0.0
            return float(viewer.total / span) if span else 0.0

        def _motion_of(self, labeling, viewer, hz=None):
            """Ход записи для трека: ПРОФИЛЬ по кадрам (`labeling.motion_profile`).

            Сводка (`m_per_frame`) -- медиана достоверных кадров профиля, сдвиг
            между СОСЕДНИМИ кадрами. Пары через 10 кадров (`motion_pairs`) остаются
            только справочными числами в ответе (`pairs`/`pairs_median`): именно
            они давали алиас -- на `roundT_doubleT` печаталось 0.750 вместо 1.500,
            на `roundT_squareT...` -1.325 вместо +1.250 (см. `check_motion.py`).
            Первая пара больше не задаёт ход, поэтому ход не зависит от того, на
            какой кадр записи попала пара.

            `hz` -- частота для км/ч: при записи клиент присылает свою, и ход в
            карточке обязан считаться по ней же, иначе км/ч не сойдутся.
            """
            pairs = labeling.motion_pairs(viewer.total)[:1]
            return labeling.measure_motion(viewer.db_path,
                                           xyz_at=self._xyz_at(viewer),
                                           total=viewer.total,
                                           hz=float(hz or self._hz_of(viewer)
                                                    or labeling.HZ_DEFAULT),
                                           pairs=pairs or None)

        def _meta_json(self, viewer):
            """meta записи; в режиме корня добавляем список записей для селектора.

            В одиночном режиме ключа `bags` нет -- ответ байт-в-байт как раньше.
            """
            body = viewer.meta()
            if bags.multi:
                body['bags'] = bags.as_json()
            return body

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

        def _web_asset(self, path):
            """Отдать файл клиента из web/ по имени (`/app.js`, `/safetylayer.js`).

            Отдельные маршруты на каждый файл приходилось дописывать руками и легко
            забыть про новый слой, поэтому отдаём любой файл веб-каталога с
            расширением: путь проверяется на выход из каталога.
            """
            rel = path.lstrip('/')
            root = os.path.normpath(WEB_DIR)
            full = os.path.normpath(os.path.join(root, rel))
            ext = os.path.splitext(full)[1].lower()
            if (not full.startswith(root + os.sep) or ext not in ASSET_TYPES
                    or not os.path.isfile(full)):
                return None
            self._static(rel, ASSET_TYPES[ext])
            return True

        def do_GET(self):
            url = urlparse(self.path)
            path = url.path
            # `/` — рабочий интерфейс проекта: клиент из web/ (его меню и есть стандарт
            # заказчика). Собранный React-клиент на главную НЕ выходит: его панель
            # выглядит иначе, и заказчик просил ровно ту же картину, что в web/.
            # `/legacy` убран -- страница ровно одна.
            if path in ('/', '/index.html'):
                self._web_asset('/index.html')
                return
            if path == '/react':
                if self._client('index.html'):
                    return
                self._web_asset('/index.html')
                return
            if path.startswith('/assets/') and self._client(path):
                # Сборка React-клиента: index.html ссылается на /assets/*.js|css, а без
                # этого маршрута отдавалась только страница (её скрипты отвечали 404,
                # и /react был пустым). Файл берётся из app/client/dist, путь
                # проверяется `_client` на выход из каталога; нет файла -- идём дальше.
                return
            if self._web_asset(path):
                # Любой файл web/ с известным расширением (app.js, track3d.js,
                # three.module.min.js и все слои) уже отдан выше -- отдельные
                # маршруты на каждый файл не нужны.
                return
            if path == '/set':
                q = parse_qs(url.query)

                def num(key, cast=float):
                    v = q.get(key, [None])[0]
                    return None if v in (None, '') else cast(v)

                def apply(viewer):
                    try:
                        body = json.dumps(viewer.set_params(
                            axis=num('axis'), gauge_mm=num('gauge', int),
                            contact=num('contact', int), offset=num('offset'),
                            side=num('side', int),
                            tunnel_half=num('tunnel_half'),
                            tunnel_low=num('tunnel_low'),
                            tunnel_high=num('tunnel_high'),
                            tunnel_contact=num('tunnel_contact', int))).encode('utf-8')
                    except Exception as exc:  # noqa: BLE001
                        return self._json({'error': str(exc)}, 400, ensure_ascii=True)
                    return self._send(body, 'application/json; charset=utf-8')

                return self._route_bag(url, apply)
            if path == '/version':
                body = json.dumps({'version': assets_version()}).encode('utf-8')
                return self._send(body, 'application/json; charset=utf-8')
            if path == '/meta':
                return self._route_bag(url, lambda viewer: self._send(
                    json.dumps(self._meta_json(viewer)).encode('utf-8'),
                    'application/json; charset=utf-8'))
            if path == '/bags':
                # Диагностика каталога: что живо сейчас, сколько открытий и вытеснений.
                return self._json(bags.status(), 200, ensure_ascii=True)
            if path == '/profile':
                q = parse_qs(url.query)
                name = (q.get('bag', [None])[0]) or bags.default_name
                if not bags.has(name):
                    return self._error_json(
                        KeyError(f'unknown bag {name!r}: доступны '
                                 f'{", ".join(bags.names)}'), 404)
                return self._json({'bag': name,
                                   'profile': bags.profiles.get(name) or None})
            if path == '/track':
                q = parse_qs(url.query)
                try:
                    frame = int(q.get('frame', ['0'])[0])
                except ValueError:
                    frame = 0

                def send_track(viewer):
                    index = max(0, min(frame, viewer.total - 1))
                    try:
                        body = viewer.track.json_bytes(viewer.bag_name, index,
                                                       viewer.db_path)
                    except ImportError as exc:   # модуль не найден -- фича, не падение
                        return self._error_json(exc, 503)
                    except Exception as exc:  # noqa: BLE001
                        return self._error_json(f'build_track: {exc!r}', 500)
                    return self._send(body, 'application/json; charset=utf-8')

                return self._route_bag(url, send_track)
            if path == '/objects':
                q = parse_qs(url.query)
                try:
                    frame = int(q.get('idx', q.get('frame', ['0']))[0])
                except ValueError:
                    frame = 0

                def send_objects(viewer):
                    index = max(0, min(frame, viewer.total - 1))
                    try:
                        import objects.to_viewer as to_viewer
                    except ImportError as exc:   # модуль не найден -- фича, не падение
                        return self._error_json(exc, 503)
                    # Ось кадра собирает `track_geometry.build_track`, и она же --
                    # единственная причина, по которой этот маршрут отвечает 500:
                    # сборка может сорваться на ВРЕМЕННО нерабочем файле (его
                    # правит параллельный редактор) или на чтении записи. Одна
                    # повторная попытка через паузу: 500 на маршруте читается как
                    # «объекты из замеров сломаны», хотя это разовый сбой сборки.
                    last = None
                    for attempt in range(2):
                        try:
                            payload = to_viewer.build_all(viewer.bag_name, index,
                                                          to_viewer.load_specs())
                            break
                        except Exception as exc:     # noqa: BLE001
                            last = exc
                            if attempt == 0:
                                time.sleep(1.5)
                    else:
                        return self._error_json(f'objects: {last!r}', 500)
                    return self._json(jsonable(payload))

                return self._route_bag(url, send_objects)
            if path == '/motion':
                # Ход машины по записи (м/кадр): та же математика, что в
                # check_motion.py, но с кэшем по записи -- панель разметки спрашивает
                # его на каждую смену записи, а сам замер стоит секунды.
                def send_motion(viewer):
                    try:
                        labeling = self._labeling()
                        motion = self._motion_of(labeling, viewer)
                    except Exception as exc:  # noqa: BLE001
                        return self._error_json(f'motion: {exc!r}', 500)
                    return self._json({'bag': viewer.bag_name, 'motion': motion})

                return self._route_bag(url, send_motion)
            if path == '/label/poses':
                # Клип запечённой позы целиком (все моменты) одним ответом: клиент
                # выбирает момент по времени САМ, той же формулой, что `labeling`
                # (`pose_moment` в web/labellayer.js), и во время воспроизведения
                # не ходит на сервер за каждым кадром позы. Двоичные массивы --
                # base64: float32 вершины (T*N*3) и uint32 лица (M*3), порядок
                # «момент за моментом», как в `frames.npz`.
                q = parse_qs(url.query)

                def send_poses(viewer):
                    try:
                        labeling = self._labeling()
                    except Exception as exc:  # noqa: BLE001
                        return self._error_json(exc, 503)
                    cls = str(q.get('class', ['person'])[0])
                    pose = str(q.get('pose', ['standing'])[0])
                    clip = labeling.pose_clip(cls, pose)
                    if clip is None:
                        return self._json({
                            'ok': False, 'class': cls, 'pose': pose,
                            'error': (f'нет запечённой позы для класса {cls!r} и позы '
                                      f'{pose!r}: меш будет примитивом'),
                        })
                    V = np.ascontiguousarray(clip['V'], dtype='<f4')
                    F = np.ascontiguousarray(clip['F'], dtype='<u4')
                    return self._json({
                        'ok': True, 'class': cls, 'pose': pose,
                        'asset': clip['asset'], 'clip': clip['clip'],
                        'play': clip['play'], 'play_hint': clip['play_hint'],
                        'samples': int(clip['samples']),
                        'duration_s': float(clip['duration_s']),
                        'period_s': float(clip['period']),
                        'times_s': [round(float(v), 5) for v in clip['times']],
                        'vertex_count': int(V.shape[1]),
                        'face_count': int(F.shape[0]),
                        'height_m': float(clip['height_m']),
                        'file': clip['path'],
                        # Оси уже КАДРА ((x, -z, y) из манифеста): клиенту остаётся
                        # поворот объекта и постановка на опору.
                        'axes': 'frame (x across, y along, z up)',
                        'vertices_b64': base64.b64encode(V.reshape(-1).tobytes()).decode('ascii'),
                        'faces_b64': base64.b64encode(F.reshape(-1).tobytes()).decode('ascii'),
                    }, 200, ensure_ascii=True)

                return self._route_bag(url, send_poses)
            if path == '/frame':
                # Кадр в режиме zones несёт ось СВОЕГО кадра: если геометрии кадра
                # в кэше /track ещё нет, она собирается внутри frame_bytes и
                # переиспользуется /track (иначе рельс красился бы зелёным).
                q = parse_qs(url.query)
                try:
                    idx = int(q.get('idx', ['0'])[0])
                except ValueError:
                    idx = 0

                def send_frame(viewer):
                    mode = q.get('mode', [viewer.mode])[0]
                    if mode not in bag_reader.COLOR_MODES:
                        mode = viewer.mode
                    return self._send(viewer.frame_bytes(idx, mode),
                                      'application/octet-stream')

                return self._route_bag(url, send_frame)
            if path == '/labels':
                # Лёгкий маршрут: только метки зон кадра (12 байт заголовка + n
                # байт uint8), без позиций. Нужен для перекраски уже нарисованного
                # кадра, когда геометрия пути кадра посчитана: метка rail считается
                # по оси ЭТОГО кадра (сборка по требованию внутри RailZone).
                q = parse_qs(url.query)
                try:
                    idx = int(q.get('idx', ['0'])[0])
                except ValueError:
                    idx = 0

                def send_labels(viewer):
                    return self._send(viewer.label_bytes(idx),
                                      'application/octet-stream')

                return self._route_bag(url, send_labels)
            if path == '/favicon.ico':
                return self._send(b'', 'image/x-icon', 204)
            return self._send(b'not found', 'text/plain; charset=utf-8', 404)

        def _post_label(self, url, path):
            """Разметка: кадр -> предложение бокса -> предпросмотр -> запись.

            Логика вся в `labeling.py`, здесь только разбор тела и выбор кадра:
            запись для всех трёх ручек -- та, что просят через `?bag=`.
            """
            try:
                labeling = self._labeling()
            except Exception as exc:  # noqa: BLE001
                return self._error_json(exc, 503)

            if path == '/label/propose':
                def propose(viewer):
                    body = self._body()
                    frame = int(body.get('frame') or 0)
                    click = body.get('click')
                    if not click or len(click) != 3:
                        return self._error_json('click -- три числа (x, y, z) в системе лидара', 400)
                    xyz = self._xyz_at(viewer)(frame)[0]
                    got = labeling.propose(np.asarray(xyz, dtype=np.float64),
                                           [float(v) for v in click],
                                           db_path=viewer.db_path,
                                           frame_index=frame)
                    got['frame'] = max(0, min(frame, viewer.total - 1))
                    got['record'] = viewer.bag_name
                    return self._json(jsonable(got))

                return self._route_bag(url, propose)

            if path == '/label/preview':
                def preview(viewer):
                    body = self._body()
                    frame = int(body.get('frame', body.get('ref_frame')) or 0)
                    cls = str(body.get('class') or 'person')
                    center, size = body.get('center'), body.get('size')
                    if not center or not size or len(center) != 3 or len(size) != 3:
                        return self._error_json('нужны center и size -- по три числа', 400)
                    # `pose_time_s` -- время позы от опорного кадра объекта, с:
                    # панель считает его по кадру (`t = (f - ref)/hz`) и присылает
                    # явно, чтобы нарисованный меш позы был тем же, по которому
                    # посчитана трассировка. `base_z` -- опора (низ объекта).
                    base_z = body.get('base_z')
                    xyz, inten, ring = labeling.frame_points(self._xyz_at(viewer), frame)
                    # Ключ кэша лучей кадра: лучи зависят только от кадра, серия
                    # preview при перетаскивании бокса берёт их из кэша labeling.
                    got = labeling.preview(
                        cls, str(body.get('pose') or 'unknown'),
                        [float(v) for v in center], [float(v) for v in size],
                        float(body.get('yaw') or 0.0), float(body.get('pitch') or 0.0),
                        float(body.get('roll') or 0.0),
                        xyz=xyz, intensity=inten, ring=ring, db_path=viewer.db_path,
                        pose_time_s=float(body.get('pose_time_s') or 0.0),
                        base_z=(None if base_z is None else float(base_z)),
                        frame_key=(viewer.bag_name, int(frame)))
                    got['removed_indices'] = self._hidden_indices(
                        viewer, frame, got.pop('removed_mask'))
                    got['counts']['points_removed_drawn'] = len(got['removed_indices'])
                    got['frame'] = max(0, min(frame, viewer.total - 1))
                    got['record'] = viewer.bag_name
                    return self._json(jsonable(got))

                return self._route_bag(url, preview)

            def save(viewer):
                body = self._body()
                objects = body.get('objects')
                if not isinstance(objects, list) or not objects:
                    return self._error_json('objects -- непустой список объектов', 400)
                # Запись знает сервер, а не клиент: без db_path модель пути
                # искалась бы по пустому имени и уехала бы на fallback-геометрию.
                for obj in objects:
                    if isinstance(obj, dict):
                        obj.setdefault('db_path', viewer.db_path)
                out_dir = str(body.get('out_dir') or labeling.MAIN_LABELS_DIR)
                hz = float(body.get('hz') or self._hz_of(viewer) or labeling.HZ_DEFAULT)
                motion = self._motion_of(labeling, viewer, hz)
                report = labeling.save(viewer.bag_name, objects,
                                       self._xyz_at(viewer), out_dir=out_dir,
                                       hz=hz, motion=motion)
                return self._json(jsonable(report))

            return self._route_bag(url, save)

        def do_POST(self):
            """POST /profile?bag= -- сохранить профиль вида записи в JSON-файл.

            Пишем только по явному действию из панели («Сохранить вид для этой
            записи»), файл читается обратно через GET /profile.
            Маршруты разметки (`/label/propose`, `/label/preview`, `/label/save`)
            уходят в `_post_label`: у них другой контракт, но та же запись по `?bag=`.
            """
            url = urlparse(self.path)
            if url.path in ('/label/propose', '/label/preview', '/label/save'):
                return self._post_label(url, url.path)
            if url.path != '/profile':
                return self._send(b'not found', 'text/plain; charset=utf-8', 404)
            name = (parse_qs(url.query).get('bag', [None])[0]) or bags.default_name
            if not bags.has(name):
                return self._error_json(
                    KeyError(f'unknown bag {name!r}: доступны {", ".join(bags.names)}'),
                    404)
            try:
                length = int(self.headers.get('Content-Length') or 0)
                raw = self.rfile.read(length) if length > 0 else b'{}'
                payload = json.loads(raw.decode('utf-8'))
                if not isinstance(payload, dict):
                    raise ValueError('профиль должен быть объектом JSON')
                # analysis_dist клиент не знает: подставляем своё значение, но
                # только если запись уже живёт (иначе не будим её ради записи).
                live = bags.live_viewer(name)
                extra = {'analysis_dist': live.analysis_dist} if live else None
                saved = bags.profiles.save(name, payload, extra=extra)
            except Exception as exc:  # noqa: BLE001
                return self._error_json(f'{exc!r}', 400)
            self._json({'bag': name, 'profile': saved})

    return Handler


def main():
    parser = argparse.ArgumentParser(
        description='three.js web viewer for rosbag2 .db3 point clouds')
    parser.add_argument('bag_dir', nargs='?', default='for_hackathon/doubleT_obstacle',
                        help='папка одной записи (внутри *.db3) или корень с '
                             'папками записей (например for_hackathon)')
    parser.add_argument('--bag', default=None,
                        help='запись по умолчанию в режиме корня (по умолчанию -- '
                             'первая по алфавиту)')
    parser.add_argument('--max-live-bags', type=int, default=MAX_LIVE_BAGS,
                        help=f'сколько записей держать открытыми одновременно '
                             f'(LRU, по умолчанию {MAX_LIVE_BAGS}); состояние записи '
                             f'тяжёлое -- облако анализа и кэш кадров')
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
    parser.add_argument('--contact-rail', action='store_true',
                        help='вернуть третью нить (контактный рельс из ПТЭ, ось ± 1.45 м) '
                             'в модель рельсов: по умолчанию её нет -- детектор видит '
                             'две нити, а ручная опция убрана из панели')
    parser.add_argument('--client-dir', default=CLIENT_DIR,
                        help='каталог собранного React-клиента (app/client/dist); '
                             'если его нет -- отдаётся старый клиент из web/')
    parser.add_argument('--track-cache', default=TRACK_CACHE_DIR,
                        help='каталог хранилища готовой геометрии пути '
                             '(<каталог>/<запись>/<кадр>.json.gz, его наполняет '
                             f'precompute_track.py; по умолчанию {TRACK_CACHE_DIR}); '
                             'none/пусто -- выключить хранилище (геометрия кадров '
                             'собирается на месте, как раньше)')
    parser.add_argument('--open', action='store_true',
                        help='open the page in the default browser')
    args = parser.parse_args()

    if not os.path.isdir(args.bag_dir) and not os.path.isfile(args.bag_dir):
        raise SystemExit(f'{args.bag_dir}: ни папки, ни файла')
    Viewer.args_axis = args.axis_x      # совместимость: атрибут класса
    try:
        bags = BagCatalog(args.bag_dir, args, max_live=args.max_live_bags)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc))
    viewer = bags.open_default()        # запись по умолчанию греем сразу, как раньше
    print(f'Reading: {viewer.db_path}')

    url = f'http://{args.host}:{args.port}/'
    print(f'BAGS        : {len(bags.names)} '
          + ('запись' if not bags.multi else 'записей')
          + f' ({", ".join(bags.names)}); по умолчанию {bags.default_name}, '
          + f'профилей вида {len(bags.profiles.all())}, живых не больше '
          + f'{bags.max_live} (LRU)')
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
    tunnel = viewer.tunnel_result
    print(f'TUNNEL      : занято {tunnel["bins_blocked"]} из {tunnel["bins_total"]} бинов '
          f'{[[round(a, 1), round(b, 1)] for a, b in tunnel["spans"]]}; '
          f'полуширина {tunnel["half_width"]:.2f} м, низ {tunnel["z_low"]:.2f} / '
          f'верх {tunnel["z_high"]:.2f} м над УГР, контактный рельс '
          f'{"исключается" if tunnel["exclude_contact"] else "не исключается"}')
    for name, variant in viewer.path['variants'].items():
        print(f'PATH {name:9s}: ' + (
            f'{len(variant["y"])} узлов, x {variant["x"].min():.2f}..{variant["x"].max():.2f}'
            if variant['available'] else f'unavailable ({variant.get("error")})'))
    track_status = viewer.track.status()
    print('TRACK       : ' + (
        f'track_geometry найден, /track отдаёт геометрию пути '
        f'(кэш {viewer.track.max_cache} кадров)'
        if track_status['available']
        else f'track_geometry недоступен ({track_status["error"]}) -- /track ответит 503'))
    # Хранилище precompute: кадр из него отдаётся без сборки, поэтому важно видеть
    # сразу, сколько кадров записи в нём есть (и что дальше сборка на месте).
    store = track_status.get('store')
    if store is not None:
        print(f'TRACK CACHE : {store["dir"]} -- {store["frames"]} кадров по '
              f'{store["bags"]} записям; у «{bags.default_name}» '
              f'{store["own_frames"]} из {viewer.total} '
              + ('(полное покрытие: /track и метка rail без сборки)'
                 if store['own_frames'] >= viewer.total else
                 '(неполное: недостающие кадры собираются на месте, один раз за '
                 'сессию предупреждение в лог)'))
    else:
        print('TRACK CACHE : выключено (--track-cache none) -- геометрия кадров '
              'собирается на месте, как раньше')
    print('CLIENT      : рабочий клиент из web/ (страница /); '
          'собранный React-клиент больше не раздаётся')
    # Клиент подключает слой объектов пути мягко (dynamic import): без этого файла
    # страница работает, но объекты пути не рисуются -- лучше видеть это сразу.
    layer_js = os.path.join(WEB_DIR, 'track3d.js')
    print(f'LAYER       : web/track3d.js '
          + (f'найден ({os.path.getsize(layer_js)} Б), '
             f'/track3d.js отдаётся, объекты пути будут'
             if os.path.isfile(layer_js) else
             'НЕ найден -- объекты пути не нарисуются, облако точек и «лента '
             'рельсов» продолжат работать'))
    print(f'Serving {"1 запись" if not bags.multi else str(len(bags.names)) + " записей"} '
          f'({viewer.total} кадров в «{bags.default_name}») at {url}   '
          f'(Ctrl+C to stop)')
    # Источник метки rail проверяем сразу: у записи по умолчанию он уже посчитан
    # (нити берутся из того же кэша, что и /track), а на кадре видно время.
    rzone = viewer.rail_zone
    ref_lines = rzone.lines(viewer.path_frame)     # только peek кэша /track
    if ref_lines:
        nsides = sum(1 for i in ref_lines if i.get('nodes') is not None)
        nnodes = sum(len(i['nodes']) for i in ref_lines if i.get('nodes') is not None)
        src = next((i['node_source'] for i in ref_lines if i.get('node_source')), None)
        print(f'RAIL ZONE   : метка rail -- по оси /track '
              f'({len(ref_lines)} нити опорного кадра {viewer.path_frame}); '
              f'ось по полилинии у {nsides} нитей, узлов {nnodes} (источник: {src}); '
              f'±{rzone.half_x:.3f} м по X, {rzone.head_below:.3f} м вниз от коронки '
              f'головки / {rzone.head_above:.3f} м вверх; ниже плоскости пола метки '
              f'rail нет')
    else:
        print(f'RAIL ZONE   : ЗАПАСНОЙ путь -- полоса zones с rail_low = 0.0 '
              f'(ниже пола не красит), причина: {rzone._reason}')
    if bags.multi:
        print(f'  Переключение записей -- в панели, блок «Данные» (или ?bag=<имя>); '
              f'живых держим не больше {bags.max_live}, остальные закрываются.')

    if args.open:
        import webbrowser
        webbrowser.open(url)

    httpd = ViewerServer((args.host, args.port),
                         make_handler(bags, args.client_dir))
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print('\nStopped.')
    finally:
        bags.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())