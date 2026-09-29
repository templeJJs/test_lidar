"""Общие средства замера объектов в записях: кадры, ось пути, поперечные профили.

Соглашения, обязательные для всех замеров объектов (иначе числа разных агентов
несравнимы):

  * система координат записи: сенсор в (0, 0, 0), X поперёк пути, Y вдоль
    (ход вперёд -- ОТРИЦАТЕЛЬНЫЙ Y), Z вверх, метры;
  * поперёк -- ``u = x - axis_x(y)``, смещение ОТ ОСИ ПУТИ, плюс вправо по ходу;
  * вдоль -- ``y`` как есть;
  * по высоте -- либо ``z`` (абсолютный), либо ``z_rel`` = z - ``head_z(y)``
    (УГР на ЭТОМ ``y``), либо ``z_shelf`` = z - ``shelf_z(u, y)`` (полка пола);
  * ось пути и колея НЕ подгоняются заново: они берутся из проверенного
    ``track_geometry.build_track``. Своя подгонка оси у каждого агента даёт свою
    колею, и объекты перестают сходиться между собой;
  * кадры читаются поштучно с маленьким кэшем: кадр -- 170-350 тыс. точек,
    держать десятки кадров в памяти нельзя.

Точка входа для замера: ``Axis`` (ось, колея, УГР, полка) и ``uz()`` (перевод
облака кадра в координаты объекта). Пример:

    import objects.common as C
    ax = C.axis('roundT_doubleT', 100)
    xyz, inten = C.load('roundT_doubleT', 100, max_dist=60)
    u, y, z, z_rel, z_shelf = C.uz(xyz, ax)
"""

from __future__ import annotations

import json
import os
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np

import bag_reader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(ROOT, 'for_hackathon')
# Записи, загруженные через веб-вьюер (POST /upload в web_viewer.py), -- здесь;
# `bag_path` смотрит сюда, когда в DATASET_DIR записи нет.
UPLOADS_DIR = os.path.join(ROOT, 'uploads')
SPECS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'specs')

# Датумы по высоте: имена для поля "z_datum" в спецификации объекта.
DATUM_RAIL_HEAD = 'rail_head'      # над верхом головки ходового рельса (УГР)
DATUM_FLOOR_SHELF = 'floor_shelf'  # над измеренной полкой пола между рельсами
DATUM_ABSOLUTE = 'absolute'        # абсолютный z записи

_LOADERS: dict = {}
_AXES: 'OrderedDict[tuple, Axis]' = OrderedDict()
_AX_CACHE = 8

# Запасной кэш оси на диске. Нужен из-за РЕАЛЬНОЙ ситуации: `track_geometry.py`
# правится параллельно и на минуты остаётся нерабочим (проверено несколько раз:
# `node_origin`, `_link_prev`), а от него зависят и /track, и /objects, и все
# замеры. Кэш хранит ТОЛЬКО ось кадра (десятки КБ на кадр, файлов не больше
# AXIS_CACHE_MAX) -- это не кэш кадров и не кэш мешей, память процесса он не
# раздувает. Запись помечается `axis_from_cache=True`, чтобы геометрия, собранная
# по кэшу, не выдавалась за свежий замер.
AXIS_CACHE_DIR = os.path.join(ROOT, '.cache', 'axis')
AXIS_CACHE_MAX = 200
AXIS_CACHE_READ = os.environ.get('OBJECTS_AXIS_CACHE', '1') not in ('0', 'false', 'no')

# Предел памяти на процесс. Замер объекта столько не требует: кадр -- 160-350
# тыс. точек (единицы МБ), осевые меши `track_geometry` -- сотни МБ на кадр.
# Предел заведён после случая, когда вспомогательная обвязка агента (кэш осей и
# разобранных кадров по всей записи) раздула процесс до ДЕСЯТКОВ гигабайт и
# машина ушла в свап. Сторож останавливает процесс сам.
MEM_LIMIT_MB = float(os.environ.get('OBJECTS_MEM_LIMIT_MB', '1500'))
MEM_FATAL = os.environ.get('OBJECTS_MEM_FATAL', '1') not in ('0', 'false', 'no')
_MEM_THREAD = None


def memory_mb() -> float:
    """RSS текущего процесса в МБ (psutil; без него -- 0.0, проверок не будет)."""
    try:
        import psutil
    except ImportError:
        return 0.0
    try:
        return psutil.Process(os.getpid()).memory_info().rss / (1024.0 * 1024.0)
    except Exception:                                  # noqa: BLE001
        return 0.0


def watch_memory(limit_mb: float = None, fatal: bool = None,
                 interval_s: float = 3.0) -> float:
    """Включить сторож памяти (один раз на процесс). Возвращает текущий RSS в МБ.

    Сторож поднимается автоматически при первом `load()` или `axis()`: пока
    процесс ниже предела -- молчит; выше -- печатает предупреждение с числом;
    выше ДВУХ пределов -- останавливает процесс, чтобы он не утопил машину.
    Отключается переменной окружения `OBJECTS_MEM_LIMIT_MB` (предел) и
    `OBJECTS_MEM_FATAL=0` (не убивать).
    """
    global _MEM_THREAD
    limit = float(MEM_LIMIT_MB if limit_mb is None else limit_mb)
    hard = MEM_FATAL if fatal is None else fatal
    now = memory_mb()
    if _MEM_THREAD is not None or not hard:
        return now
    import threading
    import time

    def loop():
        warned = False
        while True:
            time.sleep(interval_s)
            mb = memory_mb()
            if mb <= 0:
                continue
            if mb > 2.0 * limit:
                print(f'\n!! ПАМЯТЬ {mb:.0f} МБ > 2 x предела {limit:.0f} МБ: '
                      f'замер так не делается (кэш кадров/осей по всей записи?). '
                      f'Процесс остановлен.', flush=True)
                os._exit(9)
            if mb > limit and not warned:
                warned = True
                print(f'\n!  ПАМЯТЬ {mb:.0f} МБ > предела {limit:.0f} МБ: '
                      f'уменьши окно (max_dist), не держи кадры и оси пачкой, '
                      f'обрабатывай по одному кадру.', flush=True)

    _MEM_THREAD = threading.Thread(target=loop, daemon=True)
    _MEM_THREAD.start()
    return now


# ------------------------------------------------------------------- каталог

def bag_names() -> list:
    """Имена записей в порядке каталога (каждая -- папка с .db3)."""
    if not os.path.isdir(DATASET_DIR):
        return []
    return sorted(d for d in os.listdir(DATASET_DIR)
                  if os.path.isdir(os.path.join(DATASET_DIR, d)))


def bag_path(name: str) -> str:
    """Путь к .db3 записи. Папка или файл -- принимается и то, и другое."""
    if name.endswith('.db3'):
        return name
    try:
        return bag_reader.find_db3(os.path.join(DATASET_DIR, name))
    except (FileNotFoundError, OSError):
        # Запись, загруженная через веб (web_viewer.py /upload): в DATASET_DIR
        # её нет, она в UPLOADS_DIR.
        return bag_reader.find_db3(os.path.join(UPLOADS_DIR, name))


def frame_count(name: str) -> int:
    """Число кадров PointCloud2 в записи."""
    return len(_loader(name))


def load(name: str, index: int, max_dist: float = 60.0, behind: float = 2.0):
    """Кадр записи, обрезанный по дальности. Возвращает ``(xyz, intensity)``.

    Копия не делается: ``trim`` возвращает срез-маску нового массива, а сам кадр
    остаётся в кэше загрузчика (маленьком -- см. ``_loader``).
    """
    watch_memory()
    frame = _loader(name)[int(index)]
    xyz, intensity, _ring = bag_reader.trim(
        frame.xyz, frame.intensity, None, max_dist=max_dist, behind=behind)
    return xyz, intensity


def load_ring(name: str, index: int, max_dist: float = 60.0, behind: float = 2.0):
    """То же, но с номером луча (``ring``) -- нужен, чтобы отличить железо от стены."""
    frame = _loader(name)[int(index)]
    return bag_reader.trim(frame.xyz, frame.intensity, frame.ring,
                           max_dist=max_dist, behind=behind)


def _loader(name: str) -> bag_reader.BagFrames:
    """Загрузчик записи: кэш 2 кадра, БЕЗ фоновой предзагрузки.

    Предзагрузка тянет кадры вперёд и держит их в памяти пачкой -- для замеров
    это лишние сотни мегабайт (а при обвязке, которая читает всю запись, и
    десятки гигабайт: так и случилось, см. `watch_memory`).
    """
    bag = _LOADERS.get(name)
    if bag is None:
        bag = bag_reader.BagFrames(bag_path(name), cache_size=2, prefetch_ahead=0)
        _LOADERS[name] = bag
    return bag


def spread_frames(name: str, count: int, margin: float = 0.06) -> list:
    """``count`` кадров, равномерно разбросанных по всей записи.

    Замер по одному кадру -- это замер одной точки пути: на разных участках
    (прямая, кривая, у платформы, у гермозатвора) объект выглядит по-разному.
    Первые и последние кадры отступаются на ``margin`` -- там лента ещё не
    разогналась.
    """
    total = frame_count(name)
    if total <= 0:
        return []
    lo = int(round(margin * total))
    hi = max(lo, total - 1 - int(round(margin * total)))
    if count <= 1:
        return [lo]
    idx = np.linspace(lo, hi, int(count))
    return sorted({int(round(v)) for v in idx})


# ----------------------------------------------------------------------- ось

@dataclass
class Axis:
    """Ось пути и датумы кадра: колея, УГР, полка пола.

    ``head_m``/``head_c`` -- измеренная линия верха головки ``z = m*y + c``; она
    важна, потому что уклон ленты 0.5-1.9 % даёт на 30 м окна 0.15-0.6 м, и без
    этой линии объект на дальнем конце «повисает» относительно рельса.
    """

    name: str
    index: int
    s: float                       # ось пары: x = s*y + i
    i: float
    gauge_axis_m: float            # колея ось-ось (по центрам головок)
    gauge_inner_m: float           # колея по внутренним граням
    half_gauge_m: float
    head_m: float
    head_c: float
    head_m_side: dict = field(default_factory=dict)
    head_c_side: dict = field(default_factory=dict)
    shelf_z: float = float('nan')  # полка пола на u=0, y=y_ref (из служебного профиля)
    shelf_slope_u: float = 0.0
    shelf_slope_y: float = 0.0
    shelf_y_ref: float = 0.0
    shelf_window_y: tuple = ()
    shelf_m: float = float('nan')   # ИЗМЕРЕННАЯ линия полки: z = m*y + c (u=0)
    shelf_c: float = float('nan')
    shelf_source: str = 'not_measured'
    trough: dict = field(default_factory=dict)
    head_top_z: dict = field(default_factory=dict)   # скаляр из build_track -- сверка
    head_above_adjacent: dict = field(default_factory=dict)  # независимый замер высоты
    y_visible: tuple = ()
    floor_plane: dict = field(default_factory=dict)  # НЕНАДЁЖНА (см. notes)
    notes: tuple = ()
    head_source: str = 'measured'
    axis_from_cache: bool = False   # ось взята из запасного кэша (не свежий замер)
    axis_cache_source: int = -1     # из какого кадра взята, если из соседнего

    # --- координаты
    def axis_x(self, y):
        """X оси пути на этом ``y``."""
        return self.s * np.asarray(y, dtype=np.float64) + self.i

    def u(self, xyz):
        """Поперечное смещение от оси пути (плюс вправо по ходу)."""
        xyz = np.asarray(xyz)
        return xyz[:, 0] - self.axis_x(xyz[:, 1])

    def rail_center_x(self, y, side: int):
        """X центра головки нити: ``side`` = -1 левая (x<0), +1 правая."""
        return self.axis_x(y) + side * self.half_gauge_m

    def head_z(self, y):
        """Z верха головки (УГР) на этом ``y`` -- линия по кадру."""
        return self.head_m * np.asarray(y, dtype=np.float64) + self.head_c

    def shelf_z_at(self, u, y):
        """Z полки пола на ``(u, y)``.

        По ``y`` -- ИЗМЕРЕННАЯ линия полки (``shelf_m``/``shelf_c``): служебный
        ``shelf_slope_y`` из профиля шумный (окно подгонки 10 м даёт ошибку
        наклона до 0.5 %, а на 25 м это 12 см -- проверено на roundT_doubleT,
        где он выходил -0.41 % против измеренных +0.08 % у пола и рельса).
        Поперёк -- служебный ``shelf_slope_u``: он измерен в плотной зоне и
        держит полку/лоток.
        """
        u = np.asarray(u, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if np.isfinite(self.shelf_m):
            base = self.shelf_m * y + self.shelf_c
        else:
            base = self.shelf_z + self.shelf_slope_y * (y - self.shelf_y_ref)
        return base + self.shelf_slope_u * u

    def z_rel_head(self, z, y):
        return np.asarray(z, dtype=np.float64) - self.head_z(y)

    def z_rel_shelf(self, z, u, y):
        return np.asarray(z, dtype=np.float64) - self.shelf_z_at(u, y)

    def head_line_check_mm(self) -> float:
        """Расхождение линии верха головки со скаляром build_track -- грубая сверка."""
        y_ref = 0.5 * (self.y_visible[0] + self.y_visible[1]) if self.y_visible else 0.0
        if not self.head_top_z:
            return float('nan')
        got = float(self.head_z(y_ref))
        want = float(np.mean(list(self.head_top_z.values())))
        return 1000.0 * (got - want)

    def as_dict(self) -> dict:
        return {'bag': self.name, 'frame': self.index, 's': self.s, 'i': self.i,
                'gauge_axis_mm': 1000.0 * self.gauge_axis_m, 'half_gauge_m': self.half_gauge_m,
                'head_m': self.head_m, 'head_c': self.head_c,
                'head_slope_pct': 100.0 * self.head_m,
                'shelf_z': self.shelf_z, 'shelf_slope_u': self.shelf_slope_u,
                'shelf_slope_y': self.shelf_slope_y,
                'y_visible': list(self.y_visible),
                'head_source': self.head_source,
                'axis_from_cache': bool(self.axis_from_cache),
                'axis_cache_source': (self.axis_cache_source
                                      if self.axis_cache_source >= 0 else None),
                'head_line_vs_scalar_mm': self.head_line_check_mm()}


def _axis_cache_path(name: str, index: int) -> str:
    return os.path.join(AXIS_CACHE_DIR, f'{name}_{int(index)}.pkl')


def _axis_cache_put(ax: 'Axis') -> None:
    """Положить ось в запасной кэш (best-effort, с ограничением числа файлов)."""
    if not AXIS_CACHE_READ:
        return
    try:
        import pickle
        os.makedirs(AXIS_CACHE_DIR, exist_ok=True)
        path = _axis_cache_path(ax.name, ax.index)
        tmp = path + '.tmp'
        with open(tmp, 'wb') as fh:
            pickle.dump(ax, fh, protocol=4)
        os.replace(tmp, path)                    # читатель не увидит половину файла
        files = sorted((os.path.getmtime(f), f) for f in os.listdir(AXIS_CACHE_DIR)
                       if f.endswith('.pkl'))
        for _ts, old in files[:max(0, len(files) - AXIS_CACHE_MAX)]:
            try:
                os.remove(os.path.join(AXIS_CACHE_DIR, old))
            except OSError:
                pass
    except Exception:                            # noqa: BLE001
        pass                                     # кэш не обязателен -- молча пропускаем


def _axis_cache_get(name: str, index: int):
    """Ось из запасного кэша (или None). Помечается axis_from_cache=True."""
    if not AXIS_CACHE_READ:
        return None
    try:
        import pickle
        path = _axis_cache_path(name, index)
        if not os.path.exists(path):
            return None
        with open(path, 'rb') as fh:
            ax = pickle.load(fh)
        ax.axis_from_cache = True
        return ax
    except Exception:                            # noqa: BLE001
        return None


def _axis_cache_get_nearest(name: str, index: int):
    """Ось из кэша БЛИЖАЙШЕГО кадра той же записи (или None).

    Зачем: `track_geometry.py` правится параллельно и на минуты остаётся
    нерабочим, а смотреть объекты надо на любом кадре. Ось -- линии
    (`x = s*y + i`, верх головки, полка), они меняются вдоль пути медленно, и
    ось соседнего кадра даёт ту же геометрию в пределах миллиметров-сантиметров.
    Происхождение помечается (`axis_from_cache`, `axis_cache_source`), чтобы это
    не выдавалось за замер этого кадра.
    """
    if not AXIS_CACHE_READ or not os.path.isdir(AXIS_CACHE_DIR):
        return None
    best = None
    prefix = f'{name}_'
    for f in os.listdir(AXIS_CACHE_DIR):
        if not f.startswith(prefix) or not f.endswith('.pkl'):
            continue
        tail = f[len(prefix):-4]
        if not tail.isdigit():
            continue
        dist = abs(int(tail) - int(index))
        if best is None or dist < best[0]:
            best = (dist, int(tail))
    if best is None:
        return None
    cached = _axis_cache_get(name, best[1])
    if cached is not None:
        cached.axis_cache_source = best[1]
    return cached


def axis(name: str, index: int) -> Axis:
    """Ось и датумы кадра: ``track_geometry.build_track`` + линия УГР по облаку."""
    watch_memory()
    key = (name, int(index))
    cached = _AXES.get(key)
    if cached is not None:
        _AXES.move_to_end(key)
        return cached
    ax = _build_axis(name, int(index))
    _AXES[key] = ax
    while len(_AXES) > _AX_CACHE:
        _AXES.popitem(last=False)
    return ax


def _build_axis(name: str, index: int) -> Axis:
    import time

    import track_geometry

    db = bag_path(name)
    # track_geometry.py -- большой файл, который правится параллельно: в момент
    # сохранения он на секунды может оказаться нерабочим (проверено -- NameError
    # в середине правки). Замер не должен падать из-за чужого сохранения, поэтому
    # пробуем несколько раз, а в ошибке говорим, что делать.
    last = None
    attempts = 4
    for attempt in range(attempts):
        try:
            data = track_geometry.build_track(db, index, draw_sleepers=False)
            break
        except Exception as exc:                      # noqa: BLE001
            last = exc
            if attempt < attempts - 1:
                time.sleep(3.0)
    else:
        # Параллельный редактор может держать файл нерабочим минутами: тогда
        # берём ось из запасного кэша и ЧЕСТНО помечаем это.
        cached = _axis_cache_get(name, index) or _axis_cache_get_nearest(name, index)
        if cached is not None:
            return cached
        raise RuntimeError(f'{name}#{index}: build_track недоступен после {attempts} '
                           f'попыток ({type(last).__name__}: {last}), и оси нет в '
                           f'запасном кэше. Если файл правится другим агентом -- '
                           f'подождать и повторить замер.')
    meta = data.get('meta') or {}
    center = data.get('axis') or {}
    rails = data.get('rails') or []
    if not center or len(rails) < 2:
        raise RuntimeError(f'{name}#{index}: build_track не дал ось пары')

    head_top, head_slope = {}, {}
    above_adjacent = {}
    for r in rails:
        h = r.get('head') or {}
        head_top[r['side']] = float(h.get('top_z', float('nan')))
        head_slope[r['side']] = h.get('y_range')
        val = h.get('height_above_adjacent_m')
        above_adjacent[r['side']] = None if val is None else float(val)

    prof = meta.get('floor_profile') or {}
    floor = data.get('floor') or {}
    ax = Axis(
        name=name, index=index,
        s=float(center['s']), i=float(center['i']),
        gauge_axis_m=float(meta.get('gauge_axis_mm', 1592.0)) / 1000.0,
        gauge_inner_m=float(meta.get('gauge_inner_mm', 1520.0)) / 1000.0,
        half_gauge_m=float(meta.get('gauge_axis_mm', 1592.0)) / 2000.0,
        head_m=float('nan'), head_c=float('nan'),
        head_top_z=head_top,
        head_above_adjacent=above_adjacent,
        y_visible=tuple(sorted(v[0] for v in head_slope.values() if v)),
        floor_plane={k: float(v) for k, v in floor.items() if isinstance(v, (int, float))},
        notes=tuple(meta.get('notes') or ()),
    )
    ax.y_visible = (ax.y_visible[0], max(v[1] for v in head_slope.values()))
    if prof:
        ax.shelf_z = float(prof.get('shelf_z', float('nan')))
        ax.shelf_slope_u = float(prof.get('shelf_slope_u', 0.0) or 0.0)
        ax.shelf_slope_y = float(prof.get('shelf_slope_y', 0.0) or 0.0)
        win = prof.get('window_y') or []
        ax.shelf_window_y = tuple(win)
        ax.shelf_y_ref = float(np.mean(win)) if win else 0.0
        ax.trough = dict(prof.get('trough') or {})

    # Линия верха головки: скаляр из build_track на всю ленту не годится -- уклон
    # ленты на 30 м окна даёт до 0.6 м, и объект «повисает». Мерим линию по облаку.
    xyz, _ = load(name, index, max_dist=60.0, behind=2.0)
    m, c, per_side = _head_line(xyz, ax)
    if np.isfinite(m) and np.isfinite(c):
        ax.head_m, ax.head_c = m, c
        ax.head_source = 'measured_line'
        for side, (ms, cs) in per_side.items():
            ax.head_m_side[side], ax.head_c_side[side] = ms, cs
    else:
        scalar = float(np.nanmean(list(head_top.values())))
        ax.head_m, ax.head_c = 0.0, scalar
        ax.head_source = 'build_track_scalar'
    # Линия полки: тот же приём, что и для головки. Рельс лежит на полу, поэтому
    # две линии обязаны быть почти параллельны -- это проверяется в check_common.
    if np.isfinite(ax.shelf_z):
        sm, sc = _shelf_line(xyz, ax)
        if np.isfinite(sm):
            ax.shelf_m, ax.shelf_c, ax.shelf_source = sm, sc, 'measured_line'
    _axis_cache_put(ax)
    return ax


def _shelf_line(xyz, ax: Axis, bin_m: float = 2.0, u_inner: float = 0.58,
                u_outer: float = 0.72, z_half: float = 0.15, min_bins: int = 4,
                mad_k: float = 3.0):
    """Линия ``z = m*y + c`` ПРИЛЕГАЮЩЕЙ поверхности у самой нити.

    Это уровень, от которого отсчитана высота головки (известное измерение:
    0.159..0.186 м). Полосу приходится брать УЗКОЙ, потому что пол поперёк не
    плоский: замерено по всем записям, что уровень пола ступенькой уходит вниз
    от нити к центру пути -- на y = -11.5 м головка стоит выше полосы
    |u| 0.65..0.72 на 0.15-0.16 м, выше полосы 0.55..0.65 на 0.20-0.21 м,
    выше 0.45..0.55 на 0.23-0.27 м, а в центре (у лотка) ещё ниже. Широкая
    полоса 0.40..0.70 захватывает этот спуск и занижает уровень на 40-60 мм
    против независимого замера `track_geometry` (height_above_adjacent_m).

    Верхняя граница 0.72 -- край подрельсовой площадки: дальше начинается
    бок нити (0.72..0.90 даёт всего 0.03-0.07 м до головки, то есть это железо,
    а не пол). Окно по высоте (``z_half``) -- вокруг служебной полки профиля.
    """
    if not len(xyz) or not np.isfinite(ax.shelf_z):
        return float('nan'), float('nan')
    y = xyz[:, 1].astype(np.float64)
    u = ax.u(xyz)
    z = xyz[:, 2].astype(np.float64)
    z_ref = ax.shelf_z + ax.shelf_slope_y * (y - ax.shelf_y_ref) + ax.shelf_slope_u * u
    au = np.abs(u)
    mask = (au >= u_inner) & (au <= u_outer) & (np.abs(z - z_ref) <= z_half)
    if mask.sum() < 100:
        return float('nan'), float('nan')
    ys, zs = y[mask], z[mask]
    # по каждой стороне отдельно, затем обе линии складываются: пол может быть с
    # поперечным уклоном, и среднее двух сторон -- честный уровень «полки»
    lines = []
    for sign in (-1.0, 1.0):
        sel_side = np.sign(u[mask]) == sign
        if sel_side.sum() < 50:
            continue
        idx = np.floor((ys[sel_side] - ys.min()) / bin_m).astype(np.int64)
        by, bz = [], []
        for b in np.unique(idx):
            sel = idx == b
            if sel.sum() < 20:
                continue
            by.append(float(np.median(ys[sel_side][sel])))
            bz.append(float(np.median(zs[sel_side][sel])))
        if len(by) >= min_bins:
            lines.append(_robust_line(np.asarray(by), np.asarray(bz), mad_k=mad_k))
    lines = [ln for ln in lines if np.isfinite(ln[0])]
    if not lines:
        return float('nan'), float('nan')
    return (float(np.mean([ln[0] for ln in lines])),
            float(np.mean([ln[1] for ln in lines])))


def _head_line(xyz, ax: Axis, bin_m: float = 1.0, u_half: float = 0.20,
               min_bins: int = 4, mad_k: float = 3.0,
               lo_above_shelf: float = 0.02, hi_above_shelf: float = 0.45):
    """Линия ``z = m*y + c`` верха головки каждой нити, по верхам Y-бинов.

    Полоса нити (``u_half`` вокруг центра головки) сама по себе НЕ определяет
    верх головки: в круглом туннеле над ней проходит свод, и «самое высокое в
    полосе» оказывается потолком (проверено: уклон выходил -0.16, высота над
    полкой 3.4 м). Поэтому окно по высоте задаётся ФИЗИЧЕСКИ -- от измеренной
    полки пола: головка лежит в ``lo_above_shelf..hi_above_shelf`` над ней.
    Полка несёт уклон ленты (``shelf_slope_y``), поэтому окно едет вдоль пути
    вместе с рельсом.

    В бине берётся 99-й перцентиль: площадка головки плоская, одиночные
    выбросы вверх (пыль, искры, отражение) отсекаются, а смещение вниз от
    истинной коронки минимально (95-й перцентиль давал -20 мм и уводил датум).
    """
    if not len(xyz):
        return float('nan'), float('nan'), {}
    y = xyz[:, 1].astype(np.float64)
    u = ax.u(xyz)
    z = xyz[:, 2].astype(np.float64)
    shelf = ax.shelf_z_at(u, y)
    if not np.isfinite(shelf).all():
        shelf = np.full(len(y), np.nanmean(list(ax.head_top_z.values())))
    z_above = z - shelf
    per_side, lines = {}, []
    for side in (-1, 1):
        mask = np.abs(u - side * ax.half_gauge_m) <= u_half
        mask &= (z_above >= lo_above_shelf) & (z_above <= hi_above_shelf)
        if mask.sum() < 50:
            continue
        ys, zs = y[mask], z[mask]
        idx = np.floor((ys - ys.min()) / bin_m).astype(np.int64)
        by, bz = [], []
        for b in np.unique(idx):
            sel = idx == b
            if sel.sum() < 20:
                continue
            by.append(float(np.median(ys[sel])))
            bz.append(float(np.percentile(zs[sel], 99.0)))
        if len(by) < min_bins:
            continue
        m, c = _robust_line(np.asarray(by), np.asarray(bz), mad_k=mad_k)
        if np.isfinite(m):
            per_side[side] = (m, c)
            lines.append((m, c))
    if not lines:
        return float('nan'), float('nan'), per_side
    return (float(np.mean([l[0] for l in lines])),
            float(np.mean([l[1] for l in lines])), per_side)


def _robust_line(x, y, mad_k: float = 3.0, iters: int = 3):
    """МНК-прямая с отбраковкой по MAD. Возвращает ``(m, c)``."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    keep = np.ones(len(x), dtype=bool)
    m = c = float('nan')
    for _ in range(max(1, iters)):
        if keep.sum() < 3:
            break
        m, c = np.polyfit(x[keep], y[keep], 1)
        res = y - (m * x + c)
        mad = np.median(np.abs(res - np.median(res)))
        if not np.isfinite(mad) or mad <= 0.0:
            break
        sigma = 1.4826 * mad
        nxt = np.abs(res - np.median(res)) <= mad_k * sigma
        if nxt.sum() < 3 or np.array_equal(nxt, keep):
            break
        keep = nxt
    return float(m), float(c)


# ------------------------------------------------------------------ перевод

def az(*args):
    """Совместимость: ``uz`` -- актуальное имя (см. ниже)."""
    return uz(*args)


def uz(xyz, ax: Axis, with_shelf: bool = True):
    """Облако кадра в координаты объекта: ``(u, y, z, z_rel, z_shelf)``.

    ``z_rel`` -- над верхом головки ходового рельса на том же ``y`` (УГР),
    ``z_shelf`` -- над измеренной полкой пола. Оба датума отдаются всегда:
    первый годится для всего, что стоит на пути, второй -- для напольного
    оборудования и стен.
    """
    xyz = np.asarray(xyz)
    x = xyz[:, 0].astype(np.float64)
    y = xyz[:, 1].astype(np.float64)
    z = xyz[:, 2].astype(np.float64)
    u = x - ax.axis_x(y)
    z_rel = z - ax.head_z(y)
    if not with_shelf:
        return u, y, z, z_rel, None
    return u, y, z, z_rel, z - ax.shelf_z_at(u, y)


def in_band(u, z, u_lo, u_hi, z_lo, z_hi):
    """Маска полосы: ``u`` в [u_lo, u_hi] и ``z`` (любой датум) в [z_lo, z_hi]."""
    return (np.asarray(u) >= u_lo) & (np.asarray(u) <= u_hi) & \
           (np.asarray(z) >= z_lo) & (np.asarray(z) <= z_hi)


def y_window(y, y_lo, y_hi):
    y = np.asarray(y)
    return (y >= y_lo) & (y <= y_hi)


# ---------------------------------------------------------------- статистика

def robust(v) -> dict:
    """Медиана, перцентили, размах, счёт -- единый вид статистики для отчётов."""
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v)]
    if not len(v):
        return {'n': 0}
    return {'n': int(len(v)), 'median': float(np.median(v)),
            'p05': float(np.percentile(v, 5.0)), 'p95': float(np.percentile(v, 95.0)),
            'min': float(v.min()), 'max': float(v.max()),
            'std': float(v.std())}


def profile_u(u, z, u_lo: float, u_hi: float, step: float = 0.05,
              stat='median', weights=None) -> dict:
    """Поперечный профиль: по бинам ``u`` -- статистика ``z``.

    Возвращает массивы ``u_center``, ``z`` (``stat``), ``p05``, ``p95``, ``n``.
    Пустые бины -- ``nan`` (профиль не досыпается, отсутствие точек важно).
    """
    u = np.asarray(u, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    edges = np.arange(u_lo, u_hi + 1e-9, step)
    if len(edges) < 2:
        raise ValueError('profile_u: пустой диапазон u')
    idx = np.floor((u - u_lo) / step).astype(np.int64)
    keep = (idx >= 0) & (idx < len(edges) - 1)
    idx, zz = idx[keep], z[keep]
    ww = None if weights is None else np.asarray(weights, dtype=np.float64)[keep]
    centers = 0.5 * (edges[:-1] + edges[1:])
    med = np.full(len(centers), np.nan)
    p05 = np.full(len(centers), np.nan)
    p95 = np.full(len(centers), np.nan)
    cnt = np.zeros(len(centers), dtype=np.int64)
    for b in np.unique(idx):
        sel = idx == b
        cnt[b] = int(sel.sum())
        zv = zz[sel]
        if ww is not None:
            med[b] = _weighted_median(zv, ww[sel])
        elif stat == 'median':
            med[b] = float(np.median(zv))
        elif stat == 'max':
            med[b] = float(np.percentile(zv, 99.0))
        elif stat == 'min':
            med[b] = float(np.percentile(zv, 1.0))
        elif stat == 'mean':
            med[b] = float(zv.mean())
        else:
            raise ValueError(f'profile_u: неизвестная статистика {stat!r}')
        p05[b] = float(np.percentile(zv, 5.0))
        p95[b] = float(np.percentile(zv, 95.0))
    return {'u': centers, 'z': med, 'p05': p05, 'p95': p95, 'n': cnt,
            'step_m': float(step), 'stat': stat}


def _weighted_median(v, w):
    order = np.argsort(v)
    v, w = v[order], w[order]
    cum = np.cumsum(w)
    if cum[-1] <= 0:
        return float('nan')
    return float(v[np.searchsorted(cum, 0.5 * cum[-1])])


def periodicity(z, y, spacing_lo: float = 0.20, spacing_hi: float = 1.50,
                min_cycles: int = 6) -> dict:
    """Периодика вдоль пути: лучший шаг и амплитуда по сетке шагов.

    Прореженный ряд ``(y, z)`` приводится к равномерному шагу 0.02 м, из него
    вычитается прямая (уклон ленты), дальше сканируется период: для каждого
    кандидата ``p`` берётся проекция на sin/cos с этим периодом (подгонка фазы и
    амплитуды). Отдаётся амплитуда в мм, число периодов в окне и «шумовой пол» --
    амплитуда самой сильной гармоники на соседних (нерезонансных) периодах.

    Это ЧЕСТНАЯ проверка «есть ли периодика»: реальная шпала даёт амплитуду
    на порядок выше шумового пола, а не «похоже на правду».
    """
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    keep = np.isfinite(y) & np.isfinite(z)
    y, z = y[keep], z[keep]
    if len(y) < 20:
        return {'ok': False, 'why': 'слишком мало точек'}
    order = np.argsort(y)
    y, z = y[order], z[order]
    step = 0.02
    grid = np.arange(y[0], y[-1], step)
    if len(grid) < 32:
        return {'ok': False, 'why': 'короткое окно'}
    zg = np.interp(grid, y, z)
    # тренд (уклон ленты) убираем прямой, иначе низкие периоды «ловят» наклон
    m, c = np.polyfit(grid, zg, 1)
    resid = zg - (m * grid + c)
    span = grid[-1] - grid[0]
    best_amp_m = -1.0
    best = {'amp_mm': 0.0, 'spacing_m': float('nan'), 'cycles': 0.0, 'phase_rad': 0.0}
    amps = []
    grid_p = np.arange(spacing_lo, spacing_hi, 0.005)
    for p in grid_p:
        cycles = span / p
        if cycles < min_cycles:
            continue
        w = 2.0 * np.pi / p
        s = np.sin(w * grid)
        co = np.cos(w * grid)
        a = 2.0 * np.mean(resid * s)
        b = 2.0 * np.mean(resid * co)
        amp = float(np.hypot(a, b))
        amps.append(amp)
        if amp > best_amp_m:
            best_amp_m = amp
            best = {'amp_mm': amp * 1000.0, 'spacing_m': float(p),
                    'cycles': float(cycles),
                    'phase_rad': float(np.arctan2(b, a))}
    if not amps:
        return {'ok': False, 'why': 'в окне меньше min_cycles периодов'}
    amps = np.asarray(amps)
    # шумовой пол: медиана амплитуд по всем кандидатам -- у случайного ряда она
    # того же порядка, что и максимум; настоящая периодика выбивается в разы
    floor = float(np.median(amps)) * 1000.0
    best['ok'] = True
    best['noise_floor_mm'] = floor
    best['amp_over_floor'] = best['amp_mm'] / floor if floor > 0 else float('inf')
    best['window_m'] = float(span)
    best['n_points'] = int(len(y))
    return best


# ------------------------------------------------------------------- кластеры

def cluster2d(u, z, du: float = 0.05, dz: float = 0.05, min_pts: int = 12,
              u_lo: float = None, u_hi: float = None, z_lo: float = None,
              z_hi: float = None, connectivity: int = 1) -> dict:
    """Связные сгустки в плоскости ``(u, z)`` -- поиск поперечной формы объекта.

    Проекция полосы вдоль пути на ``(u, z)``: выступ, щиток, колонна, кабельный
    лоток видны как плотные связные облачка, стена/свод -- как сплошная полоса.
    Возвращает метки, число кластеров и по каждому: рамку, центр, счёт точек и
    «плотность» -- долю занятых ячеек внутри рамки (1.0 -- сплошной объект,
    меньше -- рыхлое облако, то есть скорее шум).
    """
    from scipy import ndimage

    u = np.asarray(u, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    m = np.ones(len(u), dtype=bool)
    if u_lo is not None:
        m &= u >= u_lo
    if u_hi is not None:
        m &= u <= u_hi
    if z_lo is not None:
        m &= z >= z_lo
    if z_hi is not None:
        m &= z <= z_hi
    if m.sum() < min_pts:
        return {'labels': np.zeros(0, dtype=np.int64), 'mask': m, 'objects': [],
                'why': 'точек меньше min_pts'}
    uu, zz = u[m], z[m]
    iu = np.floor((uu - uu.min()) / du).astype(np.int64)
    iz = np.floor((zz - zz.min()) / dz).astype(np.int64)
    grid = np.zeros((iz.max() + 1, iu.max() + 1), dtype=np.int32)
    np.add.at(grid, (iz, iu), 1)
    struct = (np.ones((3, 3), dtype=bool) if connectivity == 2
              else np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool))
    labels, n = ndimage.label(grid >= 1, structure=struct)
    objects = []
    for lab in range(1, n + 1):
        cells = labels == lab
        ncell = int(cells.sum())
        cnt = int(grid[cells].sum())
        if cnt < min_pts:
            continue
        rows, cols = np.nonzero(cells)
        span_r = int(rows.max() - rows.min() + 1)
        span_c = int(cols.max() - cols.min() + 1)
        box_cells = span_r * span_c
        objects.append({
            'label': int(lab), 'n_points': cnt, 'n_cells': ncell,
            # density -- доля занятых ячеек ВНУТРИ РАМКИ кластера (1.0 -- сплошной
            # объект, меньше -- рыхлое облако, то есть скорее шум). Нормировать на
            # всю сетку нельзя: у мелкой детали на большой сетке выходил ноль при
            # любой форме (дефект, найденный агентом по колоннам).
            'density': float(ncell / box_cells) if box_cells else 0.0,
            # points_per_cell -- точек на ячейку рамки: ЭТО и различает сплошную
            # поверхность от рыхлого облака. Занятость ячеек тут не работает:
            # связный сгусток почти всегда заполняет свою рамку целиком.
            'points_per_cell': float(cnt / box_cells) if box_cells else 0.0,
            'u_span_m': float(span_c * du), 'z_span_m': float(span_r * dz),
            'u_range': [float(uu.min() + cols.min() * du), float(uu.min() + (cols.max() + 1) * du)],
            'z_range': [float(zz.min() + rows.min() * dz), float(zz.min() + (rows.max() + 1) * dz)],
            'u_center': float(uu.min() + (cols.mean() + 0.5) * du),
            'z_center': float(zz.min() + (rows.mean() + 0.5) * dz),
            'cells': ncell})
    objects.sort(key=lambda o: -o['n_points'])
    lab_pts = np.zeros(len(uu), dtype=np.int64)
    lab_pts[:] = labels[iz, iu]
    return {'labels': lab_pts, 'mask': m, 'objects': objects,
            'du': du, 'dz': dz, 'u0': float(uu.min()), 'z0': float(zz.min())}


# ------------------------------------------------------------------ запись

def spec_write(slug: str, spec: dict, keep_history: bool = False) -> str:
    """Записать спецификацию объекта в ``objects/specs/<slug>.json``.

    ``keep_history=True`` дописывает запись в ``<slug>.history.jsonl`` -- чтобы
    было видно, как менялись числа от прогона к прогону (важно: замер по одному
    кадру и по всем участкам даёт разные результаты, и это должно быть видно).
    """
    os.makedirs(SPECS_DIR, exist_ok=True)
    path = os.path.join(SPECS_DIR, f'{slug}.json')
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(spec, fh, ensure_ascii=False, indent=2, sort_keys=False)
    if keep_history:
        with open(os.path.join(SPECS_DIR, f'{slug}.history.jsonl'), 'a',
                  encoding='utf-8') as fh:
            fh.write(json.dumps(spec, ensure_ascii=False) + '\n')
    return path


def jnum(obj):
    """Привести numpy-числа/массивы к обычным -- для json без сюрпризов."""
    if isinstance(obj, dict):
        return {k: jnum(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jnum(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [jnum(v) for v in obj.tolist()]
    if isinstance(obj, (np.floating, np.integer)):
        val = obj.item()
        return None if isinstance(val, float) and not np.isfinite(val) else val
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def close_all():
    """Закрыть загрузчики (в скриптах -- в конце, чтобы не держать соединения)."""
    for bag in _LOADERS.values():
        bag.close()
    _LOADERS.clear()
    _AXES.clear()