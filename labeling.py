"""Разметка объектов на пути: замер хода, предложение бокса, трассировка, запись.

Первая часть инструмента разметки. Человек ставит объект мышью (или просит
систему предложить габарит по клику), точки объекта считаются ТРАССИРОВКОЙ
реальных лучей кадра по мешу-примитиву, а разметка ложится в `labels/` в формате,
под который обучается модель. Маршруты `web_viewer.py` (`/motion`,
`/label/propose`, `/label/preview`, `/label/save`) -- тонкие: вся геометрия здесь.

Что внутри
----------
* `measure_motion` -- ход машины по записи. Тот же замер, что в `check_motion.py`
  (совмещение профиля стены между кадрами), с кэшем по записи: `/motion` ходит
  на каждую смену записи в панели, а сам замер стоит секунды.
* `PathFrame` -- путевая система КАДРА: ось (`u = 0`) и УГР (`h = 0`) в
  координатах кадра. Ось -- полилиния модели пути (`detector/track_models.json`)
  плюс доворот самого кадра (`detector.core.axis_pose`). Это важно: `u`/`h`
  объекта обязаны считаться в ТОЙ ЖЕ системе, что `u`/`h` детектора, иначе
  «детектор подтвердил» проверялось бы в другой системе координат.
* `propose` -- кластер вокруг луча клика -> бокс (центр, размеры, поворот по оси
  пути) + статус автоматики: `detector` (детектор видит объект в этом габарите),
  `cluster` (кластер есть, детектор не подтвердил), `manual` (не предложено).
* `trace_object` -- точки, которые УВИДЕЛ БЫ лидар: реальные лучи кадра
  (представитель луча = ближайший возврат), примитив из `sim/objects.py`
  (`_person` = цилиндр + сфера, `_suitcase` = бокс) и окклюзия по дальности
  ближайшего возврата -- как в `validation/synth.py`, только вместо AABB меш.
* `save` -- запись `labels/`: `index.jsonl`, карточка `<record>/<id>.json`,
  точки `<record>/<id>.npz` (локальная система объекта), `manifest.json`,
  `README.md`; в карточку и манифест идёт ревизия (md5 `detector/track_models.json`,
  md5 `track_geometry.py`, дата).

Координаты (как везде в проекте): X поперёк, Y вдоль (вперёд это -Y), Z вверх.
Путевые: `u = X - ось(y)` поперёк пути, `h = Z - УГР(y)` над уровнем головки
рельса, вдоль пути -- `along_m = -y` (растёт вперёд).

Никакой сборки геометрии пути здесь нет: ось берётся из готовой модели, кадр --
через параметр `xyz_at(frame) -> (xyz, intensity[, ring])`, чтобы запись не
открывалась второй раз (вьюер держит её живой).
"""
from __future__ import annotations

import datetime
import hashlib
import json
import math
import os
import tempfile

import numpy as np

import zones
from validation import synth

# ---------------------------------------------------------------- константы

HZ_DEFAULT = 10.0                       # частота кадров записей (замерено 9.84..10.05)
CLASSES = ('person', 'equipment', 'other')
POSES = ('standing', 'walking', 'running', 'lying', 'fallen', 'unknown')
DEFAULT_POSE = {'person': 'standing', 'equipment': 'unknown', 'other': 'unknown'}
AUTOMATION = ('detector', 'cluster', 'manual')

MAIN_LABELS_DIR = 'labels'
RECORD_README = 'README.md'

# Пары кадров замера хода -- те же, что у `check_motion.py` при fa=100, fb=110,
# step=20 (он печатает сдвиг первой пары и медиану м/кадр по всем).
MOTION_PAIRS = ((100, 110), (120, 130), (140, 150), (160, 170))
MOTION_SPAN = 10                        # кадров в паре (как fb - fa)
MOTION_SIDE = -1                        # профиль левой стены

# Полоса объекта вокруг луча клика: отметает пол/рельс/свод, но оставляет тело.
BAND_H_LOW_M = 0.12                     # над ПОЛОМ (не над УГР): 0.16 м -- высота рельса
BAND_H_HIGH_M = 2.60                    # выше этой высоты -- свод, лотки, кабели
BAND_U_HALF_M = 3.0                     # поперёк от клика
WINDOW_Y_HALF_M = 3.0                   # вдоль от клика: длинную структуру не берём
CLUSTER_CELL = (0.30, 0.15, 0.15)        # воксель (вдоль, поперёк, по высоте), м
SEED_R_M = 0.45                         # радиус семени вокруг клика, м
PERCENTILE_LO, PERCENTILE_HI = 2.0, 98.0
SIZE_MIN = (0.30, 0.25, 0.20)           # минимумы (ширина, высота, длина), м
SIZE_MAX = (2.50, 2.80, 3.00)           # максимумы: длиннее -- это стена/платформа
CONFIRM_TOL_Y_M = 1.5                   # допуск «детектор нашёл тот же объект», м
CONFIRM_TOL_U_M = 1.0

# Замер «подтверждения»: детектор на кадре с внесённым объектом.
DETECT_CACHE = 8

_MOTION_CACHE = {}
_MODEL_CACHE = {}


# ---------------------------------------------------------------- утилиты

def _md5(path):
    try:
        with open(path, 'rb') as fh:
            return hashlib.md5(fh.read()).hexdigest()
    except OSError:
        return None


def root_dir():
    return os.path.dirname(os.path.abspath(__file__))


def revision():
    """Ревизия разметки: версии того, от чего зависят геометрия и детектор."""
    root = root_dir()
    return {
        'track_models_md5': _md5(os.path.join(root, 'detector', 'track_models.json')),
        'track_geometry_md5': _md5(os.path.join(root, 'track_geometry.py')),
        'date': datetime.datetime.now().isoformat(timespec='seconds'),
    }


def _model_for(db_path):
    """Модель пути записи (кэш: таблица снимков читается один раз на процесс)."""
    key = os.path.abspath(db_path)
    hit = _MODEL_CACHE.get(key)
    if hit is None:
        from detector.profiles import model_for_db

        hit = model_for_db(db_path)
        _MODEL_CACHE[key] = hit
    return hit


def _detector_module():
    from detector import DetectorConfig, detect

    return detect, DetectorConfig


# ------------------------------------------------------------------- ход

def motion_pairs(total, span=MOTION_SPAN, pairs=MOTION_PAIRS):
    """Пары (a, b) для замера хода: как в `check_motion.py`, но по длине записи."""
    got = [(a, b) for a, b in pairs if b < total]
    if got:
        return got
    # Короткая запись: 4 пары равномерно по доступным кадрам.
    if total <= span + 2:
        return []
    step = max(span, (total - span - 2) // 4)
    out = []
    a = 2
    while a + span < total and len(out) < 4:
        out.append((a, a + span))
        a += step
    return out


def measure_motion(db_path, xyz_at=None, hz=HZ_DEFAULT, total=None, pairs=None):
    """Ход машины по записи: м/кадр, км/ч, остаток -- тем же замером, что check_motion.

    Положительный ход -- объект в мире ПРИБЛИЖАЕТСЯ к сенсору (сдвиг профиля
    стены в +Y системы лидара): так его и печатает `check_motion.py`
    (0.42..0.82 м/кадр у едущих записей, -1.32 у обратного хода, 0.00 у стоящей).

    `xyz_at(frame)` -- источник кадра; без него профиль считается по своей
    `BagFrames`, как в исходном скрипте.
    """
    import check_motion

    key = (os.path.abspath(db_path), int(total or 0),
           tuple(map(int, pairs[0])) if pairs else (-1, -1))
    hit = _MOTION_CACHE.get(key)
    if hit is not None:
        return hit

    if pairs:
        used_pairs = list(pairs)
    else:
        used_pairs = motion_pairs(int(total)) if total else list(MOTION_PAIRS)
    rates, details = [], []
    for a, b in used_pairs:
        got = check_motion.best_shift(db_path, int(a), int(b), MOTION_SIDE, xyz_at=xyz_at)
        if got is None:
            continue
        rates.append(got[0] / float(b - a))
        details.append({'frames': [int(a), int(b)], 'shift_m': float(got[0]),
                        'm_per_frame': float(got[0] / float(b - a)),
                        'residual_m': float(got[1]), 'bands': int(got[2])})
    if not rates:
        out = {
            'record': os.path.basename(os.path.dirname(os.path.abspath(db_path))),
            'm_per_frame': 0.0, 'kmh': 0.0, 'residual_m': None, 'hz': float(hz),
            'source': 'wall-profile', 'measured': False, 'pairs': [],
            'error': 'не оценить: профиль стены не совместился',
        }
    else:
        median = float(np.median(rates))
        out = {
            'record': os.path.basename(os.path.dirname(os.path.abspath(db_path))),
            'm_per_frame': median,
            'kmh': median * float(hz) * 3.6,
            'residual_m': details[0]['residual_m'],
            'hz': float(hz),
            'source': 'wall-profile',
            'measured': True,
            'pairs': details,
            # Ход считаем замеренным, если остаток осмысленный (иначе это нижняя
            # оценка -- см. шапку check_motion.py, порог 0.05/0.20 м).
            'residual_ok': bool(details[0]['residual_m'] < 0.20),
        }
    _MOTION_CACHE[key] = out
    return out


# ------------------------------------------------------- путевая система

class PathFrame:
    """Путевая система кадра: ось и УГР кадра в координатах самого кадра.

    `u = x - axis_x(y)`, `h = z - ugr_z(y)`. `axis_x` -- полилиния модели
    (каноническая система) плюс доворот кадра (`axis_pose`): ровно та формула,
    которой детектор считает `u` своего облака. Без доворота объект, поставленный
    по модели, уезжает от измеренной оси на метры (замерено 0.9 м на 10 м и 4 м
    на 40 м между кадрами).
    """

    def __init__(self, model, pose=None):
        self.model = model
        self.pose = pose

    def axis_x(self, y):
        y_arr = np.asarray(y, dtype=np.float64)
        x = np.asarray(self.model.axis_x(y_arr), dtype=np.float64)
        if self.pose is not None and self.pose.applied:
            x = x + self.pose.line(y_arr)
        return x

    def ugr_z(self, y):
        return np.asarray(self.model.ugr_z(y), dtype=np.float64)

    def floor_z(self, y):
        """Плоскость пола: УГР минус высота рельса (объект стоит на полу)."""
        return self.ugr_z(y) - zones.RAIL_HEIGHT

    def terms(self, xyz):
        """(y, u, h) для облака (N, 3) в координатах кадра."""
        xyz = np.asarray(xyz, dtype=np.float64)
        y = xyz[:, 1]
        u = xyz[:, 0] - self.axis_x(y)
        h = xyz[:, 2] - self.ugr_z(y)
        return y, u, h

    def to_xyz(self, y, u, h):
        """Точка по путевым координатам в системе кадра."""
        y_arr = np.asarray(y, dtype=np.float64)
        return np.column_stack([self.axis_x(y_arr) + float(u), y_arr,
                                self.ugr_z(y_arr) + float(h)])

    def slope(self, y, dy=0.5):
        """dx_axis/dy -- направление оси в плане (для поворота бокса)."""
        y = float(y)
        return float((self.axis_x(y + dy) - self.axis_x(y - dy)) / (2.0 * dy))

    def grade(self, y, dy=0.5):
        """dz_УГР/dy -- уклон пути в плане (для тангажа бокса)."""
        y = float(y)
        return float((self.ugr_z(y + dy) - self.ugr_z(y - dy)) / (2.0 * dy))


def path_frame(xyz, model=None, db_path=None):
    """Путевая система кадра: модель пути + доворот ЭТОГО кадра."""
    from detector.core import axis_pose

    model = model if model is not None else _model_for(db_path)
    pose = axis_pose(np.asarray(xyz, dtype=np.float64), model)
    return PathFrame(model, pose)


# ------------------------------------------------------------ предложение

def _voxel_adjacency(keys):
    """Список соседей по 26 окрестности для набора вокселей (готовая таблица)."""
    lut = {tuple(int(v) for v in k): i for i, k in enumerate(keys)}
    adj = [()] * len(keys)
    offs = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
            for dz in (-1, 0, 1) if (dx or dy or dz)]
    for i, k in enumerate(keys):
        near = []
        for dx, dy, dz in offs:
            j = lut.get((k[0] + dx, k[1] + dy, k[2] + dz))
            if j is not None:
                near.append(j)
        adj[i] = tuple(near)
    return adj


def _cluster_around(xyz, click, frame: PathFrame, cell=CLUSTER_CELL,
                    seed_r=SEED_R_M, window_y_half=WINDOW_Y_HALF_M):
    """Кластер точек вокруг клика в путевых координатах.

    Отсекаем известные поверхности (пол, рельсы, свод) полосой по высоте над
    ПОЛОМ, клик берём семенем, дальше -- связные (26 соседей) воксели. Именно
    связность, а не радиус: тело человека и стена рядом -- разные компоненты, а
    радиус склеил бы их в один габарит.

    Возвращает (idx точек в xyz, (y, u, hf)) или None.
    """
    y, u, h = frame.terms(xyz)
    hf = h + zones.RAIL_HEIGHT                      # высота над полом
    cy = float(click[1])
    cu = float(click[0] - frame.axis_x(np.asarray([click[1]]))[0])
    cw = float(click[2] - frame.ugr_z(np.asarray([click[1]]))[0]) + zones.RAIL_HEIGHT

    band = ((hf > BAND_H_LOW_M) & (hf < BAND_H_HIGH_M)
            & (np.abs(u - cu) < BAND_U_HALF_M)
            & (np.abs(y - cy) < window_y_half))
    if not band.any():
        return None

    dy, du, dh = cell
    iy = np.floor((y[band] - cy) / dy).astype(np.int64)
    iu = np.floor((u[band] - cu) / du).astype(np.int64)
    ih = np.floor((hf[band] - cw) / dh).astype(np.int64)
    keys, inv = np.unique(np.stack([iy, iu, ih], axis=1), axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    adj = _voxel_adjacency(keys)

    near = (np.abs(y[band] - cy) < seed_r) & (np.abs(u[band] - cu) < seed_r) \
        & (np.abs(hf[band] - cw) < seed_r)
    if not near.any():
        return None
    seeds = np.unique(inv[near])

    seen = np.zeros(len(keys), dtype=bool)
    seen[seeds] = True
    stack = [int(v) for v in seeds]
    while stack:
        i = stack.pop()
        for j in adj[i]:
            if not seen[j]:
                seen[j] = True
                stack.append(j)

    pts = seen[inv]
    idx = np.flatnonzero(band)[pts]
    return idx, (y[idx], u[idx], hf[idx])


def _box_from_cluster(cy, cu, chf, frame):
    """Бокс по кластеру: центр, размеры (ширина, высота, длина), поворот по оси пути."""
    y_lo, y_hi = np.percentile(cy, [PERCENTILE_LO, PERCENTILE_HI])
    u_lo, u_hi = np.percentile(cu, [PERCENTILE_LO, PERCENTILE_HI])
    h_lo, h_hi = np.percentile(chf, [PERCENTILE_LO, PERCENTILE_HI])
    # Низ -- до пола: у человека на 56 м видимых точек ниже 0.2 м нет, а стоять
    # он должен на полу, иначе бокс висит в воздухе.
    h_lo = max(0.0, float(h_lo))
    length = float(np.clip(y_hi - y_lo, SIZE_MIN[2], SIZE_MAX[2]))
    width = float(np.clip(u_hi - u_lo, SIZE_MIN[0], SIZE_MAX[0]))
    height = float(np.clip(h_hi - h_lo, SIZE_MIN[1], SIZE_MAX[1]))
    y_c = float(0.5 * (y_lo + y_hi))
    u_c = float(0.5 * (u_lo + u_hi))
    h_c = float(h_lo + 0.5 * height)
    center = frame.to_xyz(np.asarray([y_c]), u_c, h_c)[0]
    slope = frame.slope(y_c)
    grade = frame.grade(y_c)
    yaw = math.degrees(math.atan2(-slope, 1.0))      # 0 -- длина бокса вдоль пути
    pitch = math.degrees(math.atan2(grade, 1.0))
    return {
        'center': [float(v) for v in center],
        'size': [float(width), float(height), float(length)],
        'yaw': float(yaw), 'pitch': float(pitch), 'roll': 0.0,
        'extents': {'u': [float(u_lo), float(u_hi)], 'h_floor': [float(h_lo), float(h_hi)],
                    'y': [float(y_lo), float(y_hi)]},
    }


def frame_points(xyz_at, frame):
    """(xyz, intensity, ring) кадра через источник `xyz_at`."""
    got = xyz_at(int(frame))
    xyz = np.asarray(got[0], dtype=np.float64)
    inten = np.asarray(got[1], dtype=np.float64) if len(got) > 1 and got[1] is not None else None
    ring = np.asarray(got[2], dtype=np.int64) if len(got) > 2 and got[2] is not None else None
    return xyz, inten, ring


def propose(xyz, click, model=None, db_path=None, frame_index=None, xyz_at=None,
            cfg=None):
    """Предложить бокс по клику: кластер вокруг луча + статус автоматики.

    `xyz` -- облако кадра (N, 3) в системе лидара, `click` -- точка клика в той же
    системе (её даёт ближайшая к лучу точка кадра). Класс не выбирается: здесь
    только геометрия и статус.

    Ответ:
      `automation` -- `detector` (детектор видит объект в предложенном габарите),
                      `cluster` (кластер предложен, детектор не подтвердил),
                      `manual` (не предложено -- ставить руками);
      `center`/`size`/`yaw`/`pitch`/`roll` -- предложенная геометрия;
      `path` -- та же точка в путевых координатах (вдоль, поперёк, над полом);
      `cluster` -- сколько точек и какие размахи дал кластер;
      `detector` -- что нашёл детектор на этом кадре (для панели).
    """
    model = model if model is not None else _model_for(db_path)
    xyz = np.asarray(xyz, dtype=np.float64)
    frame = path_frame(xyz, model)
    out = {
        'ok': False, 'automation': 'manual', 'reason': '',
        'center': [float(v) for v in click], 'size': list(SIZE_MIN),
        'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0,
        'cluster': None, 'detector': None,
        'path': None, 'pose_suggest': None,
    }
    if xyz.shape[0] == 0:
        out['reason'] = 'кадр пуст'
        return out

    got = _cluster_around(xyz, click, frame,
                          cell=(cfg or {}).get('cell', CLUSTER_CELL),
                          seed_r=(cfg or {}).get('seed_r', SEED_R_M))
    if got is None or got[0].size < (cfg or {}).get('min_points', 8):
        out['reason'] = ('вокруг луча нет кластера: в полосе объекта '
                         f'{0 if got is None else got[0].size} точек')
        det = _detect_only(xyz, model, cfg)
        out['detector'] = det
        return out

    idx, (cy, cu, chf) = got
    box = _box_from_cluster(cy, cu, chf, frame)
    out.update({k: box[k] for k in ('center', 'size', 'yaw', 'pitch', 'roll')})
    out['cluster'] = {
        'n_points': int(idx.size),
        'extents': box['extents'],
        'cell': list((cfg or {}).get('cell', CLUSTER_CELL)),
    }
    y_c = float(np.median(cy))
    u_c = float(np.median(cu))
    h_c = float(np.median(chf)) - zones.RAIL_HEIGHT
    out['path'] = {
        'along_m': float(-y_c), 'lateral_m': u_c, 'height_m': h_c,
        'axis_x_m': float(frame.axis_x(np.asarray([y_c]))[0]),
        'ugr_z_m': float(frame.ugr_z(np.asarray([y_c]))[0]),
        'axis_slope': frame.slope(y_c),
    }
    # Класс по размахам -- только подсказка (человек примерно 1.6..1.9 м в высоту
    # и узкий), выбор класса всё равно в панели.
    width, height, _length = out['size']
    out['pose_suggest'] = ('person' if height >= 1.2 and width <= 1.2 else 'equipment')

    # Подтверждение: детектор видит объект ЭТОГО кадра в предложенном габарите.
    best, det = _detect_detail(xyz, model, cfg)
    out['detector'] = det
    # Сравниваем с КЛАСТЕРОМ клика, а не с нулём: у дальнего объекта y большое
    # (человек на кадре 13 doubleT_obstacle -- y=-55.7), и «|y| помещается в
    # допуск» не проходит никогда, хотя детектор нашёл ровно этот объект.
    if best is not None and abs(best[0] - y_c) <= max(CONFIRM_TOL_Y_M, 0.5 * out['size'][2] + 1.0) \
            and abs(best[1] - u_c) <= max(CONFIRM_TOL_U_M, 0.5 * out['size'][0] + 0.6):
        out['ok'] = True
        out['automation'] = 'detector'
        out['detector_match'] = best[2]
        out['reason'] = ('детектор подтвердил объект в габарите: '
                         f'y={best[0]:.1f} u={best[1]:+.2f} точек={best[2]["points"]}')
    else:
        out['ok'] = True
        out['automation'] = 'cluster'
        out['reason'] = (f'кластер {int(idx.size)} точек, детектор объект не подтвердил'
                         f' (ближайшее препятствие: '
                         + (f'y={det["obstacles"][0]["y_m"]:.1f}' if det['obstacles'] else 'нет')
                         + ')')
    return out


def _detect_detail(xyz, model, cfg=None):
    """Детектор на облаке: (лучшее препятствие как (y, u, dict)|None, ответ)."""
    detect, DetectorConfig = _detector_module()
    conf = DetectorConfig()
    if cfg and cfg.get('config') is not None:
        conf = cfg['config']
    res = detect(np.asarray(xyz, dtype=np.float64), model, conf)
    obstacles = [o.to_dict() for o in res.obstacles]
    best = None
    for o in res.obstacles:
        if best is None or o.confidence > best[2]['confidence']:
            best = (float(o.y_m), float(o.u_m), o.to_dict())
    return best, {'n_obstacles': len(obstacles), 'obstacles': obstacles,
                  'points_total': int(res.points_total)}


def _detect_only(xyz, model, cfg=None):
    _best, det = _detect_detail(xyz, model, cfg)
    return det


# ------------------------------------------------------------ трассировка

def object_mesh(cls, size, center, yaw=0.0, pitch=0.0, roll=0.0):
    """Меш объекта класса `cls`: примитив `sim/objects.py`, повёрнутый и поставленный.

    `size` -- (ширина поперёк, высота, длина вдоль). Человек -- цилиндр туловища
    со сферой головы (`_person`), всё остальное -- бокс (`_suitcase`); размеры
    берутся из разметки, а не из библиотечных умолчаний. Локальные оси меша:
    X -- поперёк, Y -- вдоль, Z -- вверх, начало координат -- ЦЕНТР бокса (у
    примитивов начало на полу, поэтому поднимаем на полвысоты).

    Возвращает (меш в системе кадра, фактический размер примитива).
    """
    from sim import objects as sim_objects

    width, height, length = (float(size[0]), float(size[1]), float(size[2]))
    if cls == 'person':
        # У человека размеры поперёк и вдоль берутся из ширины: примитив --
        # цилиндр, длина вдоль пути отдельного смысла не имеет.
        om = sim_objects._person(height_m=height, width_m=width)
    else:
        om = sim_objects._suitcase(length_m=width, height_m=height, depth_m=length)
    mesh = om.mesh
    mesh.translate((0.0, 0.0, -height / 2.0))         # центр бокса -> начало координат
    rot = mesh.get_rotation_matrix_from_xyz(
        (math.radians(roll), math.radians(pitch), math.radians(yaw)))
    mesh.rotate(rot, center=(0.0, 0.0, 0.0))
    mesh.translate(tuple(float(v) for v in center))
    mesh.compute_vertex_normals()
    return mesh, tuple(float(v) for v in om.size_m)


def frame_rays(xyz, ring=None):
    """Представители лучей кадра: (направления, дальности, кратность, индексы).

    Представитель луча -- БЛИЖАЙШИЙ возврат (он и решает видимость), кратность --
    сколько возвратов на луче (у Hesai 128 в этих записях почти ровно 2). Ключ
    луча -- (кольцо, ячейка азимута), как в `validation/synth.py`: без него
    «луч» -- это не луч лидара, а произвольная точка.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    r = np.linalg.norm(xyz, axis=1)
    keep = r > 1e-9
    if not keep.any():
        return (np.zeros((0, 3)), np.zeros(0), np.zeros(0, dtype=np.int64),
                np.zeros(0, dtype=np.int64))
    directions = np.zeros_like(xyz)
    directions[keep] = xyz[keep] / r[keep, None]
    keys = synth.ray_keys(xyz, None if ring is None else np.asarray(ring, dtype=np.int64))
    order = np.lexsort((r, keys))
    ks = keys[order]
    first = np.empty(ks.size, dtype=bool)
    first[0] = True
    first[1:] = ks[1:] != ks[:-1]
    rep = order[first]
    mult = np.diff(np.append(np.flatnonzero(first), ks.size))
    return directions[rep], r[rep], mult, rep


def trace_object(xyz, ring, mesh):
    """Точки объекта по РЕАЛЬНЫМ лучам кадра с окклюзией.

    Лучи -- те, что лидар действительно излучил в этом кадре (представители);
    пересечение -- с мешем объекта (`open3d.t.geometry.RaycastingScene`), а не с
    его AABB: точка возврата лежит на поверхности примитива. Луч считается
    попавшим, если поверхность объекта БЛИЖЕ ближайшего возврата кадра на этом
    луче (дальше -- объект за препятствием, фон его загораживает).

    Возвращает dict: `points` (N, 3) в системе кадра, `ray_index` (индекс
    представителя), `t_hit`, `mult` (возвратов на луч), `removed` (маска точек
    кадра, закрытых объектом), `n_rays`, `n_points`.
    """
    import open3d as o3d

    xyz = np.asarray(xyz, dtype=np.float64)
    dirs, ranges, mult, rep = frame_rays(xyz, ring)
    out = {'points': np.zeros((0, 3)), 'ray_index': np.zeros(0, dtype=np.int64),
           't_hit': np.zeros(0), 'mult': np.zeros(0, dtype=np.int64),
           'removed': np.zeros(xyz.shape[0], dtype=bool),
           'n_rays': int(dirs.shape[0]), 'rays_hit': 0, 'n_points': 0,
           'points_removed': 0}
    if dirs.shape[0] == 0:
        return out

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    rays = np.concatenate([np.zeros((dirs.shape[0], 3), dtype=np.float32),
                           dirs.astype(np.float32)], axis=1)
    ans = scene.cast_rays(o3d.core.Tensor(rays, o3d.core.float32))
    t_hit = np.asarray(ans['t_hit'].numpy(), dtype=np.float64)
    # Допуск 1e-3 м: «в тех же сантиметрах» фон на луче -- это не окклюзия.
    visible = np.isfinite(t_hit) & (t_hit < ranges - 1e-3)
    n_hit = int(np.count_nonzero(visible))
    out['rays_hit'] = n_hit
    if n_hit == 0:
        return out

    dir_v = dirs[visible]
    t_v = t_hit[visible]
    mult_v = mult[visible]
    obj_pts = np.repeat(dir_v, mult_v, axis=0) * np.repeat(t_v, mult_v)[:, None]
    out['points'] = obj_pts
    out['ray_index'] = rep[visible]
    out['t_hit'] = t_v
    out['mult'] = mult_v
    out['n_points'] = int(obj_pts.shape[0])
    # Тень: все точки кадра на попавших лучах -- ЗА объектом (представитель луча
    # ближайший), значит для детектора они удаляются, иначе препятствие
    # «просвечивает» фоном.
    rep_r = ranges
    keys_all = synth.ray_keys(xyz, None if ring is None else np.asarray(ring, dtype=np.int64))
    hit_keys = keys_all[rep[visible]]
    out['removed'] = np.isin(keys_all, hit_keys)
    out['points_removed'] = int(np.count_nonzero(out['removed']))
    del rep_r
    return out


def reflectance_for(xyz, intensity, ray_index, mult=None):
    """Отражательная способность точек объекта -- ПО ТОЧКАМ, а не по лучам.

    Своей шкалы у интенсивности нет (сырые единицы 0..255), а объект ставится на
    ту же дальность, что и соседние точки кадра, поэтому честнее взять шкалу у
    САМОГО кадра -- значение ближайшего возврата на луче. Это тот же приём, что
    бутстрап интенсивности в `validation/synth.py`, только без случайности.

    На один луч приходится НЕСКОЛЬКО точек (меш даёт вход и выход, а кратность
    луча `mult` сохраняет плотность кадра), поэтому значение луча повторяется
    столько раз, сколько точек он дал: иначе `reflectance` (N_rays,) не совпадал
    бы с `points` (N_points,) и разметка была бы непригодна для обучения.
    """
    if intensity is None:
        return None
    idx = np.asarray(ray_index, dtype=np.int64)
    if idx.size == 0:
        return np.zeros(0)
    vals = np.asarray(intensity, dtype=np.float64)[idx]
    if mult is not None:
        m = np.asarray(mult, dtype=np.int64)
        if m.size == vals.size:
            vals = np.repeat(vals, m)
    return vals


def preview(cls, pose, center, size, yaw=0.0, pitch=0.0, roll=0.0, xyz=None,
            intensity=None, ring=None, model=None, db_path=None, cfg=None):
    """Что увидел бы лидар: точки объекта по лучам кадра + ответ детектора.

    Точки считаются трассировкой (см. `trace_object`), шум дальности НЕ
    добавляется: разметка -- эталон, а не измерение; те же точки (без шума) идут
    в `.npz` при сохранении, поэтому «что видно в панели» = «что записано».
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    mesh, prim_size = object_mesh(cls, size, center, yaw, pitch, roll)
    trace = trace_object(xyz, ring, mesh)
    refl = reflectance_for(xyz, intensity, trace['ray_index'], trace['mult'])
    out = {
        'class': cls, 'pose': pose,
        'center': [float(v) for v in center],
        'size': [float(v) for v in size],
        'yaw': float(yaw), 'pitch': float(pitch), 'roll': float(roll),
        'primitive_size': list(prim_size),
        'n_points': int(trace['n_points']),
        'counts': {'rays': int(trace['n_rays']), 'rays_hit': int(trace['rays_hit']),
                   'points_removed': int(trace['points_removed'])},
        'points': [[round(float(v), 4) for v in p] for p in trace['points']],
        'reflectance': (None if refl is None
                        else [round(float(v), 3) for v in refl]),
        'detector': None,
        'confirmed': False,
    }
    if xyz.shape[0] == 0:
        return out

    model = model if model is not None else _model_for(db_path)
    bg = xyz[~trace['removed']]
    aug = np.vstack([bg, trace['points']]) if trace['n_points'] else bg
    frame = path_frame(xyz, model)
    y_c, u_c, _h_c = _terms_of_center(frame, center)
    out['path'] = {'along_m': float(-y_c), 'lateral_m': float(u_c),
                   'height_m': float(_h_c)}
    best, det = _detect_detail(aug, model, cfg)
    det['points_augmented'] = int(aug.shape[0])
    out['detector'] = det
    if best is not None:
        tol_y = max(CONFIRM_TOL_Y_M, 0.5 * float(size[2]) + 1.0)
        tol_u = max(CONFIRM_TOL_U_M, 0.5 * float(size[0]) + 0.6)
        if abs(best[0] - y_c) <= tol_y and abs(best[1] - u_c) <= tol_u:
            out['confirmed'] = True
            out['detector_match'] = best[2]
    return out


def _terms_of_center(frame, center):
    y = np.asarray([float(center[1])])
    u = float(center[0]) - float(frame.axis_x(y)[0])
    h = float(center[2]) - float(frame.ugr_z(y)[0])
    return float(y[0]), u, h


# ------------------------------------------------------------------- запись

def _atomic_write(path, data: bytes):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(path)),
                               prefix='.tmp-', suffix=os.path.basename(path))
    try:
        with os.fdopen(fd, 'wb') as fh:
            fh.write(data)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _instance_at(center, ref_frame, frame, speed):
    """Центр на кадре: сдвиг ВДОЛЬ пути на замеренный ход.

    Трек объекта -- путевая система: в мире объект стоит, а сенсор едет. Ход
    (`speed_m_per_frame`, положительный -- сенсор приближается к объекту) замерен
    сдвигом профиля стены по Y системы лидара, поэтому и проекция -- сдвиг по Y:
    y(f) = y_ref + v*(f - ref). Поперёк и по высоте объект не ползёт -- он стоит
    «на месте пути», а не «на месте экрана».
    """
    dy = float(speed) * (int(frame) - int(ref_frame))
    return [float(center[0]), float(center[1]) + dy, float(center[2])]


def _local_points(points, center, yaw, pitch, roll, prim_size=None, cls=None):
    """Точки объекта в ЛОКАЛЬНОЙ системе объекта: центр в нуле, оси -- оси бокса."""
    if points.size == 0:
        return points.reshape(0, 3)
    rot = _rotation(yaw, pitch, roll)
    return (np.asarray(points, dtype=np.float64) - np.asarray(center, dtype=np.float64)) @ rot


def _rotation(yaw, pitch, roll):
    """R = Rx(roll) @ Ry(pitch) @ Rz(yaw) -- та же матрица, что у open3d
    `get_rotation_matrix_from_xyz((roll, pitch, yaw))`, которой повёрнут меш."""
    rx, ry, rz = (math.radians(roll), math.radians(pitch), math.radians(yaw))
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rx @ Ry @ Rz


def _next_seq(index_path, record):
    """Следующий номер объекта в записи: id = <record>-NNNN."""
    top = 0
    if os.path.isfile(index_path):
        with open(index_path, encoding='utf-8') as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if row.get('record') != record:
                    continue
                tail = str(row.get('id', '')).rsplit('-', 1)[-1]
                if tail.isdigit():
                    top = max(top, int(tail))
    return top + 1


def _read_index(index_path):
    rows = []
    if os.path.isfile(index_path):
        with open(index_path, encoding='utf-8') as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except ValueError:
                        continue
    return rows


def _manifest(rows, rev):
    by_class, by_record = {}, {}
    for row in rows:
        by_class[row['class']] = by_class.get(row['class'], 0) + 1
        rec = by_record.setdefault(row['record'], {'objects': 0, 'by_class': {},
                                                   'frames': 0, 'points': 0})
        rec['objects'] += 1
        rec['by_class'][row['class']] = rec['by_class'].get(row['class'], 0) + 1
        n = len(row.get('frames') or [])
        rec['frames'] += n
        rec['points'] += int(row.get('n_points') or 0)
    return {
        'objects': len(rows),
        'by_class': dict(sorted(by_class.items())),
        'records': dict(sorted(by_record.items())),
        'classes': list(CLASSES),
        'poses': list(POSES),
        'revision': rev,
        'updated': datetime.datetime.now().isoformat(timespec='seconds'),
    }


README_TEXT = """# Разметка объектов на пути (`labels/`)

Разметка сделана инструментом «Разметка» вьюера (`web_viewer.py` + `labeling.py`,
режим в панели) и лежит здесь в формате, под который обучается модель. Файлы
мелкие: точки объекта -- десятки КБ, всю разметку можно отдать человеку через git.

## Что где

| файл | что внутри |
|---|---|
| `index.jsonl` | по строке на объект: `id`, `record`, `class`, `pose`, `frames`, `n_points`, `automation` |
| `<record>/<id>.json` | карточка: геометрия по кадрам (`instances`), трек, ревизия |
| `<record>/<id>.npz` | точки объекта в ЛОКАЛЬНОЙ системе объекта: `points` (N, 3), `reflectance` (N,) если есть |
| `manifest.json` | счётчики по классам и записям, ревизия |

## Как читать

```python
import json, numpy as np
rows = [json.loads(l) for l in open('labels/index.jsonl', encoding='utf-8')]
card = json.load(open('labels/%s/%s.json' % (rows[0]['record'], rows[0]['id']), encoding='utf-8'))
pts  = np.load('labels/%s/%s.npz' % (card['record'], card['id']))['points']
```

`points` -- в системе объекта: начало в центре бокса, X поперёк пути, Y вдоль,
Z вверх; ориентация -- `yaw`/`pitch`/`roll` из соответствующего `instance`
(`R = Rx(roll) @ Ry(pitch) @ Rz(yaw)`). В систему кадра точки переводятся как
`points @ R.T + center`.

`instances[]` -- по одному на кадр из `frames`: `center` (система лидара этого
кадра), `yaw`/`pitch`/`roll`, `size` (ширина поперёк, высота, длина вдоль),
`automation` и признак подтверждения детектором.

## Трек и ход

Объект живёт в путевой системе: `track.speed_m_per_frame` -- ход сенсора
вдоль пути (положительный -- сенсор приближается), источник `measured`
(совмещение профиля стены между кадрами, `check_motion.py`) или `manual`
(число из панели). Проекция на кадр -- сдвиг вдоль Y:
`y(f) = y_ref + speed*(f - ref)`, `x` и `z` не меняются: объект стоит на месте
пути, а не на месте экрана.

## Что есть фон (негативы)

**Неотмеченное -- негативы.** Точки кадра, не попавшие ни в один объект,
считаются фоном: туннель, пол, рельсы, стены, платформа. Точки объекта -- это
ТРАССИРОВКА реальных лучей кадра по мешу-примитиву (`sim/objects.py`) с
окклюзией, без шума дальности: разметка -- эталон, а не измерение. Точки фона на
лучах, попавших в объект, в кадре ЗА объектом (их число -- `points_removed` в
`preview`), поэтому при обучении к негативам их относить нельзя.

`automation` в строке индекса -- происхождение габарита: `detector` (детектор
видит объект в этом габарите), `cluster` (габарит предложен кластером, детектор
не подтвердил), `manual` (поставлено руками). Для обучения это метка качества,
для разбора -- след автоматики.

## Ревизии

Карточка и `manifest.json` несут `revision`: md5 `detector/track_models.json`
(модель пути, по которой считаются `u`/`h`), md5 `track_geometry.py` (геометрия
пути и рельсов) и дату записи. Совпадение ревизии значит, что `u`/`h` и
окклюзия считались по той же геометрии пути; при расхождении разметку нужно
пересчитать перед обучением.
"""


def save(record, objects, xyz_at, out_dir=MAIN_LABELS_DIR, hz=HZ_DEFAULT,
         motion=None, revision_info=None):
    """Записать разметку в `labels/`: карточки, npz, index.jsonl, manifest, README.

    `objects` -- список объектов от панели:
      `class`, `pose`, `ref_frame`, `center`, `size`, `yaw`/`pitch`/`roll`,
      `speed_m_per_frame`, `speed_source` (`measured`|`manual`), `frames`,
      `automation`.
    Точки объекта считаются ЗДЕСЬ (трассировкой по кадрам разметки): то, что
    видел клиент в предпросмотре, -- без шума, те же лучи, тот же меш.

    Возвращает отчёт: пути файлов, id, число точек, счётчики.
    """
    out_dir = os.path.abspath(out_dir)
    index_path = os.path.join(out_dir, 'index.jsonl')
    rev = revision_info or revision()
    written, errors, next_seq = [], [], _next_seq(index_path, record)
    rows = _read_index(index_path)
    by_id = {row['id']: row for row in rows}

    model = None
    for obj in objects:
        try:
            cls = str(obj.get('class') or 'other')
            if cls not in CLASSES:
                raise ValueError(f'неизвестный класс {cls!r}: доступны {", ".join(CLASSES)}')
            pose = str(obj.get('pose') or DEFAULT_POSE[cls])
            if pose not in POSES:
                raise ValueError(f'неизвестная поза {pose!r}: доступны {", ".join(POSES)}')
            frames = sorted({int(f) for f in (obj.get('frames') or [])})
            if not frames:
                raise ValueError('не указаны кадры разметки (frames)')
            ref = int(obj.get('ref_frame', frames[0]))
            center = [float(v) for v in obj['center']]
            size = [float(v) for v in obj['size']]
            if len(center) != 3 or len(size) != 3:
                raise ValueError('center и size -- по три числа')
            if min(size) <= 0.0:
                raise ValueError(f'размеры должны быть положительными, получено {size}')
            speed = float(obj.get('speed_m_per_frame') or 0.0)
            source = str(obj.get('speed_source') or 'measured')
            if source not in ('measured', 'manual'):
                raise ValueError(f'источник хода {source!r}')
            automation = str(obj.get('automation') or 'manual')
            if automation not in AUTOMATION:
                raise ValueError(f'статус автоматики {automation!r}')

            xyz, inten, ring = frame_points(xyz_at, ref)
            if model is None:
                model = _model_for(obj.get('db_path') or '')
            frame_pf = path_frame(xyz, model)
            mesh, prim_size = object_mesh(cls, size, center, obj.get('yaw', 0.0),
                                          obj.get('pitch', 0.0), obj.get('roll', 0.0))
            trace = trace_object(xyz, ring, mesh)
            refl = reflectance_for(xyz, inten, trace['ray_index'], trace['mult'])
            pts = np.asarray(trace['points'], dtype=np.float64)
            local = _local_points(pts, center, obj.get('yaw', 0.0), obj.get('pitch', 0.0),
                                  obj.get('roll', 0.0))

            obj_id = str(obj.get('id') or f'{record}-{next_seq:04d}')
            if obj.get('id'):
                by_id.pop(obj_id, None)
            else:
                next_seq += 1
            y_c, u_c, h_c = _terms_of_center(frame_pf, center)
            card = {
                'id': obj_id,
                'record': record,
                'class': cls,
                'pose': pose,
                'frames': frames,
                'n_points': int(local.shape[0]),
                'automation': automation,
                'size': [float(v) for v in size],
                'created': datetime.datetime.now().isoformat(timespec='seconds'),
                'track': {
                    'speed_m_per_frame': speed,
                    'speed_kmh': speed * float(hz) * 3.6,
                    'source': source,
                    'reference_frame': ref,
                    'measured_m_per_frame': (motion or {}).get('m_per_frame'),
                    'hz': float(hz),
                    'rule': 'y(f) = y_ref + speed_m_per_frame * (f - ref_frame)',
                },
                'path': {
                    'along_m': float(-y_c), 'lateral_m': float(u_c), 'height_m': float(h_c),
                    'axis_x_m': float(frame_pf.axis_x(np.asarray([y_c]))[0]),
                    'ugr_z_m': float(frame_pf.ugr_z(np.asarray([y_c]))[0]),
                    'yaw_deg': float(obj.get('yaw', 0.0)),
                },
                'primitive': {'kind': 'person' if cls == 'person' else 'box',
                              'size_m': list(prim_size),
                              'source': 'sim/objects.py'},
                'counts': {'rays_hit': int(trace['rays_hit']),
                           'points_removed': int(trace['points_removed']),
                           'n_points': int(local.shape[0])},
                'instances': [
                    {
                        'frame': int(f),
                        'center': _instance_at(center, ref, f, speed),
                        'yaw': float(obj.get('yaw', 0.0)),
                        'pitch': float(obj.get('pitch', 0.0)),
                        'roll': float(obj.get('roll', 0.0)),
                        'size': [float(v) for v in size],
                        'automation': automation,
                        'detector_confirmed': bool(automation == 'detector'),
                    }
                    for f in frames
                ],
                'revision': rev,
            }
            rec_dir = os.path.join(out_dir, record)
            npz_path = os.path.join(rec_dir, f'{obj_id}.npz')
            card_path = os.path.join(rec_dir, f'{obj_id}.json')
            os.makedirs(rec_dir, exist_ok=True)
            _atomic_write(npz_path, _npz_bytes(local, refl))
            _atomic_write(card_path, json.dumps(card, ensure_ascii=False,
                                                indent=1).encode('utf-8'))
            by_id[obj_id] = {
                'id': obj_id, 'record': record, 'class': cls, 'pose': pose,
                'frames': frames, 'n_points': int(local.shape[0]),
                'automation': automation,
            }
            written.append({'id': obj_id, 'card': card_path, 'npz': npz_path,
                            'n_points': int(local.shape[0]),
                            'size_bytes': os.path.getsize(npz_path)})
        except Exception as exc:                     # noqa: BLE001 -- отчёт в ответе
            errors.append({'object': obj.get('id') or obj.get('class'), 'error': str(exc)})

    all_rows = [by_id[k] for k in sorted(by_id, key=lambda i: _id_sort_key(by_id[i]))]
    index_body = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in all_rows)
    _atomic_write(index_path, index_body.encode('utf-8'))
    manifest = _manifest(all_rows, rev)
    _atomic_write(os.path.join(out_dir, 'manifest.json'),
                  json.dumps(manifest, ensure_ascii=False, indent=1).encode('utf-8'))
    _atomic_write(os.path.join(out_dir, RECORD_README), README_TEXT.encode('utf-8'))
    return {
        'record': record,
        'dir': out_dir,
        'index': index_path,
        'manifest': os.path.join(out_dir, 'manifest.json'),
        'readme': os.path.join(out_dir, RECORD_README),
        'written': written,
        'errors': errors,
        'objects_total': len(all_rows),
        'counts': manifest,
    }


def _id_sort_key(row):
    rec = str(row.get('record') or '')
    tail = str(row.get('id') or '')
    num = 0
    if '-' in tail:
        tail_num = tail.rsplit('-', 1)[-1]
        num = int(tail_num) if tail_num.isdigit() else 0
    return (rec, num, str(row.get('id') or ''))


def _npz_bytes(points, reflectance):
    """Содержимое .npz в память: точки (N, 3) float32 и reflectance (N,) при наличии."""
    import io

    buf = io.BytesIO()
    payload = {'points': np.asarray(points, dtype=np.float32).reshape(-1, 3)}
    if reflectance is not None:
        payload['reflectance'] = np.asarray(reflectance, dtype=np.float32).reshape(-1)
    np.savez_compressed(buf, **payload)
    return buf.getvalue()