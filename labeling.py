"""Разметка объектов на пути: замер хода, предложение бокса, трассировка, запись.

Первая часть инструмента разметки. Человек ставит объект мышью (или просит
систему предложить габарит по клику), точки объекта считаются ТРАССИРОВКОЙ
реальных лучей кадра по мешу-примитиву, а разметка ложится в `labels/` в формате,
под который обучается модель. Маршруты `web_viewer.py` (`/motion`,
`/label/propose`, `/label/preview`, `/label/save`) -- тонкие: вся геометрия здесь.

Что внутри
----------
* `measure_motion`/`motion_profile` -- ход машины по записи как ПРОФИЛЬ по кадрам:
  сдвиг профиля стены между СОСЕДНИМИ кадрами (там сдвиг ~1 м и минимум резкий;
  на паре через 10 кадров окно поиска давало алиас -- 0.750 вместо 1.500 у
  `roundT_doubleT`), качество -- по остатку, плюс независимая перекрёстная
  проверка кластерами ярких точек (отражателями) на тех же кадрах. Кэш по записи:
  `/motion` ходит на каждую смену записи в панели, а сам замер стоит секунды.
* `PathFrame` -- путевая система КАДРА: ось (`u = 0`) и УГР (`h = 0`) в
  координатах кадра. Ось -- полилиния модели пути (`detector/track_models.json`)
  плюс доворот самого кадра (`detector.core.axis_pose`). Это важно: `u`/`h`
  объекта обязаны считаться в ТОЙ ЖЕ системе, что `u`/`h` детектора, иначе
  «детектор подтвердил» проверялось бы в другой системе координат.
* `propose` -- кластер вокруг луча клика -> бокс (центр, размеры, поворот по оси
  пути) + статус автоматики: `detector` (детектор видит объект в этом габарите),
  `cluster` (кластер есть, детектор не подтвердил), `manual` (не предложено).
* `trace_object` -- точки, которые УВИДЕЛ БЫ лидар: реальные лучи кадра
  (представитель луча = ближайший возврат), меш объекта (у человека -- запечённая
  поза `assets/_poses`, у остальных -- примитив `sim/objects.py`) и окклюзия по
  дальности ближайшего возврата -- как в `validation/synth.py`, только вместо AABB
  меш. Тень (все точки кадра на лучах, попавших в объект) отдаётся маской
  `removed`; в панели вьюера эти точки гасятся по ВСЕМ объектам кадра сразу.
  Поза человека: `pose_rule` (идёт по траектории / стоит) -> `pose_clip` ->
  `pose_moment`/`pose_vertices` -> `place_pose`; тот же расчёт в клиенте
  (`web/labellayer.js`), поэтому рисунок и счётчик не расходятся.
* `trajectory_normalize`/`trajectory_at` -- траектория объекта: две точки
  (основание) и секунды; отсюда скорость = расстояние / время, а позиция на кадре
  интерполируется ПО ВРЕМЕНИ (`_instance_center`).
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
import re
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

# Версия формата разметки. Карточки, записанные до появления поля `format`,
# читаются как версия 1 (только опорный кадр: `points` в .npz, тень числом в
# `counts`). Схема уже менялась при живых карточках, а ревизия md5 ловит правки
# КОДА, но не схемы -- поэтому версия схемы теперь лежит в самих данных:
# в карточке (`format`, подробно), в строке `index.jsonl` (`format`, номер) и в
# `manifest.json` (`format`, как в карточке).
FORMAT_VERSION = 2
FORMAT_INFO = {
    'version': FORMAT_VERSION,
    'name': 'labels-gt',
    # v2: трассировка на КАЖДОМ кадре из frames (v1 -- только на опорном).
    'per_frame_gt': True,
    # Ключ луча тот же, что в трассировке: validation/synth.ray_keys
    # (ring * 3600 + ячейка азимута 0.1°; без ring -- только азимут).
    'ray_key': 'ring*3600 + azimuth_bin_0.1deg',
    # Ключи .npz пер-кадрового GT: concatenation по кадрам из card['frames'],
    # границы -- gt_offsets (F+1). Точки объекта на кадре восстанавливаются как
    # направление представителя луча * gt_t_hit (повторено gt_mult раз), тень --
    # mask = isin(ray_keys(кадра), gt_ray_keys).
    'npz_gt_keys': ['frames', 'gt_offsets', 'gt_ray_keys', 'gt_t_hit', 'gt_mult'],
    'npz_points': 'points (N, 3) + reflectance (N,) -- опорный кадр (как в v1)',
    'changes': 'v1: точки только на опорном кадре, тень числом; '
               'v2: + пер-кадровый GT, поле format, counts по кадрам в instances',
}

# Пары кадров замера хода -- те же, что у `check_motion.py` при fa=100, fb=110,
# step=20 (он печатает сдвиг первой пары и медиану м/кадр по всем).
MOTION_PAIRS = ((100, 110), (120, 130), (140, 150), (160, 170))
MOTION_SPAN = 10                        # кадров в паре (как fb - fa)
MOTION_SIDE = -1                        # профиль левой стены
# Ход -- ПРОФИЛЬ по кадрам, а не одно число на запись. Считается сдвигом профиля
# стены между СОСЕДНИМИ кадрами: на паре через 10 кадров сдвиг ~7-8 м, а окно
# поиска ±25 м на периодической структуре стены даёт ПЛОСКУЮ поверхность остатка
# (замерено: five лучших сдвигов 0.009-0.010 при разбросе +6.5..+8.0 м), то есть
# выбирается алиас и знак может перевернуться: у roundT_doubleT хранилось +0.750
# вместо +1.500, у roundT_squareT... -1.325 вместо +1.250.
MOTION_NEIGHBOUR_STEP = 1               # сдвиг между соседними кадрами
MOTION_NEIGHBOUR_SPAN_M = 2.5           # окно поиска для соседних кадров, м
MOTION_RES_OK = 0.05                    # остаток < -- совмещение резкое, кадру верим
MOTION_RES_BAD = 0.20                   # остаток > -- кадр недостоверен (не выдаём за факт)
MOTION_PROFILE_MAX_FRAMES = 1200        # предохранитель: длиннее записи не бывает
# Алиас: у периодической стены поверхность остатка плоская -- несколько сдвигов
# отличаются на 0.02 м остатка (замерено на squareT: r=0.003 при d = -1.25...-0.8
# и +0.25..+0.75 при истинном сдвиге +1.35 м). Такие кадры помечаются `alias`, а
# выбор между равными кандидатами делает независимая проверка (отражатели):
# если её оценка совпала с одним из кандидатов, кадр идёт как `alias-fixed`.
MOTION_AMBIG_TOL_M = 0.020              # «остатки не отличить», м
MOTION_AMBIG_SPREAD_M = 0.50            # и сдвиги различаются сильнее этого, м
MOTION_ALIAS_TOL_M = 0.35               # насколько кандидат может отличаться от проверки
# Перекрёстная проверка -- другим методом и на тех же кадрах: кластеры ярких точек
# (отражатели). Они стабильны и дают ход по ИДЕНТИЧНОСТИ объектов, а не по профилю
# стены: два независимых метода, расхождение -- повод не верить кадру.
REFL_PCT = 98.0                         # перцентиль интенсивности -- «яркие»
REFL_CELL_M = 0.40                      # воксель кластеризации отражателей, м
REFL_MIN_POINTS = 3                     # меньше точек -- не отражатель, а отблеск
REFL_MATCH_UV_M = 1.00                  # допуск совпадения поперёк/по высоте, м
REFL_MATCH_MAX_SHIFT_M = 4.00           # |сдвиг| больше -- это не тот отражатель
REFL_MIN_PAIRS = 6                      # меньше пар -- метод не даёт числа
REFL_MAX_SPREAD_M = 0.35                # разброс сдвигов по парам, м

# Полоса объекта вокруг луча клика: отметает пол/рельс/свод, но оставляет тело.
BAND_H_LOW_M = 0.12                     # над ИЗМЕРЕННОЙ поверхностью (не над УГР): 0.16 м -- высота рельса
BAND_H_HIGH_M = 2.60                    # выше этой высоты -- свод, лотки, кабели
BAND_U_HALF_M = 3.0                     # поперёк от клика
WINDOW_Y_HALF_M = 3.0                   # вдоль от клика: длинную структуру не берём
CLUSTER_CELL = (0.30, 0.15, 0.15)        # воксель (вдоль, поперёк, по высоте), м
SEED_R_M = 0.45                         # радиус семени вокруг клика, м
# Запасные пути поиска, когда под кликом связного кластера нет: клик -- это точка
# облака под курсором, и она бывает одиночным отблеском в стороне от объекта
# (или точкой пола под ним). Тогда ищем ближайший кластер в плане и, если и он не
# набрал точек, -- всё, что попало в шар вокруг клика.
PLAN_R_M = 1.20                         # радиус поиска «ближайшего кластера», в плане, м
BALL_R_M = 0.70                         # радиус шара вокруг клика, м
# Запасные пути ищут ТЕЛО объекта, а не пол: у полосы объекта низ -- 0.12 м (лежащий
# человек тоже объект), а рядом с кликом по полу в эту высоту попадают рельс и стыки
# плит -- из них выходил бокс 0.3×0.25×3.0 м «вдоль всего пути».
PLAN_H_LOW_M = 0.30                     # выше этого над полом -- тело, а не покрытие
PERCENTILE_LO, PERCENTILE_HI = 2.0, 98.0

# ОПОРА ОБЪЕКТА -- ИЗМЕРЕННАЯ поверхность под ним, а не уровень по модели пути.
# Раньше опора бралась от плоскости «УГР минус высота рельса» (`PathFrame.floor_z`),
# и это один уровень на всю запись: пол тоннеля наклонный (~2 %), между рельсами он
# ниже на 0.16..0.40 м (лоток -- заказчик сказал, что это часть пола), сбоку --
# полка. Из-за этого человек в лотке «висел» на 0.37-0.40 м над измеренной
# поверхностью (`doubleT_obstacle` кадр 13: опора -1.736 при поверхности -2.133).
# Поверхность измеряется полосой в плане вокруг центра объекта: вдоль ±Y, поперёк
# ±U, поверхность -- низкий перцентиль z точек полосы (та же мера, что у замера
# заказчика: 2-й перцентиль z точек в окне |x| < 1.2 м и |dy| < 0.5 м).
SURFACE_WIN_Y_M = 0.50                  # вдоль центра объекта, м
SURFACE_WIN_U_M = 0.40                  # поперёк центра, м
SURFACE_PCT = 2.0                       # перцентиль z в полосе -- поверхность
SURFACE_MIN_POINTS = 20                 # меньше точек -- полосу расширяем (см. _surface_z)
SURFACE_WIDEN = (2.0, 4.0)              # во сколько раз расширять полосу поперёк/вдоль
# Страховка от мусора (одиночные отблески ниже пола): измеренная поверхность не
# может отличаться от модельного пола больше, чем на эти величины.
SURFACE_MAX_DROP_M = 0.80               # ниже модельного пола (лоток)
SURFACE_MAX_LIFT_M = 1.50               # выше модельного пола (полка, платформа)
SIZE_MIN = (0.30, 0.25, 0.20)           # минимумы (ширина, высота, длина), м
SIZE_MAX = (2.50, 2.80, 3.00)           # максимумы: длиннее -- это стена/платформа
# Габарит объекта, поставленного РУКАМИ (автоматика не предложила), когда рядом с
# кликом ИЗМЕРИТЬ нечего: последний запас, а не рабочее значение. Если точки рядом
# есть, габарит берётся по ним (`_manual_geometry`), а панель показывает источник
# размера (`size_source`), чтобы «неизвестный» габарит не выглядел измеренным.
MANUAL_SIZE = (0.60, 1.00, 1.00)
MANUAL_BAND_U_M = 1.20                  # поперёк от клика, м: из чего меряем габарит вручную
MANUAL_BAND_Y_M = 1.20                  # вдоль от клика, м
MANUAL_MIN_POINTS = 12                  # меньше -- это отблески, а не габарит
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
    """Ход машины по записи: ПРОФИЛЬ по кадрам + сводка (м/кадр, км/ч, остаток).

    Положительный ход -- объект в мире ПРИБЛИЖАЕТСЯ к сенсору (сдвиг профиля
    стены в +Y системы лидара): так его и печатает `check_motion.py`
    (0.42..0.82 м/кадр у едующих записей, -1.32 у обратного хода, 0.00 у стоящей).

    Основной замер -- сдвиг между СОСЕДНИМИ кадрами (`motion_profile`): он даёт
    `profile` (м/кадр на каждый кадр), `profile_ok` (где остаток резкий) и
    перекрёстную проверку отражателями. Сводка (`m_per_frame`, `kmh`, `pairs`,
    `residual_m`, `source`, `measured`) сохранена: её читают клиент и карточка.

    `pairs` остаётся для совместимости (старый вызов `check_motion.best_shift` по
    парам через 10 кадров): числа пар уходят в ответ как `pairs`/`pairs_median`, но
    СВОДКА берётся из профиля. Раньше именно эти пары и задавали `m_per_frame` --
    и на периодической стене давали алиас (0.750 вместо 1.500 у `roundT_doubleT`).

    `xyz_at(frame)` -- источник кадра; без него профиль считается по своей
    `BagFrames`, как в исходном скрипте.
    """
    import check_motion

    key = (os.path.abspath(db_path), int(total or 0),
           tuple(map(int, pairs[0])) if pairs else (-1, -1))
    hit = _MOTION_CACHE.get(key)
    if hit is not None:
        return hit

    prof = motion_profile(db_path, xyz_at=xyz_at, total=total, hz=hz)
    out = dict(prof)
    if pairs:
        rates, details = [], []
        for a, b in pairs:
            got = check_motion.best_shift(db_path, int(a), int(b), MOTION_SIDE,
                                          xyz_at=xyz_at)
            if got is None:
                continue
            rates.append(got[0] / float(b - a))
            details.append({'frames': [int(a), int(b)], 'shift_m': float(got[0]),
                            'm_per_frame': float(got[0] / float(b - a)),
                            'residual_m': float(got[1]), 'bands': int(got[2])})
        out['pairs'] = details
        out['pairs_median'] = float(np.median(rates)) if rates else None
    _MOTION_CACHE[key] = out
    return out


def motion_profile(db_path, xyz_at=None, total=None, hz=HZ_DEFAULT,
                   step=MOTION_NEIGHBOUR_STEP, span_m=MOTION_NEIGHBOUR_SPAN_M,
                   cross=True, max_frames=MOTION_PROFILE_MAX_FRAMES):
    """Ход записи ПРОФИЛЕМ по кадрам: сдвиг профиля стены между соседними кадрами.

    `profile[f]` -- ход на кадре `f` (переход f -> f+step), м/кадр; `None` там, где
    совмещение не совпало (остаток > `MOTION_RES_BAD`) или кадра нет (последние
    `step` кадров). `profile_ok[f]` -- остаток резкий (`< MOTION_RES_OK`), такому
    кадру можно верить; `profile_residual[f]` -- остаток.

    Почему профиль, а не одно число: ход меняется по записи (поезд разгоняется и
    тормозит, на `squareT_platform_squareT_switch` он долго стоит), а пара через
    10 кадров с окном ±25 м на периодической стене даёт плоскую поверхность
    остатка -- выбирается алиас, и знак переворачивается (`roundT_doubleT`: 0.750
    вместо 1.500).

    `cross` -- посчитать ещё и независимую проверку (кластеры ярких точек) по тем
    же кадрам: `cross_profile[f]` (сдвиг по идентичности отражателей), `cross_ok[f]`,
    `cross_pairs[f]`, `cross_diff[f]` = профиль - проверка. Расхождение методов --
    признак, что кадру верить нельзя (в `profile_flags[f]` это `bad`/`differs`).

    Возвращает сводку с профилем; ключи сводки -- те же, что у старого замера
    (`m_per_frame`, `kmh`, `residual_m`, `hz`, `source`, `measured`), плюс
    `profile`, `profile_frames`, `profile_ok`, `profile_flags`, `reliable_frac`,
    `profile_min`/`profile_max`, `cross_*`.
    """
    import check_motion

    record = os.path.basename(os.path.dirname(os.path.abspath(db_path)))
    total = int(total or 0)
    src = xyz_at
    if src is None:
        from visualize_bag import BagFrames

        frames = BagFrames(db_path, cache_size=2)
        total = total or len(frames)
        src = lambda f: frames[f]                       # noqa: E731 (xyz, intensity, ts)
    n = min(total, int(max_frames))
    prof = [None] * n
    prof_raw = [None] * n
    res = [None] * n
    ok = [False] * n
    alias = [False] * n
    cross_prof = [None] * n
    cross_ok = [False] * n
    cross_pairs = [0] * n
    # Один проход по кадрам: держим только ПРЕДЫДУЩИЙ кадр (облако 350k точек --
    # 8 МБ, всю запись в памяти не удержать), поэтому и сдвиг профиля, и
    # перекрёстная проверка считаются сразу для пары (f - step, f).
    prev = None                                  # (frame, xyz, inten, grid, prof)
    for f in range(n):
        got = src(int(f))
        xyz = np.asarray(got[0], dtype=np.float64)
        inten = (np.asarray(got[1], dtype=np.float64)
                 if len(got) > 1 and got[1] is not None else None)
        grid, wp = check_motion.wall_profile_points(xyz, MOTION_SIDE)
        if prev is not None and prev[0] + step == f:
            a = prev[0]
            cands = check_motion.profile_candidates(prev[3], prev[4], grid, wp,
                                                    span_m=span_m)
            # Независимая проверка -- ДО выбора сдвига: она разрешает алиас.
            if cross and prev[2] is not None and inten is not None:
                got_cross = reflector_shift(prev[1], prev[2], xyz, inten)
                if got_cross is not None:
                    cross_prof[a] = float(got_cross[0]) / float(step)
                    cross_ok[a] = True
                    cross_pairs[a] = int(got_cross[2])
            if cands:
                best_d, best_r = cands[0][0], cands[0][1]
                prof_raw[a] = best_d / float(step)
                # Кандидаты «не отличить по остатку»: их и считаем алиасом.
                rivals = [c for c in cands
                          if c[1] <= best_r + MOTION_AMBIG_TOL_M
                          and abs(c[0] - best_d) > MOTION_AMBIG_SPREAD_M]
                chosen = best_d
                if rivals and cross_prof[a] is not None:
                    near = min(cands, key=lambda c: abs(c[0] - cross_prof[a] * step))
                    if abs(near[0] - cross_prof[a] * step) <= MOTION_ALIAS_TOL_M:
                        chosen, alias[a] = near[0], True
                picked = next((c for c in cands if c[0] == chosen), cands[0])
                res[a] = float(picked[1])
                if not rivals or alias[a]:
                    if picked[1] < MOTION_RES_OK:
                        prof[a] = chosen / float(step)
                        ok[a] = True
        prev = (f, xyz, inten, grid, wp)

    values = [v for v in prof if v is not None]
    reliable = [v for v, good in zip(prof, ok) if good]
    differences = [v - c for v, c in zip(prof, cross_prof) if v is not None and c is not None]
    flags = []
    for i in range(n):
        if prof_raw[i] is None:
            flags.append('none')
        elif not ok[i] and res[i] is not None and res[i] >= MOTION_RES_OK:
            flags.append('weak')
        elif prof[i] is None:
            flags.append('alias')            # кандидаты равны, проверки нет
        elif alias[i]:
            flags.append('alias-fixed')      # алиас разрешён проверкой
        elif cross_prof[i] is not None and abs(prof[i] - cross_prof[i]) > MOTION_RES_BAD:
            flags.append('differs')
        else:
            flags.append('ok')
    median = float(np.median(reliable)) if reliable else (
        float(np.median(values)) if values else 0.0)
    out = {
        'record': record,
        'm_per_frame': median,
        'kmh': median * float(hz) * 3.6,
        'residual_m': float(np.median([r for r in res if r is not None])) if any(
            r is not None for r in res) else None,
        'hz': float(hz),
        'source': 'wall-profile-neighbour',
        'measured': bool(reliable),
        'pairs': [],
        # --- профиль по кадрам ---
        'profile': prof,
        'profile_raw': prof_raw,            # что дал один профиль стены (до разрешения алиаса)
        'profile_alias_fixed': alias,       # кадры, где алиас разрешён проверкой
        'profile_frames': list(range(n)),
        'profile_ok': ok,
        'profile_residual': res,
        'profile_flags': flags,
        'profile_step': int(step),
        'profile_span_m': float(span_m),
        'reliable_frac': (len(reliable) / float(n)) if n else 0.0,
        'alias_fixed_frac': (sum(alias) / float(n)) if n else 0.0,
        'profile_min': (min(reliable) if reliable else None),
        'profile_max': (max(reliable) if reliable else None),
        'profile_std': (float(np.std(reliable)) if reliable else None),
        'cross_profile': cross_prof,
        'cross_ok': cross_ok,
        'cross_pairs': cross_pairs,
        'cross_diff': differences,
        'cross_diff_max': (max(abs(v) for v in differences) if differences else None),
    }
    return out


def _motion_values(motion):
    """Значения профиля хода из ответа `measure_motion`/`motion_profile` (или None)."""
    if not isinstance(motion, dict):
        return None
    prof = motion.get('profile')
    if not prof:
        return None
    return list(prof)


def motion_at(motion, frame, default=None):
    """Ход на КАДРЕ по профилю: м/кадр (знаменатель -- кадр), None если нет.

    У недостоверных кадров значения нет: подставлять туда медиану записи в
    геометрии нельзя молча -- вызывающий решает сам (`motion_between` это делает
    осознанно и считает такие кадры ходом по ближайшему достоверному).
    """
    prof = _motion_values(motion)
    if prof is None:
        return default
    f = int(frame)
    if 0 <= f < len(prof) and prof[f] is not None:
        return float(prof[f])
    return default


def motion_between(motion, frame_from, frame_to, default=None):
    """Смещение вдоль пути между кадрами: ИНТЕГРАЛ профиля, а не v * df.

    Ход записи меняется по кадрам, поэтому «сколько объект уехал между кадрами
    ref и f» -- это сумма ходов всех кадров между ними, а не одно число на
    разность номеров. У недостоверных кадров (и у последнего кадра профиля, где
    перехода ещё нет) берётся медиана достоверных -- иначе разрыв профиля
    обнулял бы ход на этих кадрах, а объект дёргался бы.

    Без профиля (`default` -- старое одно число) возвращается `default * (to - from)`.
    """
    a, b = int(frame_from), int(frame_to)
    prof = _motion_values(motion)
    if prof is None or not isinstance(motion, dict):
        return (0.0 if default is None else float(default)) * (b - a)
    fill = motion.get('m_per_frame')
    if fill is None:
        vals = [v for v in prof if v is not None]
        fill = float(np.median(vals)) if vals else 0.0
    step = int(motion.get('profile_step') or 1)
    total = 0.0
    f, target = a, b
    rng = range(f, target, step) if b >= a else range(f - step, target - step, -step)
    for i in rng:
        if 0 <= i < len(prof) and prof[i] is not None:
            sign = 1.0 if b >= a else -1.0
            total += sign * float(prof[i])
        else:
            sign = 1.0 if b >= a else -1.0
            total += sign * float(fill)
    return total


def _bright_centroids(xyz, inten, pct=REFL_PCT, cell=REFL_CELL_M,
                      min_points=REFL_MIN_POINTS):
    """Кластеры ЯРКИХ точек (отражателей): (центроид, сколько точек).

    Отражатели стабильны и видны в каждом кадре: их идентичность даёт ход вторым,
    независимым от профиля стены методом (профиль -- про структуру стены, здесь --
    про конкретные объекты). Кластеризация -- воксели (вдоль/поперёк/вверх),
    как у кластера объекта, только ячейка крупнее и точек нужно меньше: на 60 м
    отражатель -- это единицы возвратов.
    """
    if inten is None or xyz.shape[0] == 0:
        return np.zeros((0, 3)), np.zeros(0)
    thr = float(np.percentile(inten, pct))
    sel = inten >= thr
    if int(sel.sum()) < min_points:
        return np.zeros((0, 3)), np.zeros(0)
    p = xyz[sel]
    keys = np.floor(p / cell).astype(np.int64)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    cnt = np.bincount(inv, minlength=len(uniq))
    keep = cnt >= min_points
    if not keep.any():
        return np.zeros((0, 3)), np.zeros(0)
    cent = np.zeros((int(keep.sum()), 3), dtype=np.float64)
    weights = np.zeros(int(keep.sum()), dtype=np.float64)
    idx_keep = np.flatnonzero(keep)
    where = np.zeros(len(uniq), dtype=np.int64) - 1
    where[idx_keep] = np.arange(idx_keep.size)
    for j, u in enumerate(idx_keep):
        m = inv == u
        cent[j] = p[m].mean(axis=0)
        weights[j] = float(m.sum())
    return cent, weights


def reflector_shift(xyz_a, inten_a, xyz_b, inten_b, cell=REFL_CELL_M,
                    match_uv=REFL_MATCH_UV_M, max_shift=REFL_MATCH_MAX_SHIFT_M,
                    min_pairs=REFL_MIN_PAIRS, max_spread=REFL_MAX_SPREAD_M):
    """Ход между кадрами по ИДЕНТИЧНОСТИ ярких кластеров: (dy, спред, пар) | None.

    Для каждого отражателя кадра a ищется ближайший в кадре b (поперёк и по высоте
    -- в пределах `match_uv`, вдоль -- в пределах `max_shift`), пара берётся только
    взаимно-однозначная, ход -- медиана разностей по парам. `dy = y_b - y_a`:
    тот же знак, что у сдвига профиля стены (положительный -- сенсор приближается).

    Число пар и спред -- честные показатели качества: меньше `min_pairs` пар или
    спред больше `max_spread` -- метод числа не даёт (None), а не «примерно так».
    """
    ca, wa = _bright_centroids(xyz_a, inten_a, cell=cell)
    cb, wb = _bright_centroids(xyz_b, inten_b, cell=cell)
    if len(ca) < min_pairs or len(cb) < min_pairs:
        return None
    dy = np.full((len(ca), len(cb)), np.nan)
    du = np.abs(ca[:, None, 0] - cb[None, :, 0])
    dz = np.abs(ca[:, None, 2] - cb[None, :, 2])
    dyv = cb[None, :, 1] - ca[:, None, 1]
    good = (du <= match_uv) & (dz <= match_uv) & (np.abs(dyv) <= max_shift)
    # Взаимно-однозначность: пара берётся, только если она лучшая и слева, и справа.
    cost = np.where(good, du + dz, np.inf)
    left = np.argmin(cost, axis=1)
    right = np.argmin(cost, axis=0)
    rows = np.arange(len(ca))
    pairs = [(i, int(left[i])) for i in rows
             if np.isfinite(cost[i, left[i]]) and int(right[left[i]]) == i]
    if len(pairs) < min_pairs:
        return None
    vals = np.array([dyv[i, j] for i, j in pairs], dtype=np.float64)
    med = float(np.median(vals))
    spread = float(np.median(np.abs(vals - med)))
    if spread > max_spread:
        return None
    return med, spread, int(len(pairs))


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


def _surface_z(xyz, frame: PathFrame, y_c, u_c, y_half=SURFACE_WIN_Y_M,
               u_half=SURFACE_WIN_U_M):
    """ИЗМЕРЕННАЯ поверхность под точкой (y_c, u_c) плана: (z | None, сведения).

    Полоса в плане вокруг центра объекта (вдоль ±`y_half`, поперёк ±`u_half`),
    поверхность -- низкий перцентиль z точек полосы. Так опора учитывает МЕСТО
    объекта: лоток между рельсами (ниже модельного пола на 0.16..0.40 м), полку
    сбоку, наклон пола (~2 %), а не один уровень на всю запись.

    Полоса расширяется (`SURFACE_WIDEN`), если точек мало: у дальнего объекта
    возвратов в узкой полосе не набирается. Измеренная поверхность ограничена
    относительно модельного пола (`SURFACE_MAX_DROP_M`/`SURFACE_MAX_LIFT_M`): ниже
    него бывает лоток, выше -- полка, но одиночный отблеск в метре под полом
    поверхностью не является.

    Возвращает `(z, info)`; `z is None` -- мерить нечем (полоса пуста), опора
    берётся по модельному полу.
    """
    y, u, _h = frame.terms(xyz)
    model = float(frame.floor_z(np.asarray([y_c]))[0])
    n = 0
    for k in (1.0,) + tuple(SURFACE_WIDEN):
        yh, uh = y_half * k, u_half * k
        sel = (np.abs(y - y_c) <= yh) & (np.abs(u - u_c) <= uh)
        n = int(np.count_nonzero(sel))
        if n < SURFACE_MIN_POINTS:
            continue
        z = float(np.percentile(xyz[sel, 2], SURFACE_PCT))
        lo, hi = model - SURFACE_MAX_DROP_M, model + SURFACE_MAX_LIFT_M
        return float(min(max(z, lo), hi)), {
            'n_points': n, 'win_y_m': float(yh), 'win_u_m': float(uh),
            'raw_z_m': z, 'model_floor_z_m': model,
            'clamped': bool(z < lo or z > hi),
        }
    return None, {'n_points': n, 'win_y_m': float(y_half * SURFACE_WIDEN[-1]),
                  'win_u_m': float(u_half * SURFACE_WIDEN[-1]),
                  'raw_z_m': None, 'model_floor_z_m': model, 'clamped': False}


def _cluster_around(xyz, click, frame: PathFrame, cell=CLUSTER_CELL,
                    seed_r=SEED_R_M, window_y_half=WINDOW_Y_HALF_M, surface=None):
    """Кластер точек вокруг клика в путевых координатах.

    Отсекаем известные поверхности (пол, рельсы, свод) полосой по высоте над
    ПОЛОМ, клик берём семенем, дальше -- связные (26 соседей) воксели. Именно
    связность, а не радиус: тело человека и стена рядом -- разные компоненты, а
    радиус склеил бы их в один габарит.

    Возвращает (idx точек в xyz, (y, u, hf)) или None.
    """
    band, y, u, hf, (cy, cu, cw) = _click_band(xyz, click, frame,
                                               window_y_half=window_y_half,
                                               surface=surface)
    if band is None:
        return None
    near = (np.abs(y[band] - cy) < seed_r) & (np.abs(u[band] - cu) < seed_r) \
        & (np.abs(hf[band] - cw) < seed_r)
    if not near.any():
        return None
    return _component(band=band, y=y, u=u, hf=hf,
                      anchor=(cy, cu, cw), cell=cell, seeds_mask=near)


def _click_band(xyz, click, frame, window_y_half=WINDOW_Y_HALF_M,
                u_half=BAND_U_HALF_M, h_low=BAND_H_LOW_M, h_high=BAND_H_HIGH_M,
                surface=None):
    """Точки полосы объекта вокруг клика: (band, y, u, hf, (cy, cu, cw)).

    Полоса отсекает известные поверхности по высоте над ОПОРОЙ (не над УГР и не
    над модельным полом) и ограничивает окно вдоль/поперёк клика. Отсчёт от
    опоры, а не от модельного пола, -- потому что пол в месте клика может быть
    ниже модели на 0.4 м (лоток между рельсами): тогда полоса «от пола» срезала
    бы ноги стоящего в лотке человека, и его бокс висел бы в воздухе.
    Клик возвращается в тех же путевых координатах: `cu` -- поперёк от оси пути
    кадра, `cw` -- над опорой.
    """
    y, u, _h = frame.terms(xyz)
    cy = float(click[1])
    cu = float(click[0] - frame.axis_x(np.asarray([click[1]]))[0])
    # Отсчёт высоты -- от ОПОРЫ (измеренной поверхности под кликом), а если её
    # измерить не удалось -- от модельного пола кадра (УГР минус высота рельса).
    model = float(frame.floor_z(np.asarray([cy]))[0])
    ref = model if surface is None else float(surface)
    hf = xyz[:, 2] - ref                             # высота над опорой
    cw = float(click[2] - ref)                       # клик над опорой
    band = ((hf > h_low) & (hf < h_high)
            & (np.abs(u - cu) < u_half)
            & (np.abs(y - cy) < window_y_half))
    if not band.any():
        return None, y, u, hf, (cy, cu, cw)
    return band, y, u, hf, (cy, cu, cw)


def _component(band, y, u, hf, anchor, cell, seeds_mask):
    """Связная (26 соседей) компонента вокселей вокруг семян `seeds_mask`.

    Семена -- точки полосы `band`, попавшие в окно вокруг `anchor` (клика).
    Возвращает (индексы точек в xyz, (y, u, hf) этих точек) или None.
    """
    cy, cu, cw = anchor
    dy, du, dh = cell
    iy = np.floor((y[band] - cy) / dy).astype(np.int64)
    iu = np.floor((u[band] - cu) / du).astype(np.int64)
    ih = np.floor((hf[band] - cw) / dh).astype(np.int64)
    keys, inv = np.unique(np.stack([iy, iu, ih], axis=1), axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    if not seeds_mask.any():
        return None
    adj = _voxel_adjacency(keys)
    seeds = np.unique(inv[seeds_mask])

    seen = np.zeros(len(keys), dtype=bool)
    seen[seeds] = True
    stack = [int(v) for v in seeds]
    while stack:
        i = stack.pop()
        for j in adj[i]:
            if not seen[j]:
                seen[j] = True
                stack.append(j)

    idx = np.flatnonzero(band)[seen[inv]]
    return idx, (y[idx], u[idx], hf[idx])


def _cluster_plan(xyz, click, frame, cell=CLUSTER_CELL, plan_r=PLAN_R_M, surface=None):
    """Ближайший к клику кластер В ПЛАНЕ, а не под лучом.

    Клик -- точка облака под курсором, и под курсором бывает одиночный отблеск
    (или точка пола рядом с объектом): связного кластера вокруг него нет, а объект
    рядом -- есть. Здесь семенем берётся самая близкая к клику точка в плане (в
    пределах `plan_r`), дальше та же связность: чужой объект в габарит не попадёт,
    потому что он -- отдельная компонента.
    """
    band, y, u, hf, (cy, cu, _cw) = _click_band(xyz, click, frame,
                                                h_low=PLAN_H_LOW_M, surface=surface)
    if band is None:
        return None
    yr, ur = y[band], u[band]
    d2 = (yr - cy) ** 2 + (ur - cu) ** 2
    near = d2 <= plan_r * plan_r
    if not near.any():
        return None
    seed = np.zeros(near.shape, dtype=bool)
    seed[int(np.argmin(np.where(near, d2, np.inf)))] = True
    return _component(band=band, y=y, u=u, hf=hf, anchor=(cy, cu, _cw),
                      cell=cell, seeds_mask=seed)


def _cluster_ball(xyz, click, frame, r=BALL_R_M, cell=CLUSTER_CELL, surface=None):
    """Всё, что попало в шар радиуса `r` вокруг клика (без связности).

    Последний запасной путь: на дальней дистанции объект -- редкие точки, и
    связность их не собирает (воксели не соприкасаются). Габарит по такому шару
    грубее кластера, поэтому он и идёт после него.
    """
    band, y, u, hf, (cy, cu, cw) = _click_band(xyz, click, frame,
                                               h_low=PLAN_H_LOW_M, surface=surface)
    if band is None:
        return None
    d2 = (y[band] - cy) ** 2 + (u[band] - cu) ** 2 + (hf[band] - cw) ** 2
    near = d2 <= r * r
    idx = np.flatnonzero(band)[near]
    if idx.size == 0:
        return None
    return idx, (y[idx], u[idx], hf[idx])


def _box_from_cluster(cy, cu, chf, frame, base_z=None, surface=None):
    """Бокс по кластеру: центр, размеры (ширина, высота, длина), поворот по оси пути.

    `chf` -- высота точек НАД ОПОРОЙ (см. `_click_band`): и отсчёт, и абсолютные
    высоты берутся от одной поверхности -- ИЗМЕРЕННОЙ под объектом (`surface`).
    Пока отсчёт был от модельного пола, бокс вставал на него, а не на ту
    поверхность, что реально измерена в этом месте: в лотке между рельсами он
    оказывался на 0.37-0.40 м выше (жалоба «человек висит в воздухе»), а на
    рельсе/полке наоборот -- внутри головки.

    `base_z` -- высота опорной поверхности, на которую ставится объект (точка
    клика): низ бокса встаёт на неё, но НЕ выше низа самого кластера -- иначе тело
    объекта осталось бы вне бокса. Если `base_z` не задан, низ -- по кластеру.
    """
    y_lo, y_hi = np.percentile(cy, [PERCENTILE_LO, PERCENTILE_HI])
    u_lo, u_hi = np.percentile(cu, [PERCENTILE_LO, PERCENTILE_HI])
    h_lo, h_hi = np.percentile(chf, [PERCENTILE_LO, PERCENTILE_HI])
    y_c = float(0.5 * (y_lo + y_hi))
    u_c = float(0.5 * (u_lo + u_hi))
    if surface is None:
        surface = float(frame.floor_z(np.asarray([y_c]))[0])
    surface = float(surface)
    # Низ -- до опоры: у человека на 56 м видимых точек ниже 0.2 м нет, а стоять
    # он должен на поверхности, иначе бокс висит в воздухе.
    h_lo = max(0.0, float(h_lo))
    length = float(np.clip(y_hi - y_lo, SIZE_MIN[2], SIZE_MAX[2]))
    width = float(np.clip(u_hi - u_lo, SIZE_MIN[0], SIZE_MAX[0]))
    height = float(np.clip(float(h_hi) - h_lo, SIZE_MIN[1], SIZE_MAX[1]))
    floor_c = surface
    z_lo, z_hi = floor_c + h_lo, floor_c + max(float(h_hi), h_lo)
    base = z_lo if base_z is None else max(floor_c, min(float(base_z), z_lo))
    h_c = base + 0.5 * height
    center = [float(frame.axis_x(np.asarray([y_c]))[0]) + u_c, y_c, h_c]
    slope = frame.slope(y_c)
    grade = frame.grade(y_c)
    yaw = math.degrees(math.atan2(-slope, 1.0))      # 0 -- длина бокса вдоль пути
    pitch = math.degrees(math.atan2(grade, 1.0))
    return {
        'center': [float(v) for v in center],
        'size': [float(width), float(height), float(length)],
        'yaw': float(yaw), 'pitch': float(pitch), 'roll': 0.0,
        'base_z': float(base),
        'extents': {'u': [float(u_lo), float(u_hi)], 'h_floor': [float(h_lo), float(h_hi)],
                    'y': [float(y_lo), float(y_hi)], 'z': [z_lo, z_hi],
                    'floor_z': floor_c},
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
      `center`/`size`/`yaw`/`pitch`/`roll` -- предложенная геометрия (ОСНОВАНИЕ
                      бокса стоит на поверхности клика: `center.z = z_ref + высота/2`);
      `click` -- точка клика, `z_ref` -- высота ОПОРНОЙ поверхности, на которую
                      встал объект, `base` -- `[x, y, z_ref]` этой опоры;
      `surface` -- из чего выбрана опора (`click` -- точка клика, `cluster` --
                      низ кластера, ниже клика) плюс числа: клик, пол ПОД
                      ОБЪЕКТОМ (измеренный: лоток/полка/уклон учтены), пол по
                      модели пути, низ кластера;
      `path` -- та же точка в путевых координатах (вдоль, поперёк, над полом);
      `cluster` -- сколько точек и какие размахи дал кластер;
      `search` -- откуда взят кластер: `click` (под лучом), `plan` (ближайший в
                      плане, клик попал в отблеск рядом), `ball` (шар вокруг
                      клика -- точек мало, связность не собирает), None (не нашли);
      `size_source` -- откуда габарит: `cluster` (по найденному кластеру),
                      `band` (точки рядом с кликом, ручная постановка),
                      `default` (мерить нечего -- габарит по умолчанию);
      `detector` -- что нашёл детектор на этом кадре (для панели).
    """
    model = model if model is not None else _model_for(db_path)
    xyz = np.asarray(xyz, dtype=np.float64)
    frame = path_frame(xyz, model)
    click = [float(v) for v in np.asarray(click, dtype=np.float64).reshape(3)]
    model_floor = float(frame.floor_z(np.asarray([click[1]]))[0])
    # Опора -- ИЗМЕРЕННАЯ поверхность под местом объекта, а не модельный пол:
    # пол тоннеля наклонный, между рельсами он ниже модельного на 0.16..0.40 м.
    click_u = float(click[0] - frame.axis_x(np.asarray([click[1]]))[0])
    measured, meas_info = _surface_z(xyz, frame, click[1], click_u) \
        if xyz.shape[0] else (None, {})
    click_floor = model_floor if measured is None else float(measured)
    z_click = max(click_floor, click[2])
    if xyz.shape[0] == 0:
        return {
            'ok': False, 'automation': 'manual', 'reason': 'кадр пуст',
            'center': [click[0], click[1], z_click + 0.5 * MANUAL_SIZE[1]],
            'size': list(MANUAL_SIZE), 'size_source': 'default',
            'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0,
            'click': click, 'z_ref': z_click, 'base': [click[0], click[1], z_click],
            'surface': {'click_z_m': float(click[2]), 'floor_z_m': click_floor,
                        'model_floor_z_m': model_floor, 'measured': False,
                        'used': 'click'},
            'cluster': None, 'detector': None, 'search': None,
            'path': None, 'pose_suggest': None,
        }

    min_points = (cfg or {}).get('min_points', 8)
    cell = (cfg or {}).get('cell', CLUSTER_CELL)
    # Три попытки подряд: под лучом (клик внутри объекта) -> ближайший кластер в
    # плане (клик попал в отблеск рядом) -> шар вокруг клика (точек мало, связность
    # не работает). Первая, что набрала точек, и даёт габарит.
    got = _cluster_around(xyz, click, frame, cell=cell,
                          seed_r=(cfg or {}).get('seed_r', SEED_R_M),
                          surface=click_floor)
    search = 'click' if (got is not None and got[0].size >= min_points) else None
    found = 0 if got is None else int(got[0].size)
    if search is None:
        got = _cluster_plan(xyz, click, frame, cell=cell, surface=click_floor)
        found = max(found, 0 if got is None else int(got[0].size))
        if got is not None and got[0].size >= min_points:
            search = 'plan'
    if search is None:
        got = _cluster_ball(xyz, click, frame, cell=cell, surface=click_floor)
        found = max(found, 0 if got is None else int(got[0].size))
        if got is not None and got[0].size >= min_points:
            search = 'ball'
    if search is None:
        # Ничего: объект ставится руками, но габарит -- по тому, что рядом ИЗМЕРИМО
        # (полоса вокруг клика), а не жёсткой константой. Константа -- последний
        # запас, и `size_source` говорит панели, что габарит не измерен.
        size, z_ref, source, n_band = _manual_geometry(xyz, click, frame, z_click,
                                                       surface=click_floor)
        out = {
            'ok': False, 'automation': 'manual',
            'reason': (f'ни под кликом, ни в {PLAN_R_M:.1f} м рядом нет кластера '
                       f'(максимум {found} точек)'
                       + ('; габарит взят по точкам рядом' if source == 'band'
                          else '; точек рядом нет — габарит по умолчанию')),
            'center': [click[0], click[1], z_ref + 0.5 * size[1]],
            'size': list(size), 'size_source': source,
            'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0,
            'click': click, 'z_ref': z_ref, 'base': [click[0], click[1], z_ref],
            'surface': {'click_z_m': float(click[2]), 'floor_z_m': click_floor,
                        'model_floor_z_m': model_floor,
                        'measured': measured is not None,
                        'surface_win_m': [meas_info.get('win_y_m'),
                                          meas_info.get('win_u_m')],
                        'surface_points': meas_info.get('n_points'),
                        'used': 'click'},
            'cluster': (None if n_band <= 0 else {'n_points': int(n_band), 'extents': None}),
            'detector': _detect_only(xyz, model, cfg),
            'search': None, 'path': None, 'pose_suggest': None,
        }
        return out

    idx, (cy, cu, chf) = got
    # Поверхность пересчитываем ПОД ЦЕНТРОМ найденного кластера: объект стоит на
    # том, что измерено под ним, а не под точкой клика (у человека в габарите они
    # расходятся, если клик попал в отблеск рядом). Из двух измерений берём НИЖНЕЕ:
    # клик по полу обязан дать опору ровно на ЭТОМ полу, даже если рядом нашёлся
    # кластер, чей центр лежит на полке выше (`doubleT_platform`: клик по лотку, а
    # кластер -- 2.5x2x3 м по краю платформы).
    y_c0 = 0.5 * float(np.sum(np.percentile(cy, [PERCENTILE_LO, PERCENTILE_HI])))
    u_c0 = 0.5 * float(np.sum(np.percentile(cu, [PERCENTILE_LO, PERCENTILE_HI])))
    surf_c, surf_c_info = _surface_z(xyz, frame, y_c0, u_c0)
    surface = click_floor if surf_c is None else min(float(surf_c), click_floor)
    z_click = max(surface, click[2])
    box = _box_from_cluster(cy, cu, chf, frame, base_z=z_click, surface=surface)
    out = {
        'ok': False, 'automation': 'manual', 'reason': '',
        'size_source': 'cluster', 'search': search,
    }
    out.update({k: box[k] for k in ('center', 'size', 'yaw', 'pitch', 'roll')})
    out['click'] = click
    out['z_ref'] = box['base_z']
    out['base'] = [float(out['center'][0]), float(out['center'][1]), box['base_z']]
    out['surface'] = {
        'click_z_m': float(click[2]), 'floor_z_m': float(box['extents']['floor_z']),
        'model_floor_z_m': model_floor,
        'cluster_bottom_z_m': float(box['extents']['z'][0]),
        'measured': surf_c is not None,
        'surface_click_z_m': float(click_floor),
        'surface_center_z_m': (None if surf_c is None else float(surf_c)),
        'surface_win_m': [surf_c_info.get('win_y_m'), surf_c_info.get('win_u_m')],
        'surface_points': surf_c_info.get('n_points'),
        'used': ('click' if abs(box['base_z'] - z_click) < 1e-9 else 'cluster'),
    }
    out['cluster'] = {
        'n_points': int(idx.size),
        'extents': box['extents'],
        'cell': list(cell),
    }
    y_c = float(np.median(cy))
    u_c = float(np.median(cu))
    # Высота кластера -- над УГР (путевые координаты), поэтому из высоты над
    # опорой её переводим через саму опору.
    h_c = float(np.median(chf)) + float(surface) \
        - float(frame.ugr_z(np.asarray([y_c]))[0])
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
    where = {'click': 'вокруг луча', 'plan': 'рядом с кликом', 'ball': 'в шаре вокруг клика'}
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
        out['reason'] = (f'кластер {int(idx.size)} точек ({where[search]}), '
                         f'детектор объект не подтвердил'
                         f' (ближайшее препятствие: '
                         + (f'y={det["obstacles"][0]["y_m"]:.1f}' if det['obstacles'] else 'нет')
                         + ')')
    return out


def _manual_geometry(xyz, click, frame, z_click, surface=None):
    """Габарит для РУЧНОЙ постановки: то, что реально измеримо рядом с кликом.

    Автоматика не нашла объект -- но это не значит, что мерить нечего: в полосе
    вокруг клика (`MANUAL_BAND_*`, полоса та же, что у кластера, но низ выше --
    покрытие и рельс не объект) обычно есть точки -- по ним и берётся ширина,
    высота и длина. Жёсткая константа -- последний запас, и она же берётся, когда
    полоса хоть и не пуста, но габарита не даёт: точек мало (`MANUAL_MIN_POINTS`)
    или они лежат ПОЛОСОЙ (ни ширины, ни высоты -- это стык плит или кабель, а не
    тело). Габарит-крошка лучше не показывать измеренным.

    `surface` -- измеренная поверхность под кликом: от неё (а не от модельного
    пола) отсчитывается высота полосы, поэтому в лотке полоса не срезает низ тела.

    Возвращает (size, z_ref, источник, сколько точек в полосе).
    """
    band, y, u, hf, (cy, cu, _cw) = _click_band(
        xyz, click, frame, u_half=MANUAL_BAND_U_M, window_y_half=MANUAL_BAND_Y_M,
        h_low=PLAN_H_LOW_M, surface=surface)
    if band is not None:
        n = int(np.count_nonzero(band))
        if n >= MANUAL_MIN_POINTS:
            cu_b, cy_b, ch_b = u[band], y[band], hf[band]
            u_lo, u_hi = np.percentile(cu_b, [PERCENTILE_LO, PERCENTILE_HI])
            y_lo, y_hi = np.percentile(cy_b, [PERCENTILE_LO, PERCENTILE_HI])
            h_lo, h_hi = np.percentile(ch_b, [PERCENTILE_LO, PERCENTILE_HI])
            width = float(np.clip(u_hi - u_lo, SIZE_MIN[0], SIZE_MAX[0]))
            length = float(np.clip(y_hi - y_lo, SIZE_MIN[2], SIZE_MAX[2]))
            height = float(np.clip(h_hi - max(0.0, float(h_lo)), SIZE_MIN[1], SIZE_MAX[1]))
            strip = width <= SIZE_MIN[0] + 1e-9 and height <= SIZE_MIN[1] + 1e-9
            if not strip:
                return (width, height, length), float(z_click), 'band', n
    return tuple(MANUAL_SIZE), float(z_click), 'default', 0


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


# ------------------------------------------------------------ запечённые позы

# Поза человека -- НАСТОЯЩИЙ меш из `assets/_poses/` (манифест
# `assets/_poses/manifest.json`, контракт `docs/poses-contract.md`), а не поворот
# примитива. Здесь: клип по классу и позе (`class_map`), момент по времени,
# склейка соседних моментов и постановка в кадр. Тот же расчёт повторён в клиенте
# (`web/labellayer.js`: poseMoment/poseVertices/placePose) -- иначе нарисованный
# меш разошёлся бы с тем, по которому считана трассировка (а числа закрытых точек
# обязаны совпадать: их считает сервер, по ЭТОМУ мешу).
#
# Числа из манифеста: рост стоящего 1.7492..1.7504 м, вершины УЖЕ в метрах
# (`unit_convention.do_not_scale_again`), начало координат -- под ногами, меш
# Y-up, в кадр переводится как (x, y, z) -> (x, -z, y).
POSE_ROOT = os.path.join(root_dir(), 'assets', '_poses')
POSE_MANIFEST_PATH = os.path.join(POSE_ROOT, 'manifest.json')
POSE_PLAY = ('loop', 'one_shot', 'static')
# Позы, которые в сцене играет ПРАВИЛО, а не человек: у объекта с траекторией
# движение должно выглядеть идущим, у стоящего -- стоящим. Явные lying/fallen/
# running не переписываются: это осознанный выбор тела, а не «по умолчанию».
POSE_WALK, POSE_STAND = 'walking', 'standing'

_POSE_CACHE = {}


def pose_manifest(path=None):
    """Манифест запечённых поз (кэш: файл 31 МБ читается один раз на процесс)."""
    key = os.path.abspath(path or POSE_MANIFEST_PATH)
    hit = _POSE_CACHE.get(('manifest', key))
    if hit is None:
        try:
            with open(key, encoding='utf-8') as fh:
                hit = json.load(fh)
        except (OSError, ValueError):
            hit = {}
        _POSE_CACHE[('manifest', key)] = hit
    return hit


def pose_spec(cls, pose):
    """Описание клипа позы классу: `class_map` манифеста (или None).

    Позы нет в карте -- берётся `unknown` (нейтральный пешеход 1.75 м). Меши поз
    -- ТОЛЬКО человек: оборудование и прочее рисуются примитивом `sim/objects.py`.
    """
    if str(cls) != 'person':
        return None
    cmap = pose_manifest().get('class_map') or {}
    key = str(pose or 'unknown').lower()
    spec = cmap.get(key) or cmap.get('unknown')
    return spec or None


def pose_clip(cls, pose, manifest_path=None):
    """Клип позы: вершины в осях КАДРА, лица и правила проигрывания (или None).

    * `V` (T, N, 3) float32 -- уже в осях кадра ((x, -z, y) из манифеста);
    * `F` (M, 3) int32 -- лица (у всех моментов клипа одни и те же);
    * `times` (T,) -- времена моментов ОТ ПЕРВОГО (у `Death`-моментов 8..13 первый
      локальный момент -- начало прохода, а не 2.154 с исходного клипа);
    * `period` -- длительность цикла (`loop`, по контракту -- `duration_s` клипа)
      или размах времен (`one_shot`); `play`; `samples` -- сколько моментов взято.
    """
    spec = pose_spec(cls, pose)
    if not spec:
        return None
    man = pose_manifest()
    asset = next((a for a in man.get('assets') or [] if a.get('id') == spec.get('asset')), None)
    clip = None if asset is None else next(
        (c for c in asset.get('clips') or [] if c.get('name') == spec.get('clip')), None)
    if clip is None:
        return None
    bundle = clip.get('bundle_npz')
    path = None if not bundle else os.path.join(root_dir(), bundle)
    key = ('clip', os.path.abspath(path) if path else None,
           tuple(spec.get('samples') or ()) if isinstance(spec.get('samples'), (list, tuple)) else 'all')
    if key in _POSE_CACHE:
        return _POSE_CACHE[key]
    if not path or not os.path.isfile(path):
        _POSE_CACHE[key] = None
        return None
    with np.load(path) as z:
        times = np.asarray(z['times'], dtype=np.float64).reshape(-1)
        V = np.asarray(z['vertices'], dtype=np.float32).reshape(times.size, -1, 3)
        F = np.asarray(z['faces'], dtype=np.int32).reshape(-1, 3)
    samples = spec.get('samples')
    if isinstance(samples, (list, tuple)):
        idx = np.asarray([int(i) for i in samples], dtype=np.int64)
        idx = idx[(idx >= 0) & (idx < V.shape[0])]
        if idx.size:
            V, times = V[idx], times[idx]
    # Y-up (исходный FBX) -> Z-up кадра: (x, y, z) -> (x, -z, y).
    V = np.stack([V[:, :, 0], -V[:, :, 2], V[:, :, 1]], axis=2).astype(np.float32)
    play = str(spec.get('play') or ('loop' if clip.get('loop') else 'one_shot')).lower()
    if play not in POSE_PLAY:
        play = 'one_shot'
    if V.shape[0] <= 1:
        play = 'static'
    times = times - float(times[0])
    period = (float(clip.get('duration_s') or 0.0) if play == 'loop' else float(times[-1]))
    out = {
        'V': V, 'F': F, 'times': times, 'period': float(period), 'play': play,
        'asset': str(spec.get('asset')), 'clip': str(spec.get('clip')),
        'samples': int(V.shape[0]), 'duration_s': float(clip.get('duration_s') or 0.0),
        'path': os.path.relpath(path, root_dir()).replace('\\', '/'),
        'height_m': float(V[:, :, 2].max() - V[:, :, 2].min()) if V.shape[0] else 0.0,
        'play_hint': str(spec.get('play') or ''),
        'why': str(spec.get('why') or ''),
    }
    _POSE_CACHE[key] = out
    return out


def pose_rule(cls, pose, has_trajectory=False):
    """Какая поза играет в сцене: правило, а не ручная настройка на каждый кадр.

    Есть траектория -> объект ЕДЕТ, и он должен выглядеть идущим (`walking`);
    траектории нет -> стоит (`standing`). Так же читается `unknown` -- это поза по
    умолчанию («не выбрана»), а не осознанное тело. Явные `lying`/`fallen`/
    `running` не переписываются: лежащий на траектории -- тоже возможный случай.
    Та же функция в клиенте: `web/labellayer.js:poseInScene`.
    """
    key = str(pose or 'unknown').lower()
    if key in ('standing', 'walking', 'unknown'):
        return POSE_WALK if has_trajectory else POSE_STAND
    if key == 'running':
        return 'running'
    return key if key in POSES else POSE_STAND


def pose_moment(clip, t):
    """(индекс момента, доля склейки, время внутри клипа) по времени `t`, с.

    `loop` -- по контракту: индекс = floor((t mod d)/d * T), где d -- длительность
    клипа (последний интервал отдаётся последнему моменту, склейка -- в первый).
    `one_shot` -- один проход по размаху времен: доля t/span линейно ложится на
    выбранные моменты, в конце -- последний момент. `static` -- всегда первый.
    """
    V = clip['V']
    total = int(V.shape[0])
    play = clip['play']
    period = float(clip['period'])
    t = float(t or 0.0)
    if play == 'static' or total <= 1 or period <= 0.0:
        return 0, 0.0, 0.0
    if play == 'loop':
        span = math.fmod(t, period)
        if span < 0.0:
            span += period
        frac = span / period
        i = min(int(frac * total), total - 1)
        return i, float(frac * total - i), float(span)
    frac = min(max(t / period, 0.0), 1.0)
    x = frac * (total - 1)
    i = min(int(x), total - 2)
    return i, float(x - i), float(min(max(t, 0.0), period))


def pose_vertices(clip, t):
    """Вершины позы на времени `t`: линейная склейка соседних моментов (N, 3).

    Линейная склейка -- то, что предписал контракт (`playback.interpolation`):
    максимальное смещение вершины за шаг у ходьбы 0.151 м, у стояния 0.011 м.
    """
    V = clip['V']
    total = int(V.shape[0])
    i, w, t_in = pose_moment(clip, t)
    if total <= 1 or w <= 0.0:
        return np.asarray(V[min(i, total - 1)], dtype=np.float32), i, 0.0, t_in
    j = (i + 1) % total if clip['play'] == 'loop' else min(i + 1, total - 1)
    blended = (V[i].astype(np.float64) * (1.0 - w) + V[j].astype(np.float64) * w)
    return blended.astype(np.float32), i, float(w), t_in


def place_pose(vertices, center, base_z, yaw=0.0, pitch=0.0, roll=0.0):
    """Поставить меш позы в кадр: низ на опору, середина габарита -- в центр.

    Три шага, каждый проверяем числом (тот же расчёт в клиенте,
    `web/labellayer.js:placePose`):

    1. начало координат меша -> якорь `[середина габарита по X и Y, низ по Z]`:
       у стоящего это «под ногами» (origin манифеста), у лежащего -- середина
       тела, иначе лежащий вылез бы из своего бокса на метр с лишним;
    2. поворот `R = Rx(roll)·Ry(pitch)·Rz(yaw)` ВОКРУГ ЯКОРЯ (человек остаётся
       стоять там, где стоял: поворот не сносит его вбок);
    3. сдвиг в `[center.x, center.y, base_z]` так, что НИЗ меша встаёт РОВНО на
       опору: подъём = `base_z - (base_z + мин_z_после_поворота)`. Обычно это
       подъём вверх (лежащий провалился бы в пол на 0.065 м), но у тела, повёрнутого
       на пару градусов, низ уходит на миллиметры ВЫШЕ опоры -- и тогда сдвиг
       опускает меш обратно: «основание на z_ref» -- это равно, а не «не ниже».

    Размеры бокса (`size`) здесь НЕ участвуют: вершины запечены в метрах, и
    `unit_convention.do_not_scale_again` запрещает масштабировать их ещё раз.
    Возвращает (точки (N, 3), подъём, якорь).
    """
    V = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    anchor = np.array([0.5 * (float(V[:, 0].min()) + float(V[:, 0].max())),
                       0.5 * (float(V[:, 1].min()) + float(V[:, 1].max())),
                       float(V[:, 2].min())], dtype=np.float64)
    rotated = (V - anchor) @ _rotation(yaw, pitch, roll).T
    lift = float(base_z) - (float(base_z) + float(rotated[:, 2].min()))
    placed = rotated + np.array([float(center[0]), float(center[1]),
                                 float(base_z) + lift], dtype=np.float64)
    return placed, float(lift), anchor


def pose_mesh(cls, pose, size, center, yaw=0.0, pitch=0.0, roll=0.0, t=0.0,
              base_z=None):
    """Меш человека по ЗАПЕЧЁННОЙ позе + сведения о клипе (None, если позы нет).

    `t` -- время позы от опорного кадра объекта, с (см. `pose_moment`).
    `base_z` -- высота опорной поверхности; по умолчанию -- низ бокса
    (`center.z - size.z/2`), то есть ровно то место, где стоял примитив.
    """
    import open3d as o3d

    clip = pose_clip(cls, pose)
    if clip is None:
        return None
    verts, i, w, t_in = pose_vertices(clip, t)
    base = float(base_z) if base_z is not None else float(center[2]) - 0.5 * float(size[1])
    placed, lift, anchor = place_pose(verts, center, base, yaw, pitch, roll)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(placed)
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(clip['F'], dtype=np.int32))
    mesh.compute_vertex_normals()
    info = {
        'source': 'poses', 'asset': clip['asset'], 'clip': clip['clip'],
        'file': clip['path'], 'play': clip['play'], 'play_hint': clip['play_hint'],
        'samples': clip['samples'], 'moment': int(i), 'blend': float(w),
        'pose_time_s': float(t), 'time_in_clip_s': float(t_in),
        'period_s': float(clip['period']), 'duration_s': clip['duration_s'],
        'times_s': [round(float(v), 4) for v in clip['times']],
        'height_m': float(clip['height_m']),
        'base_z_m': base, 'lift_m': float(lift),
        'anchor': [round(float(v), 4) for v in anchor],
        'vertices': int(placed.shape[0]),
        'faces': int(np.asarray(clip['F']).shape[0]),
        'rule': ('loop: индекс = floor((t mod period)/period * T); '
                 'one_shot: доля t/span линейно по выбранным моментам; '
                 'static: только выбранный момент'),
    }
    return mesh, info


def object_geometry(cls, pose, size, center, yaw=0.0, pitch=0.0, roll=0.0,
                    pose_time=0.0, base_z=None, mesh_source=None):
    """Меш объекта: ЗАПЕЧЁННАЯ поза человека или примитив `sim/objects.py`.

    Возвращает (mesh, info): `info['source']` -- `poses` (вершины из
    `assets/_poses`, `pose_time` секунд от опорного кадра) или `primitive`
    (жёсткий наклон примитива -- запас, когда позы нет).

    `mesh_source='primitive'` -- явный выбор примитива вместо позы: нужен, чтобы
    перегенерировать карточку, записанную ещё по примитиву, с ТЕМИ ЖЕ точками
    (меш позы даст другой силуэт и другое число точек). Без него -- как всегда:
    у человека поза, у остальных примитив.
    """
    if str(cls) == 'person' and str(mesh_source or '').lower() != 'primitive':
        got = pose_mesh(cls, pose, size, center, yaw, pitch, roll, t=pose_time,
                        base_z=base_z)
        if got is not None:
            return got
    mesh, prim_size = object_mesh(cls, size, center, yaw, pitch, roll, pose=pose)
    return mesh, {
        'source': 'primitive', 'asset': None, 'clip': None, 'pose': str(pose),
        'play': None, 'samples': 1, 'moment': 0, 'blend': 0.0,
        'pose_time_s': float(pose_time), 'primitive_size': list(prim_size),
        'height_m': (float(prim_size[1]) if prim_size else None),
        'lift_m': 0.0,
        'rule': 'примитив sim/objects.py с жёстким наклоном позы',
    }


# ------------------------------------------------------------ трассировка

# Поза -> поворот тела ВОКРУГ ОСНОВАНИЯ (низа бокса): ось, угол в градусах.
# Лёжа/упал -- четверть оборота: тело ложится вдоль пути (вокруг поперечной оси X)
# или поперёк (вокруг продольной оси Y). Идёт/бежит -- наклон тела вперёд вокруг
# поперечной оси (шаг у примитива не отрисовать: настоящие позы -- запечённые меши
# из `assets/`, см. tools/bake_pose.js; здесь честный минимум -- жёсткий наклон).
POSE_POSE_ROTATION = {
    'lying': ('x', 90.0), 'fallen': ('y', 90.0),
    'walking': ('x', 12.0), 'running': ('x', 22.0),
}


def object_mesh(cls, size, center, yaw=0.0, pitch=0.0, roll=0.0, pose='unknown'):
    """Меш объекта класса `cls`: примитив `sim/objects.py`, повёрнутый и поставленный.

    `size` -- (ширина поперёк, высота, длина вдоль). Человек -- цилиндр туловища
    со сферой головы (`_person`), всё остальное -- бокс (`_suitcase`); размеры
    берутся из разметки, а не из библиотечных умолчаний. Локальные оси меша:
    X -- поперёк, Y -- вдоль, Z -- вверх, начало координат -- ЦЕНТР бокса (у
    примитивов начало на полу, поэтому поднимаем на полвысоты).

    `pose` МЕНЯЕТ ГЕОМЕТРИЮ, а не только карточку: `lying`/`fallen` кладут тело
    на четверть оборота (вдоль пути или поперёк), `walking`/`running` наклоняют
    его вперёд (`POSE_POSE_ROTATION`). Поворот позы идёт вокруг ОСНОВАНИЯ бокса, и
    поза, провалившаяся ниже опоры, поднимается ровно на провал: тело лежит/стоит
    на той поверхности, где стоит объект, а не в полу. `size` при этом не
    переписывается: бокс -- объявленный габарит объекта, и поза вправе занимать
    только его часть (иначе размер, раздутый позой, поехал бы в карточку и в
    детектор).

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
    base_local = (0.0, 0.0, -height / 2.0)           # низ бокса -- опорная поверхность
    axis_angle = POSE_POSE_ROTATION.get(str(pose or 'unknown').lower())
    if axis_angle:
        axis, deg = axis_angle
        rot_pose = mesh.get_rotation_matrix_from_axis_angle(
            _axis_angle_vector(axis, deg))
        mesh.rotate(rot_pose, center=base_local)
        # Тело повёрнуто вокруг опоры, но само оно объёмное: пол-толщины у лежащего
        # (и уголок -- у наклонившегося) оказалось бы НИЖЕ опоры, то есть в полу.
        # Поднимаем ровно на провал: низ позы встаёт на ту же поверхность, на
        # которой стоял объект. Только вверх -- поза, которая и так выше опоры,
        # не опускается.
        drop = float(base_local[2] - mesh.get_axis_aligned_bounding_box().min_bound[2])
        if drop > 0.0:
            mesh.translate((0.0, 0.0, drop))
    rot = mesh.get_rotation_matrix_from_xyz(
        (math.radians(roll), math.radians(pitch), math.radians(yaw)))
    mesh.rotate(rot, center=(0.0, 0.0, 0.0))
    mesh.translate(tuple(float(v) for v in center))
    mesh.compute_vertex_normals()
    return mesh, tuple(float(v) for v in om.size_m)


def _axis_angle_vector(axis, degrees):
    """Вектор поворота для open3d: направление -- ось, длина -- угол в радианах.

    Обычный numpy-массив, НЕ `o3d.utility.Vector3dVector`: в open3d 0.19 этот
    конструктор на векторе чисел падает (`Unable to cast Python instance of type
    <class 'float'>`), а `get_rotation_matrix_from_axis_angle` numpy принимает.
    """
    v = {'x': (1.0, 0.0, 0.0), 'y': (0.0, 1.0, 0.0), 'z': (0.0, 0.0, 1.0)}[axis]
    return np.asarray(v, dtype=np.float64) * math.radians(float(degrees))


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
    кадра, закрытых объектом -- тень), `n_rays`, `n_points`.

    ТЕНЬ (`removed`) -- все точки кадра, чей ключ луча (кольцо, ячейка азимута
    0.1°) совпал с лучом, попавшим в объект. Это ТОЧНО точки ЗА объектом: луч
    считается попавшим, если поверхность объекта ближе БЛИЖАЙШЕГО возврата кадра
    на этом луче (`t_hit < ranges - 1e-3`), а ближайший возврат -- представитель
    ячейки; значит все возвраты этой ячейки дальше поверхности, то есть за телом.
    Проверяемо числом: `points_removed` ≈ `rays_hit` × средняя кратность луча
    (у этих записей кратность почти ровно 2, без второго возврата за `max_dist`
    он в кадр не попадает -- разницу панель показывает как `points_removed_drawn`).
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
            intensity=None, ring=None, model=None, db_path=None, cfg=None,
            pose_time_s=0.0, base_z=None, mesh_source=None):
    """Что увидел бы лидар: точки объекта по лучам кадра + ответ детектора.

    Точки считаются трассировкой (см. `trace_object`), шум дальности НЕ
    добавляется: разметка -- эталон, а не измерение; те же точки (без шума) идут
    в `.npz` при сохранении, поэтому «что видно в панели» = «что записано».

    Меш -- по КЛАССУ И ПОЗЕ (`object_geometry`): у человека это запечённый меш
    (`assets/_poses`, момент по `pose_time_s`), у остальных -- примитив
    `sim/objects.py`. `primitive_bbox` -- габарит ЭТОГО меша (у человека -- позы),
    им поза видна числом; `pose_info` -- клип, момент, время, правило склейки.

    `removed_mask` -- маска точек КАДРА, закрытых объектом (numpy bool по `xyz`,
    НЕ для JSON: маршрут `/label/preview` переводит её в индексы того массива,
    который рисует страница, и отдаёт как `removed_indices`). В маске и ТЕНЬ: все
    точки кадра на лучах, попавших в объект (см. `trace_object`).
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    mesh, mesh_info = object_geometry(cls, pose, size, center, yaw, pitch, roll,
                                      pose_time=pose_time_s, base_z=base_z,
                                      mesh_source=mesh_source)
    trace = trace_object(xyz, ring, mesh)
    refl = reflectance_for(xyz, intensity, trace['ray_index'], trace['mult'])
    bbox = mesh.get_axis_aligned_bounding_box()
    box = {'min': [round(float(v), 4) for v in bbox.min_bound],
           'max': [round(float(v), 4) for v in bbox.max_bound],
           'extent': [round(float(v), 4) for v in bbox.get_extent()]}
    out = {
        'class': cls, 'pose': pose,
        'center': [float(v) for v in center],
        'size': [float(v) for v in size],
        'yaw': float(yaw), 'pitch': float(pitch), 'roll': float(roll),
        'primitive_size': mesh_info.get('primitive_size'),
        'primitive_bbox': box,
        'mesh_bbox': box,
        'mesh_source': mesh_info.get('source'),
        'pose_info': mesh_info,
        'removed_mask': trace['removed'],
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


# ---------------------------------------------------------------- траектория

# Траектория -- «откуда и куда идёт и сколько секунд он идти будет»: две точки
# (ставятся по облаку, как объект) и длительность. Отсюда скорость = расстояние /
# время. Точки -- ОСНОВАНИЕ объекта (опорная поверхность, `z_ref`), поэтому объект,
# едущий по траектории, стоит на том же, на чём стоял в точке постановки.

TRAJ_MIN_SECONDS = 0.05            # короче -- деления на ноль, а не длительность
TRAJ_RULE = ('base(f) = start + (end - start)*clamp(t/seconds, 0, 1)'
             ' + (0, speed_m_per_frame*(f - ref_frame), 0), t = (f-ref_frame)/hz')


def trajectory_normalize(traj):
    """Траектория из панели -> запись для карточки (или None, если её нет).

    Панель присылает `{'start': [x, y, z_ref], 'end': [...], 'seconds': s}`.
    Скорость считается ЗДЕСЬ, а не в панели: расстояние по двум точкам и время --
    это и есть определение скорости, и в карточку обязан попасть тот же расчёт.
    """
    if not isinstance(traj, dict):
        return None
    start, end = traj.get('start'), traj.get('end')
    if not (isinstance(start, (list, tuple)) and len(start) == 3
            and isinstance(end, (list, tuple)) and len(end) == 3):
        return None
    seconds = float(traj.get('seconds') or 0.0)
    if not (seconds > 0.0):
        return None
    a = [float(v) for v in start]
    b = [float(v) for v in end]
    distance = float(math.dist(a, b))
    return {
        'start': a, 'end': b, 'seconds': float(seconds),
        'distance_m': distance, 'speed_mps': distance / seconds,
        'speed_kmh': distance / seconds * 3.6,
        'source': str(traj.get('source') or 'manual'),
    }


def trajectory_at(traj, t):
    """Точка основания на траектории во время `t`, с: линейная склейка концов.

    До начала -- начало, после конца -- КОНЕЦ (а не продолжение прямой): «в
    конце -- на конечной точке». Интерполяция по ВРЕМЕНИ, а не по номеру кадра,
    поэтому картинка не зависит от темпа воспроизведения.
    """
    a = np.asarray(traj['start'], dtype=np.float64)
    b = np.asarray(traj['end'], dtype=np.float64)
    seconds = float(traj['seconds'])
    frac = 0.0 if seconds <= 0.0 else min(max(float(t) / seconds, 0.0), 1.0)
    return (a + (b - a) * frac).tolist()


def _profile_round(prof, digits=3):
    """Профиль хода для карточки: числа с округлением, `None` -- без значения.

    Профиль -- это ход ЗАПИСИ (общий для всех объектов), но в карточке он нужен:
    по нему видно, на каких кадрах объект ехал, а на каких запись стояла, и
    воспроизводится проекция объекта на кадр (`rule_profile`).
    """
    if not prof:
        return None
    return [None if v is None else round(float(v), digits) for v in prof]


def _instance_center(center, ref_frame, frame, speed, traj=None, hz=HZ_DEFAULT,
                     size=None, motion=None):
    """Центр объекта на кадре: траектория (если задана) + ход сенсора.

    Без траектории -- ровно как раньше (`_instance_at`). С траекторией объект
    едет по ней: основание = точка траектории в момент `t = (f-ref)/hz`, а к ней
    добавляется ход сенсора вдоль Y (траектория записана в системе ОПОРНОГО кадра,
    а сенсор за кадр сдвигается на `speed_m_per_frame`).

    `motion` -- профиль хода записи (`measure_motion`): тогда сдвиг сенсора --
    ИНТЕГРАЛ профиля между кадрами (`motion_between`), а не `speed * (f - ref).
    Профиль нужен потому, что ход меняется по кадрам (поезд разгоняется и тормозит):
    одно число на запись уводит объект тем сильнее, чем дальше кадр от опорного.
    """
    if motion is not None:
        dy = motion_between(motion, ref_frame, frame, default=speed)
    else:
        dy = float(speed) * (int(frame) - int(ref_frame))
    if traj is None:
        return [float(center[0]), float(center[1]) + dy, float(center[2])]
    t = (int(frame) - int(ref_frame)) / float(hz or HZ_DEFAULT)
    base = trajectory_at(traj, t)
    half = 0.5 * float(size[1]) if size else (float(center[2]) - float(base[2]))
    return [float(base[0]), float(base[1]) + dy, float(base[2]) + float(half)]


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
        'format': dict(FORMAT_INFO),
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

Читатель формата -- `labels_io.py` в корне проекта: `LabelsReader` собирает
для любой пары (запись, кадр) облако из `.db3`, позиции объектов, их точки
и маску закрытого объектами фона.

## Что где

| файл | что внутри |
|---|---|
| `index.jsonl` | по строке на объект: `id`, `record`, `class`, `pose`, `pose_scene`, `frames`, `n_points`, `automation`, `format` (номер версии), `trajectory` (начало, конец, секунды, скорость) |
| `<record>/<id>.json` | карточка: `format` (версия схемы, подробно), геометрия по кадрам (`instances` с пер-кадровыми счётчиками), трек, меш позы (`mesh`), траектория, ревизия |
| `<record>/<id>.npz` | точки объекта в ЛОКАЛЬНОЙ системе объекта: `points` (N, 3), `reflectance` (N,) если есть; с версии 2 -- ещё пер-кадровый GT: `frames`, `gt_offsets`, `gt_ray_keys`, `gt_t_hit`, `gt_mult` |
| `manifest.json` | счётчики по классам и записям, `format`, ревизия |

## Формат и версии

Поле `format` есть в карточке (подробно: номер, ключи .npz, что менялось), в
строке `index.jsonl` (номер) и в `manifest.json`. Карточка без `format` --
версия 1: трассировка только на опорном кадре (`points` в .npz), тень фона --
одним числом `counts.points_removed`. Версия 2 добавила трассировку на КАЖДОМ
кадре из `frames`: в .npz рядом с точками лежат ЛУЧИ (ключ `ring*3600 +
ячейка азимута 0.1°`, как в трассировке), дальности до поверхности объекта и
кратность возвратов -- по кадру; границы кадров -- `gt_offsets`. Точки и тень
восстанавливаются по самому кадру (направление луча -- представитель кадра),
поэтому GT хранится компактно и не зависит от порядка точек в облаке.

## Как читать

```python
import labels_io
reader = labels_io.LabelsReader('labels', 'for_hackathon')
scene = reader.frame('doubleT_obstacle', 15)   # облако + объекты + тень
for obj in scene['objects']:
    print(obj['id'], obj['center'], obj['points'].shape,
          int(obj['shadow_mask'].sum()))
```

Вручную: `points` -- в системе объекта, перевод в систему кадра описан ниже.

`points` -- в системе объекта: начало в центре бокса, X поперёк пути, Y вдоль,
Z вверх; ориентация -- `yaw`/`pitch`/`roll` из соответствующего `instance`
(`R = Rx(roll) @ Ry(pitch) @ Rz(yaw)`). В систему кадра точки переводятся как
`points @ R.T + center`.

`instances[]` -- по одному на кадр из `frames`: `center` (система лидара этого
кадра), `yaw`/`pitch`/`roll`, `size` (ширина поперёк, высота, длина вдоль),
`automation` и признак подтверждения детектором.

## Меш: запечённая поза человека

У объекта класса `person` точки считаны трассировкой по НАСТОЯЩЕМУ мешу позы из
`assets/_poses/` (манифест `assets/_poses/manifest.json`, контракт
`docs/poses-contract.md`): клип по классу и позе (`class_map`), момент по времени
`pose_time_s` от опорного кадра, склейка соседних моментов. Вершины уже в метрах
(`unit_convention.do_not_scale_again`) и НЕ масштабируются под `size`: бокс --
объявленный габарит, поза -- измеренное тело. Что именно сыграно, видно в карточке
(`mesh.clip`, `mesh.moment`, `mesh.play`, `mesh.pose_time_s`, `mesh.height_m`) и в
панели вьюера. Поза в сцене выбирается ПРАВИЛОМ (`pose_rule`): у объекта с
траекторией играет `walking`, без неё -- `standing`; явные `lying`/`fallen`/
`running` не переписываются. При отсутствии данных позы меш падает на примитив
`sim/objects.py` (`mesh.source = primitive`) -- это запас, а не рабочий путь.

## Траектория движения

У объекта может быть траектория: `start` и `end` (точки ОСНОВАНИЯ объекта,
ставятся по облаку, как сам объект) и `seconds` -- сколько он идёт. Их даёт
`trajectory`: `distance_m = |end - start|`, `speed_mps = distance_m / seconds`,
`speed_kmh`, `speed_m_per_frame = speed_mps / hz`; правило проекции на кадр --
`trajectory.rule`:

```
base(f) = start + (end - start) * clamp(t / seconds, 0, 1)
          + (0, speed_m_per_frame * (f - ref_frame), 0),   t = (f - ref_frame) / hz
```

То есть позиция интерполируется ПО ВРЕМЕНИ (не по номеру кадра): при разном темпе
воспроизведения объект стоит в одном и том же месте; после `seconds` -- на конечной
точке. Второе слагаемое -- ход сенсора: траектория записана в системе ОПОРНОГО
кадра, а сенсор за кадр сдвигается по Y на замеренный ход записи.

## Трек и ход

Объект живёт в путевой системе: `track.speed_m_per_frame` -- ход сенсора
вдоль пути (положительный -- сенсор приближается), источник `measured`
(совмещение профиля стены между кадрами, `check_motion.py`) или `manual`
(число из панели). Проекция на кадр -- сдвиг вдоль Y:
`y(f) = y_ref + speed*(f - ref)`, `x` и `z` не меняются: объект стоит на месте
пути, а не на месте экрана. У объекта с траекторией к этому сдвигу добавляется
свой ход по траектории (см. выше).

## Что есть фон (негативы)

**Неотмеченное -- негативы.** Точки кадра, не попавшие ни в один объект,
считаются фоном: туннель, пол, рельсы, стены, платформа. Точки объекта -- это
ТРАССИРОВКА реальных лучей кадра по мешу объекта (у человека -- по запечённой позе,
см. выше) с окклюзией, без шума дальности: разметка -- эталон, а не измерение.

Точки фона, закрытые объектом, в разметку как фон не идут: это ТЕНЬ -- все точки
кадра на лучах, попавших в объект (ключ луча -- кольцо и ячейка азимута 0.1°;
поверхность объекта ближе ближайшего возврата кадра на этом луче, значит все точки
луча лежат ЗА объектом). С версии 2 тень -- МАСКА по каждому кадру: ключи лучей
лежат в .npz (`gt_ray_keys`), маска восстанавливается по кадру; их число --
`points_removed` в `preview` и в `instances[]` карточки. В панели вьюера
эти точки гасятся (`removed_indices`), причём по ВСЕМ объектам кадра сразу.

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


# id карточки становится ИМЕНЕМ ФАЙЛА (labels/<record>/<id>.json), а приходит он
# из тела запроса: без проверки id вида '../../x' писал бы вне labels/. Разрешаем
# только то, что сами и генерируем.
_ID_RE = re.compile(r'[A-Za-z0-9_-]{1,64}')


def save(record, objects, xyz_at, out_dir=MAIN_LABELS_DIR, hz=HZ_DEFAULT,
         motion=None, revision_info=None):
    """Записать разметку в `labels/`: карточки, npz, index.jsonl, manifest, README.

    `objects` -- список объектов от панели:
      `class`, `pose`, `ref_frame`, `center`, `size`, `yaw`/`pitch`/`roll`,
      `speed_m_per_frame`, `speed_source` (`measured`|`manual`), `frames`,
      `automation`, `z_ref`, `trajectory` (`start`/`end`/`seconds` -- «откуда,
      куда и сколько секунд»).
    Точки объекта считаются ЗДЕСЬ (трассировкой по кадрам разметки): то, что
    видел клиент в предпросмотре, -- без шума, те же лучи, тот же меш (у человека --
    запечённый меш позы, см. `object_geometry`).

    Возвращает отчёт: пути файлов, id, число точек, счётчики.
    """
    # Каталог записи приходит из тела запроса: относительный путь считаем от
    # корня проекта, а выход НАРУЖУ запрещаем -- иначе клиент мог бы писать
    # файлы куда угодно.
    root = root_dir()
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(root, out_dir)
    out_dir = os.path.abspath(out_dir)
    if out_dir != root and not out_dir.startswith(root + os.sep):
        raise ValueError(f'out_dir должен лежать внутри проекта: {out_dir!r}')
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
            # Траектория: две точки и секунды. Скорость -- отсюда, а не от панели.
            traj = trajectory_normalize(obj.get('trajectory'))
            if (isinstance(obj.get('trajectory'), dict) and traj is None):
                raise ValueError('траектория: нужны start и end (по три числа) '
                                 f'и seconds > 0, получено {obj.get("trajectory")!r}')
            # Поза, которая играет в СЦЕНЕ: правило (идёт по траектории / стоит).
            # Панель считает его тем же правилом (`pose_rule`) и присылает готовым.
            pose_scene = str(obj.get('pose_scene') or pose_rule(cls, pose, bool(traj)))
            if pose_scene not in POSES:
                raise ValueError(f'поза сцены {pose_scene!r}')

            xyz, inten, ring = frame_points(xyz_at, ref)
            if model is None:
                model = _model_for(obj.get('db_path') or '')
            frame_pf = path_frame(xyz, model)
            # Трассировка -- на КАЖДОМ кадре из frames (v1 трассировала только
            # опорный). Меш на кадре -- по позе СЦЕНЫ (та же, что в предпросмотре
            # и на экране: «человек + траектория» даёт точки ИДУЩЕГО), центр -- по
            # правилу проекции (`_instance_center`: траектория по времени + ход
            # сенсора интегралом профиля), момент позы -- `pose_time_s` от
            # опорного кадра плюс (f - ref)/hz. Выбор разметчика остаётся в
            # карточке как `pose`.
            mesh_source = (None if obj.get('mesh_source') is None
                           else str(obj.get('mesh_source')))
            pose_time0 = float(obj.get('pose_time_s') or 0.0)
            traces = {}
            mesh_info = None
            gt_ray_keys, gt_t_hit, gt_mult = [], [], []
            gt_offsets = [0]
            for f in frames:
                if f == ref:
                    xyz_f, inten_f, ring_f = xyz, inten, ring
                else:
                    xyz_f, inten_f, ring_f = frame_points(xyz_at, f)
                center_f = _instance_center(center, ref, f, speed, traj,
                                            hz=hz, size=size, motion=motion)
                t_f = pose_time0 + (int(f) - ref) / float(hz or HZ_DEFAULT)
                base_f = (trajectory_at(traj, t_f)[2] if traj is not None
                          else obj.get('z_ref'))
                mesh_f, info_f = object_geometry(
                    cls, pose_scene, size, center_f, obj.get('yaw', 0.0),
                    obj.get('pitch', 0.0), obj.get('roll', 0.0),
                    pose_time=t_f, base_z=base_f, mesh_source=mesh_source)
                tr = trace_object(xyz_f, ring_f, mesh_f)
                tr['center'] = center_f
                tr['frame'] = int(f)
                traces[f] = tr
                if f == ref:
                    mesh_info = info_f
                # Пер-кадровый GT хранится ЛУЧАМИ, а не точками: ключи лучей,
                # дальность до поверхности и кратность возвратов. Точки и тень
                # восстанавливаются по самому кадру (направление представителя
                # луча -- из кадра), формат -- на ~4 меньше точечного, а
                # повторяющиеся кадры (стоящий объект у стоящего сенсора) сжаты
                # npz почти в ноль.
                if tr['n_points']:
                    keys_f = synth.ray_keys(
                        xyz_f, None if ring_f is None else np.asarray(ring_f, dtype=np.int64))
                    hit = keys_f[np.asarray(tr['ray_index'], dtype=np.int64)]
                    gt_ray_keys.append(np.asarray(hit, dtype=np.int64))
                    gt_t_hit.append(np.asarray(tr['t_hit'], dtype=np.float32))
                    gt_mult.append(np.asarray(tr['mult'], dtype=np.int32))
                else:
                    gt_ray_keys.append(np.zeros(0, dtype=np.int64))
                    gt_t_hit.append(np.zeros(0, dtype=np.float32))
                    gt_mult.append(np.zeros(0, dtype=np.int32))
                gt_offsets.append(gt_offsets[-1] + int(gt_ray_keys[-1].size))
            trace = traces[ref]
            refl = reflectance_for(xyz, inten, trace['ray_index'], trace['mult'])
            pts = np.asarray(trace['points'], dtype=np.float64)
            local = _local_points(pts, center, obj.get('yaw', 0.0), obj.get('pitch', 0.0),
                                  obj.get('roll', 0.0))

            obj_id = str(obj.get('id') or f'{record}-{next_seq:04d}')
            if not _ID_RE.fullmatch(obj_id):
                raise ValueError(f'id {obj_id!r}: разрешены латиница, цифры, «-» и «_»')
            if obj.get('id'):
                by_id.pop(obj_id, None)
            else:
                next_seq += 1
            y_c, u_c, h_c = _terms_of_center(frame_pf, center)
            card = {
                'id': obj_id,
                'record': record,
                'format': dict(FORMAT_INFO),
                'class': cls,
                'pose': pose,
                'pose_scene': pose_scene,
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
                    'rule': ('y(f) = y_ref + speed_m_per_frame * (f - ref_frame)'
                             if not (motion or {}).get('profile') else
                             'y(f) = y_ref + сумма(motion_profile_m_per_frame[f] по кадрам '
                             'от reference_frame до f); speed_m_per_frame -- медиана профиля'),
                    'motion_source': (motion or {}).get('source'),
                    'motion_reliable_frac': (motion or {}).get('reliable_frac'),
                    'motion_profile_step': (motion or {}).get('profile_step'),
                    'motion_profile_m_per_frame': _profile_round(
                        (motion or {}).get('profile')),
                    'motion_cross_m_per_frame': _profile_round(
                        (motion or {}).get('cross_profile')),
                    'motion_differs_frames': [
                        int(i) for i, flag in enumerate((motion or {}).get('profile_flags') or [])
                        if flag == 'differs'],
                    'rule_profile': ('y(f) = y_ref + сумма(profile[f] по кадрам от ref до f)'
                                     ', профиль -- ход кадра из motion_profile_m_per_frame'),
                },
                'path': {
                    'along_m': float(-y_c), 'lateral_m': float(u_c), 'height_m': float(h_c),
                    'axis_x_m': float(frame_pf.axis_x(np.asarray([y_c]))[0]),
                    'ugr_z_m': float(frame_pf.ugr_z(np.asarray([y_c]))[0]),
                    'yaw_deg': float(obj.get('yaw', 0.0)),
                },
                'mesh': {
                    'source': mesh_info.get('source'),
                    'asset': mesh_info.get('asset'), 'clip': mesh_info.get('clip'),
                    'play': mesh_info.get('play'), 'moment': mesh_info.get('moment'),
                    'samples': mesh_info.get('samples'),
                    'pose_time_s': mesh_info.get('pose_time_s'),
                    'height_m': mesh_info.get('height_m'),
                    'base_z_m': mesh_info.get('base_z_m'),
                    'lift_m': mesh_info.get('lift_m'),
                    'vertices': mesh_info.get('vertices'),
                    'faces': mesh_info.get('faces'),
                    'rule': mesh_info.get('rule'),
                },
                'primitive': {'kind': 'person' if cls == 'person' else 'box',
                              'size_m': list(mesh_info.get('primitive_size') or []),
                              'pose': pose,
                              'source': ('assets/_poses' if mesh_info.get('source') == 'poses'
                                         else 'sim/objects.py')},
                # Опорная поверхность (низ бокса): панель ставит основание на неё,
                # поэтому она уходит в карточку -- по ней видно, стоит объект на
                # полу, на платформе или на поднятии.
                'base_z_m': (None if obj.get('z_ref') is None
                             else float(obj['z_ref'])),
                'counts': {'rays_hit': int(trace['rays_hit']),
                           'points_removed': int(trace['points_removed']),
                           'n_points': int(local.shape[0]),
                           'frames_traced': len(traces)},
                'instances': [
                    {
                        'frame': int(f),
                        'center': _instance_center(center, ref, f, speed, traj,
                                                   hz=hz, size=size, motion=motion),
                        'yaw': float(obj.get('yaw', 0.0)),
                        'pitch': float(obj.get('pitch', 0.0)),
                        'roll': float(obj.get('roll', 0.0)),
                        'size': [float(v) for v in size],
                        'automation': automation,
                        'detector_confirmed': bool(automation == 'detector'),
                        # Пер-кадровый GT (v2): трассировка ЭТОГО кадра; сами
                        # точки/тень -- в .npz по ключам лучей (см. FORMAT_INFO).
                        'rays_hit': int(traces[f]['rays_hit']),
                        'n_points': int(traces[f]['n_points']),
                        'points_removed': int(traces[f]['points_removed']),
                    }
                    for f in frames
                ],
                'revision': rev,
            }
            if traj is not None:
                # Траектория движения: начало, конец, секунды и скорость -- то,
                # ради чего объект вообще «едет» (`trajectory_at`).
                card['trajectory'] = dict(
                    traj, reference_frame=ref, hz=float(hz),
                    speed_m_per_frame=traj['speed_mps'] / float(hz or HZ_DEFAULT),
                    play=pose_scene,
                    rule=TRAJ_RULE,
                    frames_span=float(traj['seconds']) * float(hz or HZ_DEFAULT),
                )
            rec_dir = os.path.join(out_dir, record)
            npz_path = os.path.join(rec_dir, f'{obj_id}.npz')
            card_path = os.path.join(rec_dir, f'{obj_id}.json')
            os.makedirs(rec_dir, exist_ok=True)
            gt_payload = {
                'frames': np.asarray(frames, dtype=np.int32),
                'gt_offsets': np.asarray(gt_offsets, dtype=np.int32),
                'gt_ray_keys': (np.concatenate(gt_ray_keys) if gt_ray_keys
                                else np.zeros(0, dtype=np.int64)),
                'gt_t_hit': (np.concatenate(gt_t_hit) if gt_t_hit
                             else np.zeros(0, dtype=np.float32)),
                'gt_mult': (np.concatenate(gt_mult) if gt_mult
                            else np.zeros(0, dtype=np.int32)),
            }
            _atomic_write(npz_path, _npz_bytes(local, refl, gt_payload))
            _atomic_write(card_path, json.dumps(card, ensure_ascii=False,
                                                indent=1).encode('utf-8'))
            row = {
                'id': obj_id, 'record': record, 'class': cls, 'pose': pose,
                'pose_scene': pose_scene,
                'format': FORMAT_VERSION,
                'frames': frames, 'n_points': int(local.shape[0]),
                'automation': automation,
            }
            if traj is not None:
                # В индекс -- коротко: начало, конец, секунды, скорость.
                row['trajectory'] = {
                    'start': [round(float(v), 4) for v in traj['start']],
                    'end': [round(float(v), 4) for v in traj['end']],
                    'seconds': round(float(traj['seconds']), 3),
                    'distance_m': round(float(traj['distance_m']), 4),
                    'speed_mps': round(float(traj['speed_mps']), 4),
                    'speed_kmh': round(float(traj['speed_kmh']), 3),
                }
            by_id[obj_id] = row
            written.append({'id': obj_id, 'card': card_path, 'npz': npz_path,
                            'n_points': int(local.shape[0]),
                            'frames_traced': len(traces),
                            'gt_rays': int(gt_offsets[-1]),
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


def _npz_bytes(points, reflectance, gt=None):
    """Содержимое .npz в память: точки, reflectance и пер-кадровый GT (v2).

    `points` (N, 3) float32 и `reflectance` (N,) -- опорный кадр, как в v1.
    `gt` (dict или None) -- пер-кадровый GT: `frames` (F,), `gt_offsets`
    (F+1,), `gt_ray_keys`/`gt_t_hit`/`gt_mult` -- конкатенация по кадрам;
    кадр f занимает срез [gt_offsets[i], gt_offsets[i+1]).
    """
    import io

    buf = io.BytesIO()
    payload = {'points': np.asarray(points, dtype=np.float32).reshape(-1, 3)}
    if reflectance is not None:
        payload['reflectance'] = np.asarray(reflectance, dtype=np.float32).reshape(-1)
    if gt:
        for key in ('frames', 'gt_offsets', 'gt_ray_keys', 'gt_t_hit', 'gt_mult'):
            payload[key] = np.asarray(gt[key])
    np.savez_compressed(buf, **payload)
    return buf.getvalue()