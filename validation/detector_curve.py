"""Кривая «дистанция -> P(обнаружения)» по ТРЁМ каналам РЕАЛЬНОЙ цепочки.

`run_range_curve.py` считает не детектор, а упрощённый критерий («>= 5 точек в
окне +-1.5 м»). Здесь в тот же протокол инъекций поставлены сами каналы цепочки,
и по каждому снимается булев результат:

  * `габарит` -- `detector.core.detect(...)` (пороги `DetectorConfig()` по
    умолчанию, модель оси `detector.profiles.model_for_db`): событие = есть
    препятствие в окне +-`WINDOW_M` от ИСТИННОГО центра объекта;
  * `фон` -- остаток кадра относительно фона на луч из `stitch_bg`
    (`per_ray_residual` + `corridor_gate` + `count_clusters`): пороги модуля
    (>= 5 точек, размах >= 0.3 м, по высоте >= 0.4 м); событие = кластер в том же
    окне. Ближняя зона (`near_m = 60` из `empty_frames_report`) НЕ отсекается:
    она нужна воксельному варианту фона на дальней части тоннеля, а пустые окна
    выбирались тем же критерием на всём коридоре;
  * `контур` -- свободная ширина (`detector.contour.free_width` +
    `WidthTracker`): порог модуля `WIDTH_NARROW_M` = 0.3 м и правило
    `PERSIST_BINS` = 2 бина подряд; эталон -- медиана ширины по 15 пустым кадрам
    той же записи (`WINDOW` модуля), кадр с объектом оценивается ОДИН против
    этого эталона (каждый трекер получает свою копию истории, иначе предыдущая
    постановка портила бы медиану следующей).

Правила цепочки: `1-из-3` (любой канал -- консервативно безопасно) и `2-из-3`.

ПРОТОКОЛ: 6 записей x 10 пустых кадров x 3 поперечных положения (+-0.7175 и 0) x
дистанции 25..150 шагом 25 м -- как в `run_range_curve` (окно кадров берётся из
его кэша `validation/out/empty_windows.json`, объект -- `synth.insert_object`,
0.30 x 0.30 x 1.00 м).

ДВЕ ПОСАДКИ ОБЪЕКТА (`--bases`). Замерено: МНК-пол `synth.fit_floor_ab` лежит
НИЖЕ УГР модели пути детектора на 0.10..1.25 м (по записям и дистанциям:
doubleT_obstacle -0.57..-0.84 м, roundT_doubleT -0.21..-0.10 м). Детектор же
считает высоту `h` от УГР и объём начинает с `h_low_m` = 0.30 м, поэтому объект
1 м, поставленный низом на МНК-пол, попадает в объём только верхушкой:

  * `floor` -- низ на МНК-пол (`synth.fit_floor_ab`), как в `run_range_curve`:
    это исходный протокол, и он честно показывает цену расхождения датумов;
  * `ugr` -- низ на УГР модели пути детектора (линейная МНК по `ugr_z_m`,
    расхождение линии с полилинией -- 0.00..0.06 м): объект стоит на рельсовом
    уровне, как настоящий (человек в `doubleT_obstacle` виден детектором на
    h = 0.32..1.16 м над УГР), и канал габарита сравнивается с объектом в СВОЁМ
    объёме.

Ось постановки не подменяется: `synth.axis_from_track` (середина нитей
`track_geometry`) и полилиния детектора в снимке сходятся до 1.5 мм (замерено на
25 м), то есть поперечная координата у обоих одна и та же.

ЛОЖНЫЕ: те же каналы и правила прогоняются на ПУСТЫХ кадрах в ДВУХ наборах.

  * `block` -- непрерывный блок по 60 кадров на запись с индекса
    `REFERENCE_FRAMES` (12), всего 360 >= 300. Блок НЕ выбирается по пустоте,
    иначе канал фона давал бы ноль тавтологически: он берётся как есть, а кадры
    с известным объектом (человек в `doubleT_obstacle`, 46 кадров из
    `validation/out/fp_per_hour_events.csv`) исключаются из знаменателя всех
    каналов и правил -- они не ложные. Событие -- срабатывание В ЛЮБОМ МЕСТЕ
    коридора, то есть то же определение, что в `fp_per_hour`, поэтому эти числа с
    ним и сверяются (замерено: канал габарита даёт на блоке ровно 46 событий
    fp_per_hour и ни одного сверх, FP после вычета человека -- 0);
  * `window` -- кадры окна испытаний БЕЗ объекта, событие в том же окне +-2 м на
    каждой дистанции D и с той же историей трекера. Это единственные FP,
    СРАВНИМЫЕ с P по испытаниям: у канала контура числа наборов `block` и
    `window` различаются на порядок, потому что в бою история трекера к сцене не
    подходит (датчик поворачивается между кадрами), а в испытаниях подходит.

Выход: `validation/out/detector_curve_summary.csv` (`kind=distance` -- дистанция ->
вероятности по каналам и правилам и совместные пропуски против произведения;
`kind=empty` -- FP по каналам и правилам, `frame_set` различает наборы),
`validation/out/detector_curve_trials.csv` (сырые испытания: все инъекции и все
пустые кадры) и `validation/out/detector_curve.txt` (таблицы и диагностика).

Запуск (из корня проекта, ~5 мин на этой машине: 2160 инъекций + 720 пустых кадров,
замер -- в `detector_curve.txt`):

    python -m validation.detector_curve
    python -m validation.detector_curve --frames 2 --distances 25 50   # проверка
"""

import argparse
import copy
import csv
import glob
import json
import os
import sys
import time
import warnings

import numpy as np

from . import stitch_bg as bg
from .synth import (GAUGE_HALF_M, OBJECT_SIZE_DEFAULT, axis_from_track,
                    build_rings_meta, el_deg, insert_object, placement_check)

DISTANCES = (25.0, 50.0, 75.0, 100.0, 125.0, 150.0)
LATERALS = (-GAUGE_HALF_M, 0.0, GAUGE_HALF_M)
LATERAL_NAMES = ('лево', 'центр', 'право')
BASES = ('floor', 'ugr')
BASE_NAMES = {'floor': 'низ на МНК-пол (протокол run_range_curve)',
              'ugr': 'низ на УГР модели пути (объём детектора)'}
CHANNELS = ('gauge', 'bg', 'contour')
FRAMES_PER_BAG = 10
REFERENCE_FRAMES = 12          # окно фона: как в `run_range_curve.empty_window`
FP_FRAMES_PER_BAG = 60         # кадров на запись для FP: 6 x 60 = 360 >= 300
SCAN_LIMIT_EXTRA = 200         # докуда искать пустые кадры для истории трекера
WINDOW_M = 2.0                 # окно события вокруг истинного центра объекта
REVISION_FILES = ('core.py', 'profiles.py', 'track_models.json')
RING_BAG_NAMES = ('doubleT_obstacle', 'doubleT_platform', 'roundT_doubleT',
                  'roundT_pressureGate_roundT', 'roundT_squareT_pressureGate_squareT',
                  'squareT_platform_squareT_switch')

FIELDS = (
    'kind', 'base', 'bag', 'frame', 'distance_m', 'lateral_m', 'lateral', 'size_m',
    'seed', 'in_range', 'known_object', 'empty', 'frame_set',
    'points_object', 'rays_hit', 'points_removed', 'placement_free',
    'points_inside_box_top', 'floor_minus_ugr_m',
    'gauge_hit', 'gauge_objects', 'gauge_y_m', 'gauge_err_m', 'gauge_bins_blocked',
    'bg_hit', 'bg_clusters', 'bg_resid_points', 'bg_err_m',
    'contour_hit', 'contour_events', 'contour_narrow_bins', 'contour_narrow_m',
    'contour_min_width_m', 'contour_baseline_m',
    'any_hit', 'two_of_three',
)


def project_root():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


def bag_db_paths(root=None, names=RING_BAG_NAMES):
    root = project_root() if root is None else root
    out = []
    for name in names:
        found = sorted(glob.glob(os.path.join(root, 'for_hackathon', name, '*.db3')))
        if found:
            out.append((name, found[0]))
    return out


def detector_config():
    """Пороги детектора по умолчанию -- их и проверяем, не подгоняются."""
    from detector.core import DetectorConfig  # noqa: PLC0415

    return DetectorConfig()


def model_for_record(db_path):
    from detector.profiles import model_for_db  # noqa: PLC0415

    return model_for_db(db_path)


def detector_revision():
    """md5 файлов `detector/` на текущий момент -- паспорт чисел отчёта.

    `detector/**` правится параллельно, а модель оси читается из
    `detector/track_models.json` (и перечитывается на каждой записи). Замерено
    на этом прогоне: подмена снимка В СЕРЕДИНЕ прогона уводит `valid_far_m` с
    -60.75 на -61.25 и стоит двух кадров человека из 46 -- поэтому ревизия
    снимается ДО первого кадра, а модели всех записей грузятся сразу, и числа
    отчёта относятся к одной ревизии.
    """
    import hashlib  # noqa: PLC0415

    import detector  # noqa: PLC0415

    base = os.path.dirname(os.path.abspath(detector.__file__))
    out = {}
    for name in REVISION_FILES:
        try:
            with open(os.path.join(base, name), 'rb') as handle:
                out[name] = hashlib.md5(handle.read()).hexdigest()
        except OSError:
            out[name] = '?'
    return out


# ---------------------------------------------------------------------------
# Кадры, фон, посадка объекта
# ---------------------------------------------------------------------------

def load_frame(frames, index):
    """Кадр -> (points float64, ring, intensity)."""
    frame = frames[index]
    points = np.asarray(frame.xyz, dtype=np.float64)
    ring = None if frame.ring is None else np.asarray(frame.ring, dtype=np.int64)
    intensity = (None if frame.intensity is None
                 else np.asarray(frame.intensity, dtype=np.float64))
    return points, ring, intensity


def ugr_line(model, y_lo=-150.0, y_hi=0.0):
    """Линия УГР модели пути (МНК по `ugr_z_m`): (a, b) для z = a + b*y.

    `insert_object` умеет ставить низ объекта только на ПЛОСКОСТЬ, а УГР модели --
    полилиния. Расхождение МНК-линии с полилинией в диапазоне испытаний -- до
    0.06 м (замерено), поэтому подмена допустима и она единственная: иначе объект
    встаёт на дно лотка, а не на рельсовый уровень.

    Возвращает (a, b, max_отклонение_линии_от_полилинии).
    """
    y = np.asarray(model.y_nodes, dtype=np.float64)
    z = np.asarray(model.ugr_z_m, dtype=np.float64)
    good = np.isfinite(y) & np.isfinite(z)
    sel = good & (y >= float(y_lo)) & (y <= float(y_hi))
    if np.count_nonzero(sel) < 2:
        return (float(np.median(z[good])) if np.any(good) else 0.0, 0.0, 0.0)
    b, a = np.polyfit(y[sel], z[sel], 1)
    line = a + b * y[sel]
    return (float(a), float(b), float(np.max(np.abs(line - z[sel]))))


def rings_for_points(points, elev_deg):
    """Кольцо каждой точки объекта -- ближайшее по elevation кольцо кадра.

    Точки объекта стоят на РЕАЛЬНЫХ лучах кадра (`insert_object` берёт их
    направления), поэтому elevation попадает в свой кольцевой уровень с
    точностью 1e-6° (замерено), и присвоение однозначно. Без ring канал «фон на
    луч» не может взять ячейку фона (`background['median'][ring, az_bin]`), и
    остаток по объекту вообще не считается.
    """
    elev = np.asarray(elev_deg, dtype=np.float64)
    good = np.flatnonzero(np.isfinite(elev))
    out = np.zeros(np.asarray(points).shape[0], dtype=np.int64)
    if good.size == 0 or out.size == 0:
        return out
    e = el_deg(np.asarray(points, dtype=np.float64))
    idx = np.abs(e[:, None] - elev[good][None, :]).argmin(axis=1)
    return good[idx]


def merge_ring(ring, mask, points_new, inserted, elev_deg):
    """Ring кадра с внесённым объектом: фон без тени + кольца точек объекта."""
    n_bg = int(inserted['background_count'])
    n_new = int(np.asarray(points_new).shape[0])
    if n_new == n_bg or ring is None:
        return ring
    background_ring = np.asarray(ring)[~np.asarray(mask)]
    object_ring = rings_for_points(np.asarray(points_new, dtype=np.float64)[n_bg:], elev_deg)
    return np.concatenate([background_ring, object_ring])


def frame_metas(points, ring, intensity, axis_nodes, ugr_ab):
    """Метаданные кадра для обоих посадок: 'floor' (МНК-пол) и 'ugr' (УГР модели).

    Пол считается ОДИН раз на кадр (`build_rings_meta` без `floor_ab`), посадка
    `ugr` -- та же мета с подменённым `floor_ab`: сетка колец, ось и массивы кадра
    у посадок общие, различается только уровень низа объекта.
    """
    floor_meta = build_rings_meta(points, ring, intensity, axis_nodes=axis_nodes)
    ugr_meta = dict(floor_meta)
    ugr_meta['floor_ab'] = (float(ugr_ab[0]), float(ugr_ab[1]))
    ugr_meta['floor_source'] = 'УГР модели пути детектора (МНК по ugr_z_m)'
    ugr_meta['floor_rms_m'] = float(ugr_ab[2]) if len(ugr_ab) > 2 else float('nan')
    return {'floor': floor_meta, 'ugr': ugr_meta}


# ---------------------------------------------------------------------------
# Каналы
# ---------------------------------------------------------------------------

def gauge_obstacles(xyz, model, cfg):
    """Габарит: y найденных препятствий (пороги `DetectorConfig()` по умолчанию).

    Детектор работает только на участке, где ось достоверна (`valid_range`),
    поэтому попадание истинного центра в этот участок считается отдельно: вне его
    ноль канала -- не свойство канала, а отсутствие данных для него.
    """
    from detector.core import detect  # noqa: PLC0415

    result = detect(xyz, model=model, cfg=cfg)
    ys = np.array([float(o.y_m) for o in (result.obstacles or [])], dtype=np.float64)
    return {'ys': ys, 'objects': int(ys.size),
            'bins_blocked': int(getattr(result, 'bins_blocked', 0))}


def background_clusters(points, ring, background, floor_ab, axis_nodes):
    """Фон на луч: остаток в коридоре и центры его кластеров по y.

    Пороги берутся у `stitch_bg` (`MIN_CLUSTER_POINTS`, `MIN_CLUSTER_EXTENT_M`,
    `MIN_CLUSTER_EXTENT_Z_M`) и не подгоняются. Гейт коридора обязателен: без
    него остаток забит полом и лотком. Центр кластера -- медиана его точек по y.
    """
    residue = bg.per_ray_residual(points, ring, background)
    if residue.shape[0]:
        residue = residue[bg.corridor_gate(residue, floor_ab, axis_nodes)]
    clusters = bg.count_clusters(residue, min_extent=bg.MIN_CLUSTER_EXTENT_M,
                                 min_extent_z=bg.MIN_CLUSTER_EXTENT_Z_M)
    ys = np.array([float(np.median(c['points'][:, 1])) for c in clusters])
    return {'ys': ys, 'clusters': int(ys.size), 'resid_points': int(residue.shape[0])}


def width_history(frames_points, model):
    """Свободная ширина по списку кадров: (списки half_left, half_right, y)."""
    from detector import contour  # noqa: PLC0415

    left, right, y = [], [], None
    for points in frames_points:
        y, a, b = contour.free_width(points, model)
        left.append(a)
        right.append(b)
    return left, right, y


def width_tracker(left_list, right_list):
    """`WidthTracker` с историей из кадров; пороги модуля не меняются.

    История короче `contour.WINDOW` не годится: трекер до её заполнения молчит
    (это его правило), поэтому вызывающий обязан дать не меньше `WINDOW` кадров.
    """
    from detector import contour  # noqa: PLC0415

    totals = [np.asarray(a, dtype=np.float64) + np.asarray(b, dtype=np.float64)
              for a, b in zip(left_list, right_list)]
    if len(totals) < contour.WINDOW:
        return None
    return contour.WidthTracker(_half=totals[-contour.WINDOW:])


def width_baseline(left_list, right_list):
    """Медиана свободной ширины по истории -- то, с чем трекер сравнивает кадр."""
    totals = np.vstack([np.asarray(a, dtype=np.float64) + np.asarray(b, dtype=np.float64)
                        for a, b in zip(left_list, right_list)])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.asarray(np.nanmedian(totals, axis=0), dtype=np.float64)


def contour_width(points, model, tracker):
    """Контур: свободная ширина кадра и события `WidthTracker` против истории.

    Возвращает `dists` -- дистанции событий (то, что выдаёт трекер: сужение >=
    `WIDTH_NARROW_M` в `PERSIST_BINS` бинах подряд) и `y`/`total` кадра для
    диагностики сужения (эталон считает вызывающий и передаёт в
    `width_narrowing`). Каждый вызов копирует историю: иначе событие предыдущей
    постановки осталось бы в медиане и портило следующую.
    """
    from detector import contour  # noqa: PLC0415

    y_w, left, right = contour.free_width(points, model)
    total = np.asarray(left, dtype=np.float64) + np.asarray(right, dtype=np.float64)
    out = {'y': y_w, 'total': total, 'dists': np.zeros(0)}
    if tracker is None:
        return out
    events = copy.deepcopy(tracker).update(left, right, y_w)
    out['dists'] = np.array([float(e['distance_m']) for e in events], dtype=np.float64)
    return out


def width_narrowing(contour_out, baseline, truth_y=None, window_m=WINDOW_M):
    """Диагностика: сколько бинов подряд сужено и насколько (в окне от D).

    `narrow_bins` считается по порогу САМОГО модуля (`WIDTH_NARROW_M`), но без
    правила персистентности: так видно, что канал молчит «трекер требует два бина»,
    а не «объект не сузил ширину».
    """
    from detector import contour  # noqa: PLC0415

    y_w = np.asarray(contour_out['y'], dtype=np.float64)
    total = np.asarray(contour_out['total'], dtype=np.float64)
    delta = total - np.asarray(baseline, dtype=np.float64)
    near = (np.ones(delta.shape, dtype=bool) if truth_y is None
            else np.abs(y_w - float(truth_y)) <= float(window_m))
    narrowed = np.isfinite(delta) & (delta <= -float(contour.WIDTH_NARROW_M)) & near
    run = best = 0
    for flag in narrowed.tolist():
        run = run + 1 if flag else 0
        best = max(best, run)
    finite = np.isfinite(delta) & near
    if not np.any(finite):
        return {'narrow_bins': int(best), 'narrow_m': 0.0, 'min_width_m': None,
                'baseline_m': None}
    return {'narrow_bins': int(best), 'narrow_m': float(-np.min(delta[finite])),
            'min_width_m': float(np.min(total[finite])),
            'baseline_m': float(np.median(np.asarray(baseline)[finite]))}


def hit_any(dists):
    """Событие канала «где угодно в коридоре» -- определение FP как в fp_per_hour."""
    return bool(np.asarray(dists).size)


def hit_near(values, truth_y, window_m=WINDOW_M):
    """Событие канала в окне +-`window_m` от истинного центра объекта.

    `values` -- координаты событий: у габарита и фона это y (отрицательные, тоннель
    уходит в −Y), у контура -- уже дистанции (положительные). Сравнение идёт по
    ДИСТАНЦИИ (`|value|` против `|truth_y|`), поэтому годится и то, и другое.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return False
    if truth_y is None:
        return True
    target = abs(float(truth_y))
    return bool(np.any(np.abs(np.abs(values) - target) <= float(window_m)))


def nearest_err(ys, truth_y):
    """Ближайшее событие канала к истинному центру: (координата, |разница|)."""
    ys = np.asarray(ys, dtype=np.float64)
    if ys.size == 0 or truth_y is None:
        return None, None
    err = np.abs(np.abs(ys) - abs(float(truth_y)))
    k = int(np.argmin(err))
    return float(ys[k]), float(err[k])


def is_empty(points, ring, background, floor_ab, axis_nodes):
    """Пустой кадр -- критерий `run_range_curve.empty_window` (см. докстроку).

    Возвращает `(пусто, факты)`, где в фактах лежат остаток, кластеры и их центры
    по y: и пустота, и попадание в окно считаются по ОДНОМУ остатку, без
    повторного счёта.
    """
    facts = background_clusters(points, ring, background, floor_ab, axis_nodes)
    facts['empty'] = bool(facts['clusters'] == 0)
    return facts['empty'], facts


def empty_row(name, index, frame_set, distance_m, facts, gauge, contour_out, diag, hits,
              known_events):
    """Строка пустого кадра: каналы уже посчитаны, здесь только сборка."""
    truth_y = None if distance_m == '' else -float(distance_m)
    gy, gerr = nearest_err(gauge['ys'], truth_y)
    by, berr = nearest_err(facts['ys'], truth_y)
    return {
        'kind': 'empty', 'base': '', 'bag': name, 'frame': int(index),
        'frame_set': frame_set,
        'distance_m': distance_m, 'lateral_m': '', 'lateral': '', 'size_m': '',
        'seed': '', 'in_range': '', 'known_object': int((name, index) in known_events),
        'bg_hit': int(bool(facts['clusters'])), 'bg_clusters': int(facts['clusters']),
        'bg_resid_points': int(facts['resid_points']),
        'bg_err_m': ('' if berr is None else round(berr, 3)),
        'gauge_hit': int(hits[0]), 'gauge_objects': int(gauge['objects']),
        'gauge_y_m': ('' if gy is None else round(gy, 3)),
        'gauge_err_m': ('' if gerr is None else round(gerr, 3)),
        'gauge_bins_blocked': int(gauge['bins_blocked']),
        'contour_hit': int(hits[2]), 'contour_events': int(contour_out['dists'].size),
        'contour_narrow_bins': int(diag['narrow_bins']),
        'contour_narrow_m': round(float(diag['narrow_m']), 3),
        'contour_min_width_m': ('' if diag['min_width_m'] is None
                                else round(diag['min_width_m'], 3)),
        'contour_baseline_m': ('' if diag['baseline_m'] is None
                               else round(diag['baseline_m'], 3)),
        'empty': int(facts['empty']),
        'any_hit': int(any(hits)), 'two_of_three': int(sum(hits) >= 2),
    }


# ---------------------------------------------------------------------------
# Прогон одной записи
# ---------------------------------------------------------------------------

def window_start(cache_path, name, frames_n):
    """Начало окна кадров из кэша `run_range_curve` (0, если кэша нет)."""
    try:
        with open(cache_path, encoding='utf-8') as handle:
            cache = json.load(handle)
    except (OSError, ValueError):
        return 0
    entry = cache.get('%s|%d' % (name, frames_n)) or {}
    return int(entry.get('start', 0))


def run_record(name, db_path, distances=DISTANCES, frames_n=FRAMES_PER_BAG,
               bases=BASES, fp_frames=FP_FRAMES_PER_BAG, start=0,
               known_events=(), size_m=OBJECT_SIZE_DEFAULT, model=None, verbose=True):
    """Испытания одной записи: инъекции (все базы) + пустые кадры (FP).

    Возвращает `(trial_rows, empty_rows, info)`. Кадры читаются в порядке
    возрастания индекса (кэш `bag_reader` рассчитан на последовательный доступ);
    окно испытаний и блок пустых кадров читаются дважды -- это дешевле, чем
    держать сотни кадров в памяти. `model` передаётся вызывающим, чтобы модель оси
    была загружена ДО первого кадра (см. `detector_revision`).
    """
    import bag_reader  # noqa: PLC0415

    from detector import contour  # noqa: PLC0415

    root = project_root()
    model = model if model is not None else model_for_record(db_path)
    cfg = detector_config()
    a_u, b_u, dev_u = ugr_line(model)
    y_far, y_near = model.valid_range(cfg.max_range_m, cfg.min_range_m)
    info = {'bag': name, 'db_path': db_path, 'model': model.name,
            'model_source': model.source, 'valid_far_m': float(y_far),
            'y_near_m': float(y_near), 'ugr_line_m': (a_u, b_u),
            'ugr_line_dev_m': dev_u, 'window_start': int(start), 'frames_n': int(frames_n),
            'fp_frames': int(fp_frames), 'frames_total': 0, 'empty_frac_fp': float('nan'),
            'contour_history': 0, 'axis_source': '', 'bg_stable_frac': float('nan'),
            'floor_minus_ugr_m': {}, 'seconds': 0.0}

    t_bag = time.time()
    frames = bag_reader.BagFrames(db_path, cache_size=24, prefetch_ahead=8)
    try:
        info['frames_total'] = int(len(frames))
        # 1. Фон и опорные величины -- как в `run_range_curve.empty_window`.
        reference = [load_frame(frames, i)[:2] for i in range(REFERENCE_FRAMES)]
        background = bg.per_ray_background(reference)
        floor_ab = bg.fit_floor_ab(reference[0][0])[:2]
        axis_nodes = bg._axis_nodes(reference[0][0], floor_ab)
        info['bg_stable_frac'] = float(background['stable_frac'])

        # 2. Окно испытаний + метаданные кадра по обеим посадкам.
        window = [load_frame(frames, start + k) for k in range(frames_n)]
        try:
            nodes, axis_meta = axis_from_track(db_path, start)
            info['axis_source'] = str(axis_meta.get('axis_source', '?'))
        except Exception as exc:  # noqa: BLE001 -- ось по кадру, если модуль недоступен
            nodes, info['axis_source'] = None, 'оценка по кадру (%s)' % type(exc).__name__
        items = []
        for points, ring, intensity in window:
            metas = frame_metas(points, ring, intensity, nodes, (a_u, b_u, dev_u))
            items.append({'points': points, 'ring': ring, 'elev': metas['floor']['elev_deg'],
                          'meta': metas})

        # 3. История ширины: кадры фона + пустые кадры блока (если 12 < WINDOW).
        warm_left, warm_right, _y_warm = width_history([p for p, _r in reference], model)
        info['contour_history'] = len(warm_left)

        # 4. Перепись блока пустых кадров (и добор истории по хвосту записи).
        block = list(range(REFERENCE_FRAMES, REFERENCE_FRAMES + int(fp_frames)))
        census = {}
        scan_end = min(len(frames), max(block[-1] + 1 if block else 0,
                                        REFERENCE_FRAMES + SCAN_LIMIT_EXTRA))
        index = REFERENCE_FRAMES
        while index < scan_end:
            points, ring, _i = load_frame(frames, index)
            empty, facts = is_empty(points, ring, background, floor_ab, axis_nodes)
            if index < REFERENCE_FRAMES + int(fp_frames):
                census[index] = facts
            if empty and len(warm_left) < contour.WINDOW:
                left, right, _y = width_history([points], model)
                warm_left.extend(left)
                warm_right.extend(right)
                info['contour_history'] = len(warm_left)
            index += 1
            if index > REFERENCE_FRAMES + int(fp_frames) and len(warm_left) >= contour.WINDOW:
                break
        tracker = width_tracker(warm_left, warm_right)
        baseline = width_baseline(warm_left, warm_right)

        # 5. Ложные: те же каналы на пустых кадрах, двумя наборами.
        #    `block` (кадры 12..) -- событие в ЛЮБОМ месте коридора, определение как
        #    в `fp_per_hour`; `window` (кадры окна испытаний БЕЗ объекта) -- событие
        #    в окне +-2 м на каждой дистанции D, то есть то же самое, что считается
        #    у испытаний, и с той же историей трекера. Оба набора нужны: первый
        #    даёт полевую частоту, второй -- сравнимую с P величину на дистанции.
        empty_rows = []
        for index in sorted(census):
            points, ring, _i = load_frame(frames, index)
            facts = census[index]
            gauge = gauge_obstacles(points, model, cfg)
            cw = contour_width(points, model, tracker)
            diag = width_narrowing(cw, baseline)
            hits = [hit_any(gauge['ys']), facts['clusters'] > 0,
                    hit_any(cw['dists'])]
            empty_rows.append(empty_row(name, index, 'block', '', facts, gauge, cw, diag,
                                        hits, known_events))
        for k in range(frames_n):
            index = start + k
            points, ring, _i = load_frame(frames, index)
            _empty, facts = is_empty(points, ring, background, floor_ab, axis_nodes)
            gauge = gauge_obstacles(points, model, cfg)
            cw = contour_width(points, model, tracker)
            for D in distances:
                truth_y = -float(D)
                diag = width_narrowing(cw, baseline, truth_y=truth_y)
                hits = [hit_near(gauge['ys'], truth_y), hit_near(facts['ys'], truth_y),
                        hit_near(cw['dists'], truth_y)]
                empty_rows.append(empty_row(name, index, 'window', float(D), facts, gauge,
                                           cw, diag, hits, known_events))
        block_rows = [r for r in empty_rows if r['frame_set'] == 'block']
        info['empty_frac_fp'] = (float(np.mean([r['empty'] for r in block_rows]))
                                 if block_rows else float('nan'))

        # 6. Инъекции: кадр x база x дистанция x поперечное положение.
        trial_rows = []
        for k, item in enumerate(items):
            index = start + k
            for base in bases:
                meta = item['meta'][base]
                if base not in info['floor_minus_ugr_m']:
                    y0 = -25.0
                    info['floor_minus_ugr_m'][base] = float(
                        meta['floor_ab'][0] + meta['floor_ab'][1] * y0 - (a_u + b_u * y0))
                for D in distances:
                    for lateral, lateral_name in zip(LATERALS, LATERAL_NAMES):
                        seed = k * 1000 + int(D)
                        points_new, mask, inserted = insert_object(
                            item['points'], meta, D, lateral, size_m=size_m, seed=seed)
                        points_new = np.asarray(points_new, dtype=np.float64)
                        ring_new = merge_ring(item['ring'], mask, points_new, inserted,
                                              item['elev'])
                        truth_y = float(inserted['center'][1])
                        place = placement_check(item['points'], meta, D, lateral, size_m=size_m)
                        gauge = gauge_obstacles(points_new, model, cfg)
                        bgc = background_clusters(points_new, ring_new, background,
                                                  floor_ab, axis_nodes)
                        cw = contour_width(points_new, model, tracker)
                        diag = width_narrowing(cw, baseline, truth_y=truth_y)
                        hits = [hit_near(gauge['ys'], truth_y), hit_near(bgc['ys'], truth_y),
                                hit_near(cw['dists'], truth_y)]
                        gy, gerr = nearest_err(gauge['ys'], truth_y)
                        by, berr = nearest_err(bgc['ys'], truth_y)
                        trial_rows.append({
                            'kind': 'injection', 'base': base, 'bag': name,
                            'frame': int(index), 'distance_m': float(D),
                            'lateral_m': float(lateral), 'lateral': lateral_name,
                            'size_m': 'x'.join('%.2f' % v for v in size_m),
                            'seed': int(seed),
                            'in_range': int(bool(y_far <= truth_y <= y_near)),
                            'known_object': 0,
                            'points_object': int(inserted['points_inserted']),
                            'rays_hit': int(inserted['rays_hit']),
                            'points_removed': int(inserted['points_removed']),
                            'placement_free': int(bool(place['free'])),
                            'points_inside_box_top': int(place['points_inside_top']),
                            'floor_minus_ugr_m': round(float(
                                meta['floor_ab'][0] + meta['floor_ab'][1] * truth_y
                                - (a_u + b_u * truth_y)), 3),
                            'gauge_hit': int(hits[0]), 'gauge_objects': int(gauge['objects']),
                            'gauge_y_m': ('' if gy is None else round(gy, 3)),
                            'gauge_err_m': ('' if gerr is None else round(gerr, 3)),
                            'gauge_bins_blocked': int(gauge['bins_blocked']),
                            'bg_hit': int(hits[1]), 'bg_clusters': int(bgc['clusters']),
                            'bg_resid_points': int(bgc['resid_points']),
                            'bg_err_m': ('' if berr is None else round(berr, 3)),
                            'contour_hit': int(hits[2]), 'contour_events': int(cw['dists'].size),
                            'contour_narrow_bins': int(diag['narrow_bins']),
                            'contour_narrow_m': round(float(diag['narrow_m']), 3),
                            'contour_min_width_m': ('' if diag['min_width_m'] is None
                                                    else round(diag['min_width_m'], 3)),
                            'contour_baseline_m': ('' if diag['baseline_m'] is None
                                                   else round(diag['baseline_m'], 3)),
                            'any_hit': int(any(hits)), 'two_of_three': int(sum(hits) >= 2),
                        })
            if verbose:
                print('   %-40s кадр %3d готов (%.1f с)' % (name, index, time.time() - t_bag),
                      flush=True)
    finally:
        frames.close()
    info['seconds'] = time.time() - t_bag
    if verbose and tracker is None:
        print('   %-40s контур: истории меньше %d кадров -- канал не оценивается'
              % (name, contour.WINDOW), flush=True)
    return trial_rows, empty_rows, info


# ---------------------------------------------------------------------------
# Сводка
# ---------------------------------------------------------------------------

def rate(rows, key):
    if not rows:
        return float('nan')
    return float(np.mean([1.0 if r[key] else 0.0 for r in rows]))


def joint_misses(rows, tag=''):
    """Пропуски по каналам, совместные пропуски и произведение вероятностей.

    `indep_*` -- произведение отдельных вероятностей пропуска: если каналы
    независимы, совместный пропуск обязан равняться произведению. Больше
    произведения -- каналы связаны (один сенсор, одна геометрия).
    """
    out = {}
    if not rows:
        return out
    miss = {c: np.array([not r['%s_hit' % c] for r in rows], dtype=bool) for c in CHANNELS}
    for c in CHANNELS:
        out['miss_%s%s' % (c, tag)] = float(miss[c].mean())
    for a, b in (('gauge', 'bg'), ('gauge', 'contour'), ('bg', 'contour')):
        out['joint_miss_%s_%s%s' % (a, b, tag)] = float((miss[a] & miss[b]).mean())
        out['indep_%s_%s%s' % (a, b, tag)] = out['miss_%s%s' % (a, tag)] * out['miss_%s%s' % (b, tag)]
    all_miss = miss['gauge'] & miss['bg'] & miss['contour']
    out['joint_miss_all%s' % tag] = float(all_miss.mean())
    out['indep_all%s' % tag] = (out['miss_gauge%s' % tag] * out['miss_bg%s' % tag]
                               * out['miss_contour%s' % tag])
    return out


def summarize(trial_rows, empty_rows, distances, bases, fph_total):
    """Сводка: строка на (посадка, дистанция) + строка FP на (посадку, набор кадров).

    Наборов пустых кадров два: 'block' (кадры 12..71, событие в любом месте
    коридора) и 'window' (кадры окна испытаний без объекта, история трекера
    совпадает со сценой -- сравнимо с P по испытаниям).
    """
    entries = []
    for base in bases:
        rows = [r for r in trial_rows if r['base'] == base]
        for D in distances:
            sel = [r for r in rows if r['distance_m'] == D]
            visible = [r for r in sel if r['rays_hit'] > 0]
            in_range = [r for r in sel if r['in_range']]
            entry = {
                'kind': 'distance', 'base': base, 'distance_m': float(D),
                'trials': len(sel), 'trials_in_range': len(in_range),
                'trials_visible': len(visible),
                'p_gauge': rate(sel, 'gauge_hit'), 'p_bg': rate(sel, 'bg_hit'),
                'p_contour': rate(sel, 'contour_hit'),
                'p_any': rate(sel, 'any_hit'), 'p_two_of_three': rate(sel, 'two_of_three'),
                'p_gauge_in_range': rate(in_range, 'gauge_hit'),
                'p_gauge_visible': rate(visible, 'gauge_hit'),
                'p_bg_visible': rate(visible, 'bg_hit'),
                'p_contour_visible': rate(visible, 'contour_hit'),
                'p_any_visible': rate(visible, 'any_hit'),
                'p_two_of_three_visible': rate(visible, 'two_of_three'),
                'points_object_median': (float(np.median([r['points_object'] for r in sel]))
                                         if sel else float('nan')),
            }
            entry.update(joint_misses(sel, ''))
            entry.update(joint_misses(visible, '_visible'))
            entries.append(entry)

    # Набор `block` (кадры 12..): событие в ЛЮБОМ месте коридора -- то же
    # определение, что в `fp_per_hour`, поэтому эти числа с ним и сверяются.
    raw = [r for r in empty_rows if r['frame_set'] == 'block']
    empty = [r for r in raw if not r['known_object']]
    known = [r for r in raw if r['known_object']]
    known_with_event = [r for r in known if r['gauge_hit']]
    known_missed = [r for r in known if not r['gauge_hit']]
    entries.append({
        'kind': 'empty', 'base': '', 'frame_set': 'block', 'distance_m': '',
        'trials': len(empty), 'trials_in_range': '', 'trials_visible': len(known),
        'p_gauge': rate(empty, 'gauge_hit'), 'p_bg': rate(empty, 'bg_hit'),
        'p_contour': rate(empty, 'contour_hit'),
        'p_any': rate(empty, 'any_hit'),
        'p_two_of_three': rate(empty, 'two_of_three'),
        'p_gauge_in_range': '', 'p_gauge_visible': '', 'p_bg_visible': '',
        'p_contour_visible': '', 'p_any_visible': '', 'p_two_of_three_visible': '',
        'points_object_median': '',
        'fp_frames_all': len(raw), 'fp_frames_known_object': len(known),
        'fp_gauge_raw': rate(raw, 'gauge_hit'),
        'fp_gauge_known_event': len(known_with_event),
        'fp_gauge_known_missed': len(known_missed),
        'fp_per_hour_gauge_raw': rate(raw, 'gauge_hit') * float(fph_total),
        'fp_gauge': rate(empty, 'gauge_hit') * float(fph_total),
        'fp_bg': rate(empty, 'bg_hit') * float(fph_total),
        'fp_contour': rate(empty, 'contour_hit') * float(fph_total),
        'fp_any': rate(empty, 'any_hit') * float(fph_total),
        'fp_two_of_three': rate(empty, 'two_of_three') * float(fph_total),
    })

    # Кадры окна испытаний без объекта: FP в том же окне +-2 м на дистанции D --
    # единственная величина, которую можно сравнивать с P по испытаниям.
    for D in distances:
        rows = [r for r in empty_rows
                if r['frame_set'] == 'window' and r['distance_m'] == float(D)
                and not r['known_object']]
        entries.append({
            'kind': 'empty', 'base': '', 'frame_set': 'window',
            'distance_m': float(D),
            'trials': len(rows), 'trials_in_range': '', 'trials_visible': '',
            'p_gauge': rate(rows, 'gauge_hit'), 'p_bg': rate(rows, 'bg_hit'),
            'p_contour': rate(rows, 'contour_hit'),
            'p_any': rate(rows, 'any_hit'),
            'p_two_of_three': rate(rows, 'two_of_three'),
            'p_gauge_in_range': '', 'p_gauge_visible': '', 'p_bg_visible': '',
            'p_contour_visible': '', 'p_any_visible': '', 'p_two_of_three_visible': '',
            'points_object_median': '',
            'fp_frames_all': len(rows), 'fp_frames_known_object': '',
            'fp_gauge_raw': rate(rows, 'gauge_hit'), 'fp_gauge_known_event': '',
            'fp_gauge_known_missed': '',
            'fp_per_hour_gauge_raw': rate(rows, 'gauge_hit') * float(fph_total),
            'fp_gauge': rate(rows, 'gauge_hit') * float(fph_total),
            'fp_bg': rate(rows, 'bg_hit') * float(fph_total),
            'fp_contour': rate(rows, 'contour_hit') * float(fph_total),
            'fp_any': rate(rows, 'any_hit') * float(fph_total),
            'fp_two_of_three': rate(rows, 'two_of_three') * float(fph_total),
        })
    return entries


# ---------------------------------------------------------------------------
# Текст и файлы
# ---------------------------------------------------------------------------

def render_report(entries, infos, trial_rows, empty_rows, known_events, elapsed,
                  fph_total, frames_n, fp_frames, bases, revision=None):
    lines = []
    lines.append('Три канала реальной цепочки на протоколе инъекций '
                 '`run_range_curve`: %d записей x %d кадров x 3 положения x дистанции %s м'
                 % (len(infos), frames_n, '/'.join('%.0f' % d for d in sorted({
                     r['distance_m'] for r in trial_rows if r['kind'] == 'injection'}))))
    lines.append('объект %s м; окно события +-%.1f м от истинного центра; '
                 'пороги каналов -- их собственные, не подгонялись'
                 % ('x'.join('%.2f' % v for v in OBJECT_SIZE_DEFAULT), WINDOW_M))
    lines.append('габарит -- detector.core.detect (DetectorConfig по умолчанию, '
                 'ось -- track_models.json); фон -- stitch_bg.per_ray_residual '
                 '(>= %d точек, размах %.1f м, по высоте %.1f м)'
                 % (bg.MIN_CLUSTER_POINTS, bg.MIN_CLUSTER_EXTENT_M, bg.MIN_CLUSTER_EXTENT_Z_M))
    from detector import contour as _contour  # noqa: PLC0415

    lines.append('контур -- detector.contour.free_width + WidthTracker (сужение %.2f м, '
                 '%d бина подряд, история %d пустых кадров)'
                 % (_contour.WIDTH_NARROW_M, _contour.PERSIST_BINS, _contour.WINDOW))
    if revision:
        lines.append('ревизия detector/ (снимок ДО первого кадра): core.py %s, '
                     'profiles.py %s, track_models.json %s'
                     % tuple(revision[key][:12] for key in REVISION_FILES))
        lines.append('  detector/** правится параллельно: найдено, что подмена '
                     'track_models.json В СЕРЕДИНЕ прогона меняет valid_far_m '
                     '(-60.75 -> -61.25) и стоит 2 кадров человека из 46')
    lines.append('')
    lines.append('посадка объекта (пол -- МНК `synth.fit_floor_ab`, УГР -- модель детектора):')
    for base in bases:
        lines.append('  %-6s %s' % (base, BASE_NAMES[base]))
    lines.append('')
    lines.append('записи:')
    for info in infos:
        lines.append('  %-40s кадров %4d | окно %3d..%3d | ось %s | '
                     'участок габарита %7.2f..%.2f м | фон устойчив %.1f %% | '
                     'кадров FP %d (пустых %.0f %%) | пол-УГР %s'
                     % (info['bag'], info['frames_total'], info['window_start'],
                        info['window_start'] + info['frames_n'] - 1,
                        info['model_source'], info['valid_far_m'], info['y_near_m'],
                        100.0 * info['bg_stable_frac'], info['fp_frames'],
                        100.0 * info['empty_frac_fp'],
                        '/'.join('%+.2f' % v for v in info['floor_minus_ugr_m'].values())))
    lines.append('')
    for base in bases:
        lines.append('P(обнаружения) -- посадка `%s`: %s' % (base, BASE_NAMES[base]))
        lines.append(' дист | испыт | в оси | видим | P(габарит) | P(габ|в оси) (n) | '
                     'P(фон) | P(контур) | 1-из-3 | 2-из-3 | точек объекта, медиана')
        lines.append('-' * 132)
        for e in entries:
            if e['kind'] != 'distance' or e['base'] != base:
                continue
            bar = '#' * int(round(20 * e['p_any']))
            lines.append('%5.0f | %5d | %5d | %5d  | %5.1f %% | %s (%4d) | '
                         '%5.1f %% | %6.1f %% | %5.1f %% %-5s | %5.1f %% | %6.1f'
                         % (e['distance_m'], e['trials'], e['trials_in_range'],
                            e['trials_visible'], 100 * e['p_gauge'],
                            _pct(e['p_gauge_in_range']), e['trials_in_range'],
                            100 * e['p_bg'], 100 * e['p_contour'],
                            100 * e['p_any'], bar, 100 * e['p_two_of_three'],
                            e['points_object_median']))
        lines.append('')
    empty_entries = [e for e in entries if e['kind'] == 'empty']
    block_entry = next((e for e in empty_entries if e.get('frame_set') == 'block'), None)
    window_entries = [e for e in empty_entries if e.get('frame_set') == 'window']
    lines.append('Ложные на пустых кадрах. Набор `block`: %d кадров на запись x %d записей '
                 '= %d, из них с известным объектом %d (человек в doubleT_obstacle, из '
                 'fp_per_hour_events.csv) -- исключены из знаменателя; знаменатель %d'
                 % (fp_frames, len(infos),
                    (block_entry['fp_frames_all'] if block_entry else 0),
                    (block_entry['fp_frames_known_object'] if block_entry else 0),
                    (block_entry['trials'] if block_entry else 0)))
    if block_entry is not None and block_entry['trials'] < 300:
        lines.append('  ВНИМАНИЕ: знаменатель меньше 300 кадров -- это меньше заданного '
                     'минимума, FP/ч по нему не показателен')
    lines.append('событие в наборе `block` = срабатывание в ЛЮБОМ месте коридора (как '
                 'fp_per_hour); частота кадров записей %.2f кадр/ч (кадры/часы '
                 'metadata.yaml), по ней FP/ч' % fph_total)
    if block_entry is not None:
        lines.append('')
        lines.append(' набор `block` (кадры 12..61+ каждой записи; история трекера '
                     'совпадает со сценой лишь частично -- как в бою)')
        lines.append(' правило          | событий | FP/кадр | FP/ч (по кадрам/часам записей)')
        lines.append('-' * 72)
        for key, name in (('p_gauge', 'габарит'), ('p_bg', 'фон'), ('p_contour', 'контур'),
                          ('p_any', '1-из-3'), ('p_two_of_three', '2-из-3')):
            per_frame = block_entry[key]
            per_hour = block_entry['fp_gauge' if key == 'p_gauge' else
                                   'fp_%s' % key.split('_', 1)[1]]
            lines.append(' %-15s | %7.0f | %7.5f | %s'
                         % (name, per_frame * block_entry['trials'], per_frame,
                            _fmt_rate(per_hour)))
    if window_entries:
        lines.append('')
        lines.append(' набор `window` (кадры испытаний БЕЗ объекта, событие в том же окне '
                     '+-2 м на дистанции D; история трекера -- та же)')
        lines.append('  это единственные FP, сравнимые с P по испытаниям; канал фона здесь '
                     'пуст ПО ПОСТРОЕНИЮ (эти кадры и есть критерий пустоты)')
        lines.append(' дист | кадров | FP(габарит) | FP(фон) | FP(контур) | FP(1-из-3) | '
                     'FP(2-из-3)')
        lines.append('-' * 76)
        for e in window_entries:
            lines.append('%5.0f | %6d | %11.5f | %7.5f | %10.5f | %10.5f | %10.5f'
                         % (e['distance_m'], e['trials'], e['p_gauge'], e['p_bg'],
                            e['p_contour'], e['p_any'], e['p_two_of_three']))
    lines.append('')
    lines.append('сверка канала габарита с `python -m validation.fp_per_hour` '
                 '(validation/out/fp_per_hour_events.csv, 46 событий -- человек в '
                 'doubleT_obstacle):')
    if block_entry is not None:
        raw_events = int(round(block_entry['fp_gauge_raw'] * block_entry['fp_frames_all']))
        fp_events = int(round(block_entry['p_gauge'] * block_entry['trials']))
        same = (raw_events == block_entry['fp_gauge_known_event']
                and block_entry['fp_gauge_known_missed'] == 0 and fp_events == 0)
        lines.append('  кадры блока FP: %d; кадров с известным объектом по '
                     'fp_per_hour_events.csv %d, из них с событием габарита %d '
                     '(без события %d)'
                     % (block_entry['fp_frames_all'], block_entry['fp_frames_known_object'],
                        block_entry['fp_gauge_known_event'],
                        block_entry['fp_gauge_known_missed']))
        lines.append('  событий габарита на блоке %d, из них ЛОЖНЫХ (кадры без известного '
                     'объекта) %d = %.5f/кадр; сверка: %s'
                     % (raw_events, fp_events, block_entry['p_gauge'],
                        'события габарита -- ровно кадры fp_per_hour, лишних нет'
                        if same else
                        'РАСХОЖДЕНИЕ с fp_per_hour (см. числа выше)'))
    lines.append('')
    lines.append('совместные пропуски (joint miss) против произведения вероятностей '
                 'пропуска по отдельности:')
    lines.append(' дист | P(проп. габ) P(проп. фон) P(проп. конт) | joint габ+фон '
                 '(произв.) | joint габ+конт (произв.) | joint фон+конт (произв.) | '
                 'joint все (произв.)')
    lines.append('-' * 150)
    for base in bases:
        lines.append(' посадка `%s` (все испытания):' % base)
        for e in entries:
            if e['kind'] != 'distance' or e['base'] != base:
                continue
            lines.append('%5.0f | %5.3f %5.3f %5.3f | %5.3f (%5.3f) | %5.3f (%5.3f) | '
                         '%5.3f (%5.3f) | %5.3f (%5.3f)'
                         % (e['distance_m'], e['miss_gauge'], e['miss_bg'], e['miss_contour'],
                            e['joint_miss_gauge_bg'], e['indep_gauge_bg'],
                            e['joint_miss_gauge_contour'], e['indep_gauge_contour'],
                            e['joint_miss_bg_contour'], e['indep_bg_contour'],
                            e['joint_miss_all'], e['indep_all']))
        pooled = [r for r in trial_rows if r['base'] == base]
        vis = [r for r in pooled if r['rays_hit'] > 0]
        for label, rows in (('все', pooled), ('объект видим (rays_hit>0)', vis)):
            jm = joint_misses(rows)
            lines.append('  %-28s n=%4d | P(проп. габ/фон/конт) %5.3f %5.3f %5.3f | '
                         'joint габ+фон %5.3f против %5.3f | габ+конт %5.3f против %5.3f | '
                         'фон+конт %5.3f против %5.3f | все три %5.3f против %5.3f'
                         % (label, len(rows), jm['miss_gauge'], jm['miss_bg'],
                            jm['miss_contour'],
                            jm['joint_miss_gauge_bg'], jm['indep_gauge_bg'],
                            jm['joint_miss_gauge_contour'], jm['indep_gauge_contour'],
                            jm['joint_miss_bg_contour'], jm['indep_bg_contour'],
                            jm['joint_miss_all'], jm['indep_all']))
        lines.append('')
    lines.append('вывод о независимости -- по отношению joint/произведение в блоке ниже:')
    for base in bases:
        rows = [r for r in trial_rows if r['base'] == base]
        jm = joint_misses(rows)
        pair = [('габ+фон', jm['joint_miss_gauge_bg'], jm['indep_gauge_bg']),
                ('габ+конт', jm['joint_miss_gauge_contour'], jm['indep_gauge_contour']),
                ('фон+конт', jm['joint_miss_bg_contour'], jm['indep_bg_contour']),
                ('все три', jm['joint_miss_all'], jm['indep_all'])]
        for label, joint, indep in pair:
            ratio = (joint / indep) if indep > 0 else float('inf')
            lines.append('  %-6s %-9s joint %.3f против произведения %.3f --> '
                         'отношение %s (%s)'
                         % (base, label, joint, indep, _fmt_rate(ratio),
                            'СВЯЗАНЫ' if joint > indep + 1e-12 else 'не отличаются'))
    lines.append('')
    lines.append('прогон %.1f с' % elapsed)
    return '\n'.join(lines)


def _fmt_rate(value):
    if value != value:
        return '--'
    if value == 0:
        return '0'
    return '%.3g' % value


def _pct(value):
    """Процент или '--': у вероятности по подвыборке нулевой длины её нет."""
    if value != value:
        return '--'
    return '%.1f %%' % (100.0 * value)


def write_outputs(trial_rows, empty_rows, entries, text, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    rows = list(trial_rows) + list(empty_rows)
    trials_path = os.path.join(out_dir, 'detector_curve_trials.csv')
    with open(trials_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction='ignore',
                                restval='')
        writer.writeheader()
        writer.writerows(rows)
    fields = ['kind', 'base', 'frame_set', 'distance_m', 'trials', 'trials_in_range',
              'trials_visible',
              'p_gauge', 'p_gauge_in_range', 'p_gauge_visible', 'p_bg', 'p_bg_visible',
              'p_contour', 'p_contour_visible', 'p_any', 'p_any_visible',
              'p_two_of_three', 'p_two_of_three_visible', 'points_object_median',
              'miss_gauge', 'miss_bg', 'miss_contour',
              'joint_miss_gauge_bg', 'indep_gauge_bg',
              'joint_miss_gauge_contour', 'indep_gauge_contour',
              'joint_miss_bg_contour', 'indep_bg_contour',
              'joint_miss_all', 'indep_all',
              'miss_gauge_visible', 'miss_bg_visible', 'miss_contour_visible',
              'joint_miss_gauge_bg_visible', 'indep_gauge_bg_visible',
              'joint_miss_gauge_contour_visible', 'indep_gauge_contour_visible',
              'joint_miss_bg_contour_visible', 'indep_bg_contour_visible',
              'joint_miss_all_visible', 'indep_all_visible',
              'fp_frames_all', 'fp_frames_known_object', 'fp_gauge_raw',
              'fp_per_hour_gauge_raw', 'fp_gauge', 'fp_bg', 'fp_contour', 'fp_any',
              'fp_two_of_three']
    summary_path = os.path.join(out_dir, 'detector_curve_summary.csv')
    with open(summary_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore',
                                restval='')
        writer.writeheader()
        writer.writerows(entries)
    text_path = os.path.join(out_dir, 'detector_curve.txt')
    with open(text_path, 'w', encoding='utf-8') as handle:
        handle.write(text + '\n')
    return trials_path, summary_path, text_path


def read_known_events(path, fallback=None):
    """Кадры с известным настоящим объектом -- из отчёта `fp_per_hour`.

    `fallback` -- второй путь на случай, когда прогон пишет в другой каталог:
    без известных кадров человек в `doubleT_obstacle` попал бы в знаменатель FP.
    """
    for candidate in (path, fallback):
        if not candidate:
            continue
        known = set()
        try:
            with open(candidate, encoding='utf-8') as handle:
                for row in csv.DictReader(handle):
                    try:
                        known.add((row['bag'], int(row['frame'])))
                    except (KeyError, TypeError, ValueError):
                        continue
        except OSError:
            continue
        if known:
            return known
    return set()


def bag_duration_h(db_path):
    """Длительность записи из `metadata.yaml` (часы) -- как в `fp_per_hour`."""
    import re  # noqa: PLC0415

    meta = os.path.join(os.path.dirname(db_path), 'metadata.yaml')
    try:
        with open(meta, encoding='utf-8') as handle:
            text = handle.read()
    except OSError:
        return None
    found = re.search(r'^  duration:\s*\n\s+nanoseconds:\s*(\d+)', text, re.M)
    return int(found.group(1)) / 1e9 / 3600.0 if found else None


def frames_per_hour(infos):
    """Частота кадров по записям: сумма кадров / сумма часов (fp_per_hour)."""
    frames = 0
    hours = 0.0
    for info in infos:
        duration = bag_duration_h(info['db_path'])
        if not duration:
            continue
        frames += info['frames_total']
        hours += duration
    return (frames / hours) if hours > 0 else float('nan')


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Кривая «дистанция -> P(обнаружения)» по трём каналам цепочки')
    parser.add_argument('--bags', nargs='*', default=None, help='имена записей')
    parser.add_argument('--frames', type=int, default=FRAMES_PER_BAG,
                        help='кадров окна на запись (протокол: %d)' % FRAMES_PER_BAG)
    parser.add_argument('--distances', type=float, nargs='*', default=None)
    parser.add_argument('--bases', nargs='*', default=list(BASES), choices=list(BASES))
    parser.add_argument('--fp-frames', type=int, default=FP_FRAMES_PER_BAG,
                        help='кадров на запись для FP (>= 50 при 6 записях)')
    parser.add_argument('--out-dir', default=None)
    args = parser.parse_args(argv)

    root = project_root()
    out_dir = args.out_dir or os.path.join(root, 'validation', 'out')
    bags = bag_db_paths(root, args.bags or RING_BAG_NAMES)
    if not bags:
        print('нет записей for_hackathon/*/*.db3')
        return 2
    distances = tuple(args.distances) if args.distances else DISTANCES
    bases = tuple(args.bases)
    known_events = read_known_events(
        os.path.join(out_dir, 'fp_per_hour_events.csv'),
        fallback=os.path.join(root, 'validation', 'out', 'fp_per_hour_events.csv'))
    cache_path = os.path.join(out_dir, 'empty_windows.json')

    # Единый снимок детектора: ревизия снимается ДО первого кадра, модели всех
    # записей грузятся сразу (иначе подмена track_models.json в середине прогона
    # даст отчёт из двух ревизий -- измерено, см. `detector_revision`).
    revision = detector_revision()
    models = {}
    for name, db in bags:
        try:
            models[name] = model_for_record(db)
        except Exception as exc:  # noqa: BLE001 -- detect() сам подставит модель
            print('  !! модель %s не построена (%s) -- считает detect' % (name, type(exc).__name__),
                  flush=True)
            models[name] = None

    print('ревизия detector/ на начало прогона: core.py %s, profiles.py %s, '
          'track_models.json %s'
          % tuple(revision[key][:12] for key in REVISION_FILES), flush=True)
    print('дистанции %s; посадки %s; кадров окна %d; кадров FP на запись %d; '
          'известных объектов в блоке FP: %d'
          % ('/'.join('%.0f' % d for d in distances), '/'.join(bases), args.frames,
             args.fp_frames, len(known_events)), flush=True)

    t_start = time.time()
    trial_rows, empty_rows, infos = [], [], []
    for name, db in bags:
        print(' %s: %s' % (name, os.path.basename(db)), flush=True)
        try:
            rows, empties, info = run_record(
                name, db, distances=distances, frames_n=args.frames, bases=bases,
                fp_frames=args.fp_frames, start=window_start(cache_path, name, args.frames),
                known_events=known_events, model=models.get(name))
        except Exception as exc:  # noqa: BLE001 -- одну запись терять нельзя
            print('  !! %s упала: %s: %s -- продолжаю' % (name, type(exc).__name__, exc),
                  flush=True)
            continue
        trial_rows.extend(rows)
        empty_rows.extend(empties)
        infos.append(info)
        print('  %-40s испытаний %d, пустых кадров %d, %.1f с'
              % (name, len(rows), len(empties), info['seconds']), flush=True)
    if not trial_rows:
        print('испытаний не получилось -- нечего писать')
        return 2
    elapsed = time.time() - t_start

    after = detector_revision()
    changed = [key for key in REVISION_FILES if after[key] != revision[key]]
    fph_total = frames_per_hour(infos)
    entries = summarize(trial_rows, empty_rows, distances, bases, fph_total)
    text = render_report(entries, infos, trial_rows, empty_rows, known_events, elapsed,
                         fph_total, args.frames, args.fp_frames, bases, revision)
    paths = write_outputs(trial_rows, empty_rows, entries, text, out_dir)
    print()
    print(text)
    print()
    print('испытаний %d, пустых кадров %d, прогон %.1f с' % (len(trial_rows),
                                                             len(empty_rows), elapsed))
    print('файлы: %s' % ', '.join(os.path.basename(p) for p in paths))
    print('ревизия detector/ после прогона %s (числа отчёта -- снимок начала прогона)'
          % ('ИЗМЕНИЛАСЬ: ' + ', '.join(changed) if changed else 'не менялась'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())