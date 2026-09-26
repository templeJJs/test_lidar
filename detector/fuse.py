"""Взвешенное слияние каналов детектора (лог-правдоподобие Chair--Varshney).

Счётные правила `validation.detector_curve` заведомо субоптимальны: `1-из-3`
наследует FP худшего канала (контур: 0.946/кадр на блоке), `2-из-3` обнуляет
FP, но режет полноту ниже одиночного канала габарита. Здесь вместо счёта
голосов -- взвешенная сумма срабатываний:

    S(d) = sum_i hit_i * w_i(d),   w_i(d) = ln(Pd_i(d) / Pf*_i(d))

где Pd_i(d) -- полнота канала на дистанции d по испытаниям инъекций, а
Pf*_i(d) -- частота ложных срабатываний:

  * Pf из набора `window` (пустые кадры испытаний, событие в том же окне
    +-WINDOW_M на той же дистанции) -- единственный набор, сравнимый с P по
    испытаниям (см. шапку `validation.detector_curve`);
  * нулевой Pf заменяется верхней границей «правила трёх»: 3/n_block, если
    канал чист и на блоке (n=314 > 60, и событие блока «в любом месте
    коридора» шире оконного -- его ноль сильнее), иначе 3/n_window. Для
    контура блок НЕ репрезентативен (история трекера не сцены), поэтому его
    bound -- всегда 3/n_window;
  * Pd = 0 и Pf = 0 одновременно -> w = 0: канал на этой дистанции не даёт
    свидетельства ни за, ни против (канал вне зоны применимости -- не мисс);
  * Pd = 0 при Pf > 0 -> w = ln((0.5/n_trials)/Pf) < 0: срабатывание канала,
    который на этой дистанции никогда не видел объект, -- свидетельство
    ПРОТИВ (это почти наверняка ложное).

Два правила подсчёта: мисс канала НЕ голосует против (в S входят только
hit-члены), а канал вне зоны применимости (габарит вне valid_range оси,
контур без истории трекера) -- «нет свидетельства», а не мисс.

Порог -- эмпирический Нейман--Пирсон: gamma = max S по всем негативным
испытаниям + запас MARGIN_DEFAULT (негативы блока не имеют дистанции и
оцениваются худшим случаем -- max S по всем дистанциям весов). FP = 0 на
калибровочных данных по построению; на новых данных это не гарантия, а
оценка, и запас -- её цена.

`Obstacle.confidence` как байесовский вес НЕ используется: эта эвристика не
калибрована в Pd/Pf. Мягкое (soft) слияние отложено.

Калибровка по умолчанию (`DEFAULT_WEIGHTS`, `DEFAULT_GAMMA`) измерена полным
прогоном `python -m validation.detector_curve` (посадка `ugr`, объект
0.30 x 0.30 x 1.00 м); функции `fit_weights`/`calibrate_gamma` повторяют её на
свежих строках испытаний.
"""

import math

CHANNELS = ('gauge', 'bg', 'contour')
RULE_OF_THREE = 3.0      # верхняя граница «правила трёх» для нулевого счётчика
MARGIN_DEFAULT = 0.25    # запас Неймана--Пирсона поверх max S негативов (лог-единицы)
WINDOW_M = 2.0           # окно события вокруг центра объекта (как detector_curve)

# Калибровка полного прогона validation.detector_curve, посадка ugr
# (2880 строк: 2160 инъекций + 360 block + 360 window). Веса w_i(d) =
# ln(Pd_i(d)/Pf*_i(d)); gamma = max S по негативам (0.442 -- контур на 50 м)
# + запас 0.25.
DEFAULT_WEIGHTS = {  # (канал, дистанция): w
    ('gauge', 25.0): 4.495, ('bg', 25.0): 1.760, ('contour', 25.0): 0.223,
    ('gauge', 50.0): 4.053, ('bg', 50.0): 2.097, ('contour', 50.0): 0.442,
    ('gauge', 75.0): 0.000, ('bg', 75.0): 0.000, ('contour', 75.0): 0.154,
    ('gauge', 100.0): 0.000, ('bg', 100.0): 0.000, ('contour', 100.0): 0.000,
    ('gauge', 125.0): 0.000, ('bg', 125.0): 2.454, ('contour', 125.0): 0.000,
    ('gauge', 150.0): 0.000, ('bg', 150.0): 0.000, ('contour', 150.0): 0.000,
}
DEFAULT_GAMMA = 0.692    # 0.442 (max S негативов: контур, 50 м) + 0.25 запаса


def row_hits(row, channels=CHANNELS):
    """Вектор срабатываний каналов из строки испытаний (0/1; '' -> 0)."""
    out = []
    for channel in channels:
        value = row.get('%s_hit' % channel, 0)
        out.append(int(value) if value not in ('', None) else 0)
    return out


def channel_weight(n_hit, n_trials, n_fp_window, n_window,
                   n_fp_block=0, n_block=0):
    """Вес канала ln(Pd/Pf*) по счётчикам (см. шапку модуля).

    `n_fp_block`/`n_block` -- ложные канала на блоке пустых кадров: нужны
    только чтобы ужесточить bound нулевого Pf (3/314 вместо 3/60), когда канал
    чист на ОБОИХ наборах.
    """
    if n_hit == 0 and n_fp_window == 0:
        return 0.0                       # канал молчит и там, и там: нет свидетельства
    pd = (n_hit / n_trials) if n_hit else (0.5 / n_trials)
    if n_fp_window:
        pf = n_fp_window / n_window
    elif n_fp_block == 0 and n_block:
        pf = RULE_OF_THREE / n_block     # оба набора чисты: bound по большему n
    else:
        pf = RULE_OF_THREE / n_window
    return math.log(pd / pf)


def fit_weights(trial_rows, window_rows, block_rows, distances, base,
                channels=CHANNELS):
    """Веса {(канал, дистанция): w} по строкам испытаний одной посадки."""
    weights = {}
    for distance in distances:
        distance = float(distance)
        trials = [r for r in trial_rows
                  if r.get('base') == base and float(r['distance_m']) == distance]
        window = [r for r in window_rows if float(r['distance_m']) == distance]
        for j, channel in enumerate(channels):
            weights[(channel, distance)] = channel_weight(
                n_hit=sum(row_hits(r, channels)[j] for r in trials),
                n_trials=len(trials),
                n_fp_window=sum(row_hits(r, channels)[j] for r in window),
                n_window=len(window),
                n_fp_block=sum(row_hits(r, channels)[j] for r in block_rows),
                n_block=len(block_rows))
    return weights


def score_hits(hits, distance, weights, channels=CHANNELS):
    """S(d) = sum hit_i * w_i(d): мисс не голосует, вне зоны -- нет свидетельства."""
    return sum(h * weights.get((c, float(distance)), 0.0)
               for h, c in zip(hits, channels))


def score_hits_max(hits, weights, channels=CHANNELS):
    """Худший случай для кадра без дистанции: max S по всем дистанциям весов."""
    distances = {d for _c, d in weights}
    return max((score_hits(hits, d, weights, channels) for d in distances),
               default=0.0)


def calibrate_gamma(window_rows, block_rows, weights, margin=MARGIN_DEFAULT,
                    channels=CHANNELS):
    """Эмпирический Нейман--Пирсон: max S по всем негативам + запас.

    Негативы `window` оцениваются на своей дистанции, негативы `block` (без
    дистанции) -- худшим случаем `score_hits_max`.
    """
    best = 0.0
    for row in window_rows:
        best = max(best, score_hits(row_hits(row, channels),
                                    float(row['distance_m']), weights, channels))
    for row in block_rows:
        best = max(best, score_hits_max(row_hits(row, channels), weights, channels))
    return best + float(margin)


def apply_weighted(trial_rows, empty_rows, distances, bases,
                   margin=MARGIN_DEFAULT, channels=CHANNELS):
    """Калибровка по строкам прогона и простановка `weighted_hit` всем строкам.

    Веса и gamma -- свои у каждой посадки (Pd посадок различаются); негативы
    общие, и пустой кадр считается сработавшим, если срабатывает ХОТЯ БЫ ОДНА
    калибровка (консервативная оценка FP). Возвращает паспорт калибровки:
    {'weights': {base: ...}, 'gamma': {base: ...}, 'margin': margin}.
    """
    window_rows = [r for r in empty_rows
                   if r.get('frame_set') == 'window' and not int(r['known_object'] or 0)]
    block_rows = [r for r in empty_rows
                  if r.get('frame_set') == 'block' and not int(r['known_object'] or 0)]
    weights_by_base, gamma_by_base = {}, {}
    for base in bases:
        weights = fit_weights(trial_rows, window_rows, block_rows, distances, base,
                              channels)
        gamma = calibrate_gamma(window_rows, block_rows, weights, margin, channels)
        weights_by_base[base] = weights
        gamma_by_base[base] = gamma
        for row in trial_rows:
            if row.get('base') == base:
                row['weighted_hit'] = int(
                    score_hits(row_hits(row, channels), float(row['distance_m']),
                               weights, channels) >= gamma)
    for row in empty_rows:
        hits = row_hits(row, channels)
        if row.get('frame_set') == 'block' or row.get('distance_m') in ('', None):
            row['weighted_hit'] = int(any(
                score_hits_max(hits, weights_by_base[b], channels) >= gamma_by_base[b]
                for b in bases))
        else:
            row['weighted_hit'] = int(any(
                score_hits(hits, float(row['distance_m']), weights_by_base[b],
                           channels) >= gamma_by_base[b] for b in bases))
    return {'weights': weights_by_base, 'gamma': gamma_by_base, 'margin': margin}


def fuse_frame(points, ring, model, cfg=None, background=None, base_ab=None,
               axis_nodes=None, tracker=None, distance_m=None,
               weights=None, gamma=None, window_m=WINDOW_M):
    """Один кадр через три канала цепочки и взвешенное правило.

    Тонкая обёртка: `detector.core.detect` (габарит), `validation.stitch_bg`
    (фон на луч: остаток относительно фона `background` в гейте коридора
    `base_ab`/`axis_nodes`), `detector.contour.WidthTracker` (контур;
    `tracker` с историей, кадр оценивается на КОПИИ -- история вызывающего не
    портится). Пороги каналов -- их собственные, здесь не подгоняются.

    `distance_m` -- дистанция, на которой оценивается событие (окно
    +-`window_m`); None -- событие «в любом месте коридора» и худший случай по
    дистанциям весов. Веса/порог по умолчанию -- измеренная калибровка
    (`DEFAULT_WEIGHTS`, `DEFAULT_GAMMA`).

    Возвращает {'events': {канал: [дистанции событий]}, 'hits': (0/1, ...),
    'score': S, 'decision': bool}.
    """
    import copy  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415

    from .core import DetectorConfig, detect  # noqa: PLC0415
    from . import contour as _contour  # noqa: PLC0415
    from validation import stitch_bg as _bg  # noqa: PLC0415  (validation -> detector цикла нет: импорт ленивый)

    weights = DEFAULT_WEIGHTS if weights is None else weights
    gamma = DEFAULT_GAMMA if gamma is None else gamma
    cfg = DetectorConfig() if cfg is None else cfg

    def near(values, positive):
        values = np.asarray(values, dtype=np.float64)
        if values.size == 0:
            return False
        if distance_m is None:
            return True
        target = float(distance_m) if positive else -float(distance_m)
        return bool(np.any(np.abs(values - target) <= float(window_m)))

    # Габарит: y препятствий (отрицательные, тоннель в -Y) -> дистанции |y|.
    result = detect(np.asarray(points, dtype=np.float64), model=model, cfg=cfg)
    gauge_dists = [abs(float(o.y_m)) for o in (result.obstacles or [])]

    # Фон на луч: кластеры остатка в гейте коридора, центр -- медиана y.
    bg_dists = []
    if background is not None and ring is not None:
        residue = _bg.per_ray_residual(points, ring, background)
        if residue.shape[0] and base_ab is not None and axis_nodes is not None:
            residue = residue[_bg.corridor_gate(residue, base_ab, axis_nodes)]
        clusters = _bg.count_clusters(residue)
        bg_dists = [abs(float(np.median(c['points'][:, 1]))) for c in clusters]

    # Контур: события WidthTracker на копии истории (дистанции положительные).
    contour_dists = []
    if tracker is not None:
        _y, left, right = _contour.free_width(np.asarray(points, dtype=np.float64),
                                              model)
        events = copy.deepcopy(tracker).update(left, right, _y)
        contour_dists = [float(e['distance_m']) for e in events]

    hits = (int(near(gauge_dists, positive=False)),
            int(near(bg_dists, positive=False)),
            int(near(contour_dists, positive=True)))
    if distance_m is None:
        score = score_hits_max(hits, weights)
    else:
        distance = float(distance_m)
        if (CHANNELS[0], distance) not in weights:   # ближайшая калиброванная
            distance = min({d for _c, d in weights}, key=lambda d: abs(d - distance))
        score = score_hits(hits, distance, weights)
    return {'events': {'gauge': gauge_dists, 'bg': bg_dists,
                       'contour': contour_dists},
            'hits': hits, 'score': float(score), 'decision': bool(score >= gamma)}
