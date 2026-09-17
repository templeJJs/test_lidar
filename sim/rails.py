"""Поиск рельсовых нитей по облаку: линии головок, выступающие над ложем.

Зачем отдельный модуль: `dataset_facts._rail_bands` (убран из артефакта как мусор)
искал просто локальные максимумы гистограммы по X в полосе высоты головки, и в
записях находилось восемь «нитей» на 2.5 м при стенах 9.2 м. Первая попытка здесь
же -- трекинг линий по пикам X в ячейках вдоль пути -- на синтетике находил ровно
два рельса, а на записях давала десятки линий: тот же признак дают неровности
ложа, ведь точки в полосе высоты головки есть у любого бугра.

Признак, который отделяет головку от бугра: головка ВЫСТУПАЕТ над своим ложем.
Для каждой ячейки вдоль пути и каждого бина по X сравниваем высоту точек бина с
уровнем ложа рядом с ним (|Δx| 0.25…0.60 м): у головки разница 0.10…0.16 м, у
бугра -- около нуля. Дальше обычный трекинг нити вдоль пути с пропуском ячеек.
"""
from __future__ import annotations

import numpy as np

import zones

HEAD_LOW_M = 0.08               # головка рельса R65: 16 см над ложем, берём с запасом
HEAD_HIGH_M = 0.28
BED_OUTER_M = (0.25, 0.60)      # где меряем ложе рядом с бином
RAIL_ABOVE_BED_M = 0.06         # насколько головка должна быть выше своего ложа
BIN_X_M = 0.04
CELL_Y_M = 0.50
CELL_MIN_POINTS = 2             # меньше точек в бине -- нечего сравнивать
LINK_TOL_M = 0.10
GAP_CELLS = 3                   # сколько ячеек подряд можно пропустить
MIN_CELLS = 6                   # короче 3 м -- не нить
MIN_POINTS = 60
MERGE_TOL_M = 0.12
WALL_INSET_M = 0.30             # рельсов за нитью стены не бывает
FLOOR_BAND_HALF_M = 2.0         # пол для отсчёта высоты берём в полосе пути


def _prepared(xyz: np.ndarray, walls: list):
    """Точки внутри нитей стен с высотой над ложем: (x, y, z_rel, lo, hi)."""
    a, b, _ = zones.fit_floor(xyz, y_min=-40.0, y_max=-2.0, x_center=0.0,
                              x_half=FLOOR_BAND_HALF_M)
    z_rel = xyz[:, 2] - (a + b * xyz[:, 1])
    x, y = xyz[:, 0], xyz[:, 1]
    if walls and len(walls) >= 2:
        lo = float(min(walls)) + WALL_INSET_M
        hi = float(max(walls)) - WALL_INSET_M
    else:
        lo, hi = -2.5, 2.5
    inside = (x >= lo) & (x <= hi)
    return x[inside], y[inside], z_rel[inside], lo, hi


def _rails_in_cell(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                   y_lo: float, y_hi: float) -> list:
    """Нити в одной ячейке вдоль пути: [(x, сколько точек)] по бинам, что выступают.

    Для бина берём точки в полосе высоты головки и сравниваем их медиану с медианой
    точек ложа рядом (|Δx| 0.25…0.60 м). Головка выше ложа на 0.10…0.16 м, бугор
    ложа -- на нуль, поэтому порог 0.06 м их разделяет.
    """
    m = (y >= y_lo) & (y < y_hi)
    if int(np.count_nonzero(m)) < 4:
        return []
    xs, zs = x[m], z[m]
    out = []
    edges = np.arange(xs.min(), xs.max() + BIN_X_M, BIN_X_M)
    for i in range(edges.size - 1):
        in_bin = (xs >= edges[i]) & (xs < edges[i + 1])
        n = int(np.count_nonzero(in_bin))
        if n < CELL_MIN_POINTS:
            continue
        level = float(np.median(zs[in_bin]))
        if not (HEAD_LOW_M <= level <= HEAD_HIGH_M):
            continue
        near = (np.abs(xs - 0.5 * (edges[i] + edges[i + 1])) >= BED_OUTER_M[0]) \
            & (np.abs(xs - 0.5 * (edges[i] + edges[i + 1])) <= BED_OUTER_M[1])
        if int(np.count_nonzero(near)) < CELL_MIN_POINTS:
            continue
        bed = float(np.median(zs[near]))
        if level - bed >= RAIL_ABOVE_BED_M:
            out.append((float(0.5 * (edges[i] + edges[i + 1])), n))
    return out


def rail_lines(xyz: np.ndarray, walls: list = None, y_min: float = -60.0,
               y_max: float = -2.0, min_cells: int = MIN_CELLS) -> list:
    """Нити рельсов: x при y=0, наклон, длина, число точек.

    Трекинг идёт ОТ сенсора вдаль (по убыванию Y): в каждой ячейке берутся нити
    (`_rails_in_cell`), и к треку привязывается ближайшая к продолжению линии.
    Ячейку можно пропустить (вдали возвратов мало), но не больше `GAP_CELLS`
    подряд. Затем нить достраивается МНК по всем своим точкам, а близкие нити
    сливаются: одна головка даёт несколько бинов.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.shape[0] < 200:
        return []
    x, y, z, _lo, _hi = _prepared(xyz, walls)
    if x.size < MIN_POINTS:
        return []
    first = int(np.floor((y.min() - y_min) / CELL_Y_M))
    last = int(np.floor((y.max() - y_min) / CELL_Y_M))
    yc = lambda c: y_min + (c + 0.5) * CELL_Y_M            # noqa: E731
    peaks = {c: _rails_in_cell(x, y, z, y_min + c * CELL_Y_M, y_min + (c + 1) * CELL_Y_M)
             for c in range(first, last + 1)}

    tracks = []
    for c in range(last, first - 1, -1):
        for px, n in peaks.get(c, ()):
            tracks.append({'cells': [c], 'pts': [n], 'y': [yc(c)], 'x': [px]})
        for t in tracks:
            if t['cells'][-1] != c:
                continue
            slope = ((t['x'][-1] - t['x'][-2]) / (t['y'][-1] - t['y'][-2])
                     if len(t['y']) >= 2 else 0.0)
            for gap in range(1, GAP_CELLS + 1):
                nxt = c - gap
                if nxt < first:
                    break
                want = t['x'][-1] + slope * (yc(nxt) - t['y'][-1])
                cand = [(abs(px - want), px, n) for px, n in peaks.get(nxt, ())]
                if not cand:
                    continue
                d, px, n = min(cand)
                if d > LINK_TOL_M + 0.03 * (gap - 1):
                    continue
                t['cells'].append(nxt)
                t['pts'].append(n)
                t['y'].append(yc(nxt))
                t['x'].append(px)
                break

    fitted = []
    for t in tracks:
        if len(t['cells']) < min_cells or int(np.sum(t['pts'])) < MIN_POINTS:
            continue
        yy = np.asarray(t['y'])
        xx = np.asarray(t['x'])
        w = np.asarray(t['pts'], dtype=float)
        if float(np.ptp(yy)) < 1e-6:
            continue
        k, b = np.polyfit(yy, xx, 1, w=w)          # x = k*y + b
        fitted.append({'x0': float(b), 'slope': float(k), 'points': int(np.sum(w)),
                       'y_from': float(yy.max()), 'y_to': float(yy.min()),
                       'length_m': float(np.ptp(yy))})
    fitted.sort(key=lambda t: -t['points'])
    kept = []
    for t in fitted:
        if any(abs(t['x0'] - q['x0']) <= MERGE_TOL_M for q in kept):
            continue
        kept.append(t)
    kept.sort(key=lambda t: t['x0'])
    for t in kept:
        for key in ('x0', 'slope', 'y_from', 'y_to', 'length_m'):
            t[key] = round(t[key], 3)
    return kept


def track_axes(lines: list, gauge_m: float = 1.520, tol_m: float = 0.15,
               min_points: int = 300, min_length_m: float = 10.0) -> list:
    """Оси путей по найденным нитям: пары на расстоянии колеи.

    Отвечает на вопрос «сколько путей видно и где». Нитей в записях находится
    много (ложе даёт линии не хуже рельсов), поэтому пары ранжируются по слабейшей
    из двух нитей: у настоящего пути ОБЕ нити длинные и густые, у случайной пары
    хотя бы одна короткая.
    """
    axes = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            d = abs(lines[j]['x0'] - lines[i]['x0'])
            if abs(d - gauge_m) > tol_m:
                continue
            points = min(lines[i]['points'], lines[j]['points'])
            length = min(lines[i]['length_m'], lines[j]['length_m'])
            if points < min_points or length < min_length_m:
                continue
            axes.append({'axis_x': round(0.5 * (lines[i]['x0'] + lines[j]['x0']), 3),
                         'gauge_m': round(d, 3),
                         'points': int(points), 'length_m': round(length, 1)})
    axes.sort(key=lambda a: -a['points'])
    kept = []
    for a in axes:
        if any(abs(a['axis_x'] - q['axis_x']) <= 0.30 for q in kept):
            continue
        kept.append(a)
    kept.sort(key=lambda a: a['axis_x'])
    return kept

TRACK_AXIS_CLUSTER_M = 0.12    # нити одной головки в разных кадрах гуляют по x
TRACK_AXIS_MIN_FRAMES = 2       # обе нити должны быть видны хотя бы в двух кадрах
TRACK_AXIS_MIN_POINTS = 120     # слабейшая нить пары: меньше -- это не рельс
TRACK_AXIS_MIN_LENGTH_M = 3.5


def _cluster_lines(per_frame: list) -> list:
    """Слить нити из разных кадров в группы по x (одна головка -- одна группа)."""
    flat = []
    for frame_idx, lines in enumerate(per_frame):
        for line in lines:
            flat.append((float(line['x0']), int(line['points']),
                         float(line['length_m']), frame_idx))
    flat.sort(key=lambda t: t[0])
    clusters = []
    for x0, points, length, frame_idx in flat:
        if clusters and abs(x0 - clusters[-1]['x']) <= TRACK_AXIS_CLUSTER_M:
            g = clusters[-1]
            g['x'] = (g['x'] * g['n'] + x0) / (g['n'] + 1)
            g['n'] += 1
            g['points'] = max(g['points'], points)
            g['length_m'] = max(g['length_m'], length)
            g['frames'].add(frame_idx)
        else:
            clusters.append({'x': x0, 'n': 1, 'points': points, 'length_m': length,
                             'frames': {frame_idx}})
    for g in clusters:
        g['x'] = round(g['x'], 3)
        g['length_m'] = round(g['length_m'], 1)
        g['frames'] = len(g['frames'])
    return clusters


def track_axis(frame_clouds: list, walls: list, gauge_m: float = 1.520,
               tol_m: float = 0.20) -> dict:
    """Ось пути по кадрам записи: лучшая пара нитей на расстоянии колеи.

    Нити одной головки в разных кадрах гуляют по x, поэтому они сливаются в
    группы (`_cluster_lines`), и у группы считается, в скольких кадрах её видели,
    сколько точек и какая длина у самой длинной нити. Пара на расстоянии колеи
    выбирается по слабейшей нити: у настоящего пути ОБЕ нити длинные и густые.

    Зачем: сцена ставит путь на `x = 0`, а у записи ось может стоять не там.
    Возврат: `{'axis_x', 'gauge_m', 'points', 'length_m', 'frames'}` или `{}`.
    """
    frames = len(frame_clouds)
    clusters = _cluster_lines([rail_lines(np.asarray(c, dtype=np.float64), walls=walls)
                               for c in frame_clouds])
    clusters.sort(key=lambda g: -g['points'])
    best = {}
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            a, b = clusters[i], clusters[j]
            gap = abs(b['x'] - a['x'])
            if abs(gap - gauge_m) > tol_m:
                continue
            weak_points = min(a['points'], b['points'])
            weak_length = min(a['length_m'], b['length_m'])
            weak_frames = min(a['frames'], b['frames'])
            if (weak_points < TRACK_AXIS_MIN_POINTS or weak_length < TRACK_AXIS_MIN_LENGTH_M
                    or weak_frames < min(TRACK_AXIS_MIN_FRAMES, frames)):
                continue
            score = weak_points * weak_length
            if score > best.get('score', 0.0):
                best = {'score': score,
                        'axis_x': round(0.5 * (a['x'] + b['x']), 3),
                        'gauge_m': round(gap, 3), 'points': int(weak_points),
                        'length_m': round(weak_length, 1), 'frames': int(weak_frames)}
    best.pop('score', None)
    return best
