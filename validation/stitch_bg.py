"""Негативы из нескольких кадров: фон и остаток кадра относительно фона.

Второй, независимый от `synth` источник проверки: вместо внесения объекта --
склейка фона по нескольким кадрам и поиск того, что в кадре отличается от фона.
Сенсор в тоннеле почти статичен (измерено: медиана |ΔR| соседних кадров < 50 мм),
поэтому разница кадра с фоном -- это либо настоящая помеха, либо ложное
срабатывание. Оба числа нужны: первое даёт положительные примеры без разметки,
второе -- порог шума.

Два независимых варианта, оба наружу:

  * per-ray: фон = медиана дальности по ячейкам (кольцо, азимут 0.1°). Ячейка
    годится, только если она УСТОЙЧИВА: p90 |R - медиана| < 60 мм. Гейт
    устойчивости обязателен -- на лучах, которые в кадре то есть, то нет
    (края, редкие кольца), медиана мусорная и без гейта остаток лезет из шума.
    Остаток = текущая точка ближе фона на tau = 0.15 м.
  * voxel: фон = множество вокселей 0.25 м, занятых минимум в `min_hits` кадрах.
    Остаток = точки в вокселях, которых в фоне не было. ВАЖНО: этот вариант
    работает только там, где сенсор СТАТИЧЕН. Замерено: медиана |ΔR| соседних
    кадров 8 мм у doubleT_obstacle, 20 мм у roundT_doubleT и 56-164 мм у
    остальных четырёх записей -- там сдвиг геометрии двигает поверхности из
    вокселя в воксель, и остаток выходит 100-4600 точек на пустом кадре.

Число кластеров: точки остатка кластеризуются по вокселям 0.5 м (связность 26),
кластер принимается при >= 5 точках, размахе >= 0.3 м и размахе по ВЫСОТЕ
>= 0.4 м (помеха обязана иметь высоту). Дополнительно остаток ограничивается
габаритом М (|поперёк оси| <= 1.36 м, 0.4..3.7 м над полом) и отсекается
ближняя зона y > -5 м (там собственный монтаж сенсора). Кластеризация -- свой
union-find на numpy (scipy в валидацию не тянем).
"""

import argparse
import glob
import os
import warnings

import numpy as np

from .synth import RING_COUNT, az_bins, fit_floor_ab, range_grid

TAU_M = 0.15              # остаток по дальности: точка ближе фона на tau
GATE_P90_M = 0.06         # гейт устойчивости ячейки фона
GATE_MIN_FRAMES = 3       # ячейка принимается, если наблюдалась в >= этом числе кадров
VOXEL_M = 0.25            # воксель фона
VOXEL_MIN_HITS = 4        # воксель фона: занят минимум в стольких кадрах окна
CLUSTER_VOXEL_M = 0.5     # воксель кластеризации остатка
MIN_CLUSTER_POINTS = 5    # кластер остатка: минимум точек
MIN_CLUSTER_EXTENT_M = 0.3   # и минимум размаха по любой оси
MIN_CLUSTER_EXTENT_Z_M = 0.4  # и минимум размаха по высоте (помеха должна иметь высоту)
CORRIDOR_HALF_M = 1.36    # полуширина коридора (габарит М, от оси пути)
CORRIDOR_HEIGHT_M = (0.4, 3.7)   # высота над полом: ниже -- рельсы/лоток, выше -- потолок


# ---------------------------------------------------------------------------
# Кластеризация (numpy, без scipy)
# ---------------------------------------------------------------------------

def _encode(idx, dims):
    return (idx[:, 0] * dims[1] + idx[:, 1]) * dims[2] + idx[:, 2]


def cluster_labels(points, voxel=CLUSTER_VOXEL_M, connectivity=26):
    """Метка кластера на каждую точку: связные воксели `voxel` (26-связность)."""
    points = np.asarray(points, dtype=np.float64)
    n = points.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int64), 0
    if connectivity == 6:
        offsets = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    else:
        offsets = [(dx, dy, dz)
                   for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
                   if (dx, dy, dz) > (0, 0, 0)]

    lo = points.min(axis=0) - float(voxel)
    idx = np.floor((points - lo) / float(voxel)).astype(np.int64)
    dims = idx.max(axis=0) + 2
    code = _encode(idx, dims)

    order = np.argsort(code, kind='stable')
    codes = code[order]
    uniq, first = np.unique(codes, return_index=True)
    uniq_idx = idx[order[first]]

    parent = np.arange(uniq.size)

    def find(a):
        root = a
        while parent[root] != root:
            root = parent[root]
        while parent[a] != root:
            parent[a], a = root, parent[a]
        return root

    def union(a, b):
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[rb] = ra

    for off in offsets:
        n_idx = uniq_idx + np.asarray(off, dtype=np.int64)
        n_code = _encode(n_idx, dims)
        pos = np.searchsorted(uniq, n_code)
        pos_clipped = np.minimum(pos, uniq.size - 1)
        ok = (pos < uniq.size) & (uniq[pos_clipped] == n_code)
        nbr = pos_clipped[ok]
        for a, b in zip(np.flatnonzero(ok), nbr):
            union(a, b)

    roots = np.array([find(i) for i in range(uniq.size)], dtype=np.int64)
    _, dense = np.unique(roots, return_inverse=True)
    lab_uniq = dense.astype(np.int64)
    lab = np.empty(n, dtype=np.int64)
    lab[order] = lab_uniq[np.searchsorted(uniq, codes)]
    return lab, int(lab_uniq.max() + 1) if lab_uniq.size else 0


def cluster_stats(points, labels, n_labels):
    """По кластеру: точки, число, размах по осям, центр."""
    points = np.asarray(points, dtype=np.float64)
    out = []
    for lab in range(n_labels):
        sel = points[labels == lab]
        if sel.shape[0] == 0:
            continue
        extent = sel.max(axis=0) - sel.min(axis=0)
        out.append({'points': sel, 'n': int(sel.shape[0]), 'extent': extent,
                    'extent_max': float(extent.max()),
                    'extent_z': float(extent[2]),
                    'center': sel.mean(axis=0)})
    return out


def count_clusters(points, voxel=CLUSTER_VOXEL_M, min_points=MIN_CLUSTER_POINTS,
                   min_extent=0.0, min_extent_z=0.0):
    """Кластеры точек: связность по вокселям `voxel`, фильтр по точкам и размаху.

    `min_extent` -- минимум размаха по ЛЮБОЙ оси, `min_extent_z` -- отдельно по
    высоте: помеха обязана иметь высоту над полом (кромка балласта или кабельный
    лоток размах по X/Y дают, а по Z -- нет).
    """
    points = np.asarray(points, dtype=np.float64)
    if points.shape[0] < min_points:
        return []
    labels, n_labels = cluster_labels(points, voxel=voxel)
    return [c for c in cluster_stats(points, labels, n_labels)
            if c['n'] >= min_points and c['extent_max'] >= min_extent
            and c['extent_z'] >= min_extent_z]


# ---------------------------------------------------------------------------
# Фон на луч (per-ray)
# ---------------------------------------------------------------------------

def per_ray_background(frames, tau=TAU_M, gate_p90=GATE_P90_M, gate_min_frames=GATE_MIN_FRAMES):
    """Фон по ячейкам (кольцо, азимут) из окна кадров.

    `frames` -- список (points, ring). Возвращает dict с сетками:
      'median'      медиана дальности ячейки;
      'spread_p90'  p90 |R - медиана| по кадрам окна;
      'observed'    в скольких кадрах ячейка была;
      'stable'      observed >= gate_min_frames и spread_p90 < gate_p90;
      'stable_frac' доля устойчивых ячеек от наблюдавшихся.
    """
    grids = []
    for points, ring in frames:
        grids.append(range_grid(points, ring, ring_count=RING_COUNT))
    stack = np.stack(grids)
    with np.errstate(all='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        median = np.nanmedian(stack, axis=0)
        spread = np.nanpercentile(np.abs(stack - median), 90, axis=0)
        observed = np.count_nonzero(np.isfinite(stack), axis=0)
    stable = (observed >= int(gate_min_frames)) & np.isfinite(median) & (spread < gate_p90)
    observed_any = observed > 0
    return {'median': median.astype(np.float32), 'spread_p90': spread.astype(np.float32),
            'observed': observed, 'stable': stable, 'n_frames': len(grids),
            'tau_m': float(tau), 'gate_p90_m': float(gate_p90),
            'stable_frac': float(stable.sum() / max(int(observed_any.sum()), 1)),
            'observed_frac': float(observed_any.sum() / median.size)}


def per_ray_residual(points, ring, background, tau=None):
    """Точки, которые ближе фона на tau в УСТОЙЧИВЫХ ячейках."""
    points = np.asarray(points, dtype=np.float64)
    if points.shape[0] == 0:
        return np.zeros((0, 3))
    tau = float(background['tau_m'] if tau is None else tau)
    ref = background['median'][ring, az_bins(points)]
    stable = background['stable'][ring, az_bins(points)]
    r = np.linalg.norm(points, axis=1)
    an = stable & np.isfinite(ref) & (r < ref - tau)
    return points[an]


# ---------------------------------------------------------------------------
# Фон по вокселям
# ---------------------------------------------------------------------------

def voxel_keys(points, voxel=VOXEL_M, origin=None):
    """Ключи вокселей (упакованы в int64) и сетка индексов."""
    points = np.asarray(points, dtype=np.float64)
    origin = (np.array([-12.0, -260.0, -10.0]) if origin is None
              else np.asarray(origin, dtype=np.float64))
    idx = np.floor((points - origin) / float(voxel)).astype(np.int64)
    dims = np.array([1024, 4096, 1024], dtype=np.int64)
    idx = np.clip(idx, 0, dims - 1)
    key = (idx[:, 0] * dims[1] + idx[:, 1]) * dims[2] + idx[:, 2]
    return key, idx, origin


def voxel_background(frames, voxel=VOXEL_M, min_hits=VOXEL_MIN_HITS):
    """Фон = воксели, занятые минимум в `min_hits` кадрах окна.

    `frames` -- список (points, ring). Возвращает dict 'keys' (отсортированные
    ключи фона), 'voxel_m', 'min_hits', 'count'.
    """
    counts = {}
    for points, _ring in frames:
        key, _idx, origin = voxel_keys(points, voxel)
        for k in np.unique(key):
            counts[int(k)] = counts.get(int(k), 0) + 1
    keys = np.array(sorted(k for k, v in counts.items() if v >= int(min_hits)),
                    dtype=np.int64)
    return {'keys': keys, 'voxel_m': float(voxel), 'min_hits': int(min_hits),
            'count': int(keys.size), 'n_frames': len(frames)}


def voxel_residual(points, background, voxel=None):
    """Точки в вокселях, которых не было в фоне."""
    points = np.asarray(points, dtype=np.float64)
    if points.shape[0] == 0:
        return np.zeros((0, 3))
    key, _idx, _origin = voxel_keys(points, voxel or background['voxel_m'])
    bg = background['keys']
    pos = np.searchsorted(bg, key)
    pos_clipped = np.minimum(pos, max(bg.size - 1, 0))
    seen = (pos < bg.size) & (bg[pos_clipped] == key)
    return points[~seen]


# ---------------------------------------------------------------------------
# Гейт по коридору пути (поперёк оси + высота над полом)
# ---------------------------------------------------------------------------

def corridor_gate(points, floor_ab, axis_nodes, half_m=CORRIDOR_HALF_M,
                  height_m=CORRIDOR_HEIGHT_M, min_distance_m=5.0):
    """Маска точек внутри габарита: |поперёк оси| <= half_m, высота от пола в height_m.

    Без гейта остаток забит полом, стенками лотка и рельсом: они дают ложные
    кластеры вдоль всего тоннеля (измерено: 13-27 м из 46 м «помех» на кромке
    балласта). Гейт нужен, чтобы считать именно помехи в габарите.

    `min_distance_m` отсекает ближнюю зону (y > -min_distance_m): туда попадает
    собственный монтаж сенсора -- у doubleT_platform он даёт кластер с 47-78
    точками на 2 м впереди, который иначе считается «помехой».
    """
    points = np.asarray(points, dtype=np.float64)
    if points.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    ys, xs = axis_nodes[:, 1], axis_nodes[:, 0]
    x_axis = np.interp(points[:, 1], ys, xs)
    lateral = points[:, 0] - x_axis
    z_rel = points[:, 2] - (floor_ab[0] + floor_ab[1] * points[:, 1])
    return ((np.abs(lateral) <= half_m) & (z_rel >= height_m[0]) & (z_rel <= height_m[1])
            & (points[:, 1] <= -float(min_distance_m)))


# ---------------------------------------------------------------------------
# Прогон: пустые кадры обоими способами
# ---------------------------------------------------------------------------

def empty_frames_report(bag_dir, start=0, bg_frames=15, eval_frames=10, verbose=True,
                        gate=True, near_m=60.0):
    """Фон по `bg_frames` кадрам с `start`, остаток по следующим `eval_frames`.

    Возвращает dict с числами по каждому способу: остаток на кадр (число точек и
    число кластеров) и их устойчивость по кадрам окна.

    `near_m` -- окно ближней зоны: кластеры считаются для остатка с y >= -near_m
    (0 отключает ограничение). Дальше по ходу кадры расходятся по максимальной
    дальности (+-30 м: 175..209 м в одной записи), и воксельный остаток набирает
    «новые» воксели дальнего хвоста тоннеля, которых в фоне просто не было
    (замерено: кластеры y = -130..-207 м). Числа по всему коридору тоже
    возвращаются -- как 'clusters_all_corridor'.
    """
    db = sorted(glob.glob(os.path.join(bag_dir, '*.db3')))[0]
    window = start + bg_frames + eval_frames
    import bag_reader  # noqa: PLC0415

    frames = bag_reader.BagFrames(db, cache_size=min(window + 2, bg_frames + 3),
                                  prefetch_ahead=0)

    def load(i):
        frame = frames[i]
        pts = np.asarray(frame.xyz, dtype=np.float64)
        ring = None if frame.ring is None else np.asarray(frame.ring, dtype=np.int64)
        return pts, ring

    bg_list = [load(i) for i in range(start, start + bg_frames)]
    floor_ab = fit_floor_ab(bg_list[0][0])[:2]
    axis_nodes = _axis_nodes(bg_list[0][0], floor_ab)
    per_ray = per_ray_background(bg_list)
    voxel = voxel_background(bg_list)

    rows = {'per_ray': [], 'voxel': []}
    for i in range(start + bg_frames, window):
        pts, ring = load(i)
        for name, residue in (('per_ray', per_ray_residual(pts, ring, per_ray)),
                              ('voxel', voxel_residual(pts, voxel))):
            if gate:
                residue = residue[corridor_gate(residue, floor_ab, axis_nodes)]
            near = residue[residue[:, 1] >= -float(near_m)] if near_m else residue
            clusters = count_clusters(near, min_extent=MIN_CLUSTER_EXTENT_M,
                                      min_extent_z=MIN_CLUSTER_EXTENT_Z_M)
            boxes = count_clusters(near, min_extent=MIN_CLUSTER_EXTENT_M)
            far_clusters = count_clusters(residue, min_extent=MIN_CLUSTER_EXTENT_M,
                                          min_extent_z=MIN_CLUSTER_EXTENT_Z_M)
            rows[name].append({'frame': i, 'points': int(residue.shape[0]),
                               'points_near': int(near.shape[0]),
                               'clusters': len(clusters), 'clusters_flat': len(boxes),
                               'clusters_all_corridor': len(far_clusters),
                               'cluster_points': [c['n'] for c in clusters],
                               'cluster_extent_z': [round(c['extent_z'], 2) for c in boxes]})
    frames.close()
    report = {'bag': os.path.basename(bag_dir.rstrip('/\\')), 'start': start,
              'bg_frames': bg_frames, 'eval_frames': eval_frames,
              'voxel_bg_cells': voxel['count'], 'voxel_m': voxel['voxel_m'],
              'per_ray_stable_frac': per_ray['stable_frac'],
              'per_ray_observed_frac': per_ray['observed_frac'],
              'gate': bool(gate), 'near_m': float(near_m), 'rows': rows}
    if verbose:
        print(render_report(report))
    return report


def _axis_nodes(points, floor_ab):
    """Ось пути по кадру без track_geometry (тот тянет open3d)."""
    from .synth import estimate_axis_nodes

    nodes = estimate_axis_nodes(points, floor_ab=floor_ab)
    if nodes.shape[0] < 2:
        nodes = np.array([[0.0, -300.0], [0.0, 3.0]], dtype=np.float64)
    return nodes


def render_report(report):
    lines = ['%-38s фон %2d кадров с %d | воксель %.2f м: ячеек фона %d | устойчивых '
             'ячеек луча %.1f %% (наблюдалось %.1f %%) | ближняя зона y >= -%.0f м'
             % (report['bag'], report['bg_frames'], report['start'],
                report['voxel_m'], report['voxel_bg_cells'],
                100.0 * report['per_ray_stable_frac'],
                100.0 * report['per_ray_observed_frac'], report['near_m'])]
    for name in ('per_ray', 'voxel'):
        rows = report['rows'][name]
        pts = np.array([r['points'] for r in rows])
        cls = np.array([r['clusters'] for r in rows])
        flat = np.array([r['clusters_flat'] for r in rows])
        allc = np.array([r['clusters_all_corridor'] for r in rows])
        label = 'на луч (tau 0.15 м)' if name == 'per_ray' else 'воксельный   '
        lines.append('   %s: остаток точек на кадр med %5.0f p90 %5.0f max %5d | '
                     'кластеров (высота >=0.4 м) med %.0f p90 %.0f max %d | '
                     'без правила высоты med %.0f max %d | по всему коридору med %.0f max %d'
                     % (label, np.median(pts), np.percentile(pts, 90), pts.max(),
                        np.median(cls), np.percentile(cls, 90), cls.max(),
                        np.median(flat), flat.max(),
                        np.median(allc), allc.max()))
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Фон из кадров и остаток кадра')
    parser.add_argument('--bags', nargs='*', default=None,
                        help='каталоги записей (по умолчанию все for_hackathon/*)')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--bg-frames', type=int, default=15)
    parser.add_argument('--eval-frames', type=int, default=10)
    parser.add_argument('--no-gate', action='store_true',
                        help='без гейта коридора: считать остаток по всему кадру')
    parser.add_argument('--near-m', type=float, default=60.0,
                        help='ближняя зона (y >= -near_m) для подсчёта кластеров; '
                             '0 -- без ограничения')
    parser.add_argument('--frames-per-bag', type=int, default=1,
                        help='сколько начальных кадров окна проверить на запись')
    args = parser.parse_args(argv)

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    bags = args.bags or sorted(
        d for d in glob.glob(os.path.join(root, 'for_hackathon', '*'))
        if glob.glob(os.path.join(d, '*.db3')))
    for bag in bags:
        for k in range(args.frames_per_bag):
            empty_frames_report(bag, start=args.start + k, bg_frames=args.bg_frames,
                                eval_frames=args.eval_frames, gate=not args.no_gate,
                                near_m=args.near_m)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())