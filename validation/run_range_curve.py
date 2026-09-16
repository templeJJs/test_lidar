"""Протокол «дистанция -> точки объекта / доля обнаружений».

180 испытаний на дистанцию: 6 записей x 10 кадров x 3 поперечных положения
(лево/центр/право колеи, +-0.7175 м и 0). Дистанции 25..300 м шагом 25 м.
В каждый кадр вносится объект 0.3 x 0.3 x 1 м (`synth.insert_object`) на ось
пути, и считается:

  * ЧИСЛО ТОЧЕК ОБЪЕКТА -- сколько точек дал объект (равно числу удалённых точек
    фона: объект загораживает луч, и точка фона на нём исчезает -- сохраняется
    общее число точек кадра);
  * РАЗЛИЧИМ ЛИ ОБЪЕКТ -- простой габаритный критерий: кластер >= 5 точек в окне
    +-1.5 м от истинного центра объекта с размахом по высоте >= 0.3 м. Это НЕ
    детектор (его пишет другой агент), а нижняя оценка «сколько нужно точек,
    чтобы объект вообще было видно».

Какие кадры берутся: только ПУСТЫЕ (без настоящих помех в габарите). Выбор
автоматический: по первым 12 кадрам записи строится фон на луч (приём из
`stitch_bg`), затем кадр считается пустым, если остаток в габарите не даёт ни
одного кластера; берётся первое окно из 10 подряд пустых кадров. Результат
кэшируется в `validation/out/empty_windows.json`, чтобы прогон был воспроизводим
без повторного сканирования.

Каждая постановка ПРОВЕРЯЕТСЯ по самому кадру (`synth.placement_check`): сколько
измеренных точек попало внутрь бокса и пуста ли его верхняя половина. За краем
мешей нитей ось продолжается прямой, и на кривых продолжение может уехать в
стену; колонка `placement_free` отделяет «объект не виден, потому что далеко»
от «объект случайно поставлен в стену». По всем 2160 испытаниям постановка
свободна, кроме 4 % на 25-50 м: низ бокса пересекается с лотком, который лежит
ровно на оси пути (лоток 0.4 x 0.36 м).

Выход: `validation/out/range_curve.csv` (все 2160 испытаний),
`validation/out/range_curve_summary.csv` и `range_curve.txt` (таблица с числами).
"""

import argparse
import csv
import glob
import json
import os
import time
import warnings

import numpy as np

from . import stitch_bg as bg
from .synth import (GAUGE_HALF_M, OBJECT_SIZE_DEFAULT, axis_from_track,
                    build_rings_meta, insert_object, object_point_counts,
                    placement_check)

DISTANCES = tuple(range(25, 301, 25))
LATERALS = (-GAUGE_HALF_M, 0.0, GAUGE_HALF_M)
LATERAL_NAMES = ('лево', 'центр', 'право')
FRAMES_PER_BAG = 10
REFERENCE_FRAMES = 12          # окно, по которому строится фон при выборе пустых кадров
WINDOW_M = 1.5                 # окно критерия различимости вокруг центра объекта
MIN_CLUSTER_POINTS = 5
MIN_EXTENT_Z_M = 0.3
RING_BAG_NAMES = ('doubleT_obstacle', 'doubleT_platform', 'roundT_doubleT',
                  'roundT_pressureGate_roundT', 'roundT_squareT_pressureGate_squareT',
                  'squareT_platform_squareT_switch')


def project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def bag_db_paths(root=None, names=RING_BAG_NAMES):
    root = project_root() if root is None else root
    out = []
    for name in names:
        found = sorted(glob.glob(os.path.join(root, 'for_hackathon', name, '*.db3')))
        if found:
            out.append((name, found[0]))
    return out


# ---------------------------------------------------------------------------
# Выбор пустых кадров
# ---------------------------------------------------------------------------

def empty_window(db_path, want=FRAMES_PER_BAG, reference=REFERENCE_FRAMES,
                 scan_limit=None, verbose=False):
    """Первое окно из `want` подряд пустых кадров и статистика по нему.

    Пустой = остаток относительно фона на луч (по первым `reference` кадрам) не
    даёт ни одного кластера в габарите. Возвращает dict со 'start', 'n_frames',
    'stable_frac' фона, 'scanned' и 'empty_frames' (сколько кадров проверено).
    """
    import bag_reader  # noqa: PLC0415

    frames = bag_reader.BagFrames(db_path, cache_size=reference + 4, prefetch_ahead=0)
    n_frames = len(frames)

    def load(i):
        frame = frames[i]
        pts = np.asarray(frame.xyz, dtype=np.float64)
        ring = None if frame.ring is None else np.asarray(frame.ring, dtype=np.int64)
        return pts, ring

    ref = [load(i) for i in range(min(reference, n_frames))]
    background = bg.per_ray_background(ref)
    floor_ab = bg.fit_floor_ab(ref[0][0])[:2]
    axis_nodes = bg._axis_nodes(ref[0][0], floor_ab)

    limit = n_frames if scan_limit is None else min(n_frames, int(scan_limit))
    empty = []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for i in range(limit):
            pts, ring = load(i)
            residue = bg.per_ray_residual(pts, ring, background)
            if residue.shape[0]:
                residue = residue[bg.corridor_gate(residue, floor_ab, axis_nodes)]
            clusters = bg.count_clusters(residue, min_extent=0.3, min_extent_z=0.4)
            empty.append(len(clusters) == 0)
    frames.close()

    start = None
    run = 0
    for i, ok in enumerate(empty):
        run = run + 1 if ok else 0
        if run >= want:
            start = i - want + 1
            break
    if start is None:
        start = 0
        if verbose:
            print('   пустого окна из %d кадров не нашлось: взяты кадры с 0' % want)
    return {'start': int(start), 'n_frames': int(want), 'stable_frac': background['stable_frac'],
            'observed_frac': background['observed_frac'], 'scanned': int(limit),
            'empty_frames': int(sum(empty)), 'n_bag_frames': int(n_frames)}


def empty_windows(bags, cache_path, want=FRAMES_PER_BAG, scan_limit=None, verbose=True):
    cache = {}
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, encoding='utf-8') as fh:
            cache = json.load(fh)
    out = {}
    for name, db in bags:
        key = '%s|%d' % (name, want)
        if key in cache and cache[key].get('n_bag_frames'):
            out[name] = cache[key]
            continue
        info = empty_window(db, want=want, scan_limit=scan_limit, verbose=verbose)
        out[name] = info
        cache[key] = info
        if verbose:
            print('   %-40s окно кадров %3d..%3d | фон: устойчивых ячеек луча %.1f %%, '
                  'пустых кадров %d из %d'
                  % (name, info['start'], info['start'] + want - 1,
                     100.0 * info['stable_frac'], info['empty_frames'], info['scanned']))
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'w', encoding='utf-8') as fh:
            json.dump(cache, fh, ensure_ascii=False, indent=2)
    return out


# ---------------------------------------------------------------------------
# Прогон
# ---------------------------------------------------------------------------

def run_bag(name, db_path, start, distances=DISTANCES, frames_n=FRAMES_PER_BAG,
            laterals=LATERALS, size_m=OBJECT_SIZE_DEFAULT, seed_base=0,
            verbose=True):
    """Все испытания одной записи: кадры окна x дистанции x поперечные положения."""
    import bag_reader  # noqa: PLC0415

    frames = bag_reader.BagFrames(db_path, cache_size=frames_n + 4, prefetch_ahead=0)

    def load(index):
        frame = frames[index]
        pts = np.asarray(frame.xyz, dtype=np.float64)
        ring = None if frame.ring is None else np.asarray(frame.ring, dtype=np.int64)
        intensity = (None if frame.intensity is None
                     else np.asarray(frame.intensity, dtype=np.float64))
        return pts, ring, intensity

    try:
        # Ось пути берётся по ОДНОМУ кадру записи и переиспользуется на все её
        # кадры: сенсор и путь в записи не меняются, а build_track стоит ~1 с.
        try:
            nodes, axis_meta = axis_from_track(db_path, start)
        except Exception as exc:  # noqa: BLE001
            nodes, axis_meta = None, {'axis_source':
                                      'оценка по кадру (%s)' % type(exc).__name__}
    except Exception:  # noqa: BLE001
        frames.close()
        raise

    rows = []
    try:
        for k in range(frames_n):
            index = start + k
            pts, ring, intensity = load(index)
            meta = build_rings_meta(pts, ring, intensity, axis_nodes=nodes)
            meta['axis_source'] = axis_meta['axis_source']
            for D in distances:
                for lat, lat_name in zip(laterals, LATERAL_NAMES):
                    seed = seed_base + k * 1000 + int(D)
                    points_new, mask, ins = insert_object(
                        pts, meta, D, lat, size_m=size_m, seed=seed)
                    counts = object_point_counts(
                        points_new, ins, cluster_fn=bg.count_clusters,
                        window_m=WINDOW_M, min_points=MIN_CLUSTER_POINTS,
                        min_vertical_extent_m=MIN_EXTENT_Z_M)
                    place = placement_check(pts, meta, D, lat, size_m=size_m)
                    rows.append({
                        'bag': name, 'frame': int(index), 'distance_m': float(D),
                        'lateral_m': float(lat), 'lateral': lat_name,
                        'size_m': 'x'.join('%.2f' % v for v in size_m),
                        'points_object': counts['points_object'],
                        'rays_hit': int(ins['rays_hit']),
                        'points_removed': int(ins['points_removed']),
                        'rays_candidate': int(ins['rays_candidate']),
                        'clusters': int(counts['clusters']),
                        'extent_z_m': round(float(counts['vertical_extent_m']), 3),
                        'visible': bool(counts['visible']),
                        'floor_z_m': round(float(ins['floor_z']), 3),
                        'center_x_m': round(float(ins['center'][0]), 3),
                        'axis_extrapolated': bool(ins['axis_extrapolated']),
                        'points_inside_box': int(place['points_inside']),
                        'points_inside_box_top': int(place['points_inside_top']),
                        'nearest_surface_m': round(place['nearest_m'], 3),
                        'placement_free': bool(place['free']),
                        'bg_points': int(ins['n_points_in']),
                    })
            if verbose:
                print('   %-40s кадр %3d готов (%.1f с)'
                      % (name, index, time.time() - _T0), flush=True)
    finally:
        frames.close()
    return rows


_T0 = time.time()


def summarize(rows, distances=DISTANCES):
    """Таблица «дистанция -> медиана точек, доля обнаружений» (и по положениям)."""
    lat_keys = ('left', 'center', 'right')
    summary = []
    for D in distances:
        sel = [r for r in rows if r['distance_m'] == D]
        pts = np.array([r['points_object'] for r in sel], dtype=np.float64)
        vis = np.array([1.0 if r['visible'] else 0.0 for r in sel])
        entry = {
            'distance_m': float(D), 'trials': len(sel),
            'points_median': float(np.median(pts)),
            'points_p10': float(np.percentile(pts, 10)),
            'points_p90': float(np.percentile(pts, 90)),
            'points_max': float(pts.max()) if pts.size else 0.0,
            'points_zero_frac': float((pts == 0).mean()) if pts.size else 0.0,
            'detection_frac': float(vis.mean()) if vis.size else 0.0,
        }
        free = [r for r in sel if r['placement_free']]
        entry['trials_free'] = len(free)
        entry['placement_free_frac'] = float(len(free) / len(sel)) if sel else 0.0
        if free:
            fv = np.array([1.0 if r['visible'] else 0.0 for r in free])
            fp = np.array([r['points_object'] for r in free], dtype=np.float64)
            entry['detection_frac_free'] = float(fv.mean())
            entry['points_median_free'] = float(np.median(fp))
        else:
            entry['detection_frac_free'] = 0.0
            entry['points_median_free'] = 0.0
        for lat_name, lat_key in zip(LATERAL_NAMES, lat_keys):
            sub = [r for r in sel if r['lateral'] == lat_name]
            sp = np.array([r['points_object'] for r in sub], dtype=np.float64)
            sv = np.array([1.0 if r['visible'] else 0.0 for r in sub])
            entry['points_median_%s' % lat_key] = float(np.median(sp))
            entry['detection_frac_%s' % lat_key] = float(sv.mean())
        summary.append(entry)
    return summary


def render_table(summary, rows, window_info=None):
    lines = []
    lines.append('объект %s м, окно критерия +-%.1f м, кластер >= %d точек, '
                 'размах по высоте >= %.1f м'
                 % ('x'.join('%.2f' % v for v in OBJECT_SIZE_DEFAULT), WINDOW_M,
                    MIN_CLUSTER_POINTS, MIN_EXTENT_Z_M))
    lines.append('')
    lines.append(' дист | испыт | точек объекта median p10..p90 (max) | доля обнаруж. | '
                 'median по лево/центр/право | доля по лево/центр/право | '
                 'постановка свободна: доля, доля обнаруж.')
    lines.append('-' * 150)
    for e in summary:
        bar = '#' * int(round(20 * e['detection_frac']))
        lines.append('%6.0f | %5d | %7.1f %5.1f..%-5.1f (%4.0f) | %5.1f %% %-20s | '
                     '%5.1f %5.1f %5.1f | %4.0f%% %4.0f%% %4.0f%% | %4.0f%% %4.0f%%'
                     % (e['distance_m'], e['trials'], e['points_median'],
                        e['points_p10'], e['points_p90'], e['points_max'],
                        100 * e['detection_frac'], bar,
                        e['points_median_left'], e['points_median_center'],
                        e['points_median_right'],
                        100 * e['detection_frac_left'], 100 * e['detection_frac_center'],
                        100 * e['detection_frac_right'],
                        100 * e['placement_free_frac'], 100 * e['detection_frac_free']))
    zero = [e for e in summary if e['points_median'] == 0]
    if zero:
        lines.append('')
        lines.append('медиана точек объекта равна нулю с %.0f м (дальше объект за '
                     'измеренной поверхностью -- тоннель его закрывает)'
                     % zero[0]['distance_m'])
    if window_info:
        lines.append('')
        lines.append('кадры (пустые, выбраны автоматически по фону на луч):')
        for name, info in window_info.items():
            lines.append('  %-40s кадры %3d..%3d | устойчивых ячеек фона %.1f %% | '
                         'пустых %d из %d проверенных'
                         % (name, info['start'], info['start'] + info['n_frames'] - 1,
                            100.0 * info['stable_frac'], info['empty_frames'],
                            info['scanned']))
    return '\n'.join(lines)


def write_outputs(rows, summary, text, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    raw = os.path.join(out_dir, 'range_curve.csv')
    with open(raw, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summ = os.path.join(out_dir, 'range_curve_summary.csv')
    with open(summ, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    table = os.path.join(out_dir, 'range_curve.txt')
    with open(table, 'w', encoding='utf-8') as fh:
        fh.write(text + '\n')
    return raw, summ, table


def main(argv=None):
    parser = argparse.ArgumentParser(description='Кривая «дистанция -> обнаружение»')
    parser.add_argument('--bags', nargs='*', default=None)
    parser.add_argument('--frames', type=int, default=FRAMES_PER_BAG)
    parser.add_argument('--scan-limit', type=int, default=None,
                        help='сколько кадров проверять при выборе пустого окна')
    parser.add_argument('--distances', type=float, nargs='*', default=None)
    parser.add_argument('--out-dir', default=None)
    parser.add_argument('--no-cache', action='store_true')
    args = parser.parse_args(argv)

    root = project_root()
    out_dir = args.out_dir or os.path.join(root, 'validation', 'out')
    cache_path = None if args.no_cache else os.path.join(out_dir, 'empty_windows.json')
    bags = bag_db_paths(root, args.bags or RING_BAG_NAMES)
    distances = tuple(args.distances) if args.distances else DISTANCES

    global _T0
    _T0 = time.time()
    print('выбор пустых кадров:')
    windows = empty_windows(bags, cache_path, want=args.frames, scan_limit=args.scan_limit)
    print('прогон:')

    rows = []
    for name, db in bags:
        start = windows[name]['start']
        rows.extend(run_bag(name, db, start, distances=distances, frames_n=args.frames))
    summary = summarize(rows, distances=distances)
    text = render_table(summary, rows, windows)
    paths = write_outputs(rows, summary, text, out_dir)
    print()
    print(text)
    print()
    print('испытаний %d; файлы: %s' % (len(rows), ', '.join(os.path.basename(p) for p in paths)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())