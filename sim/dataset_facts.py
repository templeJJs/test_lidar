"""Сводка фактов по всем нашим записям -- основа для генератора сцен.

Зачем: генератор не должен выдумывать параметры туннеля. Всё, что он подставляет
(ширина и высота сечения, форма свода, уклон пола, размах поворотов, число
азимутальных колонок, разброс интенсивности), берётся из реальных записей и лежит
в `sim/dataset_facts.json`.

Ширину туннеля и нити стен меряем НАШИМ ЖЕ детектором (`zones.detect_walls`) и
нашим фитом пола (`zones.fit_floor`) -- он проверен на этих данных. Самодельные
кластеры гистограммы давали мусор: «стена» находилась на x = -0.31 (это путь, а
не стена), а дрейф стен выходил 10 м вместо реального метра.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

import bag_reader
import zones
from sim.sensor import azimuth_span_deg


def _params(a: float, b: float) -> zones.ZoneParams:
    return zones.ZoneParams(floor_a=float(a), floor_b=float(b), floor_band=0.30)


def walls_of(cloud: np.ndarray, a: float, b: float) -> list:
    """Нити стен через наш детектор: две доминирующие вертикальные полосы."""
    if cloud.shape[0] < 200:
        return []
    w = zones.detect_walls(cloud, _params(a, b))
    return [round(float(v), 3) for v in sorted(w)]


SECTION_X_BIN_M = 0.25               # ячейка по X в профиле сечения
SECTION_WALL_MARGIN_M = 0.50         # профиль берём немного шире нитей стен
SECTION_MIN_POINTS = 30              # меньше точек в ячейке -- профиль не построить


def section_profile(x: np.ndarray, z_rel: np.ndarray, y: np.ndarray,
                    walls: list, y_window: tuple = (-14.0, -2.0)) -> list:
    """Форма сечения: верхняя и нижняя огибающие по X в ближней зоне.

    Зачем: свод у нас прямоугольный (плоский потолок на 4.1 м), а у записей
    сечение бывает арочным, с платформой или с соседней конструкцией за стеной.
    Огибающие -- то, что генератор потом обязан повторить, и то, по чему это
    можно проверить.

    Полоса по X берётся вокруг стен самой записи (±0.5 м), поэтому дальние
    конструкции в профиль не попадают случайно: видно ровно сечение. Огибающие --
    перцентили (5 и 95), а не минимум и максимум: одиночный выброс не должен
    рисовать «свод».
    """
    if not walls or len(walls) < 2:
        return []
    sel = (y >= y_window[0]) & (y <= y_window[1])
    x, z = x[sel], z_rel[sel]
    if x.size < SECTION_MIN_POINTS:
        return []
    lo_x = float(min(walls)) - SECTION_WALL_MARGIN_M
    hi_x = float(max(walls)) + SECTION_WALL_MARGIN_M
    edges = np.arange(lo_x, hi_x + SECTION_X_BIN_M, SECTION_X_BIN_M)
    out = []
    for i in range(edges.size - 1):
        m = (x >= edges[i]) & (x < edges[i + 1])
        if int(np.count_nonzero(m)) < SECTION_MIN_POINTS:
            continue
        out.append({'x': round(float(0.5 * (edges[i] + edges[i + 1])), 2),
                    'top': round(float(np.percentile(z[m], 95)), 2),
                    'low': round(float(np.percentile(z[m], 5)), 2),
                    'n': int(np.count_nonzero(m))})
    return out


def measure_bag(bag_dir: str, frames: int = 5, bin_len: float = 10.0) -> dict:
    """Замерить одну запись: сенсор, сцена и дрейф геометрии по длине."""
    db = bag_reader.find_db3(bag_dir)
    fr = bag_reader.BagFrames(db, cache_size=3, prefetch_ahead=0)
    try:
        idxs = np.linspace(0, len(fr) - 1, min(frames, len(fr))).astype(int)
        parts, ring, inten = [], None, []
        for i in idxs:
            f = fr[i]
            if f.ring is not None and ring is None:
                ring = f.ring
            parts.append(f.xyz[:, :3].astype(np.float64))
            if f.intensity is not None:
                inten.append(f.intensity)
        timestamps = fr.timestamps
        total = len(fr)
    finally:
        fr.close()

    c = np.concatenate(parts)
    x, y, z = c[:, 0], c[:, 1], c[:, 2]
    per_frame = len(c) / len(parts)
    columns = int(round(per_frame / (int(ring.max()) + 1))) if ring is not None else None

    # пол: наклонная плоскость. Полосу берём вокруг сенсора -- иначе основания стен
    # тянут нижнюю огибающую вниз (замерено: 0.10 против 0.03 в полосе)
    a, b, rms = zones.fit_floor(c, y_min=-40.0, y_max=-2.0, x_center=0.0, x_half=2.0)
    z_rel = z - (a + b * y)

    walls = walls_of(c, a, b)
    out = {
        'bag': os.path.basename(os.path.normpath(bag_dir)),
        'frames': total,
        'duration_s': round(float((timestamps[-1] - timestamps[0]) / 1e9), 2),
        'points_per_frame': int(per_frame),
        'azimuth_columns': columns,
        # поле зрения прибора: у широкой записи 247°, у узкой 101° -- хвостовой
        # сектор не снимается вовсе, поэтому сцена обязана иметь то же поле
        'azimuth_span_deg': azimuth_span_deg(c),
        'floor': {'a': round(float(a), 4), 'b': round(float(b), 6),
                  'grade_pct': round(float(b) * 100.0, 3), 'rms': round(float(rms), 4)},
        'walls_x': walls,
        'width_m': round(abs(walls[-1] - walls[0]), 3) if len(walls) >= 2 else None,
        # форма сечения (свод/ложе) в ближней зоне: ровно то, что генератор обязан
        # повторить; прежнее поле rail_bands было мусором (локальные пики
        # гистограммы по X) и убрано
        'section': section_profile(x, z_rel, y, walls),
        'y_visible_m': [round(float(np.percentile(y, 5)), 1),
                        round(float(np.percentile(y, 95)), 1)],
    }
    if inten:
        ii = np.concatenate(inten)
        out['intensity'] = {'median': float(np.median(ii)),
                            'p90': float(np.percentile(ii, 90)),
                            'share_gt25': round(float(np.mean(ii > 25)), 4)}

    # --- дрейф геометрии по длине: из него делаются случайные повороты
    per_bin = []
    for y0 in np.arange(np.percentile(y, 2), np.percentile(y, 98), bin_len):
        m = (y >= y0) & (y < y0 + bin_len)
        if int(np.count_nonzero(m)) < 500:
            continue
        w = walls_of(c[m], a, b)
        if len(w) >= 2:
            per_bin.append((float(y0 + bin_len / 2), w[0], w[-1]))

    centres = []
    if per_bin:
        left_ref, right_ref = per_bin[0][1], per_bin[0][2]
        for yc, left, right in per_bin:
            if (abs(left - left_ref) <= 1.5 and abs(right - right_ref) <= 1.5
                    and 2.5 <= right - left <= 14.0):
                left_ref, right_ref = left, right
                centres.append({'y': round(yc, 1), 'left': left, 'right': right,
                                'width': round(right - left, 3)})
    if len(centres) >= 3:
        ys = np.array([c_['y'] for c_ in centres])
        lefts = np.array([c_['left'] for c_ in centres])
        rights = np.array([c_['right'] for c_ in centres])
        widths = np.array([c_['width'] for c_ in centres])
        yc = ys - ys.mean()
        out['along_length'] = {
            'bins': centres,
            'width_m': [round(float(widths.min()), 3), round(float(widths.max()), 3)],
            'left_drift_m': round(float(lefts.max() - lefts.min()), 3),
            'right_drift_m': round(float(rights.max() - rights.min()), 3),
            'left_slope': round(float(np.polyfit(yc, lefts, 1)[0]), 4),
            'right_slope': round(float(np.polyfit(yc, rights, 1)[0]), 4),
            'centre_drift_m': round(float((0.5 * (lefts + rights)).max()
                                          - (0.5 * (lefts + rights)).min()), 3),
        }
    elif centres:
        out['along_length'] = {'bins': centres, 'bins_accepted': len(centres),
                               'width_m': [min(c_['width'] for c_ in centres),
                                           max(c_['width'] for c_ in centres)]}
    return out


def _span(values: list) -> list | None:
    """Диапазон по списку, отбрасывая None (у части записей метрика не снялась)."""
    vals = [v for v in values if v is not None]
    return [min(vals), max(vals)] if vals else None


def measure_all(root: str = 'for_hackathon', frames: int = 5) -> dict:
    bags = sorted(os.path.normpath(p) for p in glob.glob(os.path.join(root, '*')))
    facts = {}
    for bag_dir in bags:
        if not os.path.isdir(bag_dir) or not glob.glob(os.path.join(bag_dir, '*.db3')):
            continue
        try:
            facts[os.path.basename(bag_dir)] = measure_bag(bag_dir, frames)
        except Exception as exc:                       # noqa: BLE001
            facts[os.path.basename(bag_dir)] = {'error': str(exc)}

    ok = [f for f in facts.values() if 'floor' in f]
    al = [f['along_length'] for f in ok if 'along_length' in f]
    ranges = {
        'bags_measured': len(ok),
        'tunnel_width_m': _span([f.get('width_m') for f in ok]),
        'grade_pct': _span([f['floor']['grade_pct'] for f in ok]),
        'sensor_height_m': _span([round(-f['floor']['a'], 3) for f in ok]),
        'points_per_frame': _span([f['points_per_frame'] for f in ok]),
        'azimuth_columns': sorted({f['azimuth_columns'] for f in ok
                                   if f.get('azimuth_columns')}),
        'azimuth_span_deg': _span([f.get('azimuth_span_deg') for f in ok]),
        'intensity_share_gt25': _span([f['intensity']['share_gt25'] for f in ok
                                       if 'intensity' in f]),
        'walls_x': sorted({w for f in ok for w in f.get('walls_x', [])}),
        'visible_y_m': _span([v for f in ok for v in f.get('y_visible_m', [])]),
        # высота сечения: у записей свод бывает на 2.8…4.9 м над ложем, а у
        # генератора он пока плоский на 4.1 -- это и есть ближайшая задача
        'section_top_m': _span([max(b['top'] for b in f['section'])
                                for f in ok if f.get('section')]),
        'section_low_m': _span([min(b['low'] for b in f['section'])
                                for f in ok if f.get('section')]),
        'centre_drift_m': _span([a_['centre_drift_m'] for a_ in al
                                 if 'centre_drift_m' in a_]),
        'width_spread_m': _span([round(a_['width_m'][1] - a_['width_m'][0], 3) for a_ in al]),
    }
    return {'bags': facts, 'generator_ranges': ranges}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='факты по всем записям для генератора сцен')
    ap.add_argument('--root', default='for_hackathon')
    ap.add_argument('--frames', type=int, default=5)
    ap.add_argument('--out', default='sim/dataset_facts.json')
    args = ap.parse_args(argv)

    data = measure_all(args.root, args.frames)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as fh:
        json.dump(data, fh, ensure_ascii=False, indent=1)

    rng = data['generator_ranges']
    print(f'записей замерено: {rng.get("bags_measured")} -> {args.out}')
    for key in ('tunnel_width_m', 'grade_pct', 'sensor_height_m', 'points_per_frame',
                'azimuth_columns', 'azimuth_span_deg', 'intensity_share_gt25',
                'walls_x', 'visible_y_m', 'section_top_m', 'section_low_m',
                'centre_drift_m', 'width_spread_m'):
        print(f'  {key:22s} {rng.get(key)}')
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())