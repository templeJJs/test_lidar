"""Самопроверка: воспроизведение уже измеренных чисел сенсора и метода внесения.

Ничего нового не измеряет -- печатает «измерено / известно» по списку фактов,
которые уже были получены исследованием, чтобы было видно, что модуль стоит на
них, а не на догадках. Два блока:

  1. ФАКТЫ СЕНСОРА (Hesai 128, кадры for_hackathon): кольца и их elevation,
     плотное вертикальное ядро, шаг и фаза азимута по кольцам, дальность, шум
     дальности между кадрами, выпадение лучей, интенсивность против дальности,
     кратность возвратов на луч.
  2. ЧИСЛА ОБЪЕКТА: сколько точек даёт объект 0.3 x 0.3 x 1 м на пустых кадрах
     на 25/50/75/100/200 м (ориентир: ~264 / 60 / 22-28 / 3-16 / 0).

Запуск:  python -m validation.selfcheck [--frame-bags ...] [--no-object]
"""

import argparse
import glob
import os
import time
import warnings

import numpy as np

from . import synth

KNOWN = {
    'rings': 128,
    'elev_range': (14.40, -25.12),
    'dense_core_rings': 64,
    'dense_core_band': (-6.07, 1.97),
    'dense_core_step': (0.086, 0.2),
    'az_step_center': 0.1,
    'az_step_periphery': 0.2,
    'az_phase_std_steps': 0.3,
    'range_m': (143.0, 209.0),
    'noise_median_mm': 8.0,
    'noise_p90_mm': (16.0, 24.0),
    'dropout_frac': (0.005, 0.006),
    'intensity_slope': -0.20,
    'intensity_median_near': 11.0,
    'intensity_median_100m': 6.0,
    'intensity_zero_frac': 0.029,
}

FACT_BAGS = ('doubleT_obstacle', 'doubleT_platform', 'roundT_doubleT',
             'roundT_pressureGate_roundT', 'roundT_squareT_pressureGate_squareT',
             'squareT_platform_squareT_switch')

# Кадры для блока 2 -- те же пустые кадры, что и в исследовании.
OBJECT_PICKS = (('doubleT_obstacle', 0), ('roundT_doubleT', 75),
                ('roundT_squareT_pressureGate_squareT', 270),
                ('squareT_platform_squareT_switch', 0))
OBJECT_KNOWN = {25: 264, 50: 60, 75: 25, 100: (3, 16), 200: 0}


def db_paths(root, names=FACT_BAGS):
    out = []
    for name in names:
        found = sorted(glob.glob(os.path.join(root, 'for_hackathon', name, '*.db3')))
        if found:
            out.append((name, found[0]))
    return out


def _cell_min_ranges(points, ring, az_round=20):
    """Ячейки (кольцо, азимут 0.05°) -> ближайшая дальность: ключ -> R."""
    az = synth.az_deg(points)
    key = ring * (360 * az_round) + np.round(az * az_round).astype(np.int64)
    r = np.linalg.norm(points, axis=1)
    order = np.argsort(r)
    key = key[order]
    r = r[order]
    uniq, first = np.unique(key, return_index=True)
    return uniq, r[first]


def _az_cells_per_6deg(points, ring, elev_deg, el_lo, el_hi, az_span=3.0):
    """Ячеек азимута на 6° (|az| < az_span) в полосе elevation -- плотность сетки.

    Шаг азимута, посчитанный как медиана внутрикольцевых разностей, даёт 0.1°,
    но это не значит, что лучи стоят через 0.1°: в полосе elevation -1.4..-2.3°
    (а это ровно та полоса, куда уходят лучи к объекту дальше 75 м) кольца
    наблюдаются с пропусками 0.2-0.9°, средний шаг выходит 0.15-0.19°.
    """
    az = synth.az_deg(points)
    in_band = (elev_deg >= el_lo) & (elev_deg <= el_hi)
    rings = np.flatnonzero(np.isfinite(elev_deg) & in_band)
    counts = []
    for k in rings:
        m = (ring == k) & (np.abs(az) < az_span)
        cells = np.unique(synth.az_bins(points[m]))
        if cells.size > 3:
            counts.append(cells.size / (2 * az_span))
    return float(np.median(counts)) if counts else float('nan')


def measure_sensor_facts(bags, frame_index=0, verbose=True):
    """Факты сенсора по кадрам: по одному кадру на запись + пары кадров для шума."""
    import bag_reader  # noqa: PLC0415

    facts = {'per_bag': []}
    noise_med, noise_p90, dropouts = [], [], []
    slope_samples, i_near, i_far, zeros = [], [], [], []
    multiplicity = []
    for name, db in bags:
        frames = bag_reader.BagFrames(db, cache_size=4, prefetch_ahead=0)
        try:
            i0 = frame_index if frame_index < len(frames) - 1 else 0
            pts, ring, intensity = synth.load_frame(db, i0)
            grid = synth.ring_grid(pts, ring)
            el = grid['elev_deg']
            step = grid['az_step_deg']
            phase = grid['az_phase']
            r = np.linalg.norm(pts, axis=1)
            delta_el = np.abs(np.diff(el[np.isfinite(el)]))
            core = delta_el < 0.2
            core_rings = [el[i] for i in range(len(delta_el)) if core[i]]
            center_rings = np.flatnonzero(np.isfinite(step) & (step < 0.15))
            periph_rings = np.flatnonzero(np.isfinite(step) & (step >= 0.15))
            entry = {
                'bag': name, 'frame': i0,
                'rings': int(np.count_nonzero(np.isfinite(el))),
                'elev_min': float(np.nanmin(el)), 'elev_max': float(np.nanmax(el)),
                'dense_core_rings': int(np.count_nonzero(core)),
                'dense_core_band': (float(np.min(core_rings)) if core_rings else float('nan'),
                                    float(np.max(core_rings)) if core_rings else float('nan')),
                'dense_core_step': (float(np.min(delta_el[core])) if np.any(core) else float('nan'),
                                    float(np.max(delta_el[core])) if np.any(core) else float('nan')),
                'az_step_center': float(np.median(step[center_rings])) if center_rings.size else float('nan'),
                'az_step_periphery': float(np.median(step[periph_rings])) if periph_rings.size else float('nan'),
                'rings_az_step_center': int(center_rings.size),
                'rings_az_step_periphery': int(periph_rings.size),
                'az_phase_std_steps': float(np.nanstd(phase[center_rings])) if center_rings.size else float('nan'),
                'range_min': float(r.min()), 'range_max': float(r.max()),
                'range_median': float(np.median(r)),
                'periphery_step_max': float(np.nanmax(step)),
                'periphery_step_p90_max': float(max(
                    (np.percentile(np.diff(np.sort(synth.az_deg(pts[ring == k]))),
                                   90) for k in range(128)
                     if np.count_nonzero(ring == k) > 20), default=float('nan'))),
                'az_cells_dense_band': _az_cells_per_6deg(pts, ring, el, -4.4, -3.2),
                'az_cells_mid_band': _az_cells_per_6deg(pts, ring, el, -2.3, -1.4),
            }
            facts['per_bag'].append(entry)

            n_rays = np.unique(synth.ray_keys(pts, ring)).size
            multiplicity.append(len(pts) / max(n_rays, 1))

            # Шум дальности и выпадение лучей: пара кадров одной записи.
            j0 = min(i0 + 1, len(frames) - 1)
            f2 = frames[j0]
            pts2 = np.asarray(f2.xyz, dtype=np.float64)
            ring2 = np.asarray(f2.ring, dtype=np.int64)
            k1, r1 = _cell_min_ranges(pts, ring)
            k2, r2 = _cell_min_ranges(pts2, ring2)
            common, i_a, i_b = np.intersect1d(k1, k2, return_indices=True)
            dr = np.abs(r1[i_a] - r2[i_b])
            entry['noise_median_mm'] = 1000 * float(np.median(dr))
            entry['noise_p90_mm'] = 1000 * float(np.percentile(dr, 90))
            entry['dropout_frac'] = 1.0 - common.size / max(k1.size, 1)
            entry['returns_per_ray'] = multiplicity[-1]
            noise_med.append(entry['noise_median_mm'])
            noise_p90.append(entry['noise_p90_mm'])
            dropouts.append(entry['dropout_frac'])

            if intensity is not None:
                m = intensity > 0
                rs, i_int = r[m], intensity[m]
                edges = np.arange(5, 205, 10)
                rm, im = [], []
                for lo in edges[:-1]:
                    q = (rs >= lo) & (rs < lo + 10)
                    if int(np.count_nonzero(q)) < 200:
                        continue
                    rm.append(np.sqrt(lo * (lo + 10)))
                    im.append(float(np.median(i_int[q])))
                if len(rm) > 3:
                    entry['intensity_slope'] = float(np.polyfit(np.log(rm), np.log(im), 1)[0])
                    slope_samples.append(entry['intensity_slope'])
                band = (rs >= 5) & (rs < 25)
                if np.any(band):
                    entry['intensity_median_near'] = float(np.median(i_int[band]))
                    i_near.append(entry['intensity_median_near'])
                band = (rs >= 90) & (rs < 110)
                if np.any(band):
                    entry['intensity_median_100m'] = float(np.median(i_int[band]))
                    i_far.append(entry['intensity_median_100m'])
                entry['intensity_zero_frac'] = float(np.mean(intensity <= 0))
                zeros.append(entry['intensity_zero_frac'])
        finally:
            frames.close()
    facts['noise_median_mm'] = float(np.median(noise_med)) if noise_med else float('nan')
    facts['noise_p90_mm'] = float(np.median(noise_p90)) if noise_p90 else float('nan')
    facts['dropout_frac'] = float(np.median(dropouts)) if dropouts else float('nan')
    facts['intensity_slope'] = float(np.median(slope_samples)) if slope_samples else float('nan')
    facts['intensity_median_near'] = float(np.median(i_near)) if i_near else float('nan')
    facts['intensity_median_100m'] = float(np.median(i_far)) if i_far else float('nan')
    facts['intensity_zero_frac'] = float(np.median(zeros)) if zeros else float('nan')
    facts['returns_per_ray'] = float(np.median(multiplicity)) if multiplicity else float('nan')
    facts['reference_bag'] = (facts['per_bag'][0]['bag'] if facts['per_bag'] else '')
    facts['range_span'] = (float(min(e['range_min'] for e in facts['per_bag'])),
                           float(max(e['range_max'] for e in facts['per_bag'])))
    facts['range_max_span'] = (float(min(e['range_max'] for e in facts['per_bag'])),
                               float(max(e['range_max'] for e in facts['per_bag'])))
    return facts


def render_facts(facts):
    per_bag = facts['per_bag']
    ref = per_bag[0] if per_bag else {}
    lines = ['ФАКТЫ СЕНСОРА (по одной записи на кадр)']
    lines.append('%-38s %5s %-16s %7s %-16s %9s %9s %s'
                 % ('запись', 'колец', 'elev min..max', 'ядро', 'ядро elev',
                    'Δel min', 'Δel max', 'Φ колец 0.1/0.2°'))
    for e in per_bag:
        lines.append('%-38s %5d %6.2f..%-7.2f %7d %6.2f..%-7.2f %9.3f %9.3f %d/%d'
                     % (e['bag'], e['rings'], e['elev_min'], e['elev_max'],
                        e['dense_core_rings'], e['dense_core_band'][0],
                        e['dense_core_band'][1], e['dense_core_step'][0],
                        e['dense_core_step'][1],
                        e['rings_az_step_center'], e['rings_az_step_periphery']))
    lines.append('')
    lines.append('%-34s %-24s %-22s %s' % ('величина', 'эталон: %s' % ref.get('bag', '-'),
                                           'по всем записям', 'известно'))
    zeros = [e for e in per_bag if 'intensity_zero_frac' in e]
    rows = [
        ('колец в кадре', '%.0f' % ref.get('rings', float('nan')), '-',
         format(KNOWN['rings'])),
        ('elev min/max, °', '%.2f / %.2f' % (ref.get('elev_max', float('nan')),
                                             ref.get('elev_min', float('nan'))), '-',
         '%.2f / %.2f' % KNOWN['elev_range']),
        ('плотное ядро, колец', '%.0f' % ref.get('dense_core_rings', float('nan')), '-',
         format(KNOWN['dense_core_rings'])),
        ('ядро: Δel min/max, °', '%.3f..%.3f' % ref.get('dense_core_step', (float('nan'),) * 2),
         '-', '%.3f..%.2f' % KNOWN['dense_core_step']),
        ('ядро: полоса elev, °', '%.2f..%.2f' % ref.get('dense_core_band', (float('nan'),) * 2),
         '-', '%.2f..%.2f' % KNOWN['dense_core_band']),
        ('Δφ центральных колец, °', '%.2f' % ref.get('az_step_center', float('nan')), '-',
         '%.2f' % KNOWN['az_step_center']),
        ('Δφ периферии, °', '%.2f' % ref.get('az_step_periphery', float('nan')), '-',
         '%.2f' % KNOWN['az_step_periphery']),
        ('периферия: макс Δφ, °', '%.2f' % ref.get('periphery_step_max', float('nan')),
         'p90 шага до %.2f' % ref.get('periphery_step_p90_max', float('nan')),
         'до 0.89'),
        ('фаза азимута: std, шагов', '%.2f' % ref.get('az_phase_std_steps', float('nan')), '-',
         '%.2f' % KNOWN['az_phase_std_steps']),
        ('дальность: максимум по кадрам, м', '%.0f' % ref.get('range_max', float('nan')),
         '%.0f..%.0f' % facts['range_max_span'],
         '%.0f..%.0f' % KNOWN['range_m']),
        ('шум дальности: медиана, мм', '%.1f' % ref.get('noise_median_mm', float('nan')),
         '%.0f..%.0f' % (min(e['noise_median_mm'] for e in per_bag),
                         max(e['noise_median_mm'] for e in per_bag)),
         '%.0f' % KNOWN['noise_median_mm']),
        ('шум дальности: p90, мм', '%.1f' % ref.get('noise_p90_mm', float('nan')),
         '%.0f..%.0f' % (min(e['noise_p90_mm'] for e in per_bag),
                         max(e['noise_p90_mm'] for e in per_bag)),
         '%.0f..%.0f' % KNOWN['noise_p90_mm']),
        ('выпадение лучей, %', '%.2f' % (100 * ref.get('dropout_frac', float('nan'))),
         '%.2f..%.2f' % (100 * min(e['dropout_frac'] for e in per_bag),
                         100 * max(e['dropout_frac'] for e in per_bag)),
         '%.1f..%.1f' % (100 * KNOWN['dropout_frac'][0], 100 * KNOWN['dropout_frac'][1])),
        ('I ∝ R^p', '%.2f' % ref.get('intensity_slope', float('nan')),
         '%.2f..%.2f' % (min((e['intensity_slope'] for e in zeros), default=float('nan')),
                         max((e['intensity_slope'] for e in zeros), default=float('nan'))),
         '%.2f' % KNOWN['intensity_slope']),
        ('I медиана 5..25 м', '%.1f' % ref.get('intensity_median_near', float('nan')),
         '%.1f..%.1f' % (min((e['intensity_median_near'] for e in zeros), default=float('nan')),
                         max((e['intensity_median_near'] for e in zeros), default=float('nan'))),
         '%.0f' % KNOWN['intensity_median_near']),
        ('I медиана ~100 м', '%.1f' % ref.get('intensity_median_100m', float('nan')),
         '%.1f..%.1f' % (min((e['intensity_median_100m'] for e in zeros), default=float('nan')),
                         max((e['intensity_median_100m'] for e in zeros), default=float('nan'))),
         '%.0f' % KNOWN['intensity_median_100m']),
        ('доля I = 0', '%.2f %%' % (100 * ref.get('intensity_zero_frac', float('nan'))),
         '%.2f..%.2f %%' % (100 * min((e['intensity_zero_frac'] for e in zeros), default=float('nan')),
                            100 * max((e['intensity_zero_frac'] for e in zeros), default=float('nan'))),
         '%.1f %%' % (100 * KNOWN['intensity_zero_frac'])),
        ('возвратов на луч', '%.2f' % ref.get('returns_per_ray', float('nan')),
         '%.2f..%.2f' % (min(e['returns_per_ray'] for e in per_bag),
                         max(e['returns_per_ray'] for e in per_bag)),
         'не измерялось ранее'),
        ('ячеек азимута на 1°, полоса el -3.2..-4.4',
         '%.1f' % ref.get('az_cells_dense_band', float('nan')),
         '%.1f..%.1f' % (min((e['az_cells_dense_band'] for e in per_bag), default=float('nan')),
                         max((e['az_cells_dense_band'] for e in per_bag), default=float('nan'))),
         '≈10 (Δφ = 0.1°)'),
        ('ячеек азимута на 1°, полоса el -1.4..-2.3',
         '%.1f' % ref.get('az_cells_mid_band', float('nan')),
         '%.1f..%.1f' % (min((e['az_cells_mid_band'] for e in per_bag), default=float('nan')),
                         max((e['az_cells_mid_band'] for e in per_bag), default=float('nan'))),
         'не измерялось: шаг азимута 0.1°'),
    ]
    for name, got, all_bags, want in rows:
        lines.append('%-34s %-24s %-22s %s' % (name, got, all_bags, want))
    lines.append('')
    lines.append('Шум дальности по записям (пара соседних кадров): медиана ΔR %.0f..%.0f мм, '
                 'выпадение лучей %.2f..%.2f %%.' % (
                     min(e['noise_median_mm'] for e in per_bag),
                     max(e['noise_median_mm'] for e in per_bag),
                     100 * min(e['dropout_frac'] for e in per_bag),
                     100 * max(e['dropout_frac'] for e in per_bag)))
    fast = [e['bag'] for e in per_bag if e['noise_median_mm'] < 30]
    lines.append('Медиана ΔR < 30 мм только у: %s. В остальных записях сенсор между '
                 'кадрами сдвигается (100-170 мм), поэтому фон на луч там невалиден.'
                 % (', '.join(fast) if fast else 'нет'))
    return '\n'.join(lines)


def measure_object_counts(root, picks=OBJECT_PICKS, distances=(25, 50, 75, 100, 200),
                          lateral_step=0.2, lateral_span=1.4, verbose=True):
    """Точки объекта на пустых кадрах: по колеям и максимум по поперечному сносу."""
    out = []
    lat_grid = np.arange(-lateral_span, lateral_span + 1e-9, lateral_step)
    for name, frame_index in picks:
        db = sorted(glob.glob(os.path.join(root, 'for_hackathon', name, '*.db3')))[0]
        pts, ring, intensity = synth.load_frame(db, frame_index)
        meta = synth.build_rings_meta(pts, ring, intensity, db_path=db,
                                      frame_index=frame_index)
        row = {'bag': name, 'frame': frame_index, 'counts': {}, 'sweep': {}}
        for D in distances:
            counts = []
            for lat in synth.GAUGE_HALF_M * np.array([-1.0, 0.0, 1.0]):
                ins = synth.insert_object(pts, meta, D, float(lat), seed=1)[2]
                counts.append(int(ins['points_inserted']))
            sweep = []
            for lat in lat_grid:
                ins = synth.insert_object(pts, meta, D, float(lat), seed=1)[2]
                sweep.append(int(ins['points_inserted']))
            row['counts'][D] = counts
            row['sweep'][D] = (int(np.max(sweep)), float(lat_grid[int(np.argmax(sweep))]))
        out.append(row)
        if verbose:
            print('   %-40s кадр %3d: %s' % (
                name, frame_index,
                ' '.join('D%3d L/C/R %s' % (D, row['counts'][D]) for D in distances)),
                flush=True)
    return out


def render_object_counts(rows, distances=(25, 50, 75, 100, 200)):
    lines = ['ЧИСЛА ОБЪЕКТА 0.3 x 0.3 x 1.0 м (пустые кадры, медиана по 4 записям)']
    lines.append('%8s | %-24s | %-24s | %s'
                 % ('дист, м', 'точек лево/центр/право', 'макс по сносу ±1.4 м',
                    'ориентир из исследования'))
    for D in distances:
        left = [r['counts'][D][0] for r in rows]
        center = [r['counts'][D][1] for r in rows]
        right = [r['counts'][D][2] for r in rows]
        smax = [r['sweep'][D][0] for r in rows]
        lines.append('%8d | %4.0f %4.0f %4.0f (med %4.0f) | %4.0f %4.0f %4.0f (med %4.0f) | %s'
                     % (D, np.median(left), np.median(center), np.median(right),
                        np.median([left, center, right]),
                        min(smax), np.median(smax), max(smax), np.median(smax),
                        format_known(D)))
    return '\n'.join(lines)


def format_known(D):
    known = OBJECT_KNOWN.get(D)
    if isinstance(known, tuple):
        return '%d..%d' % known
    return '%d' % known


def main(argv=None):
    parser = argparse.ArgumentParser(description='Самопроверка валидации')
    parser.add_argument('--no-object', action='store_true')
    parser.add_argument('--frame-index', type=int, default=0)
    args = parser.parse_args(argv)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        t0 = time.time()
        facts = measure_sensor_facts(db_paths(root), frame_index=args.frame_index)
        print(render_facts(facts))
        print('\n(%.1f с на факты сенсора)' % (time.time() - t0))
        if not args.no_object:
            print('\nЧИСЛА ОБЪЕКТА:')
            rows = measure_object_counts(root)
            print()
            print(render_object_counts(rows))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())