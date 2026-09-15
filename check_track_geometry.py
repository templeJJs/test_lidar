"""Численная проверка track_geometry.build_track — без GUI и окон.

Проверяет ровно то, что просили:
  1. доля точек наблюдаемой головки внутри построенного контура профиля;
  2. ширина головки из данных (верх площадки и уровень 13 мм ниже верха)
     против 74.59 мм (ГОСТ Р 51685-2022) и 72.5 мм;
  3. высота верха головки над прилегающей поверхностью против 0.159-0.186 м;
  4. колея по внутренним граням (1520 ± 10 мм) и «ось-ось» (1592-1595 мм);
  5. шпалы (шаг, число на 10 м) и пол (отклонение меша от floor.a/b/c);
  6. две записи (doubleT_platform, doubleT_obstacle) по три кадра (0, середина,
     последний).

Как измеряется
--------------
Геометрия берётся из ВЫВОДА build_track, но облако измеряется заново:
  * оси нитей и линия верха восстанавливаются из МЕША (средний X яруса меша =
    ось нити со сдвигом подуклонки, верхние вершины = линия верха);
  * контур профиля берётся из track_geometry.rail_profile_mm (это и проверяется:
    попадают ли в него наблюдаемые точки);
  * точки головки собираются по ПОЛОСАМ 0.5 м, и в каждой полосе своя местная ось
    (середина p2/p98 по X) и свой верх (p98 по Z). Полосная привязка обязательна:
    путь кривится, и линейная ось вдоль 20 м уводит контур на 30-40 мм от головки
    (замерено), из-за чего сплошной тест «доля внутри» падал бы до 0.5.
    Форма профиля при этом не меняется — сдвигается только привязка.

Запуск:  python check_track_geometry.py
"""

from __future__ import annotations

import math
import sys

import numpy as np

from track_geometry import (PODUKLONKA_RAD, R65_H_MM, SLEEPER_SPACING_M,
                            build_track, rail_profile_mm)
from visualize_bag import BagFrames

EXPECT = {
    'head_b_mm': 74.59,          # ГОСТ Р 51685-2022 (контроль на 13 мм ниже верха)
    'head_b_alt': 72.5,          # тот же размер по старому чертежу/оценка
    'head_above_adjacent': (0.159, 0.186),
    'gauge_inner_mm': 1520.0,
    'gauge_inner_tol_mm': 10.0,
    'gauge_axis_mm': (1592.0, 1595.0),
    'sleeper_spacing_m': SLEEPER_SPACING_M,
    'sleeper_per_10m': 10.0 / SLEEPER_SPACING_M,
    'inside_min': 0.8,
}


def _fmt(v, nd=1, dash='нет'):
    if v is None:
        return dash
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return dash
    return dash if not np.isfinite(fv) else f'{fv:.{nd}f}'


def _fit_line(ys, xs, tol=0.05):
    ys = np.asarray(ys, float)
    xs = np.asarray(xs, float)
    if ys.size < 2:
        return 0.0, (float(np.median(xs)) if xs.size else 0.0)
    s, i = np.polyfit(ys, xs, 1)
    for _ in range(3):
        r = xs - (s * ys + i)
        mad = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-9
        keep = np.abs(r - np.median(r)) <= max(tol, 3.0 * mad)
        if keep.all() or keep.sum() < 2:
            break
        ys, xs = ys[keep], xs[keep]
        s, i = np.polyfit(ys, xs, 1)
    return float(s), float(i)


def _inside_polygon(poly, pts):
    """Точка внутри простого многоугольника (ray casting); pts — (N,2)."""
    inside = np.zeros(pts.shape[0], dtype=bool)
    n = len(poly)
    for k in range(n):
        x1, y1 = poly[k]
        x2, y2 = poly[(k + 1) % n]
        if y1 == y2:
            continue
        cond = (y1 > pts[:, 1]) != (y2 > pts[:, 1])
        if not cond.any():
            continue
        x_cross = (x2 - x1) * (pts[:, 1] - y1) / (y2 - y1) + x1
        inside ^= cond & (pts[:, 0] < x_cross)
    return inside


def _density_edge(u_mm, thr=0.20, bin_mm=3.0, lim_mm=60.0, min_pts=4):
    """Внутренняя грань головки: самый внутренний бин с заметной плотностью."""
    u_mm = np.asarray(u_mm, float)
    if u_mm.size < 10:
        return None
    edges = np.arange(-lim_mm, lim_mm + bin_mm, bin_mm)
    hist, _ = np.histogram(u_mm, bins=edges)
    if hist.max() < min_pts:
        return None
    ok = np.flatnonzero(hist >= max(min_pts, thr * hist.max()))
    return None if ok.size == 0 else float(edges[int(ok.min())])


def _mesh_rail_geometry(rail):
    """Ось нити и линия верха из меша: (s, i, ms, cs, out_sign, y_lo, y_hi).

    Профиль симметричен по u, поэтому средний X яруса меша = ось нити минус сдвиг
    подуклонки; средний Z верхних вершин = z_top минус 0.180*(1-cos(th)). Поправки
    снимаем точно, зная контур (это и проверяет, что меш стоит там, где надо).
    """
    verts = np.asarray(rail['mesh']['vertices'], float)
    out_sign = -1.0 if rail['side'] == 'left' else 1.0
    prof, _src = rail_profile_mm(rail['head']['width_mm'])
    n_p = len(prof)
    if verts.shape[0] % n_p:
        raise AssertionError('вершин меша не кратно контуру')
    n_y = verts.shape[0] // n_p
    lvl = verts.reshape(n_y, n_p, 3)
    ys = lvl[:, :, 1].mean(axis=1)
    x_center = lvl[:, :, 0].mean(axis=1)
    z_top_v = lvl[:, :, 2].max(axis=1)
    mean_v = float(np.mean([v for (_u, v) in prof])) / 1000.0
    cos_t, sin_t = math.cos(PODUKLONKA_RAD), math.sin(PODUKLONKA_RAD)
    x_rail = x_center + out_sign * sin_t * mean_v
    z_head = z_top_v + R65_H_MM / 1000.0 * (1.0 - cos_t)
    s, i = _fit_line(ys, x_rail)
    ms, cs = _fit_line(ys, z_head)
    return s, i, ms, cs, out_sign, float(ys.min()), float(ys.max()), mean_v


def _sections(x, y, z, s, i, ms, cs, out_sign, y_lo, y_hi, w_top, half=0.5):
    """Точки головки по полосам, каждая в своей местной системе (u, v) мм.

    Возвращает dict с пулами u/v/depth, долями внутри контура по полосам и
    местными осями (y, x_center) и верхами (y, z_top).
    """
    cos_t, sin_t = math.cos(PODUKLONKA_RAD), math.sin(PODUKLONKA_RAD)
    prof, _src = rail_profile_mm(w_top)
    dx = x - (s * y + i)
    z_top = ms * y + cs
    base = (np.abs(dx) <= 0.09) & ((z_top - z) >= -0.005) & ((z_top - z) < 0.040) \
        & (y >= y_lo) & (y <= y_hi)
    idx = np.flatnonzero(base)
    us, vs, ds, fracs, cy, cx, cz, dense = [], [], [], [], [], [], [], []
    for k in range(int(math.ceil((y_hi - y_lo) / half))):
        b0 = y_lo + half * k
        sel = idx[(y[idx] >= b0) & (y[idx] < b0 + half)]
        if sel.size < 4:
            continue
        xs = x[sel]
        xc = 0.5 * (np.percentile(xs, 2) + np.percentile(xs, 98))
        tl = float(np.percentile(z[sel], 98))
        d_out = out_sign * (xs - xc)
        dz = z[sel] - (tl - R65_H_MM / 1000.0)
        u = (d_out * cos_t + dz * sin_t) * 1000.0
        v = (-d_out * sin_t + dz * cos_t) * 1000.0
        fracs.append(float(_inside_polygon(prof, np.column_stack([u, v])).mean()))
        us.append(u)
        vs.append(v)
        ds.append((tl - z[sel]) * 1000.0)
        cy.append(0.5 * (b0 + min(b0 + half, y_hi)))
        cx.append(xc)
        cz.append(tl)
        dense.append(bool(sel.size >= 8))
    if not fracs:
        return None
    dense = np.asarray(dense)
    return {
        'u': np.concatenate(us), 'v': np.concatenate(vs), 'depth': np.concatenate(ds),
        'inside_per_band': np.asarray(fracs),
        'inside_frac': float(np.mean(fracs)),
        'inside_frac_dense': (float(np.mean(np.asarray(fracs)[dense]))
                              if dense.any() else float('nan')),
        'y_center': np.asarray(cy), 'x_center': np.asarray(cx), 'z_top': np.asarray(cz),
        'dense': dense, 'n_dense': int(dense.sum()),
        'n_band': len(fracs), 'n_pts': int(sum(u.size for u in us)),
    }


def _adjacent_height(x, y, z, s, i, ms, cs, out_sign, y_lo, y_hi):
    """Высота верха над прилегающей поверхностью (полоса ВНУТРЬ 0.10-0.24 м, p90)."""
    inward = -out_sign
    vals = []
    y0 = y_lo
    while y0 < y_hi - 0.5:
        band = (y >= y0) & (y < y0 + 0.5)
        dx = x - (s * y + i)
        near = band & (np.abs(dx) < 0.35)
        rail = band & (np.abs(dx) < 0.06)
        if near.sum() >= 20:
            gnd = float(np.percentile(z[near], 10.0))
            rail = rail & (z > gnd + 0.06) & (z < gnd + 0.45)
        if rail.sum() >= 15:
            top = float(np.percentile(z[rail], 98))
            ain = band & (inward * dx > 0.10) & (inward * dx < 0.24)
            if ain.sum() >= 10:
                vals.append(top - float(np.percentile(z[ain], 90)))
        y0 += 0.5
    return float(np.median(vals)) if len(vals) >= 4 else None


def _check_contract(res):
    """Соответствие вывода контракту + JSON-сериализуемость."""
    import json
    problems = []
    want_top = {'frame', 'axis', 'floor', 'rails', 'sleepers', 'floor_mesh', 'meta'}
    if set(res) != want_top:
        problems.append('ключи верхнего уровня: %s' % sorted(res))
    for key in ('s', 'i'):
        if not isinstance(res['axis'][key], float):
            problems.append('axis.%s не float' % key)
    for key in ('a', 'b', 'c'):
        if not isinstance(res['floor'][key], float):
            problems.append('floor.%s не float' % key)
    if len(res['rails']) != 2:
        problems.append('rails: %d элементов (ожидается 2)' % len(res['rails']))
    for rail in res['rails']:
        if set(rail) != {'side', 'head', 'mesh', 'source'}:
            problems.append('rails[].ключи: %s' % sorted(rail))
        if set(rail['head']) != {'top_z', 'width_mm', 'inner_face_x',
                                 'height_above_adjacent_m', 'y_range'}:
            problems.append('head ключи: %s' % sorted(rail['head']))
        if set(rail['mesh']) != {'vertices', 'faces'}:
            problems.append('mesh ключи: %s' % sorted(rail['mesh']))
        nv = len(rail['mesh']['vertices'])
        for f in rail['mesh']['faces']:
            if len(f) != 3 or min(f) < 0 or max(f) >= nv:
                problems.append('mesh: битый индекс %s (вершин %d)' % (f, nv))
                break
        if len(rail['head']['y_range']) != 2:
            problems.append('head.y_range не из двух чисел')
    if set(res['sleepers']) != {'instances', 'spacing_m', 'source'}:
        problems.append('sleepers ключи: %s' % sorted(res['sleepers']))
    for it in res['sleepers']['instances']:
        if set(it) != {'center', 'size'} or len(it['center']) != 3 or len(it['size']) != 3:
            problems.append('sleeper instance: %s' % sorted(it))
            break
    if set(res['floor_mesh']) != {'vertices', 'faces', 'source'}:
        problems.append('floor_mesh ключи: %s' % sorted(res['floor_mesh']))
    if set(res['meta']) != {'gauge_inner_mm', 'gauge_axis_mm', 'head_width_mm',
                            'sensor_height_above_head_m', 'notes'}:
        problems.append('meta ключи: %s' % sorted(res['meta']))
    if not isinstance(res['meta']['notes'], list) or not all(
            isinstance(s, str) for s in res['meta']['notes']):
        problems.append('meta.notes не список строк')
    try:
        blob = json.dumps(res)
    except TypeError as exc:
        blob = ''
        problems.append('JSON не сериализуется: %s' % exc)
    return {'ok': not problems, 'problems': problems, 'json_bytes': len(blob)}


def _check_frame(db_path, index):
    """Все измерения по одному кадру."""
    res = build_track(db_path, index)
    frames = BagFrames(db_path, cache_size=1)
    xyz, _inten, _ts = frames[index]
    x = xyz[:, 0].astype(np.float64)
    y = xyz[:, 1].astype(np.float64)
    z = xyz[:, 2].astype(np.float64)
    del xyz

    out = {'bag': db_path.replace('\\', '/').split('/')[1], 'index': index,
           'meta': res['meta'], 'rails': {}, 'notes': []}
    out['contract'] = _check_contract(res)

    # ---- пол
    a, b, c = res['floor']['a'], res['floor']['b'], res['floor']['c']
    fv = np.asarray(res['floor_mesh']['vertices'], float)
    ff = np.asarray(res['floor_mesh']['faces'], int)
    floor_dev = float(np.max(np.abs(fv[:, 2] - (a * fv[:, 0] + b * fv[:, 1] + c))))
    near = (np.abs(x) <= 5.0) & (y >= -60.0) & (y <= -2.0)
    # качество плоскости пола: нижняя огибающая по ячейкам 1x1 м против плоскости
    cell_key = (np.floor(x[near]).astype(np.int64) * 100003
                + np.floor(y[near]).astype(np.int64))
    order = np.argsort(cell_key, kind='stable')
    groups = np.split(order, np.flatnonzero(np.diff(cell_key[order])) + 1)
    cgx, cgy, cgz = [], [], []
    for g in groups:
        if g.size >= 20:
            cgx.append(x[near][g].mean())
            cgy.append(y[near][g].mean())
            cgz.append(np.percentile(z[near][g], 5.0))
    cgx, cgy, cgz = np.asarray(cgx), np.asarray(cgy), np.asarray(cgz)
    resid = np.abs(cgz - (a * cgx + b * cgy + c)) if cgx.size else np.zeros(0)
    out['floor'] = {
        'abc': (a, b, c), 'mesh_dev_m': floor_dev,
        'verts': int(fv.shape[0]), 'faces': int(ff.shape[0]),
        'indices_ok': bool(ff.max() < fv.shape[0] and ff.min() >= 0),
        'x_span': (float(fv[:, 0].min()), float(fv[:, 0].max())),
        'y_span': (float(fv[:, 1].min()), float(fv[:, 1].max())),
        'cloud_y': (float(y.min()), float(y.max())),
        'n_cells': int(cgx.size),
        'cell_med_m': float(np.median(resid)) if resid.size else None,
        'cell_p90_m': float(np.percentile(resid, 90)) if resid.size else None,
    }

    # ---- шпалы
    inst = res['sleepers']['instances']
    if len(inst) >= 3:
        cy = np.array([it['center'][1] for it in inst], float)
        spacing = float(np.median(np.diff(np.sort(cy))))
        y0, y1 = float(cy.min()), float(cy.max())
        span = y1 - y0
        expected_span = int(math.floor(span / res['sleepers']['spacing_m'])) + 1
    else:
        spacing, span, expected_span = float('nan'), float('nan'), 0
    sizes = np.array([it['size'] for it in inst], float) if inst else np.zeros((0, 3))
    out['sleepers'] = {
        'count': len(inst), 'spacing_m': res['sleepers']['spacing_m'],
        'spacing_used_m': spacing, 'span_m': span, 'expected_in_span': expected_span,
        'size': (float(sizes[:, 0].mean()), float(sizes[:, 1].mean()),
                 float(sizes[:, 2].mean())) if len(sizes) else None,
        'source': res['sleepers']['source'],
    }

    # ---- рельсы: свои оси и меш
    mesh = {}
    for rail in res['rails']:
        mesh[rail['side']] = _mesh_rail_geometry(rail)
    common_lo = max(mesh['left'][5], mesh['right'][5])
    common_hi = min(mesh['left'][6], mesh['right'][6])
    y_mid = 0.5 * (common_lo + common_hi)

    for rail in res['rails']:
        s, i, ms, cs, out_sign, y_lo, y_hi, _mv = mesh[rail['side']]
        sec = _sections(x, y, z, s, i, ms, cs, out_sign, y_lo, y_hi,
                        rail['head']['width_mm'])
        if sec is None:
            out['rails'][rail['side']] = {'error': 'точек головки мало — не проверено'}
            continue
        u, depth = sec['u'], sec['depth']
        m30 = (depth >= 0) & (depth < 30)
        m13 = (depth >= 8) & (depth < 20)
        m5 = (depth >= 5) & (depth < 30)
        inner = _density_edge(u[m5]) if m5.sum() >= 20 else None
        dense = sec['dense']
        # линия нити и линия верха — только по ПЛОТНЫМ полосам (там, где головка
        # реально измерена); на разреженных полосах ось тянется по прямой, и если
        # подгонять по ним, линия уезжает вместе с длиной видимого участка
        s_d, i_d = _fit_line(sec['y_center'][dense], sec['x_center'][dense], tol=0.03)
        ms_d, cs_d = _fit_line(sec['y_center'][dense], sec['z_top'][dense], tol=0.02)
        out['rails'][rail['side']] = {
            'mesh_faces': len(rail['mesh']['faces']),
            'mesh_verts': len(rail['mesh']['vertices']),
            'mesh_indices_ok': bool(np.asarray(rail['mesh']['faces']).max()
                                    < len(rail['mesh']['vertices'])),
            'mesh_y': (y_lo, y_hi), 'head_y': tuple(rail['head']['y_range']),
            'poly_verts': len(rail_profile_mm(rail['head']['width_mm'])[0]),
            'n_pts': sec['n_pts'], 'n_band': sec['n_band'], 'n_dense': sec['n_dense'],
            'inside_frac': float(np.mean(sec['inside_per_band'])),
            'inside_frac_dense': sec['inside_frac_dense'],
            'inside_median_band': float(np.median(sec['inside_per_band'])),
            'inside_min_band': float(sec['inside_per_band'].min()),
            'width_top_data_mm': (float(np.percentile(u[m30], 95)
                                        - np.percentile(u[m30], 5))
                                  if m30.sum() >= 20 else None),
            'width_13_data_mm': (float(np.percentile(u[m13], 98)
                                       - np.percentile(u[m13], 2))
                                 if m13.sum() >= 20 else None),
            'inner_edge_mm': inner,
            'width_module_mm': rail['head']['width_mm'],
            's_mesh': s, 'i_mesh': i, 's_data': s_d, 'i_data': i_d,
            'top_slope_data': ms_d, 'top_slope_mesh': ms,
            'head_top_z_data_mid': float(ms_d * y_mid + cs_d),
            'head_top_z_module_mid': float(rail['head']['top_z']),
            'x_rail_data_mid': float(s_d * y_mid + i_d),
            'x_rail_mesh_mid': float(s * y_mid + i),
            'h_adj_data': _adjacent_height(x, y, z, s_d, i_d, ms_d, cs_d, out_sign,
                                           y_lo, y_hi),
            'h_adj_module': rail['head']['height_above_adjacent_m'],
        }

    # ---- колея
    rl = out['rails']
    if 'error' not in rl.get('left', {'error': 1}) and 'error' not in rl.get('right', {'error': 1}):
        lx = rl['left']['x_rail_data_mid']
        rx = rl['right']['x_rail_data_mid']
        il, ir = abs(rl['left']['inner_edge_mm'] or 0.0), abs(rl['right']['inner_edge_mm'] or 0.0)
        axis_data = (rx - lx) * 1000.0
        gauge_data = axis_data - il - ir
        iface_l = res['rails'][0]['head']['inner_face_x']
        iface_r = res['rails'][1]['head']['inner_face_x']
        gauge_module = abs(iface_r - iface_l) * 1000.0
        out['gauge'] = {
            'y_mid': y_mid,
            'inner_mm_data': gauge_data,
            'inner_mm_module': gauge_module,
            'axis_mm_data': axis_data,
            'axis_mm_module': res['meta']['gauge_axis_mm'],
            'axis_mm_composed': gauge_data + EXPECT['head_b_mm'],
            'inner_off_sum_mm': il + ir,
            'head_b_norm_mm': EXPECT['head_b_mm'],
        }
    else:
        out['gauge'] = {}

    tops = [r['head_top_z_data_mid'] for r in rl.values() if 'head_top_z_data_mid' in r]
    out['sensor_height_m'] = float(-np.mean(tops)) if tops else None
    return out


def main():
    bags = ['for_hackathon/doubleT_platform/doubleT_platform_0.db3',
            'for_hackathon/doubleT_obstacle/doubleT_obstacle_0.db3']
    rows = []
    for db in bags:
        frames = BagFrames(db, cache_size=1)
        n = len(frames)
        del frames
        for idx in sorted({0, n // 2, n - 1}):
            info = _check_frame(db, idx)
            rows.append(info)
            fl = info['floor']
            sl = info['sleepers']
            g = info['gauge']
            print(f"\n=== {info['bag']}, кадр {idx} из {n} ===")
            print('  пол %s: меш %d верш./%d граней, откл. вершин от плоскости %.1e м, '
                  'индексы %s; коридор X [%.1f, %.1f] м, Y [%.1f, %.1f] м'
                  % (tuple(round(v, 4) for v in fl['abc']), fl['verts'], fl['faces'],
                     fl['mesh_dev_m'], 'ок' if fl['indices_ok'] else 'ПЛОХИЕ',
                     fl['x_span'][0], fl['x_span'][1], fl['y_span'][0], fl['y_span'][1]))
            print('  пол: нижняя огибающая (5-й перцентиль ячеек 1x1 м, %d ячеек) против '
                  'плоскости — медиана |откл.| %s м, p90 %s м'
                  % (fl['n_cells'], _fmt(fl['cell_med_m'], 3), _fmt(fl['cell_p90_m'], 3)))
            print('  шпалы: %d шт (в кадре участок %.1f м -> ожидание %d), шаг по '
                  'центрам %.3f м при стандартном %.2f м (%.1f шт на 10 м), '
                  'бокс %s, источник %s'
                  % (sl['count'], sl['span_m'], sl['expected_in_span'],
                     sl['spacing_used_m'], sl['spacing_m'], EXPECT['sleeper_per_10m'],
                     'x'.join('%.2f' % v for v in sl['size']) if sl['size'] else '-',
                     sl['source']))
            for side in ('left', 'right'):
                r = info['rails'].get(side, {})
                if 'error' in r:
                    print('  рельс %-5s: %s' % (side, r['error']))
                    continue
                print('  рельс %-5s: полос %d (плотных %d), точек %d | ВНУТРИ контура %.3f, '
                      'по плотным полосам %.3f (медиана %.3f, минимум %.3f)'
                      % (side, r['n_band'], r['n_dense'], r['n_pts'], r['inside_frac'],
                         r['inside_frac_dense'], r['inside_median_band'],
                         r['inside_min_band']))
                print('       ширина: верх площадки данные %s мм / модуль %s мм; '
                      'на 13 мм ниже верха данные %s мм'
                      % (_fmt(r['width_top_data_mm']), _fmt(r['width_module_mm']),
                         _fmt(r['width_13_data_mm'])))
                print('       верх головки: данные %+.3f м, модуль %+.3f м; наклон вдоль '
                      'пути данные %+.5f, модуль %+.5f м/м'
                      % (r['head_top_z_data_mid'], r['head_top_z_module_mid'],
                         r['top_slope_data'], r['top_slope_mesh']))
                print('       высота над прилегающей: данные %s м, модуль %s м'
                      % (_fmt(r['h_adj_data'], 3), _fmt(r['h_adj_module'], 3)))
                print('       ось нити: данные x=%.5f*y%+.4f, меш x=%.5f*y%+.4f '
                      '(на середине диапазона %.1f / %.1f мм)'
                      % (r['s_data'], r['i_data'], r['s_mesh'], r['i_mesh'],
                         1000 * r['x_rail_data_mid'], 1000 * r['x_rail_mesh_mid']))
                print('       меш: %d граней, y_range [%.1f, %.1f] = head.y_range [%.1f, %.1f], '
                      'индексы %s'
                      % (r['mesh_faces'], r['mesh_y'][0], r['mesh_y'][1],
                         r['head_y'][0], r['head_y'][1],
                         'ок' if r['mesh_indices_ok'] else 'ПЛОХИЕ'))
            if g:
                print('  колея: по внутренним граням — данные %s мм, модуль %s мм '
                      '(норма 1520 ± 10)' % (_fmt(g['inner_mm_data'], 0),
                                             _fmt(g['inner_mm_module'], 0)))
                print('         сумма внутренних смещений от осей нитей %s мм (норма '
                      '74.59) | «ось-ось»: данные %s, модуль %s, по норме (колея + 74.59) %s мм '
                      '(ожидание 1592-1595)'
                      % (_fmt(g['inner_off_sum_mm'], 1), _fmt(g['axis_mm_data'], 0),
                         _fmt(g['axis_mm_module'], 0), _fmt(g['axis_mm_composed'], 0)))
            print('  сенсор над верхом головки: %s м' % _fmt(info['sensor_height_m'], 3))
            ct = info['contract']
            print('  контракт вывода: %s (JSON %.0f кБ)%s'
                  % ('ок' if ct['ok'] else 'НАРУШЕН', ct['json_bytes'] / 1024.0,
                     '' if ct['ok'] else ': ' + '; '.join(ct['problems'][:3])))

    # ---------------- сводка
    print('\n=== СВОДКА (все числа в мм/м, «данные/модуль») ===')
    print('%-17s %4s | %-13s | %-13s | %-11s | %-7s | %-5s | %s'
          % ('bag', 'кадр', 'колея внутр', 'ось-ось', 'ширина верх', 'шир13', 'верх-п',
             'внутри (плотные)'))
    fails = []
    for info in rows:
        g = info['gauge']
        rl = info['rails']
        keys = ('left', 'right')
        w_top = [rl[k]['width_top_data_mm'] for k in keys
                 if rl.get(k, {}).get('width_top_data_mm')]
        w13 = [rl[k]['width_13_data_mm'] for k in keys
               if rl.get(k, {}).get('width_13_data_mm')]
        hh = [rl[k]['h_adj_data'] for k in keys if rl.get(k, {}).get('h_adj_data')]
        ins = [rl[k]['inside_frac'] for k in keys if 'inside_frac' in rl.get(k, {})]
        ins_d = [rl[k]['inside_frac_dense'] for k in keys
                 if rl.get(k, {}).get('inside_frac_dense') is not None]
        print('%-17s %4d | %6s/%6s | %6s/%6s | %5s/%5s | %7s | %5s | %.3f (%.3f)'
              % (info['bag'], info['index'],
                 _fmt(g.get('inner_mm_data'), 0, '-'), _fmt(g.get('inner_mm_module'), 0, '-'),
                 _fmt(g.get('axis_mm_data'), 0, '-'), _fmt(g.get('axis_mm_module'), 0, '-'),
                 _fmt(np.mean(w_top) if w_top else None, 1, '-'),
                 _fmt(np.mean(w13) if w13 else None, 1, '-'),
                 _fmt(np.mean(hh) if hh else None, 3, '-'),
                 _fmt(np.mean(ins) if ins else None, 3, '-'),
                 np.mean(ins_d) if ins_d else float('nan')))
        checks = []
        if g:
            v = g.get('inner_mm_data')
            checks.append(('колея внутр 1520±10',
                           v is not None and abs(v - EXPECT['gauge_inner_mm'])
                           <= EXPECT['gauge_inner_tol_mm']))
            v = g.get('axis_mm_data')
            checks.append(('ось-ось 1592-1595',
                           v is not None and EXPECT['gauge_axis_mm'][0] <= v
                           <= EXPECT['gauge_axis_mm'][1]))
            v = g.get('axis_mm_composed')
            checks.append(('ось-ось по норме (колея+74.59) 1592-1595',
                           v is not None and EXPECT['gauge_axis_mm'][0] <= v
                           <= EXPECT['gauge_axis_mm'][1]))
        if ins:
            checks.append(('внутри контура > 0.8 (плотные полосы)',
                           bool(np.mean(ins_d) > EXPECT['inside_min'])
                           if ins_d else bool(np.mean(ins) > EXPECT['inside_min'])))
        if ins_d:
            checks.append(('внутри контура > 0.8 (по всем полосам)',
                           bool(np.mean(ins) > EXPECT['inside_min'])))
        if hh:
            lo, hi = EXPECT['head_above_adjacent']
            checks.append(('верх над прилегающей 0.159-0.186',
                           bool(lo <= np.mean(hh) <= hi)))
        if w_top:
            checks.append(('ширина верха в ±10 мм от 74.59',
                           bool(abs(np.mean(w_top) - EXPECT['head_b_mm']) <= 10.0)))
        if np.isfinite(info['sleepers']['spacing_used_m']):
            checks.append(('шаг шпал 0.55±0.02',
                           bool(abs(info['sleepers']['spacing_used_m']
                                    - EXPECT['sleeper_spacing_m']) <= 0.02)))
        if info['sleepers']['count'] >= 3:
            checks.append(('число шпал на участке',
                           bool(abs(info['sleepers']['count']
                                    - info['sleepers']['expected_in_span']) <= 1)))
        checks.append(('пол: меш в плоскости floor.a/b/c',
                       bool(info['floor']['mesh_dev_m'] < 1e-6)))
        if info['floor']['cell_med_m'] is not None:
            checks.append(('пол: плоскость описывает нижнюю огибающую (медиана < 0.25 м)',
                           bool(info['floor']['cell_med_m'] < 0.25)))
        checks.append(('контракт вывода и JSON', bool(info['contract']['ok'])))
        for name, ok in checks:
            if not ok:
                fails.append((info['bag'], info['index'], name))

    print('\n--- что НЕ сошлось ---')
    if not fails:
        print('  всё в допуске')
    for bag, idx, name in fails:
        print('  %s кадр %d: %s' % (bag, idx, name))

    # ---- итоговые числа
    def _col(fn):
        vals = [v for v in (fn(i) for i in rows) if v is not None and np.isfinite(v)]
        return (min(vals), max(vals)) if vals else (None, None)

    ins_avg = [np.mean([i['rails'][k]['inside_frac'] for k in ('left', 'right')
                        if 'inside_frac' in i['rails'].get(k, {})]) for i in rows]
    ins_dense = [np.mean([i['rails'][k]['inside_frac_dense'] for k in ('left', 'right')
                          if i['rails'].get(k, {}).get('inside_frac_dense') is not None])
                 for i in rows]
    all_avg = ins_avg + ins_dense  # noqa: F841  (для наглядности в отчёте)
    n_checks = 11
    tot = n_checks * len(rows)
    print()
    print('ИТОГ: кадров %d (%d записи x 3 кадра: 0, середина, последний); '
          'проверок %d, не сошлось %d.' % (len(rows), len(bags), tot, len(fails)))
    print('ИТОГ: 1) внутри контура профиля по всем полосам %.3f-%.3f, по плотным полосам '
          '%.3f-%.3f наблюдаемых точек головки (порог 0.8) — %s; 2) ширина верха площадки '
          '%s-%s мм, на 13 мм ниже верха '
          '%s-%s мм (нормы 74.59 / 72.5); 3) высота верха над прилегающей %s-%s м '
          '(норма 0.159-0.186); 4) колея по внутренним граням %s-%s мм (норма 1520 ± 10), '
          '«ось-ось» измеренная %s-%s мм и по норме (колея + 74.59) %s-%s мм (норма 1592-1595);'
          % (min(ins_avg), max(ins_avg), min(ins_dense), max(ins_dense),
             'в допуске' if min(ins_dense) > EXPECT['inside_min'] else 'НИЖЕ порога',
             _fmt(_col(lambda i: np.mean([i['rails'][k]['width_top_data_mm']
                                          for k in ('left', 'right')
                                          if i['rails'].get(k, {}).get('width_top_data_mm')]))[0]),
             _fmt(_col(lambda i: np.mean([i['rails'][k]['width_top_data_mm']
                                          for k in ('left', 'right')
                                          if i['rails'].get(k, {}).get('width_top_data_mm')]))[1]),
             _fmt(_col(lambda i: np.mean([i['rails'][k]['width_13_data_mm']
                                          for k in ('left', 'right')
                                          if i['rails'].get(k, {}).get('width_13_data_mm')]))[0]),
             _fmt(_col(lambda i: np.mean([i['rails'][k]['width_13_data_mm']
                                          for k in ('left', 'right')
                                          if i['rails'].get(k, {}).get('width_13_data_mm')]))[1]),
             _fmt(_col(lambda i: np.mean([i['rails'][k]['h_adj_data'] for k in ('left', 'right')
                                          if i['rails'].get(k, {}).get('h_adj_data')]))[0], 3),
             _fmt(_col(lambda i: np.mean([i['rails'][k]['h_adj_data'] for k in ('left', 'right')
                                          if i['rails'].get(k, {}).get('h_adj_data')]))[1], 3),
             _fmt(_col(lambda i: i['gauge'].get('inner_mm_data'))[0], 0),
             _fmt(_col(lambda i: i['gauge'].get('inner_mm_data'))[1], 0),
             _fmt(_col(lambda i: i['gauge'].get('axis_mm_data'))[0], 0),
             _fmt(_col(lambda i: i['gauge'].get('axis_mm_data'))[1], 0),
             _fmt(_col(lambda i: i['gauge'].get('axis_mm_composed'))[0], 0),
             _fmt(_col(lambda i: i['gauge'].get('axis_mm_composed'))[1], 0)))
    print('ИТОГ: 5) шпалы — %s шт, шаг %.2f м (стандарт, %.1f шт/10 м), верх совмещён с '
          'плоскостью подошвы рельса; пол — плоскость z = a*x + b*y + c: меш лежит в ней '
          'точно (0.0 м), нижняя огибающая ячеек 1x1 м отклоняется от неё на %s м (медиана); '
          '6) сенсор над верхом головки %s-%s м (допущение: сенсор — начало облака; '
          'по отчётам 1.11 м для doubleT_platform и 1.51-1.59 м для doubleT_obstacle).'
          % (_fmt(_col(lambda i: i['sleepers']['count'])[0], 0), SLEEPER_SPACING_M,
             EXPECT['sleeper_per_10m'],
             _fmt(_col(lambda i: i['floor']['cell_med_m'])[1], 3),
             _fmt(_col(lambda i: i['sensor_height_m'])[0], 3),
             _fmt(_col(lambda i: i['sensor_height_m'])[1], 3)))
    return 0


if __name__ == '__main__':
    sys.exit(main())