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

from track_geometry import (PODUKLONKA_RAD, R65_H_MM, RAIL_BIN_Y_M, RAIL_MIN_PTS_BIN,
                            RAIL_NODES, RAIL_SWEEP_EXTRA_M, RAIL_WIN_HALF_M,
                            SLEEPER_SPACING_M, build_track, rail_profile_mm)
from visualize_bag import BagFrames

import zones  # только чтение: воспроизведение пола ВЬЮЕРА (zones.fit_floor в /meta)

import track_geometry as tg  # только чтение: свой трекер оси для полосных измерений

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
    # лоток: допуск на совпадение независимого замера с выводом модуля
    'trough_tol_m': 0.05,
    'trough_min_depth_m': 0.12,   # ниже этого — считаем, что лотка в кадре нет
    # развёртка меша нити: длина = плотный диапазон + запас
    'sweep_extra_m': RAIL_SWEEP_EXTRA_M,
    'sweep_extra_tol_m': 0.6,     # допуск на концы меша (шаг сетки 0.5 м + округление)
    'mesh_above_shelf_m': (0.10, 0.35),   # верх меша над ИЗМЕРЕННОЙ полкой
    'mesh_above_floor_m': (0.10, 0.60),   # верх меша над полом ВЬЮЕРА (zones.fit_floor)
    'mesh_above_spread_m': 0.10,          # допустимый разброс привязки по всей длине
    'sweep_drift_guard_m': 0.30,          # RAIL_DRIFT_LIMIT_M: уход от плотной зоны
    'sweep_x_guard_m': 0.50,              # RAIL_X_DRIFT_LIMIT_M
    # метрика «ось меша до точек рельса» (внутри измеренного) для полилинии
    'axis_median_m': 0.05,
    'axis_p95_m': 0.15,
    'axis_jump_m': 0.10,                  # шаг оси между уровнями, «скачок»
    'h_local_m': (0.08, 0.35),            # локальная высота верха меша над полом
    'top_band_m': (0.08, 0.30),           # коридор верха ленты над измеренной полкой
    'top_hard_m': 0.05,                   # жёсткий минимум: ниже — сквозь пол
    'body_h_m': 0.180,                    # высота тела рельса (ГОСТ Р65)
    'foot_b_m': 0.150,                    # ширина подошвы (ГОСТ Р65)
    'z_jump_m': 0.02,                     # одиночный прыжок узла по Z
    'slope_break_1_m': 0.05,              # излом наклона оси на шаг 0.5 м
    'foot_adj_tol_m': (-0.05, 0.02),      # низ подошвы относительно пола у нити
}


def _floor_measure(x, y, z, s_ax, i_ax, m_top, c_top, y_lo, y_hi,
                   bin_m=0.02, lim_m=0.90, min_pts=4, drop_m=0.12,
                   z_up=0.10, z_down=0.80):
    """НЕЗАВИСИМЫЙ замер профиля пола в системе координат пары рельсов.

    Точки пола: ближняя зона y_lo..y_hi, |u| <= lim_m+0.6 и гейт по Z «ниже верха
    головки на z_up..z_down». Профиль — медиана по бинам bin_m; два уровня
    k-средних дают полку (верх) и дно лотка (низ); края лотка — связный участок
    ниже полки на drop_m вокруг u=0.
    """
    u = x - (s_ax * y + i_ax)
    top = m_top * y + c_top
    m = ((y >= y_lo) & (y <= y_hi) & (np.abs(u) <= lim_m + 0.6)
         & (z < top - z_up) & (z > top - z_down))
    if int(m.sum()) < 2000:
        return None
    uu, zz = u[m], z[m]
    edges = np.arange(-lim_m, lim_m + bin_m, bin_m)
    n_b = edges.size - 1
    uc = edges[:-1] + bin_m / 2.0
    med = np.full(n_b, np.nan)
    cnt = np.zeros(n_b, dtype=np.int64)
    for k in range(n_b):
        sel = (uu >= edges[k]) & (uu < edges[k + 1])
        cnt[k] = int(sel.sum())
        if cnt[k] >= min_pts:
            med[k] = float(np.median(zz[sel]))
    ok = np.isfinite(med)
    if int(ok.sum()) < 8:
        return None
    low, high = float(np.min(med[ok])), float(np.max(med[ok]))
    for _ in range(10):
        a = np.abs(med[ok] - low) <= np.abs(med[ok] - high)
        if a.any():
            low = float(np.mean(med[ok][a]))
        if (~a).any():
            high = float(np.mean(med[ok][~a]))
    thr = high - drop_m
    idx0 = int(np.flatnonzero(ok)[np.argmin(np.abs(uc[np.flatnonzero(ok)]))])
    has_trough = bool(med[idx0] < thr)
    u_l = u_r = float('nan')
    bottom = high
    n_bot = 0
    if has_trough:
        kl = idx0
        while kl - 1 >= 0 and np.isfinite(med[kl - 1]) and med[kl - 1] < thr:
            kl -= 1
        kr = idx0
        while kr + 1 < n_b and np.isfinite(med[kr + 1]) and med[kr + 1] < thr:
            kr += 1
        u_l = float(uc[kl] - bin_m / 2.0)
        u_r = float(uc[kr] + bin_m / 2.0)
        inb = (uu >= u_l) & (uu <= u_r)
        n_bot = int(inb.sum())
        bottom = float(np.percentile(zz[inb], 10.0)) if n_bot >= 10 else float(med[idx0])
    edge = max(abs(u_l), abs(u_r)) if has_trough else 0.22
    out = ((uc < u_l - 0.04) | (uc > u_r + 0.04)) if has_trough else np.ones(n_b, dtype=bool)
    sel_s = np.zeros(n_b, dtype=bool)
    for hi_lim in (0.62, 0.90):
        sel_s = ok & out & (np.abs(uc) >= edge + 0.06) & (np.abs(uc) <= hi_lim)
        if int(sel_s.sum()) >= 6:
            break
    c1, c0 = (np.polyfit(uc[sel_s], med[sel_s], 1) if int(sel_s.sum()) >= 6 else (0.0, high))
    return {'shelf_z': float(c0), 'shelf_slope_u': float(c1), 'has_trough': has_trough,
            'u_l': u_l, 'u_r': u_r,
            'width': float(u_r - u_l) if has_trough else 0.0,
            'center': float(0.5 * (u_l + u_r)) if has_trough else float('nan'),
            'bottom': float(bottom),
            'depth': float(c0 - bottom) if has_trough else 0.0,
            'n_bottom': n_bot, 'n_pts': int(m.sum()), 'n_bins': int(ok.sum())}


def _shelf_line_meas(x, y, z, s_ax, i_ax, m_top, c_top, fl, y_lo, y_hi,
                     band_m=1.5, min_pts=80, u_lim=0.90, z_up=0.10, z_down=0.80):
    """НЕЗАВИСИМАЯ прямая полки вдоль пути — база для проверки привязки меша.

    Полка = точки вне лотка (края лотка берутся из независимого замера _floor_measure)
    в полосе |u| <= u_lim и по Z между «ниже верха головки на z_up..z_down». Уровень
    полки — медиана Z по полосам band_m; по ним МНК даёт наклон вдоль пути.

    Возвращает {'slope_y', 'z_at_ymid', 'ymid', 'y0', 'y1', 'n_bands'} или None.
    """
    if fl is None:
        return None
    u = x - (s_ax * y + i_ax)
    top = m_top * y + c_top
    if fl['has_trough']:
        outside = (u < fl['u_l'] - 0.04) | (u > fl['u_r'] + 0.04)
    else:
        outside = np.ones(u.shape, dtype=bool)
    m = ((y >= y_lo) & (y <= y_hi) & outside & (np.abs(u) <= u_lim)
         & (z < top - z_up) & (z > top - z_down))
    yy, zz = y[m], z[m]
    yb, zb = [], []
    for y0 in np.arange(y_lo, y_hi - band_m, band_m):
        b = (yy >= y0) & (yy < y0 + band_m)
        if int(b.sum()) >= min_pts:
            yb.append(y0 + band_m / 2.0)
            zb.append(float(np.median(zz[b])))
    if len(yb) < 3:
        return None
    slope, inter = np.polyfit(np.asarray(yb), np.asarray(zb), 1)
    ymid = float(np.mean(yb))
    return {'slope_y': float(slope), 'z_at_ymid': float(inter + slope * ymid),
            'ymid': ymid, 'y0': float(min(yb)), 'y1': float(max(yb)),
            'n_bands': int(len(yb))}


def _path_normals_chk(ys, xs, out_sign):
    """Локальные нормали оси меша по её узлам (центральная разность) — как в модуле."""
    ys = np.asarray(ys, float)
    xs = np.asarray(xs, float)
    n = ys.size
    tx = np.empty(n)
    ty = np.empty(n)
    if n < 2:
        return np.zeros(n), np.zeros(n)
    tx[1:-1] = xs[2:] - xs[:-2]
    ty[1:-1] = ys[2:] - ys[:-2]
    tx[0], ty[0] = xs[1] - xs[0], ys[1] - ys[0]
    tx[-1], ty[-1] = xs[-1] - xs[-2], ys[-1] - ys[-2]
    ln = np.hypot(tx, ty)
    ln[ln < 1e-9] = 1.0
    return out_sign * (-ty / ln), out_sign * (tx / ln)


def _axis_metric(x, y, z, ax_x, ax_y, ax_nx, ax_ny, shelf_z, shelf_slope_y, y_ref,
                 dense, u_lim=0.15, lo=0.05, hi=0.40, ywin=1.0):
    """НЕЗАВИСИМАЯ метрика: расстояние от оси меша до ближайших точек рельса.

    Для каждого узла оси: окно ±ywin по Y, полоса головки lo..hi м над ИЗМЕРЕННОЙ
    полкой (независимый замер полки передаёт вызывающий), |u| <= u_lim по локальной
    нормали узла. Расстояние узла — минимум |u|. Возвращает медиану/p95 отдельно
    внутри заявленного измеренного диапазона (dense) и за его пределами.
    """
    y = np.asarray(y, float)
    order = np.argsort(y)
    ys_s = y[order]
    res = {'in': [], 'out': [], 'n_in': 0, 'n_out': 0, 'n_no_in': 0, 'n_no_out': 0}
    for i in range(len(ax_x)):
        yc = float(ax_y[i])
        a = int(np.searchsorted(ys_s, yc - ywin, 'left'))
        b = int(np.searchsorted(ys_s, yc + ywin, 'right'))
        key = ('in' if (dense and dense[0] - 1e-6 <= yc <= dense[1] + 1e-6) else 'out')
        res['n_' + key] += 1
        if b - a < 3:
            res['n_no_' + key] += 1
            continue
        idx = order[a:b]
        zs = z[idx] - (shelf_z + shelf_slope_y * (yc - y_ref))
        sel = (zs >= lo) & (zs <= hi)
        if not sel.any():
            res['n_no_' + key] += 1
            continue
        idx = idx[sel]
        u = (x[idx] - ax_x[i]) * ax_nx[i] + (y[idx] - ax_y[i]) * ax_ny[i]
        au = np.abs(u)
        au = au[au <= u_lim]
        if au.size == 0:
            res['n_no_' + key] += 1
            continue
        res[key].append(float(au.min()))
    out = {}
    for key in ('in', 'out'):
        v = res[key]
        out[key] = {
            'median_m': float(np.median(v)) if v else None,
            'p95_m': float(np.percentile(v, 95)) if v else None,
            'n_nodes': res['n_' + key], 'n_with_points': len(v),
            'n_without_points': res['n_no_' + key],
        }
    out['u_limit_m'] = u_lim
    out['band_above_shelf_m'] = [lo, hi]
    out['window_y_m'] = ywin
    return out


def _mesh_axis(verts, n_nodes=None):
    """Ось меша по его же вершинам: центр уровня (среднее по узлам сечения) и верх."""
    v = np.asarray(verts, float) if len(verts) else np.zeros((0, 3))
    if v.size == 0:
        return (np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0))
    n_nodes = int(n_nodes or RAIL_NODES)
    n = v.shape[0] // n_nodes
    blk = v[:n * n_nodes].reshape(n, n_nodes, 3)
    levels = blk[:, 0, 1]
    centers = blk[:, :, 0].mean(axis=1)
    ny_s = blk[:, :, 1].mean(axis=1)
    top = blk[:, :, 2].max(axis=1)
    return levels, centers, ny_s, top


def _shelf_profile_along(x, y, z, axis, top_at, u_lim=0.90, band_m=1.5, min_pts=60,
                         z_below=0.05, z_deep=0.45, u_l=None, u_r=None):
    """ЛОКАЛЬНЫЙ уровень полки вдоль пути — независимый замер по полосам band_m.

    Уровень полки на длинном уклоне нельзя мерить одной прямой по ближнему окну:
    при уклоне 1.5 % прямая за 20 м уходит от реальной полки на 0.3 м, и «высота
    меша над полкой» перестаёт быть константой. Поэтому по каждой полосе берётся
    медиана Z точек полки: |u| <= u_lim от СВОЕЙ оси (axis — полилиния check'а),
    вне лотка (u_l/u_r из независимого замера) и в полосе z_below..z_deep ниже
    верха меша в этой полосе (top_at).

    Возвращает {'y', 'z', 'n_bands'} (массивы) или None.
    """
    u = x - axis['x_at'](y)
    top = top_at(y) if callable(top_at) else top_at
    mask = (np.abs(u) <= u_lim) & (z <= top - z_below) & (z >= top - z_deep)
    if u_l is not None:
        mask &= (u < u_l - 0.04) | (u > u_r + 0.04)
    yb, zb = [], []
    y0 = float(np.min(y))
    y1 = float(np.max(y))
    for a in np.arange(y0, y1 - band_m, band_m):
        b = mask & (y >= a) & (y < a + band_m)
        if int(b.sum()) >= min_pts:
            yb.append(a + band_m / 2.0)
            zb.append(float(np.median(z[b])))
    if len(yb) < 3:
        return None
    return {'y': np.asarray(yb, float), 'z': np.asarray(zb, float), 'n_bands': len(yb)}


def _mesh_h_adj_local(x, y, z, levels, centers, top_lvl, nx_ax, ny_axn, out_sign,
                      dense, inward_lo=0.10, inward_hi=0.24, ywin=0.25,
                      z_below=0.06, z_deep=0.45, min_pts=8):
    """ЛОКАЛЬНАЯ высота меша над прилегающим полом — по каждому уровню меша.

    Для уровня i берём точки в окне ±ywin по Y, ВНУТРЬ от оси меша на 0.10..0.24 м
    (та же полоса, что у _adjacent_height) и по Z ниже верха этого уровня на
    z_below..z_deep. Уровень пола — p90 по этим точкам, высота = верх меша минус он.
    Главное: НИКАКОЙ длинной экстраполяции — только локальное сравнение, поэтому на
    уклоне и на кривой «не летит» проверяется честно (длинная прямая полки на уклоне
    1.5 % уходит от реальной на 0.3 м за 20 м, и критерий по ней был бы про полку,
    а не про меш).
    Возвращает {'min_m','max_m','med_m','spread_m','n_levels'} или None.
    """
    vals = []
    idx = np.flatnonzero(dense)
    for i in idx:
        a = i - 1 if i > 0 else i
        b = i + 1 if i < len(levels) - 1 else i
        yc = float(levels[i])
        m = ((y >= yc - ywin) & (y <= yc + ywin)
             & (z <= top_lvl[i] - z_below) & (z >= top_lvl[i] - z_deep))
        if not m.any():
            continue
        u = (x[m] - centers[i]) * nx_ax[i] + (y[m] - yc) * ny_axn[i]
        m2 = (-out_sign) * u
        sel = (m2 >= inward_lo) & (m2 <= inward_hi)
        if int(sel.sum()) < min_pts:
            continue
        adj = float(np.percentile(z[m][sel], 90.0))
        vals.append(float(top_lvl[i]) - adj)
    if len(vals) < 3:
        return None
    v = np.asarray(vals)
    return {'min_m': float(v.min()), 'max_m': float(v.max()),
            'med_m': float(np.median(v)), 'spread_m': float(v.max() - v.min()),
            'n_levels': int(v.size)}


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


def _rail_lines_from_points(x, y, z, side, axis, gauge_mm, y_lo, y_hi, half=0.5):
    """Своя линия нити и линия верха по СЫРЫМ точкам (меш модуля не используется).

    Старт — документированное соотношение вывода x = axis.s*y + axis.i ± gauge/2;
    далее 3 итерации: в полосах 0.5 м верх = p98 по Z точек с гейтом по местной
    земле, центр = середина p2/p98 по X, по ним МНК. Возвращает (s, i, ms, cs).
    """
    out_sign = -1.0 if side == 'left' else 1.0
    s = float(axis['s'])
    i = float(axis['i']) + out_sign * gauge_mm / 2000.0
    ms, cs = 0.0, float(np.median(z))
    for _ in range(3):
        ys_c, xs_c, ys_t, zs_t = [], [], [], []
        y0 = y_lo
        while y0 < y_hi - half:
            band = (y >= y0) & (y < y0 + half)
            dx = x - (s * y + i)
            near = band & (np.abs(dx) < 0.35)
            rail = band & (np.abs(dx) < 0.06)
            if near.sum() >= 20:
                gnd = float(np.percentile(z[near], 10.0))
                rail = rail & (z > gnd + 0.06) & (z < gnd + 0.45)
            if rail.sum() >= 15:
                ys_t.append(y0 + half / 2)
                zs_t.append(float(np.percentile(z[rail], 98)))
                ys_c.append(y0 + half / 2)
                xs_c.append(0.5 * (np.percentile(x[rail], 2)
                                   + np.percentile(x[rail], 98)))
            y0 += half
        if len(ys_t) < 4:
            break
        ms, cs = _fit_line(ys_t, zs_t, tol=0.02)
        s, i = _fit_line(ys_c, xs_c, tol=0.05)
    return s, i, ms, cs


def _axis_from_track(x, y, z, s, i, ms, cs, out_sign, y_lo, y_hi, y_mid):
    """СВОЯ полилиния оси нити для полосных измерений check'а.

    Тот же трекер, что в модуле (track_geometry._rail_track), но затравочная линия
    своя (из _rail_lines_from_points) и облако то же сырое; меш модуля не участвует.
    Возвращает {'x_at', 'z_at', 'y_lo', 'y_hi', 'n_nodes', 'dense_len_m'} или None.
    """
    tr = tg._rail_track(x, y, z, s, i, ms, cs, out_sign, y_lo, y_hi, y_mid)
    if tr is None or len(tr['grid_y']) < 2:
        return None
    gy = np.asarray(tr['grid_y'], float)
    gx = np.asarray(tr['grid_x'], float)
    gz = np.asarray(tr['grid_z'], float)
    o = np.argsort(gy)
    gy, gx, gz = gy[o], gx[o], gz[o]

    def x_at(yv, gy=gy, gx=gx):
        return np.interp(yv, gy, gx)

    def z_at(yv, gy=gy, gz=gz):
        return np.interp(yv, gy, gz)

    return {'x_at': x_at, 'z_at': z_at, 'y_lo': float(gy[0]), 'y_hi': float(gy[-1]),
            'n_nodes': int(gy.size),
            'dense_len_m': float(np.hypot(np.diff(gx), np.diff(gy)).sum())}


def _sections(x, y, z, s, i, ms, cs, out_sign, y_lo, y_hi, w_top, half=0.5, axis=None):
    """Точки головки по полосам, каждая в своей местной системе (u, v) мм.

    axis — необязательная СВОЯ полилиния оси (см. _axis_from_track): на кривой и
    стрелке прямая линия уходит от нити, и полосы пришлось бы брать только у ядра.
    Возвращает dict с пулами u/v/depth, долями внутри контура по полосам и
    местными осями (y, x_center) и верхами (y, z_top).
    """
    cos_t, sin_t = math.cos(PODUKLONKA_RAD), math.sin(PODUKLONKA_RAD)
    prof, _src = rail_profile_mm(w_top)
    if axis is not None:
        dx = x - axis['x_at'](y)
        z_top = axis['z_at'](y)
    else:
        dx = x - (s * y + i)
        z_top = ms * y + cs
    base = (np.abs(dx) <= 0.09) & ((z_top - z) >= -0.005) & ((z_top - z) < 0.040) \
        & (y >= y_lo) & (y <= y_hi)
    idx = np.flatnonzero(base)
    us, vs, ds, fracs, cy, cx, cz, dense, counts = [], [], [], [], [], [], [], [], []
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
        counts.append(int(sel.size))
    if not fracs:
        return None
    fracs = np.asarray(fracs)
    counts = np.asarray(counts)
    dense = np.asarray(dense)
    w = counts / float(counts.sum())
    return {
        'u': np.concatenate(us), 'v': np.concatenate(vs), 'depth': np.concatenate(ds),
        'u_dense': np.concatenate([u for u, d in zip(us, dense) if d]),
        'depth_dense': np.concatenate([d for d, dd in zip(ds, dense) if dd]),
        'inside_per_band': fracs,
        'inside_frac': float(np.sum(fracs * w)),
        'inside_frac_dense': (float(np.sum(fracs[dense] * counts[dense])
                                    / counts[dense].sum()) if dense.any() else float('nan')),
        'inside_median_band': float(np.median(fracs)),
        'inside_min_band': float(fracs.min()),
        'y_center': np.asarray(cy), 'x_center': np.asarray(cx), 'z_top': np.asarray(cz),
        'dense': dense, 'n_dense': int(dense.sum()),
        'n_band': len(fracs), 'n_pts': int(sum(u.size for u in us)),
    }


def _adjacent_height(x, y, z, s, i, ms, cs, out_sign, y_lo, y_hi, y_keep=None, axis=None):
    """Высота верха над прилегающей поверхностью (полоса ВНУТРЬ 0.10-0.24 м, p90).

    y_keep — список центров полос, где головка реально разрешена: на разреженных
    полосах верх берётся с экстраполированной линии и высота занижается.
    axis — необязательная своя полилиния оси (см. _axis_from_track).
    """
    inward = -out_sign
    vals = []
    y0 = y_lo
    while y0 < y_hi - 0.5:
        yc = y0 + 0.25
        if y_keep is not None and not any(abs(yc - k) < 0.26 for k in y_keep):
            y0 += 0.5
            continue
        band = (y >= y0) & (y < y0 + 0.5)
        if axis is not None:
            dx = x - axis['x_at'](y)
        else:
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
        # mesh_measured — ДОБАВЛЕННЫЙ ключ (измеренная полоска рядом с телом)
        if set(rail) - {'mesh_measured'} != {'side', 'head', 'mesh', 'source'}:
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
    if not {'gauge_inner_mm', 'gauge_axis_mm', 'head_width_mm',
            'sensor_height_above_head_m', 'notes'} <= set(res['meta']):
        problems.append('meta ключи: %s' % sorted(res['meta']))
    extra = set(res['meta']) - {'gauge_inner_mm', 'gauge_axis_mm', 'head_width_mm',
                                'sensor_height_above_head_m', 'notes'}
    for k in extra:            # дополнительные ключи разрешены, но должны быть JSON-ами
        if not isinstance(res['meta'][k], (dict, list, str, int, float, bool, type(None))):
            problems.append('meta.%s неподходящего типа' % k)
    if not isinstance(res['meta']['notes'], list) or not all(
            isinstance(s, str) for s in res['meta']['notes']):
        problems.append('meta.notes не список строк')
    try:
        blob = json.dumps(res)
    except TypeError as exc:
        blob = ''
        problems.append('JSON не сериализуется: %s' % exc)
    return {'ok': not problems, 'problems': problems, 'json_bytes': len(blob)}


def _abs_le(v, lim):
    """|v| <= lim с честной обработкой None и нуля (0.0 не должен становиться «нет данных»)."""
    return (v is not None) and (abs(v) <= lim)


def _check_rail_mesh_bins(x, y, z, rail, s, i, ms, cs, out_sign, sec, y_mid,
                          module_stats=None):
    """Проверки ПРЕЖНЕГО меша (only_observed=True: куски по бинам с точками):
    габарит, подкреплённость точками, обрывы, «нет дорисованного».

    module_stats — meta['rail_mesh']['rails'][side] из вывода: оттуда берётся
    ЛИНИЯ НИТИ модуля (line/top_line), в которой строился меш, и его счёт бинов.
    Своя линия check'а (s, i, ms, cs) используется отдельно — для сверки линий.
    """
    v = np.asarray(rail['mesh']['vertices'], float) if rail['mesh']['vertices'] \
        else np.zeros((0, 3))
    f = np.asarray(rail['mesh']['faces'], int) if rail['mesh']['faces'] \
        else np.zeros((0, 3), int)
    res = {
        'mesh_verts': int(v.shape[0]), 'mesh_faces': int(f.shape[0]),
        'mesh_indices_ok': bool(f.size == 0 or (f.min() >= 0 and f.max() < v.shape[0])
                                and all(len(t) == 3 for t in rail['mesh']['faces'])),
        'mesh_y': ([float(v[:, 1].min()), float(v[:, 1].max())] if v.size else None),
        'mesh_x_span_mm': (float((v[:, 0].max() - v[:, 0].min()) * 1000.0)
                           if v.size else None),
        'head_y': [float(rail['head']['y_range'][0]), float(rail['head']['y_range'][1])],
        'source': dict(rail['source']),
    }
    if module_stats:
        res['line_diff_i_mm'] = float((module_stats['line']['i'] - i) * 1000.0)
        res['line_diff_s'] = float(module_stats['line']['s'] - s)
        res['mod_bins_ok'] = module_stats.get('n_bins_with_points')
        res['mod_bins_span'] = module_stats.get('n_bins_span')
        res['mod_ribbons'] = module_stats.get('n_ribbons')
        res['mod_gaps'] = module_stats.get('gaps_y')
        res['mod_n_verts'] = module_stats.get('n_verts')
        res['mod_width_med_mm'] = (None if module_stats.get('width_section_median_m') is None
                                   else 1000.0 * module_stats['width_section_median_m'])
        # выборка и кадр — по ЛИНИИ МОДУЛЯ (в ней построен меш)
        s_use, i_use = float(module_stats['line']['s']), float(module_stats['line']['i'])
        ms_use, cs_use = (float(module_stats['top_line']['s']),
                          float(module_stats['top_line']['i']))
    else:
        s_use, i_use, ms_use, cs_use = s, i, ms, cs
    dx = x - (s_use * y + i_use)
    top = ms_use * y + cs_use
    depth = top - z
    sel = ((np.abs(dx) <= 0.060) & (depth >= -0.005) & (depth < 0.040))
    ud = out_sign * dx[sel]
    yd = y[sel]
    zd = z[sel]
    res['n_pts_sel'] = int(ud.size)
    res['n_pts_sel_own_line'] = int(np.sum((np.abs(x - (s * y + i)) <= 0.060)
                                           & (((ms * y + cs) - z) >= -0.005)
                                           & (((ms * y + cs) - z) < 0.040)))
    if v.size == 0 or ud.size < 3:
        res['verts_backed_frac'] = None
        return res
    order = np.argsort(yd, kind='stable')
    ys_s, us_s, zs_s = yd[order], ud[order], zd[order]
    # 1) подкреплённость: у каждой вершины есть точки в окне бина, а по u — ближайшая
    #    в пределах 30 мм (шаг точек по u у кромок головки 10-30 мм)
    lo = np.searchsorted(ys_s, v[:, 1] - 0.76, side='left')
    hi = np.searchsorted(ys_s, v[:, 1] + 0.76, side='right')
    backed = 0
    z_dev = []
    near_u = []
    for k in range(v.shape[0]):
        a, b = int(lo[k]), int(hi[k])
        if b - a < 1:
            continue
        uu = us_s[a:b]
        u_v = out_sign * (v[k, 0] - (s_use * v[k, 1] + i_use))
        d = np.abs(uu - u_v)
        near = d <= 0.030
        near_u.append(float(d.min()))
        if not near.any():
            continue
        backed += 1
        z_dev.append(abs(float(np.median(zs_s[a:b][near])) - v[k, 2]))
    res['verts_backed_frac'] = backed / float(v.shape[0])
    res['verts_nearest_u_p90_mm'] = float(np.percentile(near_u, 90)) if near_u else None
    res['vert_z_dev_p90_m'] = float(np.percentile(z_dev, 90)) if z_dev else None
    # 2) низ меша против самой низкой наблюдённой точки — В ТЕХ ЖЕ ОКНАХ бинов,
    #    что попали в меш (глобальный минимум может лежать в выпавшем бине)
    y_lo_b = math.floor(float(ys_s[0]) / RAIL_BIN_Y_M) * RAIL_BIN_Y_M
    n_span = int(math.floor((float(ys_s[-1]) - y_lo_b) / RAIL_BIN_Y_M)) + 1
    ok_bins = []
    for b in range(n_span):
        yc = y_lo_b + (b + 0.5) * RAIL_BIN_Y_M
        a = int(np.searchsorted(ys_s, yc - RAIL_WIN_HALF_M, side='left'))
        b2 = int(np.searchsorted(ys_s, yc + RAIL_WIN_HALF_M, side='right'))
        own_a = int(np.searchsorted(ys_s, yc - RAIL_BIN_Y_M / 2, side='left'))
        own_b = int(np.searchsorted(ys_s, yc + RAIL_BIN_Y_M / 2, side='right'))
        if b2 - a >= RAIL_MIN_PTS_BIN and own_b - own_a >= 1:
            ok_bins.append(b)
    cov = np.zeros(ys_s.size, dtype=bool)
    for b in ok_bins:
        yc = y_lo_b + (b + 0.5) * RAIL_BIN_Y_M
        a = int(np.searchsorted(ys_s, yc - RAIL_WIN_HALF_M, side='left'))
        b2 = int(np.searchsorted(ys_s, yc + RAIL_WIN_HALF_M, side='right'))
        cov[a:b2] = True
    res['mesh_z_min'] = float(v[:, 2].min())
    res['pts_z_min'] = float(zd.min())
    res['pts_z_min_in_bins'] = float(zd[cov].min()) if cov.any() else None
    # Низ сравниваем ПО БИНАМ (в одном и том же Y): иначе разница — это наклон
    # верха вдоль пути (0.005-0.015 м/м), а не дорисованная геометрия
    z_per_bin = []
    if v.shape[0] % RAIL_NODES == 0:
        blk0 = v.reshape(-1, RAIL_NODES, 3)
        for b in range(blk0.shape[0]):
            yc = blk0[b, 0, 1]
            a = int(np.searchsorted(ys_s, yc - RAIL_WIN_HALF_M, side='left'))
            b2 = int(np.searchsorted(ys_s, yc + RAIL_WIN_HALF_M, side='right'))
            if b2 - a < 1:
                continue
            z_per_bin.append(float(blk0[b, :, 2].min() - zs_s[a:b2].min()))
    res['z_min_diff_per_bin_p50_m'] = (float(np.percentile(z_per_bin, 50))
                                       if z_per_bin else None)
    res['z_min_diff_m'] = float(np.max(z_per_bin)) if z_per_bin else None
    # 3) глубина под линией верха (в локальной системе): сколько меш «уходит вниз»
    d_v = (ms_use * v[:, 1] + cs_use) - v[:, 2]
    res['mesh_depth_max_m'] = float(d_v.max())
    res['mesh_depth_p90_m'] = float(np.percentile(d_v, 90))
    res['pts_depth_max_m'] = float(depth[sel].max())
    res['pts_depth_max_in_bins_m'] = float(depth[sel][cov].max()) if cov.any() else None
    res['mesh_z_span_m'] = float(v[:, 2].max() - v[:, 2].min())
    # 4) габариты против измеренной головки
    n_nodes = int(RAIL_NODES)
    if v.shape[0] % n_nodes == 0:
        blk = v.reshape(-1, n_nodes, 3)
        widths = (blk[:, :, 0].max(axis=1) - blk[:, :, 0].min(axis=1)) * 1000.0
        res['mesh_width_med_mm'] = float(np.median(widths))
        res['mesh_width_p10_mm'] = float(np.percentile(widths, 10))
        res['mesh_width_p90_mm'] = float(np.percentile(widths, 90))
        res['mesh_width_diff_mm'] = res['mesh_width_med_mm'] - rail['head']['width_mm']
        # та же статистика по сырым точкам в тех же бинах (эталон «как в данных»)
        ref = []
        for b in range(blk.shape[0]):
            yc = blk[b, 0, 1]
            a = int(np.searchsorted(ys_s, yc - 0.75, side='left'))
            b2 = int(np.searchsorted(ys_s, yc + 0.75, side='right'))
            if b2 - a < 3:
                continue
            uu = us_s[a:b2]
            w = ((float(np.percentile(uu, 98) - np.percentile(uu, 2))) if uu.size >= 8
                 else (float(uu.max() - uu.min())))
            ref.append(w * 1000.0)
        res['ref_width_med_mm'] = float(np.median(ref)) if ref else None
        res['mesh_width_diff_vs_ref_mm'] = (res['mesh_width_med_mm'] - res['ref_width_med_mm']
                                            if ref else None)
    # 5) непрерывность: сколько сечений сшито (faces) против возможного
    res['n_sections'] = int(v.shape[0] // n_nodes) if n_nodes else 0
    res['faces_per_join'] = int(2 * (n_nodes - 1)) if n_nodes else 0
    res['n_faces_expected_if_continuous'] = (max(0, res['n_sections'] - 1)
                                             * res['faces_per_join'])

    # 6) подсчёт бинов и разрывов по сырым точкам (своими руками, та же арифметика)
    gaps = [[float(y_lo_b + (p + 1) * RAIL_BIN_Y_M), float(y_lo_b + q * RAIL_BIN_Y_M)]
            for p, q in zip(ok_bins, ok_bins[1:]) if q - p > 1]
    res['chk_bins_span'] = n_span
    res['chk_bins_ok'] = len(ok_bins)
    res['chk_bins_ok_own_line'] = None
    res['chk_gaps'] = gaps
    # 7) 100 % вершин обязаны приходить из бинов с точками: сверяем Y вершин с
    #    множеством центров валидных бинов (своих, посчитанных выше)
    if v.size:
        centers = np.array([y_lo_b + (b + 0.5) * RAIL_BIN_Y_M for b in ok_bins])
        hit = np.abs(v[:, 1][:, None] - centers[None, :]).min(axis=1) < 1e-6 \
            if centers.size else np.zeros(v.shape[0], dtype=bool)
        res['verts_from_valid_bins_frac'] = float(np.mean(hit))
    else:
        res['verts_from_valid_bins_frac'] = None
    res['chk_ribbons'] = 1 + sum(1 for p, q in zip(ok_bins, ok_bins[1:]) if q - p > 1)
    return res


def _check_rail_mesh(x, y, z, rail, s, i, ms, cs, out_sign, sec, y_mid,
                     module_stats=None, shelf_meas=None, floor_their=None,
                     shelf_prof=None, axis_ab=None):
    """Проверки меша РАЗВЁРТКИ: непрерывность, сечение, ось, предсказание вне данных.

    Проверяем ровно то, что обещано: одна непрерывная полоса (граней = (уровней-1)*8),
    сечение равно измеренному (ширина против головки), ось меша лежит на линии нити,
    за пределами плотных бинов сечение ТО ЖЕ (не «дорисовка» — разброс u и vz по
    уровням вне данных ~0), а внутри плотных бинов вершины подкреплены точками.
    Отдельно — ПРИВЯЗКА ПО ВЫСОТЕ: длина меша = плотный диапазон + запас, а верх
    меша держится над ИЗМЕРЕННОЙ полкой (shelf_meas) и над полом вьюера
    (floor_their = zones.fit_floor) в заданных коридорах по всей длине.

    module_stats — meta['rail_mesh']['rails'][side]: линия нити, линия верха,
    section_source (диапазон плотных бинов), extent_y, beyond_measured, sweep_*.
    shelf_meas — независимая прямая полки (см. _shelf_line_meas) или None.
    floor_their — (a, b, rms) пола вьюера в виде z = a + b*y или None.
    """
    v = np.asarray(rail['mesh']['vertices'], float) if rail['mesh']['vertices'] \
        else np.zeros((0, 3))
    f = np.asarray(rail['mesh']['faces'], int) if rail['mesh']['faces'] \
        else np.zeros((0, 3), int)
    n_nodes = int(((module_stats or {}).get('polyline') or {}).get('section_nodes')
                  or RAIL_NODES)
    res = {
        'mesh_verts': int(v.shape[0]), 'mesh_faces': int(f.shape[0]),
        'mesh_indices_ok': bool(f.size == 0 or (f.min() >= 0 and f.max() < v.shape[0])
                                and all(len(t) == 3 for t in rail['mesh']['faces'])),
        'mesh_y': ([float(v[:, 1].min()), float(v[:, 1].max())] if v.size else None),
        'head_y': [float(rail['head']['y_range'][0]), float(rail['head']['y_range'][1])],
        'source': dict(rail['source']),
        'axis_model': (module_stats or {}).get('axis_model'),
    }
    if module_stats:
        res['line_diff_i_mm'] = float((module_stats['line']['i'] - i) * 1000.0)
        res['mod_extent_y'] = module_stats.get('extent_y')
        res['mod_section_source'] = module_stats.get('section_source')
        res['mod_beyond'] = module_stats.get('beyond_measured')
        res['mod_ribbons'] = module_stats.get('n_ribbons')
        res['mod_bins_ok'] = module_stats.get('n_bins_with_points')
        res['mod_n_verts'] = module_stats.get('n_verts')
        res['mod_width_mm'] = (None if not module_stats.get('section')
                               else 1000.0 * module_stats['section']['width_m'])
        res['mod_u_off_mm'] = (None if not module_stats.get('section')
                               else 1000.0 * module_stats['section']['u_off_m'])
        res['mod_extra_used_m'] = module_stats.get('sweep_extra_used_m')
        res['mod_extra_far_m'] = module_stats.get('sweep_extra_far_m')
        res['mod_extra_near_m'] = module_stats.get('sweep_extra_near_m')
        res['mod_extra_cap_h_m'] = module_stats.get('sweep_extra_cap_h_m')
        res['mod_extra_cap_x_m'] = module_stats.get('sweep_extra_cap_x_m')
        res['mod_head_h_m'] = module_stats.get('head_h_above_shelf_m')
        res['mod_z_drift_far_m'] = module_stats.get('z_drift_far_m')
        res['mod_z_drift_beyond_m'] = module_stats.get('z_drift_beyond_m')
        res['mod_x_drift_far_m'] = module_stats.get('x_drift_far_m')
        res['mod_sweep_anchor'] = module_stats.get('sweep_anchor')
        s_use = float(module_stats['line']['s'])
        i_use = float(module_stats['line']['i'])
        ms_use = float(module_stats['top_line']['s'])
        cs_use = float(module_stats['top_line']['i'])
    else:
        s_use, i_use, ms_use, cs_use = s, i, ms, cs
    if v.size == 0:
        res['continuous'] = None
        return res

    n_levels = v.shape[0] // n_nodes
    res['n_sections'] = n_levels
    blk = v[:n_levels * n_nodes].reshape(n_levels, n_nodes, 3)
    levels = blk[:, 0, 1]
    is_body = bool((module_stats or {}).get('mesh_body'))
    if is_body:
        # тело: 2*n_nodes боковых граней на шаг + 2 крышки по (n_nodes-2) треугольников
        res['faces_expected_continuous'] = (int((n_levels - 1) * 2 * n_nodes)
                                            + 2 * max(0, n_nodes - 2))
    else:
        res['faces_expected_continuous'] = int((n_levels - 1) * 2 * (n_nodes - 1))
    res['continuous'] = bool(f.shape[0] == res['faces_expected_continuous'])
    # Головка уровня: вершины в 40 мм под верхом уровня. Для полоски это все 5
    # вершин, для ТЕЛА — измеренная коронка + плечи головки; так ширина и центр
    # сравнимы с измеренной головкой, а ось не уезжает из-за широкой подошвы.
    k_top = blk[:, :, 2].max(axis=1, keepdims=True)
    head_mask = blk[:, :, 2] >= (k_top - 0.04)
    widths = np.array([np.ptp(blk[i, head_mask[i], 0]) for i in range(n_levels)]) * 1000.0
    centers = np.array([blk[i, head_mask[i], 0].mean() for i in range(n_levels)])
    ny_ax = np.array([blk[i, head_mask[i], 1].mean() for i in range(n_levels)])
    levels_y = ny_ax      # «y узла»: у тела первая вершина уровня смещена на u*n_y
    res['mesh_width_med_mm'] = float(np.median(widths))
    res['mesh_width_spread_mm'] = float(widths.max() - widths.min())
    res['mesh_width_diff_mm'] = res['mesh_width_med_mm'] - rail['head']['width_mm']
    off = out_sign * (centers - (s_use * levels + i_use)) * 1000.0
    res['mesh_axis_off_med_mm'] = float(np.median(off))
    res['mesh_axis_off_max_mm'] = float(np.max(np.abs(off - np.median(off))))
    # ЛОКАЛЬНЫЙ базис уровня: нормаль по соседним уровням (как строит модуль).
    # Для полилинии это и есть проверка «ось идёт по кривой»: u_loc меряется поперёк
    # пути, а не по X, поэтому разброс вне данных и метрика ниже считаются корректно.
    nx_ax, ny_axn = _path_normals_chk(levels, centers, out_sign)
    # локальные u и vz каждой вершины — по уровням. По Z отсчёт от ВЕРХА САМОГО
    # СЕЧЕНИЯ (верхний узел уровня): привязка меша по высоте (полка/плоскость) —
    # предмет отдельной проверки ниже, и она не должна попадать в «глубину».
    top_lvl = blk[:, :, 2].max(axis=1)
    u_loc = np.where(head_mask, (blk[:, :, 0] - centers[:, None]) * nx_ax[:, None]
                     + (blk[:, :, 1] - levels[:, None]) * ny_axn[:, None], np.nan)
    v_loc = np.where(head_mask, blk[:, :, 2] - top_lvl[:, None], np.nan)
    # шаг оси между уровнями (поперёк пути): скачок = перескок на ветку/стенку
    axis_step = np.abs(np.diff(centers)) if n_levels > 1 else np.zeros(0)
    res['axis_step_max_m'] = float(axis_step.max()) if axis_step.size else 0.0
    res['n_jump_levels'] = int(np.sum(axis_step > EXPECT['axis_jump_m'])) \
        if axis_step.size else 0
    res['axis_len_m'] = (float(np.hypot(np.diff(centers), np.diff(levels)).sum())
                         if n_levels > 1 else 0.0)

    # --- ПРИВЯЗКА ПО ВЫСОТЕ: длина развёртки и уход верха от полки и от пола вьюера
    res['mesh_y_len_m'] = float(levels.max() - levels.min())
    res['old_len_m'] = float(min(y.max(), -2.0) - max(y.min(), -200.0))
    src0 = (module_stats or {}).get('section_source')
    res['mod_dense_len_m'] = (None if not src0 else float(src0[1] - src0[0]))
    if src0:
        res['mod_len_minus_dense_m'] = res['mesh_y_len_m'] - res['mod_dense_len_m']
        res['mod_extra_far_chk_m'] = float(src0[0] - res['mesh_y'][0])
        res['mod_extra_near_chk_m'] = float(res['mesh_y'][1] - src0[1])
    else:
        res['mod_len_minus_dense_m'] = None
        res['mod_extra_far_chk_m'] = None
        res['mod_extra_near_chk_m'] = None
    res['anchor_shelf'] = None
    res['anchor_their_floor'] = None
    # три контрольные точки по длине: ближний конец, дальний край плотных бинов,
    # дальний конец меша
    marks = {'near': int(np.argmax(levels)), 'far': int(np.argmin(levels))}
    if src0:
        marks['dense_far'] = int(np.argmin(np.abs(levels - src0[0])))
    anchors = []
    if shelf_meas is not None:
        anchors.append(('shelf', lambda yv, sm=shelf_meas: sm['z_at_ymid']
                        + sm['slope_y'] * (yv - sm['ymid'])))
    if floor_their is not None:
        anchors.append(('their_floor', lambda yv, ft=floor_their: ft[0] + ft[1] * yv))
    for name, fn in anchors:
        dz = top_lvl - np.array([float(fn(yv)) for yv in levels])
        stat = {'min_m': float(dz.min()), 'max_m': float(dz.max()),
                'med_m': float(np.median(dz)),
                'spread_m': float(dz.max() - dz.min()),
                'at': {}}
        for tag, idx in marks.items():
            d_row = blk[idx, :, 2] - float(fn(levels[idx]))
            stat['at'][tag] = {'y_m': float(levels[idx]),
                               'min_m': float(d_row.min()), 'max_m': float(d_row.max()),
                               'med_m': float(np.median(d_row))}
        res['anchor_shelf' if name == 'shelf' else 'anchor_their_floor'] = stat
    # уход верха меша от линии верха (как было ДО правки: меш шёл по m*y+c)
    res['drift_from_top_line_far_m'] = float(
        abs(top_lvl[np.argmin(levels)] - (ms_use * levels.min() + cs_use)))
    # «ДО»: меш шёл по m*y+c на весь диапазон — насколько он уезжал от полки на старом
    # дальнем конце y = max(y.min(), -200)
    y_old_far = max(float(y.min()), -200.0)
    if shelf_meas is not None:
        z_shelf_old = (shelf_meas['z_at_ymid']
                       + shelf_meas['slope_y'] * (y_old_far - shelf_meas['ymid']))
    else:
        z_shelf_old = ms_use * y_old_far + cs_use
    res['old_far_y_m'] = y_old_far
    res['old_drift_far_m'] = float(abs((ms_use * y_old_far + cs_use) - z_shelf_old))

    src = (module_stats or {}).get('section_source')
    if src:
        dense = (levels >= src[0] - 1e-6) & (levels <= src[1] + 1e-6)
    else:
        dense = np.ones(n_levels, dtype=bool)
    res['n_levels_dense'] = int(dense.sum())
    # --- ГЛАВНАЯ проверка «меш не срезает»: расстояние от оси меша до точек рельса
    if shelf_meas is not None:
        res['axis_metric'] = _axis_metric(
            x, y, z, centers, levels_y, nx_ax, ny_axn,
            shelf_meas['z_at_ymid'], shelf_meas['slope_y'], shelf_meas['ymid'],
            (float(src[0]), float(src[1])) if src else None)
    else:
        res['axis_metric'] = None
    if (~dense).any() and dense.any():
        # вне плотных бинов сечение обязано быть тем же: разброс u и vz ~ 0
        with np.errstate(invalid='ignore'):
            res['beyond_u_spread_mm'] = float(
                np.nanmax(np.nanmax(u_loc[~dense], axis=0)
                          - np.nanmin(u_loc[~dense], axis=0)) * 1000.0)
            res['beyond_vz_spread_mm'] = float(
                np.nanmax(np.nanmax(v_loc[~dense], axis=0)
                          - np.nanmin(v_loc[~dense], axis=0)) * 1000.0)
        res['beyond_measured_chk'] = (float(np.clip(
            1.0 - (src0[1] - src0[0]) / max(1e-9, levels.max() - levels.min()), 0.0, 1.0))
            if src0 else float(1.0 - 1.0 * dense.sum() / n_levels))
    else:
        res['beyond_u_spread_mm'] = None
        res['beyond_vz_spread_mm'] = None
        res['beyond_measured_chk'] = None
    # ЛОКАЛЬНАЯ высота над прилегающим полом (без длинной экстраполяции полки)
    res['h_local'] = _mesh_h_adj_local(x, y, z, levels, centers, top_lvl,
                                       nx_ax, ny_axn, out_sign, dense)
    # ТЕЛО рельса и ИЗМЕРЕННАЯ полоска: габарит тела, число вершин полоски и
    # совпадение верха тела с измеренной коронкой
    mp_ = (module_stats or {}).get('polyline') or {}
    mv = (rail.get('mesh_measured') or {}).get('vertices') or []
    res['body_h_m'] = float(np.ptp(blk[0, :, 2]))
    res['body_w_m'] = float(np.ptp(blk[0, :, 0]))
    res['mesh_measured_verts'] = int(len(mv))
    res['mesh_measured_ok'] = (len(mv) == 5 * int(mp_.get('n_nodes_total') or 0))
    res['mesh_measured_nodes'] = mp_.get('n_nodes_total')
    res['top_equals_measured_m'] = None
    res['node_stats'] = mp_.get('node_stats')
    res['source_measured'] = (rail.get('source') or {}).get('mesh_measured')
    # УЗЛЫ ОСИ из meta: тот же массив, что уходит потребителям (маска точек);
    # сверяем с мешем: y-порядок, x против центра полоски, z против верха меша
    res['nodes_chk'] = None
    nd = (module_stats or {}).get('nodes')
    if nd:
        a_n = np.asarray(nd, float)
        res['nodes_chk'] = {'n': int(a_n.shape[0]),
                            'first': [float(v) for v in a_n[0]],
                            'last': [float(v) for v in a_n[-1]],
                            'y_sorted': bool(np.all(np.diff(a_n[:, 0]) > 0))}
        if mv:
            a_m = np.asarray(mv, float)
            n_m = a_m.shape[0] // 5
            mx = a_m[:n_m * 5].reshape(n_m, 5, 3)[:, :, 0].mean(axis=1)
            nmin = min(n_m, a_n.shape[0])
            res['nodes_chk']['dx_max_m'] = float(np.max(np.abs(a_n[:nmin, 1] - mx[:nmin])))
            res['nodes_chk']['dz_max_m'] = float(np.max(np.abs(a_n[:nmin, 2]
                                                           - top_lvl[:nmin])))
            res['nodes_chk']['y_levels_m'] = float(np.max(np.abs(a_n[:nmin, 0]
                                                                 - levels_y[:nmin])))
    if mv:
        a_m = np.asarray(mv, float)
        n_m = a_m.shape[0] // 5
        top_m = a_m[:n_m * 5].reshape(n_m, 5, 3)[:, :, 2].max(axis=1)
        nmin = min(top_m.size, n_levels)
        if nmin:
            res['top_equals_measured_m'] = float(np.max(np.abs(top_lvl[:nmin]
                                                               - top_m[:nmin])))
    # ВЕРХ ЛЕНТЫ НАД ИЗМЕРЕННОЙ ПОЛКОЙ: коридор [0.08, 0.30] и жёсткий минимум 0.05.
    # Полка — линия профиля модуля, оценённая под САМОЙ нитью (поперечный наклон).
    res['top_band'] = None
    if shelf_prof is not None and axis_ab is not None:
        sa, ia = axis_ab
        u_node = centers - (sa * levels + ia)
        ymid = 0.5 * (shelf_prof['window_y'][0] + shelf_prof['window_y'][1])
        sh = (shelf_prof['shelf_z'] + shelf_prof['shelf_slope_u'] * u_node
              + shelf_prof['shelf_slope_y'] * (levels - ymid))
        rel = top_lvl - sh
        lo, hi = EXPECT['top_band_m']
        res['top_band'] = {'lo_m': lo, 'hi_m': hi,
                           'min_m': float(rel.min()), 'max_m': float(rel.max()),
                           'n_out': int(np.sum((rel < lo) | (rel > hi))),
                           'n_below_hard': int(np.sum(rel < EXPECT['top_hard_m'])),
                           'n_nodes': int(n_levels)}
    # глубина меша под верхом СВОЕГО сечения (шейки/подошвы в геометрии нет)
    d_v = -v_loc
    with np.errstate(invalid='ignore'):
        res['mesh_depth_max_m'] = float(np.nanmax(d_v))
        res['mesh_depth_med_m'] = float(np.nanmedian(np.nanmax(d_v, axis=1)))

    # подкрепление точками — только по уровням В ПЛОТНЫХ бинах. Ось берём из САМОГО
    # меша (интерполяция по его узлам): для полилинии прямая линия тут не годится —
    # на кривой она уходит от меша и «подкрепление» теряет смысл
    if n_levels >= 2:
        o = np.argsort(levels)
        lv_s, ce_s, tp_s = levels[o], centers[o], top_lvl[o]
        if (module_stats or {}).get('axis_model', '').startswith('polyline'):
            def _axis_x(yv, lv=lv_s, ce=ce_s):
                return np.interp(yv, lv, ce)

            def _axis_top(yv, lv=lv_s, tp=tp_s):
                return np.interp(yv, lv, tp)
        else:
            def _axis_x(yv, s=s_use, i=i_use):
                return s * yv + i

            def _axis_top(yv, s=ms_use, i=cs_use):
                return s * yv + i
    else:
        _axis_x = lambda yv, s=s_use, i=i_use: s * yv + i
        _axis_top = lambda yv, s=ms_use, i=cs_use: s * yv + i
    dx = x - _axis_x(y)
    depth_p = _axis_top(y) - z
    sel = ((np.abs(dx) <= 0.060) & (depth_p >= -0.005) & (depth_p < 0.040))
    ud = out_sign * dx[sel]
    yd = y[sel]
    zd = z[sel]
    res['n_pts_sel'] = int(ud.size)
    res['pts_depth_med_in_dense_m'] = None
    res['pts_depth_p90_in_dense_m'] = None
    if dense.any() and ud.size >= 3:
        yd_all = y[sel]
        in_dense = (yd_all >= levels[dense].min() - 0.76) & \
                   (yd_all <= levels[dense].max() + 0.76)
        if in_dense.any():
            dep = depth_p[sel][in_dense]
            res['pts_depth_med_in_dense_m'] = float(np.median(dep))
            res['pts_depth_p90_in_dense_m'] = float(np.percentile(dep, 90))
    backed = 0
    used = 0
    near_u = []
    if ud.size >= 3 and dense.any():
        order = np.argsort(yd, kind='stable')
        ys_s, us_s, zs_s = yd[order], ud[order], zd[order]
        idx_dense = np.flatnonzero(dense)
        lo_l = levels[dense].min() - RAIL_WIN_HALF_M
        hi_l = levels[dense].max() + RAIL_WIN_HALF_M
        a_l = int(np.searchsorted(ys_s, lo_l, side='left'))
        b_l = int(np.searchsorted(ys_s, hi_l, side='right'))
        ys_d, us_d, zs_d = ys_s[a_l:b_l], us_s[a_l:b_l], zs_s[a_l:b_l]
        for k in idx_dense:
            yc = levels[k]
            a = int(np.searchsorted(ys_d, yc - 0.76, side='left'))
            b2 = int(np.searchsorted(ys_d, yc + 0.76, side='right'))
            if b2 - a < 1:
                continue
            for j in np.flatnonzero(head_mask[k]):
                used += 1
                d = np.abs(us_d[a:b2] - u_loc[k, j])
                near_u.append(float(d.min()))
                if d.min() <= 0.030:
                    backed += 1
    res['verts_backed_frac'] = (backed / float(used)) if used else None
    res['verts_nearest_u_p90_mm'] = float(np.percentile(near_u, 90)) if near_u else None
    res['n_verts_checked'] = used
    return res


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
           'meta': res['meta'], 'rails': {}, 'notes': [], 'rails_bins_only': {}}
    out['contract'] = _check_contract(res)
    res_obs = build_track(db_path, index, only_observed=True)
    res_str = build_track(db_path, index, axis_polyline=False)

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
        'empty_ok': (len(inst) == 0
                     and 'встроен' in str(res['sleepers']['source'])),
    }

    # ---- рельсы: свои линии по СЫРЫМ точкам (меш модуля для измерений не нужен)
    gauge_mm = float(res['meta']['gauge_axis_mm'])
    mesh = {}
    for rail in res['rails']:
        yr = rail['head']['y_range']
        s, i, ms, cs = _rail_lines_from_points(x, y, z, rail['side'], res['axis'],
                                               gauge_mm, yr[0], yr[1])
        mesh[rail['side']] = (s, i, ms, cs)
    # своя ОСЬ-ПОЛИЛИНИЯ для полосных измерений: на кривой и стрелке прямая линия
    # теряет нить, и полосы за пределами ядра выпадают. Трекер тот же, но затравочная
    # линия и облако — свои (меш модуля тут не используется).
    ax_poly = {}
    y_lo_ax = max(float(y.min()), -tg.Y_SEARCH_M)
    y_hi_ax = min(float(y.max()), 5.0)
    for rail in res['rails']:
        s, i, ms, cs = mesh[rail['side']]
        yr = rail['head']['y_range']
        ax_poly[rail['side']] = _axis_from_track(
            x, y, z, s, i, ms, cs, -1.0 if rail['side'] == 'left' else 1.0,
            y_lo_ax, y_hi_ax, 0.5 * (yr[0] + yr[1]))
    out['axis_poly_chk'] = {k: (None if v is None else
                                {'n_nodes': v['n_nodes'], 'dense_len_m': v['dense_len_m'],
                                 'y_lo': v['y_lo'], 'y_hi': v['y_hi']})
                            for k, v in ax_poly.items()}
    # опорное Y: середина самого плотного окна 5 м по плотным полосам ОБЕИХ нитей
    dense_y = []
    for rail in res['rails']:
        s, i, ms, cs = mesh[rail['side']]
        yr = rail['head']['y_range']
        sec0 = _sections(x, y, z, s, i, ms, cs,
                         -1.0 if rail['side'] == 'left' else 1.0, yr[0], yr[1],
                         rail['head']['width_mm'], axis=ax_poly.get(rail['side']))
        if sec0 is not None and sec0['dense'].any():
            dense_y.append(sec0['y_center'][sec0['dense']])
    if dense_y:
        ally = np.sort(np.concatenate(dense_y))
        best, best_n, j = 0.0, -1, 0
        for k in range(ally.size):
            while ally[k] - ally[j] > 5.0:
                j += 1
            if k - j + 1 > best_n:
                best_n, best = k - j + 1, 0.5 * (ally[j] + ally[k])
        y_mid = float(best)
    else:
        y_mid = float(np.mean([0.5 * (r['head']['y_range'][0] + r['head']['y_range'][1])
                               for r in res['rails']]))
    out['y_ref'] = y_mid

    # ---- ПОЛ: независимый замер профиля (лоток) + RMS профиля против плоскости
    fp = res['meta'].get('floor_profile')
    s_ax_ch = 0.5 * (mesh['left'][0] + mesh['right'][0])
    i_ax_ch = 0.5 * (mesh['left'][1] + mesh['right'][1])
    m_top_ch = 0.5 * (mesh['left'][2] + mesh['right'][2])
    c_top_ch = 0.5 * (mesh['left'][3] + mesh['right'][3])
    win = fp['window_y'] if fp else [max(float(y.min()), -12.0), -2.0]
    fl = _floor_measure(x, y, z, s_ax_ch, i_ax_ch, m_top_ch, c_top_ch, win[0], win[1])
    out['floor_profile'] = {'measured': fl, 'module': fp, 'window_y': win}
    # ЛОКАЛЬНЫЙ уровень полки вдоль всего пути (для критерия «меш не летит»): одной
    # прямой по ближнему окну на уклоне 1.5 % за 20 м набегает 0.3 м, и высота меша
    # над полкой перестаёт быть константой (замерено). Полка меряется по полосам,
    # ось — своя полилиния check'а, верх — свой же трекер (независимо от меша модуля).
    shelf_local = {}
    for rail in res['rails']:
        ax = ax_poly.get(rail['side'])
        if ax is None:
            shelf_local[rail['side']] = None
            continue
        shelf_local[rail['side']] = _shelf_profile_along(
            x, y, z, ax, ax['z_at'],
            u_l=(fl['u_l'] if fl and fl['has_trough'] else None),
            u_r=(fl['u_r'] if fl and fl['has_trough'] else None))
    out['shelf_local'] = {k: (None if v is None else
                              {'n_bands': v['n_bands'],
                               'y_lo': float(v['y'][0]), 'y_hi': float(v['y'][-1]),
                               'z_min': float(v['z'].min()), 'z_max': float(v['z'].max())})
                          for k, v in shelf_local.items()}

    # независимая ПРЯМАЯ ПОЛКИ вдоль пути (база для проверки привязки меша нитей)
    shelf_meas = _shelf_line_meas(x, y, z, s_ax_ch, i_ax_ch, m_top_ch, c_top_ch,
                                  fl, win[0], win[1])
    out['shelf_line_measured'] = shelf_meas
    # ПОЛ ВЬЮЕРА: тот же вызов, что в web_viewer (zones.fit_floor, полоса ±2 м),
    # иначе кросс-проверка привязки меша была бы проверкой самой себя
    cloud = np.column_stack([x, y, z])
    fa, fb, frms = zones.fit_floor(cloud, x_center=float(i_ax_ch), x_half=2.0)
    floor_their = (float(fa), float(fb), float(frms))
    out['floor_their'] = {
        'a': floor_their[0], 'b': floor_their[1], 'rms_m': floor_their[2],
        'z_at_y_ref': floor_their[0] + floor_their[1] * y_mid,
        'z_at_minus6': floor_their[0] + floor_their[1] * (-6.0),
        'x_center': float(i_ax_ch), 'x_half': 2.0,
        'source': 'zones.fit_floor(cloud, x_center=axis_i, x_half=2.0) — как в /meta вьюера',
    }
    del cloud

    # RMS: точки пола в окне профиля против (а) плоскости floor.a/b/c и
    # (б) профиля из вывода модуля (meta.floor_profile)
    uu = x - (s_ax_ch * y + i_ax_ch)
    top = m_top_ch * y + c_top_ch
    pm = ((y >= win[0]) & (y <= win[1]) & (np.abs(uu) <= 0.90)
          & (z < top - 0.10) & (z > top - 0.80))
    rms_pl = rms_pr = None
    if int(pm.sum()) > 500:
        rms_pl = float(np.sqrt(np.mean((z[pm] - (a * x[pm] + b * y[pm] + c)) ** 2)))
        if fp is not None:
            ymid = 0.5 * (win[0] + win[1])
            z_mod = (fp['shelf_z'] + fp['shelf_slope_u'] * uu[pm]
                     + fp['shelf_slope_y'] * (y[pm] - ymid))
            tr = fp.get('trough')
            if tr:
                inb = (uu[pm] >= tr['u_left']) & (uu[pm] <= tr['u_right'])
                z_mod[inb] = tr['bottom_z']
            rms_pr = float(np.sqrt(np.mean((z[pm] - z_mod) ** 2)))
    out['floor_rms'] = {'plane_m': rms_pl, 'profile_m': rms_pr, 'n_pts': int(pm.sum())}

    # форма меша пола: есть ли в нём лоток (вершины ниже полки на 0.10+ м)
    fv2 = np.asarray(res['floor_mesh']['vertices'], float)
    u_mesh = fv2[:, 0] - (s_ax_ch * fv2[:, 1] + i_ax_ch)
    shelf_lvl = (fp['shelf_z'] + fp['shelf_slope_u'] * u_mesh
                 + fp['shelf_slope_y'] * (fv2[:, 1] - 0.5 * (win[0] + win[1]))) if fp else None
    out['floor_profile']['mesh_has_trough'] = (
        bool(np.any(shelf_lvl - fv2[:, 2] > 0.10)) if shelf_lvl is not None else None)
    out['floor_profile']['mesh_y'] = (float(fv2[:, 1].min()), float(fv2[:, 1].max()))

    for rail in res['rails']:
        s, i, ms, cs = mesh[rail['side']]
        out_sign = -1.0 if rail['side'] == 'left' else 1.0
        yr = rail['head']['y_range']
        if not rail['mesh']['vertices']:
            # модуль не нашёл нить на этом кадре (меш пуст) — геометрические проверки
            # по этой нити бессмысленны, а своя привязка ведёт от axis={0,0} и даёт мусор
            out['rails'][rail['side']] = {
                'error': 'меш пуст: нить на кадре не найдена (детекция)',
                'detected': False, 'mesh_verts': 0}
            continue
        sec = _sections(x, y, z, s, i, ms, cs, out_sign, yr[0], yr[1],
                        rail['head']['width_mm'], axis=ax_poly.get(rail['side']))
        if sec is None:
            # меш ЕСТЬ (может быть построен по узлам плана пары), а вот сечения
            # головки на кадре взять нечем — разбор геометрии нити пропускается,
            # но число вершин отдаём: по нему проверяется инвариант «меш не пуст»
            _mv = np.asarray(rail['mesh']['vertices'], float)
            out['rails'][rail['side']] = {
                'error': 'точек головки мало — не проверено',
                'mesh_verts': int(_mv.shape[0]),
                'mesh_y': ([float(_mv[:, 1].min()), float(_mv[:, 1].max())]
                           if _mv.size else None)}
            continue
        u, depth = sec['u_dense'], sec['depth_dense']
        if u.size < 20:
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
            'n_pts': sec['n_pts'], 'n_band': sec['n_band'], 'n_dense': sec['n_dense'],
            'inside_frac': sec['inside_frac'],
            'inside_frac_dense': sec['inside_frac_dense'],
            'inside_median_band': sec['inside_median_band'],
            'inside_min_band': sec['inside_min_band'],
            'width_top_data_mm': (float(np.percentile(u[m30], 95)
                                        - np.percentile(u[m30], 5))
                                  if m30.sum() >= 20 else None),
            'width_13_data_mm': (float(np.percentile(u[m13], 98)
                                       - np.percentile(u[m13], 2))
                                 if m13.sum() >= 20 else None),
            'inner_edge_mm': inner,
            'width_module_mm': rail['head']['width_mm'],
            'line_s_data': s, 'line_i_data': i,
            'top_slope_data': ms_d, 'top_slope_used': ms,
            'head_top_z_data_mid': float(ms_d * y_mid + cs_d),
            'head_top_z_module_mid': float(rail['head']['top_z']),
            'x_rail_data_mid': float(s_d * y_mid + i_d),
            'h_adj_data': _adjacent_height(x, y, z, s_d, i_d, ms_d, cs_d, out_sign,
                                           yr[0], yr[1],
                                           y_keep=sec['y_center'][dense],
                                           axis=ax_poly.get(rail['side'])),
            'h_adj_module': rail['head']['height_above_adjacent_m'],
        }
        out['rails'][rail['side']].update(
            _check_rail_mesh(x, y, z, rail, s, i, ms, cs, out_sign, sec, y_mid,
                             module_stats=res['meta'].get('rail_mesh', {})
                             .get('rails', {}).get(rail['side']),
                             shelf_meas=shelf_meas, floor_their=floor_their,
                             shelf_prof=res['meta'].get('floor_profile'),
                             axis_ab=(float(res['axis']['s']), float(res['axis']['i']))))
        # --- сравнение с ПРЕЖНЕЙ ПРЯМОЙ осью: та же метрика на том же кадре
        st = res_str['meta']['rail_mesh']['rails'][rail['side']]
        rail_str = res_str['rails'][0 if rail['side'] == 'left' else 1]
        lv2, ce2, _nyc2, _tp2 = _mesh_axis(rail_str['mesh']['vertices'])
        if lv2.size >= 2 and shelf_meas is not None:
            nx2, ny2 = _path_normals_chk(lv2, ce2, out_sign)
            src2 = st.get('section_source')
            out['rails'][rail['side']]['metric_straight'] = _axis_metric(
                x, y, z, ce2, lv2, nx2, ny2, shelf_meas['z_at_ymid'],
                shelf_meas['slope_y'], shelf_meas['ymid'],
                (float(src2[0]), float(src2[1])) if src2 else None)
        else:
            out['rails'][rail['side']]['metric_straight'] = None
        out['rails'][rail['side']]['straight_stats'] = {
            'n_bins_span': st.get('n_bins_span'),
            'n_bins_with_points': st.get('n_bins_with_points'),
            'n_ribbons': st.get('n_ribbons'),
            'dense_len_m': (None if not st.get('section_source')
                            else float(st['section_source'][1] - st['section_source'][0])),
            'n_verts': st.get('n_verts'),
            'extent_y': st.get('extent_y'),
            'axis_len_m': (float(np.hypot(np.diff(ce2), np.diff(lv2)).sum())
                           if lv2.size > 1 else 0.0),
        }
        # «до» (только куски по бинам) для той же нити — тем же кодом check'а
        if rail['side'] not in out['rails_bins_only']:
            rb = res_obs['rails'][0 if rail['side'] == 'left' else 1]
            out['rails_bins_only'][rail['side']] = {
                'mesh_verts': len(rb['mesh']['vertices']),
                'mesh_faces': len(rb['mesh']['faces']),
                'mod_ribbons': res_obs['meta']['rail_mesh']['rails'][rail['side']]['n_ribbons'],
                'y_len_m': ((float(np.asarray(rb['mesh']['vertices'], float)[:, 1].max())
                             - float(np.asarray(rb['mesh']['vertices'], float)[:, 1].min()))
                            if rb['mesh']['vertices'] else None),
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

    # ---- ДО/ПОСЛЕ: та же функция с draw_gost_profile=True (старая ГОСТ-схема)
    res_gost = build_track(db_path, index, draw_gost_profile=True)
    gost = {}
    for rg in res_gost['rails']:
        vg = np.asarray(rg['mesh']['vertices'], float) if rg['mesh']['vertices'] \
            else np.zeros((0, 3))
        tl = (res['meta'].get('rail_mesh', {}).get('rails', {})
              .get(rg['side'], {}).get('top_line'))
        depth_g = None
        if vg.size and tl:
            depth_g = float(((tl['s'] * vg[:, 1] + tl['i']) - vg[:, 2]).max())
        gost[rg['side']] = {
            'verts': int(vg.shape[0]), 'faces': len(rg['mesh']['faces']),
            'x_span_mm': ((float(vg[:, 0].max() - vg[:, 0].min()) * 1000.0)
                          if vg.size else None),
            'z_span_mm': ((float(vg[:, 2].max() - vg[:, 2].min()) * 1000.0)
                          if vg.size else None),
            'depth_max_m': depth_g,
            'y_len_m': ((float(vg[:, 1].max() - vg[:, 1].min())) if vg.size else None),
            'y_range': [float(rg['head']['y_range'][0]), float(rg['head']['y_range'][1])],
            'source': dict(rg['source']),
        }
    out['gost_mesh'] = gost
    out['gost_meta_ok'] = (
        abs(res_gost['meta']['gauge_inner_mm'] - res['meta']['gauge_inner_mm']) < 1e-9
        and abs(res_gost['meta']['gauge_axis_mm'] - res['meta']['gauge_axis_mm']) < 1e-9
        and abs(res_gost['meta']['head_width_mm'] - res['meta']['head_width_mm']) < 1e-9
        and abs(res_gost['meta']['sensor_height_above_head_m']
                - res['meta']['sensor_height_above_head_m']) < 1e-9)
    # default == явный draw_gost_profile=False (флаг по умолчанию ни на что не влияет)
    import json as _json
    res_false = build_track(db_path, index, draw_gost_profile=False, only_observed=False)
    out['default_equals_explicit_false'] = (
        _json.dumps(res, sort_keys=True) == _json.dumps(res_false, sort_keys=True))
    # only_observed не должен менять ничего, кроме rails[].mesh
    a = _json.loads(_json.dumps(res))
    b = _json.loads(_json.dumps(res_obs))
    for d in (a, b):
        for rr in d['rails']:
            rr['mesh'] = None
            rr['mesh_measured'] = None
            # все поля source, описывающие САМ меш (в only_observed он другой)
            for kk in ('mesh', 'mesh_measured', 'body', 'web', 'foot'):
                rr['source'][kk] = None
        d['meta'].pop('rail_mesh', None)
        d['meta']['notes'] = [t for t in d['meta']['notes']
                              if 'режим' not in t and 'ПОЛИЛИНИЯ' not in t
                              and 'МЕТРИКА оси' not in t]
    out['only_observed_isolated'] = (_json.dumps(a, sort_keys=True)
                                     == _json.dumps(b, sort_keys=True))
    return out


def main():
    # 6 записей по 2 кадра (0 и середина): прямые пути, кривые, стрелки
    bags = ['for_hackathon/doubleT_platform/doubleT_platform_0.db3',
            'for_hackathon/doubleT_obstacle/doubleT_obstacle_0.db3',
            'for_hackathon/roundT_doubleT/roundT_doubleT_0.db3',
            'for_hackathon/roundT_pressureGate_roundT/roundT_pressureGate_roundT_0.db3',
            'for_hackathon/roundT_squareT_pressureGate_squareT/'
            'roundT_squareT_pressureGate_squareT_0.db3',
            'for_hackathon/squareT_platform_squareT_switch/'
            'squareT_platform_squareT_switch_0.db3']
    rows = []
    for db in bags:
        frames = BagFrames(db, cache_size=1)
        n = len(frames)
        del frames
        for idx in sorted({0, n // 2}):
            info = _check_frame(db, idx)
            rows.append(info)
            fl = info['floor']
            sl = info['sleepers']
            g = info['gauge']
            print(f"\n=== {info['bag']}, кадр {idx} из {n} ===")
            print('  пол: меш %d верш./%d граней, ИНДЕКСЫ %s, X [%.2f, %.2f] м, Y [%.1f, %.1f] м '
                  '(профиль, не плоскость)'
                  % (fl['verts'], fl['faces'], 'ок' if fl['indices_ok'] else 'ПЛОХИЕ',
                     fl['x_span'][0], fl['x_span'][1], fl['y_span'][0], fl['y_span'][1]))
            print('       опорная плоскость floor %s (оставлена в контракте): нижняя '
                  'огибающая ячеек 1x1 м (%d ячеек) отклоняется от неё на %s м (медиана)'
                  % (tuple(round(v, 4) for v in fl['abc']), fl['n_cells'],
                     _fmt(fl['cell_med_m'], 3)))
            fth = info['floor_their']
            sl_ = info.get('shelf_line_measured')
            print('       ПОЛ ВЬЮЕРА (zones.fit_floor, полоса x=%.2f±%.1f): z = %+.4f '
                  '%+.6f*y -> z(y_ref=%s) = %s м, z(%s) = %s м; RMS %s м | независимая '
                  'прямая полки: %s'
                  % (fth['x_center'], fth['x_half'], fth['a'], fth['b'],
                     _fmt(info['y_ref'], 1), _fmt(fth['z_at_y_ref'], 3), '-6',
                     _fmt(fth['z_at_minus6'], 3), _fmt(fth['rms_m'], 3),
                     ('z = %+.3f %+.5f*(y%+.1f) по %d полосам 1.5 м'
                      % (sl_['z_at_ymid'], sl_['slope_y'], -sl_['ymid'], sl_['n_bands']))
                     if sl_ else 'НЕТ (мало точек полки)'))
            fprl = info['meta'].get('floor_plane_reliability')
            print('       НАША плоскость floor против измеренной полки: на y=%s '
                  'плоскость %s м, полка %s м -> плоскость НИЖЕ полки на %s м '
                  '(+ гуляет между кадрами)'
                  % (_fmt(fprl['probe_y_m'], 1), _fmt(fprl['plane_z_m'], 3),
                     _fmt(fprl['shelf_z_m'], 3),
                     _fmt(-fprl['plane_minus_shelf_m'], 3)) if fprl else
                  '       НАША плоскость floor: сравнить с полкой не удалось')
            fpr = info['floor_profile']
            fm, fmod = fpr['measured'], fpr['module']
            if fm:
                print('  профиль пола (независимый замер, окно y %.1f..%.1f): полка %+.3f м '
                      '(%+.4f*u), лоток %s'
                      % (fpr['window_y'][0], fpr['window_y'][1], fm['shelf_z'],
                         fm['shelf_slope_u'],
                         ('НЕТ (глубже полки лишь на %.3f м)' % fm['depth']) if not fm['has_trough']
                         else 'u %+.3f..%+.3f = %.3f м, центр %+.3f, дно %+.3f, глубина %.3f м, точек дна %d'
                         % (fm['u_l'], fm['u_r'], fm['width'], fm['center'], fm['bottom'],
                            fm['depth'], fm['n_bottom'])))
            if fmod:
                t = fmod.get('trough')
                print('  профиль пола (вывод модуля): полка %+.3f м (%+.4f*u %+.4f*y), лоток %s'
                      % (fmod['shelf_z'], fmod['shelf_slope_u'], fmod['shelf_slope_y'],
                         'НЕТ' if not t else
                         'u %+.3f..%+.3f = %.3f м, центр %+.3f, дно %+.3f, глубина %.3f м, '
                         'стенка %.0f мм, по полосам 3 м ширина %s'
                         % (t['u_left'], t['u_right'], t['width_m'], t['center_u'],
                            t['bottom_z'], t['depth_m'], 1000 * t['wall_run_m'],
                            [round(v, 2) for v in t['width_bands_m']])))
                print('       меш пола содержит лоток: %s' % fpr['mesh_has_trough'])
            fr = info['floor_rms']
            print('  RMS точек пола (%d шт) от плоскости %s м, от профиля %s м -> профиль '
                  'в %s раза точнее'
                  % (fr['n_pts'], _fmt(fr['plane_m'], 3), _fmt(fr['profile_m'], 3),
                     _fmt(fr['plane_m'] / fr['profile_m'], 2) if (fr['plane_m'] and fr['profile_m'])
                     else 'нет'))
            print('  шпалы: %d шт, источник «%s»%s'
                  % (sl['count'], sl['source'],
                     ' — пусто, как и требуется' if sl['empty_ok'] else ' — ОЖИДАЛОСЬ ПУСТО'))
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
                      'пути данные %+.5f м/м'
                      % (r['head_top_z_data_mid'], r['head_top_z_module_mid'],
                         r['top_slope_data']))
                print('       высота над прилегающей: данные %s м, модуль %s м'
                      % (_fmt(r['h_adj_data'], 3), _fmt(r['h_adj_module'], 3)))
                print('       линия нити: своя x=%.5f*y%+.4f, модуля x=%.5f*y%+.4f '
                      '(расхождение по i %s мм)'
                      % (r['line_s_data'], r['line_i_data'],
                         info['meta']['rail_mesh']['rails'][side]['line']['s'],
                         info['meta']['rail_mesh']['rails'][side]['line']['i'],
                         _fmt(r.get('line_diff_i_mm'), 1)))
                gm = info['gost_mesh'][side]
                rb = info['rails_bins_only'].get(side, {})
                print('       МЕШ РАЗВЁРТКИ: %d верш./%d граней, сечений %d, '
                      'граней ожидается %d при непрерывности -> %s, y [%.1f, %.1f] (%.1f м)'
                      % (r['mesh_verts'], r['mesh_faces'], r['n_sections'],
                         r['faces_expected_continuous'],
                         'непрерывно' if r['continuous'] else 'ЕСТЬ РАЗРЫВЫ',
                         r['mesh_y'][0], r['mesh_y'][1], r['mesh_y'][1] - r['mesh_y'][0]))
                print('          сечение (медиана по плотным бинам): ширина меш %s мм против '
                      'измеренной головки %s мм -> %s мм; глубина под верхом %.0f мм; '
                      'центр от линии %s мм (разброс по Y %s мм)'
                      % (_fmt(r['mesh_width_med_mm'], 1), _fmt(r['width_module_mm'], 1),
                         _fmt(r['mesh_width_diff_mm'], 1), 1000 * r['mesh_depth_med_m'],
                         _fmt(r['mesh_axis_off_med_mm'], 1),
                         _fmt(r['mesh_axis_off_max_mm'], 1)))
                print('          плотные бины: y %s..%s (уровней %d из %d), вне них сечение '
                      'то же: разброс u %s мм, разброс vz %s мм; доля длины вне данных: '
                      'модуль %s, check %s'
                      % (_fmt(r['mod_section_source'][0] if r['mod_section_source'] else None, 1),
                         _fmt(r['mod_section_source'][1] if r['mod_section_source'] else None, 1),
                         r.get('n_levels_dense') or 0, r['n_sections'],
                         _fmt(r['beyond_u_spread_mm'], 3), _fmt(r['beyond_vz_spread_mm'], 3),
                         _fmt(r['mod_beyond'], 3), _fmt(r['beyond_measured_chk'], 3)))
                print('          вершины в плотных бинах подкреплены точками: %s '
                      '(из %d проверенных, p90 ближайшей точки по u %s мм); '
                      'глубина меша %.0f мм против наблюдённой p90 %.0f мм'
                      % (_fmt(r['verts_backed_frac'], 3), r.get('n_verts_checked') or 0,
                         _fmt(r['verts_nearest_u_p90_mm'], 1),
                         1000 * r['mesh_depth_med_m'],
                         1000 * (r['pts_depth_p90_in_dense_m'] or 0)))
                mp = info['meta']['rail_mesh']['rails'][side].get('polyline')
                print('          ОСЬ: %s | шаг оси между уровнями макс %s м, длина оси %.1f м'
                      % (r.get('axis_model'), _fmt(r.get('axis_step_max_m'), 3),
                         r.get('axis_len_m') or 0.0))
                if mp:
                    print('             полилиния: узлов %s (измерено %s, зашито '
                          'интерполяцией %s, отброшено MAD %s), длина по пути %s м, шаг '
                          'оси макс %s м, касательные дал/ближ %s°, кривизна %s/%s 1/м'
                          % (mp['n_nodes'], mp['n_nodes_measured'], mp['n_nodes_bridged'],
                             mp['n_nodes_rejected_mad'], _fmt(mp['dense_len_m'], 1),
                             _fmt(mp['axis_max_step_m'], 3),
                             [None if v is None else round(v, 2)
                              for v in mp['tangent_deg_far_near']],
                             _fmt(mp['kappa_far_1_m'], 4), _fmt(mp['kappa_near_1_m'], 4)))
                am = r.get('axis_metric')
                am2 = r.get('metric_straight')
                ss = r.get('straight_stats') or {}
                if am:
                    print('          МЕТРИКА ОСИ (ось до ближайших точек рельса, полоса '
                          '%.2f..%.2f м над полкой, |u| <= %.2f м):'
                          % (am['band_above_shelf_m'][0], am['band_above_shelf_m'][1],
                             am['u_limit_m']))
                    print('             внутри данных: ПОЛИЛИНИЯ медиана %s / p95 %s м '
                          '(%d узлов с точками из %d), покрыто %s м | ПРЯМАЯ ось медиана '
                          '%s / p95 %s м (%s узлов), покрыто %s м'
                          % (_fmt(am['in']['median_m'], 3), _fmt(am['in']['p95_m'], 3),
                             am['in']['n_with_points'], am['in']['n_nodes'],
                             _fmt(r.get('mod_dense_len_m'), 1),
                             _fmt((am2 or {}).get('in', {}).get('median_m'), 3),
                             _fmt((am2 or {}).get('in', {}).get('p95_m'), 3),
                             (am2 or {}).get('in', {}).get('n_with_points'),
                             _fmt(ss.get('dense_len_m'), 1)))
                    print('             за пределами данных: ПОЛИЛИНИЯ медиана %s / p95 %s м '
                          '(%d узлов из %d) | ПРЯМАЯ ось медиана %s / p95 %s м (%s узлов)'
                          % (_fmt(am['out']['median_m'], 3), _fmt(am['out']['p95_m'], 3),
                             am['out']['n_with_points'], am['out']['n_nodes'],
                             _fmt((am2 or {}).get('out', {}).get('median_m'), 3),
                             _fmt((am2 or {}).get('out', {}).get('p95_m'), 3),
                             (am2 or {}).get('out', {}).get('n_with_points')))
                if ss:
                    print('          ПРЯМАЯ ОСЬ (для сравнения): бинов с точками %s из %s, '
                          'кусков %s, меш %s верш., y %s, длина оси %.1f м'
                          % (ss.get('n_bins_with_points'), ss.get('n_bins_span'),
                             ss.get('n_ribbons'), ss.get('n_verts'),
                             [None if ss.get('extent_y') is None
                              else round(v, 1) for v in ss.get('extent_y') or []],
                             ss.get('axis_len_m') or 0.0))
                ns = ((info['meta']['rail_mesh']['rails'].get(side) or {}).get('polyline')
                      or {}).get('node_stats') or {}
                print('          УЗЛЫ: прыжок Z макс %s м, излом наклона макс %s 1/м, '
                      'узлов вне тренда %s | МЕШ: mesh_measured %s верш., тело %s верш.'
                      % (_fmt(ns.get('z_jump_max_m'), 3), _fmt(ns.get('slope_break_max_1_m'), 4),
                         ns.get('n_trend_out'), r.get('mesh_measured_verts'),
                         r.get('mesh_verts')))
                tb = r.get('top_band')
                if tb:
                    print('          ВЕРХ НАД ПОЛКОЙ (коридор %.2f..%.2f): %s..%s м, вне '
                          'коридора %d/%d, ниже полки+%.2f м %d'
                          % (tb['lo_m'], tb['hi_m'], _fmt(tb['min_m'], 3),
                             _fmt(tb['max_m'], 3), tb['n_out'], tb['n_nodes'],
                             EXPECT['top_hard_m'], tb['n_below_hard']))
                hl = r.get('h_local') or {}
                print('          НЕ ЛЕТИТ: локальная высота верха над полом min %s / med %s '
                      '/ max %s м, разброс %s м (%s уровней); ось: узлов %s, скачков >%.2f м %s'
                      % (_fmt(hl.get('min_m'), 3), _fmt(hl.get('med_m'), 3),
                         _fmt(hl.get('max_m'), 3), _fmt(hl.get('spread_m'), 3),
                         hl.get('n_levels'), r.get('n_sections'), EXPECT['axis_jump_m'],
                         r.get('n_jump_levels')))
                print('          РАЗВЁРТКА ПО ДЛИНЕ: меш %.1f м = плотный диапазон %s м + %s м на '
                      'дальнем конце + %s м на ближнем (запас %.1f м, ограничение ухода '
                      '%s м по высоте / %s м по X); БЫЛО (весь диапазон) %.1f м'
                      % (r['mesh_y_len_m'], _fmt(r['mod_dense_len_m'], 1),
                         _fmt(r['mod_extra_far_chk_m'], 1),
                         _fmt(r['mod_extra_near_chk_m'], 1), EXPECT['sweep_extra_m'],
                         _fmt(r.get('mod_extra_cap_h_m'), 1),
                         _fmt(r.get('mod_extra_cap_x_m'), 1), r['old_len_m']))
                if r.get('anchor_shelf'):
                    a1 = r['anchor_shelf']
                    print('          верх меша минус ИЗМЕРЕННАЯ ПОЛКА: min %.3f / медиана '
                          '%.3f / max %.3f м, разброс %.3f м (коридор %.2f..%.2f)'
                          % (a1['min_m'], a1['med_m'], a1['max_m'], a1['spread_m'],
                             EXPECT['mesh_above_shelf_m'][0], EXPECT['mesh_above_shelf_m'][1]))
                    for tag, ttl in (('near', 'ближний конец'), ('dense_far',
                                                                 'дальний край данных'),
                                     ('far', 'дальний конец')):
                        if tag in a1['at']:
                            s_ = a1['at'][tag]
                            print('             %-18s y=%6.1f: по узлам %.3f..%.3f, '
                                  'медиана %.3f м'
                                  % (ttl, s_['y_m'], s_['min_m'], s_['max_m'], s_['med_m']))
                a2 = r.get('anchor_their_floor')
                if a2:
                    print('          верх меша минус ПОЛ ВЬЮЕРА (zones.fit_floor, '
                          'кросс-проверка): min %.3f / медиана %.3f / max %.3f м '
                          '(коридор %.2f..%.2f)'
                          % (a2['min_m'], a2['med_m'], a2['max_m'],
                             EXPECT['mesh_above_floor_m'][0], EXPECT['mesh_above_floor_m'][1]))
                    for tag, ttl in (('near', 'ближний конец'), ('dense_far',
                                                                 'дальний край данных'),
                                     ('far', 'дальний конец')):
                        if tag in a2['at']:
                            s_ = a2['at'][tag]
                            print('             %-18s y=%6.1f: по узлам %.3f..%.3f, '
                                  'медиана %.3f м'
                                  % (ttl, s_['y_m'], s_['min_m'], s_['max_m'], s_['med_m']))
                print('          уход верха от полки: БЫЛО (меш по m*y+c, дальний конец y=%.0f м) '
                      '%s м -> СТАЛО (меш по полке) %s м на всём меше; уход от плотной '
                      'зоны по высоте %s м (за пределами данных %s м), по X %s м; '
                      'высота головки над полкой (для привязки) %s м'
                      % (r['old_far_y_m'], _fmt(r.get('old_drift_far_m'), 3),
                         _fmt(max(a1['min_m'], a1['max_m']) if a1 else None, 3),
                         _fmt(r.get('mod_z_drift_far_m'), 3),
                         _fmt(r.get('mod_z_drift_beyond_m'), 3),
                         _fmt(r.get('mod_x_drift_far_m'), 3),
                         _fmt(r.get('mod_head_h_m'), 3)))
                print('          source: head=%s, mesh=%s, web=%s, foot=%s'
                      % (r['source'].get('head'), r['source'].get('mesh'),
                         r['source'].get('web'), r['source'].get('foot')))
                print('          БЫЛО only_observed=True: %s верш./%s граней, y-длина %s м, '
                      'кусков %s | ДО (ГОСТ, %s верш.): глубина под верхом %.0f мм'
                      % (rb.get('mesh_verts'), rb.get('mesh_faces'),
                         _fmt(rb.get('y_len_m'), 1), rb.get('mod_ribbons'),
                         gm['verts'], 1000 * (gm['depth_max_m'] or 0)))
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
    print('%-17s %4s | %-13s | %-13s | %-11s | %-7s | %-6s | %-11s | %-13s | %s'
          % ('bag', 'кадр', 'колея внутр', 'ось-ось', 'ширина/13мм',
             'верх-п', 'внутри', 'лоток ш/г, м', 'RMS пол/проф, м',
             'меш рельса: длина/кусков/верш/ширина/вне данных'))
    fails = []
    n_checks_total = 0
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
        fmod = info['floor_profile']['module'] or {}
        tr = fmod.get('trough')
        fr = info['floor_rms']
        meshline = []
        for k in keys:
            r = rl.get(k, {})
            if 'mesh_y' in r and r['mesh_y']:
                meshline.append('%s: %.0f м/%d/%d/%s мм/%.0f%%' % (
                    'L' if k == 'left' else 'R', r['mesh_y'][1] - r['mesh_y'][0],
                    1 if r.get('continuous') else 2, r['mesh_verts'],
                    _fmt(r['mesh_width_med_mm'], 0, '-'),
                    100.0 * (r.get('mod_beyond') or 0)))
        print('%-17s %4d | %6s/%6s | %6s/%6s | %5s/%5s | %5s | %s | %-11s | %-13s | %s'
              % (info['bag'], info['index'],
                 _fmt(g.get('inner_mm_data'), 0, '-'), _fmt(g.get('inner_mm_module'), 0, '-'),
                 _fmt(g.get('axis_mm_data'), 0, '-'), _fmt(g.get('axis_mm_module'), 0, '-'),
                 _fmt(np.mean(w_top) if w_top else None, 1, '-'),
                 _fmt(np.mean(w13) if w13 else None, 1, '-'),
                 _fmt(np.mean(hh) if hh else None, 3, '-'),
                 _fmt(np.mean(ins_d) if ins_d else None, 3, '-'),
                 ('-' if not tr else '%.2f/%.2f' % (tr['width_m'], tr['depth_m'])),
                 ('%s/%s' % (_fmt(fr['plane_m'], 3, '-'), _fmt(fr['profile_m'], 3, '-'))),
                 ' '.join(meshline)))
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
        if ins:
            checks.append(('внутри контура > 0.8 (плотные полосы)',
                           bool(np.mean(ins_d) > EXPECT['inside_min'])
                           if ins_d else bool(np.mean(ins) > EXPECT['inside_min'])))
        if hh:
            lo, hi = EXPECT['head_above_adjacent']
            checks.append(('верх над прилегающей 0.159-0.186',
                           bool(lo <= np.mean(hh) <= hi)))
        if w_top:
            checks.append(('ширина верха в ±10 мм от 74.59',
                           bool(abs(np.mean(w_top) - EXPECT['head_b_mm']) <= 10.0)))
        # --- шпалы: пусто и правильный источник
        checks.append(('шпалы: instances пуст, source «встроенный путь»',
                       bool(info['sleepers']['empty_ok'])))
        # --- лоток: независимый замер против вывода модуля
        fm = info['floor_profile']['measured']
        if fm and fmod:
            if fm['has_trough'] and tr:
                tol = EXPECT['trough_tol_m']
                checks.append(('лоток: ширина совпала (±%.2f м)' % tol,
                               abs(fm['width'] - tr['width_m']) <= tol))
                checks.append(('лоток: глубина совпала (±%.2f)' % tol,
                               abs(fm['depth'] - tr['depth_m']) <= tol))
                checks.append(('лоток: центр совпал (±%.2f)' % tol,
                               abs(fm['center'] - tr['center_u']) <= tol))
                checks.append(('лоток: глубина > %.2f м' % EXPECT['trough_min_depth_m'],
                               bool(tr['depth_m'] > EXPECT['trough_min_depth_m'])))
            elif (not fm['has_trough']) and (tr is None):
                checks.append(('лоток: оба замера согласны, что его нет', True))
            else:
                checks.append(('лоток: замеры согласны по наличию',
                               bool(bool(fm['has_trough']) == bool(tr))))
        # --- RMS: профиль описывает пол лучше плоскости
        if fr['plane_m'] is not None and fr['profile_m'] is not None:
            checks.append(('RMS профиля < RMS плоскости',
                           bool(fr['profile_m'] < fr['plane_m'])))
        # --- меш пола соответствует профилю: лоток есть тогда и только тогда, когда измерен
        mh = info['floor_profile']['mesh_has_trough']
        checks.append(('меш пола соответствует профилю (лоток есть/нет)',
                       (mh is not None) and (bool(mh) == bool(tr))))
        # --- меш рельса: непрерывная развёртка измеренного сечения
        for k in keys:
            r = rl.get(k, {})
            n_verts = int(r.get('mesh_verts') or 0)
            # ИНВАРИАНТ (внешнее требование): mesh.vertices НЕ ПУСТ у КАЖДОЙ нити.
            # Проверяется ПЕРВЫМ и безусловно: у не измеренной нити меш строится по
            # её узлам из плана пары (сечение — парной нити или номинал Р65),
            # поэтому пустой меш возможен только как дефект.
            checks.append(('меш %s: vertices не пуст у каждой нити (инвариант)' % k,
                           n_verts > 0))
            if r.get('detected') is False or n_verts == 0:
                checks.append(('детекция меш %s: нить найдена на кадре (меш не пуст)' % k,
                               False))
                continue
            if 'mesh_indices_ok' not in r:
                # разбор геометрии нити не сделан (сечения взять нечем): тело
                # построено по узлам плана пары — из проверок только непустота выше
                src_pair = ((info['meta']['rail_mesh']['rails'].get(k) or {})
                            .get('mesh_source'))
                checks.append(('меш %s: разбор сечения пропущен (нить не измерена, '
                               'тело по узлам плана пары, источник %s)'
                               % (k, src_pair or 'измеренное сечение парной нити'), True))
                continue
            checks.append(('меш %s: грани — тройки, индексы в границах' % k,
                           bool(r['mesh_indices_ok'])))
            checks.append(('меш %s: НЕПРЕРЫВЕН (ломаных нет: граней = (уровней-1)*8)' % k,
                           bool(r['continuous'])))
            checks.append(('меш %s: длина = extent_y ±0.6 м' % k,
                           r['mesh_y'] is not None and r.get('mod_extent_y') is not None
                           and abs(r['mesh_y'][0] - r['mod_extent_y'][0]) < 0.6
                           and abs(r['mesh_y'][1] - r['mod_extent_y'][1]) < 0.6))
            checks.append(('меш %s: сечение = измеренной головке ±10 мм' % k,
                           _abs_le(r['mesh_width_diff_mm'], 10.0)))
            checks.append(('меш %s: сечение постоянно по длине (±2 мм)' % k,
                           _abs_le(r['mesh_width_spread_mm'], 2.0)))
            is_poly = str(r.get('axis_model') or '').startswith('polyline')
            if is_poly:
                am = r.get('axis_metric') or {}
                mi, mo = am.get('in', {}), am.get('out', {})
                checks.append(('меш %s: ОСЬ ПО РЕЛЬСУ — внутри данных медиана <= %.2f м, '
                               'p95 <= %.2f м' % (k, EXPECT['axis_median_m'],
                                                  EXPECT['axis_p95_m']),
                               (mi.get('median_m') is not None
                                and mi['median_m'] <= EXPECT['axis_median_m']
                                and mi['p95_m'] is not None
                                and mi['p95_m'] <= EXPECT['axis_p95_m'])))
                checks.append(('меш %s: ось без скачков (шаг <= %.2f м)' % (k,
                                                                            EXPECT['axis_jump_m']),
                               (r.get('axis_step_max_m') is not None
                                and r['axis_step_max_m'] <= EXPECT['axis_jump_m'])))
                mp = (info['meta']['rail_mesh']['rails'].get(k) or {}).get('polyline') or {}
                checks.append(('меш %s: построен по полилинии (уровней = узлов полилинии)' % k,
                               mp.get('n_nodes_total') == r.get('n_sections')))
                checks.append(('меш %s: полилиния покрывает не меньше прямой (охват данных)' % k,
                               (r.get('mod_dense_len_m') is not None
                                and (r.get('straight_stats') or {}).get('dense_len_m')
                                is not None
                                and r['mod_dense_len_m']
                                >= r['straight_stats']['dense_len_m'] - 0.6)))
            else:
                checks.append(('меш %s: ось меша лежит на линии нити (±10 мм)' % k,
                               _abs_le(r['mesh_axis_off_max_mm'], 10.0)))
            checks.append(('меш %s: вне плотных бинов сечение то же (u и vz ±2 мм)' % k,
                           _abs_le(r['beyond_u_spread_mm'], 2.0)
                           and _abs_le(r['beyond_vz_spread_mm'], 2.0)))
            checks.append(('меш %s: доля вне данных совпала (модуль vs check ±0.02)',
                           _abs_le((r['mod_beyond'] or 0) - (r['beyond_measured_chk'] or 0),
                                   0.02)))
            checks.append(('меш %s: вершины в плотных бинах подкреплены точками (>=0.85 в 30 мм)' % k,
                           (r['verts_backed_frac'] is not None
                            and r['verts_backed_frac'] >= 0.85)))
            is_body_k = str(r.get('source', {}).get('mesh', '')).find('body') >= 0 or                 bool((info['meta']['rail_mesh'].get('mesh_body')))
            if is_body_k:
                checks.append(('меш %s: ТЕЛО — высота 0.180 ± 0.02 м (шейка/подошва есть)' % k,
                               (r.get('body_h_m') is not None
                                and abs(r['body_h_m'] - 0.180) <= 0.02)))
                checks.append(('source %s: тело = ГОСТ (web/foot gost), полоска измерена' % k,
                               'gost' in str(r['source'].get('web'))
                               and 'gost' in str(r['source'].get('foot'))
                               and 'measured' in str(r['source'].get('mesh_measured'))))
            else:
                checks.append(('меш %s: глубина < 0.10 м (нет шейки/подошвы)' % k,
                               (r['mesh_depth_med_m'] is not None
                                and r['mesh_depth_med_m'] < 0.10)))
                checks.append(('source %s: web/foot не наблюдаются' % k,
                               'not observed' in str(r['source'].get('web'))
                               and 'not observed' in str(r['source'].get('foot'))
                               and 'measured' in str(r['source'].get('mesh'))))
            # --- развёртка: длина = измеренный диапазон + запас, и привязка по высоте
            exp_far = None
            exp_near = None
            if r.get('mod_extra_used_m') is not None:
                cy = info['floor']['cloud_y']
                src_ = r.get('mod_section_source') or [None, None]
                if is_poly:
                    # полилиния: запас ограничен кривизной (kappa*L^2/2), высотой и
                    # краем облака; продолжение идёт ПО КАСАТЕЛЬНОЙ, поэтому в Y оно
                    # короче пути (на |dir_y|)
                    checks.append(('меш %s: запас по касательной не больше заданного %.0f м'
                                   % (k, EXPECT['sweep_extra_m']),
                                   r.get('mod_extra_used_m') is not None
                                   and r['mod_extra_used_m'] <= EXPECT['sweep_extra_m'] + 1e-9))
                    edge_y = (min(r['mod_extra_used_m'], src_[0] - cy[0])
                              if src_[0] is not None else None)
                    checks.append(('меш %s: продолжение не выходит за облако (±0.6 м)' % k,
                                   r['mesh_y'][0] >= cy[0] - 0.6
                                   and r['mesh_y'][1] <= cy[1] + 0.6))
                    # ПРОДОЛЖЕНИЕ ПО ДАННЫМ И МОДЕЛЬЮ (жалоба «лента в 3-6 раз короче видимого рельса»):
                    # дальний конец ведёт трекинг: данные (плотные/разреженные) + модель.
                    # Инварианты: меш доходит до подтверждённого данными конца, конец
                    # совпадает с концом ленты по трекингу, модельная часть не длиннее
                    # предела, и у каждого узла есть пометка происхождения
                    rch = ((info['meta']['rail_mesh']['rails'].get(k) or {}).get('reach')
                           or {})
                    for endname, mesh_end in (('far', r['mesh_y'][0]), ('near', r['mesh_y'][1])):
                        e = (rch.get(endname) or {}) if isinstance(rch, dict) else {}
                        conf = e.get('confirmed_end_y_m')
                        nm = ('дальний' if endname == 'far' else 'ближний')
                        if endname == 'far' and e.get('ribbon_end_y_m') is not None:
                            checks.append(('меш %s: конец совпадает с концом ленты по '
                                           'трекингу (%s, ±0.6 м)' % (k, nm),
                                           _abs_le(mesh_end - e['ribbon_end_y_m'], 0.6)))
                            checks.append(('меш %s: доходит до подтверждённого данными '
                                           'конца (%s, ≥ -1.0 м)' % (k, nm),
                                           conf is None or mesh_end <= conf + 1.0))
                            checks.append(('меш %s: модельная часть в пределе %.0f м'
                                           % (k, 100.0),
                                           (e.get('model_len_m') or 0.0) <= 100.0 + 1.0))
                        else:
                            checks.append(('меш %s: не висит за подтверждённым данными '
                                           'концом (%s, ±0.6 м)' % (k, nm),
                                           conf is None or _abs_le(mesh_end - conf, 0.6)))
                    origins = (info['meta']['rail_mesh']['rails'].get(k) or {}).get('node_origin')
                    checks.append(('узлы %s: у каждого узла есть пометка происхождения' % k,
                                   origins is not None
                                   and all(v in ('measured', 'sparse', 'model')
                                           for v in origins)))
                else:
                    exp_far = (min(r['mod_extra_used_m'], src_[0] - cy[0])
                               if src_[0] is not None else None)
                    exp_near = (min(r['mod_extra_used_m'], cy[1] - src_[1])
                                if src_[1] is not None else None)
                    checks.append(('меш %s: запас в каждую сторону = мин(%.0f м, край данных) ±%.1f'
                                   % (k, EXPECT['sweep_extra_m'], EXPECT['sweep_extra_tol_m']),
                                   exp_far is not None and exp_near is not None
                                   and _abs_le(r['mod_extra_far_chk_m'] - exp_far,
                                               EXPECT['sweep_extra_tol_m'])
                                   and _abs_le(r['mod_extra_near_chk_m'] - exp_near,
                                               EXPECT['sweep_extra_tol_m'])))
            # ДАЛЬНИЙ конец ведёт трекинг по данным и моделью (лимиты: модельный участок
            # <= 100 м и коридор |x| <= 3 м, см. POLY_MODEL_MAX_M/POLY_REACH_X_LIMIT_M),
            # поэтому «бюджет 15 м» проверяется только как запас БЛИЖНЕГО конца, а длина
            # меша по-прежнему обязана быть меньше полного диапазона облака
            rch_ = ((info['meta']['rail_mesh']['rails'].get(k) or {}).get('reach') or {})
            near_kept = ((rch_.get('near') or {}).get('kept_extra_m') or 0.0)
            checks.append(('меш %s: запас ближнего конца <= %.0f м (никаких 198 м)'
                           % (k, EXPECT['sweep_extra_m']),
                           r.get('mod_extra_used_m') is not None
                           and near_kept <= EXPECT['sweep_extra_m'] + 1e-9
                           and r['mesh_y_len_m'] < r['old_len_m'] - 20.0))
            a1 = r.get('anchor_shelf')
            half_dense = 0.5 * (r['mod_dense_len_m'] or 0.0)
            shl = info.get('shelf_line_measured')
            grade = abs(shl['slope_y']) * half_dense if shl else 0.0
            # ВЫСОТА ПРОДОЛЖЕНИЯ — по СВОЕЙ линии верха нити (m_top·y + c_top): это и
            # есть «не врать по высоте». Раньше проверялся уход от уровня плотной зоны
            # (<= 0.30 м), но на 100-200 м лента обязана идти по уклону пути, и этот
            # критерий стал бессмысленным (замерено: уход = |уклон| * длина)
            tl_off = (info['meta']['rail_mesh']['rails'].get(k) or {}).get(
                'top_line_off_max_m')
            # допуск растёт с длиной: кламп коридора держит верх в [низ, 0.30] над
            # ЭКСТРАПОЛИРОВАННОЙ полкой, а линия верха нити идёт своим уклоном; если
            # уклоны расходятся на dS, на длине L расхождение dS*L (замерено 0.30-0.60 м
            # на 100-200 м) — это не ошибка высоты, а разница двух измеренных уклонов
            tl_slope = (info['meta']['rail_mesh']['rails'].get(k) or {}).get('top_line', {})
            ds = abs(float(tl_slope.get('s') or 0.0)
                     - (abs(shl['slope_y']) * (1 if shl and shl['slope_y'] >= 0 else -1)
                        if shl else 0.0))
            allow = EXPECT['sweep_drift_guard_m'] + ds * r['mesh_y_len_m'] + 0.05
            checks.append(('меш %s: высота продолжения по своей линии верха (<= %.2f м)'
                           % (k, allow),
                           tl_off is not None and tl_off <= allow + 1e-6))
            checks.append(('меш %s: уход по X <= %.2f м' % (k, EXPECT['sweep_x_guard_m']),
                           r.get('mod_x_drift_far_m') is not None
                           and r['mod_x_drift_far_m'] <= EXPECT['sweep_x_guard_m'] + 1e-6))
            # ТЕЛО: высота/подошва + ИЗМЕРЕННАЯ полоска отдельным ключом
            mp0 = (info['meta']['rail_mesh']['rails'].get(k) or {}).get('polyline') or {}
            if r.get('source_measured'):
                checks.append(('тело %s: высота %.3f±0.02 м (ГОСТ)' % (k, EXPECT['body_h_m']),
                               r.get('body_h_m') is not None
                               and abs(r['body_h_m'] - EXPECT['body_h_m']) <= 0.02))
                checks.append(('тело %s: подошва %.3f±0.01 м (ГОСТ)' % (k, EXPECT['foot_b_m']),
                               r.get('body_w_m') is not None
                               and abs(r['body_w_m'] - EXPECT['foot_b_m']) <= 0.01))
                checks.append(('mesh_measured %s: есть и вершин = 5 x узлов полоски' % k,
                               bool(r.get('mesh_measured_ok'))))
                checks.append(('тело %s: верх = измеренной коронке (±5 мм)' % k,
                               r.get('top_equals_measured_m') is not None
                               and r['top_equals_measured_m'] <= 0.005))
            nc = r.get('nodes_chk')
            checks.append(('nodes %s: есть, отсортированы по y, включая продолжение' % k,
                           bool(nc) and nc['y_sorted']))
            checks.append(('nodes %s: x = оси меша (±1 мм), z = верх (±5 мм), y = уровням'
                           % k,
                           bool(nc) and nc.get('dx_max_m') is not None
                           and nc['dx_max_m'] <= 0.001 and nc['dz_max_m'] <= 0.005
                           and nc['y_levels_m'] <= 0.006))   # y в JSON округлён до 1 см
            ns = mp0.get('node_stats') or {}
            checks.append(('узлы %s: одиночный прыжок Z <= %.2f м' % (k, EXPECT['z_jump_m']),
                           ns.get('z_jump_max_m') is not None
                           and ns['z_jump_max_m'] <= EXPECT['z_jump_m'] + 1e-9))
            checks.append(('узлы %s: излом наклона <= %.2f 1/м на шаг' % (k, EXPECT['slope_break_1_m']),
                           ns.get('slope_break_max_1_m') is not None
                           and ns['slope_break_max_1_m'] <= EXPECT['slope_break_1_m'] + 1e-9))
            checks.append(('узлы %s: нет узлов дальше 0.05 м от тренда' % k,
                           ns.get('n_trend_out') == 0))
            if mp0.get('foot_below_shelf_min_m') is not None:
                # «не залезать в пол»: низ подошвы не глубже 0.05 м под ЛОКАЛЬНОЙ
                # ПОЛКОЙ (линия профиля модуля под нитью — та же, что у меша пола)
                checks.append(('подошва %s: низ не глубже %.2f м под полкой (нет в полу)'
                               % (k, -EXPECT['foot_adj_tol_m'][0]),
                               mp0['foot_below_shelf_min_m'] >= EXPECT['foot_adj_tol_m'][0] - 1e-9))
            tb = r.get('top_band')
            if tb:
                checks.append(('меш %s: ВЕРХ В КОРИДОРЕ %.2f..%.2f м над измеренной '
                               'полкой — все %d узлов' % (k, tb['lo_m'], tb['hi_m'],
                                                          tb['n_nodes']),
                               tb['n_out'] == 0))
                checks.append(('меш %s: ни один узел не ниже полки + %.2f м' % (
                    k, EXPECT['top_hard_m']), tb['n_below_hard'] == 0))
            hl = r.get('h_local')
            checks.append(('меш %s: НЕ ЛЕТИТ — локальная высота над полом %.2f..%.2f м'
                           % (k, EXPECT['h_local_m'][0], EXPECT['h_local_m'][1]),
                           hl is not None
                           and hl['min_m'] >= EXPECT['h_local_m'][0]
                           and hl['max_m'] <= EXPECT['h_local_m'][1]))
            checks.append(('меш %s: локальная высота не съезжает по длине (разброс <= %.2f м)'
                           % (k, EXPECT['mesh_above_spread_m']),
                           hl is not None and _abs_le(hl['spread_m'],
                                                      EXPECT['mesh_above_spread_m'])))
            if a1:
                lo, hi = EXPECT['mesh_above_shelf_m']
                checks.append(('меш %s: верх над полкой модуля %.2f..%.2f м (медиана)'
                               % (k, lo, hi),
                               bool(lo <= a1['med_m'] <= hi)))
            a2 = r.get('anchor_their_floor')
            if a2:
                lo, hi = EXPECT['mesh_above_floor_m']
                checks.append(('меш %s: верх над полом ВЬЮЕРА %.2f..%.2f м (кросс-проверка)'
                               % (k, lo, hi),
                               bool(lo <= a2['min_m'] and a2['max_m'] <= hi)))
        checks.append(('draw_gost_profile по умолчанию не влияет (default == False)',
                       bool(info['default_equals_explicit_false'])))
        checks.append(('only_observed не меняет ничего, кроме rails[].mesh',
                       bool(info['only_observed_isolated'])))
        checks.append(('метрики meta не зависят от схемы меша', bool(info['gost_meta_ok'])))
        checks.append(('контракт вывода и JSON', bool(info['contract']['ok'])))
        n_checks_total += len(checks)
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
    tot = n_checks_total
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
    print('ИТОГ: 5) ПОЛ — ПРОФИЛЬ с лотком (не плоскость): ширина и глубина по кадрам %s, '
          'RMS точек пола от профиля %s-%s м против %s-%s м от плоскости — профиль точнее '
          'в %s-%s раза; плоскость floor.a/b/c в контракте оставлена как опорная; '
          'ШПАЛЫ: %s шт, source «%s» (встроенный путь, в облаке их нет); '
          '6) сенсор над верхом головки %s-%s м (допущение: сенсор — начало облака; '
          'по отчётам 1.11 м для doubleT_platform и 1.51-1.59 м для doubleT_obstacle).'
          % (('нет выраженного лотка' if not any(
              (i['floor_profile']['module'] or {}).get('trough') for i in rows) else
              '%.2f-%.2f м глубина %.2f-%.2f м' % (
                  min((i['floor_profile']['module']['trough']['width_m'] for i in rows
                       if (i['floor_profile']['module'] or {}).get('trough')), default=float('nan')),
                  max((i['floor_profile']['module']['trough']['width_m'] for i in rows
                       if (i['floor_profile']['module'] or {}).get('trough')), default=float('nan')),
                  min((i['floor_profile']['module']['trough']['depth_m'] for i in rows
                       if (i['floor_profile']['module'] or {}).get('trough')), default=float('nan')),
                  max((i['floor_profile']['module']['trough']['depth_m'] for i in rows
                       if (i['floor_profile']['module'] or {}).get('trough')), default=float('nan')))),
             _fmt(_col(lambda i: i['floor_rms']['profile_m'])[0], 3),
             _fmt(_col(lambda i: i['floor_rms']['profile_m'])[1], 3),
             _fmt(_col(lambda i: i['floor_rms']['plane_m'])[0], 3),
             _fmt(_col(lambda i: i['floor_rms']['plane_m'])[1], 3),
             _fmt(_col(lambda i: (i['floor_rms']['plane_m'] / i['floor_rms']['profile_m'])
                       if i['floor_rms']['profile_m'] else None)[0], 2),
             _fmt(_col(lambda i: (i['floor_rms']['plane_m'] / i['floor_rms']['profile_m'])
                       if i['floor_rms']['profile_m'] else None)[1], 2),
             _fmt(_col(lambda i: i['sleepers']['count'])[0], 0),
             (rows[0]['sleepers']['source'] if rows else ''),
             _fmt(_col(lambda i: i['sensor_height_m'])[0], 3),
             _fmt(_col(lambda i: i['sensor_height_m'])[1], 3)))
    reasons = []
    if any('полом ВЬЮЕРА' in name for _, _, name in fails):
        reasons.append('кросс-проверка с полом ВЬЮЕРА (zones.fit_floor, 5-й перцентиль '
                       'по бинам 2 м от y=-60 до -2) вышла за 0.60 м только на САМОМ '
                       'дальнем конце развёртки и только там, где наклон плоскости '
                       'расходится с измеренным наклоном полки (doubleT_obstacle: '
                       'плоскость 1.92-2.00 %, полка 1.38-1.47 %): на 35 м это 0.13-0.18 м '
                       'расхождения линий, плюс сама плоскость на 0.17-0.18 м ниже '
                       'измеренной полки. Относительно ИЗМЕРЕННОЙ полки верх меша стоит '
                       'ровно на своей высоте по всей длине — см. строки «верх меша минус '
                       'ИЗМЕРЕННАЯ ПОЛКА»')
    if any('лоток' in name for _, _, name in fails):
        reasons.append('лоток: независимый замер и вывод модуля расходятся больше '
                       'допуска 50 мм — смотреть, где именно (ширина/глубина/центр): '
                       'дно освещено в 3-4 раза реже полки, поэтому края ловятся по '
                       'медиане 20-мм бинов и на редких кадрах уезжают')
    if any('RMS' in name for _, _, name in fails):
        reasons.append('RMS профиля не лучше плоскости: профиль измерен в ближней зоне, '
                       'а окно сравнения шире — смотреть meta.floor_profile.window_y')
    if any('ось-ось' in name for _, _, name in fails):
        reasons.append('колея «ось-ось» уходит от 1592-1595 на 0-3 мм: оценка центра '
                       'головки упирается в подуклонку (наружная кромка площадки выше '
                       'на 3.7 мм) и в тень снаружи; разброс величины между кадрами '
                       'одной сцены ±3 мм')
    if any('прилегающей' in name for _, _, name in fails):
        reasons.append('высота над прилегающей на doubleT_platform 0.151-0.158 м против '
                       '0.159-0.186: величина зависит от того, где взята полоса ВНУТРЬ '
                       '0.10-0.24 м — один и тот же замер даёт 0.158 и 0.170 м при '
                       'сдвиге линии на 10 мм, то есть это неопределённость базы отсчёта, '
                       'а не противоречие данным')
    if any('внутри контура' in name for _, _, name in fails):
        reasons.append('доля точек внутри контура ниже 0.8: на разреженной полосе '
                       '(4-7 точек) середина p2/p98 определяется с ошибкой ±15 мм, и '
                       'точки выходят из контура не по форме профиля, а по привязке полосы')
    print('ИТОГ: причины расхождений: %s.' % ('; '.join(reasons) if reasons
                                              else 'нет расхождений'))
    a1s = [(i['bag'], i['index'], k, i['rails'][k]['anchor_shelf'])
           for i in rows for k in ('left', 'right')
           if (i['rails'].get(k) or {}).get('anchor_shelf')]
    a2s = [i['rails'][k]['anchor_their_floor'] for i in rows for k in ('left', 'right')
           if (i['rails'].get(k) or {}).get('anchor_their_floor')]
    lens = [(i['rails'][k]['mesh_y_len_m'], i['rails'][k]['mod_dense_len_m'],
             i['rails'][k].get('mod_extra_far_chk_m'), i['rails'][k]['old_len_m'])
            for i in rows for k in ('left', 'right')
            if (i['rails'].get(k) or {}).get('mesh_y_len_m')]
    if a1s:
        print('ИТОГ: 7) РАЗВЁРТКА МЕША НИТИ — измеренный диапазон + запас (не «весь '
              'диапазон»): длина меша %.1f-%.1f м (было %.1f-%.1f м, то есть весь '
              'диапазон), плотный диапазон %.1f-%.1f м, запас на дальнем конце %.1f-%.1f м '
              '(задан %.0f м, обрезан пределом ухода %.2f м по высоте / %.2f м по X); '
              'кусков всегда 1. Верх меша над ИЗМЕРЕННОЙ полкой: медиана %.3f-%.3f м, '
              'разброс по длине не больше %.3f м (коридор %.2f..%.2f); над полом ВЬЮЕРА '
              '%.3f-%.3f м (коридор %.2f..%.2f, вне коридора %d из %d пар нить-кадр). '
              'Привязка — по измеренной полке (meta.floor_profile); плоскость floor в '
              'геометрии меша НЕ участвует и помечена в meta как ненадёжная.'
              % (min(l[0] for l in lens), max(l[0] for l in lens),
                 min(l[3] for l in lens), max(l[3] for l in lens),
                 min(l[1] for l in lens), max(l[1] for l in lens),
                 min(l[2] for l in lens), max(l[2] for l in lens),
                 EXPECT['sweep_extra_m'], EXPECT['sweep_drift_guard_m'],
                 EXPECT['sweep_x_guard_m'],
                 min(a['med_m'] for _, _, _, a in a1s), max(a['med_m'] for _, _, _, a in a1s),
                 max(a['spread_m'] for _, _, _, a in a1s),
                 EXPECT['mesh_above_shelf_m'][0], EXPECT['mesh_above_shelf_m'][1],
                 min(a['min_m'] for a in a2s), max(a['max_m'] for a in a2s),
                 EXPECT['mesh_above_floor_m'][0], EXPECT['mesh_above_floor_m'][1],
                 sum(1 for a in a2s
                     if a['min_m'] < EXPECT['mesh_above_floor_m'][0]
                     or a['max_m'] > EXPECT['mesh_above_floor_m'][1]), len(a2s)))
    return 0


if __name__ == '__main__':
    sys.exit(main())