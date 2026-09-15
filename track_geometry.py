"""Геометрия пути по одному кадру PointCloud2: ось, рельсы, шпалы, пол.

Публичный контракт — ровно одна функция `build_track(db_path, frame_index, floor_ab=None)`.
Она возвращает JSON-совместимый dict (см. её докстроку). Модуль ничего не рисует,
не открывает окон, не ходит в сеть и не имеет эффектов при импорте.

Что берётся из данных, а что из стандарта
-----------------------------------------
Наблюдаема только головка рельса — её верх и верхние ~35-40 мм боковых граней;
шейка и подошва закрыты свесом головки и не видны ни при каком угле (замерено
отдельно: видимая глубина под верхом 35 мм для doubleT_platform и 40 мм для
doubleT_obstacle). Поэтому:

  * всё выше 38 мм под верхом головки — ИЗМЕРЕНО (ширина площадки, высота верха,
    её наклон вдоль пути, положение оси и колея);
  * ниже — профиль Р65 по ГОСТ Р 51685-2022, табл. 2: H=180, b(головка)=74.59,
    e(шейка)=18, B(подошва)=150, m(перо подошвы)=11.25 мм;
  * шпалы в облаке НЕ наблюдаются (отдельный замер: ни рельефа, ни комба
    плотности) — строятся по ГОСТ 78-2004.

Локальная система профиля (мм) — «поперёк, вверх», как просит контракт
----------------------------------------------------------------------
  u — поперёк рельса, в положительную сторону — НАРУЖУ от оси пути
      (для левой нити наружу = −X, для правой = +X);
  v — вверх;
  начало координат — центр НИЖНЕЙ плоскости подошвы рельса: v=0 — низ подошвы,
  v=180 — верх головки нового рельса Р65.

Точка профиля (u, v) садится в мир как
  x = x_rail(y) + out_sign * (u*cosθ − v*sinθ)
  z = z_base(y)   +            (u*sinθ + v*cosθ)
где x_rail(y) = s*y + i — ось нити, z_base(y) = z_top(y) − 0.180 м — плоскость
подошвы, out_sign = −1 для 'left' и +1 для 'right', θ = подуклонка 1:20 = 2.86°
(ВСН 94-77, п. 3.17, внутрь колеи: верх рельса наклоняется к середине пути).
Обратный переход: u = (Δout*cosθ + Δz*sinθ), v = −Δout*sinθ + Δz*cosθ, где
Δout = out_sign*(x − x_rail), Δz = z − z_base.

Соглашения вывода
-----------------
  * 'left' — нить с МЕНЬШИМ X, 'right' — с большим (взгляд вдоль +Y, Z вверх);
  * 'axis' {s, i} — середина между осями нитей: x = s*y + i. Оси самих нитей
    лежат на axis ± gauge_axis/2 (у каждой нити свой наклон, см. meta['notes']),
    поэтому по выводу модель восстанавливается;
  * head.top_z и head.inner_face_x — значения линий в середине ЯДРА детекции
    (участок, где нити измерены плотно; вдоль пути они линейны, наклоны — в
    meta['notes']). Колея и «ось-ось» в meta тоже даны на этом уровне.
  * шпала — бокс с осями X/Y/Z: size = габаритный бокс шпалы, развёрнутой по
    оси пути (в контракте у instance нет угла поворота);
  * head.inner_face_x — внутренняя грань головки на 13 мм ниже верха (уровень
    измерения колеи по ПТЭ / 2288р).

Единицы — метры (мм только там, где в имени есть _mm).
"""

from __future__ import annotations

import math
import warnings

import numpy as np

from visualize_bag import BagFrames

# --------------------------------------------------------------- стандарт Р65
R65_H_MM = 180.00        # полная высота рельса
R65_HEAD_B_MM = 74.59    # ширина головки (контроль на 13 мм ниже верха)
R65_WEB_E_MM = 18.00     # толщина шейки
R65_FOOT_B_MM = 150.00   # ширина подошвы
R65_FOOT_M_MM = 11.25    # высота пера подошвы

HEAD_CONTROL_DEPTH_MM = 13.0   # уровень контроля ширины головки (ГОСТ Р 51685-2022, прил. П)
HEAD_MEASURED_DEPTH_MM = 38.0  # видимая глубина под верхом головки (замерено 35-40)
HEAD_BOTTOM_V_MM = R65_H_MM - HEAD_MEASURED_DEPTH_MM   # 142.0 — низ измеренной зоны
HEAD_CONTROL_V_MM = R65_H_MM - HEAD_CONTROL_DEPTH_MM   # 167.0
WEB_TOP_V_MM = 135.0            # переход головка/шейка (чертёж, вторичный источник)
FOOT_FILLET_U_MM = 40.0         # докуда перо подошвы идёт горизонтально
FOOT_TO_WEB_V_MM = 45.0         # верх перехода пера в шейку (аппроксимация галтели)
HEAD_FILLET_U_MM = 30.0         # ширина головки у её низа (аппроксимация галтели)

PODUKLONKA_RATIO = 20.0
PODUKLONKA_RAD = math.atan(1.0 / PODUKLONKA_RATIO)     # 2.862°

GAUGE_INNER_NOMINAL_MM = 1520.0                          # ПТЭ п. 45
GAUGE_AXIS_NOMINAL_MM = GAUGE_INNER_NOMINAL_MM + R65_HEAD_B_MM   # 1594.6

# шпала: деревянная тип II по ГОСТ 78-2004 табл. 1 (2750x230x160 мм) — этот тип
# стандарт предписывает приёмо-отправочным и станционным путям; тип I (250x180) —
# главным путям 1-2 классов.
SLEEPER_LENGTH_M = 2.75
SLEEPER_WIDTH_M = 0.23
SLEEPER_HEIGHT_M = 0.16
SLEEPER_SPACING_M = 0.55      # эпюра 1600-2000 шт/км, станционная

# -------------------------------------------------------- окна поиска нитей
X_LO, X_HI = -3.0, 3.0        # коридор поиска вокруг сенсора, м
X_BIN = 0.02                  # поперечный шаг огибающего профиля, м
Y_HALF = 0.5                  # полбина огибающей; полоса детекции = 3 полбины
Y_SEARCH_M = 45.0             # дальше нитей в этих данных нет (замерено)
PROM_MIN_M = 0.035            # минимальное превышение гребня над соседями с двух сторон
RIDGE_W_LO, RIDGE_W_HI = 0.025, 0.15   # ширина площадки головки на 15 мм ниже верха
GAUGE_TOL_M = 0.20            # допуск пары «ось-ось»
MID_MAX_M = 0.60              # своя колея — ближе 0.6 м к сенсору (наблюдено 0.03...0.18)

HEAD_VIS_HALF_M = 0.05        # полуокно по X при проверке «есть точки головки»
HEAD_VIS_DZ_M = 0.035         # полуокно по Z при проверке «есть точки головки»
HEAD_VIS_MIN_PTS = 1          # минимум точек в полбине, чтобы нить считалась видимой
HEAD_VIS_MAX_GAP_M = 2.5      # допустимый разрыв видимости (дальше головка редка)
HEAD_VIS_DENSE_PTS = 3        # плотность, при которой участок считается надёжным


# ------------------------------------------------------------------ плоскость
def _floor_plane(x, y, z):
    """Робастная плоскость пола z = a*x + b*y + c по 5-му перцентилю ячеек 1x1 м.

    5-й перцентиль ячейки — нижняя огибающая поверхности. Дальше МНК по ячейкам
    с отбраковкой по MAD: одиночные ячейки со сваями и сваями мусора иначе
    перекашивают плоскость. Полоса подгонки |x| <= 5 м, y в [-60, -2] (дальше пол
    почти не виден; ячейки вплотную к сенсору тоже исключены).
    """
    sel = (np.abs(x) <= 5.0) & (y >= -60.0) & (y <= -2.0)
    if sel.sum() < 2000:
        sel = np.ones(x.size, dtype=bool)

    xs, ys, zs = x[sel], y[sel], z[sel]
    key = np.floor(xs).astype(np.int64) * 100003 + np.floor(ys).astype(np.int64)
    order = np.argsort(key, kind='stable')
    groups = np.split(order, np.flatnonzero(np.diff(key[order])) + 1)

    gx, gy, gz = [], [], []
    for g in groups:
        if g.size >= 20:
            gx.append(xs[g].mean())
            gy.append(ys[g].mean())
            gz.append(np.percentile(zs[g], 5.0))
    if len(gx) < 6:
        return (0.0, 0.0, float(np.percentile(z, 5.0)))

    gx, gy, gz = np.asarray(gx), np.asarray(gy), np.asarray(gz)
    A = np.column_stack([gx, gy, np.ones(gx.size)])
    w = np.ones(gx.size)
    for _ in range(8):
        coef, *_ = np.linalg.lstsq(A * w[:, None], gz * w, rcond=None)
        r = gz - A @ coef
        mad = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-6
        w = 1.0 / np.sqrt(1.0 + (r / (3.0 * mad)) ** 2)
    return (float(coef[0]), float(coef[1]), float(coef[2]))


# ----------------------------------------------------------- огибающий профиль
def _envelope(x, y, z, y_lo, y_hi):
    """Огибающая Z по ячейкам «полбина (0.5 м) x X-бин (20 мм)».

    Возвращает (env, cnt, order, starts): env — второй максимум Z в ячейке
    (одиночный выброс не должен поднимать огибающую), order/starts — точки,
    разложенные по полбинам, чтобы не перефильтровывать облако в каждой полосе.
    """
    n_bins = int(round((X_HI - X_LO) / X_BIN))
    n_half = int(math.ceil((y_hi - y_lo) / Y_HALF)) + 1
    iy = np.floor((y - y_lo) / Y_HALF).astype(np.int64)
    ix = np.floor((x - X_LO) / X_BIN).astype(np.int64)
    ok = (ix >= 0) & (ix < n_bins) & (iy >= 0) & (iy < n_half)
    key = iy[ok] * n_bins + ix[ok]
    zz = z[ok]
    order = np.lexsort((zz, key))
    ks, zs = key[order], zz[order]
    starts = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
    ends = np.r_[starts[1:], ks.size]
    cnt = ends - starts
    env = np.full((n_half, n_bins), np.nan)
    C = np.zeros((n_half, n_bins), dtype=np.int64)
    r, c = np.divmod(ks[starts], n_bins)
    env[r, c] = np.where(cnt >= 2, zs[ends - 2], zs[ends - 1])
    C[r, c] = cnt
    idx_all = np.flatnonzero(ok)[order]
    half_start = np.searchsorted(iy[idx_all], np.arange(n_half + 1))
    return env, C, idx_all, half_start


def _ridge_candidates(env):
    """Пики огибающей, похожие на головку рельса.

    Гребень головки — локальный максимум, превышающий соседей на PROM_MIN_M с
    ОБЕИХ сторон (это и отличает рельс от кромки лотка или стены), с шириной
    площадки на 15 мм ниже верха в RIDGE_W_LO..RIDGE_W_HI.
    """
    from numpy.lib.stride_tricks import sliding_window_view

    n = env.size
    if n < 5:
        return []
    e = env.copy()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        e[1:-1] = np.nanmedian(np.stack([env[:-2], env[1:-1], env[2:]]), axis=0)

    local_max = np.zeros(n, dtype=bool)
    local_max[1:-1] = (e[1:-1] >= e[:-2]) & (e[1:-1] >= e[2:])

    half = 8
    lpad = np.full(n + 2 * half, np.inf)
    lpad[2 * half:] = np.where(np.isfinite(e), e, np.inf)
    rpad = np.full(n + 2 * half, np.inf)
    rpad[:n] = np.where(np.isfinite(e), e, np.inf)
    lmin = sliding_window_view(lpad, half + 1).min(axis=1)
    rmin = sliding_window_view(rpad, half + 1).min(axis=1)

    out = []
    for i in np.flatnonzero(local_max & np.isfinite(e)):
        prom_l = e[i] - lmin[i]
        prom_r = e[i] - rmin[i]
        if not (np.isfinite(prom_l) and np.isfinite(prom_r)):
            continue
        if min(prom_l, prom_r) < PROM_MIN_M:
            continue
        level = e[i] - 0.015
        j = i
        while j > 0 and np.isfinite(e[j]) and e[j] > level:
            j -= 1
        k = i
        while k < n - 1 and np.isfinite(e[k]) and e[k] > level:
            k += 1
        width = (k - j) * X_BIN
        if not (RIDGE_W_LO <= width <= RIDGE_W_HI):
            continue
        out.append((int(i), float(min(prom_l, prom_r))))

    out.sort(key=lambda t: t[0])
    merged = []
    for c in out:
        if merged and (c[0] - merged[-1][0]) * X_BIN <= 0.06:
            if c[1] > merged[-1][1]:
                merged[-1] = c
        else:
            merged.append(c)
    return merged


def _bands(x, y, z, y_lo, y_hi):
    """Полосы детекции: [(y_center, [(x, z, prom), ...]), ...].

    Кандидат обязан быть гребнем над МЕСТНОЙ поверхностью: 10-й перцентиль Z в
    ±0.35 м вокруг него должен быть ниже верха на 0.08-0.45 м. Без этого гейта
    на 20-40 м в пары попадали кромки потолка (их превышение 3-4 м против 0.15 м
    у рельса), и своя колея уезжала на чужую линию.
    """
    env, _cnt, idx_all, half_start = _envelope(x, y, z, y_lo, y_hi)
    n_half = env.shape[0]
    xc = X_LO + X_BIN * (np.arange(env.shape[1]) + 0.5)
    bands = []
    for b in range(max(0, n_half - 2)):
        v = env[b:b + 3]
        finite = np.isfinite(v)
        if not finite.any():
            bands.append((y_lo + (b + 1.5) * Y_HALF, []))
            continue
        e = np.where(finite.any(axis=0),
                     np.max(np.where(finite, v, -np.inf), axis=0), np.nan)
        sl = idx_all[half_start[b]:half_start[min(b + 3, n_half)]]
        xs, zs = x[sl], z[sl]
        peaks = []
        for (i, prom) in _ridge_candidates(e):
            xi = float(xc[i])
            win = np.abs(xs - xi) <= 0.35
            if win.sum() >= 10:
                gnd = float(np.percentile(zs[win], 10.0))
                h = float(e[i]) - gnd
                if not (0.08 <= h <= 0.45):
                    continue
            peaks.append((xi, float(e[i]), prom))
        bands.append((y_lo + (b + 1.5) * Y_HALF, peaks))
    return bands


def _band_pairs(band):
    """Пары пиков полосы, подходящие по колее (ось-ось)."""
    yc, peaks = band
    for a in range(len(peaks)):
        for b in range(a + 1, len(peaks)):
            xa, xb = peaks[a][0], peaks[b][0]
            if abs((xb - xa) - GAUGE_AXIS_NOMINAL_MM / 1000.0) > GAUGE_TOL_M:
                continue
            yield xa, xb, min(peaks[a][2], peaks[b][2]), yc


def _fit_line(ys, xs, tol=0.06, iters=3, pre_tol=None):
    """Прямая x = s*y + i с отбраковкой выбросов по MAD.

    pre_tol — грубый предфильтр по медиане ДО первой подгонки. Без него два
    выброса (например, точки потолка, случайно попавшие в узкое окно линии)
    разворачивают начальную МНК так, что MAD уже не помогает.
    """
    ys = np.asarray(ys, dtype=np.float64)
    xs = np.asarray(xs, dtype=np.float64)
    if pre_tol is not None and xs.size >= 4:
        keep0 = np.abs(xs - np.median(xs)) <= pre_tol
        if keep0.sum() >= 2:
            ys, xs = ys[keep0], xs[keep0]
    if ys.size < 2:
        return 0.0, (float(np.median(xs)) if xs.size else 0.0)
    s, i = np.polyfit(ys, xs, 1)
    for _ in range(iters):
        r = xs - (s * ys + i)
        med = np.median(r)
        mad = 1.4826 * np.median(np.abs(r - med)) + 1e-9
        keep = np.abs(r - med) <= max(tol, 3.0 * mad)
        if keep.all() or keep.sum() < 2:
            break
        ys, xs = ys[keep], xs[keep]
        s, i = np.polyfit(ys, xs, 1)
    return float(s), float(i)


def _core_run(ys, gap=1.5, window=10.0):
    """Непрерывный участок полос вокруг самого плотного окна.

    Дальние одиночные полосы не должны удлинять «ядро»: там головка уже не
    разрешается, и линия верха, подогнанная по ним, уезжает на постель.
    """
    ys = np.asarray(ys, dtype=np.float64)
    if ys.size < 3:
        return 0, ys.size
    order = np.argsort(ys)
    ys_s = ys[order]
    best_k, best_cnt = 0, -1
    j = 0
    for i in range(ys_s.size):
        while ys_s[i] - ys_s[j] > window:
            j += 1
        if i - j + 1 > best_cnt:
            best_cnt, best_k = i - j + 1, (i + j) // 2
    anchor = best_k
    lo = hi = anchor
    k = anchor - 1
    while k >= 0 and ys_s[k + 1] - ys_s[k] <= gap:
        lo = k
        k -= 1
    k = anchor + 1
    while k < ys_s.size and ys_s[k] - ys_s[k - 1] <= gap:
        hi = k
        k += 1
    keep = order[lo:hi + 1]
    return keep


def _find_rails(x, y, z, y_lo, y_hi, notes):
    """Ось и нити: голосование по серединам пар «два гребня на колее».

    Нити прямые, поэтому середина пары гребней своей колеи лежит на одной прямой
    x = s*y + i. Каждая полоса голосует за такую прямую; берутся сильнейшие
    направления с подавлением соседей, для каждого по полосам набираются пары
    (берётся пара, ближайшая к прогнозу) и подгоняются две линии нитей. Ядро
    трека — непрерывный участок вокруг самого плотного окна 10 м. Из годных
    треков выбирается своя колея: ближайшая к сенсору.

    Возвращает dict {'s_l','i_l','s_r','i_r','n_band','mid','core_lo','core_hi'} или None.
    """
    bands = _bands(x, y, z, y_lo, y_hi)
    slopes = np.arange(-0.05, 0.0501, 0.002)
    i0, di = -3.0, 0.01
    n_i = int(round(6.0 / di)) + 1
    acc = np.zeros((slopes.size, n_i))
    rows = np.arange(slopes.size)

    for band in bands:
        for xa, xb, _prom, yc in _band_pairs(band):
            b = np.round(((xa + xb) / 2.0 - slopes * yc - i0) / di).astype(np.int64)
            k = (b >= 0) & (b < n_i)
            if np.any(k):
                np.add.at(acc, (rows[k], b[k]), 1.0)

    peaks = []
    rest = acc.copy()
    for _ in range(8):
        kb, ib = np.unravel_index(np.argmax(rest), rest.shape)
        if rest[kb, ib] < 4:
            break
        s_mid = float(slopes[kb])
        i_mid = i0 + di * ib
        peaks.append((s_mid, i_mid))
        diag = np.abs((slopes[:, None] * (-5.0) + i0 + di * np.arange(n_i)[None, :])
                      - (s_mid * (-5.0) + i_mid))
        rest[diag < 0.10] = 0.0

    tracks = []
    for s_mid, i_mid in peaks:
        samples = []
        for band in bands:
            yc, _peaks_b = band
            pred = s_mid * yc + i_mid
            if abs(pred) > 1.2:
                continue
            best = None
            for xa, xb, _prom, _yc in _band_pairs(band):
                d = abs((xa + xb) / 2.0 - pred)
                if d > 0.10:
                    continue
                if best is None or d < best[0]:
                    best = (d, xa, xb, yc)
            if best is not None:
                samples.append((best[3], best[1], best[2]))
        if len(samples) < 4:
            continue
        arr = np.asarray(samples, dtype=np.float64)
        keep = _core_run(arr[:, 0])
        arr = arr[keep]
        if len(arr) < 4:
            continue
        for _ in range(3):
            s_l, i_l = _fit_line(arr[:, 0], arr[:, 1])
            s_r, i_r = _fit_line(arr[:, 0], arr[:, 2])
            r_l = arr[:, 1] - (s_l * arr[:, 0] + i_l)
            r_r = arr[:, 2] - (s_r * arr[:, 0] + i_r)
            keep = ((np.abs(r_l - np.median(r_l)) < 0.06)
                    & (np.abs(r_r - np.median(r_r)) < 0.06))
            if keep.all() or keep.sum() < 4:
                break
            arr = arr[keep]
        if len(arr) < 4:
            continue
        s_l, i_l = _fit_line(arr[:, 0], arr[:, 1])
        s_r, i_r = _fit_line(arr[:, 0], arr[:, 2])
        tracks.append({'s_l': s_l, 'i_l': i_l, 's_r': s_r, 'i_r': i_r,
                       'n_band': int(len(arr)),
                       'mid': float(((s_l * (-5.0) + i_l) + (s_r * (-5.0) + i_r)) / 2.0),
                       'core_lo': float(arr[:, 0].min()),
                       'core_hi': float(arr[:, 0].max())})

    if not tracks:
        notes.append('нити не найдены: пар гребней на колее 1594.6 ± 200 мм нет')
        return None

    best_n = max(t['n_band'] for t in tracks)
    own = [t for t in tracks if t['n_band'] >= 0.6 * best_n and abs(t['mid']) <= MID_MAX_M]
    if not own:
        own = [min(tracks, key=lambda t: abs(t['mid']))]
        notes.append('нити найдены, но своя колея дальше %.2f м от сенсора — '
                     'взята ближайшая' % MID_MAX_M)
    own.sort(key=lambda t: abs(t['mid']))
    return own[0]


def _visibility_range(x, y, z, s, i, m_top, c_top, y_lo, y_hi, y_core):
    """Диапазон Y, где у нити есть точки головки (под ними и растягивается меш).

    Полбина 0.5 м считается видимой, если в окне |X − линия| <= 50 мм и
    |Z − верх| <= 35 мм есть хоть одна точка: рельс занимает 0.5 м полосы своей
    длиной, и «диапазон, где есть точки рельса» — это ровно объединение таких
    полбин (так и написано в контракте). Ряд обходится от ядра детекции наружу,
    разрывы до 2.5 м допускаются. Возвращает (lo, hi, n_bands, n_dense):
    n_dense — полосы, где точек не меньше HEAD_VIS_DENSE_PTS (надёжная часть).
    """
    n = int(math.ceil((y_hi - y_lo) / Y_HALF)) + 1
    if n < 3:
        return y_lo, y_hi, 0, 0
    iy = np.floor((y - y_lo) / Y_HALF).astype(np.int64)
    ok = (iy >= 0) & (iy < n)
    seen = np.zeros(n, dtype=np.int64)
    if np.any(ok):
        yy = y[ok]
        hit = ((np.abs(x[ok] - (s * yy + i)) <= HEAD_VIS_HALF_M)
               & (np.abs(z[ok] - (m_top * yy + c_top)) <= HEAD_VIS_DZ_M))
        np.add.at(seen, iy[ok][hit], 1)
    good = seen >= HEAD_VIS_MIN_PTS
    if not good.any():
        return y_core, y_core, 0, 0
    start = int(np.clip(math.floor((y_core - y_lo) / Y_HALF), 0, n - 1))
    if not good[start]:
        near = np.flatnonzero(good)
        start = int(near[np.argmin(np.abs(near - start))])
    max_gap = int(round(HEAD_VIS_MAX_GAP_M / Y_HALF))
    lo = hi = start
    for step in (-1, 1):
        gap = 0
        k = start + step
        while 0 <= k < n:
            if good[k]:
                gap = 0
                if step < 0:
                    lo = k
                else:
                    hi = k
            else:
                gap += 1
                if gap > max_gap:
                    break
            k += step
    n_dense = int(np.sum(seen[lo:hi + 1] >= HEAD_VIS_DENSE_PTS))
    n_bands = int(hi - lo + 1)
    return y_lo + lo * Y_HALF, y_lo + (hi + 1) * Y_HALF, n_bands, n_dense


def _density_edge(u_mm, inward_negative, bin_mm=3.0, thr=0.20, lim_mm=60.0, min_pts=4):
    """Внутренняя грань головки по обрыву плотности.

    Экстремальные перцентили у неё шумные: за гранью есть редкий хвост точек
    (тень/постель), и p2 уезжает на 10-30 мм. Берём самый внутренний бин, в
    котором ещё есть плотность: не меньше 4 точек и не меньше 20 % от максимума.
    inward_negative=True — внутрь колеи это меньшие u.
    """
    u_mm = np.asarray(u_mm, dtype=np.float64)
    if u_mm.size < 10:
        return None
    edges = np.arange(-lim_mm, lim_mm + bin_mm, bin_mm)
    hist, _ = np.histogram(u_mm, bins=edges)
    if hist.max() < min_pts:
        return None
    ok = np.flatnonzero(hist >= max(min_pts, thr * hist.max()))
    if ok.size == 0:
        return None
    k = int(ok.min()) if inward_negative else int(ok.max())
    edge = edges[k] if inward_negative else edges[k + 1]
    return float(edge)


def _head_measurements(x, y, z, side, s, i, core_lo, core_hi):
    """Измерения головки нити: линия верха, ширины, внутренняя грань, высота над полом.

    Два прохода. Первый: по полосам 0.5 м берётся 98-й перцентиль Z точек в
    |X − линия| <= 60 мм и по ним подгоняется ЛИНИЯ ВЕРХА (верх вдоль пути
    наклонён, у doubleT_obstacle на 14 мм/м: без этого глубина «под верхом»
    гуляет на ±10 мм внутри полосы и ширина меряется не на той грани).

    Второй проход — по полосам 0.5 м, в КАЖДОЙ СВОЯ поперечная привязка:
    ось полосы = середина p2/p98 по X точек полосы, верх полосы = p98 по Z.
    Иначе кривизна пути вдоль 20 м смазывает u на 30-40 мм (замерено).
    В локальной координате u (наружу положительно) считаются
      * width_top_mm — p95−p5 поперечного размаха точек в 0..30 мм под верхом
        (видимая ширина площадки; наружная сторона головки всегда в тени);
      * width_13_mm — p98−p2 в 8..20 мм под верхом (уровень контроля ширины
        головки по ГОСТ Р 51685-2022);
      * inner_off_mm — внутренняя грань головки по обрыву плотности (см.
        _density_edge) в 5..30 мм под верхом, отсчёт от местной оси внутрь колеи;
      * center_off_m — насколько местные оси смещены от линии нити (медиана):
        по нему линия нити доцентровывается на головку;
      * прилегающая поверхность — 90-й перцентиль Z в 0.10..0.24 м ВНУТРЬ от
        оси нити, по полосам; высота = верх полосы минус эта поверхность
        (замерено, что именно так получаются 0.159-0.186 м).
    """
    lam = -1.0 if side == 'left' else 1.0     # +1 = наружу; для правой нити это +X
    inward = -lam
    out_sign = lam
    rows = []
    n_band = max(2, int((core_hi - core_lo) / 0.5))
    dx = x - (s * y + i)
    for k in range(n_band + 1):
        y0 = core_lo + 0.5 * k
        band = (y >= y0) & (y < y0 + 0.5)
        near = band & (np.abs(dx) < 0.35)
        rail = band & (np.abs(dx) < 0.06)
        if near.sum() >= 20:
            gnd = float(np.percentile(z[near], 10.0))
            rail = rail & (z > gnd + 0.06) & (z < gnd + 0.45)
        if rail.sum() < 15:
            continue
        top = float(np.percentile(z[rail], 98))
        row = [y0 + 0.25, top, np.nan]
        ain = band & (inward * dx > 0.10) & (inward * dx < 0.24)
        if ain.sum() >= 10:
            row[2] = float(np.percentile(z[ain], 90))
        rows.append(row)

    if len(rows) < 4:
        return None
    arr = np.asarray(rows, dtype=np.float64)
    s_top, i_top = _fit_line(arr[:, 0], arr[:, 1], tol=0.02, pre_tol=0.30)

    m_adj = np.isfinite(arr[:, 2])
    h_adj = None
    if m_adj.sum() >= 4:
        h_adj = float(np.median(arr[m_adj, 1] - arr[m_adj, 2]))

    # второй проход: по полосам со СВОЕЙ поперечной привязкой
    z_top_line = s_top * y + i_top
    us, ds, offs, cys, cxs = [], [], [], [], []
    for k in range(n_band + 1):
        y0 = core_lo + 0.5 * k
        band = (y >= y0) & (y < y0 + 0.5) & (np.abs(dx) < 0.09)
        band &= (z_top_line - z) >= -0.005
        band &= (z_top_line - z) < 0.040
        idx_b = np.flatnonzero(band)
        if idx_b.size < 8:
            continue
        xs = x[idx_b]
        xc = 0.5 * (np.percentile(xs, 2) + np.percentile(xs, 98))
        tl = float(np.percentile(z[idx_b], 98))
        d_out = out_sign * (xs - xc)
        dz = z[idx_b] - (tl - R65_H_MM / 1000.0)
        us.append((d_out * math.cos(PODUKLONKA_RAD) + dz * math.sin(PODUKLONKA_RAD))
                  * 1000.0)
        ds.append((tl - z[idx_b]) * 1000.0)
        offs.append(xc - (s * (y0 + 0.25) + i))
        cys.append(y0 + 0.25)
        cxs.append(xc)

    if not us:
        return None
    u_head = np.concatenate(us)
    d_head = np.concatenate(ds)
    if u_head.size < 40:
        return None
    center_off = float(np.median(offs)) if offs else 0.0
    s_cent, i_cent = _fit_line(cys, cxs, tol=0.05)

    m30 = (d_head >= 0.0) & (d_head < 30.0)
    width_top = (float(np.percentile(u_head[m30], 95) - np.percentile(u_head[m30], 5))
                 if m30.sum() >= 20 else None)
    m13 = (d_head >= 8.0) & (d_head < 20.0)
    width_13 = (float(np.percentile(u_head[m13], 98) - np.percentile(u_head[m13], 2))
                if m13.sum() >= 20 else None)
    m5 = (d_head >= 5.0) & (d_head < 30.0)
    inner = _density_edge(u_head[m5], inward_negative=True)
    if inner is None:
        inner = float(np.percentile(u_head[m5], 2)) if m5.sum() >= 20 else None

    return {
        's_top': s_top, 'i_top': i_top,
        'y_lo': float(arr[0, 0]), 'y_hi': float(arr[-1, 0]),
        'width_top_mm': width_top,
        'width_13_mm': width_13,
        'inner_off_mm': (None if inner is None else -float(inner)),
        'center_off_m': center_off,
        's_center': s_cent, 'i_center': i_cent,
        'height_above_adjacent_m': h_adj,
        'n_band': int(len(rows)),
    }


def rail_profile_mm(head_top_width_mm):
    """Контур поперечного сечения рельса в локальной системе (мм).

    Возвращает (verts, sources): verts — замкнутый многоугольник [(u, v), ...]
    (u наружу, v вверх от низа подошвы), sources — источник участка:
    'head measured' выше 38 мм под верхом, 'gost R65' ниже. Ширина верха площадки
    — измеренная (head_top_width_mm), ширина на 13 мм ниже верха — ГОСТ (74.59).
    """
    w2 = float(head_top_width_mm) / 2.0
    half = [
        (0.0, 0.0, 'gost R65'),
        (R65_FOOT_B_MM / 2.0, 0.0, 'gost R65'),
        (R65_FOOT_B_MM / 2.0, R65_FOOT_M_MM, 'gost R65'),
        (FOOT_FILLET_U_MM, R65_FOOT_M_MM, 'gost R65'),
        (R65_WEB_E_MM / 2.0, FOOT_TO_WEB_V_MM, 'gost R65'),
        (R65_WEB_E_MM / 2.0, WEB_TOP_V_MM, 'gost R65'),
        (HEAD_FILLET_U_MM, HEAD_BOTTOM_V_MM, 'gost R65'),
        (R65_HEAD_B_MM / 2.0, HEAD_CONTROL_V_MM, 'head measured'),
        (w2, R65_H_MM, 'head measured'),
    ]
    verts = [(u, v) for (u, v, _s) in half]
    sources = [s for (_u, _v, s) in half]
    for (u, v, s) in reversed(half[1:]):
        verts.append((-u, v))
        sources.append(s)
    return verts, sources


def _point_in_tri(p, a, b, c):
    """Точка внутри треугольника (барицентрический тест, включая границу)."""
    d1 = (p[0] - b[0]) * (a[1] - b[1]) - (a[0] - b[0]) * (p[1] - b[1])
    d2 = (p[0] - c[0]) * (b[1] - c[1]) - (b[0] - c[0]) * (p[1] - c[1])
    d3 = (p[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (p[1] - a[1])
    neg = (d1 <= 0) or (d2 <= 0) or (d3 <= 0)
    pos = (d1 >= 0) or (d2 >= 0) or (d3 >= 0)
    return not (neg and pos)


def _triangulate_polygon(pts):
    """Отсечение ушей для простого многоугольника (u,v), обход против часовой.

    Профиль рельса невыпуклый (шейка уже подошвы и головки), поэтому веером его
    крыть нельзя — нужны уши.
    """
    n = len(pts)
    if n < 3:
        return []
    idx = list(range(n))
    tris = []
    guard = 0
    while len(idx) > 3 and guard < 20 * n:
        guard += 1
        cut = False
        for k in range(len(idx)):
            i0 = idx[k - 1]
            i1 = idx[k]
            i2 = idx[(k + 1) % len(idx)]
            a, b, c = pts[i0], pts[i1], pts[i2]
            if (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) <= 0:
                continue
            inside = False
            for j in idx:
                if j in (i0, i1, i2):
                    continue
                if _point_in_tri(pts[j], a, b, c):
                    inside = True
                    break
            if inside:
                continue
            tris.append((i0, i1, i2))
            idx.pop(k)
            cut = True
            break
        if not cut:
            break
    if len(idx) == 3:
        tris.append((idx[0], idx[1], idx[2]))
    return tris


def _rail_mesh(verts_mm, s, i, m_top, c_top, y_lo, y_hi, out_sign, step=0.5):
    """Экструзия контура вдоль оси x = s*y + i на диапазоне Y, шаг 0.5 м."""
    ys = np.arange(y_lo, y_hi + 1e-9, step)
    if ys.size < 2:
        ys = np.array([y_lo, y_lo + step])
    n_p = len(verts_mm)
    n_y = ys.size
    cos_t = math.cos(PODUKLONKA_RAD)
    sin_t = math.sin(PODUKLONKA_RAD)
    verts = np.empty((n_y * n_p, 3))
    for k, yv in enumerate(ys):
        x_rail = s * yv + i
        z_base = (m_top * yv + c_top) - R65_H_MM / 1000.0
        for p, (u_mm, v_mm) in enumerate(verts_mm):
            u = u_mm / 1000.0
            v = v_mm / 1000.0
            verts[k * n_p + p, 0] = x_rail + out_sign * (u * cos_t - v * sin_t)
            verts[k * n_p + p, 1] = yv
            verts[k * n_p + p, 2] = z_base + (u * sin_t + v * cos_t)

    faces = []
    for k in range(n_y - 1):
        for p in range(n_p):
            a = k * n_p + p
            b = k * n_p + (p + 1) % n_p
            c = (k + 1) * n_p + (p + 1) % n_p
            d = (k + 1) * n_p + p
            faces.append([a, b, c])
            faces.append([a, c, d])
    caps = _triangulate_polygon(verts_mm)
    last = (n_y - 1) * n_p
    for (t0, t1, t2) in caps:
        faces.append([t2, t1, t0])
        faces.append([last + t0, last + t1, last + t2])
    return verts, faces


def _floor_mesh(a, b, c, y_lo, y_hi, x_lo=-5.0, x_hi=5.0, step=1.0):
    """Плоскость пола в коридоре X ±5 м, Y как у облака."""
    xs = np.arange(x_lo, x_hi + 1e-9, step)
    ys = np.arange(y_lo, y_hi + 1e-9, step)
    if xs.size < 2:
        xs = np.array([x_lo, x_hi])
    if ys.size < 2:
        ys = np.array([y_lo, y_hi])
    nx, ny = xs.size, ys.size
    verts = np.empty((nx * ny, 3))
    for j, yv in enumerate(ys):
        for k, xv in enumerate(xs):
            verts[j * nx + k] = (xv, yv, a * xv + b * yv + c)
    faces = []
    for j in range(ny - 1):
        for k in range(nx - 1):
            p0 = j * nx + k
            p1 = j * nx + k + 1
            p2 = (j + 1) * nx + k + 1
            p3 = (j + 1) * nx + k
            faces.append([p0, p1, p2])
            faces.append([p0, p2, p3])
    return verts, faces


def _sleepers(s_axis, i_axis, z_top_axis, y_lo, y_hi, spacing):
    """Шпалы по стандарту: боксы, ориентированные по оси пути."""
    if y_hi - y_lo < spacing:
        return []
    theta = math.atan(s_axis)
    sx = SLEEPER_LENGTH_M * abs(math.cos(theta))
    sy = SLEEPER_WIDTH_M + SLEEPER_LENGTH_M * abs(math.sin(theta))
    out = []
    k = 0
    while True:
        yv = y_lo + 0.5 * spacing + k * spacing
        if yv > y_hi - 0.5 * spacing + 1e-9:
            break
        z_top = z_top_axis(yv) - R65_H_MM / 1000.0
        out.append({
            'center': [float(s_axis * yv + i_axis), float(yv),
                       float(z_top - SLEEPER_HEIGHT_M / 2.0)],
            'size': [float(sx), float(sy), float(SLEEPER_HEIGHT_M)],
        })
        k += 1
    return out


def build_track(db_path: str, frame_index: int, floor_ab=None) -> dict:
    """Строит геометрию пути по кадру rosbag2 .db3.

    Параметры
    ---------
    db_path : путь к .db3 (например for_hackathon/doubleT_platform/doubleT_platform_0.db3)
    frame_index : номер кадра PointCloud2 в bag (0 .. len(BagFrames)-1)
    floor_ab : необязательная плоскость пола (a, b, c) для z = a*x + b*y + c;
               если None — оценивается по кадру (см. _floor_plane)

    Возвращает dict:
    {
      'frame': int,
      'axis':  {'s': float, 'i': float},              # середина колеи: x = s*y + i
      'floor': {'a': float, 'b': float, 'c': float},  # z = a*x + b*y + c
      'rails': [ {'side': 'left'|'right',
                  'head': {'top_z': float, 'width_mm': float, 'inner_face_x': float,
                           'height_above_adjacent_m': float, 'y_range': [y0, y1]},
                  'mesh': {'vertices': [[x,y,z], ...], 'faces': [[i,j,k], ...]},
                  'source': {'head': 'measured', 'web': 'gost R65', 'foot': 'gost R65'}},
                 ... ],                              # ровно два: left (x меньше), right
      'sleepers': {'instances': [{'center': [x,y,z], 'size': [sx,sy,sz]}, ...],
                   'spacing_m': float, 'source': 'gost'},
      'floor_mesh': {'vertices': [[x,y,z], ...], 'faces': [[i,j,k], ...],
                     'source': 'measured'},
      'meta': {'gauge_inner_mm': float, 'gauge_axis_mm': float, 'head_width_mm': float,
               'sensor_height_above_head_m': float, 'notes': [str, ...]}
    }

    Оси нитей восстанавливаются из вывода: x = axis.s*y + axis.i ± gauge_axis_mm/2000
    (у каждой нити свой наклон — числа в meta['notes']). head.top_z, head.inner_face_x,
    колея и «ось-ось» даны в середине ядра детекции (где нити измерены плотно).
    """
    frames = BagFrames(db_path, cache_size=2)
    xyz, _intensity, _ts = frames[frame_index]
    if len(xyz) == 0:
        raise ValueError(f'кадр {frame_index} в {db_path} пуст')
    x = xyz[:, 0].astype(np.float64)
    y = xyz[:, 1].astype(np.float64)
    z = xyz[:, 2].astype(np.float64)

    notes = []
    if floor_ab is None:
        a, b, c = _floor_plane(x, y, z)
        notes.append('пол оценён по кадру: 5-й перцентиль ячеек 1x1 м (|x|<=5 м, '
                     'y в [-60,-2]), МНК с отбраковкой по MAD')
    else:
        a, b, c = float(floor_ab[0]), float(floor_ab[1]), float(floor_ab[2])
        notes.append('пол взят из floor_ab, по кадру не оценивался')

    y_lo = max(float(y.min()), -Y_SEARCH_M)
    y_hi = min(float(y.max()), 5.0)
    track = _find_rails(x, y, z, y_lo, y_hi, notes)

    if track is None:
        s_axis, i_axis = 0.0, 0.0
        head_src = 'gost R65 (не измерено)'
        notes.append('головка не измерена: нити поставлены по нормативной колее '
                     '1594.6 мм от сенсора, верх = пол + 0.17 м')
    else:
        s_axis = 0.5 * (track['s_l'] + track['s_r'])
        i_axis = 0.5 * (track['i_l'] + track['i_r'])
        head_src = 'measured'

    # --- по нити: линия, линия верха, измерения
    rails = []
    for side in ('left', 'right'):
        out_sign = -1.0 if side == 'left' else 1.0
        src = head_src
        if track is None:
            s_r = s_axis
            i_r = i_axis + out_sign * GAUGE_AXIS_NOMINAL_MM / 2000.0
            m_top = b
            c_top = a * i_r + c + 0.17
            meas = {'width_top_mm': R65_HEAD_B_MM, 'width_13_mm': R65_HEAD_B_MM,
                    'inner_off_mm': R65_HEAD_B_MM / 2.0,
                    'height_above_adjacent_m': None,
                    'y_lo': y_hi - 1.0, 'y_hi': y_hi}
        else:
            s_r = track['s_l'] if side == 'left' else track['s_r']
            i_r = track['i_l'] if side == 'left' else track['i_r']
            meas = _head_measurements(x, y, z, side, s_r, i_r,
                                      track['core_lo'], track['core_hi'])
            if meas is None:
                notes.append(f'{side}: головка измерилась меньше чем в 4 полосах — '
                             'взята ширина по ГОСТ, высота не измерена')
                m_top = b
                c_top = a * i_r + c + 0.17
                meas = {'width_top_mm': R65_HEAD_B_MM, 'width_13_mm': R65_HEAD_B_MM,
                        'inner_off_mm': R65_HEAD_B_MM / 2.0,
                        'height_above_adjacent_m': None,
                        'y_lo': track['core_lo'], 'y_hi': track['core_hi']}
                src = 'gost R65 (не измерено)'
            else:
                m_top, c_top = meas['s_top'], meas['i_top']
                # ось нити — линия по центрам поперечных сечений головки (середина
                # p2/p98 точек полосы), а не по гребню: гребень смещён наружу
                # (подуклонка поднимает наружную кромку площадки, наружная сторона
                # в тени), из-за чего колея «ось-ось» завышалась на 10-20 мм
                s_r, i_r = float(meas['s_center']), float(meas['i_center'])

        rails.append({'side': side, 'out_sign': out_sign, 's': s_r, 'i': i_r,
                      'm_top': m_top, 'c_top': c_top, 'meas': meas, 'src': src})

    # --- участок, где видны точки головки
    for r in rails:
        meas = r['meas']
        core = 0.5 * (meas['y_lo'] + meas['y_hi'])
        vy_lo, vy_hi, n_bands, n_dense = _visibility_range(
            x, y, z, r['s'], r['i'], r['m_top'], r['c_top'], y_lo, y_hi, core)
        r['vis_bands'], r['vis_dense'] = n_bands, n_dense
        if vy_hi - vy_lo < 1.0:
            vy_hi = vy_lo + 1.0
            notes.append(f"{r['side']}: видимый участок короче 1 м — меш вырожден")
        r['y_lo'], r['y_hi'] = float(vy_lo), float(vy_hi)
        if n_bands > 0 and n_dense < 0.5 * n_bands:
            notes.append('%s: за пределами плотной части головка представлена '
                         'единичными точками (%d полос из %d без трёх точек) — меш там '
                         'прямая экструзия по экстраполированной оси, ширина и верх '
                         'измерены только в плотной части'
                         % (r['side'], n_bands - n_dense, n_bands))

    common_lo = max(r['y_lo'] for r in rails)
    common_hi = min(r['y_hi'] for r in rails)
    if common_hi - common_lo < 0.5:
        common_lo, common_hi = min(r['y_lo'] for r in rails), max(r['y_hi'] for r in rails)
        notes.append('общего видимого участка у нитей нет — общие величины даны '
                     'по объединению диапазонов')
    # Отчётные величины нитей (top_z, inner_face_x, колея) берём в середине ЯДРА
    # детекции — участка, где нити измерены плотно. Если брать середину всего меша,
    # колея «поедет» вместе с длиной видимого участка (нити не параллельны:
    # разность наклонов ~0.0015 м/м даёт 8 мм на 6 м).
    if track is not None:
        y_mid = 0.5 * (track['core_lo'] + track['core_hi'])
    else:
        y_mid = 0.5 * (common_lo + common_hi)

    # --- вывод по нитям
    rails_out = []
    head_widths = []
    inner_faces = []
    for r in rails:
        meas = r['meas']
        verts_mm, _srcs = rail_profile_mm(meas['width_top_mm'])
        verts, faces = _rail_mesh(verts_mm, r['s'], r['i'], r['m_top'], r['c_top'],
                                  r['y_lo'], r['y_hi'], r['out_sign'])
        inner_off = (meas['inner_off_mm'] if meas['inner_off_mm'] is not None
                     else R65_HEAD_B_MM / 2.0)
        inner_face_x = r['s'] * y_mid + r['i'] - r['out_sign'] * inner_off / 1000.0
        rails_out.append({
            'side': r['side'],
            'head': {
                'top_z': float(r['m_top'] * y_mid + r['c_top']),
                'width_mm': float(meas['width_top_mm']),
                'inner_face_x': float(inner_face_x),
                'height_above_adjacent_m': (None if meas['height_above_adjacent_m'] is None
                                            else float(meas['height_above_adjacent_m'])),
                'y_range': [r['y_lo'], r['y_hi']],
            },
            'mesh': {'vertices': [[float(v) for v in row] for row in verts],
                     'faces': [[int(f0), int(f1), int(f2)] for (f0, f1, f2) in faces]},
            'source': {'head': r['src'], 'web': 'gost R65', 'foot': 'gost R65'},
        })
        head_widths.append(float(meas['width_top_mm']))
        inner_faces.append(float(inner_face_x))

    gauge_axis_m = abs(((rails[1]['s'] * y_mid + rails[1]['i'])
                        - (rails[0]['s'] * y_mid + rails[0]['i'])))
    gauge_inner_m = abs(inner_faces[1] - inner_faces[0])

    z_top_axis = (lambda yv, rl=rails: 0.5 * (rl[0]['m_top'] * yv + rl[0]['c_top']
                                              + rl[1]['m_top'] * yv + rl[1]['c_top']))

    sleepers_y_lo = common_lo
    sleepers_y_hi = common_hi
    if sleepers_y_hi - sleepers_y_lo < SLEEPER_SPACING_M:
        notes.append('общий видимый участок короче шага шпал — шпалы не построены')
    instances = _sleepers(s_axis, i_axis, z_top_axis, sleepers_y_lo, sleepers_y_hi,
                          SLEEPER_SPACING_M)
    if not instances:
        notes.append('шпалы не построены: диапазон оси короче шага')

    f_verts, f_faces = _floor_mesh(a, b, c, float(y.min()), float(y.max()))
    z_head = 0.5 * (rails_out[0]['head']['top_z'] + rails_out[1]['head']['top_z'])

    if track is not None:
        notes.append('наклоны линий вдоль пути (м/м): ось %+.5f, левая нить %+.5f, '
                     'правая нить %+.5f; верх головки: левая %+.5f, правая %+.5f'
                     % (s_axis, track['s_l'], track['s_r'],
                        rails[0]['m_top'], rails[1]['m_top']))
        notes.append('нити найдены в %d полосах, ядро детекции y = %.1f..%.1f м; '
                     'меш тянется по всем полосам, где есть точки головки: левая '
                     'y = %.1f..%.1f м (%d полос, %d плотных), правая y = %.1f..%.1f м '
                     '(%d полос, %d плотных)'
                     % (track['n_band'], track['core_lo'], track['core_hi'],
                        rails[0]['y_lo'], rails[0]['y_hi'], rails[0]['vis_bands'],
                        rails[0]['vis_dense'], rails[1]['y_lo'], rails[1]['y_hi'],
                        rails[1]['vis_bands'], rails[1]['vis_dense']))
    notes.append('ширина головки на 13 мм ниже верха взята из ГОСТ Р 51685-2022 '
                 '(74.59 мм): лидар видит только верхние ~38 мм головки')
    notes.append('подуклонка 1:20 (2.86°) внутрь колеи — по ВСН 94-77 п. 3.17, '
                 'в данных не измерена')
    notes.append('шпалы: ГОСТ 78-2004 тип II, 2750x230x160 мм, шаг %.2f м; '
                 'в облаке шпалы не наблюдаются (ни рельефом, ни комбом плотности), '
                 'верх шпалы совмещён с плоскостью подошвы рельса' % SLEEPER_SPACING_M)
    notes.append('сенсор — начало облака (0,0,0), sensor_height_above_head_m = -top_z '
                 '(допущение, отдельно не проверено)')
    notes.append('колея по внутренним граням измерена на 13 мм ниже верха; колея '
                 '«ось-ось» больше на ширину головки')
    notes.append('колея «ось-ось» %.1f мм — по линиям нитей, каждая проведена по центрам '
                 'поперечных сечений головки (середина p2/p98 точек полосы); '
                 'нормативная связь колея_внутр + 74.59 = %.1f мм, норма 1594.6 мм'
                 % (1000.0 * gauge_axis_m, 1000.0 * gauge_inner_m + R65_HEAD_B_MM))
    notes.append('сумма внутренних смещений граней от осей нитей (колея «ось-ось» минус '
                 'колея по граням) = %.1f мм против нормативных 74.59 мм: наружная '
                 'сторона головки в тени, а подуклонка поднимает наружную кромку '
                 'площадки на 3.7 мм, поэтому оценка центра головки смещена наружу на '
                 'несколько мм — отсюда и остаточный сдвиг колеи «ось-ось»'
                 % (1000.0 * (gauge_axis_m - gauge_inner_m)))
    notes.append('пол — плоскость по нижней огибающей (5-й перцентиль ячеек 1x1 м в '
                 '|x|<=5 м, y в [-60,-2]): поверхность СТУПЕНЧАТАЯ (полка, лоток, '
                 'обочины), поэтому плоскость проходит между уровнями и между кадрами '
                 'одной сцены её положение гуляет (замерено ~0.14 м на doubleT_platform '
                 'против 0.003 м на doubleT_obstacle)')

    return {
        'frame': int(frame_index),
        'axis': {'s': float(s_axis), 'i': float(i_axis)},
        'floor': {'a': float(a), 'b': float(b), 'c': float(c)},
        'rails': rails_out,
        'sleepers': {'instances': instances, 'spacing_m': float(SLEEPER_SPACING_M),
                     'source': 'gost'},
        'floor_mesh': {'vertices': [[float(v) for v in row] for row in f_verts],
                       'faces': [[int(t0), int(t1), int(t2)] for (t0, t1, t2) in f_faces],
                       'source': 'measured'},
        'meta': {
            'gauge_inner_mm': float(1000.0 * gauge_inner_m),
            'gauge_axis_mm': float(1000.0 * gauge_axis_m),
            'head_width_mm': float(np.mean(head_widths)),
            'sensor_height_above_head_m': float(-z_head),
            'notes': notes,
        },
    }