"""
Rail detection and gauge measurement for LiDAR point clouds.

Functions extracted from visualize_bag.py for reuse and modification.
"""

import numpy as np
import open3d as o3d


def rail_x(line, y):
    """Вычисляет X рельса в точке Y.

    line = (slope, intercept)        → x = slope * y + intercept
    line = (a2, a1, a0)              → x = a2 * y² + a1 * y + a0  (парабола)
    line = {'knots_y':…, 'knots_x':…} → интерполяция по узлам (плавная кривая)
    """
    # Кусочно-линейная интерполяция по узлам — плавная, без разрывов
    if isinstance(line, dict):
        ky = line['knots_y']
        kx = line['knots_x']
        return np.interp(y, ky, kx)

    if len(line) == 2:
        return line[0] * y + line[1]
    return line[0] * y * y + line[1] * y + line[2]


def _estimate_ground_plane(points, cell_size=1.0, near_radius=5.0):
    """
    Робастная оценка плоскости земли через grid-based подход.

    Делит облако на ячейки cell_size × cell_size, берёт 5-й перцентиль Z
    в каждой ячейке (≈ уровень земли), затем RANSAC по сетке.

    near_radius: оценка строится только по точкам в радиусе near_radius
    от лидара (по XY), чтобы дальние точки не искажали плоскость.
    """
    from sklearn.linear_model import RANSACRegressor

    # Берём только ближние точки для оценки земли
    if near_radius is not None and near_radius > 0:
        dist_xy = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
        near_mask = dist_xy <= near_radius
        pts_for_ground = points[near_mask]
    else:
        pts_for_ground = points

    if len(pts_for_ground) < 100:
        pts_for_ground = points

    z = pts_for_ground[:, 2]
    x_cells = np.arange(pts_for_ground[:, 0].min() - 0.5, pts_for_ground[:, 0].max() + 0.5, cell_size)
    y_cells = np.arange(pts_for_ground[:, 1].min() - 0.5, pts_for_ground[:, 1].max() + 0.5, cell_size)
    nx = len(x_cells) - 1
    ny = len(y_cells) - 1

    if nx < 1 or ny < 1:
        return None

    # searchsorted даёт тот же результат, что (p >= x_cells[i]) & (p < x_cells[i+1])
    xi = np.searchsorted(x_cells, pts_for_ground[:, 0], side='right') - 1
    yi = np.searchsorted(y_cells, pts_for_ground[:, 1], side='right') - 1
    # Точки за пределами сетки — clip
    np.clip(xi, 0, nx - 1, out=xi)
    np.clip(yi, 0, ny - 1, out=yi)

    # Линейный индекс ячейки
    cell_id = xi * ny + yi

    # Сортируем по ячейкам — одна сортировка вместо двойного цикла
    order = np.argsort(cell_id)
    cell_id_sorted = cell_id[order]
    z_sorted = z[order]

    # Границы ячеек в отсортированном массиве
    changes = np.where(np.diff(cell_id_sorted) != 0)[0] + 1
    starts = np.empty(len(changes) + 1, dtype=np.intp)
    starts[0] = 0
    starts[1:] = changes
    ends = np.empty(len(changes) + 1, dtype=np.intp)
    ends[:-1] = changes
    ends[-1] = len(cell_id_sorted)
    unique_cells = cell_id_sorted[starts]

    gx, gy, gz = [], [], []
    for k in range(len(unique_cells)):
        count = ends[k] - starts[k]
        if count > 10:
            cid = unique_cells[k]
            cx = cid // ny
            cy = cid % ny
            cell_z = z_sorted[starts[k]:ends[k]]
            p5 = np.percentile(cell_z, 5)
            gx.append((x_cells[cx] + x_cells[cx + 1]) / 2)
            gy.append((y_cells[cy] + y_cells[cy + 1]) / 2)
            gz.append(p5)

    if len(gx) < 10:
        return None

    XY = np.column_stack([gx, gy])
    # random_state фиксирован: без него посев менял и подгонку плоскости земли,
    # и (через неё) состав точек рельсовой маски — 1516…1549 мм на одном кадре.
    ransac = RANSACRegressor(residual_threshold=0.15, max_trials=1000, random_state=0)
    ransac.fit(XY, np.array(gz))
    a, b = ransac.estimator_.coef_
    c = ransac.estimator_.intercept_

    if abs(a) > 0.2 or abs(b) > 0.2:
        return None

    return (a, b, c)


def _fit_rail_robust(y_vals, x_vals, knot_step=10.0, window_scale=1.5,
                     mad_k=3.0):
    """Робастный фит рельса: скользящее окно → узлы → интерполяция.

    Для каждого узла (шаг knot_step метров) берёт точки в окне
    ±knot_step*window_scale, отбрасывает выбросы по MAD, берёт медиану X.
    Результат — набор узлов, между которыми np.interp делает плавную
    кусочно-линейную интерполяцию. Никаких разрывов, никаких треугольников.

    Ближние узлы (много точек) → точные. Дальние (мало точек) → шумнее,
    но не тянут ближние.

    На коротких дистанциях (<30м) — обычный polyfit (tuple).

    Возвращает:
        dict {'knots_y':…, 'knots_x':…}  — если длинная дистанция
        tuple (a2, a1, a0)                — если короткая (совместимо)
    """
    y = np.array(y_vals, dtype=np.float64)
    x = np.array(x_vals, dtype=np.float64)
    y_span = y.max() - y.min()

    # Короткая дистанция — обычный polyfit
    if y_span < 30.0 or len(y) < 6:
        if y_span < 1.0 or len(y) < 3:
            return (0.0, 0.0, float(x.mean()))
        if y_span < 3.0 or len(y) < 6:
            coeffs = np.polyfit(y, x, 1)
            return (0.0, float(coeffs[0]), float(coeffs[1]))
        # MAD-фильтрация перед polyfit
        keep = np.ones(len(y), dtype=bool)
        for _ in range(2):
            yf, xf = y[keep], x[keep]
            if len(yf) < 4:
                break
            c1 = np.polyfit(yf, xf, 1)
            res = xf - (c1[0] * yf + c1[1])
            med_r = np.median(res)
            mad = np.median(np.abs(res - med_r))
            if mad <= 0:
                break
            inliers = np.abs(res - med_r) <= mad_k * 1.4826 * mad
            if inliers.all():
                break
            idx = np.where(keep)[0]
            keep[idx[~inliers]] = False
        yf, xf = y[keep], x[keep]
        if len(yf) < 3:
            yf, xf = y, x
        coeffs = np.polyfit(yf, xf, min(2, max(1, len(yf) - 1)))
        if len(coeffs) == 2:
            return (0.0, float(coeffs[0]), float(coeffs[1]))
        return (float(coeffs[0]), float(coeffs[1]), float(coeffs[2]))

    # Длинная дистанция — скользящее окно по узлам
    y_min, y_max = y.min(), y.max()
    window = knot_step * window_scale

    # Сортируем для быстрого searchsorted
    order = np.argsort(y)
    y_s = y[order]
    x_s = x[order]

    # Глобальный линейный тренд для MAD-фильтрации
    global_fit = np.polyfit(y, x, 1)

    knots_y = []
    knots_x = []

    for y_knot in np.arange(y_min, y_max + knot_step * 0.5, knot_step):
        # Точки в окне
        i_lo = np.searchsorted(y_s, y_knot - window, side='left')
        i_hi = np.searchsorted(y_s, y_knot + window, side='right')
        if i_hi - i_lo < 2:
            # Мало точек — интерполяция по глобальному тренду
            knots_y.append(y_knot)
            knots_x.append(float(global_fit[0] * y_knot + global_fit[1]))
            continue

        yw = y_s[i_lo:i_hi]
        xw = x_s[i_lo:i_hi]

        # Весá: ближе к узлу → больший вес (треугольное ядро)
        w = 1.0 - np.abs(yw - y_knot) / window
        w = np.maximum(w, 0.1)  # минимальный вес 0.1

        # MAD-фильтрация относительно взвешенной медианы
        # Сначала грубая оценка центра — взвешенная медиана
        sort_idx = np.argsort(xw)
        cumw = np.cumsum(w[sort_idx])
        center_x = float(xw[sort_idx[np.searchsorted(cumw, cumw[-1] * 0.5)]])

        residuals = xw - center_x
        mad = np.median(np.abs(residuals))
        if mad > 0:
            good = np.abs(residuals) <= mad_k * 1.4826 * mad
            if good.sum() >= 2:
                xw = xw[good]
                w = w[good]

        # Взвешенная медиана очищенных данных
        sort_idx = np.argsort(xw)
        cumw = np.cumsum(w[sort_idx])
        knot_x = float(xw[sort_idx[np.searchsorted(cumw, cumw[-1] * 0.5)]])

        knots_y.append(y_knot)
        knots_x.append(knot_x)

    if len(knots_y) < 2:
        coeffs = np.polyfit(y, x, 1)
        return (0.0, float(coeffs[0]), float(coeffs[1]))

    return {'knots_y': np.array(knots_y), 'knots_x': np.array(knots_x)}


def estimate_tunnel_axis(points, ground_plane, y_centers, slice_halfs,
                         min_wall_points=30, wall_dz_min=0.5,
                         wall_dz_max=6.0, smoothing_window=5):
    """Ось туннеля по срезам: midline между стенами.

    На каждом поперечном срезе по Y:
    1. Берём точки выше wall_dz_min над землёй (стены, потолок — не балласт)
    2. Если точек >= min_wall_points — это туннель
    3. Левая стена = перцентиль 5% по X, правая = 95%
    4. Midline = (left + right) / 2

    Робастность к разным туннелям:
    - Круглый: стены = дуги, перцентили ловят края
    - Квадратный: стены = вертикальные плоскости, перцентили точны
    - С нишами/кабелями: 5-95% перцентили игнорируют выбросы
    - Несимметричный (платформа): midline сдвигается — это правильно,
      потому что путь тоже сдвинут от геометрического центра

    Возвращает:
        axis_y: np.array — Y координаты где ось определена
        axis_x: np.array — X координаты оси (midline)
        Или (None, None) если не туннель (мало срезов с достаточным количеством точек)
    """
    if ground_plane is None:
        return None, None

    a, b, c = ground_plane

    # Предсортировка по Y для быстрого среза
    y_order = np.argsort(points[:, 1])
    pts_y = points[y_order, 1]
    pts_x = points[y_order, 0]
    pts_z = points[y_order, 2]
    # Z земли для отсортированных точек
    gz = a * pts_x + b * pts_y + c
    dz_all = pts_z - gz

    raw_y = []
    raw_x = []

    for i, y_center in enumerate(y_centers):
        sh = slice_halfs[i] if i < len(slice_halfs) else slice_halfs[-1]
        # Берём более широкий срез для стен (стены дальше от центра чем рельсы)
        wall_sh = max(sh, 1.0)

        i_lo = np.searchsorted(pts_y, y_center - wall_sh, side='left')
        i_hi = np.searchsorted(pts_y, y_center + wall_sh, side='right')

        if i_hi - i_lo < min_wall_points:
            continue

        sx = pts_x[i_lo:i_hi]
        sdz = dz_all[i_lo:i_hi]

        # Только точки стен/потолка — выше балласта
        wall_mask = (sdz > wall_dz_min) & (sdz < wall_dz_max)
        wx = sx[wall_mask]

        if len(wx) < min_wall_points:
            continue

        # Края туннеля через перцентили
        left_wall = np.percentile(wx, 5)
        right_wall = np.percentile(wx, 95)

        # Санитарная проверка: стены должны быть достаточно далеко друг от друга
        # (минимум 2м — даже самый узкий туннель шире)
        if right_wall - left_wall < 2.0:
            continue

        midline = (left_wall + right_wall) / 2.0
        raw_y.append(y_center)
        raw_x.append(midline)

    if len(raw_y) < 3:
        return None, None

    raw_y = np.array(raw_y)
    raw_x = np.array(raw_x)

    # Сглаживание — скользящая медиана, чтобы ниши/кабели/аномалии не дёргали ось
    if smoothing_window > 1 and len(raw_x) >= smoothing_window:
        half_w = smoothing_window // 2
        smoothed_x = np.copy(raw_x)
        for j in range(len(raw_x)):
            lo = max(0, j - half_w)
            hi = min(len(raw_x), j + half_w + 1)
            smoothed_x[j] = np.median(raw_x[lo:hi])
        return raw_y, smoothed_x

    return raw_y, raw_x


def _extract_peaks(points, dz, y_center, slice_half, x_bin, min_band,
                    min_bin_pts, x_search_limit):
    """Извлекает пики из одного поперечного среза. Возвращает list of (x, z)."""
    # Предполагается что points уже отсортированы по Y (или передан срез)
    bp_x = points[:, 0]
    bdz = dz

    if len(bp_x) < min_band:
        return []

    x_lo = max(bp_x.min(), -x_search_limit)
    x_hi = min(bp_x.max(), x_search_limit)
    if x_hi - x_lo < 0.5:
        return []

    x_bins = np.arange(x_lo, x_hi, x_bin)
    n_xbins = len(x_bins) - 1
    if n_xbins < 3:
        return []

    xbi = np.digitize(bp_x, x_bins) - 1
    valid_xb = (xbi >= 0) & (xbi < n_xbins)
    xbi_v = xbi[valid_xb]
    bdz_v = bdz[valid_xb]

    bin_counts = np.bincount(xbi_v, minlength=n_xbins)
    if (bin_counts >= min_bin_pts).sum() < 3:
        return []

    xb_order = np.argsort(xbi_v, kind='mergesort')
    xbi_s = xbi_v[xb_order]
    bdz_s = bdz_v[xb_order]
    bin_edges = np.zeros(n_xbins + 1, dtype=np.intp)
    np.cumsum(bin_counts, out=bin_edges[1:])

    x_centers_list = []
    z_profile_list = []
    for bi in range(n_xbins):
        if bin_counts[bi] >= min_bin_pts:
            s, e = bin_edges[bi], bin_edges[bi + 1]
            p80 = np.percentile(bdz_s[s:e], 80)
            x_centers_list.append((x_bins[bi] + x_bins[bi + 1]) / 2)
            z_profile_list.append(p80)

    if len(x_centers_list) < 3:
        return []

    x_centers = np.array(x_centers_list)
    z_profile = np.array(z_profile_list)

    rail_height = (z_profile > 0.03) & (z_profile < 0.40)
    if not rail_height.any():
        return []

    rx = x_centers[rail_height]
    rz = z_profile[rail_height]

    group_gap = max(0.15, x_bin * 2.5)
    groups = [[0]]
    for j in range(1, len(rx)):
        if rx[j] - rx[j - 1] < group_gap:
            groups[-1].append(j)
        else:
            groups.append([j])

    return [(rx[g[np.argmax(rz[g])]], rz[g].max()) for g in groups]


def _extract_trackbed(bp_x, x_bin, x_search_limit, gauge=1.52, gauge_tol=0.15,
                      own_x_range=1.5):
    """Ищет полотно пути как связную полосу плотности, ограниченную провалами.

    Полотно = непрерывная зона по X где плотность > порога, шириной
    примерно gauge (1.37..1.67м для стандартной колеи 1520±150мм).
    Границы полотна — края рельсов (внешние грани).

    Возвращает list of (x_left_edge, x_right_edge) — найденные полотна.
    x_left_edge/x_right_edge — X внешних границ полотна (≈ позиции рельсов).
    """
    if len(bp_x) < 20:
        return []

    x_lo = max(float(bp_x.min()), -x_search_limit)
    x_hi = min(float(bp_x.max()), x_search_limit)
    if x_hi - x_lo < 0.5:
        return []

    # Гистограмма плотности по X — бин чуть крупнее чем для пиков
    tb_bin = max(x_bin, 0.05)
    edges = np.arange(x_lo, x_hi + tb_bin, tb_bin)
    counts, _ = np.histogram(bp_x, bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2

    if len(counts) < 5:
        return []

    # Сглаживание
    kernel = np.ones(5) / 5.0
    smoothed = np.convolve(counts.astype(float), kernel, mode='same')

    median_density = np.median(smoothed[smoothed > 0]) if (smoothed > 0).any() else 1.0
    if median_density <= 0:
        return []

    # Бинарная маска: "есть точки" vs "пусто"
    # Порог 0.3 от медианы — ниже = провал / тень / за пределами полотна
    threshold = 0.3
    is_filled = smoothed > (median_density * threshold)

    # Найти связные отрезки "заполненных" бинов
    segments = []  # [(start_idx, end_idx)]
    in_seg = False
    seg_start = 0
    for i in range(len(is_filled)):
        if is_filled[i] and not in_seg:
            seg_start = i
            in_seg = True
        elif not is_filled[i] and in_seg:
            segments.append((seg_start, i - 1))
            in_seg = False
    if in_seg:
        segments.append((seg_start, len(is_filled) - 1))

    # Фильтр: ширина сегмента должна быть примерно gauge.
    # Полотно видно от внутренних теней (ближе к центру) до внешних краёв,
    # поэтому может быть уже колеи на ~0.3м (тени внутри рельсов).
    min_width = gauge - gauge_tol - 0.5
    max_width = gauge + gauge_tol + 0.5

    results = []
    for s_start, s_end in segments:
        x_left = centers[s_start]
        x_right = centers[s_end]
        width = x_right - x_left
        if min_width <= width <= max_width:
            midpoint = (x_left + x_right) / 2
            if abs(midpoint) <= own_x_range + 0.5:
                results.append((x_left, x_right))

    return results


def _extract_shadow_dips(bp_x, x_bin, x_search_limit):
    """Извлекает позиции провалов плотности (теней) из поперечного среза.

    Возвращает list of float — X-координаты провалов (без фильтрации по колее).
    """
    if len(bp_x) < 20:
        return []

    x_lo = max(bp_x.min(), -x_search_limit)
    x_hi = min(bp_x.max(), x_search_limit)
    if x_hi - x_lo < 0.5:
        return []

    shadow_bin = max(x_bin, 0.02)
    edges = np.arange(x_lo, x_hi + shadow_bin, shadow_bin)
    counts, _ = np.histogram(bp_x, bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2

    if len(counts) < 5:
        return []

    kernel = np.ones(5) / 5.0
    smoothed = np.convolve(counts.astype(float), kernel, mode='same')

    median_density = np.median(smoothed[smoothed > 0]) if (smoothed > 0).any() else 1.0
    if median_density <= 0:
        return []

    norm = smoothed / median_density

    dips = []
    for i in range(1, len(norm) - 1):
        if norm[i] < norm[i - 1] and norm[i] < norm[i + 1] and norm[i] < 0.5:
            dips.append(centers[i])
    if len(norm) > 0 and norm[0] < 0.3:
        dips.insert(0, centers[0])
    if len(norm) > 0 and norm[-1] < 0.3:
        dips.append(centers[-1])

    return dips


def _extract_shadow_pairs(bp_x, x_bin, x_search_limit, gauge=1.52,
                          gauge_tol=0.15, own_x_range=1.5):
    """Ищет пары провалов плотности (теней от рельсов) в поперечном срезе.

    Рельс выпирает над балластом → лидар не видит за ним → в гистограмме
    плотности по X появляются узкие провалы. Пара провалов на расстоянии
    колеи = рельсы.

    Возвращает list of (x_left, x_right) — пары позиций провалов.
    """
    dips = _extract_shadow_dips(bp_x, x_bin, x_search_limit)
    if len(dips) < 2:
        return []

    pairs = []
    for ai in range(len(dips)):
        for bi in range(ai + 1, len(dips)):
            dist = abs(dips[bi] - dips[ai])
            if abs(dist - gauge) < gauge_tol:
                xl = min(dips[ai], dips[bi])
                xr = max(dips[ai], dips[bi])
                midpoint = (xl + xr) / 2
                if abs(midpoint) <= own_x_range:
                    pairs.append((xl, xr))

    return pairs


def find_rail_lines(points, intensity, gauge=1.52, gauge_tol=0.15,
                     own_x_range=1.5, max_y_distance=25.0,
                     prev_rail_lines=None, debug=False):
    """
    Находит рельсы через непрерывные нити пиков + проверку колеи.

    Алгоритм:
    1. Grid-based RANSAC → плоскость земли
    2. Поперечные срезы по Y → Z-профиль → пики 3-40 см = головки рельсов
    3. Связывание пиков между соседними срезами в цепочки (нити):
       пик в срезе i+1 привязывается к ближайшему пику в срезе i (±300мм по X)
    4. Фильтрация: нити короче 5 срезов отбрасываются (шум)
    5. Поиск пар нитей на расстоянии колеи gauge ± gauge_tol
    6. Фит: каждая нить → набор узлов (y, x) → интерполяция

    prev_rail_lines: результат предыдущего кадра — используется как prior,
        чтобы пары не прыгали. Если задан, пары фильтруются: midpoint не
        дальше PRIOR_TOL от предыдущей позиции рельсов в этом Y.

    Возвращает:
        rail_lines: list of dict {'knots_y':…, 'knots_x':…}
        ground_plane: (a, b, c)
    """

    # Обрезаем по дальности
    if max_y_distance is not None and max_y_distance > 0:
        y_mask = np.abs(points[:, 1]) <= max_y_distance
        points = points[y_mask]
        if intensity is not None:
            intensity = intensity[y_mask]

    if len(points) < 100:
        return [], None

    # 1. Плоскость земли
    ground_plane = _estimate_ground_plane(points)
    if ground_plane is None:
        return [], None

    a, b, c = ground_plane
    z_ground = a * points[:, 0] + b * points[:, 1] + c
    dz = points[:, 2] - z_ground

    # Параметры
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    x_search_limit = own_x_range + gauge / 2 + gauge_tol

    SLICE_HALF_NEAR = 0.25
    SLICE_HALF_FAR = 2.0
    XBIN_NEAR = 0.05
    XBIN_FAR = 0.15
    ADAPT_NEAR = 10.0
    ADAPT_FAR = 200.0
    MIN_BIN_POINTS_NEAR = 1
    MIN_BIN_POINTS_FAR = 1
    MIN_BAND_POINTS_NEAR = 30
    MIN_BAND_POINTS_FAR = 10

    def _adapt(near_val, far_val, distance):
        t = np.clip((distance - ADAPT_NEAR) / (ADAPT_FAR - ADAPT_NEAR), 0.0, 1.0)
        return near_val + t * (far_val - near_val)

    # Генерируем срезы
    y_centers = []
    slice_halfs = []
    y = y_min + SLICE_HALF_NEAR
    while y < y_max - SLICE_HALF_NEAR:
        y_centers.append(y)
        dist = abs(y)
        slice_halfs.append(_adapt(SLICE_HALF_NEAR, SLICE_HALF_FAR, dist))
        step = 2.0 * _adapt(SLICE_HALF_NEAR, SLICE_HALF_FAR, dist)
        y += max(step, 0.3)

    if not y_centers:
        return [], ground_plane

    # Предсортировка по Y
    y_order = np.argsort(points[:, 1])
    pts_y_sorted = points[:, 1][y_order]
    pts_x_sorted = points[:, 0][y_order]
    dz_sorted = dz[y_order]

    # 1b. Ось туннеля — constraint для дальних дистанций.
    # Находим midline между стенами на каждом срезе ПЕРЕД поиском пиков,
    # чтобы расширить x_search_limit на поворотах (рельсы уходят по X).
    axis_y, axis_x = estimate_tunnel_axis(
        points, ground_plane, y_centers, slice_halfs)

    def _tunnel_axis_at(y_val):
        """X оси туннеля в точке Y. None если оси нет."""
        if axis_y is None:
            return None
        return float(np.interp(y_val, axis_y, axis_x))

    # Параметры constraint'ов
    TUNNEL_AXIS_TOL = 2.0    # макс. отклонение midpoint пары от оси, м
    NEAR_EXTRAP_MAX = 30.0   # до какой дальности считаем "надёжными"
    FAR_EXTRAP_START = 50.0  # с какой дальности включаем constraint экстраполяции
    EXTRAP_TOL = 1.0         # допуск экстраполяции, м

    # 2. Собираем пики по всем срезам
    slice_peaks = []  # list of list of (x, z) per slice
    for y_center in y_centers:
        dist = abs(y_center)
        slice_half = _adapt(SLICE_HALF_NEAR, SLICE_HALF_FAR, dist)
        x_bin = _adapt(XBIN_NEAR, XBIN_FAR, dist)
        min_band = int(_adapt(MIN_BAND_POINTS_NEAR, MIN_BAND_POINTS_FAR, dist))
        min_bin_pts = int(_adapt(MIN_BIN_POINTS_NEAR, MIN_BIN_POINTS_FAR, dist))

        # На повороте рельсы уходят по X вслед за осью туннеля —
        # расширяем область поиска пиков вокруг оси, а не вокруг X=0
        axis_x_here = _tunnel_axis_at(y_center)
        if axis_x_here is not None:
            x_lim = abs(axis_x_here) + own_x_range + gauge / 2 + gauge_tol
        else:
            x_lim = x_search_limit

        i_lo = np.searchsorted(pts_y_sorted, y_center - slice_half, side='right')
        i_hi = np.searchsorted(pts_y_sorted, y_center + slice_half, side='left')

        bp_x = pts_x_sorted[i_lo:i_hi]
        bdz = dz_sorted[i_lo:i_hi]

        bp_pts = np.column_stack([bp_x, np.zeros(len(bp_x)), np.zeros(len(bp_x))])
        peaks = _extract_peaks(bp_pts, bdz, y_center, slice_half, x_bin,
                               min_band, min_bin_pts, x_lim)
        slice_peaks.append(peaks)

    # 3. Находим пары в каждом срезе — fusion: пики + тени + полотно
    #
    # Три детектора работают ПАРАЛЛЕЛЬНО (не fallback):
    #   - Пики Z-профиля (головки рельсов выше земли)
    #   - Тени (пары провалов плотности на расстоянии колеи)
    #   - Полотно (связная полоса плотности шириной ~колеи)
    #
    # Все пары собираются вместе, дубли (близкие по X) мержатся.
    # Score = сумма подтверждений от каждого детектора.
    # Цепочки конкурируют по суммарному score.
    SHADOW_MATCH_TOL = 0.15  # допуск совпадения между детекторами по X, м
    TRACKBED_MATCH_TOL = 0.20

    slice_pairs = []        # list of list of (x_left, x_right)
    slice_pair_scores = []  # list of list of float

    for si, peaks in enumerate(slice_peaks):
        y_center = y_centers[si]
        dist = abs(y_center)
        axis_x_here = _tunnel_axis_at(y_center)

        # Получаем точки среза
        slice_half = _adapt(SLICE_HALF_NEAR, SLICE_HALF_FAR, dist)
        x_bin = _adapt(XBIN_NEAR, XBIN_FAR, dist)
        i_lo = np.searchsorted(pts_y_sorted, y_center - slice_half, side='right')
        i_hi = np.searchsorted(pts_y_sorted, y_center + slice_half, side='left')
        bp_x = pts_x_sorted[i_lo:i_hi]

        if axis_x_here is not None:
            x_lim = abs(axis_x_here) + own_x_range + gauge / 2 + gauge_tol
        else:
            x_lim = x_search_limit

        # Три детектора
        shadow_dips = _extract_shadow_dips(bp_x, x_bin, x_lim)
        trackbeds = _extract_trackbed(bp_x, x_bin, x_lim, gauge=gauge,
                                       gauge_tol=gauge_tol, own_x_range=own_x_range)

        def _midpoint_ok(midpoint):
            if axis_x_here is not None:
                return abs(midpoint - axis_x_here) <= TUNNEL_AXIS_TOL
            return abs(midpoint) <= own_x_range

        # Собираем кандидатов от всех детекторов: (xl, xr, source)
        # source: 'peak', 'shadow', 'trackbed'
        candidates = []

        # --- Пики ---
        # На дальних срезах пики шумнее — расширяем tolerance
        local_gauge_tol = gauge_tol + 0.05 + _adapt(0.0, 0.10, dist)
        for ai in range(len(peaks)):
            for bi in range(ai + 1, len(peaks)):
                d = abs(peaks[bi][0] - peaks[ai][0])
                if abs(d - gauge) < local_gauge_tol:
                    xl = min(peaks[ai][0], peaks[bi][0])
                    xr = max(peaks[ai][0], peaks[bi][0])
                    if _midpoint_ok((xl + xr) / 2):
                        candidates.append((xl, xr, 'peak'))

        # --- Тени ---
        if len(shadow_dips) >= 2:
            for ai in range(len(shadow_dips)):
                for bi in range(ai + 1, len(shadow_dips)):
                    d = abs(shadow_dips[bi] - shadow_dips[ai])
                    if abs(d - gauge) < gauge_tol:
                        xl = min(shadow_dips[ai], shadow_dips[bi])
                        xr = max(shadow_dips[ai], shadow_dips[bi])
                        if _midpoint_ok((xl + xr) / 2):
                            candidates.append((xl, xr, 'shadow'))

        # --- Полотно ---
        for tb_l, tb_r in trackbeds:
            if _midpoint_ok((tb_l + tb_r) / 2):
                candidates.append((tb_l, tb_r, 'trackbed'))

        # Мержим близкие кандидаты и считаем fusion score.
        # Группируем: два кандидата "одинаковые" если midpoint ± SHADOW_MATCH_TOL.
        # Внутри группы: позиция от пиков (точнее), score = сумма источников.
        pairs = []
        scores = []
        used = [False] * len(candidates)

        for ci in range(len(candidates)):
            if used[ci]:
                continue
            xl, xr, src = candidates[ci]
            mid = (xl + xr) / 2
            group = [(xl, xr, src)]
            used[ci] = True
            for cj in range(ci + 1, len(candidates)):
                if used[cj]:
                    continue
                mid2 = (candidates[cj][0] + candidates[cj][1]) / 2
                if abs(mid2 - mid) < SHADOW_MATCH_TOL:
                    group.append(candidates[cj])
                    used[cj] = True

            sources = {g[2] for g in group}

            # Позиция: приоритет пикам
            peak_entries = [g for g in group if g[2] == 'peak']
            if peak_entries:
                final_xl = peak_entries[0][0]
                final_xr = peak_entries[0][1]
            else:
                final_xl = float(np.median([g[0] for g in group]))
                final_xr = float(np.median([g[1] for g in group]))

            score = 0.0
            if 'peak' in sources:
                score += 1.0
            if 'shadow' in sources:
                score += 0.5
            if 'trackbed' in sources:
                score += 0.7

            pairs.append((final_xl, final_xr))
            scores.append(score)

        slice_pairs.append(pairs)
        slice_pair_scores.append(scores)

    # 3a. Anchor: зафиксировать "наш" путь по ближним срезам (|Y| < 5м).
    # Midpoint ближних пар = якорь. Все пары с midpoint дальше ANCHOR_TOL
    # от якоря — чужой путь, удаляем.
    ANCHOR_Y_MAX = 5.0
    ANCHOR_TOL = 1.5  # макс. отклонение midpoint от якоря, м
    anchor_mids = []
    for si, pairs in enumerate(slice_pairs):
        if pairs and abs(y_centers[si]) <= ANCHOR_Y_MAX:
            # Берём пару ближайшую к X=0
            best = min(pairs, key=lambda p: abs((p[0] + p[1]) / 2))
            anchor_mids.append((best[0] + best[1]) / 2)

    # Count pairs before anchor
    _pre_anchor = sum(len(p) for p in slice_pairs)

    if anchor_mids:
        anchor_x = float(np.median(anchor_mids))
        print(f'  [anchor] anchor_x={anchor_x:.3f}, pairs before={_pre_anchor}')
        for si in range(len(slice_pairs)):
            filtered = []
            filtered_scores = []
            for pi, (xl, xr) in enumerate(slice_pairs[si]):
                if abs((xl + xr) / 2 - anchor_x) <= ANCHOR_TOL:
                    filtered.append((xl, xr))
                    filtered_scores.append(slice_pair_scores[si][pi])
            slice_pairs[si] = filtered
            slice_pair_scores[si] = filtered_scores
        _post_anchor = sum(len(p) for p in slice_pairs)
        print(f'  [anchor] pairs after={_post_anchor}')
    else:
        print(f'  [anchor] no anchor found, pairs={_pre_anchor}')

    # 3b. Экстраполяция с ближних — дополнительный фильтр на дальних дистанциях
    near_mids_y = []
    near_mids_x = []
    for si, pairs in enumerate(slice_pairs):
        if pairs and abs(y_centers[si]) <= NEAR_EXTRAP_MAX:
            xl, xr = pairs[0]
            near_mids_y.append(y_centers[si])
            near_mids_x.append((xl + xr) / 2)

    extrap_fit = None
    if len(near_mids_y) >= 3:
        near_mids_y = np.array(near_mids_y)
        near_mids_x = np.array(near_mids_x)
        deg = 2 if (near_mids_y.max() - near_mids_y.min()) > 10.0 else 1
        extrap_fit = np.polyfit(near_mids_y, near_mids_x, deg)

    # Экстраполяция применяется ТОЛЬКО если нет оси туннеля.
    # Ось — лучший constraint, а экстраполяция квадратичная с 30м
    # промахивается далеко за пределами данных.
    if extrap_fit is not None and axis_y is None:
        for si in range(len(slice_pairs)):
            y_center = y_centers[si]
            if abs(y_center) <= FAR_EXTRAP_START:
                continue
            if not slice_pairs[si]:
                continue
            expected_mid = float(np.polyval(extrap_fit, y_center))
            filtered = []
            filtered_scores = []
            for pi, (xl, xr) in enumerate(slice_pairs[si]):
                midpoint = (xl + xr) / 2
                if abs(midpoint - expected_mid) <= EXTRAP_TOL:
                    filtered.append((xl, xr))
                    filtered_scores.append(slice_pair_scores[si][pi])
            if not filtered and slice_pairs[si]:
                best_pi = min(range(len(slice_pairs[si])),
                              key=lambda k: abs((slice_pairs[si][k][0] + slice_pairs[si][k][1]) / 2 - expected_mid))
                best = slice_pairs[si][best_pi]
                if abs((best[0] + best[1]) / 2 - expected_mid) <= EXTRAP_TOL * 2:
                    filtered = [best]
                    filtered_scores = [slice_pair_scores[si][best_pi]]
            slice_pairs[si] = filtered
            slice_pair_scores[si] = filtered_scores

    # 4. Связываем пары между срезами в цепочки
    # Каждая цепочка — последовательность пар, где оба рельса
    # сместились плавно (±LINK_TOL) между соседними срезами.
    #
    # На повороте рельсы сдвигаются на 0.5-1.0м между соседними дальними срезами.
    # Вместо огромного LINK_TOL (который бы ловил мусор) используем два режима:
    # - Если есть ось туннеля: проверяем что КОЛЕЯ сохранилась (gauge_shift)
    #   и midpoint не далеко от оси. Абсолютный сдвиг по X может быть большим.
    # - Если оси нет: классический LINK_TOL от предыдущего среза.
    LINK_TOL_NEAR = 0.30
    LINK_TOL_FAR = 0.50
    GAUGE_SHIFT_TOL = 0.20  # колея не должна скакать >200мм между срезами
    MAX_GAP = 5              # на дальних дистанциях много пустых срезов

    # chain entries: (si, xl, xr, score)
    active_chains = []   # [(last_si, last_xl, last_xr, chain)]
    finished_chains = []

    # Построить lookup: slice_idx → {pair_idx → score}
    _pair_score_lookup = []
    for si in range(len(slice_pairs)):
        lookup = {}
        for pi in range(len(slice_pairs[si])):
            lookup[pi] = slice_pair_scores[si][pi] if pi < len(slice_pair_scores[si]) else 1.0
        _pair_score_lookup.append(lookup)

    for si, pairs in enumerate(slice_pairs):
        used = set()
        new_active = []
        y_here = y_centers[si]
        dist_here = abs(y_here)
        link_tol = _adapt(LINK_TOL_NEAR, LINK_TOL_FAR, dist_here)
        has_axis = axis_y is not None

        for ch_si, ch_xl, ch_xr, chain in active_chains:
            gap = si - ch_si
            if gap > MAX_GAP + 1:
                finished_chains.append(chain)
                continue

            ch_gauge = ch_xr - ch_xl
            best_pi = None
            best_cost = 1e9

            # Макс. допустимый сдвиг midpoint пропорционален gap
            # (на повороте ось смещается ~0.3м/срез)
            max_mid_shift = 0.5 * (gap + 1) if has_axis else link_tol * 2

            for pi, (xl, xr) in enumerate(pairs):
                if pi in used:
                    continue

                if has_axis and dist_here > ADAPT_NEAR:
                    new_gauge = xr - xl
                    gauge_shift = abs(new_gauge - ch_gauge)
                    if gauge_shift > GAUGE_SHIFT_TOL:
                        continue
                    mid_shift = abs((xl + xr) / 2 - (ch_xl + ch_xr) / 2)
                    if mid_shift > max_mid_shift:
                        continue
                    cost = mid_shift
                else:
                    if abs(xl - ch_xl) >= link_tol or abs(xr - ch_xr) >= link_tol:
                        continue
                    cost = abs(xl - ch_xl) + abs(xr - ch_xr)

                if cost < best_cost:
                    best_cost = cost
                    best_pi = pi

            if best_pi is not None:
                xl, xr = pairs[best_pi]
                used.add(best_pi)
                pair_score = _pair_score_lookup[si].get(best_pi, 1.0)
                chain.append((si, xl, xr, pair_score))
                new_active.append((si, xl, xr, chain))
            else:
                new_active.append((ch_si, ch_xl, ch_xr, chain))

        # Новые пары начинают новые цепочки
        for pi, (xl, xr) in enumerate(pairs):
            if pi not in used:
                pair_score = _pair_score_lookup[si].get(pi, 1.0)
                new_active.append((si, xl, xr, [(si, xl, xr, pair_score)]))

        active_chains = new_active

    for _, _, _, chain in active_chains:
        finished_chains.append(chain)

    # 5. Лучшая цепочка — длина × средний score, ближайшая к лидару
    MIN_CHAIN_LEN = 5
    chains = [c for c in finished_chains if len(c) >= MIN_CHAIN_LEN]

    if not chains:
        return [], ground_plane

    def chain_score(c):
        mid = np.mean([(p[1] + p[2]) / 2 for p in c])
        total_score = sum(p[3] for p in c)  # сумма fusion scores
        # Жёсткий штраф: если midpoint далеко от лидара — не наш путь
        closeness = max(0.0, 1.0 - abs(mid) / own_x_range)
        return total_score * closeness
    chains.sort(key=chain_score, reverse=True)
    best_chain = chains[0]

    # 6. Строим rail_lines из цепочки пар
    from scipy.interpolate import UnivariateSpline

    chain_ys = np.array([y_centers[p[0]] for p in best_chain])
    chain_xl = np.array([p[1] for p in best_chain])
    chain_xr = np.array([p[2] for p in best_chain])

    # DEBUG: показать пары в цепочке
    for i in range(len(best_chain)):
        y = chain_ys[i]
        xl = chain_xl[i]
        xr = chain_xr[i]
        mid = (xl + xr) / 2
        print(f'    chain Y={y:6.1f}  xl={xl:+.3f}  xr={xr:+.3f}  mid={mid:+.3f}  gauge={xr-xl:.3f}')

    rail_lines = []
    for xs_raw in [chain_xl, chain_xr]:
        order = np.argsort(chain_ys)
        ys_s = chain_ys[order]
        xs_s = xs_raw[order]

        if len(ys_s) < 4 or ys_s.max() - ys_s.min() < 1.0:
            rail_lines.append((0.0, float(xs_s.mean())))
        else:
            # Линейная интерполяция по узлам цепочки — не улетает за пределы
            rail_lines.append({'knots_y': ys_s.copy(),
                               'knots_x': xs_s.copy()})

    # Левый рельс — слева от лидара (X < 0), правый — справа (X > 0)
    x0 = rail_x(rail_lines[0], 0)
    x1 = rail_x(rail_lines[1], 0)
    if x0 > x1:
        rail_lines[0], rail_lines[1] = rail_lines[1], rail_lines[0]
        x0, x1 = x1, x0

    print(f'  [rail_detection] rail X at Y=0: left={x0:.3f}, right={x1:.3f}, mid={(x0+x1)/2:.3f}')

    if debug:
        # Собираем индексы срезов из лучшей цепочки
        chain_si_set = {p[0] for p in best_chain}
        debug_info = {
            'y_centers': y_centers,
            'slice_peaks': slice_peaks,       # все пики по срезам
            'slice_pairs': slice_pairs,       # все пары после фильтрации
            'slice_pair_scores': slice_pair_scores,
            'best_chain': best_chain,         # (si, xl, xr, score)
            'chain_ys': chain_ys,
            'chain_xl': chain_xl,
            'chain_xr': chain_xr,
            'all_chains': chains,             # все цепочки >= MIN_CHAIN_LEN
            'ground_plane': ground_plane,
        }
        return rail_lines, ground_plane, debug_info

    return rail_lines, ground_plane


def _find_own_track(rail_lines):
    """Возвращает индекс пары (ti, ti+1) ближайшей к лидару (X=0)."""
    best_ti = None
    best_dist = float('inf')
    for ti in range(0, len(rail_lines), 2):
        if ti + 1 >= len(rail_lines):
            break
        mid_x = abs((rail_x(rail_lines[ti], 0) + rail_x(rail_lines[ti + 1], 0)) / 2)
        if mid_x < best_dist:
            best_dist = mid_x
            best_ti = ti
    return best_ti


def _robust_median(values, iters=3, mad_k=3.0):
    """Медиана с итеративной отбраковкой выбросов по MAD (по умолчанию 3 итерации)."""
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return None
    center = float(np.median(v))
    for _ in range(iters):
        mad = float(np.median(np.abs(v - center)))
        if mad <= 0:
            break
        keep = np.abs(v - center) <= mad_k * 1.4826 * mad
        if keep.all() or not keep.any():
            break
        v = v[keep]
        center = float(np.median(v))
    return center


def _slice_center(rail_pts, y, slice_half=0.5, center_window=None, center_iters=3,
                  min_points=0):
    """Центр нити в срезе y по оси X.

    center_window=None — прежнее поведение: медиана X внутри среза ±slice_half.
    center_window=W    — устойчивая медиана по Y-окну ±W с отбраковкой по MAD;
                         выбросы в окне (соседние объекты, шум) не тянут центр.
    min_points         — минимум точек в окне (0 = без порога, как в прежнем коде);
                         на редких дальних срезах медиана по 1-2 точкам бессмысленна.
    """
    if center_window is None:
        band = (rail_pts[:, 1] > y - slice_half) & (rail_pts[:, 1] < y + slice_half)
        if not band.any() or band.sum() < min_points:
            return None
        # np.float32 без приведения к float64 — чтобы прежнее число совпало побитово.
        return np.median(rail_pts[band, 0])
    band = np.abs(rail_pts[:, 1] - y) < center_window
    if not band.any() or band.sum() < min_points:
        return None
    return _robust_median(rail_pts[band, 0], iters=center_iters)


def _gauge_stats(values):
    """Сводка по набору срезов (метры → метры, печать в мм делает вызывающий)."""
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return None
    return {'values': v, 'mean': float(v.mean()), 'std': float(v.std()),
            'min': float(v.min()), 'max': float(v.max()), 'n': int(v.size)}


def measure_gauge(points, rail_mask, rail_lines, ground_plane, pair_index=None,
                  slice_step=1.0, slice_half=0.5, center_window=1.5,
                  center_iters=3, center_min_points=0, own_offset=0.1,
                  head_half_width=0.15, head_depth=0.013, head_top_pct=98.0,
                  head_slice_half=1.5, head_min_points=25, head_pct=95.0):
    """Колея своей пары нитей в двух базах отсчёта.

    axes  — расстояние между центрами нитей (прежняя база). Центр среза считается
            через _slice_center: при center_window=W — устойчивая медиана по
            Y-окну ±W, при center_window=None — медиана в ±slice_half (как было).
            «Среднее по срезам» не меняется.
    inner — нормативная база: расстояние между внутренними рабочими гранями
            головок на уровне head_depth ниже поверхности катания (по умолчанию
            13 мм, ПТЭ п.45). Грани берутся как head_pct/100-head_pct перцентили
            футпринта головки (точки в ±head_half_width от линии, полоса
            head_depth под верхом головки). Левая нить пары — с меньшим X, её
            внутренняя грань обращена в +X, правая — в −X. Y-диапазон тот же,
            что у базы по осям (пересечение нитей маски).

    Ничего не меняет в rail_lines/rail_mask: только читает points и маску.
    Возвращает dict: 'pair_index', 'axes', 'inner' (сводки _gauge_stats) и
    'reason' — почему база не посчитана (None, если посчитаны обе).
    """
    result = {'pair_index': pair_index, 'axes': None, 'inner': None, 'reason': None}

    if not rail_lines or ground_plane is None or len(points) == 0:
        result['reason'] = 'нет линий/плоскости/точек'
        return result

    if pair_index is None:
        pair_index = _find_own_track(rail_lines)
    if pair_index is None or pair_index + 1 >= len(rail_lines):
        result['reason'] = 'своя пара нитей не найдена'
        return result
    result['pair_index'] = pair_index

    line1 = rail_lines[pair_index]        # нить с меньшим X
    line2 = rail_lines[pair_index + 1]    # нить с большим X

    rail_pts = points[rail_mask]
    if len(rail_pts) == 0:
        result['reason'] = 'пустая рельсовая маска'
        return result

    r1 = rail_pts[np.abs(rail_pts[:, 0] - rail_x(line1, rail_pts[:, 1])) < own_offset]
    r2 = rail_pts[np.abs(rail_pts[:, 0] - rail_x(line2, rail_pts[:, 1])) < own_offset]
    if len(r1) <= 5 or len(r2) <= 5:
        result['reason'] = 'мало точек в нитях своей пары'
        return result

    y_lo = max(r1[:, 1].min(), r2[:, 1].min())
    y_hi = min(r1[:, 1].max(), r2[:, 1].max())

    dists = []
    for y in np.arange(y_lo + slice_half, y_hi - slice_half, slice_step):
        b1 = (r1[:, 1] > y - slice_half) & (r1[:, 1] < y + slice_half)
        b2 = (r2[:, 1] > y - slice_half) & (r2[:, 1] < y + slice_half)
        if b1.sum() > 0 and b2.sum() > 0:
            c1 = _slice_center(r1, y, slice_half, center_window, center_iters,
                               center_min_points)
            c2 = _slice_center(r2, y, slice_half, center_window, center_iters,
                               center_min_points)
            if c1 is not None and c2 is not None:
                dists.append(abs(c2 - c1))
    result['axes'] = _gauge_stats(dists)

    a, b, c = ground_plane
    dz = points[:, 2] - (a * points[:, 0] + b * points[:, 1] + c)

    def footprint(line):
        off = points[:, 0] - rail_x(line, points[:, 1])
        head = (np.abs(off) < head_half_width) & (dz > 0.03) & (dz < 0.40)
        if head.sum() < 30:
            return None
        top = float(np.percentile(dz[head], head_top_pct))  # поверхность катания
        band = head & (dz >= top - head_depth) & (dz <= top)
        if band.sum() < 30:
            return None
        return off[band], points[band, 1]

    fp1 = footprint(line1)
    fp2 = footprint(line2)
    if fp1 is None or fp2 is None:
        result['reason'] = ('футпринт головки не найден, база по осям тоже не посчитана'
                            if result['axes'] is None else 'футпринт головки не найден')
        return result

    o1, y1 = fp1
    o2, y2 = fp2
    inner = []
    for y in np.arange(y_lo + slice_half, y_hi - slice_half, slice_step):
        b1 = np.abs(y1 - y) < head_slice_half
        b2 = np.abs(y2 - y) < head_slice_half
        if b1.sum() < head_min_points or b2.sum() < head_min_points:
            continue
        left_inner = rail_x(line1, y) + np.percentile(o1[b1], head_pct)
        right_inner = rail_x(line2, y) + np.percentile(o2[b2], 100.0 - head_pct)
        inner.append(right_inner - left_inner)
    result['inner'] = _gauge_stats(inner)

    if result['axes'] is None and result['inner'] is None:
        result['reason'] = 'ни одна база не посчитана'
    return result


def apply_rail_lines(points, rail_lines, ground_plane, rail_radius=0.075,
                     own_track_only=False, dz_min=-0.02, dz_max=0.40,
                     refine_axis=False, refine_step=2.0, max_y_distance=25.0):
    """
    Быстрое применение найденных рельсовых линий к новому кадру.

    Маска: все точки в полосе ±rail_radius от оси рельса (по X),
    с высотой над землёй от dz_min до dz_max. Захватывает полный
    силуэт рельса — подошва, шейка, головка.

    refine_axis=True — корректирует ось рельса локально (в окнах по Y
    шириной refine_step) по медиане X точек головки, чтобы маска была
    симметрична даже на поворотах.

    own_track_only=True — только путь поезда (ближайший к лидару).
    Возвращает булеву маску (True = рельс).
    """
    n = len(points)
    rail_mask = np.zeros(n, dtype=bool)

    if not rail_lines or ground_plane is None or n == 0:
        return rail_mask

    if own_track_only:
        best_ti = _find_own_track(rail_lines)
        if best_ti is None:
            return rail_mask
        lines_to_use = [rail_lines[best_ti], rail_lines[best_ti + 1]]
    else:
        lines_to_use = list(rail_lines)

    a, b, c = ground_plane
    z_ground = a * points[:, 0] + b * points[:, 1] + c
    dz = points[:, 2] - z_ground
    height_ok = (dz > dz_min) & (dz < dz_max)
    if max_y_distance is not None and max_y_distance > 0:
        height_ok &= np.abs(points[:, 1]) <= max_y_distance
    height_idx = np.where(height_ok)[0]

    if len(height_idx) == 0:
        return rail_mask

    h_y = points[height_idx, 1]
    h_x = points[height_idx, 0]
    h_dz = dz[height_idx]

    for line in lines_to_use:
        x_rail = rail_x(line, h_y)
        x_off = h_x - x_rail

        if refine_axis:
            # Локальная коррекция оси в окнах по Y.
            # Берём только верхушку головки (top 20% по dz в окне) —
            # при боковом обзоре точки стенки рельса асимметричны
            # и сдвигают медиану, а верхушка самая узкая и точная.
            y_min, y_max = h_y.min(), h_y.max()
            corrected_off = x_off.copy()
            for y_center in np.arange(y_min, y_max + refine_step, refine_step):
                y_band = (h_y >= y_center - refine_step) & (h_y < y_center + refine_step)
                near_rail = y_band & (np.abs(x_off) < 0.08) & (h_dz > 0.10)
                if near_rail.sum() > 10:
                    # Только top 20% по высоте — верхушка головки
                    dz_thresh = np.percentile(h_dz[near_rail], 80)
                    head_top = near_rail & (h_dz >= dz_thresh)
                    if head_top.sum() > 3:
                        shift = float(np.median(x_off[head_top]))
                    else:
                        shift = float(np.median(x_off[near_rail]))
                    corrected_off[y_band] = x_off[y_band] - shift
            x_off = corrected_off

        close = np.abs(x_off) < rail_radius
        rail_mask[height_idx[close]] = True

    return rail_mask


def detect_rails(points, intensity, rail_lines=None, ground_plane=None,
                 rail_radius=0.04):
    """
    Обёртка: если rail_lines заданы — быстрое применение,
    иначе — полная детекция (find + apply).
    """
    if rail_lines is not None:
        return apply_rail_lines(points, rail_lines, ground_plane, rail_radius)

    lines, gplane = find_rail_lines(points, intensity)
    return apply_rail_lines(points, lines, gplane, rail_radius)


def build_rail_meshes(points, rail_mask, rail_lines, ground_plane,
                      rail_width=0.075, rail_height=0.15, step=0.5):
    """
    Строит 3D-меши рельсов по реальным точкам (повторяет кривые и стрелки).

    Для каждого рельса: берёт точки rail_mask, усредняет X в полосах по Y,
    строит прямоугольный профиль вдоль получившейся кривой.

    Параметры:
        rail_width: ширина профиля (по умолчанию 75 мм — головка Р65)
        rail_height: высота профиля (по умолчанию 150 мм)
        step: шаг по Y (м)
    """
    meshes = []
    if not rail_lines or ground_plane is None:
        return meshes

    a, b, c = ground_plane
    hw = rail_width / 2

    rail_pts = points[rail_mask]
    if len(rail_pts) == 0:
        return meshes

    for line in rail_lines:
        # Выбрать точки, принадлежащие этому рельсу (ближайшие к линии)
        x_expected = rail_x(line, rail_pts[:, 1])
        dx = np.abs(rail_pts[:, 0] - x_expected)
        belong = dx < 0.1
        rp = rail_pts[belong]

        if len(rp) < 5:
            continue

        # Усреднить X в полосах по Y → кривая рельса
        y_min, y_max = rp[:, 1].min(), rp[:, 1].max()
        y_vals = np.arange(y_min, y_max, step)
        if len(y_vals) < 2:
            continue

        centerline = []  # (x, y, z_base)
        for y in y_vals:
            band = (rp[:, 1] >= y - step / 2) & (rp[:, 1] < y + step / 2)
            if band.sum() > 0:
                x_center = np.median(rp[band, 0])
            else:
                x_center = rail_x(line, y)
            z_base = a * x_center + b * y + c
            centerline.append((x_center, y, z_base))

        # Построить меш вдоль centerline
        verts = []
        tris = []
        for yi, (xc, y, zb) in enumerate(centerline):
            zt = zb + rail_height
            verts.append([xc - hw, y, zb])
            verts.append([xc + hw, y, zb])
            verts.append([xc + hw, y, zt])
            verts.append([xc - hw, y, zt])

            if yi > 0:
                prev = (yi - 1) * 4
                cur = yi * 4
                for face in range(4):
                    nf = (face + 1) % 4
                    tris.append([prev + face, cur + face, cur + nf])
                    tris.append([prev + face, cur + nf, prev + nf])

        if len(centerline) > 1:
            tris.append([0, 1, 2])
            tris.append([0, 2, 3])
            last = (len(centerline) - 1) * 4
            tris.append([last, last + 2, last + 1])
            tris.append([last, last + 3, last + 2])

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(np.array(verts))
        mesh.triangles = o3d.utility.Vector3iVector(np.array(tris))
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([0.85, 0.15, 0.15])
        meshes.append(mesh)

    return meshes
