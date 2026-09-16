"""
Rail detection and gauge measurement for LiDAR point clouds.

Functions extracted from visualize_bag.py for reuse and modification.
"""

import numpy as np
import open3d as o3d


def _estimate_ground_plane(points, cell_size=1.0):
    """
    Робастная оценка плоскости земли через grid-based подход.

    Делит облако на ячейки cell_size × cell_size, берёт 5-й перцентиль Z
    в каждой ячейке (≈ уровень земли), затем RANSAC по сетке.
    """
    from sklearn.linear_model import RANSACRegressor

    z = points[:, 2]
    x_cells = np.arange(points[:, 0].min() - 0.5, points[:, 0].max() + 0.5, cell_size)
    y_cells = np.arange(points[:, 1].min() - 0.5, points[:, 1].max() + 0.5, cell_size)

    gx, gy, gz = [], [], []
    for xi in range(len(x_cells) - 1):
        xmask = (points[:, 0] >= x_cells[xi]) & (points[:, 0] < x_cells[xi + 1])
        for yi in range(len(y_cells) - 1):
            mask = xmask & (points[:, 1] >= y_cells[yi]) & (points[:, 1] < y_cells[yi + 1])
            if mask.sum() > 10:
                gx.append((x_cells[xi] + x_cells[xi + 1]) / 2)
                gy.append((y_cells[yi] + y_cells[yi + 1]) / 2)
                gz.append(np.percentile(z[mask], 5))

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


def find_rail_lines(points, intensity, gauge=1.52, gauge_tol=0.15,
                     own_x_range=1.5, max_y_distance=20.0):
    """
    Находит рельсы через поперечные Z-профили + ограничение по колее.

    own_x_range — максимальное смещение центра колеи от лидара (X=0), м.
        На прямой рельсы на ±0.76м (midpoint ≈ 0), на повороте смещаются,
        но midpoint не уходит дальше ±1.5м. Пары с midpoint за пределами
        own_x_range отбрасываются до кластеризации — не ловим чужие пути.
    max_y_distance — максимальная дальность по Y от лидара (Y=0), м.
        Точки дальше обрезаются перед поиском — на большом расстоянии
        рельсы расходятся из-за шума и разреженности облака.

    Алгоритм:
    1. Grid-based RANSAC → плоскость земли
    2. Поперечные срезы по Y → Z-профиль → пики 3-25 см над землёй = головки рельсов
    3. Пары пиков с расстоянием gauge ± gauge_tol → рельсовая колея
       (только пары с midpoint в ±own_x_range от лидара)
    4. Кластеризация пар → отдельные пути
    5. Фит линий x(y) для каждого рельса (приоритет — ближайший к лидару)

    Возвращает:
        rail_lines: list of (slope, intercept) — линии x = slope*y + intercept
        ground_plane: (a, b, c) — коэффициенты плоскости z = a*x + b*y + c
    """
    from sklearn.cluster import DBSCAN

    # Обрезаем по дальности (Y) — дальше max_y_distance рельсы уходят косо
    if max_y_distance is not None and max_y_distance > 0:
        y_mask = np.abs(points[:, 1]) <= max_y_distance
        points = points[y_mask]
        if intensity is not None:
            intensity = intensity[y_mask]

    if len(points) < 100:
        return [], None

    # 1. Robust ground plane via grid percentiles
    ground_plane = _estimate_ground_plane(points)
    if ground_plane is None:
        return [], None

    a, b, c = ground_plane

    z_ground = a * points[:, 0] + b * points[:, 1] + c
    dz = points[:, 2] - z_ground

    # 2. Scan cross-sections, find height peaks, match gauge pairs
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    all_pairs = []  # (y, x_left, x_right)

    # Ограничиваем X-диапазон поиска: лидар по центру колеи,
    # рельсы не дальше own_x_range + полколеи от X=0
    x_search_limit = own_x_range + gauge / 2 + gauge_tol

    for y_center in np.arange(y_min + 0.5, y_max - 0.5, 0.5):
        band = (points[:, 1] > y_center - 0.25) & (points[:, 1] < y_center + 0.25)
        if band.sum() < 30:
            continue

        bp = points[band]
        bdz = dz[band]

        # Ограничиваем бины по X — не ищем пики далеко от лидара
        x_lo = max(bp[:, 0].min(), -x_search_limit)
        x_hi = min(bp[:, 0].max(), x_search_limit)
        if x_hi - x_lo < gauge - gauge_tol:
            continue

        x_bins = np.arange(x_lo, x_hi, 0.05)
        if len(x_bins) < 3:
            continue

        # Z-profile: 80th percentile per X-bin (catches rail head sticking up)
        x_centers = []
        z_profile = []
        for xi in range(len(x_bins) - 1):
            xmask = (bp[:, 0] >= x_bins[xi]) & (bp[:, 0] < x_bins[xi + 1])
            if xmask.sum() > 0:
                x_centers.append((x_bins[xi] + x_bins[xi + 1]) / 2)
                z_profile.append(np.percentile(bdz[xmask], 80))

        if len(x_centers) < 3:
            continue

        x_centers = np.array(x_centers)
        z_profile = np.array(z_profile)

        # Rail peaks: 3-40cm above ground (head height; 25cm мало при
        # перекосе путей — один рельс может быть на 10+ см выше другого)
        rail_height = (z_profile > 0.03) & (z_profile < 0.40)
        if not rail_height.any():
            continue

        rx = x_centers[rail_height]
        rz = z_profile[rail_height]

        # Group adjacent bins into peaks
        groups = [[0]]
        for j in range(1, len(rx)):
            if rx[j] - rx[j - 1] < 0.15:
                groups[-1].append(j)
            else:
                groups.append([j])

        peaks = [(rx[g[np.argmax(rz[g])]], rz[g].max()) for g in groups]

        # Find pairs at gauge distance, only if midpoint near lidar
        for ai in range(len(peaks)):
            for bi in range(ai + 1, len(peaks)):
                dist = abs(peaks[bi][0] - peaks[ai][0])
                if abs(dist - gauge) < gauge_tol:
                    midpoint = (peaks[ai][0] + peaks[bi][0]) / 2
                    if abs(midpoint) <= own_x_range:
                        all_pairs.append((y_center, peaks[ai][0], peaks[bi][0]))

    if len(all_pairs) < 3:
        return [], ground_plane

    pairs = np.array(all_pairs)
    midpoints = (pairs[:, 1] + pairs[:, 2]) / 2

    # 3. Cluster pairs by track (midpoint X)
    db = DBSCAN(eps=0.5, min_samples=3).fit(midpoints.reshape(-1, 1))
    labels = db.labels_

    # 4. Fit rail lines — приоритет кластеру ближайшему к лидару (X=0)
    clusters = sorted(set(labels) - {-1})
    if not clusters:
        return [], ground_plane

    # Сортируем кластеры по близости midpoint к X=0 (свой путь первый)
    cluster_mid = {}
    for lbl in clusters:
        cluster_mid[lbl] = abs(np.median(midpoints[labels == lbl]))
    clusters.sort(key=lambda lbl: cluster_mid[lbl])

    rail_lines = []
    for lbl in clusters:
        mask = labels == lbl
        p = pairs[mask]
        y_vals = p[:, 0]

        for rail_idx in [1, 2]:  # left and right rail
            x_vals = p[:, rail_idx]
            if y_vals.max() - y_vals.min() < 1.0:
                rail_lines.append((0.0, x_vals.mean()))
            else:
                coeffs = np.polyfit(y_vals, x_vals, 1)
                rail_lines.append((coeffs[0], coeffs[1]))

    return rail_lines, ground_plane


def _find_own_track(rail_lines):
    """Возвращает индекс пары (ti, ti+1) ближайшей к лидару (X=0)."""
    best_ti = None
    best_dist = float('inf')
    for ti in range(0, len(rail_lines), 2):
        if ti + 1 >= len(rail_lines):
            break
        mid_x = abs((rail_lines[ti][1] + rail_lines[ti + 1][1]) / 2)
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

    s1, i1 = rail_lines[pair_index]        # нить с меньшим X
    s2, i2 = rail_lines[pair_index + 1]    # нить с большим X

    rail_pts = points[rail_mask]
    if len(rail_pts) == 0:
        result['reason'] = 'пустая рельсовая маска'
        return result

    r1 = rail_pts[np.abs(rail_pts[:, 0] - (s1 * rail_pts[:, 1] + i1)) < own_offset]
    r2 = rail_pts[np.abs(rail_pts[:, 0] - (s2 * rail_pts[:, 1] + i2)) < own_offset]
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

    def footprint(slope, intercept):
        off = points[:, 0] - (slope * points[:, 1] + intercept)
        head = (np.abs(off) < head_half_width) & (dz > 0.03) & (dz < 0.40)
        if head.sum() < 30:
            return None
        top = float(np.percentile(dz[head], head_top_pct))  # поверхность катания
        band = head & (dz >= top - head_depth) & (dz <= top)
        if band.sum() < 30:
            return None
        return off[band], points[band, 1]

    fp1 = footprint(s1, i1)
    fp2 = footprint(s2, i2)
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
        left_inner = (s1 * y + i1) + np.percentile(o1[b1], head_pct)
        right_inner = (s2 * y + i2) + np.percentile(o2[b2], 100.0 - head_pct)
        inner.append(right_inner - left_inner)
    result['inner'] = _gauge_stats(inner)

    if result['axes'] is None and result['inner'] is None:
        result['reason'] = 'ни одна база не посчитана'
    return result


def apply_rail_lines(points, rail_lines, ground_plane, rail_radius=0.075,
                     own_track_only=False, dz_min=-0.02, dz_max=0.40,
                     refine_axis=False, refine_step=2.0, max_y_distance=20.0):
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

    for slope, intercept in lines_to_use:
        x_rail = slope * h_y + intercept
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

    for slope, intercept in rail_lines:
        # Выбрать точки, принадлежащие этому рельсу (ближайшие к линии)
        x_expected = slope * rail_pts[:, 1] + intercept
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
                x_center = slope * y + intercept
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
