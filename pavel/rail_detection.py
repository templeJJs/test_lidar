"""
Rail detection and gauge measurement for LiDAR point clouds.

Functions extracted from visualize_bag.py for reuse and modification.
"""

import numpy as np
import open3d as o3d


def _estimate_ground_plane(points, cell_size=1.0, x_half_width=2.5, max_tilt=0.2):
    """
    Робастная оценка плоскости земли через grid-based подход.

    Делит облако на ячейки cell_size × cell_size, берёт 5-й перцентиль Z
    в каждой ячейке (≈ уровень земли), затем RANSAC по сетке.

    x_half_width — полуширина коридора вокруг лидара, по которому считается
        земля. Ограничение обязательно: на станции платформа возвышается над
        путём примерно на 1.4 м и занимает бо́льшую часть облака, поэтому
        RANSAC по всей ширине фитит плоскость через платформу. Это давало
        поперечный уклон |a| до 0.42 со скачущим знаком и, как следствие,
        отказ детекции (ground=None) на кадрах у платформы.
    """
    from sklearn.linear_model import RANSACRegressor

    if x_half_width is not None:
        near = np.abs(points[:, 0]) <= x_half_width
        if near.sum() >= 200:
            points = points[near]

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

    if abs(a) > max_tilt or abs(b) > max_tilt:
        return None

    return (a, b, c)


def estimate_ground_plane(points, cell_size=1.0, x_half_widths=(2.5, 4.0, 8.0),
                          max_tilts=(0.06, 0.12, 0.2)):
    """Плоскость земли с расширением коридора и допуска при неудаче.

    Узкий коридор (2.5 м) отсекает платформу, но на некоторых кадрах в нём
    остаётся слишком мало заполненных ячеек для RANSAC. Тогда пробуем шире:
    лучше оценка по более широкой полосе, чем полный отказ детекции.

    Допуск на уклон тоже наращивается, и сначала он жёсткий. Прежний порог
    0.2 пропускал явно ложные плоскости: на 22-43% кадров (по трём бэгам)
    поперечный уклон выходил за 0.05, доходя до 0.18, при медиане около нуля.
    Завал плоскости смещает dz, а по dz ищутся головки рельсов, так что
    профиль в срезе переставал быть профилем пути. Реальный вираж метро даёт
    уклон до ~0.1, поэтому 0.06 отсекает выбросы, не трогая настоящие кривые,
    а более мягкие ступени остаются на случай, когда иначе оценки не будет.
    """
    for max_tilt in max_tilts:
        for hw in x_half_widths:
            plane = _estimate_ground_plane(points, cell_size=cell_size,
                                           x_half_width=hw, max_tilt=max_tilt)
            if plane is not None:
                return plane
    return None


def _wmedian(values, weights):
    """Взвешенная медиана — устойчивая к выбросам замена взвешенному среднему."""
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if v.size == 0:
        return 0.0
    order = np.argsort(v)
    v, w = v[order], w[order]
    cw = np.cumsum(w)
    if cw[-1] <= 0:
        return float(np.median(v))
    return float(v[np.searchsorted(cw, 0.5 * cw[-1])])


def _fit_line_robust(y_vals, x_vals, iters=3, mad_k=2.5, min_keep=3, w=None):
    """Фит x = slope*y + intercept с итеративной отбраковкой по MAD.

    Обычный polyfit минимизирует квадрат ошибки, поэтому одна пара, пережившая
    фильтр консенсуса, всё ещё заметно тянет линию — а на плече 30 м даже
    небольшой перекос наклона даёт дециметры на дальнем конце.

    w — необязательные веса наблюдений (у np.polyfit вес входит линейно
    в невязку, то есть как sqrt от веса в квадратичной форме).
    """
    y = np.asarray(y_vals, dtype=np.float64)
    x = np.asarray(x_vals, dtype=np.float64)
    w = None if w is None else np.asarray(w, dtype=np.float64)

    coeffs = np.polyfit(y, x, 1, w=None if w is None else np.sqrt(w))
    for _ in range(iters):
        resid = x - np.polyval(coeffs, y)
        mad = float(np.median(np.abs(resid - np.median(resid))))
        if mad <= 0:
            break
        keep = np.abs(resid - np.median(resid)) <= mad_k * 1.4826 * mad
        if keep.all() or keep.sum() < min_keep:
            break
        y, x = y[keep], x[keep]
        if w is not None:
            w = w[keep]
        if y.max() - y.min() < 1.0:
            break
        coeffs = np.polyfit(y, x, 1, w=None if w is None else np.sqrt(w))

    return (float(coeffs[0]), float(coeffs[1]))


def _fit_curve_robust(y_vals, x_vals, curved=True, iters=3, mad_k=2.5,
                      min_keep=6, min_span=8.0, improve=0.85, w_range=18.0):
    """Фит x(y) прямой или параболой, с отбраковкой по MAD.

    Возвращает (slope, intercept) для прямой или (a2, a1, a0) для параболы —
    оба формата понимает rail_x().

    Квадратичный член берётся только если он реально улучшает остаток
    (на improve и больше) и если пары покрывают достаточный диапазон по Y.
    На прямом участке парабола по шумным парам изгибается на пустом месте,
    а лишняя степень свободы усиливает шум — это ровно то, из-за чего
    прежняя попытка «фитить параболой по парам» на дальности делала хуже.
    Здесь она применяется к ОСЕВОЙ линии (вдвое меньше шума, чем у нити)
    и только в пределах ближней зоны, где пары плотные.
    """
    y = np.asarray(y_vals, dtype=np.float64)
    x = np.asarray(x_vals, dtype=np.float64)

    # Вес падает с дальностью. Замер точности САМИХ пар по карте пути
    # (roundT_pressureGate_roundT): медиана отклонения растёт с 0.013 м на
    # 0-5 м до 0.047 м на 35-40 м — то есть вдвое-втрое, — но главное, что
    # взрывается доля грубых промахов: 0.0% пар с отклонением >0.2 м внутри
    # 20 м, 3.1% на 30-35 м и 7.5% на 35-40 м. Редкая, но грубо неверная
    # дальняя пара сидит на самом длинном плече и разворачивает фит сильнее
    # всех. Поэтому дальние пары участвуют, но с пониженным весом.
    w = 1.0 / (1.0 + (np.abs(y) / w_range) ** 2)

    lin = _fit_line_robust(y, x, w=w)
    if not curved or len(y) < min_keep or (y.max() - y.min()) < min_span:
        return lin

    # Робастный квадратичный фит на тех же данных
    yq, xq, wq = y, x, w
    coeffs = np.polyfit(yq, xq, 2, w=np.sqrt(wq))
    for _ in range(iters):
        resid = xq - np.polyval(coeffs, yq)
        mad = float(np.median(np.abs(resid - np.median(resid))))
        if mad <= 0:
            break
        keep = np.abs(resid - np.median(resid)) <= mad_k * 1.4826 * mad
        if keep.all() or keep.sum() < min_keep:
            break
        yq, xq, wq = yq[keep], xq[keep], wq[keep]
        if (yq.max() - yq.min()) < min_span:
            return lin
        coeffs = np.polyfit(yq, xq, 2, w=np.sqrt(wq))

    # Сравниваем на общем наборе точек, иначе выигрывает тот, кто больше
    # отбраковал. Остаток тоже взвешенный: иначе выбор модели определяют
    # дальние пары, которые как раз наименее надёжны.
    r_lin = _wmedian(np.abs(x - np.polyval(lin, y)), w)
    r_quad = _wmedian(np.abs(x - np.polyval(coeffs, y)), w)
    if r_quad < improve * r_lin:
        return (float(coeffs[0]), float(coeffs[1]), float(coeffs[2]))
    return lin


def _offset_fit(fit, dx):
    """Сдвигает фит по X на dx, сохраняя формат (прямая или парабола)."""
    if len(fit) == 3:
        return (fit[0], fit[1], fit[2] + dx)
    return (fit[0], fit[1] + dx)


def rail_x(line, y):
    """X рельса в точке Y. Понимает 2-tuple (прямая) и 3-tuple (парабола)."""
    if len(line) == 3:
        return line[0] * y * y + line[1] * y + line[2]
    return line[0] * y + line[1]


def _filter_pairs_consensus(pairs, tol=0.12, iters=5, min_keep=3,
                            seed_range=12.0, branch_tol=0.45, quad_span=8.0):
    """Отбрасывает пары, не согласованные с опорной линией midpoint кадра.

    Признаки одиночного пика (ширина, высота, intensity, число точек) ложные
    пары не разделяют — замер показал, что распределения перекрываются почти
    полностью. Разделяет пространственная согласованность: рельс непрерывен,
    поэтому midpoint колеи — гладкая функция дальности, а ложная пара
    (кабель, край платформы, чужой путь) от неё отскакивает.

    Опорная линия строится робастным фитом midpoint(y) по парам кадра и
    уточняется итеративно. Сравнение с общей линией, а не с долей ближайших
    соседей: на 25-30 м пар мало, у каждой 2-4 соседа, и подсчёт доли там
    выбрасывал верные пары — покрытие на 30 м падало со 100% до 74%.

    На стрелке midpoint раздваивается, и фит по всем парам ложится между
    ветками, поэтому линия сначала привязывается к ближней зоне (seed_range),
    где путь поезда однозначен, и ветка выбирается по близости к затравке
    (branch_tol). Габарит ведётся по своей ветке; чужая отбрасывается.
    """
    if len(pairs) < min_keep:
        return pairs

    y = pairs[:, 0]
    mid = (pairs[:, 1] + pairs[:, 2]) / 2

    # Затравка — ближняя зона: под лидаром путь однозначен даже на стрелке,
    # где дальше по ходу midpoint раздваивается. Без затравки робастный фит
    # на развилке ложится МЕЖДУ ветками и не совпадает ни с одной: на кадре
    # 255 roundT_pressureGate_roundT это давало остаток до 0.87 м при том,
    # что обе ветки сами по себе детектированы верно.
    near = np.abs(y) <= seed_range
    if near.sum() >= min_keep:
        seed = float(np.median(mid[near]))
    else:
        order = np.argsort(np.abs(y))[:max(min_keep, len(y) // 4)]
        seed = float(np.median(mid[order]))

    keep = np.abs(mid - seed) < branch_tol
    if keep.sum() < min_keep:
        keep = np.ones(len(pairs), dtype=bool)

    for _ in range(iters):
        yk, mk = y[keep], mid[keep]
        if keep.sum() < min_keep or yk.max() - yk.min() < 1.0:
            break
        # Опорная линия КРИВАЯ, когда пары покрывают достаточное плечо.
        # Прежде она была всегда прямой, и на повороте это выбрасывало
        # именно верные дальние пары: на кадре 231 roundT_pressureGate_roundT
        # путь уходит вбок с 0.04 м на 2 м до 1.89 м на 39.5 м, прямая такой
        # ход описать не может, и остаток верных пар рос с дальностью чисто
        # от неучтённой кривизны (0.14 м на 28 м, 0.48 м на 36.5 м) — все они
        # не проходили допуск 0.12 м. Оставались 20 пар из 45, причём среди
        # уцелевших дальних были пары СОСЕДНЕГО пути (вынос −1.3 м), которые
        # случайно легли близко к прямой. Они и тянули фит: ошибка осевой на
        # 30 м составляла 0.63 м.
        #
        # С квадратичной опорной проходит 40 пар из 45, и отвергнутые пять —
        # ровно те пять, что лежат в стороне от пути (14.0, 14.5, 33.5, 38.0,
        # 39.5 м). Степень выбирается по плечу: на коротком плече парабола
        # вырождается в шум, поэтому там остаётся прямая.
        deg = 2 if (yk.max() - yk.min()) >= quad_span else 1
        coeffs = np.polyfit(yk, mk, deg)
        resid = np.abs(mid - np.polyval(coeffs, y))
        new_keep = resid < tol
        if new_keep.sum() < min_keep or (new_keep == keep).all():
            break
        keep = new_keep

    return pairs[keep] if keep.sum() >= min_keep else pairs


def _filter_pairs_continuous(pairs, max_gap=10.0, min_keep=3):
    """Оставляет непрерывную по дальности цепочку пар, начиная от лидара.

    Рельс под поездом непрерывен: пары идут подряд с шагом среза (0.5 м).
    Замер разрывов ВНУТРИ настоящей цепочки на двух бэгах: медиана 0.50 м,
    p99 2.0-2.5 м, максимум 7.5 м. А на кадре 231
    roundT_pressureGate_roundT после ближних пар (до 15.5 м) шёл провал
    до 38 м, и там стояли две пары соседнего пути — с ПРАВИЛЬНОЙ колеёй
    1550 мм, поэтому по колее они неотличимы, но в стороне от пути на
    1.3-1.4 м. Будучи единственными дальними, они тянули параболу и давали
    ошибку 0.63 м на 30 м.

    Порог 10 м на порядок больше естественного разрыва и на порядок меньше
    наблюдавшегося провала, поэтому режет именно оторванные хвосты.
    Цепочка ведётся от ближней зоны: путь под поездом однозначен.
    """
    if len(pairs) < min_keep:
        return pairs

    order = np.argsort(np.abs(pairs[:, 0]))
    dist = np.abs(pairs[order, 0])
    gaps = np.diff(dist)
    cut = np.nonzero(gaps > max_gap)[0]
    if len(cut) == 0:
        return pairs

    keep_n = cut[0] + 1
    return pairs[order[:keep_n]] if keep_n >= min_keep else pairs


def _filter_pairs_self_gauge(pairs, near_range=20.0, tol=0.06, min_near=5,
                             min_keep=3):
    """Отбрасывает пары, колея которых не совпадает с колеёй ЭТОГО кадра.

    Номинальный допуск (±150 мм вокруг 1520) слишком широк, чтобы отсечь
    дальние ложные пары. Замер на roundT_pressureGate_roundT, кадры 237/240:
    настоящая колея в кадре 1573 ± 8 мм (по 39 кадрам: 1564…1592), а ложные
    пары на 39-42 м дают 1400-1450 мм — то есть внутрь ±150 мм они попадают
    и раньше проходили фильтр. При этом они лежали на 1.3-1.5 м в стороне от
    пути и, будучи единственными парами дальше 21 м, тянули фит на себя.

    Своя колея оценивается по ближней зоне, где детекция надёжна (медиана
    устойчива: разброс между кадрами 8 мм против 130-180 мм у ложных пар),
    и служит куда более острым признаком, чем номинал.
    """
    if len(pairs) < min_keep:
        return pairs

    g = pairs[:, 2] - pairs[:, 1]
    near = np.abs(pairs[:, 0]) <= near_range
    if near.sum() < min_near:
        return pairs

    own = float(np.median(g[near]))
    keep = np.abs(g - own) < tol
    return pairs[keep] if keep.sum() >= min_keep else pairs


def find_rail_lines(points, intensity, gauge=1.52, gauge_tol=0.15,
                     own_x_range=1.5, max_y_distance=40.0,
                     return_pairs=False, curved=True, fit_range=30.0):
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

    return_pairs — вернуть третьим элементом сырые пары пиков
        (N, 3) как (y, x_left, x_right) вместо только фита. Пары —
        исходные измерения положения колеи; для сшивки кадров в карту
        они ценнее линий, т.к. фит теряет информацию о каждом срезе.

    Возвращает:
        rail_lines: list of (slope, intercept) — линии x = slope*y + intercept
        ground_plane: (a, b, c) — коэффициенты плоскости z = a*x + b*y + c
        pairs: (N, 3) — только если return_pairs=True
    """
    empty_pairs = np.empty((0, 3))
    from sklearn.cluster import DBSCAN

    # Обрезаем по дальности (Y) — дальше max_y_distance рельсы уходят косо
    if max_y_distance is not None and max_y_distance > 0:
        y_mask = np.abs(points[:, 1]) <= max_y_distance
        points = points[y_mask]
        if intensity is not None:
            intensity = intensity[y_mask]

    if len(points) < 100:
        return ([], None, empty_pairs) if return_pairs else ([], None)

    # 1. Robust ground plane via grid percentiles
    ground_plane = estimate_ground_plane(points)
    if ground_plane is None:
        return ([], None, empty_pairs) if return_pairs else ([], None)

    a, b, c = ground_plane

    z_ground = a * points[:, 0] + b * points[:, 1] + c
    dz = points[:, 2] - z_ground

    # 2. Scan cross-sections, find height peaks, match gauge pairs
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    all_pairs = []  # (y, x_left, x_right)

    # Коридор поиска расширяется с дальностью, потому что путь уходит вбок.
    # Прежде предел был постоянным (own_x_range + полколеи), то есть прямым
    # коридором в системе лидара, тогда как путь изогнут: при радиусе метро
    # 300 м вынос составляет y²/2R, то есть 1.5 м уже на 30 м и 2.7 м на 40 м.
    # Замер по карте пути на roundT_pressureGate_roundT: центр колеи выходит
    # за own_x_range=1.5 м на 13% кадров на 30 м, на 28% на 40 м и на 35% на
    # 50 м (максимум 3.6 м). Настоящие рельсы оказывались ВНЕ окна поиска, и
    # внутри оставались только стены, кабели и соседний путь — отсюда и
    # «перескоки на стену» на поворотах.
    #
    # Радиус 250 м берётся с запасом к самой крутой кривой метро (300-600 м
    # по README), поэтому допуск гарантированно накрывает реальный вынос, но
    # растёт как y², а не линейно, и вблизи не размывается: на 10 м прибавка
    # всего 0.2 м, так что ложные пары у лидара отсекаются как раньше.
    CURVE_RADIUS_MIN = 250.0
    def _x_allow(y):
        return own_x_range + y * y / (2.0 * CURVE_RADIUS_MIN)

    x_search_limit = _x_allow(max(abs(y_min), abs(y_max))) + gauge / 2 + gauge_tol

    for y_center in np.arange(y_min + 0.5, y_max - 0.5, 0.5):
        band = (points[:, 1] > y_center - 0.25) & (points[:, 1] < y_center + 0.25)
        if band.sum() < 30:
            continue

        bp = points[band]
        bdz = dz[band]

        # Ограничиваем бины по X — не ищем пики далеко от ОЖИДАЕМОГО положения
        # пути на этой дальности (коридор расширяется по y²/2R, см. выше)
        slice_limit = _x_allow(abs(y_center)) + gauge / 2 + gauge_tol
        x_lo = max(bp[:, 0].min(), -slice_limit)
        x_hi = min(bp[:, 0].max(), slice_limit)
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

        # Позиция пика — центроид точек головки, а не X бина с максимумом.
        # Бин 50 мм иначе квантует результат: измеренная колея принимала
        # только значения 1400/1500/1550/1650 мм, и «разброс» в 77 мм был
        # почти целиком дискретизацией, а не реальным гулянием колеи.
        # Головка узкая (~70 мм), поэтому центроид точек в окне вокруг
        # вершины даёт положение с точностью много лучше ширины бина.
        peaks = []
        for g in groups:
            x_top = rx[g[np.argmax(rz[g])]]
            near_top = (np.abs(bp[:, 0] - x_top) < 0.06) & (bdz > 0.03) & (bdz < 0.40)
            if near_top.sum() >= 3:
                # Взвешиваем по высоте: точки вершины головки важнее склонов
                w = bdz[near_top] - 0.03
                x_sub = float(np.average(bp[near_top, 0], weights=w)) if w.sum() > 0 \
                    else float(bp[near_top, 0].mean())
            else:
                x_sub = float(x_top)
            peaks.append((x_sub, rz[g].max()))

        # Find pairs at gauge distance, only if midpoint near lidar.
        # На срез берётся ОДНА пара — с колеёй, ближайшей к номиналу.
        # Прежде в срез могли попасть две конкурирующие гипотезы (реальный
        # рельс и ложный пик от кабеля/шпалы), обе шли в фит, и ложная
        # тянула линию: на кадре 200 такие дубли давали остаток до 0.39 м.
        best = None
        for ai in range(len(peaks)):
            for bi in range(ai + 1, len(peaks)):
                dist = abs(peaks[bi][0] - peaks[ai][0])
                if abs(dist - gauge) < gauge_tol:
                    midpoint = (peaks[ai][0] + peaks[bi][0]) / 2
                    if abs(midpoint) <= _x_allow(abs(y_center)):
                        score = abs(dist - gauge)
                        if best is None or score < best[0]:
                            x_lo_p = min(peaks[ai][0], peaks[bi][0])
                            x_hi_p = max(peaks[ai][0], peaks[bi][0])
                            best = (score, x_lo_p, x_hi_p)
        if best is not None:
            all_pairs.append((y_center, best[1], best[2]))

    if len(all_pairs) < 3:
        return ([], ground_plane, empty_pairs) if return_pairs else ([], ground_plane)

    pairs = np.array(all_pairs)
    pairs = _filter_pairs_self_gauge(pairs)
    pairs = _filter_pairs_continuous(pairs)
    pairs = _filter_pairs_consensus(pairs)
    if len(pairs) < 3:
        return ([], ground_plane, empty_pairs) if return_pairs else ([], ground_plane)

    midpoints = (pairs[:, 1] + pairs[:, 2]) / 2

    # 3. Cluster pairs by track (midpoint X)
    db = DBSCAN(eps=0.5, min_samples=3).fit(midpoints.reshape(-1, 1))
    labels = db.labels_

    # 4. Fit rail lines — приоритет кластеру ближайшему к лидару (X=0)
    clusters = sorted(set(labels) - {-1})
    if not clusters:
        return ([], ground_plane, empty_pairs) if return_pairs else ([], ground_plane)

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

        if y_vals.max() - y_vals.min() < 1.0:
            for rail_idx in [1, 2]:
                rail_lines.append((0.0, float(np.median(p[:, rail_idx]))))
            continue

        # Фитим колею как целое, а не две нити независимо: рельсы жёстко
        # параллельны, поэтому осевая линия несёт всю форму пути, а нити —
        # это осевая ± полколеи. Независимый фит тратил бы две степени
        # свободы на то, что физически одна, и вдвое усиливал шум; кроме
        # того, нити могли разъехаться непараллельно, чего в пути не бывает.
        mid_vals = (p[:, 1] + p[:, 2]) / 2
        half = float(np.median(p[:, 2] - p[:, 1])) / 2

        # Фит строится по парам НЕ ДАЛЬШЕ fit_range, а за неё линия
        # экстраполируется. Дальние пары почти не несут сигнала, но сидят на
        # самом длинном плече: доля пар с отклонением >0.2 м равна 0.0% внутри
        # 20 м, 3.1% на 30-35 м и 7.5% на 35-40 м. Замер по карте пути,
        # ошибка осевой на 30 м (roundT_pressureGate_roundT):
        #
        #   пары до 40 м (было):  p95 0.461  max 1.007
        #   пары до 30 м (стало): p95 0.139  max 0.226
        #
        # То есть отбрасывание дальних пар не ухудшает, а радикально срезает
        # хвост: кривизна уверенно читается по ближней зоне, где пар много и
        # они точны, и экстраполируется дальше точнее, чем фитится по шуму.
        near_fit = np.abs(y_vals) <= fit_range
        if near_fit.sum() >= 6 and (y_vals[near_fit].max()
                                    - y_vals[near_fit].min()) >= 8.0:
            yf, mf = y_vals[near_fit], mid_vals[near_fit]
        else:
            yf, mf = y_vals, mid_vals

        center = _fit_curve_robust(yf, mf, curved=curved)
        rail_lines.append(_offset_fit(center, -half))
        rail_lines.append(_offset_fit(center, +half))

    if return_pairs:
        # Только свой путь: кластеры отсортированы по близости к лидару
        own = pairs[labels == clusters[0]] if clusters else empty_pairs
        return rail_lines, ground_plane, own
    return rail_lines, ground_plane


def _find_own_track(rail_lines):
    """Возвращает индекс пары (ti, ti+1) ближайшей к лидару (X=0)."""
    best_ti = None
    best_dist = float('inf')
    for ti in range(0, len(rail_lines), 2):
        if ti + 1 >= len(rail_lines):
            break
        # rail_x, а не line[1]: у параболы (a2, a1, a0) второй элемент —
        # наклон, а не X при нуле, и сравнение путей шло бы по чужой величине
        mid_x = abs((rail_x(rail_lines[ti], 0.0)
                     + rail_x(rail_lines[ti + 1], 0.0)) / 2)
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
                     refine_axis=True, refine_step=2.0, max_y_distance=40.0):
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
