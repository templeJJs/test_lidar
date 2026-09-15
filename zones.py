"""Разметка облака точек туннеля по зонам: rail / bed / ceiling / wall / middle.

Почему именно так:
- Пол в туннеле НАКЛОННЫЙ (уклон ~2.16%), поэтому все пороги задаются не
  абсолютным Z, а высотой НАД плоскостью пола z_floor(y) = floor_a + floor_b*y.
- Рельсы НЕ отделяются ни интенсивностью, ни геометрией точек (проверено на
  for_hackathon/doubleT_obstacle), поэтому это МОДЕЛЬ, привязанная к оси пути
  (axis_x) и ширине колеи (gauge).
- Приоритет меток: первое совпадение побеждает: rail -> bed -> ceiling -> wall
  -> middle. Реализовано перезаписью масок от младшего приоритета к старшему.

Модуль намеренно не зависит от GUI: на вход идут (N, 3) float32/float64 и
объект параметров, на выходе — метки/цвета/счётчики. Всё векторизовано
(никаких циклов по точкам) — кадр в 300k+ точек обрабатывается ~10 раз в сек.
"""
from dataclasses import dataclass

import numpy as np
import open3d as o3d

ZONE_NAMES = ('rail', 'bed', 'ceiling', 'wall', 'middle')  # label order; labels are indices into this

ZONE_COLORS = {
    'rail':    (1.00, 0.55, 0.00),   # bright orange
    'bed':     (0.35, 0.45, 0.60),   # grey-blue      ("низ")
    'ceiling': (0.85, 0.30, 0.90),   # magenta        ("потолок")
    'wall':    (0.00, 0.75, 0.70),   # turquoise
    'middle':  (0.25, 0.85, 0.35),   # green          ("средний уровень")
}

# Рельс рисовать сквозь вертикальное препятствие нельзя: линия утверждала бы
# "здесь можно ехать". Заблокированный участок красится этим цветом.
BLOCKED_COLOR = (1.00, 0.08, 0.08)

RAIL_HEIGHT = 0.16          # metres: rail head above the floor plane

# Габарит по высоте: всё выше ложа и ниже этой отметки внутри коридора считаем
# препятствием (поезд не может проехать сквозь него).
TRAIN_TOP = 3.7

# Препятствие должно ИМЕТЬ ВЫСОТУ. Счёт точек без этого условия красил красным
# кромку балласта, кабельный лоток и стенку рельса вдоль края коридора: замерено
# на doubleT_obstacle -- 13-27 м из 46 м, при высоте точек в этих бинах 0.0-0.19 м,
# тогда как у настоящего объекта на y = -38 м высота 1.0 м.
MIN_OBSTACLE_HEIGHT = 0.5    # metres, 5..95 перцентили высот внутри бина

_LABEL = {name: i for i, name in enumerate(ZONE_NAMES)}
_COLOR_TABLE = np.array([ZONE_COLORS[name] for name in ZONE_NAMES], dtype=np.float64)


@dataclass
class ZoneParams:
    floor_a: float = -1.886
    floor_b: float = 0.021585
    floor_band: float = 0.30        # metres above the floor plane -> 'bed'
    ceiling_z: float = 2.0          # absolute Z -> 'ceiling'
    axis_x: float = 1.60            # track axis lateral position
    gauge: float = 1.520            # metres
    rail_half_width: float = 0.12   # metres around each rail centre
    rail_low: float = -0.10         # metres relative to the floor plane
    rail_high: float = 0.35         # metres relative to the floor plane
    wall_x: tuple = ()              # detected wall X positions
    wall_band: float = 0.50         # metres
    detect_walls: bool = True
    contact_rail: bool = True       # рисовать/классифицировать контактный рельс
    contact_side: int = +1          # +1 -> со стороны +x от оси, -1 -> со стороны -x
    contact_offset: float = 1.46   # от ОСИ пути: gauge/2 + 0.70 (контактный снаружи ходового)    # метры от оси пути до центра контактного рельса
    contact_height: float = 0.20    # метры над плоскостью пола (головка рельса)
    contact_half_width: float = 0.10  # метры


def floor_profile(y, params):
    """Z плоскости пола: a + b*y. Пол наклонён, поэтому пороги задаются над ним.

    Работает и для скаляра, и для numpy-массива.
    """
    y_arr = np.asarray(y, dtype=np.float64)
    return params.floor_a + params.floor_b * y_arr


def fit_floor(xyz, y_min=-60.0, y_max=-2.0, bins=30, percentile=5.0, clip=2.0,
              x_center=None, x_half=None):
    """Робастная подгонка пола: (a, b, rms) для z_floor(y) = a + b*y.

    Пол -- нижняя огибающая поверхности, поэтому берём низкий перцентиль Z по
    Y-бинам, затем прямую МНК и ОТБРАКОВКУ выбросов по sigma: одиночные бины с
    конструкциями или редкими выбросами иначе перекашивают плоскость.

    Почему не мода плотного слоя: замерено, что доля моды в бине всего 1.2-3 %,
    то есть доминирующего слоя земли в этих данных нет (Z размазан: p10 -1.85,
    медиана -0.25). Поиск моды давал rms 0.68-1.6 м против 0.1 м у огибающей.

    Полосу `x_center +- x_half` задавать ОБЯЗАТЕЛЬНО: по всему кадру нижнюю
    огибающую тянут вниз основания стен и плоскость получается кривой
    (замерено: rms 0.18-0.20 по всему кадру против 0.042 в полосе пути x 2..5).
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.size == 0:
        return (float('nan'), 0.0, 0.0)

    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    sel = (y >= y_min) & (y < y_max)
    if x_center is not None and x_half is not None:
        sel &= np.abs(x - float(x_center)) <= float(x_half)
    y = y[sel]
    z = z[sel]
    if y.size == 0:
        return (float(np.median(xyz[:, 2])), 0.0, 0.0)

    edges = np.linspace(y_min, y_max, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    idx = np.clip(np.searchsorted(edges, y, side='right') - 1, 0, bins - 1)

    levels = np.full(bins, np.nan)
    for b in range(bins):
        m = idx == b
        if int(np.count_nonzero(m)) < 20:
            continue
        levels[b] = float(np.percentile(z[m], percentile))

    good = ~np.isnan(levels)
    if np.count_nonzero(good) < 2:
        return (float(np.median(xyz[:, 2])), 0.0, 0.0)

    slope, intercept = np.polyfit(centers[good], levels[good], 1)
    for _ in range(3):
        resid = levels[good] - (intercept + slope * centers[good])
        scale = float(np.median(np.abs(resid))) * 1.4826 or 1e-6
        keep = np.abs(resid) <= clip * scale
        if keep.all() or np.count_nonzero(keep) < 3:
            break
        good_idx = np.flatnonzero(good)[keep]
        good = np.zeros_like(good)
        good[good_idx] = True
        slope, intercept = np.polyfit(centers[good], levels[good], 1)

    resid = (intercept + slope * centers[good]) - levels[good]
    rms = float(np.sqrt(np.mean(resid ** 2)))
    return (float(intercept), float(slope), rms)


def detect_walls(xyz, params, min_separation=2.0, bin_width=0.10):
    """Ищем две доминирующие вертикальные стены по гистограмме X.

    Учитываем только точки ВЫШЕ пола (z - z_floor(y) >= floor_band): пол и рельсы
    дают сильные пики по X и мешали бы. Берём самый сильный бин, затем самый
    сильный бин минимум в min_separation метрах от него. Возвращаем центры
    по возрастанию; если второй не найден — один элемент.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.shape[0] == 0:
        return []

    x = xyz[:, 0]
    above = (xyz[:, 2] - floor_profile(xyz[:, 1], params)) >= params.floor_band
    xs = x[above]
    if xs.size == 0:
        return []

    x0 = float(np.floor(xs.min() / bin_width) * bin_width)
    x1 = float(np.ceil(xs.max() / bin_width) * bin_width)
    edges = np.arange(x0, x1 + bin_width, bin_width)
    counts, edges = np.histogram(xs, bins=edges)
    if counts.size == 0:
        return []

    centers = 0.5 * (edges[:-1] + edges[1:])
    first = int(np.argmax(counts))

    far = np.abs(centers - centers[first]) >= min_separation
    counts_far = np.where(far, counts, 0)
    second = int(np.argmax(counts_far))
    if counts_far[second] <= 0:
        return [float(centers[first])]

    return sorted([float(centers[first]), float(centers[second])])


def rail_x_positions(params):
    """[(x, высота_над_полом), ...] для КАЖДОГО рельса: ходовые + контактный.

    Почему рельсов три, а не два: у электрифицированного пути, кроме двух ходовых
    рельсов на `axis +- gauge/2`, есть контактный рельс. Он вынесен в сторону от
    оси на `contact_offset` и стоит на изоляторах, то есть на своей высоте
    `contact_height`, а не на `RAIL_HEIGHT`. Полуширина у него тоже своя
    (`contact_half_width`), иначе классификация "утащила" бы его в полосу
    ходового рельса. При `contact_rail=False` остаются ровно два ходовых -- это
    старый случай, и `make_rail_lines` для него работает как раньше.
    """
    half = params.gauge / 2.0
    rails = [(params.axis_x - half, RAIL_HEIGHT),
             (params.axis_x + half, RAIL_HEIGHT)]
    if params.contact_rail:
        rails.append((params.axis_x + params.contact_side * params.contact_offset,
                      params.contact_height))
    return rails


def classify(xyz, params):
    """Метки зон (uint8, индексы ZONE_NAMES). Первое совпадение побеждает.

    Порядок приоритета rail -> bed -> ceiling -> wall -> middle реализован
    наложением масок от младшего к старшему: старшие затирают младшие.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    n = xyz.shape[0]
    labels = np.full(n, _LABEL['middle'], dtype=np.uint8)
    if n == 0:
        return labels

    x = xyz[:, 0]
    zrel = xyz[:, 2] - floor_profile(xyz[:, 1], params)

    if len(params.wall_x):
        wall = np.zeros(n, dtype=bool)
        for wall_x in params.wall_x:
            wall |= np.abs(x - wall_x) < params.wall_band
        labels[wall] = _LABEL['wall']

    labels[xyz[:, 2] >= params.ceiling_z] = _LABEL['ceiling']
    labels[zrel < params.floor_band] = _LABEL['bed']

    if params.rail_half_width > 0 or params.contact_rail:
        rails = rail_x_positions(params)
        rail = np.zeros(n, dtype=bool)
        # Ходовые рельсы: прежняя полоса rail_low..rail_high ОТ плоскости пола.
        running_band = (zrel >= params.rail_low) & (zrel <= params.rail_high)
        for rail_x, _height in rails[:2]:
            rail |= (np.abs(x - rail_x) <= params.rail_half_width) & running_band
        # Контактный рельс: та же полуширина-логика, но свой коридор вокруг
        # contact_height; смещения rail_low/rail_high -- те же, что у ходового,
        # только отсчитываются от головки контактного рельса.
        if len(rails) > 2:
            contact_band = ((zrel >= params.contact_height + params.rail_low)
                            & (zrel <= params.contact_height + params.rail_high))
            for rail_x, _height in rails[2:]:
                rail |= (np.abs(x - rail_x) <= params.contact_half_width) & contact_band
        labels[rail] = _LABEL['rail']

    return labels


def zone_colors(xyz, params):
    """(N, 3) float64 цветов по метке точки; берём ровно из ZONE_COLORS."""
    labels = classify(xyz, params)
    return colors_from_labels(labels)


def colors_from_labels(labels):
    """(N, 3) float64 цветов по уже посчитанным меткам.

    Нужно, чтобы не классифицировать кадр дважды: инспектору нужны и цвета, и
    счётчики по зонам с одной и той же разметки.
    """
    labels = np.asarray(labels)
    return _COLOR_TABLE[labels.astype(np.intp, copy=False)]


def classify_counts(xyz, params):
    """{имя_зоны: количество} по всем пяти зонам (нули не опускаем)."""
    labels = classify(xyz, params)
    counts = np.bincount(labels, minlength=len(ZONE_NAMES))
    return {name: int(counts[i]) for i, name in enumerate(ZONE_NAMES)}


def _y_nodes(y_from, y_to, step):
    """Узлы вдоль пути. Общий для коридора и для линий рельсов, чтобы разметка
    засорённости и геометрия рельсов совпадали точка-в-точку."""
    ys = np.arange(y_from, y_to, step, dtype=np.float64)
    if ys.size == 0 or ys[-1] != y_to:
        ys = np.append(ys, float(y_to))
    return ys


def _bin_span(values, y, edges):
    """Высота точек в каждом Y-бине: размах перцентилей 5..95, 0 у пустых бинов.

    Нужна, чтобы отличать препятствие от тонкой поверхности: у столба, человека
    или упавшего предмета точки занимают по высоте десятки сантиметров, а у
    кромки балласта, лотка или стенки рельса -- сантиметры.
    """
    nb = edges.size - 1
    span = np.zeros(nb, dtype=np.float64)
    if values.size == 0:
        return span
    idx = np.clip(np.searchsorted(edges, y, side='right') - 1, 0, nb - 1)
    order = np.argsort(idx, kind='stable')
    idx_sorted = idx[order]
    val_sorted = values[order]
    # bounds[b] -- первый индекс бина b в отсортированном массиве, bounds[nb] -- конец
    bounds = np.searchsorted(idx_sorted, np.arange(nb + 1))
    for b in range(nb):
        lo, hi = int(bounds[b]), int(bounds[b + 1])
        if hi - lo < 2:
            continue
        seg = val_sorted[lo:hi]
        span[b] = float(np.percentile(seg, 95) - np.percentile(seg, 5))
    return span


def corridor_blocked(xyz, params, y_from, y_to, bin_width=1.0, clearance=0.30,
                     train_top=TRAIN_TOP, min_points=20,
                     min_height=MIN_OBSTACLE_HEIGHT):
    """Засорённость коридора пути вдоль Y: (узлы, маска "заблокировано").

    Поезд не может проехать сквозь вертикальное препятствие, поэтому рельс
    рисуется не сплошной линией, а разорванной там, где коридор занят. В коридор
    берём точки ВЫШЕ ложа (пол и балласт не мешают) и НИЖЕ габарита поезда
    (`train_top`) в полосе `|x - axis_x| < gauge/2 + clearance`.

    Одного счёта точек мало: бин набирает `min_points` и на тонкой поверхности,
    тянущейся вдоль края коридора (кромка балласта, лоток, стенка рельса), из-за
    чего красным становилась половина пути. Поэтому бин занят только если точки в
    нём занимают по высоте не меньше `min_height` (`_bin_span`). `min_height=0`
    возвращает прежнее поведение -- «занято по одному счёту точек».
    """
    ys = _y_nodes(y_from, y_to, bin_width)
    n = ys.size
    if n == 0:
        return ys, np.zeros(0, dtype=bool)
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.shape[0] == 0:
        return ys, np.zeros(n, dtype=bool)

    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    zrel = z - floor_profile(y, params)
    half = params.gauge / 2.0 + clearance
    inside = ((np.abs(x - params.axis_x) < half)
              & (zrel > params.floor_band)
              & (zrel < train_top))
    edges = np.append(ys - bin_width / 2.0, ys[-1] + bin_width / 2.0)
    counts, _ = np.histogram(y[inside], bins=edges)
    blocked = counts >= max(int(min_points), 1)
    if min_height > 0.0:
        blocked &= _bin_span(zrel[inside], y[inside], edges) >= float(min_height)
    return ys, blocked


def free_corridor_profile(xyz, params, y_from, y_to, bin_width=1.0, clearance=0.30,
                          train_top=TRAIN_TOP, min_points=20, axis_step=0.05,
                          bin_x=0.05):
    """Для каждого кандидата оси -- сколько Y-бинов свободно: (оси, свободно, всего).

    Поезд не может проехать сквозь вертикальное препятствие, поэтому ось пути
    ВЫБИРАЕТСЯ по максимуму свободной длины коридора. Середина между стенами --
    неверная эвристика: на `doubleT_obstacle` она ставит рельс прямо в столб.

    Двумерная гистограмма (Y, X) + кумулятивная сумма по X дают окно коридора
    за O(1) на кандидата, поэтому весь перебор -- векторный.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    ys = _y_nodes(y_from, y_to, bin_width)
    n_bins = ys.size
    if n_bins == 0:
        return np.zeros(0), np.zeros(0, dtype=int), 0

    half = params.gauge / 2.0 + clearance

    # Кандидаты -- только ВНУТРИ тоннеля. Вне стен "свободно" по определению, и
    # поиск максимума без этого ограничения уводит ось за стену (замерено: 8.75 м
    # при стенах -2.65 и +6.55). Ограничение берём по найденным стенам.
    if len(params.wall_x) >= 2:
        inner_lo = float(min(params.wall_x)) + half
        inner_hi = float(max(params.wall_x)) - half
    elif xyz.shape[0]:
        inner_lo = float(xyz[:, 0].min()) + half
        inner_hi = float(xyz[:, 0].max()) - half
    else:
        inner_lo, inner_hi = params.axis_x - 1.0, params.axis_x + 1.0
    if inner_hi <= inner_lo:
        mid = 0.5 * (inner_lo + inner_hi)
        inner_lo, inner_hi = mid - 0.5, mid + 0.5
    axes = np.arange(inner_lo, inner_hi + 1e-9, axis_step)

    if xyz.shape[0] == 0:
        return axes, np.full(axes.size, n_bins, dtype=int), n_bins

    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    zrel = z - floor_profile(y, params)
    inside = ((zrel > params.floor_band) & (zrel < train_top)
              & (y >= y_from) & (y < y_to))
    xo, yo = x[inside], y[inside]
    if xo.size == 0:
        return axes, np.full(axes.size, n_bins, dtype=int), n_bins

    yedges = np.append(ys - bin_width / 2.0, ys[-1] + bin_width / 2.0)
    xedges = np.arange(np.floor((inner_lo - half) / bin_x) * bin_x,
                       inner_hi + half + bin_x, bin_x)

    hist, _, _ = np.histogram2d(yo, xo, bins=[yedges, xedges])
    csum = np.concatenate([np.zeros((hist.shape[0], 1)), np.cumsum(hist, axis=1)], axis=1)

    lo = np.clip(np.searchsorted(xedges, axes - half, side='left'), 0, csum.shape[1] - 1)
    hi = np.clip(np.searchsorted(xedges, axes + half, side='right'), 0, csum.shape[1] - 1)
    window = csum[:, hi] - csum[:, lo]          # (n_y, n_axes)
    blocked = window >= max(int(min_points), 1)
    free = (~blocked).sum(axis=0)
    return axes, free, n_bins


def find_axis(xyz, params, y_from, y_to, **kwargs):
    """Ось пути = максимум свободной длины коридора.

    При равенстве берём ЦЕНТР самого длинного непрерывного плато максимума: край
    свободной зоны хуже, потому что одно небольшое препятствие сдвинет его в столб.
    """
    axes, free, n_bins = free_corridor_profile(xyz, params, y_from, y_to, **kwargs)
    if axes.size == 0 or n_bins == 0:
        return float(params.axis_x)
    # Нет ни одного препятствия -- ни одна ось не лучше другой, выбирать не из чего:
    # уважаем заданную (из --axis-x или дефолта), а не подставляем середину стен.
    if int(free.min()) == n_bins:
        return float(params.axis_x)
    best = int(free.max())
    if best == 0:
        return float(params.axis_x)

    ok = free == best
    runs = []
    start = None
    for i, flag in enumerate(ok):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(ok) - 1))
    a, b = max(runs, key=lambda r: r[1] - r[0])
    return float(axes[(a + b) // 2])


def blocked_spans(ys, blocked, bin_width=None):
    """Склеивает соседние заблокированные бины в диапазоны Y: [(y0, y1), ...]."""
    ys = np.asarray(ys, dtype=np.float64)
    blocked = np.asarray(blocked, dtype=bool)
    if ys.size == 0 or not bool(np.any(blocked)):
        return []
    if bin_width is None:
        bin_width = float(ys[1] - ys[0]) if ys.size > 1 else 1.0
    spans = []
    start = None
    for i, flag in enumerate(blocked):
        if flag and start is None:
            start = ys[i] - bin_width / 2.0
        elif not flag and start is not None:
            spans.append((float(start), float(ys[i - 1] + bin_width / 2.0)))
            start = None
    if start is not None:
        spans.append((float(start), float(ys[-1] + bin_width / 2.0)))
    return spans


def make_rail_lines(params, y_from=-80.0, y_to=0.0, step=1.0, blocked=None):
    """Полилинии ВСЕХ рельсов (ходовые + контактный) поверх наклонного пола.

    Число линий берётся из `rail_x_positions`, поэтому подпись совместима со
    старым случаем двух ходовых рельсов: при `contact_rail=False` получаются
    ровно две линии. Контактный рельс рисуется на `contact_height` -- он стоит
    на изоляторах и не совпадает по высоте с ходовыми, поэтому высота берётся
    для каждого рельса своя.

    Если `blocked` не задан -- длинные линии цветом рельса (профиль пола линеен,
    поэтому геометрия та же). Если задан (маска на узлах Y) -- линии рвутся по
    шагам, и шаг, у которого занят ЛЮБОЙ из концов, красится BLOCKED_COLOR:
    рельс визуально упирается в препятствие, а не проходит сквозь него.
    """
    ys = _y_nodes(y_from, y_to, step)
    n = ys.size
    rails = rail_x_positions(params)
    m = len(rails)
    z_floor = floor_profile(ys, params)
    points = np.vstack([
        np.column_stack([np.full(n, float(rail_x), dtype=np.float64), ys, z_floor + height])
        for rail_x, height in rails
    ]).astype(np.float64)

    if blocked is None:
        starts = np.arange(m, dtype=np.int32) * n
        lines = np.column_stack([starts, starts + (n - 1)]).astype(np.int32)
        colors = np.tile(np.array(ZONE_COLORS['rail'], dtype=np.float64), (m, 1))
    else:
        block = np.asarray(blocked, dtype=bool)
        if block.size != n:
            raise ValueError(f'blocked must have {n} entries, got {block.size}')
        rail_color = np.array(ZONE_COLORS['rail'], dtype=np.float64)
        stop_color = np.array(BLOCKED_COLOR, dtype=np.float64)
        per_rail = n - 1
        seg_stop = block[:-1] | block[1:]
        step_colors = np.where(seg_stop[:, None], stop_color[None, :], rail_color[None, :])
        lines = np.empty((m * per_rail, 2), dtype=np.int32)
        colors = np.empty((m * per_rail, 3), dtype=np.float64)
        seg_idx = np.arange(per_rail, dtype=np.int32)
        for r in range(m):
            s = slice(r * per_rail, (r + 1) * per_rail)
            start = r * n + seg_idx
            lines[s, 0] = start
            lines[s, 1] = start + 1
            colors[s] = step_colors

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set