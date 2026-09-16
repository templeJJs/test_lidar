"""Разметка облака точек туннеля по зонам: rail / bed / ceiling / wall / middle.

Сюда же добавлены две вещи «поверх» разметки (спека
docs/superpowers/specs/2026-09-15-path-and-safety-tunnel-design.md):

- `path_variants` -- три варианта ЛИНИИ ХОДА (коридор / рельсы / ось панели),
  по которой видно, куда ведёт путь;
- `tunnel_blocked` -- ТУННЕЛЬ БЕЗОПАСНОСТИ (габарит «М» вдоль линии хода) и
  занятость его бинов: всё, что попало внутрь, -- помеха.

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

RAIL_HEIGHT = 0.16          # metres: rail head above the floor plane == УГР (уровень головки рельса)

# Габарит по высоте: всё выше ложа и ниже этой отметки внутри коридора считаем
# препятствием (поезд не может проехать сквозь него).
TRAIN_TOP = 3.7

# Препятствие должно ИМЕТЬ ВЫСОТУ. Счёт точек без этого условия красил красным
# кромку балласта, кабельный лоток и стенку рельса вдоль края коридора: замерено
# на doubleT_obstacle -- 13-27 м из 46 м, при высоте точек в этих бинах 0.0-0.19 м,
# тогда как у настоящего объекта на y = -38 м высота 1.0 м.
MIN_OBSTACLE_HEIGHT = 0.5    # metres, 5..95 перцентили высот внутри бина

# ---------------------------------------------------------------------------
# Линия хода и туннель безопасности. Нормативы (ГОСТ 23961-80, ПТЭ метрополитенов
# п. 3.26) и числа -- из docs/superpowers/specs/2026-09-15-path-and-safety-tunnel-design.md:
#   * габарит «М» 2700 x 3750 мм, вагоны до 2712 x 3680 мм -> полуширина 1.36 м;
#   * ширина отсчитывается от ОСИ ПУТИ, высота -- от УГР (головки рельса, RAIL_HEIGHT);
#   * контактный рельс 1450 мм от оси, 160 мм над УГР, укладка слева по ходу;
#   * край платформы 1457 мм от оси, 1100 мм над УГР.
PATH_STEP = 1.0              # шаг узлов линии хода, м
PATH_BIN_X = 0.20            # ячейка поперёк пути для поиска свободного интервала, м
TUNNEL_HALF_WIDTH = 1.36     # полуширина габарита, м
TUNNEL_Z_LOW = 0.10          # низ туннеля, м НАД УГР
TUNNEL_Z_HIGH = 3.70         # верх туннеля, м НАД УГР
TUNNEL_CONTACT_HALF = 0.15   # полоса исключения контактного рельса, м от его оси
TUNNEL_CONTACT_BAND = 0.20   # +-м вокруг нормальной высоты контактного рельса
PLATFORM_FROM = 1.40         # край платформы: от..до м от оси пути
PLATFORM_TO = 1.55
PLATFORM_Z_LOW = 1.0         # высота края платформы, м над УГР
PLATFORM_Z_HIGH = 1.2

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
    contact_side: int = -1          # -1 -> слева по ходу (норма метрополитена), +1 -> справа
    contact_offset: float = 1.45    # метры от оси пути до центра контактного рельса (ПТЭ п. 3.26)
    # Высота головки контактного рельса над ПЛОСКОСТЬЮ ПОЛА: 160 мм над УГР + 0.16
    # самого УГР. Было 0.20 (= 40 мм над УГР) -- расхождение с ПТЭ, правка спеки.
    contact_height: float = RAIL_HEIGHT + 0.16
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


def _span_by_key(values, keys, n_groups):
    """Размах перцентилей 5..95 значений внутри каждой группы (0 у пустых групп).

    Общий примитив для `_bin_span` (группа -- бин вдоль пути) и для поиска
    свободного интервала поперёк пути (группа -- ячейка Y x X). Ключи -- целые
    0..n_groups-1, порядок произвольный.
    """
    span = np.zeros(int(n_groups), dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or n_groups <= 0:
        return span
    keys = np.asarray(keys, dtype=np.intp)
    order = np.argsort(keys, kind='stable')
    keys_sorted = keys[order]
    val_sorted = values[order]
    # bounds[b] -- первый индекс группы b в отсортированном массиве, bounds[n] -- конец
    bounds = np.searchsorted(keys_sorted, np.arange(int(n_groups) + 1))
    for b in range(int(n_groups)):
        lo, hi = int(bounds[b]), int(bounds[b + 1])
        if hi - lo < 2:
            continue
        seg = val_sorted[lo:hi]
        span[b] = float(np.percentile(seg, 95) - np.percentile(seg, 5))
    return span


def _bin_span(values, y, edges):
    """Высота точек в каждом Y-бине: размах перцентилей 5..95, 0 у пустых бинов.

    Нужна, чтобы отличать препятствие от тонкой поверхности: у столба, человека
    или упавшего предмета точки занимают по высоте десятки сантиметров, а у
    кромки балласта, лотка или стенки рельса -- сантиметры.
    """
    nb = edges.size - 1
    if nb <= 0:
        return np.zeros(0, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if values.size == 0:
        return np.zeros(nb, dtype=np.float64)
    # Точки вне бинов выбрасываем, а не сваливаем в крайние: облако уходит на
    # 200 м, а профиль пола линеен, и zrel таких точек раздувал размах крайнего
    # бина -- первый бин анализа (y = -40) ложно помечался занятым.
    keep = (y >= edges[0]) & (y < edges[-1])
    idx = np.clip(np.searchsorted(edges, y[keep], side='right') - 1, 0, nb - 1)
    return _span_by_key(values[keep], idx, nb)


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


# ------------------------------------------------------- линия хода вперёд


def _data_bins(xyz, ys, bin_width):
    """Маска бинов `ys`, в которых есть ХОТЬ ОДНА точка облака.

    «Узлы только там, где есть данные»: в пустом бине ни свободного интервала,
    ни оси рельсов нет -- рисовать там узел нечем.
    """
    ys = np.asarray(ys, dtype=np.float64)
    if ys.size == 0:
        return np.zeros(0, dtype=bool)
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.shape[0] == 0:
        return np.zeros(ys.size, dtype=bool)
    edges = np.append(ys - bin_width / 2.0, ys[-1] + bin_width / 2.0)
    y = xyz[:, 1]
    inside = (y >= edges[0]) & (y < edges[-1])
    if not bool(np.any(inside)):
        return np.zeros(ys.size, dtype=bool)
    idx = np.searchsorted(edges, y[inside], side='right') - 1
    return np.bincount(idx, minlength=ys.size)[:ys.size] > 0


def _fill_nan(values):
    """Пропуски -- предыдущим значением; ведущие -- первым известным.

    «Если бин занят -- берём предыдущее значение» (спека): так линия не прыгает
    на занятом бине, а продолжает последнее осмысленное направление.
    """
    src = np.asarray(values, dtype=np.float64)
    good = np.isfinite(src)
    if not bool(np.any(good)):
        return src.copy()
    idx = np.maximum.accumulate(np.where(good, np.arange(src.size), 0))
    out = src[idx]
    out[:int(np.argmax(good))] = src[int(np.argmax(good))]
    return out


def _median3(values):
    """Медиана по окну 3 бина (у краёв -- по двум доступным значениям)."""
    v = np.asarray(values, dtype=np.float64)
    n = v.size
    if n <= 2:
        return v.copy()
    return np.array([float(np.median(v[max(0, i - 1):min(n, i + 2)]))
                     for i in range(n)], dtype=np.float64)


def _longest_free_run(free):
    """Самый длинный непрерывный отрезок True -> (начало, конец) включительно."""
    best = None
    start = None
    free = np.asarray(free, dtype=bool)
    for i, flag in enumerate(free):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if best is None or (i - 1 - start) > (best[1] - best[0]):
                best = (start, i - 1)
            start = None
    if start is not None and (best is None
                              or (free.size - 1 - start) > (best[1] - best[0])):
        best = (start, free.size - 1)
    return best


def _path_band(params, xyz):
    """Полоса поперёк пути, в которой ищется свободный интервал: между стенами."""
    if len(params.wall_x) >= 2:
        return float(min(params.wall_x)), float(max(params.wall_x))
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.shape[0]:
        return float(xyz[:, 0].min()), float(xyz[:, 0].max())
    return float(params.axis_x) - 1.0, float(params.axis_x) + 1.0


def corridor_centres(xyz, params, ys, bin_width=1.0, bin_x=PATH_BIN_X,
                     min_points=20, min_height=MIN_OBSTACLE_HEIGHT,
                     train_top=TRAIN_TOP):
    """X центра самого широкого свободного интервала в каждом Y-бине (NaN -- нет).

    «Свободно» -- тот же критерий, что в `corridor_blocked`: в ячейке `bin_x`
    поперёк пути точек меньше `min_points` ИЛИ размах высот (5..95 перцентили)
    меньше `min_height`. Считаем именно ЯЧЕЙКИ, а не точки: стенка рельса или
    кабельный лоток шириной 10 см иначе разрезали бы свободную полосу.

    Полоса поиска -- между найденными стенами (`_path_band`); за стену линия не
    уходит, поэтому «внутри стен» выполняется по построению. Возвращает NaN для
    бина, где свободных ячеек нет вовсе, -- вызывающий решает, чем заполнить.
    """
    ys = np.asarray(ys, dtype=np.float64)
    n = ys.size
    out = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return out
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.shape[0] == 0:
        return out
    lo, hi = _path_band(params, xyz)
    if hi - lo < bin_x:
        return out

    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    zrel = z - floor_profile(y, params)
    yedges = np.append(ys - bin_width / 2.0, ys[-1] + bin_width / 2.0)
    xedges = np.arange(lo, hi + bin_x, bin_x, dtype=np.float64)
    nx = xedges.size - 1
    if nx <= 0:
        return out

    sel = ((y >= yedges[0]) & (y < yedges[-1])
           & (x >= xedges[0]) & (x < xedges[-1])
           & (zrel > params.floor_band) & (zrel < train_top))
    counts, _, _ = np.histogram2d(y[sel], x[sel], bins=[yedges, xedges])
    iy = np.clip(np.searchsorted(yedges, y[sel], side='right') - 1, 0, n - 1)
    ix = np.clip(np.searchsorted(xedges, x[sel], side='right') - 1, 0, nx - 1)
    spans = _span_by_key(zrel[sel], iy * nx + ix, n * nx).reshape(n, nx)
    occupied = (counts >= max(int(min_points), 1)) & (spans >= float(min_height))

    for i in range(n):
        run = _longest_free_run(~occupied[i])
        if run is not None:
            out[i] = 0.5 * (xedges[run[0]] + xedges[run[1] + 1])
    return out


def _path_variant(ys, x, z, source, error=None):
    """Один вариант линии хода в формате блока `path` из спеки.

    Значения -- numpy-массивы: JSON собирает вызывающий (web_viewer). `available`
    = False означает «рисовать нечего» (пустые y/x/z), `error` -- причину.
    """
    empty = np.zeros(0, dtype=np.float64)
    ok = error is None and np.asarray(x).size == np.asarray(ys).size and ys.size > 0
    out = {
        'y': np.asarray(ys, dtype=np.float64) if ok else empty,
        'x': np.asarray(x, dtype=np.float64) if ok else empty,
        'z': np.asarray(z, dtype=np.float64) if ok else empty,
        'source': source,
        'available': bool(ok),
    }
    if error is not None:
        out['error'] = error
    return out


def _default_rail_model(db_path=None, frame=0):
    """Ось пути из track_geometry: `x = s*y + i`.

    Импорт ленивый: модуль может быть недоступен или падать -- это обрабатывает
    `path_variants`, помечая вариант `rails` как unavailable, а не роняя вьюер.
    """
    from track_geometry import build_track
    if db_path is None:
        raise ValueError('не задан путь к bag')
    return build_track(str(db_path), int(frame))['axis']


def path_variants(xyz, params, y_from, y_to, step=PATH_STEP,
                  min_points=20, min_height=MIN_OBSTACLE_HEIGHT,
                  bin_x=PATH_BIN_X, train_top=TRAIN_TOP, rail_model=None,
                  db_path=None, frame=0):
    """Три варианта линии хода -- блок `path` из спеки (см. модульный docstring).

      * `corridor` -- центр свободного коридора по бинам `step` (сглаживание
        медианой по 3 бинам, на занятом бине -- предыдущее значение);
      * `rails` -- ось измеренных головок рельсов из `track_geometry.build_track`;
        модуль недоступен или упал -> вариант `unavailable`, вьюер не падает;
      * `axis` -- прямая по оси из панели (`params.axis_x`).

    Высота узлов -- `floor_profile(y) + RAIL_HEIGHT` (уровень головки рельса), а
    узлы только там, где в облаке есть точки. Все три варианта идут по ОДНИМ Y
    (иначе клиенту пришлось бы держать три разные шкалы вдоль пути).
    """
    ys_all = _y_nodes(y_from, y_to, step)
    has_data = _data_bins(xyz, ys_all, step)
    ys = ys_all[has_data]
    z = floor_profile(ys, params) + RAIL_HEIGHT
    no_data = None if ys.size else 'нет точек в диапазоне анализа'

    if no_data is not None:
        corridor = _path_variant(ys, ys, z, 'corridor-centre', no_data)
        rails = _path_variant(ys, ys, z, 'track_geometry', no_data)
        axis = _path_variant(ys, ys, z, 'panel-axis', no_data)
    else:
        # --- corridor
        centres = corridor_centres(xyz, params, ys_all, bin_width=step,
                                   bin_x=bin_x, min_points=min_points,
                                   min_height=min_height, train_top=train_top)[has_data]
        if bool(np.any(np.isfinite(centres))):
            corridor = _path_variant(ys, _median3(_fill_nan(centres)), z,
                                     'corridor-centre')
        else:
            corridor = _path_variant(ys, ys, z, 'corridor-centre',
                                     'нет свободного интервала ни в одном бине')

        # --- rails
        try:
            model = _default_rail_model if rail_model is None else rail_model
            data_axis = model(db_path, frame)
            s = float(data_axis['s'])
            i = float(data_axis['i'])
            x_rails = s * ys + i
            if not (np.isfinite(s) and np.isfinite(i)
                    and bool(np.all(np.isfinite(x_rails)))):
                raise ValueError('нечисловая модель оси')
            rails = _path_variant(ys, x_rails, z, 'track_geometry')
        except Exception as exc:          # noqa: BLE001 -- любая ошибка чужого модуля
            rails = _path_variant(ys, ys, z, 'track_geometry',
                                  f'{type(exc).__name__}: {exc}')

        # --- axis из панели
        axis = _path_variant(ys, np.full(ys.size, float(params.axis_x)), z,
                             'panel-axis')

    return {'step': float(step), 'z_offset': float(RAIL_HEIGHT),
            'variants': {'corridor': corridor, 'rails': rails, 'axis': axis}}


# ----------------------------------------------------- туннель безопасности


def tunnel_blocked(xyz, params, y_from, y_to, path_y=None, path_x=None,
                   half_width=TUNNEL_HALF_WIDTH, z_low=TUNNEL_Z_LOW,
                   z_high=TUNNEL_Z_HIGH, exclude_contact=True,
                   exclude_platform=True, platform_min_points=1,
                   platform_min_share=0.5,
                   bin_width=1.0, min_points=20,
                   min_height=MIN_OBSTACLE_HEIGHT):
    """Занятость туннеля безопасности по бинам вдоль линии хода.

    Сечение -- прямоугольник: полуширина `half_width` от линии, высота от `z_low`
    до `z_high` НАД УГР, то есть по zrel (над плоскостью пола) от
    `z_low + RAIL_HEIGHT` до `z_high + RAIL_HEIGHT`. Вдоль пути -- наклонное
    сечение (линия задаётся `path_x` на узлах `path_y`, интерполируется на бины);
    без линии -- прямая по оси `params.axis_x` (так туннель и ходит за ползунком
    «ось»: /set?axis= сдвигает объём и меняет занятость).

    Из объёма исключается ИЗВЕСТНОЕ оборудование, иначе оно само стало бы
    помехой: ходовые рельсы (полоса `rail_half_width` в диапазоне высот рельса),
    контактный рельс (полоса `TUNNEL_CONTACT_HALF` вокруг его нормальной высоты
    +-`TUNNEL_CONTACT_BAND`), край платформы (полоса `PLATFORM_FROM..PLATFORM_TO`
    от оси на высоте `PLATFORM_Z_LOW..PLATFORM_Z_HIGH` над УГР: бин помечается
    `platform` и помехой не считается, но только если в полосе не меньше
    `platform_min_points` точек И не меньше `platform_min_share` от точек бина --
    одной случайной точки для этого мало). Шпалы и балласт ниже низа туннеля
    отсекаются самим порогом.

    Занятость бина -- критерий `corridor_blocked`: точек >= `min_points` И размах
    высот (5..95 перцентили) >= `min_height`.

    Возвращает dict: y, x (линия по бинам), blocked, platform, counts, height_span,
    spans (склейка `blocked_spans`), bins_blocked, bins_total и параметры.
    """
    ys = _y_nodes(y_from, y_to, bin_width)
    n = ys.size
    z_lo_rel = float(z_low) + RAIL_HEIGHT
    z_hi_rel = float(z_high) + RAIL_HEIGHT

    line = np.full(n, float(params.axis_x), dtype=np.float64)
    if path_x is not None and n:
        py = np.asarray(ys if path_y is None else path_y, dtype=np.float64)
        px = np.asarray(path_x, dtype=np.float64)
        if py.size and px.size == py.size:
            order = np.argsort(py)
            line = np.interp(ys, py[order], px[order])

    result = {
        'y': ys,
        'x': line,
        'blocked': np.zeros(n, dtype=bool),
        'platform': np.zeros(n, dtype=bool),
        'counts': np.zeros(n, dtype=np.int64),
        'height_span': np.zeros(n, dtype=np.float64),
        'spans': [],
        'bins_blocked': 0,
        'bins_total': int(n),
        'half_width': float(half_width),
        'z_low': float(z_low),
        'z_high': float(z_high),
        'rail_height': float(RAIL_HEIGHT),
        'z_low_zrel': z_lo_rel,
        'z_high_zrel': z_hi_rel,
        'exclude_contact': bool(exclude_contact and params.contact_rail),
        'exclude_platform': bool(exclude_platform),
        'platform_min_points': int(platform_min_points),
        'min_points': int(min_points),
        'min_height': float(min_height),
    }
    xyz = np.asarray(xyz, dtype=np.float64)
    if n == 0 or half_width <= 0.0 or z_hi_rel <= z_lo_rel or xyz.shape[0] == 0:
        return result

    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    zrel = z - floor_profile(y, params)
    yedges = np.append(ys - bin_width / 2.0, ys[-1] + bin_width / 2.0)
    # Точки вне диапазона анализа не участвуют: облако уходит и на 200 м, а
    # профиль пола линеен и на таких y даёт бессмысленный zrel.
    in_range = (y >= yedges[0]) & (y < yedges[-1])
    xline = np.interp(y, ys, line)

    inside = ((np.abs(x - xline) <= float(half_width))
              & (zrel >= z_lo_rel) & (zrel <= z_hi_rel) & in_range)

    # --- исключения известного оборудования
    ignored = np.zeros(x.size, dtype=bool)
    rail_band = (zrel >= params.rail_low) & (zrel <= params.rail_high)
    rails = rail_x_positions(params)
    for rail_x, _height in rails[:2]:
        ignored |= (np.abs(x - rail_x) <= params.rail_half_width) & rail_band
    if exclude_contact and params.contact_rail:
        for rail_x, height in rails[2:]:
            ignored |= ((np.abs(x - rail_x) <= TUNNEL_CONTACT_HALF)
                        & (np.abs(zrel - height) <= TUNNEL_CONTACT_BAND))

    take = inside & ~ignored
    counts, _ = np.histogram(y[take], bins=yedges)

    platform = np.zeros(n, dtype=bool)
    if exclude_platform:
        plat_pts = (in_range
                    & (np.abs(x - xline) >= PLATFORM_FROM)
                    & (np.abs(x - xline) <= PLATFORM_TO)
                    & (zrel >= PLATFORM_Z_LOW + RAIL_HEIGHT)
                    & (zrel <= PLATFORM_Z_HIGH + RAIL_HEIGHT))
        if bool(np.any(plat_pts)):
            plat_idx = np.searchsorted(yedges, y[plat_pts], side='right') - 1
            plat_counts = np.bincount(plat_idx, minlength=n)[:n]
            # Кромка платформы -- сплошная конструкция: в её бине почти все точки
            # лежат в полосе платформы. Поэтому мало абсолютного минимума -- нужна
            # ещё и доля от точек бина, иначе одна случайная точка гасит настоящую
            # помеху (замерено на doubleT_obstacle: бины y=-15 с 177 точками и
            # y=-5 с 1143 гасли из-за 25 и 19 точек в полосе платформы).
            share = plat_counts / np.maximum(counts, 1)
            platform = ((plat_counts >= max(int(platform_min_points), 1))
                        & (share >= float(platform_min_share)))

    take_idx = np.searchsorted(yedges, y[take], side='right') - 1
    spans = _span_by_key(zrel[take], take_idx, n)
    blocked = ((counts >= max(int(min_points), 1))
               & (spans >= float(min_height)))
    if exclude_platform:
        blocked &= ~platform

    result['blocked'] = blocked
    result['platform'] = platform
    result['counts'] = counts.astype(np.int64)
    result['height_span'] = spans
    result['bins_blocked'] = int(np.count_nonzero(blocked))
    result['spans'] = blocked_spans(ys, blocked, bin_width)
    return result


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
    # open3d нужен ТОЛЬКО здесь (LineSet для фронта): модульный импорт тянул 447 МБ
    # зависимости на весь аналитический путь. Ленивый импорт: без open3d всё
    # остальное работает, а эта функция падает внятной ошибкой
    try:
        import open3d as o3d
    except ImportError as exc:            # pragma: no cover - зависит от окружения
        raise RuntimeError(
            'make_rail_lines: нужен open3d (только для линии рельсов во фронте) — '
            f'пакет не установлен: {exc}') from exc
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