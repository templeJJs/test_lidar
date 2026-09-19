"""Синтетическое внесение препятствия в реальный лидарный кадр -- без разметки.

Идея: не строим «своё» облако, а пользуемся УЖЕ существующими лучами кадра.
Объект (коробка) ставится в тоннель, для каждого реального луча считается
пересечение с ним (slab-метод), и точка попадает на объект ТОЛЬКО если объект
ближе уже измеренной поверхности на этом луче. Иначе объект закрыт тоннелем --
это и есть физически честная видимость. Точки фона на лучах, попавших в объект,
удаляются (тень/окклюзия): дальше по этому лучу сенсор ничего не видит.

Порядок (воспроизводит проверенный исследованием метод):
  1. сетка сенсора: медиана elevation по кольцу `eps_k`, шаг азимута `dphi_k` и
     фаза `a_k` -- по каждому кольцу из реального кадра (`ring_grid`);
  2. траектория: ось `x_c(y)` -- полилиния (`axis_nodes`, из `build_track` либо
     оценка по кадру), пол `z(y)` -- МНК по точкам пола (`fit_floor_ab`);
  3. AABB объекта: центр на оси в точке D со сносом `lateral_m`, полуразмеры
     `size_m/2` (по умолчанию объект 0.3 x 0.3 x 1 м);
  4. пересечение каждого РЕАЛЬНОГО луча с AABB, расширенным на `D*tan(theta/2)`,
     theta = 0.1° -- луч, задевший край, тоже даёт точку;
  5. `t_box < R_измеренный` -> луч видит объект; иначе объект закрыт;
  6. точки фона на этих лучах удаляются (тень);
  7. шум дальности N(0, 0.01 м), интенсивность -- бутстрап из реальных значений
     в бине D +- 10 %; опционально dropout части лучей объекта.

Единица «луч» -- ячейка сетки (кольцо, азимут 0.1°): идентификатор ячейки
`ring*3600 + floor((az+180)/0.1)`. Видимость решает БЛИЖАЙШИЙ возврат луча, но
удаляются ВСЕ точки этого луча (объект отрезает луч целиком) и вставляется
столько же точек на луч, сколько их в кадре. Измерено: в этих bag-ах на луч
почти ровно 2 возврата (346 808 точек = 2 x 173 408 - 8: у 8 лучей один возврат;
97.7 % пар с одинаковой дальностью), поэтому объект получает ту же плотность
~2 точки/луч, что и фон.

Поэтому `meta['rays_hit']` (лучей попало в объект), `meta['points_removed']`
(точек удалено тенью) и `meta['points_inserted']` (точек объекта) -- три разных
числа, и все три наружу. Без кратности (один возврат на луч) числа объекта были
бы ровно вдвое меньше -- на 25 м это 134 вместо 264 (см. selfcheck).

Поэтому `meta['rays_hit']` (лучей попало в объект), `meta['points_removed']`
(точек удалено тенью) и `meta['points_inserted']` (точек объекта) -- три разных
числа, и все три наружу. Без кратности (один возврат на луч) числа объекта были
бы ровно вдвое меньше -- на 25 м это 134 вместо 264 (см. selfcheck).

Модуль сам данных не читает: на вход идут готовые массивы. Для удобства рядом
лежат `load_frame` (rosbag2 -> массивы) и `axis_from_track` (ось из
`track_geometry`), оба с ЛЕНИВЫМ импортом: open3d не нужен, а если
`track_geometry` недоступен, ось оценивается по самому кадру. Всё векторизовано;
350 тыс. точек обрабатываются за ~25-40 мс (замер на живых кадрах).
"""

import math

import numpy as np

RING_COUNT = 128
AZ_BIN_DEG = 0.1                  # шаг ячейки азимута центральных колец Hesai 128
AZ_BINS = int(round(360.0 / AZ_BIN_DEG))
BEAM_DEG = 0.1                    # угловой шаг луча: расширение AABB = D*tan(theta/2)
RANGE_NOISE_M = 0.01              # sigma шума дальности (измерено: медиана 8 мм, p90 16-24 мм)
INTENSITY_BIN_FRAC = 0.10         # бутстрап интенсивности из бина D +- 10 %
OBJECT_SIZE_DEFAULT = (0.3, 0.3, 1.0)
# Полуширина колеи для поперечных положений постановки (лево/центр/право). Число
# ВЫВОДИТСЯ из колеи, а не назначается: `GAUGE_M_DEFAULT` -- измеренная колея по
# снимку `detector/track_models.json` (по записям 1.5885..1.5980, ось головки),
# то есть +-0.794..0.799 м. Раньше здесь стояло 0.7175 м -- ниоткуда не
# выводимое число; оно попадало ВНУТРЬ полосы оборудования детектора
# (|u -+ gauge/2| <= 0.10 м, `core.equipment_mask`), и высота силуэта на
# положениях «лево/право» срезалась маской, а не физикой. Рабочее значение --
# половина колеи ЭТОЙ записи (`gauge_half_m(model)`); константа остаётся
# запасным путём, когда модели нет (например, синтетический стенд тестов).
GAUGE_M_DEFAULT = 1.593
GAUGE_HALF_M = GAUGE_M_DEFAULT / 2.0


# ---------------------------------------------------------------------------
# 0. Базы протокола: колея постановки и независимость испытаний
# ---------------------------------------------------------------------------

def gauge_half_m(model=None):
    """Полуширина колеи для положений «лево/центр/право» по модели записи.

    Колея берётся из модели пути (`TrackModel.gauge_m`, поле снимка
    `detector/track_models.json`: 1.5885..1.5980 по шести записям, измерение по
    оси головки рельса). Без модели -- `GAUGE_HALF_M` (та же колея, среднее
    значение). См. вывод числа в комментарии к `GAUGE_HALF_M`.
    """
    gauge = getattr(model, 'gauge_m', None)
    if gauge:
        return float(gauge) / 2.0
    return GAUGE_HALF_M


def lateral_positions(model=None):
    """Поперечные положения постановки: (-полуширина, 0.0, +полуширина) колеи."""
    half = gauge_half_m(model)
    return (-half, 0.0, half)


def t_ppf_975(df):
    """97.5 %-я точка распределения Стьюдента с `df` степенями свободы.

    Приближение Хилла (Hill, 1970) -- точность лучше 1e-4 при df >= 3, scipy в
    валидацию не тянем.
    """
    z = 1.959964
    if df <= 0:
        return z
    return (z + (z ** 3 + z) / (4.0 * df)
            + (5 * z ** 5 + 16 * z ** 3 + 3 * z) / (96.0 * df ** 2)
            + (3 * z ** 7 + 19 * z ** 5 + 17 * z ** 3 - 15 * z) / (384.0 * df ** 3)
            + (79 * z ** 9 + 776 * z ** 7 + 1482 * z ** 5 - 1920 * z ** 3 - 945 * z)
            / (92160.0 * df ** 4))


def scene_rates(rows, scene_ids, hit_key):
    """Доля срабатываний ВНУТРИ каждой сцены: {сцена: доля}.

    Сцена -- один кадр записи (кластер испытаний): поперечные положения и
    посадки внутри сцены независимыми не являются, и шум дальности у них общий.
    """
    groups = {}
    for row, scene in zip(rows, scene_ids):
        groups.setdefault(scene, []).append(1.0 if row[hit_key] else 0.0)
    return {key: float(np.mean(values)) for key, values in groups.items()}


def scene_ci(rates, alpha=0.05):
    """95 %-й ДИ средней доли срабатываний, посчитанный по НЕЗАВИСИМЫМ СЦЕНАМ.

    `rates` -- доли срабатываний по сценам (`scene_rates`). Среднее по сценам
    равно среднему по испытаниям, когда сцены одной длины, поэтому интервал
    честно показывает цену переиспользования одного кадра: шум дальности и
    геометрия внутри сцены общие, и «180 независимых испытаний» -- это на самом
    деле 60 сцен. Интервал -- t-интервал по разбросу МЕЖДУ сценами; при 60
    сценах он в 1.7 раза шире, чем наивный по 180 испытаниям.
    """
    values = np.asarray(list(rates), dtype=np.float64)
    n = values.size
    if rates is None or n < 3:
        # Двух сцен мало даже для t-интервала: разброс между ними -- это разброс
        # двух чисел, и интервал выходит шире физического диапазона [0, 1].
        return (float('nan'), float('nan'))
    mean = float(values.mean())
    half = float(t_ppf_975(n - 1) * values.std(ddof=1) / np.sqrt(n))
    return (mean - half, mean + half)


# ---------------------------------------------------------------------------
# 1. Сетка сенсора
# ---------------------------------------------------------------------------

def az_deg(points):
    """Азимут, °: 0 -- прямо по ходу (-Y), положительно вправо (+X)."""
    points = np.asarray(points, dtype=np.float64)
    return np.degrees(np.arctan2(points[:, 0], -points[:, 1]))


def el_deg(points):
    """Elevation, °: положительно вверх."""
    points = np.asarray(points, dtype=np.float64)
    return np.degrees(np.arctan2(points[:, 2], np.hypot(points[:, 0], points[:, 1])))


def az_bins(points):
    """Индекс ячейки азимута 0.1° (0..3599)."""
    return np.clip(np.floor((az_deg(points) + 180.0) / AZ_BIN_DEG).astype(np.int64),
                   0, AZ_BINS - 1)


def ray_keys(points, ring=None):
    """Идентификатор луча: ring*AZ_BINS + ячейка азимута. Без ring -- только азимут."""
    bins = az_bins(points)
    if ring is None:
        return bins
    return np.asarray(ring, dtype=np.int64) * AZ_BINS + bins


def range_grid(points, ring, ring_count=RING_COUNT):
    """Ячейки (кольцо, азимут) -> БЛИЖАЙШАЯ дальность в ячейке, (ring_count, AZ_BINS).

    Кольцо с фазой азимута, своей у каждого кольца (измерено std ~0.3 шага),
    поэтому единая общая сетка не годится -- ячейки берём в системе координат
    луча (кольцо, азимут), а не как общий для кадра угловой растр.
    """
    points = np.asarray(points, dtype=np.float64)
    r = np.linalg.norm(points, axis=1)
    key = ray_keys(points, ring)
    order = np.lexsort((r, key))
    ks = key[order]
    rs = r[order]
    if ks.size == 0:
        return np.full((ring_count, AZ_BINS), np.nan, dtype=np.float32)
    first = np.empty(ks.size, dtype=bool)
    first[0] = True
    first[1:] = ks[1:] != ks[:-1]
    grid = np.full(ring_count * AZ_BINS, np.nan, dtype=np.float32)
    grid[ks[first]] = rs[first].astype(np.float32)
    return grid.reshape(ring_count, AZ_BINS)


def ring_grid(points, ring, ring_count=RING_COUNT):
    """Параметры колец по реальному кадру: eps_k, шаг и фаза азимута.

    Возвращает dict:
      'elev_deg'     (ring_count,) медиана elevation кольца (NaN, если кольцо пусто);
      'az_step_deg'  (ring_count,) медианный шаг азимута внутри кольца (0.1°/0.2°);
      'az_phase'     (ring_count,) фаза азимута в долях шага (0..1) -- фаз у колец
                     нет общей, поэтому каждая со своей.
    """
    points = np.asarray(points, dtype=np.float64)
    ring = np.asarray(ring, dtype=np.int64)
    elev = el_deg(points)
    azi = az_deg(points)

    step = np.full(ring_count, np.nan)
    phase = np.full(ring_count, np.nan)
    elev_k = np.full(ring_count, np.nan)

    order = np.argsort(ring, kind='stable')
    rings_sorted = ring[order]
    bounds = np.searchsorted(rings_sorted, np.arange(ring_count + 1))
    for k in range(ring_count):
        lo, hi = int(bounds[k]), int(bounds[k + 1])
        if hi - lo < 10:
            continue
        sel = order[lo:hi]
        elev_k[k] = float(np.median(elev[sel]))
        a = np.sort(azi[sel])
        d = np.diff(a)
        d = d[d > 1e-4]
        if d.size == 0:
            continue
        st = float(np.median(d))
        step[k] = st
        if 0.0 < st < 90.0:
            phase[k] = float(np.median(np.mod(a, st)) / st)
    return {'elev_deg': elev_k, 'az_step_deg': step, 'az_phase': phase}


# ---------------------------------------------------------------------------
# 2. Траектория: пол и ось
# ---------------------------------------------------------------------------

def fit_floor_ab(points, y_lo=-60.0, y_hi=-2.0, bins=30, percentile=5.0, clip=2.0):
    """МНК по точкам пола: (a, b, rms) для z_floor(y) = a + b*y.

    Пол -- нижняя огибающая (низкий перцентиль Z по Y-бинам) плюс прямая МНК с
    отбраковкой выбросов по MAD. Копия `zones.fit_floor` (та живёт в модуле,
    который тянет open3d, а валидация обязана работать без него).
    """
    points = np.asarray(points, dtype=np.float64)
    if points.size == 0:
        return (float('nan'), 0.0, 0.0)
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    sel = (y >= y_lo) & (y < y_hi)
    y, z = y[sel], z[sel]
    if y.size == 0:
        return (float(np.median(points[:, 2])), 0.0, 0.0)

    edges = np.linspace(y_lo, y_hi, bins + 1)
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
        return (float(np.median(points[:, 2])), 0.0, 0.0)
    slope, intercept = np.polyfit(centers[good], levels[good], 1)
    for _ in range(3):
        resid = levels[good] - (intercept + slope * centers[good])
        scale = float(np.median(np.abs(resid))) * 1.4826 or 1e-6
        keep = np.abs(resid) <= clip * scale
        if keep.all() or np.count_nonzero(keep) < 3:
            break
        idx_good = np.flatnonzero(good)[keep]
        good = np.zeros_like(good)
        good[idx_good] = True
        slope, intercept = np.polyfit(centers[good], levels[good], 1)

    resid = (intercept + slope * centers[good]) - levels[good]
    rms = float(np.sqrt(np.mean(resid ** 2)))
    return (float(intercept), float(slope), rms)


def estimate_axis_nodes(points, floor_ab=None, y_lo=-160.0, y_hi=-3.0, bin_m=2.0,
                        min_points=200, min_span_m=1.0, max_span_m=7.0):
    """Ось пути по кадру: полилиния x(y) из точек среднего уровня.

    В бине Y берём точки на высоте 0.8..2.6 м над полом и берём середину
    1..99 перцентилей X. Бин принимается, если размах по X лежит в
    (min_span_m, max_span_m): ближе по ходу это рельсы/лоток, дальше -- ДВЕ СТЕНЫ,
    и их середина и есть ось тоннеля (замерено: на 100-160 м в полосе 0.8..2.6 м
    над полом точек, кроме стен, нет). Бин с размахом больше max_span_m
    выбрасывается: там видна не ось, а развилка или примыкание.

    Это запасной путь, когда `track_geometry.build_track` недоступен, и
    продолжение там, где кончаются его меши: осевые бины доходят до y ~ -160 м,
    а меши нитей у разных записей -- только до -49..-139 м.
    """
    points = np.asarray(points, dtype=np.float64)
    if floor_ab is None:
        floor_ab = fit_floor_ab(points)[:2]
    a, b = float(floor_ab[0]), float(floor_ab[1])
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    zr = z - (a + b * y)

    edges = np.arange(y_lo, y_hi + bin_m, bin_m)
    nodes = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        m = (y >= lo) & (y < hi) & (zr > 0.8) & (zr < 2.6)
        if int(np.count_nonzero(m)) < min_points:
            continue
        xlo, xhi = np.percentile(x[m], [1.0, 99.0])
        if not (min_span_m < (xhi - xlo) < max_span_m):
            continue
        nodes.append((0.5 * (xlo + xhi), 0.5 * (lo + hi)))
    if not nodes:
        return np.zeros((0, 2))
    nodes = np.array(nodes, dtype=np.float64)
    return nodes[np.argsort(nodes[:, 1])]


def axis_from_track(db_path, frame_index, axis_polyline=True, x_limit_m=12.0):
    """Ось из `track_geometry.build_track`: середина полилиний двух нитей.

    `rails[side].nodes` у track_geometry идут как [y, x, z_top]; наружу отсюда
    ось отдаётся в соглашении модуля -- (M, 2) [x, y] по возрастанию y.

    ВАЖНО: за измеренным участком track_geometry продолжает полилинию по
    касательной, и на дальнем хвосте узлы могут уезжать (в одном из прогонов
    build_track на doubleT_obstacle видели y = -146.75 при x = 89.45 м -- в
    тоннеле шириной 16 м такого x быть не может). Поэтому узлы с
    |x| > x_limit_m отбрасываются, а из остатка берётся самый длинный
    непрерывный отрезок по y. Сколько отброшено и где кончается ось -- в мете
    ('axis_nodes_dropped', 'axis_nodes_y_range_m').

    Импорт track_geometry ленивый: он тянет visualize_bag, а с ним и open3d.
    """
    import track_geometry as tg  # noqa: PLC0415  (ленивый импорт -- см. докстроку)

    track = tg.build_track(db_path, frame_index, axis_polyline=axis_polyline)
    mesh = track.get('meta', {}).get('rail_mesh', {})
    rails = mesh.get('rails', {})
    dropped = 0
    by_side = {}
    for side in ('left', 'right'):
        side_nodes = rails.get(side, {}).get('nodes')
        if side_nodes is None or len(side_nodes) == 0:
            continue
        arr = np.asarray(side_nodes, dtype=np.float64)
        y = arr[:, 0]                      # [y, x, z_top]
        x = arr[:, 1]
        keep = (np.isfinite(y) & np.isfinite(x)
                & (np.abs(x) <= float(x_limit_m)) & (np.abs(y) <= 260.0))
        dropped += int(arr.shape[0] - np.count_nonzero(keep))
        by_side[side] = (y[keep], x[keep])

    if len(by_side) == 2:
        yl, xl = by_side['left']
        yr, xr = by_side['right']
        y_lo = max(yl.min(), yr.min())
        y_hi = min(yl.max(), yr.max())
        if y_hi > y_lo:
            y = np.linspace(y_lo, y_hi, max(2, int((y_hi - y_lo) / 0.5) + 1))
            x = 0.5 * (np.interp(y, yl, xl) + np.interp(y, yr, xr))
            source = 'track_geometry: середина полилиний нитей (rail_mesh.rails.*.nodes)'
        else:
            y, x = yl, xl
            source = 'track_geometry: нити не перекрываются по y -- взята левая'
    elif len(by_side) == 1:
        y, x = next(iter(by_side.values()))
        source = 'track_geometry: одна нить (вторая не измерена)'
    else:
        axis = track['axis']
        y = np.array([-150.0, 0.0], dtype=np.float64)
        x = axis['s'] * y + axis['i']
        source = 'track_geometry: прямая ось (нити не измерены)'

    nodes = np.column_stack([x, y])
    nodes = nodes[np.argsort(nodes[:, 1])]
    nodes = nodes[_longest_run(nodes[:, 1], max_gap_m=2.0)]
    meta = {'axis_source': source, 'axis_model': mesh.get('axis_model'),
            'frame_index': int(frame_index), 'axis_nodes_dropped': int(dropped),
            'axis_nodes_y_range_m': ([float(nodes[0, 1]), float(nodes[-1, 1])]
                                     if nodes.size else None)}
    return nodes, meta


def _longest_run(y_sorted, max_gap_m=2.0):
    """Маска самого длинного непрерывного отрезка по y (разрыв > max_gap_m его рвёт)."""
    y_sorted = np.asarray(y_sorted, dtype=np.float64)
    if y_sorted.size == 0:
        return np.zeros(0, dtype=bool)
    if y_sorted.size == 1:
        return np.ones(1, dtype=bool)
    gap = np.diff(y_sorted) > max_gap_m
    bounds = np.flatnonzero(gap)
    starts = np.concatenate([[0], bounds + 1])
    ends = np.concatenate([bounds, [y_sorted.size - 1]])
    lengths = ends - starts + 1
    best = int(np.argmax(lengths))
    mask = np.zeros(y_sorted.size, dtype=bool)
    mask[starts[best]:ends[best] + 1] = True
    return mask


def load_frame(db_path, frame_index):
    """Кадр rosbag2 -> (points (N,3) float64, ring (N,), intensity (N,)).

    Единственное место, где валидация касается данных; всё остальное -- массивы.
    """
    import bag_reader  # noqa: PLC0415

    frames = bag_reader.BagFrames(db_path, cache_size=2, prefetch_ahead=0)
    try:
        frame = frames[frame_index]
    finally:
        frames.close()
    points = np.asarray(frame.xyz, dtype=np.float64)
    ring = None if frame.ring is None else np.asarray(frame.ring, dtype=np.int64)
    intensity = None if frame.intensity is None else np.asarray(frame.intensity, dtype=np.float64)
    return points, ring, intensity


def build_rings_meta(points, ring=None, intensity=None, axis_nodes=None, floor_ab=None,
                     db_path=None, frame_index=0):
    """Метаданные кадра для `insert_object`: сетка колец, траектория, массивы кадра.

    Приоритет оси: явные `axis_nodes` -> `track_geometry.build_track` (если задан
    db_path) -> оценка по кадру. Пол: явный `floor_ab` -> МНК по кадру.

    В мете лежат и per-point массивы кадра (`ring`, `intensity`), потому что
    сигнатура `insert_object` тянет всё кадровое через `rings_meta`: они
    используются, только если длина совпадает с длиной `points`.
    """
    points = np.asarray(points, dtype=np.float64)
    ring = None if ring is None else np.asarray(ring, dtype=np.int64)
    intensity = None if intensity is None else np.asarray(intensity, dtype=np.float64)

    meta = {'ring_count': RING_COUNT, 'az_bin_deg': AZ_BIN_DEG, 'az_bins': AZ_BINS}
    if ring is not None:
        meta.update(ring_grid(points, ring))
    meta['ring'] = ring
    meta['intensity'] = intensity

    r = np.linalg.norm(points, axis=1)
    meta['range_m'] = r
    meta['range_stats_m'] = {
        'min': float(r.min()) if r.size else float('nan'),
        'max': float(r.max()) if r.size else float('nan'),
        'median': float(np.median(r)) if r.size else float('nan'),
    }

    if floor_ab is None:
        a, b, rms = fit_floor_ab(points)
        meta['floor_source'] = 'МНК по точкам пола (низкий перцентиль по Y-бинам)'
        meta['floor_rms_m'] = rms
    else:
        a, b = float(floor_ab[0]), float(floor_ab[1])
        meta['floor_source'] = 'пол задан вызывающим'
        meta['floor_rms_m'] = float('nan')
    meta['floor_ab'] = (a, b)

    if axis_nodes is not None:
        nodes = np.asarray(axis_nodes, dtype=np.float64)[:, :2]
        meta['axis_source'] = 'ось задана вызывающим'
    elif db_path is not None:
        try:
            nodes, extra = axis_from_track(db_path, frame_index)
            meta.update(extra)
            # Меши нитей кончаются раньше тоннеля (у разных записей y = -49..-139 м),
            # а дальше ось продолжается глобальным наклоном МНК по узлам (см.
            # axis_point). Продолжать её осевыми бинами кадра НЕЛЬЗЯ как опорой:
            # на несимметричных станциях середина по X -- это середина зала
            # (у doubleT_obstacle есть платформа до x = +6.8 м), и постановка
            # уезжает из колеи. Наклон осевых бинов только сохраняется в мете --
            # для справки, сколько их и куда шло продолжение.
            estimate = estimate_axis_nodes(points, floor_ab=(a, b))
            meta['axis_frame_nodes'] = int(estimate.shape[0])
            if estimate.shape[0] >= 2:
                meta['axis_frame_line'] = _axis_line(estimate)
        except Exception as exc:  # noqa: BLE001  -- нет open3d/данных: работаем без мешей
            nodes = estimate_axis_nodes(points, floor_ab=(a, b))
            meta['axis_source'] = f'оценка по кадру (build_track недоступен: {type(exc).__name__})'
    else:
        nodes = estimate_axis_nodes(points, floor_ab=(a, b))
        meta['axis_source'] = 'оценка по кадру (бины 2 м, 0.8..2.6 м над полом)'
    if nodes.shape[0] < 2:
        # Более широкая полоса захвата (до 10 м): в несимметричных залах середина
        # по X -- это середина зала, а не ось пути, но это всё же измеренное место
        # внутри тоннеля, а не выдуманный x = 0.
        wide = estimate_axis_nodes(points, floor_ab=(a, b), max_span_m=10.0)
        if wide.shape[0] >= 2:
            nodes = wide
            meta['axis_source'] = ('оценка по кадру: середина зала (полоса шире, '
                                   'узкая полоса не дала ни одного бина)')
        else:
            nodes = np.array([[0.0, -300.0], [0.0, 0.0]], dtype=np.float64)
            meta['axis_source'] = 'ось не измерена: прямая x=0'
    nodes = nodes[np.argsort(nodes[:, 1])]
    meta['axis_nodes'] = nodes
    meta['axis_nodes_y_range_m'] = [float(nodes[0, 1]), float(nodes[-1, 1])]
    meta['axis_line'] = _axis_line(nodes)
    return meta


# ---------------------------------------------------------------------------
# 3. AABB-объект на оси
# ---------------------------------------------------------------------------

def axis_point(meta, D, lateral_m=0.0):
    """Точка на оси в `D` метрах по ходу + снос `lateral_m` по локальной нормали.

    Ось параметризована координатой y: x = x_c(y). Внутри измеренного участка
    касательная и нормаль берутся из сегмента полилинии, содержащего y = -D.

    Вне участка полилиния НЕ продолжается по касательной последнего сегмента:
    локальный наклон на хвосте уводит ось из тоннеля (замерено на
    doubleT_obstacle: продолжение по касательной последнего сегмента до y = -100
    даёт x = +5.6 м, то есть объект ставится в стену, и на 100 м «ничего не
    видно» вообще ни на одном кадре). Вместо этого ось продолжается ГЛОБАЛЬНЫМ
    наклоном МНК по всем узлам (`meta['axis_line']`), привязанным к крайнему
    узлу: так продолжение непрерывно и остаётся внутри тоннеля.
    """
    nodes = np.asarray(meta['axis_nodes'], dtype=np.float64)
    ys, xs = nodes[:, 1], nodes[:, 0]
    y0 = -float(D)
    extrapolated = bool(y0 < ys[0] or y0 > ys[-1])
    if extrapolated:
        line = meta.get('axis_line') or _axis_line(nodes)
        s_global = float(line[0])
        edge = -1 if y0 > ys[-1] else 0
        y_edge, x_edge = float(ys[edge]), float(xs[edge])
        x_axis = x_edge + s_global * (y0 - y_edge)
        slope = s_global
    else:
        i1 = int(np.searchsorted(ys, y0))
        i0 = max(0, i1 - 1)
        dy = ys[i1] - ys[i0]
        slope = (xs[i1] - xs[i0]) / dy if abs(dy) > 1e-9 else 0.0
        x_axis = xs[i0] + slope * (y0 - ys[i0])
    norm = math.hypot(slope, 1.0)
    tangent = np.array([slope / norm, 1.0 / norm])
    normal = np.array([1.0 / norm, -slope / norm])
    point = np.array([x_axis, y0]) + float(lateral_m) * normal
    z_floor = float(meta['floor_ab'][0] + meta['floor_ab'][1] * point[1])
    return point, tangent, normal, z_floor, extrapolated


def _axis_line(nodes):
    """Глобальная прямая МНК по узлам оси: (наклон dx/dy, сдвиг x при y = 0)."""
    nodes = np.asarray(nodes, dtype=np.float64)
    if nodes.shape[0] < 2:
        return (0.0, 0.0)
    slope, intercept = np.polyfit(nodes[:, 1], nodes[:, 0], 1)
    return (float(slope), float(intercept))


def object_box(meta, D, lateral_m, size_m=OBJECT_SIZE_DEFAULT, beam_deg=BEAM_DEG,
               expand_z=False):
    """AABB объекта: (lo, hi, center, floor_z, half_extents, hub_m).

    Расширение на `hub_m = D*tan(theta/2)` -- луч, задевший край, тоже считается
    попавшим (угловой шаг луча theta = 0.1°, край объекта занимает его половину).
    По умолчанию расширяются только поперечная ось и глубина (как в измеренном
    исследовании ориентире: 264 точки на 25 м). `expand_z=True` добавляет то же
    расширение по высоте -- физически так же законно (шаг луча есть и по
    elevation), и даёт 274 точки на 25 м, то есть +4 %.
    Низ коробки лежит на полу (пол -> низ объекта), верх -- на полу + size_z.
    """
    size = np.asarray(size_m, dtype=np.float64).reshape(-1)
    if size.size != 3:
        raise ValueError('size_m must be a 3-vector (sx, sy, sz)')
    point, tangent, normal, z_floor, extrapolated = axis_point(meta, D, lateral_m)
    center = np.array([point[0], point[1], z_floor + 0.5 * size[2]])
    half = 0.5 * size
    hub = float(D) * math.tan(math.radians(0.5 * float(beam_deg)))
    expand = np.array([half[0] + hub, half[1] + hub,
                       half[2] + (hub if expand_z else 0.0)])
    lo = center - expand
    hi = center + expand
    return lo, hi, center, z_floor, half, hub, extrapolated


def _slab(directions, lo, hi):
    """Slab-метод: (t_near, t_far, t_near<R) считает вызывающий; здесь t_near/t_far."""
    directions = np.asarray(directions, dtype=np.float64)
    inv = np.empty_like(directions)
    tiny = np.abs(directions) < 1e-12
    np.divide(1.0, directions, out=inv, where=~tiny)
    inv[tiny] = 1e12
    t1 = lo[None, :] * inv
    t2 = hi[None, :] * inv
    t_near = np.maximum(np.minimum(t1, t2).max(axis=1), 0.0)
    t_far = np.maximum(t1, t2).min(axis=1)
    return t_near, t_far


# ---------------------------------------------------------------------------
# 4. Внесение
# ---------------------------------------------------------------------------

def insert_object(points, rings_meta, D, lateral_m, size_m=OBJECT_SIZE_DEFAULT, seed=0,
                  beam_deg=BEAM_DEG, noise_m=RANGE_NOISE_M, dropout_frac=0.0,
                  expand_z=False):
    """Внести объект `size_m` на дистанцию `D` в реальный кадр `points`.

    Объект -- AABB, лежащий на полу в точке оси D со сносом `lateral_m`.

    Возвращает `(points_new, inserted_mask, meta)`:
      * `points_new` -- кадр с объектом: фон без удалённых точек + точки объекта
        (столько же на луч, сколько их в кадре -- см. докстроку модуля), в том же
        dtype, что и вход;
      * `inserted_mask` -- маска ПО ИСХОДНЫМ точкам: True = точка удалена как
        тень объекта (луч попал в объект). `mask.sum() == meta['points_removed']`;
        точки объекта лежат в `points_new` ДОПИСАННЫМИ в конце (с индекса
        `meta['background_count']`), их число -- `meta['points_inserted']`;
      * `meta` -- ключи:
          `rays_hit`         лучей попало в объект (видимых, t_box < R);
          `points_removed`   точек фона удалено тенью (== mask.sum());
          `points_inserted`  точек объекта в `points_new`;
          `rays_candidate`   лучей прошло предфильтр конусом (диагностика);
          `rays_with_range_beyond_box`  лучей пересекли бокс, но он за поверхностью;
          `object_returns_per_ray`  средняя кратность точек объекта на луч;
          `object_intensity`, `object_range_m`, `object_ray_index`  диагностика точек
                             объекта (индекс -- в наборе видимых лучей-представителей);
          `center`/`box_min`/`box_max`/`half_m`/`hub_m`/`floor_z`/`distance_m`/
          `lateral_m`/`size_m`/`beam_deg`/`noise_sigma_m`/`dropout_frac`/`seed`/
          `n_points_in`/`background_count`/`points_new`  геометрия и счётчики.

    `expand_z=True` добавляет расширение бокса по высоте на D*tan(theta/2) --
    ориентир 264 точки на 25 м получен без него, с ним выходит 274 (см. selfcheck).

    Детерминизм: при одном `seed` результат побитово одинаков (шум и dropout
    берутся из одного `default_rng(seed)` в фиксированном порядке).
    """
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('points must be (N, 3)')
    if points.shape[0] == 0:
        raise ValueError('points is empty: не с чем пересекать лучи')
    dtype = points.dtype if points.dtype.kind == 'f' else np.float64
    pts = np.asarray(points, dtype=np.float64)
    n = pts.shape[0]

    ring = rings_meta.get('ring')
    if ring is not None and len(ring) != n:
        ring = None
    intensity = rings_meta.get('intensity')
    if intensity is not None and len(intensity) != n:
        intensity = None

    lo, hi, center, z_floor, half, hub, extrapolated = object_box(
        rings_meta, D, lateral_m, size_m, beam_deg, expand_z=expand_z)

    r = np.linalg.norm(pts, axis=1)
    keep = r > 1e-9
    if not np.any(keep):
        raise ValueError('all points are at the origin: no ray directions')
    directions = np.zeros_like(pts)
    directions[keep] = pts[keep] / r[keep, None]

    # Кандидаты: конус вокруг центра бокса. Радиус сферы, накрывающей бокс, даёт
    # строгую границу -- ни один попавший луч из конуса выпасть не может.
    rc = float(np.linalg.norm(center))
    r_box = float(np.max([half[0] + hub, half[1] + hub, half[2] + hub]))
    cos_limit = math.cos(math.atan2(r_box, max(rc - r_box, 1e-6))
                         + math.radians(0.5 * beam_deg) + 1e-3)
    cand = keep & ((directions @ (center / rc)) > cos_limit)
    cand_idx = np.flatnonzero(cand)
    meta = {
        'distance_m': float(D), 'lateral_m': float(lateral_m),
        'size_m': tuple(float(v) for v in np.asarray(size_m, dtype=np.float64).reshape(-1)),
        'center': center, 'box_min': lo, 'box_max': hi, 'floor_z': z_floor,
        'half_m': half, 'hub_m': hub, 'beam_deg': float(beam_deg),
        'noise_sigma_m': float(noise_m), 'dropout_frac': float(dropout_frac),
        'seed': int(seed), 'n_points_in': int(n), 'background_count': int(n),
        'rays_candidate': int(cand_idx.size), 'rays_hit': 0, 'rays_with_range_beyond_box': 0,
        'points_removed': 0, 'points_inserted': 0, 'points_new': int(n),
        'object_intensity': np.zeros(0), 'object_ray_index': np.zeros(0, dtype=np.int64),
        'object_range_m': np.zeros(0), 'object_returns_per_ray': 0.0,
        'axis_point': np.array([center[0], center[1]]),
        'axis_extrapolated': bool(extrapolated),
    }
    if cand_idx.size == 0:
        return pts.astype(dtype, copy=True), np.zeros(n, dtype=bool), meta

    d_cand = directions[cand_idx]
    r_cand = r[cand_idx]
    keys_cand = ray_keys(pts[cand_idx], None if ring is None else ring[cand_idx])

    # Ближайший возврат на луч решает видимость; для этого сортируем луч+дальность.
    # Заодно считаем кратность возвратов на луч: у Hesai 128 в этих bag-ах на каждый
    # луч почти ровно 2 точки (346 808 = 2 x 173 408 - 8; у 8 лучей один возврат).
    # Объект отдаёт столько же точек на луч, сколько их в кадре: иначе синтетический
    # объект выходит вдвое реже настоящего и пороги детектора по числу точек не
    # переносятся на синтетику.
    order = np.lexsort((r_cand, keys_cand))
    keys_sorted = keys_cand[order]
    first = np.empty(keys_sorted.size, dtype=bool)
    first[0] = True
    first[1:] = keys_sorted[1:] != keys_sorted[:-1]
    rep = order[first]                              # представитель луча (ближайший)
    rep_keys = keys_sorted[first]
    rep_r = r_cand[rep]
    rep_dir = d_cand[rep]
    mult = np.diff(np.append(np.flatnonzero(first), keys_sorted.size))

    t_near, t_far = _slab(rep_dir, lo, hi)
    crosses = t_far >= t_near
    visible = crosses & (t_near < rep_r)
    n_hit = int(np.count_nonzero(visible))
    meta['rays_candidate'] = int(rep_keys.size)
    meta['rays_hit'] = n_hit
    meta['rays_with_range_beyond_box'] = int(np.count_nonzero(crosses & ~visible))
    if n_hit == 0:
        return pts.astype(dtype, copy=True), np.zeros(n, dtype=bool), meta

    # Тень: удаляем ВСЕ точки кадра на лучах, попавших в объект.
    hit_keys = rep_keys[visible]
    shadow = np.isin(keys_cand, hit_keys)
    mask = np.zeros(n, dtype=bool)
    mask[cand_idx[shadow]] = True

    rng = np.random.default_rng(seed)
    t_obj = t_near[visible]
    dir_obj = rep_dir[visible]
    mult_obj = mult[visible]
    ray_kept = np.ones(dir_obj.shape[0], dtype=bool)
    if dropout_frac > 0.0:
        # dropout 0.5 % ЛУЧЕЙ: луч не регистрирует объект (фон на нём всё равно
        # удалён -- объект его загораживает независимо от того, был ли возврат).
        ray_kept = rng.random(dir_obj.shape[0]) >= float(dropout_frac)
        dir_obj = dir_obj[ray_kept]
        t_obj = t_obj[ray_kept]
        mult_obj = mult_obj[ray_kept]
    obj_pts = np.repeat(dir_obj, mult_obj, axis=0)
    if obj_pts.shape[0]:
        noise = rng.normal(0.0, float(noise_m), obj_pts.shape[0])
        obj_pts = obj_pts * (np.repeat(t_obj, mult_obj) + noise)[:, None]

    # Интенсивность -- бутстрап из реальных значений в бине D +- 10 %.
    obj_intensity = np.zeros(0)
    if intensity is not None and obj_pts.shape[0] > 0:
        band = np.abs(r - float(D)) <= INTENSITY_BIN_FRAC * float(D)
        pool = intensity[band]
        if pool.size == 0:
            pool = intensity
        obj_intensity = rng.choice(pool, size=obj_pts.shape[0], replace=True)

    bg = ~mask
    n_bg = int(np.count_nonzero(bg))
    meta['points_removed'] = int(n - n_bg)
    meta['points_inserted'] = int(obj_pts.shape[0])
    meta['points_new'] = int(n_bg + obj_pts.shape[0])
    meta['background_count'] = n_bg
    meta['object_intensity'] = np.asarray(obj_intensity, dtype=np.float64)
    meta['object_ray_index'] = np.flatnonzero(visible)[ray_kept]
    meta['object_returns_per_ray'] = (float(np.mean(mult_obj)) if mult_obj.size else 0.0)
    meta['object_range_m'] = (np.linalg.norm(obj_pts, axis=1) if obj_pts.size
                              else np.zeros(0))

    points_new = np.empty((meta['points_new'], 3), dtype=dtype)
    points_new[:n_bg] = pts[bg].astype(dtype)
    points_new[n_bg:] = obj_pts.astype(dtype)
    return points_new, mask, meta


def placement_check(points, meta, D, lateral_m, size_m=OBJECT_SIZE_DEFAULT,
                    beam_deg=BEAM_DEG):
    """Проверка постановки: не в стене/платформе ли объект.

    Возвращает dict:
      'center'            центр бокса;
      'points_inside'     измеренных точек кадра внутри бокса (без расширения);
      'points_inside_top' из них в ВЕРХНЕЙ половине бокса;
      'nearest_m'         расстояние от центра бокса до ближайшей измеренной точки;
      'free'              True, если верхняя половина бокса пуста -- объект может
                          стоять (в стену или под платформу он так не попадает).

    Почему верхняя половина, а не весь бокс: на оси пути лежит лоток шириной 0.4 м
    и глубиной 0.36 м, поэтому низ бокса (он ставится на подогнанную плоскость
    пола) пересекается с лотком -- 0-8 точек внутрь. Это осознанный артефакт
    постановки «объект на ось пути», и к видимости он почти не влияет. А вот
    объект, уехавший в стену, заполнен точками до самого верха -- это ловится.

    Зачем вообще: за краем мешей нитей ось продолжается прямой, и на кривых
    запись (roundT_*) продолжение может уехать; проверка по самому кадру отделяет
    «объект не виден, потому что далеко» от «объект случайно поставлен в стену».
    """
    lo, hi, center, z_floor, _half, _hub, _ext = object_box(
        meta, D, lateral_m, size_m, beam_deg)
    points = np.asarray(points, dtype=np.float64)
    if points.shape[0] == 0:
        return {'center': center, 'points_inside': 0, 'points_inside_top': 0,
                'nearest_m': float('inf'), 'free': True}
    in_box = np.all((points >= lo) & (points <= hi), axis=1)
    inside = int(np.count_nonzero(in_box))
    top = int(np.count_nonzero(in_box & (points[:, 2] > center[2])))
    nearest = float(np.linalg.norm(points - center, axis=1).min())
    return {'center': center, 'points_inside': inside, 'points_inside_top': top,
            'nearest_m': nearest, 'free': bool(top == 0)}


# ---------------------------------------------------------------------------
# 5. Простой критерий различимости (сам детектор пишет другой агент)
# ---------------------------------------------------------------------------

def visible_object_mask(points, meta, window_m=1.5):
    """Точки объекта в окне +-`window_m` от истинного центра по Y (плюс радиус)."""
    points = np.asarray(points, dtype=np.float64)
    if points.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    cy = float(meta['center'][1])
    return np.abs(points[:, 1] - cy) <= (window_m + 0.5 * meta['size_m'][1])


def object_point_counts(points_new, meta, cluster_fn=None, window_m=1.5,
                        min_points=5, min_vertical_extent_m=0.3):
    """Число точек объекта и различим ли он: кластер >=min_points в окне +-window_m.

    Критерий намеренно простой (габаритный): детектор пишет другой агент, а тут
    нужно только «есть ли над полом кластер объекта». `cluster_fn` -- функция
    кластеризации (`stitch_bg.count_clusters`), которая отдаёт либо массивы точек,
    либо словари с 'points'/'extent_z'; без неё считается один кластер по всем
    точкам окна.
    """
    obj = np.asarray(points_new, dtype=np.float64)[int(meta['background_count']):]
    in_window = visible_object_mask(obj, meta, window_m)
    sel = obj[in_window]
    out = {'points_object': int(obj.shape[0]), 'points_in_window': int(sel.shape[0]),
           'clusters': 0, 'vertical_extent_m': 0.0, 'visible': False}
    if sel.shape[0] == 0:
        return out
    if cluster_fn is None:
        extent = float(sel[:, 2].max() - sel[:, 2].min())
        out['clusters'] = int(sel.shape[0] >= min_points)
        out['vertical_extent_m'] = extent
        out['visible'] = bool(sel.shape[0] >= min_points and extent >= min_vertical_extent_m)
        return out
    best = 0.0
    n_ok = 0
    for cluster in cluster_fn(sel, min_points=min_points):
        if isinstance(cluster, dict):
            extent = float(cluster['extent_z'])
        else:
            cluster = np.asarray(cluster, dtype=np.float64)
            extent = float(cluster[:, 2].max() - cluster[:, 2].min())
        best = max(best, extent)
        if extent >= min_vertical_extent_m:
            n_ok += 1
    out['clusters'] = n_ok
    out['vertical_extent_m'] = best
    out['visible'] = bool(n_ok > 0)
    return out