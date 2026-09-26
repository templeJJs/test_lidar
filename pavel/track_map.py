"""
Карта пути: сшивка ближних наблюдений колеи в единую линию через SE(2)-позы.

Зачем не длинный фит с одного кадра: на 40+ м пики Z-профиля теряют
однозначность (шпалы, балласт, кабели дают ложные пары), и фит тянется за
шумом. Зато под лидаром, на 5-15 м, колея детектируется практически идеально.

Поезд проезжает 1.5 м за кадр, поэтому каждый участок пути попадает в надёжную
ближнюю зону десятки раз — с разных ракурсов. Сшив эти наблюдения одометрией,
получаем разметку пути на весь бэг, включая участки, где с текущего кадра
рельсы уже не видны.

Ключевое отличие от прежнего rail_mapper.py: точки переводятся в глобальную
систему полным преобразованием SE(2) (сдвиг И поворот), а не только сдвигом
по Y. Прежняя версия писала локальный X повёрнутого кадра прямо в карту, из-за
чего карта систематически спрямляла повороты.

Выход: <bag>/track_map.npz — осевая линия и рельсы как функция пути s.
"""

import argparse
import os
import sys
import time

import numpy as np

from odometry import compute_trajectory, transform_to_global, transform_to_local
from rail_detection import find_rail_lines
from visualize_bag import BagFrames, find_db3

# Зона сбора наблюдений колеи. Прежде 15 м — по прежней детекции, у которой
# дальше пики Z-профиля теряли однозначность (шпалы, балласт, кабели дают
# ложные пары). Полилинейная детекция ведёт нить узлами с проверкой на каждом
# шаге и сама останавливается там, где точек не хватает, поэтому зону можно
# держать шире: лишнего она не придумает, а дальние наблюдения входят в фит с
# пониженным весом (см. rng в build_map).
NEAR_RANGE = 40.0
# Шаг узлов карты по пройденному пути.
NODE_STEP = 1.0
# Ожидаемая колея (1520 мм) — для отбраковки грубо неверных пар.
GAUGE = 1.52
GAUGE_TOL = 0.12
# Отсечение соседнего пути: окно опорной линии по s и допуск поперёк.
# Допуск 0.25 м — меньше самого тесного соскока (0.30 м), но больше
# собственного разброса детекции внутри своего пути (СКО 0.19 м у нормальных
# зон, см. гистограмму в filter_branch).
BRANCH_WIN = 25.0
BRANCH_TOL = 0.25
# Полуширина окна локальной регрессии, которой сглаживается осевая.
SMOOTH_HALF = 5.0
# Асимметричный тримминг: доля отбрасываемых наблюдений с ЛЕВОГО хвоста
# (отрицательные поперечные отклонения). Соскоки детекции на соседний путь
# идут только влево, поэтому правый хвост (честный шум) не трогается.
# Величина выведена из замера доли соскоков: ε = 9.2% на 10739 наблюдениях
# roundT_pressureGate_roundT (401 узел); взято q ≈ 1.5ε — с запасом на
# разброс доли соскоков между узлами, но без задирания: лишний тримминг
# смещал бы оценку вправо. Замер на тех же данных против медианы:
# bias −7.8→+2.3 мм, RMSE 19.5→15.3 мм, max-ошибка 99→59 мм; на Монте-Карло
# с измеренными параметрами смеси выигрыш 9-11% (при ε=30% — 1.5-2×).
# Применяется ТОЛЬКО в оценивателе узла (build_map). В filter_branch опорная
# линия остаётся медианой: замер на roundT_doubleT показал, что asym-trim
# опоры (правый хвост не тримуется) в зонах с параллельным путём СПРАВА
# перетягивает опору на чужой путь — осевая уходила на +447 мм от траектории
# поезда (s=283..317), а на roundT_pG замена опоры не давала ничего (keep-сет
# после итераций совпадал с медианным точно).
TRIM_LEFT_Q = 0.15


def _wlstsq_trim_left(V, y, w, q=TRIM_LEFT_Q):
    """Взвешенный lstsq с одним раундом тримминга нижней доли q остатков.

    Соскоки на соседний путь тянут фит влево (отрицательные остатки), поэтому
    после первого фита нижние q остатков отбрасываются и фит повторяется на
    уцелевших с теми же весами. Остатки для тримминга берутся невзвешенными:
    отбраковывается фактический вынос, а не взвешенный вклад.
    """
    Vw = V * w[:, None]
    coef = np.linalg.lstsq(Vw, y * w, rcond=None)[0]
    r = y - V @ coef
    n_cut = int(np.floor(q * len(r)))
    if 0 < n_cut <= len(r) - 2:
        keep = np.argsort(r)[n_cut:]
        coef = np.linalg.lstsq(Vw[keep], y[keep] * w[keep], rcond=None)[0]
    return coef[-1]


def collect_observations(frames, poses, near_range=NEAR_RANGE, own_x_range=1.5,
                         verbose=True, progress_every=50):
    """Собирает наблюдения колеи со всех кадров в глобальной системе.

    С каждого кадра берутся только сырые пары пиков (y, x_left, x_right) из
    ближней зоны — это исходные измерения, без промежуточного фита.

    Возвращает (obs, planes):
        obs — (M, 6): global_x_left, global_y_left, global_x_right,
                      global_y_right, дальность в своём кадре, превышение
                      верха головки НАД ПЛОСКОСТЬЮ ПОЛА своего кадра
                      (nan, если детекция высоту не даёт)
        planes — (N, 3) плоскость земли на кадр (nan, если не найдена)
    """
    total = len(frames)
    obs = []
    planes = np.full((total, 3), np.nan)
    n_ok = 0
    t0 = time.time()

    if verbose:
        print(f'Сбор наблюдений колеи (ближняя зона {near_range:.0f} м)...')

    _rg = sys.modules.get('rail_geometry')
    _bag_key = getattr(frames, 'db_path', None)

    for i in range(total):
        pts, intensity, _ = frames[i]
        if _rg is not None:
            # какой кадр какой записи — для тёплой сборки и связи решений
            _rg.set_frame_context(_bag_key, i)
        _, ground, pairs = find_rail_lines(
            pts, intensity, own_x_range=own_x_range,
            max_y_distance=near_range, return_pairs=True)

        if ground is not None:
            planes[i] = ground
        if len(pairs) < 3:
            continue

        # Отбраковка пар с неверной колеёй: на стрелках и у платформы
        # попадаются пики от чужого пути и от края платформы.
        gauge = np.abs(pairs[:, 2] - pairs[:, 1])
        good = np.abs(gauge - GAUGE) < GAUGE_TOL
        if good.sum() < 3:
            continue
        pairs = pairs[good]
        n_ok += 1

        y = pairs[:, 0]
        left = transform_to_global(np.column_stack([pairs[:, 1], y]), poses[i])
        right = transform_to_global(np.column_stack([pairs[:, 2], y]), poses[i])
        # 5-й столбец — дальность в СВОЁМ кадре (ось Y смотрит назад).
        # Разброс наблюдения растёт с ней (СКО 0.077 м на 0-3 м против 0.137 м
        # на 9-12 м) при том же среднем, поэтому дальние наблюдения не врут,
        # а просто шумнее — их вес в фите понижается.
        # Высота головки: полилинейная детекция даёт её измеренной (столбцы
        # 3-4 пар), прежняя — нет.
        #
        # В карту идёт НЕ сырой z, а превышение над плоскостью пола СВОЕГО
        # кадра. SE(2)-одометрия вертикаль не отслеживает ("движение плоское",
        # transform_to_global не трогает z), поэтому сырые z разных кадров
        # лежат в разных вертикальных отсчётах: сенсор идёт по профилю пути, и
        # высота пола под ним гуляет на 325 мм за бэг (doubleT_platform).
        # Складывать такие z в одну карту — та же ошибка, что прежний rail_mapper
        # делал с X. Превышение над полом — инвариант кадра (~0.16 м, УГР),
        # его сшивать корректно, а абсолютную высоту потребитель восстановит,
        # добавив плоскость своего кадра.
        # Опора — плоскость пола В ТОЧКЕ ОСИ ПУТИ (x_mid), а не под каждой
        # нитью: поперечный наклон плоскости a сам по себе ненадёжен (завал до
        # -0.068 на doubleT_platform f50), и на плече ±0.76 м он разносит левую
        # и правую нить на 115 мм при истинной разнице около нуля. В точке оси
        # вклад a общий для обеих нитей и в превышение не входит. Проверка на
        # кадре с ровной плоскостью (a = +0.005): левая 134 мм, правая 136 мм.
        if pairs.shape[1] >= 5 and ground is not None:
            z_abs = (pairs[:, 3] + pairs[:, 4]) / 2
            x_mid = (pairs[:, 1] + pairs[:, 2]) / 2
            a_g, b_g, c_g = ground
            z_head = z_abs - (a_g * x_mid + b_g * y + c_g)
        else:
            z_head = np.full(len(pairs), np.nan)
        obs.append(np.column_stack([left, right, -y, z_head]))

        if verbose and progress_every and (i + 1) % progress_every == 0:
            print(f'  {i + 1}/{total}  кадров с колеёй {n_ok}  {time.time() - t0:.1f}s')

    if verbose:
        print(f'  готово за {time.time() - t0:.1f}s: {n_ok}/{total} кадров с колеёй')

    return (np.vstack(obs) if obs else np.empty((0, 6))), planes


def project_to_path(pts, poses, arc):
    """Проецирует точки на траекторию, возвращает (s, dist).

    Ближайшая ПОЗА — не проекция, а квантование: s мог принимать только
    столько значений, сколько кадров (267 на 6434 наблюдения при шаге поз
    1.5 м). Узлы карты идут через 1 м, то есть чаще этой сетки, поэтому
    соседние узлы получали один и тот же набор наблюдений, а продольное
    положение узла задавалось не путём, а тем, какой кадр случайно оказался
    ближе. Отсюда шаг узлов 0.90 м вместо 1.00 и изломы линии до 104° при
    том, что сама траектория гладкая (максимум 11°).

    Здесь точка проецируется на ОТРЕЗКИ ломаной траектории, а s получается
    интерполяцией вдоль отрезка — непрерывная величина.
    """
    pts = np.asarray(pts, dtype=float)
    A, B = poses[:-1, :2], poses[1:, :2]
    AB = B - A
    L2 = np.maximum((AB ** 2).sum(axis=1), 1e-12)

    # Отрезков сотни, точек тысячи — прямой перебор дешевле KD-дерева и, в
    # отличие от него, даёт именно проекцию, а не ближайший узел.
    d = pts[:, None, :] - A[None, :, :]
    t = np.clip((d * AB[None, :, :]).sum(axis=2) / L2[None, :], 0.0, 1.0)
    proj = A[None, :, :] + t[:, :, None] * AB[None, :, :]
    dist2 = ((pts[:, None, :] - proj) ** 2).sum(axis=2)

    seg = dist2.argmin(axis=1)
    rows = np.arange(len(pts))
    s = arc[seg] + t[rows, seg] * (arc[seg + 1] - arc[seg])
    return s, np.sqrt(dist2[rows, seg])


def path_frame(s_nodes, poses, arc):
    """Опорная точка и единичная нормаль пути в каждом узле.

    Нужна, потому что медиана по глобальным X и Y по отдельности корректна
    только когда путь идёт вдоль оси. На повороте облако наблюдений в окне
    узла вытянуто вдоль пути, и покоординатная медиана смешивает продольный
    разброс с поперечным. Положение узла строится в базисе пути: вдоль —
    фиксировано сеткой s, поперёк — медиана наблюдений.
    """
    px = np.interp(s_nodes, arc, poses[:, 0])
    py = np.interp(s_nodes, arc, poses[:, 1])

    # Касательная берётся по базе ±half_base метров, а не по соседним узлам:
    # одношаговая разность на шаге 1 м ловит дрожание одометрии, а на краю
    # вырождается в одностороннюю и может схлопнуться почти в ноль.
    half_base = 3.0
    ax = np.interp(s_nodes - half_base, arc, poses[:, 0])
    ay = np.interp(s_nodes - half_base, arc, poses[:, 1])
    bx = np.interp(s_nodes + half_base, arc, poses[:, 0])
    by = np.interp(s_nodes + half_base, arc, poses[:, 1])
    gx, gy = bx - ax, by - ay

    n = np.hypot(gx, gy)
    bad = n < 1e-6
    if bad.any():  # вырожденный край — берём направление ближайшего годного
        good = np.nonzero(~bad)[0]
        if len(good) == 0:
            gx, gy, n = np.ones_like(gx), np.zeros_like(gy), np.ones_like(n)
        else:
            src_i = good[np.argmin(np.abs(np.nonzero(bad)[0][:, None] - good[None, :]), axis=1)]
            gx[bad], gy[bad], n[bad] = gx[src_i], gy[src_i], n[src_i]
    gx, gy = gx / n, gy / n
    return px, py, -gy, gx  # точка + нормаль = касательная, повёрнутая на 90°


def filter_branch(s_obs, lat_c, win=BRANCH_WIN, tol=BRANCH_TOL, n_iter=4,
                  min_in_win=3):
    """Отсекает наблюдения с соседнего пути (ветка стрелки, параллельный путь).

    Это те самые «соскоки вбок»: детектор кратковременно цепляется за чужую
    пару рельсов в 0.3-0.95 м сбоку и возвращается обратно. Колея у них
    правильная (1570 мм против 1577), высота головки правильная — по одному
    наблюдению они неотличимы от своих. Отличает их то, что СВОЙ путь под
    поездом непрерывен: поезд по нему едет, значит lat(s) — гладкая функция,
    а ветка от неё отходит.

    Замер на roundT_pressureGate_roundT: 747 наблюдений (11.6%) с выносом
    < -0.30 м, сгруппированы в зонах s≈105, 280-320, 380-400 м. В 51 узле из
    400 они составляли больше половины наблюдений и перетягивали медиану —
    отсюда и кривизна карты.

    Опорная линия строится медианой в скользящем окне и уточняется
    итеративно: на первой итерации выбросы ещё тянут её, на последующих
    уже отсечены. Опора обязана быть робастной в ОБЕ стороны: ветка может
    лежать и справа, и тогда односторонний тримминг перетягивает опору на
    чужой путь (замер на roundT_doubleT: с asym-trim 15%/0 опоры осевая
    ушла на +447 мм от траектории поезда в зоне s=283..317, с медианой +18).
    Остаточное левое смещение узлов от хвоста соскоков убирается не здесь,
    а в оценивателе узла build_map (см. TRIM_LEFT_Q): на roundT_pG замена
    медианной опоры на asym-trim не меняла keep-сет вообще.
    """
    s_obs = np.asarray(s_obs, dtype=float)
    lat_c = np.asarray(lat_c, dtype=float)
    if len(s_obs) == 0:
        return np.zeros(0, dtype=bool)

    grid = np.arange(s_obs.min(), s_obs.max() + 1.0, 1.0)
    keep = np.ones(len(s_obs), dtype=bool)
    ref = np.full(len(grid), np.nan)

    for _ in range(n_iter):
        for k, g0 in enumerate(grid):
            m = keep & (np.abs(s_obs - g0) < win / 2)
            if m.sum() >= min_in_win:
                ref[k] = np.median(lat_c[m])
        ok = ~np.isnan(ref)
        if ok.sum() < 2:
            return np.ones(len(s_obs), dtype=bool)
        keep = np.abs(lat_c - np.interp(s_obs, grid[ok], ref[ok])) < tol

    return keep


def build_map(obs, poses, arc, node_step=NODE_STEP, min_obs=3, max_lat=2.5,
              smooth_half=SMOOTH_HALF):
    """Сводит наблюдения в осевую линию пути, параметризованную путём s.

    Наблюдение проецируется на траекторию (project_to_path), и его положение
    раскладывается в базисе пути на продольное s и поперечное смещение.
    Медиана берётся ТОЛЬКО по поперечному: вдоль пути узел стоит там, где его
    ставит сетка s. Покоординатная медиана по глобальным X/Y, как было раньше,
    на повороте смешивала продольный разброс с поперечным и давала изломы.

    Траектория поезда — сама по себе осевая линия, поэтому она служит опорой,
    а наблюдения уточняют положение рельсов поперёк неё.

    Возвращает dict с узлами: s, center_x/y, left_x/y, right_x/y, rail_dz,
    n_obs. rail_dz — измеренное превышение верха головки над плоскостью пола
    (nan, если детекция высоту не давала). Абсолютная высота в кадре i:
    z = a*x + b*y + c + rail_dz, где (a, b, c) — planes[i]. Прежде на это
    место подставлялась константа 0.15 м.
    """
    if len(obs) == 0:
        return None

    mid = np.column_stack([(obs[:, 0] + obs[:, 2]) / 2,
                           (obs[:, 1] + obs[:, 3]) / 2])
    s_obs, dist = project_to_path(mid, poses, arc)

    # Наблюдение дальше max_lat от траектории — чужой путь или срыв детекции:
    # поезд едет по своей колее, её центр не может уйти на метры вбок.
    keep = dist < max_lat
    s_obs, obs = s_obs[keep], obs[keep]
    if len(obs) == 0:
        return None

    # Строго внутри траектории: np.interp за её концом отдаёт одну и ту же
    # точку, касательная там вырождается (норма падает до 0.02), и нормаль
    # становится произвольной — такие узлы давали колею 13 и 737 мм.
    s_nodes = np.arange(arc[0], arc[-1], node_step)
    px, py, nx, ny = path_frame(s_nodes, poses, arc)

    # Отсев соседнего пути ДО сборки узлов: иначе в зонах стрелок чужие
    # наблюдения перетягивают медиану узла и кривят карту.
    lat_c = 0.5 * (
        (obs[:, 0] - np.interp(s_obs, s_nodes, px)) * np.interp(s_obs, s_nodes, nx)
        + (obs[:, 1] - np.interp(s_obs, s_nodes, py)) * np.interp(s_obs, s_nodes, ny)
        + (obs[:, 2] - np.interp(s_obs, s_nodes, px)) * np.interp(s_obs, s_nodes, nx)
        + (obs[:, 3] - np.interp(s_obs, s_nodes, py)) * np.interp(s_obs, s_nodes, ny))
    on_track = filter_branch(s_obs, lat_c)
    if on_track.sum() >= min_obs:
        s_obs, obs = s_obs[on_track], obs[on_track]

    out = {k: [] for k in ('s', 'center_x', 'center_y', 'left_x', 'left_y',
                           'right_x', 'right_y', 'rail_dz', 'n_obs')}

    order = np.argsort(s_obs)
    s_sorted = s_obs[order]
    obs_sorted = obs[order]

    # Поперечное смещение нитей считается один раз, в базисе ближайшего узла:
    # дальше фит идёт уже по скалярам lat(s), а не по парам координат.
    idx = np.clip(np.searchsorted(s_nodes, s_sorted), 0, len(s_nodes) - 1)
    lat_l = ((obs_sorted[:, 0] - px[idx]) * nx[idx]
             + (obs_sorted[:, 1] - py[idx]) * ny[idx])
    lat_r = ((obs_sorted[:, 2] - px[idx]) * nx[idx]
             + (obs_sorted[:, 3] - py[idx]) * ny[idx])
    rng = obs_sorted[:, 4]
    z_head = obs_sorted[:, 5] if obs_sorted.shape[1] > 5 else np.full(len(rng), np.nan)

    lo = np.searchsorted(s_sorted, s_nodes - smooth_half, side='left')
    hi = np.searchsorted(s_sorted, s_nodes + smooth_half, side='right')

    for k, s in enumerate(s_nodes):
        n_win = hi[k] - lo[k]
        if n_win < min_obs:
            continue
        sl = slice(lo[k], hi[k])

        # Веса: трикуб по расстоянию вдоль пути (ближе к узлу — весомее) и
        # понижение для наблюдений, снятых издалека внутри своего кадра.
        ds = s_sorted[sl] - s
        w = (1.0 - np.abs(ds / smooth_half) ** 3) ** 3
        w = w / (1.0 + rng[sl] / 10.0)

        # Локальная ПРЯМАЯ, а не медиана в окне ±1 м. Путь реально смещается
        # вбок (замер: плавный сход на 0.2 м за 2.5 м пути у s≈149), и медиана
        # в узком окне превращала такой скат в ступеньку: соседние узлы брали
        # разные куски ската и расходились на 0.10-0.16 м при том, что сами
        # наблюдения внутри каждого узла согласованы (ст. ошибка медианы всего
        # 0.014-0.069 м). Прямая описывает скат наклоном, а не ступенькой.
        # Фит — с одним раундом асимметричного тримминга остатков (15% слева):
        # голый lstsq не робастен, а соскоки на соседний путь (левый хвост)
        # смещали узел (замер на roundT_pG: bias −7.8 мм у медианной опоры,
        # у asym-trim +2.3 мм — см. TRIM_LEFT_Q).
        V = np.vander(ds, 2)
        fit_l = _wlstsq_trim_left(V, lat_l[sl], w)
        fit_r = _wlstsq_trim_left(V, lat_r[sl], w)
        fit_c = (fit_l + fit_r) / 2

        out['s'].append(s)
        out['left_x'].append(px[k] + fit_l * nx[k])
        out['left_y'].append(py[k] + fit_l * ny[k])
        out['right_x'].append(px[k] + fit_r * nx[k])
        out['right_y'].append(py[k] + fit_r * ny[k])
        out['center_x'].append(px[k] + fit_c * nx[k])
        out['center_y'].append(py[k] + fit_c * ny[k])
        # Превышение — медианой, а не взвешенной прямой: оно почти постоянно
        # (УГР), но отдельные узлы полилинии садятся на скол головки или на
        # закладную, и такой выброс прямую тянет, а медиану нет.
        z_win = z_head[sl]
        z_ok = z_win[~np.isnan(z_win)]
        out['rail_dz'].append(np.median(z_ok) if len(z_ok) else np.nan)
        out['n_obs'].append(n_win)

    return {k: np.array(v) for k, v in out.items()}


def smooth_planes(planes, window=15):
    """Сглаживает плоскости земли по кадрам скользящей медианой.

    Плоскость оценивается на каждом кадре независимо, и её коэффициенты
    заметно дрожат: продольный наклон b меняется между соседними кадрами
    так, что высота на 100 м прыгает на метр и больше. Путь физически
    гладкий, поэтому дрожание — шум оценки, а не рельеф.

    Пропуски (nan) заполняются интерполяцией по соседним кадрам.
    """
    planes = np.asarray(planes, dtype=float).copy()
    n = len(planes)
    idx = np.arange(n)

    for c in range(planes.shape[1]):
        col = planes[:, c]
        ok = ~np.isnan(col)
        if ok.sum() < 2:
            continue
        col = np.interp(idx, idx[ok], col[ok])
        half = max(1, window // 2)
        sm = np.empty(n)
        for i in range(n):
            lo, hi = max(0, i - half), min(n, i + half + 1)
            sm[i] = np.median(col[lo:hi])
        planes[:, c] = sm

    return planes


def map_to_frame(track, poses, frame_idx, max_range=200.0):
    """Проецирует карту в систему кадра: где путь виден с этого кадра.

    Это и есть разметка, работающая там, где детекция невозможна: положение
    рельсов на 100+ м известно из наблюдений, сделанных, когда поезд туда
    доехал.

    Возвращает dict с массивами (s, left_x, right_x, center_x, fwd) в
    локальных координатах кадра; fwd — дальность вперёд по ходу (метры).
    """
    pose = poses[frame_idx]
    left = transform_to_local(np.column_stack([track['left_x'], track['left_y']]), pose)
    right = transform_to_local(np.column_stack([track['right_x'], track['right_y']]), pose)

    # Ось Y лидара смотрит назад: вперёд по ходу — это -Y
    fwd = -(left[:, 1] + right[:, 1]) / 2
    keep = (fwd >= 0) & (fwd <= max_range)

    out = {
        's': track['s'][keep],
        'fwd': fwd[keep],
        'left_x': left[keep, 0],
        'right_x': right[keep, 0],
        'center_x': (left[keep, 0] + right[keep, 0]) / 2,
        'left_y': left[keep, 1],
        'right_y': right[keep, 1],
    }
    # Сшитое превышение головки над полом, если карта его несёт: потребителю
    # больше не нужна константа вроде 0.15 м.
    if 'rail_dz' in track:
        out['rail_dz'] = np.asarray(track['rail_dz'])[keep]
    return out


def run(bag_dir, near_range=NEAR_RANGE, node_step=NODE_STEP, min_intensity=50):
    db_path = find_db3(bag_dir)
    frames = BagFrames(db_path, cache_size=8)
    print(f'Bag: {db_path}, {len(frames)} кадров')

    traj_path = os.path.join(bag_dir, 'trajectory.npz')
    if os.path.exists(traj_path):
        traj = dict(np.load(traj_path))
        print(f'Траектория из кэша: {traj["arc"][-1]:.0f} м')
    else:
        traj = compute_trajectory(frames, min_intensity=min_intensity)
        np.savez(traj_path, **traj)

    poses, arc = traj['poses'], traj['arc']

    obs, planes = collect_observations(frames, poses, near_range=near_range)
    if len(obs) == 0:
        print('Колея не найдена ни на одном кадре')
        return

    track = build_map(obs, poses, arc, node_step=node_step)
    if track is None or len(track['s']) < 2:
        print('Не удалось построить карту')
        return

    gauge = np.hypot(track['left_x'] - track['right_x'],
                     track['left_y'] - track['right_y'])

    print(f'\nКарта пути: {len(track["s"])} узлов, '
          f'{track["s"][0]:.0f}..{track["s"][-1]:.0f} м')
    print(f'  наблюдений на узел: med {np.median(track["n_obs"]):.0f}, '
          f'min {track["n_obs"].min()}')
    print(f'  колея: med {gauge.mean() * 1000:.0f} мм, '
          f'СКО {gauge.std() * 1000:.0f} мм '
          f'(эталон {GAUGE * 1000:.0f} мм)')

    planes_sm = smooth_planes(planes)
    dz100 = np.abs(np.diff(planes_sm[:, 1])) * 100
    print(f'  плоскость земли сглажена: скачок Z на 100 м '
          f'med {np.median(dz100) * 1000:.0f} мм, p95 {np.percentile(dz100, 95) * 1000:.0f} мм')

    out = os.path.join(bag_dir, 'track_map.npz')
    np.savez(out, planes=planes_sm, planes_raw=planes, **track)
    print(f'  -> {out}')


def main():
    parser = argparse.ArgumentParser(
        description='Карта пути: сшивка ближних наблюдений колеи')
    parser.add_argument('bag_dirs', nargs='+')
    parser.add_argument('--near-range', type=float, default=NEAR_RANGE,
                        help=f'ближняя зона детекции, м (по умолчанию {NEAR_RANGE:.0f})')
    parser.add_argument('--node-step', type=float, default=NODE_STEP,
                        help='шаг узлов карты, м')
    parser.add_argument('--polyline', action='store_true',
                        help='детекция колеи ПОЛИЛИНИЕЙ через test/track_geometry.py '
                             'вместо фита прямой/параболой (см. rail_geometry.py)')
    args = parser.parse_args()

    if args.polyline:
        import rail_geometry
        rail_geometry.enable()
        print('Детекция: ПОЛИЛИНИЯ (test/track_geometry.py)')

    for bag_dir in args.bag_dirs:
        print(f'\n{"=" * 60}\n{bag_dir}\n{"=" * 60}')
        run(bag_dir, near_range=args.near_range, node_step=args.node_step)


if __name__ == '__main__':
    main()
