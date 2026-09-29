"""Картинка: ось вдаль (блок 1б) против карты пути.

Показывает на виде сверху то, чего не видно в цифрах: где ось легла на
эталон, где отстала, а где улетела в сторону. Улёт особенно важен — на
кадрах, где карта коротка, он в метрику ошибки НЕ попадает, но на картинке
виден сразу.

    # три кадра подряд, окно на экран
    python plot_axis.py roundT_pressureGate_roundT 108,123,240

    # в файл
    python plot_axis.py roundT_pressureGate_roundT 108,123,240 -o axis.png

    # самые плохие кадры бэга найти самому и нарисовать
    python plot_axis.py roundT_doubleT --worst 6

    # сравнить сглаженную ось с сырой (обе на одной картинке)
    python plot_axis.py roundT_pressureGate_roundT 123 --raw

    # плотная стена: подложить точки 6 прошлых кадров по одометрии
    python plot_axis.py roundT_pressureGate_roundT 123 --accum

    # ФИЛЬМ: вид сверху кадр за кадром, окно --accum скользит вместе с кадром
    python plot_axis.py roundT_pressureGate_roundT --movie out.mp4 \
        --movie-range 100:400:1 --accum 5 --walls

    # НАЙДЕННЫЕ СТЕНЫ (блок check_wall) поверх картинки
    python plot_axis.py roundT_pressureGate_roundT 108,123,240 --walls

    # НАХОДКИ блока 3 (препятствия) поверх картинки, вместе с коридором
    python plot_axis.py doubleT_obstacle 40 --corridor --obstacles

    # СКОЛЬКО ПРЕПЯТСТВИЙ ВО ВСЁМ БЭГЕ: замер рядом с картинкой кадра
    python plot_axis.py doubleT_platform 120 --obstacles --metrics
    python plot_axis.py doubleT_platform 120 --metrics 10 --episodes

    # профиль стены по высоте в одном бине: видно, за что цепляется кромка
    python plot_axis.py roundT_pressureGate_roundT 108 --profile 47

Что на картинке БЕЗ --walls (ось блока 1б, axis_wall):
    зелёная толстая  — путь по карте track_map.npz, эталон
    оранжевый пунктир— прямая ось блока 1 (как было бы без ведения стеной)
    красная линия    — ось блока 1б, точки раскрашены по статусу бина
    чёрные точки     — кромка стены, за которую зацепилось ведение
    серая штриховая  — reach: дальше ось не подтверждена измерением
    светло-серое     — точки кадра в высоте габарита

С --walls картинка показывает БЛОК 1в (wall_rules), и ось блока 1б с неё
УБИРАЕТСЯ вместе со своими кромками и reach. Иначе рисовались две оси разных
блоков по разным правилам, с двумя легендами про «ведёт стена», и понять, чья
линия чья, было нельзя. Сравнивать надо с эталоном, а не с предыдущей версией
себя.

С --obstacles поверх кладутся находки блока 3 (obstacles.detect):
    красная рамка    — препятствие В ГАБАРИТЕ, с подписью дальности и высоты
    серая рамка      — объект есть, но мимо коридора
    пунктирная рамка — бин не измерен ('hole'): находка настоящая, но за
                       неизвестным участком (confirmed=False)
    цветные точки    — точки самой находки: видно, из чего она собрана

    цветные линии    — стены блока check_wall, каждая на своём отрезке
                       дальности; кружки — кромки, легшие на линию, толщина
                       растёт с числом подтвердивших слоёв высоты
    ось по РЕЖИМАМ   — зелёный рельсы / голубой середина двух стен /
                       оранжевый ведёт стена / серый продолжение формы
    красный крестик  — tight: ось ближе к стене, чем бывает отступ (аномалия)
    пустой квадрат   — отступ в этом бине не измерен, взят по умолчанию

ВЕДУЩАЯ стена обведена и подписана — та, по которой ось РЕАЛЬНО построена
(последний бин в режиме LEAD). На прямой ведущей нет: там ось идёт по середине
двух стен, и выбирать сторону не нужно.

ВАЖНО про эталон: карта кончается на разной дальности в разных кадрах
(медиана 149 м, но p10 = 43 м). Зелёная линия рисуется только там, где карта
есть; где её нет — сверять не с чем, и ошибка там ничего не значит.
"""
import argparse
import os
import sys
import numpy as np

# И guard/, и корень проекта: отсюда импортируются оба — `guard.*` как пакет
# и `visualize_bag` / `track_map`, лежащие в корне.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../for_hackathon')

# Цвет несёт статус бина: чем ось ведётся именно здесь.
STATUS_STYLE = (
    ('seed',     'tab:purple', 'ось: от рельсов (ближняя зона)'),
    ('straight', 'tab:blue',   'ось: прямая, стена согласна'),
    ('wall',     'tab:red',    'ось: ведёт стена'),
    ('hole',     '0.55',       'ось: дыра — предсказание, НЕ измерение'),
)


def load(bag):
    """-> (frames, track_map, poses). Бэг можно звать коротким именем."""
    from visualize_bag import BagFrames, find_db3
    d = bag if os.path.isdir(bag) else os.path.join(ROOT, bag)
    if not os.path.isdir(d):
        sys.exit(f'нет такого бэга: {d}')
    db_path = find_db3(d)
    frames = BagFrames(db_path, cache_size=4)
    frames.db_path = db_path        # ключ тёплой сборки полилинии (rail_poly)
    tmp = os.path.join(d, 'track_map.npz')
    trp = os.path.join(d, 'trajectory.npz')
    tm = dict(np.load(tmp)) if os.path.exists(tmp) else None
    poses = np.load(trp)['poses'] if os.path.exists(trp) else None
    if tm is None:
        print(f'{bag}: нет track_map.npz — эталон не нарисуется')
    return frames, tm, poses


def ref_axis(tm, poses, i, max_range=150.0):
    """Истинная ось из карты в системе кадра -> (fwd, lat), отсортировано."""
    if tm is None or poses is None or i >= len(poses):
        return None, None
    from track_map import map_to_frame
    pr = map_to_frame(tm, poses, i, max_range=max_range)
    if len(pr['fwd']) < 2:
        return None, None
    o = np.argsort(pr['fwd'])
    return pr['fwd'][o], pr['center_x'][o]


def worst_frames(frames, tm, poses, n, step, at=(30.0, 50.0, 70.0)):
    """Кадры с наибольшей ошибкой против карты — их и интересно смотреть."""
    from guard.run import analyze
    from guard import axis_wall as AW
    rows = []
    for i in range(0, min(len(frames), len(poses) if poses is not None
                          else len(frames)), step):
        rf, rx = ref_axis(tm, poses, i)
        if rf is None:
            continue
        tr = analyze(frames[i][0])[0]['far']
        if tr is None:
            continue
        e = [abs(float(AW.lat_at(tr, r)) - float(np.interp(r, rf, rx)))
             for r in at if r <= rf.max() and r <= tr['reach']]
        if e:
            rows.append((max(e), i))
    rows.sort(reverse=True)
    if rows:
        print('худшие кадры (ошибка, м):  '
              + '  '.join(f'f{i}={e:.2f}' for e, i in rows[:n]))
    return [i for _, i in rows[:n]]


def accum_cloud(frames, poses, i, n_back):
    """Точки n_back предыдущих кадров, перенесённые в систему кадра i.

    Зачем. Один кадр даёт разреженную стену: между кольцами лидара дыры, и на
    виде сверху кромка выглядит пунктиром. Соседние кадры измеряли ту же
    неподвижную стену с точки в метре позади — их точки закрывают дыры.
    Копится ОБЛАКО, а не бины коридора (это делает guard.accum): картинке
    нужны именно точки.

    Движущийся объект при таком переносе размажется вдоль пути — поэтому
    накопление здесь только для показа стен, не для габарита.

    -> (fwd, lat, up) уже в системе кадра i, без точек самого кадра i.
    """
    from guard.geom import to_frame
    from guard.accum import _shift_along_path
    out = []
    for j in range(max(0, i - n_back), i):
        mv = _shift_along_path(poses, j, i)
        if mv is None:
            continue
        d_fwd, d_lat, d_th = mv
        f, la, up = to_frame(frames[j][0])
        # Поза старого кадра в системе текущего: сначала поворот, потом сдвиг
        # на пройденное. Знак d_* — как в guard.accum: новый кадр впереди.
        c, sn = np.cos(-d_th), np.sin(-d_th)
        fr = c * f - sn * la - d_fwd
        lr = sn * f + c * la - d_lat
        out.append((fr, lr, up))
    if not out:
        return None
    return tuple(np.concatenate(x) for x in zip(*out))


WALL_COLORS = ('tab:red', 'tab:blue', 'tab:green', 'tab:purple', 'tab:brown',
               'tab:olive', 'tab:cyan', 'magenta')


# Цвет режима ведения (блок wall_rules). Ось красится по режиму, потому что
# «ось уехала» само по себе не отладочная информация: нужно видеть, ПОЧЕМУ —
# по рельсам вела, по середине двух стен, по одной стене с отступом или просто
# продолжала форму в дыре. Пороги правятся по кадрам, где глаз видит не тот
# режим, а не по эталону: карта (track_map) сама не идеальна.
MODE_COLORS = {'rails': 'tab:green', 'center': 'tab:cyan',
               'lead': 'tab:orange', 'hold': '0.55'}
MODE_LABELS = {'rails': 'ось: рельсы', 'center': 'ось: середина двух стен',
               'lead': 'ось: ведёт стена', 'hold': 'ось: продолжение формы'}


def draw_walls(ax, fwd, lat, up, floor, max_range, axis=None, off=None):
    """Стены и ось по правилам ведения поверх вида сверху -> строка заголовка.

    Каждая стена рисуется ТОЛЬКО на своём отрезке [d_lo, d_hi]: где она
    прослежена, там и линия. Экстраполяция сюда не выносится намеренно —
    картинка должна показывать, что измерено, а не что достроено.

    Ось (блок `wall_rules`) рисуется поверх, посегментно и цветом РЕЖИМА, а
    ведущая стена подсвечивается той, по которой ось реально построена (а не
    той, что раньше обрывается, как считала старая `check_wall.leading`).
    """
    from guard import check_wall as CW
    from guard import wall_rules as WR

    walls = CW.detect(fwd, lat, up, floor, max_range=max_range)
    if not walls:
        return 'стен не найдено'

    bins = WR.build(walls, axis, off=off, max_range=max_range)
    lead, lead_side = WR.leading(bins)

    for n, w in enumerate(walls):
        col = WALL_COLORS[n % len(WALL_COLORS)]
        t = np.linspace(w.d_lo, w.d_hi, 60)
        # Толщина несёт число слоёв: чем в большем числе полос высоты стена
        # подтверждена, тем меньше шансов, что это кабель или навес.
        ax.plot(t, w.lat_at(t), color=col, lw=1.0 + 0.5 * len(w.layers),
                alpha=0.85, zorder=8,
                label=f'стена {n + 1}: {w.d_lo:.0f}-{w.d_hi:.0f} м, '
                      f'слоёв {len(w.layers)}, rms {w.rms:.2f}')
        ax.plot(w.d, w.e, 'o', color=col, ms=3, mfc='none', zorder=8)
        if w is lead:
            ax.plot(t, w.lat_at(t), color=col, lw=5.0, alpha=0.22, zorder=7)
            ax.annotate('ведущая', (w.d_hi, w.lat_at(w.d_hi)), fontsize=8,
                        color=col, xytext=(4, 4), textcoords='offset points')

    # Ось по режимам: сегмент между соседними бинами красится режимом правого
    # из них, поэтому смена цвета стоит ровно на бине, где режим сменился.
    seen = set()
    for a, b in zip(bins, bins[1:]):
        col = MODE_COLORS.get(b.mode, 'k')
        lbl = None if b.mode in seen else MODE_LABELS.get(b.mode)
        seen.add(b.mode)
        ax.plot([a.d, b.d], [a.lat, b.lat], color=col, lw=2.2, alpha=0.95,
                zorder=9, label=lbl)
    # Бины, где полугабарит подошёл к стене ближе зазора: ось не правилась
    # (на завороте отступ постоянен), но верить ей здесь меньше оснований.
    tight = [(b.d, b.lat) for b in bins if b.tight]
    if tight:
        ax.plot(*zip(*tight), 'x', color='tab:red', ms=6, mew=1.6, zorder=10,
                label='габарит у стены')
    # Отступ, взятый по умолчанию (ближней зоны не было — мерить негде).
    guess = [(b.d, b.lat) for b in bins
             if b.mode == 'lead' and not b.off_measured]
    if guess:
        ax.plot(*zip(*guess), 's', color='tab:orange', ms=4, mfc='none',
                zorder=10, label='отступ не измерен')

    return f'стен {len(walls)}, ' + WR.summary(bins)


def draw_corridor(ax, wall, max_range):
    """Габаритный коридор блока 2 поверх вида сверху -> строка для заголовка.

    Отвечает на вопрос, которого не видно по одной красной линии: упирается
    ли коридор в стену или ведётся предсказанием. Цвета те же, что в draw.py
    (одна схема на весь пакет): зелёный — бин подтверждён измерением стен,
    жёлтый — "не наблюдается", а НЕ "свободно".

    left/right/half_* в walls.trace — вынос ОТ ВЕДОМОЙ ОСИ БИНА, поэтому в
    поперечную координату картинки они возвращаются прибавлением wall['axis'].
    """
    from guard.draw import OK, HOLE, WALL

    sel = np.array([s != 'stop' for s in wall['status']])
    if not sel.any():
        return 'коридора нет'
    f = wall['fwd_mid'][sel]
    k = f <= max_range
    f, sel_idx = f[k], np.flatnonzero(sel)[k]
    axc = wall['axis'][sel_idx]
    st = wall['status'][sel_idx]
    ok = st == 'ok'

    # Кромки стен, в которые упирается коридор: пунктиром, чтобы не путать с
    # краем габарита — это разные вещи (габарит может быть уже стены).
    for key, lbl in (('left', 'кромка стены (коридор)'), ('right', None)):
        ax.plot(f, axc + wall[key][sel_idx], color=WALL, lw=1.0, ls='--',
                zorder=3, label=lbl)

    lo = axc - wall['half_left'][sel_idx]
    hi = axc + wall['half_right'][sel_idx]
    shown = set()
    for n in range(len(f) - 1):
        col = OK if (ok[n] and ok[n + 1]) else HOLE
        key = 'ok' if col is OK else 'hole'
        lbl = None
        if key not in shown:
            lbl = ('габарит подтверждён' if col is OK
                   else 'не наблюдается (не "свободно")')
            shown.add(key)
        ax.fill_between(f[n:n + 2], lo[n:n + 2], hi[n:n + 2], color=col,
                        alpha=0.22, lw=0, zorder=2, label=lbl)
        ax.plot(f[n:n + 2], lo[n:n + 2], color=col, lw=1.3, zorder=3)
        ax.plot(f[n:n + 2], hi[n:n + 2], color=col, lw=1.3, zorder=3)

    n_ok = int(ok.sum())
    return (f'коридор: {n_ok}/{len(f)} бинов измерено, '
            f'подтверждён до {wall["reach"]:.0f} м')


def draw_obstacles(ax, wall, fwd, lat, max_range):
    """Находки блока 3 (obstacles.detect) поверх вида сверху -> строка заголовка.

    Отвечает на вопрос, который коридор не решает: коридор показывает, ДОКУДА
    видно, а не ЧТО внутри. Находка рисуется тремя вещами сразу, потому что
    каждая отвечает на своё: точки находки (подсветка — видно, из чего она
    собрана и не пыль ли это), рамка по её габаритам (ширина/высота меряются
    именно так) и подпись с дальностью (её называют в тревоге).

    Красным — находка В ГАБАРИТЕ (опасна), серым — вне коридора (объект есть,
    но поезд мимо него проходит). Пустая рамка (пунктир) — `confirmed=False`:
    бин не измерен ('hole'), находка настоящая, но за неизвестным участком.

    fwd/lat — точки ТОГО ЖЕ кадра, из которых считался блок 3: point_idx в
    находке — индексы в этих массивах (без накопления, см. draw).
    """
    ob = wall.get('obstacles')
    if not ob:
        return ''
    # Считаем и рисуем только то, что попало в кадр картинки: находка за
    # правым краем в счётчике заголовка сбивает с толку — её на картинке нет.
    hits = [h for h in ob[0] if h['d'] <= max_range]
    shown = set()
    for o in hits:
        danger = o['in_gauge']
        col = 'tab:red' if danger else '0.45'
        # Поперечная координата — в системе лидара (lat_abs), как и облако:
        # 'lat' в находке отсчитан от оси габарита и на картинку не кладётся.
        xc = o['lat_abs']
        lo, hi = xc - o['width'] / 2, xc + o['width'] / 2

        pi = o.get('point_idx')
        if pi is not None and len(pi):
            pi = np.asarray(pi, dtype=int)
            pi = pi[pi < len(fwd)]
            lbl = None if 'pts' in shown else 'точки находки'
            shown.add('pts')
            ax.scatter(fwd[pi], lat[pi], s=6, c=col, linewidths=0, zorder=11,
                       label=lbl)

        key = ('g' if danger else 'o') + ('c' if o['confirmed'] else 'h')
        lbl = None
        if key not in shown:
            lbl = ('ПРЕПЯТСТВИЕ в габарите' if danger else 'объект вне габарита')
            if not o['confirmed']:
                lbl += ' (бин не измерен)'
            shown.add(key)
        r = plt_rect(o['d'], lo, hi, col, o['confirmed'])
        ax.add_patch(r)
        if lbl:
            r.set_label(lbl)
        if danger:
            ax.annotate(f'{o["d"]:.0f} м  h={o["height"]:.2f}',
                        xy=(o['d'], hi), xytext=(0, 4),
                        textcoords='offset points', ha='center', va='bottom',
                        fontsize=7, color=col, zorder=12)

    ing = [o for o in hits if o['in_gauge']]
    if not hits:
        return 'препятствий нет'
    near = min(ing, key=lambda h: h['d']) if ing else None
    return ('препятствия: {} (в габарите {}){}'.format(
        len(hits), len(ing),
        f', ближняя {near["d"]:.0f} м' if near is not None else ''))


def plt_rect(d, lo, hi, col, confirmed):
    """Рамка находки на виде сверху: по дальности — окно поиска OBST_WIN,
    поперёк — реальная ширина находки. Сплошная = бин измерен."""
    from matplotlib.patches import Rectangle
    from guard.obstacles import OBST_WIN
    return Rectangle((d - OBST_WIN / 2, lo), OBST_WIN, hi - lo,
                     fill=False, edgecolor=col, lw=1.6,
                     ls='-' if confirmed else '--', zorder=11)


def draw_rails(ax, axis):
    """Две найденные нити и ось по ним (блок 1п) — что именно нашлось.

    Рисуется только при --polyline: у прямой оси нитей нет вовсе. Нити нужны
    рядом с осью, потому что «ось уехала» без них не читается — видно ли, что
    съехала одна нить, или полилиния потеряла обе.
    """
    if not axis.get('poly') or axis.get('rails') is None:
        return ''
    (fl, ll), (fr, lr) = axis['rails']
    ax.plot(fl, ll, '-', color='tab:brown', lw=1.0, alpha=0.9, zorder=6,
            label='нить рельса (полилиния)')
    ax.plot(fr, lr, '-', color='tab:brown', lw=1.0, alpha=0.9, zorder=6)
    ax.plot(axis['knots_f'], axis['knots_l'], '-', color='magenta', lw=1.8,
            alpha=0.95, zorder=7, label='ось по рельсам (ломаная)')
    return (f'полилиния: узлов {axis["n_obs"]}, до '
            f'{axis["knots_f"].max():.0f} м, колея {axis["gauge"]:.3f} м')


def draw(ax, frames, tm, poses, i, max_range, raw, n_accum=0, walls=False,
         ylim=None, corridor=False, polyline=False, obstacles=False):
    from guard.geom import to_frame
    from guard.run import analyze
    from guard.walls import GAUGE_H_LO, GAUGE_H_HI
    from guard import axis_wall as AW

    pts = frames[i][0]
    axis, w = analyze(pts, polyline=polyline,
                      bag_key=getattr(frames, 'db_path', None), frame_index=i)
    tr = axis['far']
    if tr is None:
        ax.set_title(f'кадр {i}: ведение выключено')
        ax.set_xlim(0, max_range)
        if ylim is not None:
            ax.set_ylim(-ylim, ylim)
        ax.grid(alpha=0.3)
        return

    # Облако — только то, что в высоте габарита: пол и свод сверху мешают.
    fwd, lat, up = to_frame(pts)
    h = up - w['floor']
    m = (fwd > 0) & (fwd < max_range) & (h > GAUGE_H_LO) & (h < GAUGE_H_HI)

    # Накопленные точки соседних кадров — под точками этого кадра и светлее,
    # чтобы видно было, что именно добавилось.
    det = (fwd, lat, up)          # облако, по которому ищутся стены
    if n_accum and poses is not None:
        ac = accum_cloud(frames, poses, i, n_accum)
        if ac is not None:
            af, al, au = ac
            ah = au - w['floor']
            ma = ((af > 0) & (af < max_range)
                  & (ah > GAUGE_H_LO) & (ah < GAUGE_H_HI))
            ax.scatter(af[ma], al[ma], s=0.5, c='0.90', linewidths=0,
                       zorder=0, label=f'точки {n_accum} прошлых кадров')
            # Стены ищутся и по накопленному тоже: дыры между кольцами
            # закрыты, и на дальности в слое становится не 4 точки, а десятки.
            det = tuple(np.concatenate([a, b])
                        for a, b in zip(det, (af, al, au)))

    ax.scatter(fwd[m], lat[m], s=0.5, c='0.82', linewidths=0, zorder=1,
               label='точки кадра')

    # Эталон.
    rf, rx = ref_axis(tm, poses, i)
    if rf is not None:
        k = rf <= max_range
        ax.plot(rf[k], rx[k], color='tab:green', lw=2.6, zorder=5,
                label='путь по карте (эталон)')

    sel = np.array([s != 'stop' for s in tr['status']])
    f, la, st = tr['fwd_mid'][sel], tr['lat'][sel], tr['status'][sel]

    # Прямая ось блока 1 — что было бы без ведения стеной.
    ax.plot(f, AW.axis_at(axis, f), color='tab:orange', lw=1.4, ls=':',
            zorder=4, label='прямая ось (блок 1)')

    # Сырая ось, если просили сравнить: тот же расчёт при SMOOTH_WIN = 0.
    if raw:
        keep = AW.SMOOTH_WIN
        AW.SMOOTH_WIN = 0
        try:
            tr0 = analyze(pts)[0]['far']
        finally:
            AW.SMOOTH_WIN = keep
        s0 = np.array([s != 'stop' for s in tr0['status']])
        ax.plot(tr0['fwd_mid'][s0], tr0['lat'][s0], color='tab:red', lw=1.0,
                ls='--', alpha=0.55, zorder=5, label='ось без сглаживания')

    # Ось блока 1б (axis_wall) — ТОЛЬКО без --walls. С --walls на картинке
    # была вторая ось, нарисованная другим блоком по другим правилам: две
    # линии, две легенды («ось: ведёт стена» красными шарами от axis_wall и
    # оранжевым сегментом от wall_rules), и понять, какая чья, было нельзя.
    # Сравнивать надо с ЭТАЛОНОМ, а не с предыдущей версией себя, поэтому
    # старая ось уходит целиком, вместе со своими кромками и reach.
    if not walls:
        ax.plot(f, la, color='tab:red', lw=1.2, alpha=0.6, zorder=6)
        for name, col, lbl in STATUS_STYLE:
            k = st == name
            if k.any():
                ax.plot(f[k], la[k], 'o', color=col, ms=5, zorder=7, label=lbl)

        # Кромка, за которую зацепилось ведение.
        for key, lbl in (('wall_left', 'кромка стены'), ('wall_right', None)):
            ax.plot(f, tr[key][sel], '.', color='k', ms=3.5, zorder=6,
                    label=lbl)

        # Докуда ось подтверждена измерением.
        ax.axvline(tr['reach'], color='0.35', ls='--', lw=0.9, zorder=3)
        ax.annotate(f'reach {tr["reach"]:.0f} м', (tr['reach'], 1), xycoords=(
            'data', 'axes fraction'), fontsize=8, va='top', ha='left',
            xytext=(3, -3), textcoords='offset points', color='0.3')

    # Правилам ведения нужны ось от рельсов (истина ближней зоны) и отступ,
    # измеренный в ней же: без них LEAD берёт нижнюю границу физики и честно
    # помечает бины как неизмеренные.
    wtxt = draw_walls(ax, det[0], det[1], det[2], w['floor'], max_range,
                      axis=axis, off=tr['offset']) if walls else None
    ctxt = draw_corridor(ax, w, max_range) if corridor else None
    # Точки — этого кадра: блок 3 считался по нему (analyze выше),
    # и point_idx индексирует именно fwd/lat, а не накопленное облако.
    otxt = draw_obstacles(ax, w, fwd, lat, max_range) if obstacles else ''
    ptxt = draw_rails(ax, axis) if polyline else ''

    # Заголовок: отступ, измеренный в ближней зоне, нужен обоим блокам —
    # он и есть вход правил ведения. Остальное (бинов стеной, переход) —
    # про ось axis_wall, и с --walls её на картинке нет.
    nw = int(tr['n_wall'])
    ax.set_title(f'кадр {i}   отступ L={tr["offset"][-1]:+.2f} '
                 f'R={tr["offset"][+1]:+.2f} м'
                 + ('' if walls else
                    f'   бинов стеной {nw}'
                    + (f'   переход с {tr["led_from"]:.0f} м'
                       if not np.isnan(tr['led_from']) else '   стена не ведёт'))
                 + (f'   |   {ctxt}' if ctxt else '')
                 + (f'   |   {otxt}' if otxt else '')
                 + (f'   |   {wtxt}' if wtxt else '')
                 + (f'   |   {ptxt}' if ptxt else ''),
                 fontsize=10)
    ax.set_xlabel('дальность вперёд, м')
    ax.set_ylabel('поперёк, м')
    ax.set_xlim(0, max_range)
    if ylim is not None:
        ax.set_ylim(-ylim, ylim)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc='upper left', ncol=3, framealpha=0.9)


def draw_profile(ax, frames, tm, poses, i, at):
    """Профиль стены по ВЫСОТЕ в одном бине: видно, за что цепляется кромка.

    Тоннель круглый, и вынос стены зависит от высоты. Эта панель отвечает на
    вопрос "почему кромка села внутрь": показывает, где в этом бине точки на
    каждой высоте и какой слой выбрало ведение.
    """
    from guard.geom import to_frame
    from guard.run import analyze
    from guard import walls as W

    pts = frames[i][0]
    axis, w = analyze(pts)
    tr = axis['far']
    fwd, lat, up = to_frame(pts)
    h = up - w['floor']

    rf, rx = ref_axis(tm, poses, i)
    if rf is not None and at <= rf.max():
        a = float(np.interp(at, rf, rx))
    elif tr is not None:
        a = float(np.interp(at, tr['fwd_mid'], tr['lat']))
    else:
        a = 0.0

    m = (fwd >= at - W.BIN / 2) & (fwd < at + W.BIN / 2)
    ax.scatter(lat[m] - a, h[m], s=3, c='0.6', linewidths=0)

    # Кромка по каждому слою: p10 выноса, как её видит _edge.
    for z in np.arange(W.H_LO, W.H_HI - 1e-9, W.H_STEP):
        k = m & (h >= z) & (h < z + W.H_STEP)
        if k.sum() < W.MIN_PTS:
            continue
        d = lat[k] - a
        for side in (-1, +1):
            s = d[d * side > 0] * side
            if len(s) < W.MIN_PTS:
                continue
            v = float(np.percentile(s, W.WALL_Q)) * side
            ax.plot([v], [z + W.H_STEP / 2], 'o', color='tab:red', ms=6,
                    zorder=5)

    if tr is not None:
        lay = tr.get('layer')
        if lay is not None:
            ax.axhspan(lay, lay + W.H_STEP, color='tab:green', alpha=0.18,
                       zorder=0,
                       label=f'слой ведения {lay:.1f}-{lay + W.H_STEP:.1f} м')
    ax.axhspan(W.GAUGE_H_LO, W.GAUGE_H_HI, color='tab:blue', alpha=0.07,
               zorder=0, label='высота габарита')
    ax.axvline(0, color='tab:green', lw=1.5, zorder=4)
    ax.plot([], [], 'o', color='tab:red', ms=6, label='кромка слоя (p10)')
    ax.set_title(f'кадр {i}: профиль поперёк на {at:.0f} м '
                 f'(0 = путь по карте)', fontsize=10)
    ax.set_xlabel('поперёк от пути, м')
    ax.set_ylabel('высота над полом, м')
    ax.set_xlim(-8, 8)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc='upper left')


def make_movie(a, frames, tm, poses, plt):
    """Фильм: тот же draw() кадр за кадром, пределы осей зафиксированы.

    Окно --accum скользит вместе с кадром: на кадре i подкладываются кадры
    i-n..i-1, перенесённые одометрией. Первые n кадров бэга накопят меньше.
    """
    from matplotlib import animation

    parts = (a.movie_range.split(':') + ['', '', ''])[:3]
    beg = int(parts[0]) if parts[0] else 0
    end = int(parts[1]) if parts[1] else len(frames)
    step = int(parts[2]) if parts[2] else 1
    idx = list(range(max(0, beg), min(end, len(frames)), max(1, step)))
    if not idx:
        sys.exit('--movie-range не выбрал ни одного кадра')

    ylim = a.lat_range or a.range / 4.0

    if a.movie.endswith('.gif'):
        writer = animation.PillowWriter(fps=a.fps)
    else:
        writer = animation.FFMpegWriter(fps=a.fps, bitrate=4000)

    fig, ax = plt.subplots(figsize=(14, 6))
    name = os.path.basename(a.bag.rstrip('/'))
    with writer.saving(fig, a.movie, dpi=110):
        for n, i in enumerate(idx):
            ax.clear()
            draw(ax, frames, tm, poses, i, a.range, a.raw, a.accum, a.walls,
                 ylim=ylim, corridor=a.corridor, polyline=a.polyline,
                 obstacles=a.obstacles)
            # Номер кадра в suptitle, а не только в длинном заголовке оси:
            # в фильме глаз должен находить его в одном и том же месте, чтобы
            # плохой кадр можно было назвать и открыть отдельной картинкой.
            fig.suptitle(f'{name}   —   кадр {i}   ({n + 1}/{len(idx)})',
                         fontsize=12, y=0.998)
            fig.tight_layout()
            writer.grab_frame()
            if n % 20 == 0:
                print(f'\r{n + 1}/{len(idx)} кадров', end='', flush=True)
    plt.close(fig)
    print(f'\rготово, {len(idx)} кадров: {a.movie}')


def print_metrics(a, frames):
    """Замер по всему бэгу рядом с картинкой одного кадра.

    Заголовок картинки отвечает «сколько находок в ЭТОМ кадре», а вопрос
    «сколько препятствий в бэге» — другой: объект живёт десятки кадров, и
    сумма покадровых находок посчитала бы его десятки раз. Считает
    metrics.obstacle_count теми же блоками, с тем же --range и --accum, что
    у картинки, — чтобы числа отвечали именно ей. --accum здесь и там —
    агрегация ТОЧЕК (в run.py это --accum-pts, а не --accum: под тем именем
    там копится коридор по бинам, и препятствий он не несёт).
    """
    from guard.metrics.obstacle_count import measure, report
    bag_dir = a.bag if os.path.isdir(a.bag) else os.path.join(ROOT, a.bag)
    m = measure(frames, bag_dir, max_range=a.range, step=a.metrics,
                n_accum=a.accum, db=getattr(frames, 'db_path', None))
    print('\n'.join(report(m, a.bag, episodes=a.episodes)))
    print()


def main():
    ap = argparse.ArgumentParser(
        description='Ось вдаль против карты, вид сверху',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    ap.add_argument('bag', help='имя бэга в for_hackathon или путь')
    ap.add_argument('frames', nargs='?', default='',
                    help='номера кадров через запятую, напр. 108,123,240')
    ap.add_argument('--worst', type=int, metavar='N',
                    help='вместо списка — N худших кадров по ошибке')
    ap.add_argument('--step', type=int, default=3,
                    help='шаг перебора при --worst (по умолчанию 3)')
    ap.add_argument('-o', '--out', metavar='PNG',
                    help='сохранить в файл вместо показа окна')
    ap.add_argument('--range', type=float, default=100.0,
                    help='дальность на картинке, м (по умолчанию 100)')
    ap.add_argument('--accum', nargs='?', type=int, const=6, default=0,
                    metavar='N',
                    help='подложить точки N прошлых кадров, перенесённые '
                         'одометрией (по умолчанию 6): стена на картинке '
                         'становится плотной, дыры между кольцами закрыты')
    ap.add_argument('--walls', action='store_true',
                    help='нарисовать стены, найденные блоком check_wall '
                         '(детекция стен, про ось не знает)')
    ap.add_argument('--corridor', '--corridors', dest='corridor',
                    action='store_true',
                    help='нарисовать габаритный коридор блока 2 (walls.trace): '
                         'полоса по бинам, зелёная там где край упёрся в '
                         'измеренную стену, жёлтая где бин ведётся '
                         'предсказанием. Видно, входит ли коридор в стену на '
                         'повороте — по одной линии оси это незаметно')
    ap.add_argument('--obstacles', action='store_true',
                    help='нарисовать находки блока 3 (obstacles.detect): '
                         'точки находки, рамка по её габаритам и дальность. '
                         'Красное — в габарите (тревога), серое — объект '
                         'мимо коридора; пунктирная рамка — бин не измерен')
    ap.add_argument('--metrics', nargs='?', type=int, const=1, default=0,
                    metavar='STEP',
                    help='рядом с картинкой напечатать замер по ВСЕМУ бэгу: '
                         'сколько препятствий выдаст блок 3 (metrics.'
                         'obstacle_count). Необязательный аргумент — шаг по '
                         'кадрам (по умолчанию 1, весь бэг)')
    ap.add_argument('--episodes', action='store_true',
                    help='с --metrics: таблица эпизодов, строка на объект')
    ap.add_argument('--movie', metavar='OUT',
                    help='снять фильм (вид сверху, кадр за кадром) в .mp4 '
                         'или .gif вместо картинки; кадры берутся из '
                         '--movie-range')
    ap.add_argument('--movie-range', metavar='A:B:S', default='::1',
                    help='какие кадры в фильм: начало:конец:шаг '
                         '(по умолчанию весь бэг с шагом 1)')
    ap.add_argument('--fps', type=int, default=10,
                    help='кадров в секунду в фильме (по умолчанию 10)')
    ap.add_argument('--lat-range', type=float, default=0.0,
                    metavar='M',
                    help='зафиксировать полуширину по «поперёк», м; в фильме '
                         'по умолчанию max_range/4, иначе авто')
    # Ломаная включена ПО УМОЛЧАНИЮ: прямая ось блока 1 стоит на горстке
    # наблюдений (roundT_doubleT f107/f121 — n_obs=4, колея 1.36 м при
    # номинале 1.52, то есть зацеплена чужая пара) и съезжает вбок, а на
    # повороте и не может быть верной. Полилиния на тех же кадрах даёт 64-92
    # узла и колею 1.515-1.523 м. --no-polyline возвращает прежнюю прямую.
    ap.add_argument('--polyline', action=argparse.BooleanOptionalAction,
                    default=True,
                    help='ось блока 1 — ЛОМАНАЯ по двум нитям полилинейной '
                         'детекции (как --polyline в visualize_bag.py) вместо '
                         'прямой; обе нити и ось по ним рисуются на картинке. '
                         'Включено по умолчанию, --no-polyline выключает')
    ap.add_argument('--raw', action='store_true',
                    help='нарисовать рядом ось без сглаживания (SMOOTH_WIN=0)')
    ap.add_argument('--profile', type=float, metavar='D',
                    help='добавить профиль стены по высоте на дальности D '
                         '(вторая колонка картинки)')
    a = ap.parse_args()

    import matplotlib
    if a.out:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    frames, tm, poses = load(a.bag)

    # Замер по всему бэгу печатается ДО картинки: она блокирует поток в
    # plt.show(), и числа после неё пользователь увидел бы только закрыв окно.
    if a.metrics:
        print_metrics(a, frames)

    if a.movie:
        make_movie(a, frames, tm, poses, plt)
        return

    if a.worst:
        if tm is None or poses is None:
            sys.exit('--worst требует track_map.npz и trajectory.npz')
        idx = worst_frames(frames, tm, poses, a.worst, a.step)
    elif a.frames:
        idx = [int(x) for x in a.frames.split(',') if x.strip()]
    else:
        idx = [len(frames) // 2]
    idx = [i for i in idx if 0 <= i < len(frames)]
    if not idx:
        sys.exit('нет кадров для показа')

    ncol = 2 if a.profile is not None else 1
    fig, axes = plt.subplots(len(idx), ncol, squeeze=False,
                             figsize=(14 if ncol == 1 else 19,
                                      4.3 * len(idx)),
                             gridspec_kw=({'width_ratios': [2.4, 1]}
                                          if ncol == 2 else None))
    for r, (ax, i) in enumerate(zip(axes[:, 0], idx)):
        draw(ax, frames, tm, poses, i, a.range, a.raw, a.accum, a.walls,
             corridor=a.corridor, polyline=a.polyline,
             obstacles=a.obstacles)
        if a.profile is not None:
            draw_profile(axes[r, 1], frames, tm, poses, i, a.profile)
    fig.suptitle(os.path.basename(a.bag.rstrip('/')), fontsize=11, y=0.998)
    fig.tight_layout()

    if a.out:
        fig.savefig(a.out, dpi=115)
        print(f'сохранено: {a.out}')
    else:
        plt.show()


if __name__ == '__main__':
    main()
