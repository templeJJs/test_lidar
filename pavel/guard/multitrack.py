"""Блок: многопутность и стрелки (вторая ветка пути в кадре).

Зачем отдельный блок. track.straight_axis сознательно выбирает ОДНУ ветку —
ближнюю к лидару (`_best_branch`), остальные отбрасываются. На прямой это
правильно: ось одна. Но на стрелке и на двухпутке в кадре ДВА пути, и до
перевода стрелки из облака не определить, куда поедет поезд. Правильное
поведение — вести габарит по ВСЕМ правдоподобным веткам, а не угадывать одну.
Этот блок находит все ветки и отдаёт их наружу; выбором "нашей" он не
занимается.

Четыре замера задали конструкцию.

1. ШИРОКОЕ ОКНО ТОНИТ В МУСОРЕ, ПИК НАДО ПРОВЕРЯТЬ НА "РЕЛЬСОВОСТЬ".
   track._peaks смотрит в полосе +-2.5 м от лидара — второй путь (межосевое
   ~4 м, замер roundT_doubleT f225: ветка +4.06 м) туда не попадает. Но
   простое расширение окна даёт десятки ложных веток: стены и край платформы
   пересекают полосу высот головки и дают пики, из которых со своим рельсом
   складывается "колея" (замер squareT_switch f400, срез 8-16 м: пики
   -2.28/+2.28 с колеёй 1.44-1.56 к своим рельсам -0.82/+0.72 — 450 кадров
   подряд!). Разделяет СТРУКТУРА ПО ВЫСОТЕ: рельс — линия, почти все точки
   которой лежат в полосе головки; стена — вертикаль, в полосе головки у неё
   единицы процентов. Замер доли точек полосы в столбце +-0.15 м вокруг пика
   (срез 8-16 м, squareT_switch f0/f50/f100/f400, doubleT_platform f100):
     свои рельсы        0.66-0.90
     стены платформы    0.03-0.11
     бордюр/лоток       0.11-0.16
   Порог RAIL_FRAC=0.30 лежит между группами с двукратным запасом в обе
   стороны.

2. КОЛЕЯ ОДНА, А ВЕТОК-ДУБЛЕЙ МНОГО. Из соседних пиков одного и того же
   рельса складываются пары с почти одинаковым центром и разной колеёй
   (замер squareT_switch f125: четыре ветки с центрами -0.13..-0.06). Дубли
   сливаются по центру: два РАЗНЫХ пути не могут иметь оси ближе
   GAUGE_NOM/2 — при меньшем шаге внутренние рельсы наложились бы, то есть
   это один путь, переизмеренный дважды (аналитика из колеи, не подгонка).

3. ПЕРЕКРЁСТНАЯ ПАРА — НЕ ПУТЬ. Два реальных пути с осями c1 < c2 и колеёй
   g имеют рельсы c1+-g/2 и c2+-g/2; пара из соседних внутренних рельсов даёт
   центр (c1+c2)/2 и колею c2-c1-g, которая при межосевом 2g+-tol (на веере
   стрелки разнос проходит через 2.7-3.1 м) попадает в окно колеи. Такая
   ветка-фантом ровно посередине двух настоящих: она отбрасывается, если
   |(c2-c1) - g_фантома - GAUGE_NOM| <= GAUGE_TOL — это аналитическое условие
   "колея фантома = межосевое минус колея", новых чисел не вводит.

4. ПУТЬ ЗА СТЕНОЙ — НЕ МАРШРУТ. Соседний путь виден сквозь проёмы и сбойки
   и даёт честную пару с колеёй (замер doubleT_platform f10: ветка +4.06 м,
   колея 1.60, шесть срезов 9-28 м, между путями открыто — стены до 28 м
   нет). Геометрия кадра там неотличима от настоящей двухпутки
   (roundT_doubleT f230: та же ветка +4.06, та же открытость), и высота
   рельсов, и их доля в полосе — тоже (замеры: рельсы 0.39-0.48 м над полом
   в обоих бэгах). Различает КОНЕЦ видимости ветки: соседний путь обрывается
   там, где проём закрывается стеной между путями (f10: стена +2.2..+2.5 м с
   28 м, check_wall.detect), а свой второй путь продолжается за плечо или
   уходит к своей дальней стене (f230: стена +6.25 м — за веткой; стрелка
   f820: стена +2.45 м — дальше оси ветки +1.84 в её плече). Поэтому ветка
   помечается blocked, если стена кадра проходит между своей осью и осью
   ветки в её плече или сразу за ним (см. _wall_between).

Что выдаётся — список веток {axis, gauge, n_obs, span, own, blocked,
confidence} и флаг diverged. Коридор ветки считается штатным walls.trace
с осью ветки (та же геометрия, другая ось) — функция branch_corridor.
"""

import argparse

import numpy as np

from .geom import to_frame, floor_level
from . import track as T
from . import walls as W

# --- окно поиска чужих веток ----------------------------------------------
BRANCH_LAT_WIN = 6.0     # полуширина поиска пиков, м. Шире LAT_WIN блока 1
                         # (2.5 м — там только СВОЙ путь). Низ задан замером
                         # межосевого двухпутки: roundT_doubleT f225 ветка
                         # +4.06 м, внешний рельс +4.8 м; запас до 6.0 м
                         # покрывает и раскачку оси. Дальше не расширяем:
                         # чем шире окно, тем больше стен попадает в профиль
                         # (см. замер 1), а межосевое метро больше ~5 м
                         # в зоне детекции головок (30 м) не различить
RAIL_FRAC = 0.30         # доля точек столбца пика, лежащих в полосе высот
                         # головки: рельс >= 0.66, стена/бордюр <= 0.16
                         # (замер 1 выше, 8 кадров трёх бэгов)
PEAK_STRIP = 0.15        # полуширина столбца проверки вокруг пика, м:
                         # 3 бина профиля (LAT_BIN=0.05); уже — точки соседних
                         # рельсов/стен не попадают, шире — разбавляют долю

# --- слияние и отбор веток -------------------------------------------------
SAME_TRACK = T.GAUGE_NOM / 2   # оси двух РАЗНЫХ путей ближе быть не могут
                               # (замер 2): иначе внутренние рельсы накладываются
MIN_SEP = W.GAUGE_HALF         # ветка — отдельный путь, если её ось за
                               # пределами полугабарита своего пути (1.70 м,
                               # норматив). Ближе — или тот же путь с шумом
                               # переизмерения, или ветка в зоне крестовины,
                               # где габаритные коридоры путей всё равно
                               # слиты и раздельное ведение бессмысленно.
                               # Замер-разделитель на прямых: плита пути
                               # roundT_doubleT f0-f140 рождает ложные ветки
                               # со сносом оси до 1.61 м от своей (f30:
                               # -1.35 при своей +0.26), истинные вторые
                               # пути — 1.85+ м (f140: -1.85; f230: +4.06;
                               # стрелка f820: +1.97)


def rail_peaks(lat_s, up_s, floor, lat_win=BRANCH_LAT_WIN):
    """Пики профиля в полосе головки, отфильтрованные по "рельсовости".

    Отличие от track._peaks — параметризованное окно и проверка структуры
    по высоте (замер 1): пик принимается, только если доля точек его столбца
    в полосе головки не ниже RAIL_FRAC. Столбец считается по ВСЕМ высотам
    среза — поэтому у стены (вертикаль на весь кадр) доля мала, а у рельса
    (линия) близка к 1.
    """
    in_band = (up_s - floor >= T.H_LO) & (up_s - floor < T.H_HI)
    m = in_band & (np.abs(lat_s) < lat_win)
    if m.sum() < T.MIN_PTS_SLICE:
        return np.zeros(0)
    v = lat_s[m]
    edges = np.arange(-lat_win, lat_win + T.LAT_BIN, T.LAT_BIN)
    cnt, _ = np.histogram(v, bins=edges)
    thr = max(T.PEAK_MIN_CNT, cnt.max() * T.PEAK_MIN_FRAC)
    out = []
    i = 1
    while i < len(cnt) - 1:
        if cnt[i] >= thr and cnt[i] >= cnt[i - 1] and cnt[i] >= cnt[i + 1]:
            j = i
            while j + 1 < len(cnt) - 1 and cnt[j + 1] == cnt[i]:
                j += 1
            sel = v[(v >= edges[i - 1]) & (v < edges[j + 2])]
            if len(sel):
                c = float(sel.mean())
                strip = np.abs(lat_s - c) < PEAK_STRIP
                n_all = int(strip.sum())
                if n_all and in_band[strip].sum() / n_all >= RAIL_FRAC:
                    out.append(c)
            i = j + 1
        else:
            i += 1
    return np.array(sorted(out))


def link_branches(cands):
    """ВСЕ непрерывные цепочки пар по срезам (обобщение track._best_branch).

    Логика связывания и фильтр непрерывности — ровно те же, что в блоке 1
    (жадное связывание по свежести, прогон не короче MIN_BRANCH); отличие
    одно: вместо выбора одной ближней ветки возвращаются все прошедшие
    фильтр. Каждая ветка — список (fwd, центр, колея).
    """
    branches = []
    for fwd_s, pairs in cands:
        links = []
        for pi, (c, g) in enumerate(pairs):
            for br in branches:
                if br[-1][0] == fwd_s:
                    continue
                d = abs(br[-1][1] - c)
                if d < T.LINK_TOL:
                    links.append(((fwd_s - br[-1][0], d), pi, br))
        links.sort(key=lambda t: t[0])
        used_p, used_b = set(), set()
        for _, pi, br in links:
            if pi in used_p or id(br) in used_b:
                continue
            br.append((fwd_s, pairs[pi][0], pairs[pi][1]))
            used_p.add(pi)
            used_b.add(id(br))
        for pi, (c, g) in enumerate(pairs):
            if pi not in used_p:
                branches.append([(fwd_s, c, g)])

    def shoulder(b):
        best, n = 1, 1
        for k in range(1, len(b)):
            if b[k][0] - b[k - 1][0] > T.SLICE_LEN * 1.5:
                n = 1
            else:
                n += 1
                best = max(best, n)
        return best

    return [b for b in branches
            if len(b) >= T.MIN_BRANCH and shoulder(b) >= T.MIN_BRANCH]


def _branch_view(b):
    """Цепочка -> компактное описание (ось, колея, опоры)."""
    obs_f = np.array([f for f, _, _ in b])
    obs_c = np.array([c for _, c, _ in b])
    obs_g = np.array([g for _, _, g in b])
    # Наклон НЕ клампится, в отличие от своей оси блока 1: уходящая ветка
    # стрелки по определению имеет наклон (замер f820 squareT_switch: веер
    # до 0.2 м/м), кламп стёр бы именно то, ради чего ветка ищется.
    slope, lat0 = np.polyfit(obs_f, obs_c, 1)
    return dict(axis=dict(lat0=float(lat0), slope=float(slope)),
                center=float(np.median(obs_c)),
                gauge=float(np.median(obs_g)),
                n_obs=len(b),
                span=(float(obs_f.min()), float(obs_f.max())),
                obs_fwd=obs_f, obs_lat=obs_c)


def merge_duplicates(branches):
    """Сливает цепочки одного пути (замер 2) -> список _branch_view.

    Кластер по центру с порогом SAME_TRACK; представитель кластера —
    цепочка с наибольшим числом звеньев (она же самая длинная по плечу в
    замерах: дубли рождаются обрывами связывания и короче).
    """
    views = sorted((_branch_view(b) for b in branches),
                   key=lambda v: -v['n_obs'])
    clusters = []
    for v in views:
        for cl in clusters:
            if abs(v['center'] - cl[0]['center']) < SAME_TRACK:
                cl.append(v)
                break
        else:
            clusters.append([v])
    return [cl[0] for cl in clusters]


def drop_phantoms(branches):
    """Отбрасывает перекрёстные пары двух реальных веток (замер 3)."""
    keep = []
    for i, b in enumerate(branches):
        phantom = False
        for j, a in enumerate(branches):
            if j == i:
                continue
            for k, c in enumerate(branches):
                if k == i or k == j:
                    continue
                if a['center'] < b['center'] < c['center'] and abs(
                        (c['center'] - a['center']) - b['gauge']
                        - T.GAUGE_NOM) <= T.GAUGE_TOL:
                    phantom = True
        if not phantom:
            keep.append(b)
    return keep


def _wall_between(walls, axis, branch):
    """Стоит ли стена между своим путём и веткой (замер 4).

    walls — ВСЕ стены кадра из check_wall.detect (абсолютные координаты, ось
    им не нужна). Ветка за стеной — не маршрут: это соседний путь, видимый
    сквозь проём/сбойку. Признак — стена, которая
      1) лежит между своей осью и осью ветки (own_c < |w| < |c_B| на стороне
         ветки), и
      2) делает это у ДАЛЬНЕГО КОНЦА ветки (в плече или в пределах одного
         бина за ним): именно она заслонила путь, и видимость ветки оборвалась.
    Пункт 2 — разделитель со стрелкой: на squareT_switch f820 уходящая ветка
    (+1.84 м, плечо 9-17 м) проходит МИМО торца платформы (стена +2.45 м —
    ДАЛЬШЕ оси ветки в её плече), а на doubleT_platform f10 соседний путь
    (+4.06 м, плечо 9-28 м) обрывается ровно там, где с 28 м начинается стена
    +2.3 м между путями. Окно за концом — один бин: стена, заслонившая путь,
    начинается сразу за ним.
    """
    s = +1 if branch['center'] > T.axis_at(axis, 15.0) else -1
    lo, hi = branch['span']
    blocked, seen = 0, 0
    for w in walls:
        if w.side != s:
            continue
        a, b = max(lo, w.d_lo), min(hi + W.BIN, w.d_hi)
        if b - a < W.BIN:      # пересечения короче бина — не наблюдение
            continue
        for d in np.arange(a, b, W.BIN):
            wl = float(w.lat_at(d))
            c_b = branch['axis']['lat0'] + branch['axis']['slope'] * d
            own_d = float(T.axis_at(axis, d))
            seen += 1
            if s * wl > s * own_d and abs(wl) < abs(c_b):
                blocked += 1
    # Стена — преграда, если закрывает бОльшую часть проверенного окна:
    # одиночный бин может быть колонной в проёме. Половина — медианное
    # правило "больше чем нет", размеры окна заданы геометрией бина.
    return seen >= 2 and blocked * 2 > seen


def detect_divergence(fwd, lat, up, floor, axis=None, walls=None):
    """Все ветки пути в кадре и флаг расхождения.

    axis — ось своего пути из track.straight_axis (если None, считается
    здесь). walls — список стен кадра из check_wall.detect (если None,
    считается здесь; нужен для отсечения путей за стеной, см. замер 4).
    Своя ветка — ближайшая к своей оси; чужая ветка с осью дальше MIN_SEP
    от своей и без стены между — отдельный маршрут.

    Возвращает dict:
      branches — список: axis {lat0, slope}, gauge, n_obs, span, own,
                 blocked (за стеной), confidence
      diverged — есть ли в кадре вторая ветка-маршрут
    """
    if axis is None:
        axis = T.straight_axis(fwd, lat, up, floor)
    if walls is None:
        from .check_wall import detect as _detect_walls  # лениво: check_wall
        #                                                  тянет axis_wall
        walls = _detect_walls(fwd, lat, up, floor)
    cands = []
    for lo in np.arange(T.NEAR_FWD[0], T.NEAR_FWD[1], T.SLICE_LEN):
        hi = min(lo + T.SLICE_LEN, T.NEAR_FWD[1])
        m = (fwd >= lo) & (fwd < hi)
        if m.sum() < T.MIN_PTS_SLICE:
            continue
        pr = T._pairs(rail_peaks(lat[m], up[m], floor))
        if pr:
            cands.append(((lo + hi) / 2, pr))

    branches = drop_phantoms(merge_duplicates(link_branches(cands)))
    # Своя ось из блока 1: она измерена в узком окне и потому точнее, чем
    # ось ветки из широкого окна. Своя ветка — ОДНА, ближайшая к этой оси
    # (пометка по одиночеству: иначе дубли своего пути, не слитые из-за
    # разрыва связывания, оба попадали в own, а третья считалась чужой).
    own_c = T.axis_at(axis, 15.0)   # середина зоны измерения
    near = [b for b in branches if abs(b['center'] - own_c) < SAME_TRACK]
    own = min(near, key=lambda b: abs(b['center'] - own_c)) if near else None
    for b in branches:
        b['own'] = b is own
        b['blocked'] = (not b['own']
                        and abs(b['center'] - own_c) >= MIN_SEP
                        and _wall_between(walls, axis, b))
        # Доверие: доля срезов с парой (непрерывность) и близость колеи к
        # номиналу. Обе величины безразмерны и уже нормированы константами
        # блока 1, новых чисел не вводят.
        cont = b['n_obs'] / max(1.0, (b['span'][1] - b['span'][0])
                                / T.SLICE_LEN + 1)
        b['confidence'] = float(np.clip(
            cont * (1.0 - abs(b['gauge'] - T.GAUGE_NOM) / T.GAUGE_TOL),
            0.0, 1.0))
    others = [b for b in branches
              if not b['own'] and not b['blocked']
              and abs(b['center'] - own_c) >= MIN_SEP]
    return dict(branches=branches, diverged=bool(others), axis=axis,
                walls=walls)


def branch_corridor(fwd, lat, up, floor, branch, max_range=W.MAX_RANGE):
    """Габаритный коридор вдоль оси ветки — штатный walls.trace.

    Ось ветки подаётся как прямая (lat0/slope от цепочки пар): на плече
    измерения (9-28 м) этого хватает, дальше коридор ведётся предсказанием
    цепочки стен, как и в основном тракте.
    """
    return W.trace(fwd, lat, up, floor, branch['axis'], max_range=max_range)


def analyze_frame(pts, with_corridors=True):
    """Кадр -> (своя ось, результат detect_divergence, коридоры веток)."""
    fwd, lat, up = to_frame(pts)
    floor = floor_level(fwd, lat, up)
    det = detect_divergence(fwd, lat, up, floor)
    corridors = {}
    if with_corridors:
        for i, b in enumerate(det['branches']):
            corridors[i] = branch_corridor(fwd, lat, up, floor, b)
    return det['axis'], det, corridors


def _fmt_branch(i, b):
    a = b['axis']
    tag = 'СВОЯ' if b['own'] else ('ЗА СТЕНОЙ' if b['blocked'] else 'ВЕТКА')
    return (f"  [{i}] {tag:9s} "
            f"ось lat={a['lat0']:+.2f}{a['slope']:+.3f}*fwd  "
            f"центр {b['center']:+.2f} м  колея {b['gauge'] * 1000:.0f} мм  "
            f"срезов {b['n_obs']}  плечо {b['span'][0]:.0f}-{b['span'][1]:.0f} м  "
            f"conf {b['confidence']:.2f}")


def main():
    ap = argparse.ArgumentParser(
        description='Все ветки пути в кадре (многопутность, стрелки)')
    ap.add_argument('bag_dir')
    ap.add_argument('--frame', type=int, default=None)
    ap.add_argument('--sweep', type=int, default=0,
                    help='пройти по бэгу с этим шагом кадров')
    ap.add_argument('--corridors', action='store_true',
                    help='считать коридоры веток (по умолчанию только оси)')
    a = ap.parse_args()

    from visualize_bag import BagFrames, find_db3
    frames = BagFrames(find_db3(a.bag_dir), cache_size=2)

    if a.sweep:
        n_div = 0
        n_fr = 0
        for i in range(0, len(frames), a.sweep):
            pts = frames[i][0]
            fwd, lat, up = to_frame(pts)
            floor = floor_level(fwd, lat, up)
            det = detect_divergence(fwd, lat, up, floor)
            n_fr += 1
            n_div += det['diverged']
            desc = []
            for b in det['branches']:
                tag = '*' if b['own'] else ('#' if b['blocked'] else '')
                desc.append(f"{b['center']:+.2f}{tag}"
                            f"(g{b['gauge']:.2f},n{b['n_obs']},"
                            f"{b['span'][0]:.0f}-{b['span'][1]:.0f})")
            print(f"f{i:4d} diverged={int(det['diverged'])} "
                  f"n={len(det['branches'])}  " + '  '.join(desc))
        print(f'{a.bag_dir}: кадров {n_fr}, diverged {n_div} '
              f'({100.0 * n_div / max(n_fr, 1):.0f}%)')
        return

    fr = a.frame or 0
    pts = frames[fr][0]
    axis, det, corridors = analyze_frame(pts, with_corridors=a.corridors)
    print(f'{a.bag_dir} кадр {fr}: diverged={det["diverged"]}, '
          f'веток {len(det["branches"])}')
    for i, b in enumerate(det['branches']):
        print(_fmt_branch(i, b))
        if a.corridors and i in corridors:
            w = corridors[i]
            print(f'       коридор: подтверждён до {w["reach"]:.0f} м '
                  f'(измерено бинов {w["n_ok"]})')


if __name__ == '__main__':
    main()
