"""Замер: до какой дальности вперёд видны ОБЕ стены тоннеля.

Зачем
-----
Габарит на 100+ м по рельсам не построить: детекция нити кончается на 40 м, а
экстраполяция пути промахивается на метры (см. eval_extrapolation.py: дуга на
150 м даёт 5.78 м при полугабарите 1.7 м). Стены тоннеля параллельны пути и
видны дальше рельсов, поэтому они — следующая опора для коридора. Этот замер
отвечает, до какой дальности на них можно опираться ФАКТИЧЕСКИ.

Считается отдельно для прямых участков и кривых: в кривой лидар светит в
наружную стену, внутренняя уходит из поля зрения, и дальность падает вдвое.

Как ищется стена
----------------
НЕ глобальной гистограммой по X на весь кадр (так делает zones.detect_walls):
на кривой путь уходит вбок, и общая на кадр гистограмма размазывается. Здесь
каждый бин дальности обрабатывается отдельно, а окно поиска центрируется на
ОСИ ПУТИ в этом бине (из карты, если она есть).

ВАЖНО, как читать результат
---------------------------
Возвращается дальность, до которой обе стены видны НЕПРЕРЫВНО, — она обрывается
на первом бине, где точек не хватило. Это НЕ граница тоннеля: дальше точки
обычно есть снова (замер на roundT_doubleT f60: 145 точек на 80-90 м, провал до
21 на 90-100 м, снова 144 на 100-110 м). Лучи лидара расходятся веером, и на
дальности между кольцами появляются дыры; отражение под скользящим углом
слабеет.

Поэтому «стен не видно» значит «не наблюдается», а не «там пусто». Для решения
«есть ли препятствие в габарите» подменять одно другим нельзя: препятствие,
которое лидар не добил по дальности, выглядело бы как свободный путь.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from visualize_bag import BagFrames, find_db3  # noqa: E402
from track_map import map_to_frame  # noqa: E402

BIN = 10.0            # шаг бина по дальности, м
MAX_R = 170.0
MIN_PTS = 8           # точек на стену в бине, чтобы считать её видимой
Z_LO, Z_HI = -1.0, 2.0   # полоса высот, где стена, а не пол/потолок
SEARCH = 6.0          # полуокно поиска стены вбок от оси пути, м
MIN_OFF = 1.2         # стена не ближе этого к оси (иначе это габарит/поезд)
CURVE_DEG = 3.0       # изменение курса на 40 м, выше которого участок — кривая

def wall_reach(pts, fw, cx):
    """Дальность, до которой обе стены видны непрерывно, м."""
    y, x, z = -pts[:, 1], pts[:, 0], pts[:, 2]
    band = (z > Z_LO) & (z < Z_HI)
    reach = 0.0
    for lo in np.arange(0.0, MAX_R, BIN):
        hi = lo + BIN
        mid = lo + BIN / 2
        if fw is not None and len(fw) > 1 and fw.max() >= mid:
            axis = float(np.interp(mid, fw, cx))
        else:
            axis = 0.0
        m = band & (y >= lo) & (y < hi)
        if m.sum() < 2 * MIN_PTS:
            break
        dx = x[m] - axis
        left = (dx < -MIN_OFF) & (dx > -SEARCH)
        right = (dx > MIN_OFF) & (dx < SEARCH)
        if left.sum() < MIN_PTS or right.sum() < MIN_PTS:
            break
        reach = hi
    return reach

def curvature(poses, i, span=40.0):
    """Модуль изменения курса на span метров вперёд, град."""
    if i + 2 >= len(poses):
        return np.nan
    p = poses[i:, :2]
    d = np.hypot(*(p[1:] - p[:-1]).T).cumsum()
    j = np.searchsorted(d, span)
    if j >= len(poses) - i:
        return np.nan
    a = poses[i + j, 2] - poses[i, 2]
    return abs(np.degrees((a + np.pi) % (2 * np.pi) - np.pi))

BAGS = ['doubleT_platform', 'roundT_doubleT', 'roundT_pressureGate_roundT',
        'roundT_squareT_pressureGate_squareT', 'squareT_platform_squareT_switch',
        'doubleT_obstacle']
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pavel/for_hackathon')
print(f'{"бэг":36s} {"кадров":>6s} {"ПРЯМАЯ":>14s} {"КРИВАЯ":>14s}')
print(f'{"":36s} {"":>6s} {"med":>6s} {"p10":>7s} {"med":>6s} {"p10":>7s}')
allst, allcu = [], []
for b in BAGS:
    d = os.path.join(root, b)
    try:
        frames = BagFrames(find_db3(d), cache_size=4)
    except Exception as e:
        print(f'{b:36s}  — {e}'); continue
    tmp = os.path.join(d, 'track_map.npz'); trp = os.path.join(d, 'trajectory.npz')
    tm = dict(np.load(tmp)) if os.path.exists(tmp) else None
    tr = dict(np.load(trp)) if os.path.exists(trp) else None
    poses = tr['poses'] if tr else None
    st, cu = [], []
    step = max(1, len(frames) // 60)
    for i in range(0, len(frames), step):
        pts = frames[i][0]
        fw = cx = None
        if tm is not None and poses is not None and i < len(poses):
            pr = map_to_frame(tm, poses, i, max_range=MAX_R)
            if len(pr['fwd']) > 1:
                o = np.argsort(pr['fwd']); fw, cx = pr['fwd'][o], pr['center_x'][o]
        r = wall_reach(pts, fw, cx)
        k = curvature(poses, i) if poses is not None else np.nan
        (cu if (not np.isnan(k) and k > CURVE_DEG) else st).append(r)
    f = lambda a: (f'{np.median(a):5.0f}м {np.percentile(a,10):6.0f}м' if a else '     —      —')
    print(f'{b:36s} {len(frames):6d} {f(st)} {f(cu)}')
    allst += st; allcu += cu
print()
print(f'{"ИТОГО":36s} {"":6s} {np.median(allst):5.0f}м {np.percentile(allst,10):6.0f}м '
      f'{np.median(allcu):5.0f}м {np.percentile(allcu,10):6.0f}м' if allcu else '')
print(f'\nкритерий: >= {MIN_PTS} точек на каждую стену в бине {BIN:.0f} м,')
print(f'полоса высот {Z_LO}..{Z_HI} м, поиск в {MIN_OFF}..{SEARCH} м от оси пути')
print(f'«кривая» = изменение курса > {CURVE_DEG:.0f}° на 40 м вперёд')
print()
print('Дальность обрывается на ПЕРВОМ бине без стен. Дальше точки обычно есть')
print('снова — это дыры между кольцами лидара, а не конец тоннеля. «Не видно»')
print('читать как «не наблюдается», НЕ как «там свободно».')
print()
print('Оси пути берутся из track_map.npz бэга; если карта бэга битая, окно')
print('поиска центрируется мимо пути и дальность занижается (doubleT_obstacle).')
