"""Замер: до какой дальности wall_corridor честно держит габарит.

Отличие от eval_wall_range.py
-----------------------------
Тот замер отвечал «до какой дальности В ПРИНЦИПЕ видны обе стены» — то есть
какой потолок у метода. Этот меряет, сколько из него берёт РЕАЛЬНЫЙ алгоритм
ведения коридора, и заодно проверяет то, ради чего он написан: не цепляется ли
коридор за второй путь.

Что считается
-------------
1. reach — дальность подтверждённого габарита, медиана и p10 по кадрам,
   отдельно на прямой и в кривой. Смотреть на p10: для безопасности важен
   худший кадр, а не типичный.
2. Ширина коридора и её СКО — прокси на срывы. Коридор, перескочивший на
   соседний объём, даёт ширину 5-7 м вместо 3-4 м.
3. Доля кадров, где ширина ушла за 5 м, — прямой счётчик срывов на второй путь.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from visualize_bag import BagFrames, find_db3  # noqa: E402
from track_map import map_to_frame  # noqa: E402
import wall_corridor as wc  # noqa: E402

CURVE_DEG = 3.0

# Широкий коридор сам по себе НЕ ошибка: roundT_doubleT с кадра ~150 реально
# въезжает в двухпутный участок, и правая стена честно уходит с +2.5 на +5.5 м
# (замер по гистограмме X ближней зоны). Мерить надо не ширину, а РАЗРЫВ:
# своя стена между соседними бинами не прыгает, чужая появляется скачком.
JUMP = 1.0        # скачок выноса стены между соседними бинами, м

BAGS = ['doubleT_platform', 'roundT_doubleT', 'roundT_pressureGate_roundT',
        'roundT_squareT_pressureGate_squareT', 'squareT_platform_squareT_switch']


def curvature(poses, i, span=40.0):
    if poses is None or i + 2 >= len(poses):
        return np.nan
    p = poses[i:, :2]
    d = np.hypot(*(p[1:] - p[:-1]).T).cumsum()
    j = np.searchsorted(d, span)
    if j >= len(poses) - i:
        return np.nan
    a = poses[i + j, 2] - poses[i, 2]
    return abs(np.degrees((a + np.pi) % (2 * np.pi) - np.pi))


def main():
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pavel/for_hackathon')
    print(f'{"бэг":36s} {"ПРЯМАЯ":>13s} {"КРИВАЯ":>13s} {"ширина":>16s} {"срывы":>6s}')
    print(f'{"":36s} {"med":>6s} {"p10":>6s} {"med":>6s} {"p10":>6s} '
          f'{"med":>7s} {"СКО":>7s}')
    tot_st, tot_cu, tot_w, tot_bad, tot_n = [], [], [], 0, 0

    for b in BAGS:
        d = os.path.join(root, b)
        try:
            frames = BagFrames(find_db3(d), cache_size=4)
        except Exception as e:
            print(f'{b:36s}  — {e}')
            continue
        tmp, trp = (os.path.join(d, f) for f in ('track_map.npz', 'trajectory.npz'))
        tm = dict(np.load(tmp)) if os.path.exists(tmp) else None
        poses = np.load(trp)['poses'] if os.path.exists(trp) else None
        usable_map = tm is not None and len(tm['s']) > 20

        st, cu, ws = [], [], []
        bad = 0
        step = max(1, len(frames) // 40)
        for i in range(0, len(frames), step):
            pts = frames[i][0]
            fw = cx = None
            if usable_map and poses is not None and i < len(poses):
                pr = map_to_frame(tm, poses, i, max_range=wc.MAX_R)
                if len(pr['fwd']) > 1:
                    o = np.argsort(pr['fwd'])
                    fw, cx = pr['fwd'][o], pr['center_x'][o]
            w = wc.trace_walls(pts, fw, cx)
            ok = w['status'] == 'ok'
            wid = w['width'][ok]
            if len(wid):
                ws.append(float(np.median(wid)))
                for side in ('left_x', 'right_x'):
                    v = w[side][ok]
                    if len(v) > 1 and np.nanmax(np.abs(np.diff(v))) > JUMP:
                        bad += 1
                        break
            k = curvature(poses, i)
            (cu if (not np.isnan(k) and k > CURVE_DEG) else st).append(w['reach'])

        f = lambda a: (f'{np.median(a):5.0f}м {np.percentile(a, 10):5.0f}м'
                       if a else '    —     —')
        wtxt = (f'{np.median(ws):6.2f}м {np.std(ws):6.2f}м' if ws else '     —      —')
        n = len(st) + len(cu)
        print(f'{b:36s} {f(st)} {f(cu)} {wtxt} {bad:3d}/{n:<3d}')
        tot_st += st; tot_cu += cu; tot_w += ws; tot_bad += bad; tot_n += n

    print()
    f = lambda a: (f'{np.median(a):5.0f}м {np.percentile(a, 10):5.0f}м'
                   if a else '    —     —')
    wtxt = (f'{np.median(tot_w):6.2f}м {np.std(tot_w):6.2f}м' if tot_w else '—')
    print(f'{"ИТОГО":36s} {f(tot_st)} {f(tot_cu)} {wtxt} {tot_bad:3d}/{tot_n:<3d}')
    print()
    print(f'«срыв» = скачок выноса стены между соседними бинами > {JUMP:.1f} м:')
    print('своя стена так не прыгает, это переход на соседний объём. Саму по')
    print('себе широкую ширину за срыв НЕ считаем — на двухпутном участке')
    print('(roundT_doubleT с кадра ~150) коридор в 7 м честный.')
    print()
    print('reach — дальность ПОДТВЕРЖДЁННОГО габарита. Дальше ответ «не')
    print('наблюдается», а не «свободно»: провалы точек на дальности — дыры')
    print('между кольцами лидара, а не конец тоннеля.')


if __name__ == '__main__':
    main()
