"""Замер: можно ли вести ось пути отступом от стены.

Вопрос
------
На кривой рельсы кончаются на 40 м, а экстраполяция промахивается на метры.
Стена видна дальше и параллельна пути — значит, ось можно вести отступом от
неё. Вопрос только в том, какая у этого ОШИБКА на реальных данных.

Как меряется
------------
Коридор ведётся БЕЗ карты (axis=0) — то есть ровно так, как он работал бы в
realtime на новом участке. Полученная ось сравнивается с осью из track_map,
которая здесь служит эталоном: её ошибка 10 см на 170 м (README), то есть на
порядок меньше того, что мы меряем.

Три способа построить ось из стен:
  center — середина между двумя стенами (нужны обе)
  left   — отступ от ЛЕВОЙ стены на полширины
  right  — отступ от ПРАВОЙ стены на полширины

Полуширина берётся не константой, а по ближним бинам ТЕКУЩЕГО кадра: тоннели
разные (3.0-4.0 м), и зашивать число нельзя.

Отдельно считается, что бывает, когда видна ТОЛЬКО ОДНА стена: на кривой это
типичный случай (лидар светит в наружную стену, внутренняя уходит из поля
зрения), и именно там схема «отступ от стены» либо работает, либо нет.

Разбивка
--------
Результат разложен по типу участка, потому что «нестандартные пути» — это не
одна проблема, а три разные:
  прямая / кривая — по изменению курса на 40 м вперёд
  узкий / широкий — по измеренной ширине коридора (зал и двухпутка шире 5 м)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from visualize_bag import BagFrames, find_db3  # noqa: E402
from track_map import map_to_frame  # noqa: E402
import wall_corridor as wc  # noqa: E402

CURVE_DEG = 3.0       # изменение курса на 40 м, выше которого участок — кривая
WIDE = 5.0            # ширина коридора, выше которой это зал/двухпутка
NEAR_BINS = 4         # по скольким ближним бинам берётся полуширина тоннеля
RANGES = (30.0, 50.0, 70.0, 90.0)   # на каких дальностях сверяем ось

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


def axes_from_walls(w):
    """Три оценки оси пути по стенам + полуширина по ближним бинам.

    Возвращает (fwd, center, left_off, right_off, half, ok) — ok отмечает
    бины, где стены ИЗМЕРЕНЫ, а не достроены предсказанием.
    """
    ok = np.asarray(w['status']) == 'ok'
    L, R = w['left_x'], w['right_x']

    # Полуширина по ближним ПОДТВЕРЖДЁННЫМ бинам текущего кадра.
    near = ok & (w['fwd'] <= w['fwd'][0] + NEAR_BINS * wc.BIN)
    half = float(np.median(w['width'][near])) / 2 if near.any() else np.nan

    center = (L + R) / 2
    left_off = L + half     # от левой стены вправо
    right_off = R - half    # от правой стены влево
    return w['fwd'], center, left_off, right_off, half, ok


def main():
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pavel/for_hackathon')

    # err[(сегмент, метод, дальность)] = список ошибок, м
    err = {}
    seen_one_wall = {'кривая': 0, 'прямая': 0}
    tot_frames = {'кривая': 0, 'прямая': 0}

    for b in BAGS:
        d = os.path.join(root, b)
        try:
            frames = BagFrames(find_db3(d), cache_size=4)
        except Exception as e:
            print(f'{b}: {e}')
            continue
        tmp, trp = (os.path.join(d, f) for f in ('track_map.npz', 'trajectory.npz'))
        if not (os.path.exists(tmp) and os.path.exists(trp)):
            continue
        tm, poses = dict(np.load(tmp)), np.load(trp)['poses']
        if len(tm['s']) <= 20:
            print(f'{b}: карта битая ({len(tm["s"])} узлов), пропуск')
            continue

        step = max(1, len(frames) // 40)
        for i in range(0, len(frames), step):
            if i >= len(poses):
                break
            pr = map_to_frame(tm, poses, i, max_range=wc.MAX_R)
            if len(pr['fwd']) < 2:
                continue
            o = np.argsort(pr['fwd'])
            ref_fwd, ref_x = pr['fwd'][o], pr['center_x'][o]

            # ВЕДЁМ БЕЗ КАРТЫ — как в realtime на новом участке
            pts = frames[i][0]
            w = wc.trace_walls(pts, None, None)
            fwd, center, loff, roff, half, ok = axes_from_walls(w)
            if np.isnan(half):
                continue

            k = curvature(poses, i)
            seg = 'кривая' if (not np.isnan(k) and k > CURVE_DEG) else 'прямая'
            wid = w['width'][ok]
            wide = len(wid) and float(np.median(wid)) > WIDE
            tot_frames[seg] += 1

            for r in RANGES:
                j = int(np.argmin(np.abs(fwd - r)))
                if not ok[j] or r > ref_fwd.max():
                    continue
                truth = float(np.interp(r, ref_fwd, ref_x))
                for name, val in (('center', center[j]),
                                  ('left', loff[j]), ('right', roff[j])):
                    if np.isnan(val):
                        continue
                    key = (seg, 'широкий' if wide else 'узкий', name, r)
                    err.setdefault(key, []).append(abs(val - truth))

    # --- вывод ---
    print('Ошибка оси пути, построенной ПО СТЕНАМ без карты, против карты-эталона')
    print(f'(полуширина по {NEAR_BINS} ближним бинам текущего кадра; '
          f'полугабарит {wc.GAUGE_HALF} м)\n')

    for seg in ('прямая', 'кривая'):
        for kind in ('узкий', 'широкий'):
            rows = [(r, m) for (s, k, m, r) in err if s == seg and k == kind]
            if not rows:
                continue
            print(f'--- {seg}, {kind} профиль ---')
            print(f'{"метод":>8s} ' + ' '.join(f'{r:>13.0f}м' for r in RANGES))
            print(f'{"":8s} ' + ' '.join(f'{"med":>6s}{"p95":>8s}' for _ in RANGES))
            for m in ('center', 'left', 'right'):
                cells = []
                for r in RANGES:
                    a = err.get((seg, kind, m, r), [])
                    cells.append(f'{np.median(a):6.2f}{np.percentile(a, 95):8.2f}'
                                 if len(a) >= 5 else f'{"—":>6s}{"—":>8s}')
                print(f'{m:>8s} ' + ' '.join(cells))
            n = max((len(err.get((seg, kind, m, RANGES[0]), []))
                     for m in ('center', 'left', 'right')), default=0)
            print(f'         кадров на ближней дальности: {n}\n')

    print(f'полугабарит — {wc.GAUGE_HALF} м. Метод годится, если ошибка заметно')
    print('меньше: ошибка оси прибавляется к габариту с обеих сторон.')


if __name__ == '__main__':
    main()
