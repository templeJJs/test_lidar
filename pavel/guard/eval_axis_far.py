"""Замер оси вдаль (блок 1б) против карты пути.

Мерит ДВЕ вещи, и обе нужны одновременно:

  ошибка оси  — |ось(d) - карта(d)| на дальностях 30/50/70 м;
  скачок      — |ось(d+BIN) - ось(d)| между соседними бинами. Ось может быть
                верной в среднем и при этом ломаной: излом в 1.3 м на
                переключении режимов даёт габарит, который дёргается.

ВАЖНО про границу карты: карта кончается на 40-54 м, дальше np.interp
экстраполирует последним значением, и разница с ним ничего не значит. Поэтому
дальность входит в замер, только если она ВНУТРИ покрытия карты этого кадра.

    python eval_axis_far.py
    python eval_axis_far.py --bags roundT_pressureGate_roundT
"""
import argparse
import os
import sys

import numpy as np

# И guard/, и корень pavel/: visualize_bag.py и track_map.py лежат в корне,
# `guard.*` импортируется как пакет (как в guard/plot_axis.py).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visualize_bag import BagFrames, find_db3      # noqa: E402
from track_map import map_to_frame                 # noqa: E402
from guard.run import analyze                      # noqa: E402
from guard import axis_wall as AW                  # noqa: E402

RANGES = (30.0, 50.0, 70.0)
BAGS = ['roundT_pressureGate_roundT', 'roundT_doubleT', 'doubleT_platform']
STEP = 5          # шаг кадров: 5 даёт 100-200 кадров на бэг


def ref_axis(tm, poses, i):
    """Истинная ось из карты в системе кадра -> (fwd, lat) отсортировано."""
    pr = map_to_frame(tm, poses, i, max_range=150.0)
    if len(pr['fwd']) < 2:
        return None, None
    o = np.argsort(pr['fwd'])
    return pr['fwd'][o], pr['center_x'][o]


def run(bags, step=STEP, verbose=True, polyline=False):
    """-> dict: err[(bag, d)] = список ошибок; jump[bag] = список скачков."""
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../for_hackathon')
    err, jump, cov = {}, {}, {}

    for b in bags:
        d = os.path.join(root, b)
        tmp, trp = (os.path.join(d, f) for f in ('track_map.npz', 'trajectory.npz'))
        if not (os.path.exists(tmp) and os.path.exists(trp)):
            print(f'{b}: нет карты/траектории, пропуск')
            continue
        db = find_db3(d)
        frames = BagFrames(db, cache_size=4)
        tm, poses = dict(np.load(tmp)), np.load(trp)['poses']
        jump[b], cov[b] = [], [0, 0]

        for i in range(0, min(len(frames), len(poses)), step):
            rf, rx = ref_axis(tm, poses, i)
            if rf is None:
                continue
            axis, _ = analyze(frames[i][0], polyline=polyline,
                              bag_key=db, frame_index=i)
            tr = axis['far']
            if tr is None:
                continue

            # Ошибка на фиксированных дальностях — только внутри карты.
            for r in RANGES:
                if r > rf.max() or r > tr['reach']:
                    continue
                truth = float(np.interp(r, rf, rx))
                err.setdefault((b, r), []).append(abs(float(AW.lat_at(tr, r)) - truth))

            # Скачок между СМЕЖНЫМИ бинами: гладкость оси. Считается по
            # подтверждённому участку, где ось вообще есть. diff по маске ok
            # склеивал бы бины через дыру (бин на 40 м с бином на 60 м стали
            # бы «соседями»), поэтому скачок берём только там, где индексы
            # бинов идут подряд.
            ok = np.flatnonzero([s in ('seed', 'straight', 'wall')
                                 for s in tr['status']])
            if len(ok) >= 2:
                d = np.abs(np.diff(tr['lat'][ok]))
                jump[b] += list(d[np.diff(ok) == 1])
            cov[b][0] += 1
            cov[b][1] += int(tr['n_wall'] > 0)

    if verbose:
        report(bags, err, jump, cov)
    return err, jump, cov


def report(bags, err, jump, cov):
    print(f'{"бэг":<34s} ' + ' '.join(f'{r:>4.0f}м med  p95' for r in RANGES)
          + '   скачок med  p95   кадров  стена')
    for b in bags:
        if b not in jump:
            continue
        cells = []
        for r in RANGES:
            a = err.get((b, r), [])
            cells.append(f'{np.median(a):9.2f}{np.percentile(a, 95):5.2f}'
                         if len(a) >= 5 else f'{"—":>9s}{"—":>5s}')
        j = jump[b]
        js = (f'{np.median(j):9.2f}{np.percentile(j, 95):5.2f}'
              if len(j) >= 5 else f'{"—":>9s}{"—":>5s}')
        n, nw = cov[b]
        print(f'{b:<34s} ' + ' '.join(cells) + js
              + f'{n:8d}{(nw / max(n, 1)):7.0%}')

    # Сводка по всем бэгам: одно число на дальность, чтобы видеть регресс.
    allj = [v for b in bags for v in jump.get(b, [])]
    print()
    for r in RANGES:
        a = [v for b in bags for v in err.get((b, r), [])]
        if len(a) >= 5:
            print(f'  все бэги {r:.0f} м: med {np.median(a):.2f}  '
                  f'p95 {np.percentile(a, 95):.2f}  n={len(a)}')
    if allj:
        print(f'  скачок       : med {np.median(allj):.2f}  '
              f'p95 {np.percentile(allj, 95):.2f}  '
              f'max {np.max(allj):.2f}  n={len(allj)}')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bags', nargs='*', default=BAGS)
    ap.add_argument('--step', type=int, default=STEP)
    ap.add_argument('--polyline', action='store_true',
                    help='ось блока 1 ломаной по нитям полилинии')
    a = ap.parse_args()
    run(a.bags, a.step, polyline=a.polyline)


if __name__ == '__main__':
    main()
