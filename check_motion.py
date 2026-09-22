"""Замер хода машины по записям: совмещение профиля стены между кадрами.

  python check_motion.py [кадр_от] [кадр_до] [шаг]

Зачем: кадры записи отличаются не только поворотом, но и **сдвигом** (машина едет).
Сдвиг измеряется по сигналу с мелкой структурой — профилю СТЕНЫ (кронштейны, ниши,
трубы): гладкий профиль головки рельса для этого не годится, он почти не различает
положение. Берётся пара кадров, профиль второго смещается по y, ищется сдвиг с
наименьшим остатком; остаток показывает, насколько совмещение осмысленно.

Результат — ход в метрах за кадр и км/ч (частота кадров 10 Гц), плюс остаток:
  < 0.05 м — совмещение резкое, сдвигу можно верить;
  > 0.20 м — сигнал зашумлён, число считать нижней оценкой.

Инструмент нужен разметке: объект в движущейся записи должен приближаться к сенсору
ровно с этим ходом, а не с выдуманной скоростью.
"""
from __future__ import annotations

import sys

import numpy as np

from visualize_bag import BagFrames

RECS = ['doubleT_obstacle', 'doubleT_platform', 'roundT_doubleT',
        'roundT_pressureGate_roundT', 'roundT_squareT_pressureGate_squareT',
        'squareT_platform_squareT_switch']
HZ = 10.0
BAND_M = 0.25
Y_LO, Y_HI = -40.0, -5.0
WALL_X_M = 1.5
Z_UP_M = 1.0


def wall_profile(db: str, frame: int, side: int = -1):
    """Профиль стены: медиана x в полосах по y (стена = выше пола на 1 м, |x| > 1.5)."""
    xyz = BagFrames(db, cache_size=2)[frame][0]
    x = xyz[:, 0].astype(float)
    y = xyz[:, 1].astype(float)
    z = xyz[:, 2].astype(float)
    floor = float(np.percentile(z, 5))
    m = (z > floor + Z_UP_M) & ((x < -WALL_X_M) if side < 0 else (x > WALL_X_M))
    grid = np.arange(Y_LO, Y_HI, BAND_M)
    out = np.full(grid.size, np.nan)
    for i, yv in enumerate(grid):
        b = m & (np.abs(y - yv) <= BAND_M)
        if int(b.sum()) >= 8:
            out[i] = float(np.median(x[b]))
    return grid, out


def best_shift(db: str, fa: int, fb: int, side: int = -1, span_m: float = 25.0):
    """Сдвиг профиля fb относительно fa: (сдвиг, остаток, число полос)."""
    ga, pa = wall_profile(db, fa, side)
    gb, pb = wall_profile(db, fb, side)
    ok = np.isfinite(pa)
    best = None
    step = BAND_M
    for d in np.arange(-span_m, span_m + 1e-9, step):
        q = np.interp(ga + d, gb, pb, left=np.nan, right=np.nan)
        both = ok & np.isfinite(q)
        if int(both.sum()) < 20:
            continue
        r = float(np.median(np.abs(pa[both] - q[both])))
        if best is None or r < best[1]:
            best = (float(d), r, int(both.sum()))
    return best


def main(argv):
    fa = int(argv[0]) if len(argv) > 0 else 100
    fb = int(argv[1]) if len(argv) > 1 else 110
    step = int(argv[2]) if len(argv) > 2 else 20
    print('совмещение профиля стены: кадры %d -> %d и далее с шагом %d' % (fa, fb, step))
    print()
    print('%-38s %-9s | %-11s | %-9s | %-8s' %
          ('запись', 'кадры', 'сдвиг', 'м/кадр', 'остаток'))
    for rec in RECS:
        db = 'for_hackathon/%s/%s_0.db3' % (rec, rec)
        rows = []
        for a in range(fa, fb, step):
            b = a + (fb - fa)
            got = best_shift(db, a, b)
            if got is not None:
                rows.append(got[0] / float(fb - fa))
        if not rows:
            print('%-38s %-9s | не оценить' % (rec, '%d->%d' % (fa, fb)))
            continue
        med = float(np.median(rows))
        r = best_shift(db, fa, fb)
        verdict = 'СТОИТ' if abs(med) < 0.15 else '%.0f км/ч' % (med * HZ * 3.6)
        print('%-38s %-9s | %+6.2f м за %d | %+6.2f м/кадр | %.3f м  %s'
              % (rec, '%d->%d' % (fa, fb), (r[0] if r else float('nan')),
                 fb - fa, med, (r[1] if r else float('nan')), verdict))


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))