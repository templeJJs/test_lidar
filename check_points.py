"""Сколько точек даёт лидар: по записям, по метру пути и сколько из них рабочих.

  python check_points.py            # выборка 8 кадров на запись, ~минута

Кадры разбираются ВЫБОРОЧНО: парсить все 2488 кадров корпуса значит прочитать
десятки ГБ CDR. Разброс точек внутри записи узкий (мин/макс в пределах 1-10 %),
поэтому медиана кадра x число кадров даёт сумму с точностью порядка процента.

Рабочая доля -- точки в КОРИДОРЕ детектора (|u| <= half_width_m, высота от УГР
h_low..h_high), то есть то, что вообще может стать препятствием. Канонизация позы
кадра -- через модель (TrackModel.relative), как в самом детекторе.
"""
from __future__ import annotations

import numpy as np

from visualize_bag import BagFrames
from detector import DetectorConfig
from detector.profiles import model_for_db

RECS = ['doubleT_obstacle', 'doubleT_platform', 'roundT_doubleT',
        'roundT_pressureGate_roundT', 'roundT_squareT_pressureGate_squareT',
        'squareT_platform_squareT_switch']
SAMPLE_PER_REC = 8
M_PER_FRAME = 1.2          # ход машины за кадр (замерено по движению конца ленты)


def main() -> int:
    cfg = DetectorConfig()
    print('коридор детектора: |u| <= %.2f м, h = %.2f..%.2f м'
          % (cfg.half_width_m, cfg.h_low_m, cfg.h_high_m))
    print()
    print('%-36s %5s | %-24s | %9s | %11s | %8s | %8s'
          % ('запись', 'кадр', 'точек в кадре (выборка)', 'медиана', 'всего, млн',
             'точ/м', 'в коридоре'))
    tot_frames = 0
    tot_pts = 0.0
    for rec in RECS:
        db = 'for_hackathon/%s/%s_0.db3' % (rec, rec)
        fr = BagFrames(db, cache_size=2)
        n = len(fr)
        idx = sorted(set(np.linspace(0, n - 1, SAMPLE_PER_REC).astype(int).tolist()))
        counts = [int(fr[i][0].shape[0]) for i in idx]
        med = float(np.median(counts))
        model = model_for_db(db)
        xyz = np.asarray(fr[idx[len(idx) // 2]][0], dtype=np.float64)
        u, h = model.relative(xyz)
        inside = int(np.count_nonzero((np.abs(u) <= cfg.half_width_m)
                                      & (h >= cfg.h_low_m) & (h <= cfg.h_high_m)))
        tot_frames += n
        tot_pts += med * n
        print('%-36s %5d | %-24s | %9.0f | %11.2f | %8.0f | %7.2f %%'
              % (rec, n, 'мин %d / макс %d' % (min(counts), max(counts)),
                 med, med * n / 1e6, med / M_PER_FRAME, 100.0 * inside / counts[-1]))
    path = tot_frames * M_PER_FRAME
    print()
    print('ИТОГО: кадров %d, точек ~%.0f млн, путь ~%.2f км, ~%.0f точек/м (~%.2f млрд на км)'
          % (tot_frames, tot_pts / 1e6, path / 1000.0, tot_pts / path, tot_pts / path * 1000 / 1e9))
    print('оценка суммы: медиана кадра x число кадров (выборка %d кадров на запись)'
          % SAMPLE_PER_REC)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())