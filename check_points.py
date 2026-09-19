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



# --------------------------------------------------------- по дальности -----
# Полосы по 3D-дальности от сенсора (как её меряет лидар). Печатается доля от
# кадра, точки по корпусу и то же самое на метр пути.
RANGE_EDGES = [0.0, 1.0, 10.0, 20.0, 50.0, 100.0, float('inf')]
RANGE_LABELS = ['0..1 м', '1..10 м', '10..20 м', '20..50 м', '50..100 м', '>100 м']


def by_range() -> int:
    print()
    print('по дальности от сенсора (3D |x,y,z|), медиана по %d кадрам на запись'
          % SAMPLE_PER_REC)
    print()
    print('%-34s | %s' % ('запись', ' | '.join('%-9s' % s for s in RANGE_LABELS)))
    band_total = np.zeros(len(RANGE_LABELS))
    frames_total = 0
    for rec in RECS:
        db = 'for_hackathon/%s/%s_0.db3' % (rec, rec)
        fr = BagFrames(db, cache_size=2)
        n = len(fr)
        idx = sorted(set(np.linspace(0, n - 1, SAMPLE_PER_REC).astype(int).tolist()))
        frac = []
        counts = []
        for i in idx:
            xyz = np.asarray(fr[i][0], dtype=np.float64)
            r = np.sqrt((xyz ** 2).sum(axis=1))
            h, _ = np.histogram(r, bins=RANGE_EDGES)
            frac.append(h / xyz.shape[0] * 100.0)
            counts.append(int(xyz.shape[0]))
        med = np.median(np.asarray(frac), axis=0)
        # Оценка точек в полосе = медиана ПРОИЗВЕДЕНИЙ (точек в кадре x доля),
        # а не произведение медиан: число точек по кадрам гуляет на 4-17 %,
        # и произведение медиан смещает оценку.
        in_band = np.asarray([c * f / 100.0 for c, f in zip(counts, frac)])
        band_total += np.median(in_band, axis=0) * n
        frames_total += n
        print('%-34s | %s' % (rec, ' | '.join('%8.2f%%' % v for v in med)))
    path = frames_total * M_PER_FRAME
    print()
    print('%-12s %14s %16s %10s' % ('полоса', 'точек по корпусу', 'на метр пути', 'доля кадра'))
    for lb, v in zip(RANGE_LABELS, band_total):
        print('%-12s %14.0f %16.0f %9.1f %%' % (lb, v, v / path, 100.0 * v / band_total.sum()))
    print()
    print('НАКОПИТЕЛЬНО (внутри радиуса):')
    cum = np.cumsum(band_total)
    for lim, v in zip([1.0, 10.0, 20.0, 50.0, 100.0], cum[:-1]):
        print('  в пределах %5.0f м %14.0f точек = %8.0f на метр пути = %5.1f %% кадра'
              % (lim, v, v / path, 100.0 * v / band_total.sum()))
    return 0


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
        # Доля «в коридоре» считается НА КАЖДОМ кадре выборки и берётся медианой:
        # числитель и знаменатель обязаны быть из одного кадра (раньше inside брался
        # со среднего кадра, а делилось на последний -- расхождение до 18 %).
        shares = []
        for i in idx:
            xyz = np.asarray(fr[i][0], dtype=np.float64)
            u, h = model.relative(xyz)
            inside = int(np.count_nonzero((np.abs(u) <= cfg.half_width_m)
                                          & (h >= cfg.h_low_m) & (h <= cfg.h_high_m)))
            shares.append(100.0 * inside / xyz.shape[0])
        share = float(np.median(shares))
        tot_frames += n
        tot_pts += med * n
        print('%-36s %5d | %-24s | %9.0f | %11.2f | %8.0f | %7.2f %%'
              % (rec, n, 'мин %d / макс %d' % (min(counts), max(counts)),
                 med, med * n / 1e6, med / M_PER_FRAME, share))
    path = tot_frames * M_PER_FRAME
    print()
    print('ИТОГО: кадров %d, точек ~%.0f млн, путь ~%.2f км, ~%.0f точек/м (~%.2f млрд на км)'
          % (tot_frames, tot_pts / 1e6, path / 1000.0, tot_pts / path, tot_pts / path * 1000 / 1e9))
    print('оценка суммы: медиана кадра x число кадров (выборка %d кадров на запись)'
          % SAMPLE_PER_REC)
    by_range()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())