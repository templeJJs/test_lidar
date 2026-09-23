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
# Порог вердикта «СТОИТ», м/кадр. Обоснование -- замер по записям for_hackathon
# (те же числа -- в докстринге `labeling.measure_motion`): стоящая запись
# (doubleT_obstacle) даёт медиану |0.00| м/кадр, едущие -- от 0.42 м/кадр
# (doubleT_platform) до 1.32 у обратного хода. 0.15 м/кадр (5.4 км/ч при 10 Гц)
# разделяет классы с запасом в обе стороны: в ~5 раз выше стоящей и примерно
# втрое ниже самой медленной едущей, поэтому на этих данных вердикт не граничит
# ни с одним наблюдением. Порог -- про СТОЯЩУЮ запись, а не про «медленно едет».
STAND_M_PER_FRAME = 0.15


def wall_profile(db: str, frame: int, side: int = -1, xyz_at=None):
    """Профиль стены: медиана x в полосах по y (стена = выше пола на 1 м, |x| > 1.5).

    `xyz_at` -- необязательный источник точек `xyz_at(frame) -> (xyz, ...)`; им
    пользуется `labeling.py`, у которого кадр уже открыт (вьюер держит запись
    живой). Без него запись открывается здесь, как было.
    """
    xyz = BagFrames(db, cache_size=2)[frame][0] if xyz_at is None else xyz_at(frame)[0]
    return wall_profile_points(xyz, side)


def wall_profile_points(xyz, side: int = -1, y_lo: float = Y_LO, y_hi: float = Y_HI,
                        band_m: float = BAND_M, wall_x_m: float = WALL_X_M,
                        z_up_m: float = Z_UP_M):
    """Профиль стены по УЖЕ ЗАГРУЖЕННОМУ облаку: (сетка по y, медианы x).

    Отдельно от `wall_profile` -- потому что профиль нужен ПО КАДРАМ, а открывать
    запись на каждый кадр (как делает `wall_profile`) дороже самого замера.
    Числа те же: полоса полосы та же (|y - yv| <= band_m), медиана по x тем же
    `np.median`, поэтому профиль совпадает с `wall_profile` значение в значение.

    Внутри -- вместо 140 булевых масок по всему облаку: один раз отбор точек стены,
    сортировка по y и границы полос через `searchsorted` (замер: 0.12 с -> 0.01 с).
    """
    x = xyz[:, 0].astype(float)
    y = xyz[:, 1].astype(float)
    z = xyz[:, 2].astype(float)
    floor = float(np.percentile(z, 5))
    m = (z > floor + z_up_m) & ((x < -wall_x_m) if side < 0 else (x > wall_x_m))
    grid = np.arange(y_lo, y_hi, band_m)
    out = np.full(grid.size, np.nan)
    if not m.any():
        return grid, out
    ys = y[m]
    xs = x[m]
    order = np.argsort(ys, kind='stable')
    ys_sorted = ys[order]
    xs_sorted = xs[order]
    lo = np.searchsorted(ys_sorted, grid - band_m, side='left')
    hi = np.searchsorted(ys_sorted, grid + band_m, side='right')
    for i in range(grid.size):
        if hi[i] - lo[i] >= 8:
            out[i] = float(np.median(xs_sorted[lo[i]:hi[i]]))
    return grid, out


def profile_shift(ga, pa, gb, pb, span_m: float = 25.0, step: float = BAND_M):
    """Сдвиг профиля b относительно a по готовым профилям: (сдвиг, остаток, полосы).

    Ровно та же формула, что в `best_shift` (интерполяция профиля b в `ga + d`,
    медиана модуля разности), но профили не считаются заново -- для ПОКАДРОВОГО
    замера они уже есть. Окно поиска `span_m` задаёт вызывающий: для СОСЕДНИХ
    кадров сдвиг ~1 м, и широкое окно (±25 м) на периодической структуре стены
    выбирает алиас (замерено: пара 100->110 на squareT, остаток 0.009-0.010 у пяти
    разных сдвигов +6.5..+8.0 м).
    """
    ok = np.isfinite(pa)
    best = None
    for d in np.arange(-span_m, span_m + 1e-9, step):
        q = np.interp(ga + d, gb, pb, left=np.nan, right=np.nan)
        both = ok & np.isfinite(q)
        if int(both.sum()) < 20:
            continue
        r = float(np.median(np.abs(pa[both] - q[both])))
        if best is None or r < best[1]:
            best = (float(d), r, int(both.sum()))
    return best


def profile_candidates(ga, pa, gb, pb, span_m: float = 25.0, step: float = BAND_M):
    """Кандидаты сдвига профиля b относительно a: [(сдвиг, остаток)] по возрастанию.

    Тот же расчёт, что в `profile_shift`, но отдаются ВСЕ сдвиги с остатком: у
    периодической стены поверхность остатка имеет несколько почти равных минимумов
    (замерено на `squareT_platform_squareT_switch`: r=0.003 при d = -1.25, -1.15,
    -1.00, -0.80, +0.25, +0.60, +0.75 при истинном сдвиге +1.35 м), и по одному
    «лучшему» сдвигу отличить истину от алиаса НЕЛЬЗЯ. Выбор делает вызывающий:
    при равных кандидатах -- по независимой проверке (отражатели), иначе кадр
    помечается недостоверным.
    """
    ok = np.isfinite(pa)
    out = []
    for d in np.arange(-span_m, span_m + 1e-9, step):
        q = np.interp(ga + d, gb, pb, left=np.nan, right=np.nan)
        both = ok & np.isfinite(q)
        if int(both.sum()) < 20:
            continue
        out.append((float(d), float(np.median(np.abs(pa[both] - q[both]))),
                    int(both.sum())))
    out.sort(key=lambda t: t[1])
    return out


def best_shift(db: str, fa: int, fb: int, side: int = -1, span_m: float = 25.0,
               xyz_at=None):
    """Сдвиг профиля fb относительно fa: (сдвиг, остаток, число полос)."""
    ga, pa = wall_profile(db, fa, side, xyz_at=xyz_at)
    gb, pb = wall_profile(db, fb, side, xyz_at=xyz_at)
    return profile_shift(ga, pa, gb, pb, span_m=span_m)


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
        first = None                       # сдвиг/остаток ПЕРВОЙ пары (fa -> fb)
        for a in range(fa, fb, step):
            b = a + (fb - fa)
            got = best_shift(db, a, b)
            if got is None:
                continue
            if first is None:              # первая итерация -- это ровно пара fa->fb
                first = got
            rows.append(got[0] / float(fb - fa))
        if not rows:
            print('%-38s %-9s | не оценить' % (rec, '%d->%d' % (fa, fb)))
            continue
        med = float(np.median(rows))
        # Сдвиг и остаток первой пары уже посчитаны циклом выше -- второй раз
        # best_shift(fa, fb) не зовём (это чтение двух кадров записи).
        r = first
        verdict = 'СТОИТ' if abs(med) < STAND_M_PER_FRAME \
            else '%.0f км/ч' % (med * HZ * 3.6)
        print('%-38s %-9s | %+6.2f м за %d | %+6.2f м/кадр | %.3f м  %s'
              % (rec, '%d->%d' % (fa, fb), (r[0] if r else float('nan')),
                 fb - fa, med, (r[1] if r else float('nan')), verdict))


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))