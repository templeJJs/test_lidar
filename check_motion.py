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

Сдвиг уточняется суббинно (парабола по трём точкам сетки + мелкая дооценка
остатка, `refine_minimum`): сетка BAND_M=0.25 м квантует ход до 9 км/ч,
уточнение снимает квантование там, где минимум не плоский.

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

# Суббинное уточнение сдвига: сеточный минимум поверхности остатка (шаг BAND_M)
# уточняется параболической вершиной по трём соседним точкам сетки и дооценкой
# остатка на мелкой сетке в окрестности вершины. Без уточнения все сдвиги кратны
# BAND_M (0.25 м/кадр при 10 Гц = 9 км/ч), и сходимость с независимым замером
# (отражатели) лучше половины бина недостижима в принципе.
SUBBIN_FINE_STEP_M = 0.01   # мелкая сетка дооценки остатка
SUBBIN_FINE_GRID_M = 0.05   # ресемплинг профилей для мелкой дооценки
# Гарды уточнения (иначе -- прежнее сеточное значение): минимум не на краю
# сетки, кривизна параболы положительная и НЕ плоская (>= SUBBIN_MIN_CURVE_REL
# от остатка в минимуме), вершина внутри бина (|delta| <= половины шага).
SUBBIN_MIN_CURVE_REL = 0.02


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


def _residual_surface(ga, pa, gb, pb, span_m, step):
    """Поверхность остатка на сетке сдвигов: (сетка, остатки, полосы).

    Общий расчёт `profile_shift`/`profile_candidates`/`joint_candidates`:
    интерполяция профиля b в `ga + d`, медиана модуля разности; сдвиги с
    малым числом общих полос (< 20) -- NaN.
    """
    ok = np.isfinite(pa)
    shifts = np.arange(-span_m, span_m + 1e-9, step)
    res = np.full(shifts.size, np.nan)
    bands = np.zeros(shifts.size, dtype=int)
    for i, d in enumerate(shifts):
        q = np.interp(ga + d, gb, pb, left=np.nan, right=np.nan)
        both = ok & np.isfinite(q)
        if int(both.sum()) < 20:
            continue
        res[i] = float(np.median(np.abs(pa[both] - q[both])))
        bands[i] = int(both.sum())
    return shifts, res, bands


def residual_at(ga, pa, gb, pb, d):
    """Остаток совмещения профилей на КОНКРЕТНОМ сдвиге: (остаток, полосы)|None.

    Та же формула, что у `_residual_surface` -- чтобы остаток на уточнённом
    (суббинном) сдвиге был в той же шкале, что у сеточных кандидатов.
    """
    ok = np.isfinite(pa)
    q = np.interp(ga + float(d), gb, pb, left=np.nan, right=np.nan)
    both = ok & np.isfinite(q)
    if int(both.sum()) < 20:
        return None
    return float(np.median(np.abs(pa[both] - q[both]))), int(both.sum())


def _fine_profile(grid, prof, grid_m=SUBBIN_FINE_GRID_M):
    """Профиль, ресемплированный на мелкую сетку: (сетка, значения, маска)."""
    gf = np.arange(grid[0], grid[-1], grid_m)
    fin = np.isfinite(prof)
    pf = np.interp(gf, grid, np.where(fin, prof, 0.0))
    mf = np.interp(gf, grid, fin.astype(float))
    return gf, pf, mf


def _fine_residual_at(fa, fb, d):
    """Остаток на мелкой сетке: (остаток, точки) | None.

    Ресемплируются ОБА профиля (симметрично): односторонняя интерполяция
    профиля b на мелких сдвигах смещает дно поверхности на ~0.05-0.10 м
    (замер roundT_doubleT f100->101: мин 1.60 против 1.71 при истине 1.672
    по trajectory.npz).
    """
    gf, pa_f, ma = fa
    _, pb_f, mb = fb
    q = np.interp(gf + d, gf, pb_f, left=np.nan, right=np.nan)
    mq = np.interp(gf + d, gf, mb, left=0.0, right=0.0)
    both = (ma > 0.5) & (mq > 0.5) & np.isfinite(q)
    n = int(both.sum())
    if n < 100:
        return None
    return float(np.median(np.abs(pa_f[both] - q[both]))), n


def refine_minimum(structs, shifts, res, k, fine_step=SUBBIN_FINE_STEP_M):
    """Суббинное уточнение минимума поверхности: (сдвиг, остаток, полосы)|None.

    `structs` -- [(ga, pa, gb, pb), ...] структуры поверхности `res` (одна или
    несколько, сумма). Параболическая вершина по трём соседним точкам сетки
    (vertex = d_k + 0.5*(r_{k-1}-r_{k+1})/(r_{k-1}-2r_k+r_{k+1})), затем
    дооценка остатка на мелкой сетке `fine_step` в окрестности ±шаг сетки
    вокруг вершины. Уточнение применяется, только если минимум не на краю
    сетки, не плоский (кривизна >= SUBBIN_MIN_CURVE_REL * r_k) и вершина
    внутри бина (|delta| <= 0.5 шага); иначе None -- прежнее поведение.
    """
    if k <= 0 or k >= res.size - 1 or not np.isfinite(res[k]):
        return None
    r0, r1, r2 = res[k - 1], res[k], res[k + 1]
    if not (np.isfinite(r0) and np.isfinite(r2)):
        return None
    denom = r0 - 2.0 * r1 + r2
    if denom <= SUBBIN_MIN_CURVE_REL * max(r1, 1e-9):
        return None                                    # плоский минимум
    delta = 0.5 * (r0 - r2) / denom
    if abs(delta) > 0.5:
        return None                                    # вершина вне бина
    step = float(shifts[1] - shifts[0])
    center = float(shifts[k]) + delta * step
    lo = max(center - step, float(shifts[0]))
    hi = min(center + step, float(shifts[-1]))
    fines = [(_fine_profile(ga, pa), _fine_profile(gb, pb))
             for ga, pa, gb, pb in structs]
    best = None
    for d in np.arange(lo, hi + 1e-9, fine_step):
        r_sum, n_min = 0.0, None
        for fa, fb in fines:
            got = _fine_residual_at(fa, fb, float(d))
            if got is None:
                r_sum = None
                break
            r_sum += got[0]
            n_min = got[1] if n_min is None else min(n_min, got[1])
        if r_sum is None:
            continue
        if best is None or r_sum < best[1]:
            best = (float(d), r_sum, n_min)
    return best


def _coarse_scale(structs, d):
    """Остаток уточнённого сдвига в шкале СЕТОЧНОЙ поверхности: (r, полосы)|None.

    Мелкая дооценка (`_fine_residual_at`) ресемплирует оба профиля и потому
    даёт остаток в ДРУГОЙ шкале (на острых структурах выше: замер squareT
    f100->101 -- 1.11 против 0.71 на сетке). Смешивать шкалы в одном списке
    кандидатов нельзя, поэтому уточнённый сдвиг переводится в грубую шкалу
    через `residual_at` (для нескольких структур -- сумма).
    """
    r_sum, n_min = 0.0, None
    for ga, pa, gb, pb in structs:
        got = residual_at(ga, pa, gb, pb, d)
        if got is None:
            return None
        r_sum += got[0]
        n_min = got[1] if n_min is None else min(n_min, got[1])
    return r_sum, n_min


def joint_candidates(structs, span_m=25.0, step=BAND_M, refine=True):
    """Кандидаты сдвига по СУММЕ поверхностей остатка независимых структур.

    `structs` -- [(ga, pa, gb, pb), ...] (левая стена, правая стена, ...).
    Ложные алиасы разных структур по сдвигу почти никогда не совпадают,
    истинный минимум совпадает всегда: сумма поверхностей снимает алиасы
    (замер на squareT f100->101: у левой стены r=0.003 при пяти сдвигах
    -1.25..-0.75 и +0.25..+0.75, у правой -- широкий минимум на истинном
    +1.25; у суммы минимум единственный, на +1.25).

    Структура участвует, только если её поверхность конечна не менее чем на
    половине сетки (иначе она добавляет дырки, а не сигнал). Сдвиг выдаётся,
    только если на нём конечны ВСЕ участвующие поверхности; остаток кандидата
    -- сумма остатков, полосы -- минимум по структурам. Первый кандидат
    (минимум суммы) уточняется суббинно (`refine_minimum`), если гарды это
    разрешают; иначе остаётся сеточным, как раньше. Остаток уточнённого
    кандидата переведён в грубую шкалу (`_coarse_scale`), поэтому уточнённый
    кандидат остаётся первым БЕЗ пересортировки: его грубый остаток может
    чуть превышать остаток сеточного минимума -- это артефакт квантования
    грубой формулы, а не ухудшение совмещения.
    """
    surfaces = []
    for ga, pa, gb, pb in structs:
        shifts, res, bands = _residual_surface(ga, pa, gb, pb, span_m, step)
        if np.isfinite(res).mean() >= 0.5:
            surfaces.append((ga, pa, gb, pb, shifts, res, bands))
    if not surfaces:
        return []
    shifts = surfaces[0][4]
    valid = np.ones(shifts.size, dtype=bool)
    res_sum = np.zeros(shifts.size)
    band_min = None
    for _, _, _, _, _, res, bands in surfaces:
        fin = np.isfinite(res)
        valid &= fin
        res_sum = res_sum + np.where(fin, res, 0.0)
        band_min = bands if band_min is None else np.minimum(band_min, bands)
    out = [(float(shifts[i]), float(res_sum[i]), int(band_min[i]))
           for i in np.flatnonzero(valid)]
    out.sort(key=lambda t: t[1])
    if refine and out:
        res_masked = np.where(valid, res_sum, np.nan)
        k = int(np.nanargmin(res_masked))
        got = refine_minimum([s[:4] for s in surfaces], shifts, res_masked, k)
        if got is not None:
            coarse = _coarse_scale([s[:4] for s in surfaces], got[0])
            if coarse is not None:
                out[0] = (got[0], coarse[0], coarse[1])
    return out


def refine_to(ga, pa, gb, pb, d, span_m=25.0, step=BAND_M):
    """Суббинное уточнение в окрестности ИЗВЕСТНОГО сдвига по одной структуре.

    Для связки «совместная поверхность выбирает бин, основная структура даёт
    значение»: совместный минимум снимает алиасы, но тянуть суббинное значение
    к смещённой структуре нельзя (замер roundT_doubleT: правая сторона --
    не стена, а второй путь/окклюзии, её минимум смещён на -0.2..-0.3 м);
    поэтому бин выбирается совместно (`joint_candidates`), а уточнение идёт
    по поверхности одной (основной) структуры вокруг выбранного сдвига.
    Гарды -- те же, что у `refine_minimum`; None -- остаётся прежнее значение.
    Остаток -- в грубой шкале (`residual_at`).
    """
    shifts, res, bands = _residual_surface(ga, pa, gb, pb, span_m, step)
    if np.isfinite(res).mean() < 0.5:
        return None
    dist = np.where(np.isfinite(res), np.abs(shifts - float(d)), np.inf)
    k = int(np.argmin(dist))
    got = refine_minimum([(ga, pa, gb, pb)], shifts, res, k)
    if got is None:
        return None
    coarse = _coarse_scale([(ga, pa, gb, pb)], got[0])
    if coarse is None:
        return None
    return got[0], coarse[0], coarse[1]


def profile_shift(ga, pa, gb, pb, span_m: float = 25.0, step: float = BAND_M):
    """Сдвиг профиля b относительно a по готовым профилям: (сдвиг, остаток, полосы).

    Ровно та же формула, что раньше (интерполяция профиля b в `ga + d`,
    медиана модуля разности) плюс суббинное уточнение минимума
    (`refine_minimum`): сеточное значение кратно BAND_M, уточнённое -- нет.
    Окно поиска `span_m` задаёт вызывающий: для СОСЕДНИХ кадров сдвиг ~1 м,
    и широкое окно (±25 м) на периодической структуре стены выбирает алиас
    (замерено: пара 100->110 на squareT, остаток 0.009-0.010 у пяти разных
    сдвигов +6.5..+8.0 м).
    """
    shifts, res, bands = _residual_surface(ga, pa, gb, pb, span_m, step)
    if not np.isfinite(res).any():
        return None
    k = int(np.nanargmin(res))
    got = refine_minimum([(ga, pa, gb, pb)], shifts, res, k)
    if got is not None:
        coarse = _coarse_scale([(ga, pa, gb, pb)], got[0])
        if coarse is not None:
            return got[0], coarse[0], coarse[1]
    return float(shifts[k]), float(res[k]), int(bands[k])


def profile_candidates(ga, pa, gb, pb, span_m: float = 25.0, step: float = BAND_M):
    """Кандидаты сдвига профиля b относительно a: [(сдвиг, остаток)] по возрастанию.

    Одна структура -- частный случай `joint_candidates`; первый кандидат
    уточнён суббинно (остальные -- сеточные, как раньше). У периодической
    стены поверхность остатка имеет несколько почти равных минимумов
    (замерено на `squareT_platform_squareT_switch`: r=0.003 при d = -1.25,
    -1.15, -1.00, -0.80, +0.25, +0.60, +0.75 при истинном сдвиге +1.35 м),
    и по одному «лучшему» сдвигу отличить истину от алиаса НЕЛЬЗЯ. Выбор
    делает вызывающий: совместной поверхностью нескольких структур
    (`joint_candidates`) и/или независимой проверкой (отражатели), иначе кадр
    помечается недостоверным.
    """
    return joint_candidates([(ga, pa, gb, pb)], span_m=span_m, step=step)


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