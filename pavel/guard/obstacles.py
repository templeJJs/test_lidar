"""Блок 3. Препятствия в габарите: остаток после объяснения штатным.

Место в тракте
--------------
Блок 1 (track.py) даёт ось пути по рельсам, блок 1б (axis_wall.py) ведёт её
вдаль стеной, блок 2 (walls.py) строит коридор свободного пространства, блок
2б (structures.py) объясняет штатные конструкции внутри коридора. Здесь —
последний шаг: препятствие это то, что НЕ удалось отнести ни к чему штатному.
Цель — не видеть препятствий там, где их нет.

Метод — перенос проверенного `wall_corridor.obstacles_in_gauge` (README,
«Препятствия в габарите»: человек на ~55 м найден в 21 кадре из 21) на выход
блоков 1-2б. Три ключа, без которых метод не работает (все — замеры README):

1. Объект виден как локальный ИЗБЫТОК над фоном: пол, лоток и дальняя стена
   дают ровно ~30 точек на ячейку по всей ширине. Фон — 40-й перцентиль
   занятых ячеек, НЕ медиана: сам объект входит в выборку и завышает её
   (f190: медиана 20 при фоне 2-6 в соседних окнах).
2. Окна вдоль пути перекрываются (шаг вдвое меньше окна): иначе объект на
   стыке делится пополам и не набирает порог — терялся в 5 кадрах из 9.
3. Высота объекта — верх СПЛОШНОЙ снизу части колонки (разрыв по
   гистограмме), а не max(Z): над человеком в той же ячейке лежат точки
   свода, и высота всех находок получалась 3.70 м — признак «человек ниже
   1.5 м» не работал.

Отличие от прежнего детектора: поиск идёт по ОСТАТКУ точек после блока 2б
(`wall['structures']['mask']`), поэтому штатные ниши, кронштейны и порталы
сюда не доходят вовсе, а не отсеиваются порогами. Бины со статусом 'hole'
НЕ пропускаются (в doubleT_obstacle человек стоит ровно в таком), но находка
помечается confirmed=False.

Остаток после блока 2б — это другая статистика, чем полное облако: фон
разрежен, и на нём всплывают свои штатные объекты. Пять правил-объяснений
(все про геометрию, никакого описания конкретного объекта — прежний блок 3
был откачен именно за «ширина ≤ 2 м, высота ≤ 2.4 м»), замеры у констант:

1. КОЛОННА ДО СВОДА — конструкция тоннеля (пилон, портал). Принцип и порог
   COL_TOP_H те же, что во втором признаке structures.py.
2. СТЕНОВАЯ РЯБЬ — кластер целиком в полосе WALL_RIPPLE внутрь от
   измеренной кромки стены или за ней: кромка — перцентиль стеновых точек,
   и часть стены всегда лежит внутрь от неё (замер: 0.02-0.25 м).
3. ПЫЛЬ — точки размазаны по дальности больше DEPTH_MAX: твёрдый объект
   впереди виден лицевой гранью и компактен вдоль луча (человек на 55 м:
   p90-p10 дальностей ≤ 0.3 м), пыль рассеяния у стен и между кольцами
   размазана на 1.7-3.6 м.
4. НЕ СТОИТ НА ПОЛОТНЕ — низ колонны выше BASE_MAX: контактный рельс
   (0.60-0.67 м на четырёх бэгах), подвесы, оборудование. У человека низ
   0.10-0.23 м. Правило отключается в бинах, где пол не покрыт кольцами
   (wall['uncovered']): там низ объекта принципиально невидим.
5. ФРАГМЕНТ КОНСТРУКЦИИ — над верхом колонны в той же полосе поперёк с
   разрывом не больше допустимого разрыва колонны продолжается уже
   объяснённое. У фрагментов зазор −0.05..+0.09 м, у человека до свода
   2.0+ м. Над человеком свод объяснён, но зазор большой — он остаётся.

Ось габарита
------------
Отсчёт поперечного положения — от ведомой оси коридора `wall['axis']`: на
кривой только она повторяет изгиб пути. Исключение — когда ведомая ось не
держится ни за что: в зале doubleT_obstacle стены дальше габарита с обеих
сторон, ведение стеной с постоянным отступом теряет смысл (замер: ось на
55 м гуляет −0.4..−1.7 м между кадрами при стоящем поезде, и человек в
центре пути «выпадал» из габарита). Там, где обе стены за габаритом, а ось
измерена рельсами (axis['measured']), габарит центрируется на прямой оси
блока 1: рельсы — единственное прямое измерение пути.

CLI:
    python -m guard.obstacles for_hackathon/doubleT_obstacle --frame 40
    python -m guard.obstacles for_hackathon/doubleT_obstacle --sweep 10
"""

import argparse
import time

import numpy as np

from .geom import to_frame
from .track import axis_at
from .walls import BIN, MAX_RANGE, GAUGE_HALF, GAUGE_H_LO, GAUGE_H_HI

# --- кластеризация (константы — замеры README «Препятствия в габарите») -----
OBST_DX = 0.30        # ширина ячейки поперёк пути, м
OBST_MIN_PTS = 12     # точек в кластере, чтобы он считался объектом
OBST_MIN_H = 0.4      # минимальная высота объекта над полом, м
OBST_EXCESS = 2.5     # во сколько раз кластер должен превышать фон окна
OBST_BG_Q = 40.0      # перцентиль фона: не медиана — сам объект завышает
                      # выборку (замер f190: медиана 20 при фоне 2-6)
OBST_WIN = 5.0        # длина окна поиска вдоль пути, м
OBST_STEP = 2.5       # шаг окна, м. ВДВОЕ меньше окна — окна перекрываются,
                      # иначе объект на стыке делится пополам и не набирает
                      # порог (README: терялся в 5 кадрах из 9)
SEARCH_EXTRA = 1.5    # запас полосы поиска за габаритом, м: объект у края
                      # надо увидеть целиком, чтобы честно сказать, заходит
                      # он внутрь или нет (замер wall_corridor: человек на
                      # x=+0.7..+1.3 терял половину точек при полосе по
                      # клампам стен)
MERGE_DX = 0.6        # находки соседних перекрывающихся окон — одна, если
                      # ближе этого поперёк (перекрытие окон даёт дубли)
BG_EXCL = 2           # фон ячейки считается по ячейкам дальше этого числа
                      # ячеек от неё: сам объект не должен входить в оценку
                      # фона (та же причина, что у OBST_BG_Q). Замер f90
                      # doubleT_obstacle: человек занимает ВСЕ 3 занятые
                      # ячейки окна (44+19+23 точки, ширина до 0.8 м = 3
                      # ячейки), p40 насчитался 19.8 по его же флангам и
                      # избыток 2.2x не прошёл порог 2.5x — человек терялся.
                      # 2 ячейки = 0.6 м: край объекта шириной до 0.8 м
                      # (против него мерились) из центра не дальше 1-2 ячеек

# --- высота объекта (замеры README, см. _column_height) ---------------------
COL_GAP = 0.5         # разрыв колонки по высоте, м: над головой человека
                      # такой разрыв есть, у стены и колонны его нет
COL_STEP = 0.25       # шаг гистограммы высоты, м
COL_DENSE = 40        # от стольких точек кластера слой считается занятым от
                      # 2 точек; на разреженном (18 точек на 3 м высоты у
                      # человека на 60 м, замер f160) — от 1

# --- чем кластер остатка объясняется штатным --------------------------------
# Препятствие на пути СТОИТ на полотне. Замеры по бэгам (пробник
# _tmp_obst_probe10): низ колонны человека на 55 м — 0.10-0.23 м над полом
# (21 кадр), а у штатных продольных линий (контактный рельс и т.п.) она на
# 0.60-0.67 м (doubleT_obstacle f40/f190, roundT_pressureGate f200,
# doubleT_platform f80 — везде 0.60+), у фрагментов пилонов и оборудования
# 1.2-2.1 м. Порог посередине между полотном и линиями.
BASE_MAX = 0.5        # низ колонны выше этого — объект не стоит на пути, м
FRAG_GAP = COL_GAP    # если объяснённая конструкция продолжается над верхом
                      # колонны с разрывом не больше допустимого разрыва
                      # самой колонны — это её фрагмент. Замер: у фрагментов
                      # зазор −0.05..+0.09 м, у человека до свода 2.0+ м
FRAG_LAT = 0.45       # полоса поперёк, где ищется продолжение, м — ячейка
                      # CELL_LAT (0.4) из structures.py плюс допуск на край
COL_TOP_H = 2.60      # колонна, дотянувшаяся досюда, — конструкция тоннеля
                      # (пилон, портал): тот же принцип и значение, что
                      # COL_TOP_H в structures.py — объект ниже 2.6 м до
                      # свода не дотягивается
# Глубина объекта ВДОЛЬ луча. Твёрдый объект впереди виден лицевой гранью, и
# его точки компактны по дальности; пыль рассеяния у стен и между кольцами
# размазана по всему 5-метровому окну. Замер (пробник _tmp_obst_probe15):
# человек на 55 м — размах fwd 0.10-0.41 м (p90-p10 <= 0.3) на всех кадрах;
# ложные сгущения остатка на 75-100 м — p90-p10 1.7-3.6 м, размах 2.4-4.9 м.
# Порог — щедрый запас на рельеф лицевой поверхности любого препятствия
# (даже у вагонетки/стремянки/ящика она десятки сантиметров), а не повторение
# размера конкретного объекта.
DEPTH_MAX = 1.0       # p90-p10 дальностей точек кластера, м
# Стеновая рябь. Кромка стены — перцентиль (Q_RANK-я точка) стеновых точек
# (walls.py), поэтому часть стены ВСЕГДА лежит внутрь от кромки; на дальности
# кромка ещё и шумит. Замер глубины проникновения остатка внутрь от
# измеренной кромки (5 находок на 65-107 м, roundT_squareT_..., squareT_...,
# roundT_pressureGate): 0.02-0.25 м. Кластер, ЦЕЛИКОМ лежащий в этой полосе
# внутрь от кромки или за ней, — это стена: свободного объекта там быть не
# может (он оказался бы внутри стены). Объект шире полосы не съедается
# целиком: у него остаётся внутренняя часть.
WALL_RIPPLE = 0.30    # глубина полосы внутрь от измеренной кромки стены, м


def _column_height(zs, floor, gap=COL_GAP, step=COL_STEP):
    """Верх сплошной снизу части колонки точек, м над полом.

    Человек и свод попадают в одну ячейку по X, поэтому max(Z) описывает не
    объект, а тоннель. Высота растёт, пока слои идут подряд, и обрывается на
    первом разрыве в `gap` метров. Разрыв ищется по ГИСТОГРАММЕ, а не по
    соседним точкам: одиночные точки на стойке над головой смыкают зазор, и
    колонна человека тянулась до свода (замер doubleT_obstacle f140-f180:
    высота 3.70 м у той же цели, что в соседних кадрах давала 1.2 м).
    """
    zs = np.sort(np.asarray(zs, dtype=float) - floor)
    if len(zs) == 0:
        return 0.0
    nb = max(1, int(np.ceil((zs[-1] - zs[0]) / step)) + 1)
    cnt, edges = np.histogram(zs, bins=nb, range=(zs[0], zs[0] + nb * step))
    occ = cnt >= (2 if len(zs) >= COL_DENSE else 1)
    run = int(np.ceil(gap / step))
    top = zs[0]
    empty = 0
    for i, o in enumerate(occ):
        if o:
            top = min(edges[i + 1], zs[-1])
            empty = 0
        else:
            empty += 1
            if empty >= run:
                break
    return float(top)


def _pct_lin(vals, q):
    """Перцентиль, бит-в-бит np.percentile(vals, q, method='linear') для 1-D,
    но без накладных расходов np.percentile (он зовётся на каждый кластер
    окна). Формула ветвится по дробной части, чтобы порядок операций с
    float совпадал с numpy и результат не отличался ни в одном бите."""
    s = np.sort(vals)
    n = len(s)
    if n == 1:
        return float(s[0])
    pos = (q / 100.0) * (n - 1)
    i = int(pos)
    g = pos - i
    a = s[i]
    b = s[min(i + 1, n - 1)]
    return float(a + (b - a) * g) if g < 0.5 else float(b - (b - a) * (1 - g))


def _explain_cluster(col_top, h_pts, f_pts, x0, x1, xc, frag,
                     floor_visible, wall_l, wall_r):
    """Чем кластер остатка объясняется штатным — или None (тогда препятствие).

    Пять правил, все — про геометрию, а не про объект (никакого описания
    человека: прежний блок 3 был откачен именно за это):

    'column'   — колонна достала до свода (>= COL_TOP_H): пилон, портал,
                 створка. Тот же принцип, что в structures.py.
    'wall'     — кластер целиком в полосе WALL_RIPPLE внутрь от измеренной
                 кромки стены или за ней: это сама стена (кромка — перцентиль
                 стеновых точек, внутрь от неё всегда лежит рябь стены).
                 Только по ИЗМЕРЕННЫМ сторонам: кламп габарита стену не
                 измерял и ничего не объясняет.
    'dust'     — точки размазаны по дальности больше DEPTH_MAX: твёрдый
                 объект впереди виден лицевой гранью и компактен вдоль луча;
                 размах в метры — это пыль рассеяния у стен и между кольцами.
    'base'     — колонна НЕ стоит на полотне: её низ выше BASE_MAX над полом.
                 Контактный рельс, подвесы, оборудование над поездом.
                 Отключается, где пол не покрыт кольцами (floor_visible=False):
                 там низ объекта принципиально невидим.
    'fragment' — над верхом колонны в той же полосе поперёк продолжается
                 уже объяснённая конструкция с разрывом не больше FRAG_GAP:
                 это её нижний фрагмент, недобранный блоком 2б.

    Порядок важен: 'column' первым — у пилона низ тоже может быть выше пола
    (низ съеден объяснением), а поверхность его реальна и по глубине
    компактна. frag — не массивы, а callable() -> (de, he) объяснённых точек
    окна: правило 'fragment' идёт последним, и до него доживает мало
    кластеров, поэтому маска объяснённых точек считается лениво, только для
    доживших.
    """
    if col_top >= COL_TOP_H:
        return 'column'
    if not len(h_pts):
        return None
    if np.isfinite(wall_r) and x0 >= wall_r - WALL_RIPPLE:
        return 'wall'
    if np.isfinite(wall_l) and x1 <= wall_l + WALL_RIPPLE:
        return 'wall'
    if len(f_pts) >= 4:
        # p90-p10, не размах: одиночный отскок не растянет глубину.
        if _pct_lin(f_pts, 90.0) - _pct_lin(f_pts, 10.0) > DEPTH_MAX:
            return 'dust'
    # Низ колонны — 5-й перцентиль, а не минимум: одиночный шумовой отскок
    # от балласта не должен «приземлять» подвес.
    base = _pct_lin(h_pts, 5.0)
    if floor_visible and base > BASE_MAX:
        return 'base'
    de, he = frag()
    if len(he):
        near = np.abs(de - xc) < FRAG_LAT
        above = he[near & (he > col_top)]
        if len(above) and float(above.min()) - col_top <= FRAG_GAP:
            return 'fragment'
    return None


def gauge_centers(wall, axis):
    """Ось отсчёта габарита по бинам, м (абсолютный lat).

    Ведомая ось коридора, КРОМЕ бинов, где она не подтверждена ничем: обе
    стены дальше габарита (стена не держит ось, отступ к ней измерен в другом
    месте) и при этом ось измерена рельсами. Там — прямая ось блока 1.
    Обоснование и замер — в докстринге модуля.
    """
    center = np.asarray(wall['axis'], dtype=float).copy()
    if axis.get('measured'):
        unanchored = ((np.abs(wall['left']) > GAUGE_HALF)
                      & (wall['right'] > GAUGE_HALF))
        straight = axis_at(axis, np.asarray(wall['fwd_mid'], dtype=float))
        center[unanchored] = straight[unanchored]
    return center


def detect(fwd, lat, up, floor, wall, axis):
    """Остаток необъяснённого -> находки в габарите.

    fwd/lat/up — кадр в рабочей системе (geom.to_frame), floor — пол кадра,
    wall — коридор из walls.trace (с 'structures' из structures.explain, если
    блок 2б был включён), axis — ось блока 1.

    Возвращает (hits, summary). hits — список по окнам (после слияния
    перекрывающихся): d (дальность центра, м), lat (медиана относительно оси
    габарита), lat_abs (то же в системе лидара), width, height, n_pts,
    in_gauge (пересекается с коридором), confirmed (бин измерен, не 'hole').
    summary — счётчики кадра.
    """
    fwd = np.asarray(fwd, dtype=float)
    lat = np.asarray(lat, dtype=float)
    up = np.asarray(up, dtype=float)
    h = up - floor

    fm = np.asarray(wall['fwd_mid'], dtype=float)
    status = np.asarray(wall['status'], dtype=object)
    live = np.isfinite(wall['axis']) & (status != 'stop')
    summary = dict(resid_pts=0, windows=0, hits=0, hits_in_gauge=0,
                   hits_in_gauge_confirmed=0, detect_ms=0.0)
    if not live.any():
        return [], summary
    limit = float(fm[live].max()) + BIN / 2

    center_bins = gauge_centers(wall, axis)
    half_l = np.asarray(wall['half_left'], dtype=float)
    half_r = np.asarray(wall['half_right'], dtype=float)

    cand = (fwd >= 0) & (fwd < limit) & (h >= GAUGE_H_LO) & (h < GAUGE_H_HI)
    st = wall.get('structures')
    if st is not None:
        expl = st['mask'] & (fwd >= 0) & (fwd < limit)
        cand &= ~st['mask']   # объяснено конструкцией — не препятствие
    else:
        expl = np.zeros(len(fwd), dtype=bool)
    summary['resid_pts'] = int(cand.sum())
    # Слепые зоны колец пола: там низ объекта принципиально невидим, и
    # требовать «стоит на полотне» нельзя — ноги объекта съедает дыра.
    unc = np.asarray(wall.get('uncovered', np.zeros(len(fm), bool)), dtype=bool)

    t0 = time.perf_counter()
    # Работаем по индексам кандидатов один раз: полные маски по облаку на
    # каждое окно — это десятки миллионов операций на кадр.
    ci = np.flatnonzero(cand)
    # Ось бина -> точки: интерполяция между центрами бинов (как в
    # walls.trace) только для кандидатов — d остальных точек нужна лишь в
    # ленивой проверке 'fragment'.
    cf, cz, ch = fwd[ci], up[ci], h[ci]
    cd = lat[ci] - np.interp(cf, fm[live], center_bins[live])
    # Объяснённые конструкцией точки нужны только для проверки «фрагмент
    # конструкции» (колонна продолжается в объяснённую массу) — последнего
    # правила _explain_cluster. Доживают до него немногие, поэтому массивы
    # собираются лениво, при первом дожившем кластере.
    _expl = [None]
    def expl_pts():
        if _expl[0] is None:
            ei = np.flatnonzero(expl)
            ef = fwd[ei]
            ed = lat[ei] - np.interp(ef, fm[live], center_bins[live])
            _expl[0] = (ef, ed, h[ei])
        return _expl[0]
    # Измеренные кромки стен по сторонам (nan там, где сторона не измерена —
    # кламп габарита или предсказание дыры: они ничего не объясняют).
    lead = np.asarray(wall.get('lead', ['none'] * len(fm)), dtype=object)
    meas_l = np.where((status == 'ok') | ((status == 'gauge') & (lead == 'left')),
                      wall['left'], np.nan)
    meas_r = np.where((status == 'ok') | ((status == 'gauge') & (lead == 'right')),
                      wall['right'], np.nan)
    ok_l = np.isfinite(meas_l)
    fm_l, ed_l = fm[ok_l], meas_l[ok_l]
    ok_r = np.isfinite(meas_r)
    fm_r, ed_r = fm[ok_r], meas_r[ok_r]

    def edge_at(fm_e, ed_e, f):
        """Измеренная кромка стороны на дальности f (интерполяция по бинам)."""
        if len(fm_e) < 2:
            return np.nan
        return float(np.interp(f, fm_e, ed_e))

    hits = []
    explained = dict(base=0, fragment=0, column=0, dust=0, wall=0)
    for w0 in np.arange(0.0, limit - OBST_WIN + 1e-6, OBST_STEP):
        w1 = w0 + OBST_WIN
        mid = (w0 + w1) / 2
        i = int(np.argmin(np.abs(fm - mid)))
        if status[i] == 'stop':
            continue
        summary['windows'] += 1
        # Бины 'hole' НЕ пропускаются: в doubleT_obstacle человек на 55 м
        # попадает ровно в такой бин. Габарит там держится клампом, а находка
        # помечается confirmed=False.
        confirmed = status[i] in ('ok', 'gauge')
        m = (cf >= w0) & (cf < w1) & (np.abs(cd) < GAUGE_HALF + SEARCH_EXTRA)
        if m.sum() < OBST_MIN_PTS:
            continue
        dd, zz, h_pts_w, f_pts_w = cd[m], cz[m], ch[m], cf[m]
        # Маска объяснённых точек того же окна — лениво, на первый кластер,
        # доживший до правила 'fragment' (большинство срезается раньше).
        win_expl = [None]
        def frag_pts(w0=w0, w1=w1, win_expl=win_expl):
            if win_expl[0] is None:
                ef, ed, eh = expl_pts()
                me = (ef >= w0) & (ef < w1)
                win_expl[0] = (ed[me], eh[me])
            return win_expl[0]

        edges = np.arange(dd.min(), dd.max() + OBST_DX, OBST_DX)
        if len(edges) < 3:
            continue
        cnt, _ = np.histogram(dd, bins=edges)
        # Порог ячейки — избыток над фоном окна. Фон считается по занятым
        # ячейкам ВНЕ окрестности самой ячейки (BG_EXCL): иначе объект сам
        # завышает свой фон — на разреженном остатке после блока 2б в окне
        # может вообще не быть других ячеек (замер f90: все 3 занятые ячейки
        # окна — это человек, p40 по ним дал фон 19.8 и съел избыток).
        occ_idx = np.flatnonzero(cnt > 0)
        qual = np.zeros(len(cnt), dtype=bool)
        for c in occ_idx:
            rest = cnt[occ_idx[np.abs(occ_idx - c) > BG_EXCL]]
            bg = _pct_lin(rest, OBST_BG_Q) if len(rest) else 0.0
            qual[c] = cnt[c] >= max(OBST_MIN_PTS, OBST_EXCESS * bg)

        k = 0
        while k < len(cnt):
            if not qual[k]:
                k += 1
                continue
            j = k
            while j + 1 < len(cnt) and qual[j + 1]:
                j += 1
            sel = (dd >= edges[k]) & (dd < edges[j + 1])
            n = int(sel.sum())
            if n >= OBST_MIN_PTS:
                # Высота — верх сплошной снизу части колонки, не max(Z):
                # над объектом в той же ячейке лежат точки свода.
                hh = _column_height(zz[sel], floor)
                xc = float(np.median(dd[sel]))
                x0, x1 = float(dd[sel].min()), float(dd[sel].max())
                fmed = float(np.median(f_pts_w[sel]))
                why = _explain_cluster(hh, h_pts_w[sel], f_pts_w[sel],
                                       x0, x1, xc, frag_pts,
                                       floor_visible=not unc[i],
                                       wall_l=edge_at(fm_l, ed_l, fmed),
                                       wall_r=edge_at(fm_r, ed_r, fmed))
                if why is not None:
                    explained[why] += 1
                elif hh >= OBST_MIN_H:
                    # «В габарите» — если объект пересекается с коридором, а
                    # не если его центр внутри: человек у края опасен краем
                    # (README: для безопасности это правильная сторона ошибки).
                    inside = (x1 > -half_l[i]) and (x0 < half_r[i])
                    hits.append(dict(d=float(mid), lat=xc,
                                     lat_abs=xc + float(center_bins[i]),
                                     x_min=x0, x_max=x1, width=x1 - x0,
                                     height=hh, n_pts=n,
                                     in_gauge=bool(inside),
                                     confirmed=bool(confirmed)))
            k = j + 1

    hits = _merge_hits(hits)
    summary['detect_ms'] = (time.perf_counter() - t0) * 1000.0
    summary['explained'] = explained
    summary['hits'] = len(hits)
    summary['hits_in_gauge'] = sum(h['in_gauge'] for h in hits)
    summary['hits_in_gauge_confirmed'] = sum(
        h['in_gauge'] and h['confirmed'] for h in hits)
    return hits, summary


def _merge_hits(hits, dr=OBST_WIN, dx=MERGE_DX):
    """Схлопывает находки перекрывающихся окон: ближе dr по дальности и dx
    поперёк — один объект; остаётся находка с большим числом точек."""
    out = []
    for h in sorted(hits, key=lambda a: -a['n_pts']):
        if any(abs(h['d'] - o['d']) < dr and abs(h['lat'] - o['lat']) < dx
               for o in out):
            continue
        out.append(h)
    return sorted(out, key=lambda a: a['d'])


def _print_hits(hits, prefix=''):
    print(f'{prefix}{"дальн":>6s} {"lat":>6s} {"от оси":>7s} {"шир":>5s} '
          f'{"выс":>5s} {"точек":>6s} {"габарит":>8s} {"подтв":>6s}')
    for o in hits:
        print(f'{prefix}{o["d"]:6.1f} {o["lat_abs"]:+6.2f} {o["lat"]:+7.2f} '
              f'{o["width"]:5.2f} {o["height"]:5.2f} {o["n_pts"]:6d} '
              f'{"ДА" if o["in_gauge"] else "нет":>8s} '
              f'{"да" if o["confirmed"] else "НЕТ":>6s}')


def main():
    ap = argparse.ArgumentParser(
        description='Блок 3: препятствия в габарите из остатка после '
                    'объяснения штатным')
    ap.add_argument('bag_dir')
    ap.add_argument('--frame', type=int, default=0)
    ap.add_argument('--sweep', type=int, default=0,
                    help='пройти по бэгу с этим шагом кадров и дать сводку')
    a = ap.parse_args()

    from .run import analyze, _frames
    frames = _frames(a.bag_dir)

    if a.sweep:
        fr_hits, fr_ingauge, fr_unc = 0, 0, 0
        n_in, n_in_conf, tms = 0, 0, []
        expl_tot = dict(base=0, fragment=0, column=0, dust=0, wall=0)
        idx = range(0, len(frames), a.sweep)
        for i in idx:
            axis, wall = analyze(frames[i][0])
            fwd, lat, up = to_frame(frames[i][0])
            hits, summ = detect(fwd, lat, up, wall['floor'], wall, axis)
            tms.append(summ['detect_ms'])
            for key in expl_tot:
                expl_tot[key] += summ['explained'][key]
            ing = [h for h in hits if h['in_gauge']]
            fr_hits += bool(hits)
            n_in += len(ing)
            n_in_conf += sum(h['confirmed'] for h in ing)
            if ing:
                fr_ingauge += 1
                fr_unc += any(not h['confirmed'] for h in ing)
                for h in ing:
                    print(f'  [{i:4d}] ПРЕПЯТСТВИЕ d={h["d"]:5.1f} '
                          f'lat={h["lat_abs"]:+.2f} (от оси {h["lat"]:+.2f}) '
                          f'ш={h["width"]:.2f} в={h["height"]:.2f} '
                          f'n={h["n_pts"]}'
                          + ('' if h['confirmed'] else '  [hole, не подтв.]'))
        n = len(list(idx))
        print(f'{a.bag_dir}: {n} кадров')
        print(f'  кадров с находками        : {fr_hits}')
        print(f'  кадров с находкой в габарите: {fr_ingauge} '
              f'(из них с неподтверждённой: {fr_unc})')
        print(f'  находок в габарите        : {n_in}, подтверждённых: {n_in_conf}')
        print(f'  объяснено штатным         : полотно={expl_tot["base"]} '
              f'фрагмент={expl_tot["fragment"]} колонна={expl_tot["column"]} '
              f'пыль={expl_tot["dust"]} стена={expl_tot["wall"]}')
        print(f'  время detect, мс          : med {np.median(tms):.1f}  '
              f'max {np.max(tms):.1f}')
        return

    pts = frames[a.frame][0]
    axis, wall = analyze(pts)
    fwd, lat, up = to_frame(pts)
    hits, summ = detect(fwd, lat, up, wall['floor'], wall, axis)
    print(f'{a.bag_dir} кадр {a.frame}: остаток после конструкций '
          f'{summ["resid_pts"]} точек, находок {len(hits)}')
    _print_hits(hits, prefix='  ')


if __name__ == '__main__':
    main()
