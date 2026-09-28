"""Сборка тракта: ось прямого пути -> стены -> конструкции -> препятствия.

Порядок блоков и есть суть решения: сначала объяснить всё, что можем,
штатным (стена, конструкция тоннеля); препятствие — то, что объяснить
не удалось (блок 3).

    блок 1  track.py       ось пути по головкам рельсов (только прямая)
    блок 1б axis_wall.py   ось вдаль: где прямая не объясняет стену, её
                           ведёт стена — так получается поворот
    блок 2  walls.py       внешняя граница коридора: стены тоннеля
    блок 2б structures.py  конструкции внутри коридора
    блок 3  obstacles.py   препятствия в габарите из остатка необъяснённого
    блок 3.5 tracker.py    межкадровый трекинг находок блока 3 (sweep)

    python -m guard.run for_hackathon/doubleT_platform --frame 100
    python -m guard.run for_hackathon/doubleT_platform --frame 100 --branches
    python -m guard.run for_hackathon/doubleT_platform --sweep 20 --accum
    python -m guard.run for_hackathon/doubleT_platform --sweep 10 --accum-pts
"""

import argparse

import numpy as np

from .geom import to_frame, floor_level
from .track import straight_axis
from .axis_wall import trace_axis
from .walls import BIN, MAX_RANGE, trace
from .structures import H_HI, H_LO, explain


def analyze(pts, wall_axis=True, with_structures=True, with_obstacles=True,
            extra=None, polyline=False, bag_key=None, frame_index=None):
    """Кадр -> (ось, коридор). Один вызов на кадр.

    wall_axis — вести ось стеной там, где прямая перестала её объяснять.
    Возвращаемая ось получает поле 'bins' с осью БЛОКА 1в (wall_rules.build:
    режимы RAILS/CENTER/LEAD/HOLD) — от неё и строится коридор блока 2,
    повторяя изгиб пути. Поле 'far' (результат axis_wall.trace_axis) остаётся
    рядом: оно даёт измеренный в ближней зоне отступ и диагностику, но
    коридор по нему больше НЕ ведётся. Раньше вёлся, и правки правил ведения
    (блок 1в) в габарит не попадали вовсе — коридор шёл по другой оси, чем
    та, которую показывает plot_axis и по которой правятся пороги.

    with_structures — считать блок 2б (конструкции тоннеля) следом за стенами.
    Результат кладётся в коридор ключом 'structures' (маска объяснённых
    точек и занятость ячеек): остаток необъяснённого — вход блока 3.

    with_obstacles — считать блок 3 (препятствия в габарите) по остатку после
    блоков 1-2б. Результат — коридор ключом 'obstacles': (hits, summary) из
    obstacles.detect. Дёшево (8-28 мс на кадр), включено по умолчанию.

    polyline — ось блока 1 брать ЛОМАНОЙ по двум нитям полилинейной детекции
    (rail_poly над rail_geometry) вместо прямой. Прямая на кривой промахивается
    втрое больше полугабарита; ломаная идёт по середине измеренных нитей.
    Где нитей нет, откат на прямую (бит в бит прежнее поведение), поэтому флаг
    аддитивен. bag_key/frame_index — ключи тёплой сборки полилинии.

    extra — (fwd, lat, up) агрегированных точек прошлых кадров в системе
    текущего (см. _PointAggregator): идёт только в оценку кромок стен
    (walls.trace, параметр extra), НЕ в structures/препятствия — движущийся
    объект размазывается одометрией в полосу и должен оставаться покадровым.
    """
    fwd, lat, up = to_frame(pts)
    floor = floor_level(fwd, lat, up)
    if polyline:
        from .rail_poly import axis_or_straight
        axis = axis_or_straight(pts, fwd, lat, up, floor,
                                bag_key=bag_key, frame_index=frame_index)
    else:
        axis = straight_axis(fwd, lat, up, floor)
    axis['far'] = (trace_axis(fwd, lat, up, floor, axis) if wall_axis else None)
    # Ось блока 1в: стены кадра (check_wall, про ось не знают) плюс отступ,
    # измеренный в ближней зоне ЭТОГО кадра. Отступ берётся из axis_wall —
    # там он и мерится (measure_offset); без него LEAD берёт нижнюю границу
    # физики и честно помечает бины как неизмеренные.
    axis['bins'] = None
    if wall_axis:
        from .check_wall import detect as _detect_walls
        from . import wall_rules as WR
        found = _detect_walls(fwd, lat, up, floor, max_range=MAX_RANGE)
        if found:
            axis['bins'] = WR.build(found, axis, off=axis['far']['offset'],
                                    max_range=MAX_RANGE)
    wall = trace(fwd, lat, up, floor, axis, bins=axis['bins'], extra=extra)
    if with_structures:
        wall['structures'] = explain(fwd, lat, up, floor, axis)
    if with_obstacles:
        from .obstacles import detect as _detect_obstacles
        wall['obstacles'] = _detect_obstacles(fwd, lat, up, floor, wall, axis)
    return axis, wall


class _PointAggregator:
    """Точки последних кадров, перенесённые ПОЛНЫМ SE(2) в систему текущего.

    В отличие от accum.CorridorAccumulator (копит ВЫВОДЫ по бинам и потому
    вынужден отбраковывать кадры на кривых, см. MAX_TURN там), здесь
    переносятся сами точки: поворот учитывается точно, и на кривых агрегат
    корректен. Позы — trajectory.npz (odometry.py), конвенция та же, что в
    accum._shift_along_path: p_world = R(theta) p_local + t, локальная
    система лидара x=lat, y=-fwd. Проверка конвенции (_tmp_se2_check):
    кромки стен, посчитанные по ОДНОМУ агрегату 6 прошлых кадров, совпадают
    с кромками текущего кадра на 0.003-0.135 м (бины 30-65 м, doubleT_platform
    f100 и кривая roundT_pressureGate_roundT f100); неверный знак поворота
    дал бы на кривой систематику 0.3-0.5 м.

    Движение между кадрами ~1 м/кадр, поворот до ~0.5 град/кадр: за HISTORY
    кадров ошибка одометрии (сантиметры) на два порядка меньше бина 5 м и
    ширины окна GATE — кромкам стен не мешает.
    """

    def __init__(self, frames, poses, history):
        self.frames = frames
        self.poses = poses
        self.history = history

    def extra(self, i):
        """(fwd, lat, up) точек кадров [i-history, i) в системе кадра i."""
        if self.poses is None or i <= 0 or i >= len(self.poses):
            return None
        xi, yi, ti = (float(v) for v in self.poses[i])
        ci, si = np.cos(-ti), np.sin(-ti)
        parts = []
        for j in range(max(0, i - self.history), i):
            xj, yj, tj = (float(v) for v in self.poses[j])
            # p_i = R(t_j - t_i) p_j + R(-t_i) (t_j - t_i)
            dt = tj - ti
            c, s = np.cos(dt), np.sin(dt)
            gx, gy = xj - xi, yj - yi
            tx, ty = ci * gx - si * gy, si * gx + ci * gy
            p = np.asarray(self.frames[j][0], dtype=float)
            x = c * p[:, 0] - s * p[:, 1] + tx
            y = s * p[:, 0] + c * p[:, 1] + ty
            parts.append((-y, x, p[:, 2]))   # fwd=-y, lat=x, up=z
        if not parts:
            return None
        return tuple(np.concatenate([q[k] for q in parts]) for k in range(3))



def _frames(bag_dir, cache_size=2):
    from visualize_bag import BagFrames, find_db3
    return BagFrames(find_db3(bag_dir), cache_size=cache_size)


def main():
    ap = argparse.ArgumentParser(
        description='Ось прямого пути, стены тоннеля и габаритный коридор')
    ap.add_argument('bag_dir')
    ap.add_argument('--frame', type=int, default=0)
    ap.add_argument('--sweep', type=int, default=0,
                    help='пройти по бэгу с этим шагом кадров и дать сводку')
    ap.add_argument('--plot', nargs='?', const='', default=None, metavar='PNG',
                    help='вид сверху: без значения — окно, со значением — файл')
    ap.add_argument('--bar', action='store_true',
                    help='полоса статуса по дальности в консоли для всех кадров')
    ap.add_argument('--accum', nargs='?', type=int, const=0, default=None,
                    metavar='N',
                    help='накапливать коридор по кадрам через одометрию '
                         '(trajectory.npz): поднимает худший кадр. С --plot '
                         'N задаёт глубину истории и рисует две панели, '
                         'без накопления и с ним')
    ap.add_argument('--accum-pts', nargs='?', type=int, const=0, default=None,
                    metavar='N',
                    help='агрегировать ТОЧКИ последних N кадров (по умолчанию '
                         'HISTORY из accum) полным SE(2) из trajectory.npz в '
                         'оценку кромок стен: дальние бины кадра содержат '
                         '8-21 точку (шум перцентиля 22-35%), накопление даёт '
                         '70-190 и снижает шум кромки примерно втрое. '
                         'Препятствия не затрагиваются: агрегат идёт только '
                         'в walls.trace. Без trajectory.npz — предупреждение '
                         'и продолжение без агрегации')
    # Ломаная включена ПО УМОЛЧАНИЮ, как и в plot_axis: прямая ось блока 1
    # стоит на горстке наблюдений (roundT_doubleT f107/f121 — n_obs=4, колея
    # 1.36 м при номинале 1.52, то есть зацеплена чужая пара) и съезжает вбок,
    # а на повороте верной быть и не может. Полилиния на тех же кадрах даёт
    # 64-92 узла и колею 1.515-1.523 м. --no-polyline возвращает прямую.
    ap.add_argument('--polyline', action=argparse.BooleanOptionalAction,
                    default=True,
                    help='ось блока 1 — ЛОМАНАЯ по двум нитям полилинейной '
                         'детекции (rail_geometry/track_geometry) вместо '
                         'прямой; где нитей нет, откат на прямую. Включено по '
                         'умолчанию, --no-polyline выключает')
    ap.add_argument('--no-track', action='store_true',
                    help='выключить блок 3.5 (межкадровый трекинг находок): '
                         'сводка препятствий остаётся покадровой')
    ap.add_argument('--branches', action='store_true',
                    help='после основного тракта найти ВСЕ ветки пути в кадре '
                         '(многопутность, стрелки — guard/multitrack.py). '
                         'Дорого (считает стены кадра заново), поэтому только '
                         'по флагу и только для одиночного кадра, не в sweep')
    a = ap.parse_args()

    # Ключ бэга для тёплой сборки полилинии (кэши rail_geometry живут по
    # (db_path, frame_index)); при --polyline=False никем не читается.
    from visualize_bag import find_db3
    db = find_db3(a.bag_dir)

    # Агрегатор точек прошлых кадров. Предупреждение вместо отказа (в отличие
    # от --accum): doubleT_obstacle без trajectory.npz входит в обязательный
    # набор sweep-замеров, и отключать весь прогон из-за одного бэга нельзя.
    pagg = None
    pt_hist = 0
    pt_poses = None
    if a.accum_pts is not None:
        from .accum import HISTORY, load_poses
        pt_hist = a.accum_pts or HISTORY
        pt_poses = load_poses(a.bag_dir)
        if pt_poses is None:
            print(f'{a.bag_dir}: нет trajectory.npz — агрегация точек '
                  f'отключена.\n'
                  f'  посчитать: python odometry.py {a.bag_dir} --save')

    # Кэш кадров: при агрегации точек каждый шаг sweep читает N прошлых
    # кадров; доступ в сумме строго последовательный, поэтому кэша
    # глубина+шаг хватает, чтобы каждый кадр декодировался ровно один раз.
    frames = _frames(a.bag_dir,
                     cache_size=(pt_hist + (a.sweep or 0) + 2
                                 if a.accum_pts is not None else 2))
    if pt_poses is not None:
        pagg = _PointAggregator(frames, pt_poses, pt_hist)

    # Накопитель осмыслен только при последовательном проходе: одиночному
    # кадру копить нечего.
    acc = None
    if a.accum is not None:
        from .accum import CorridorAccumulator, HISTORY, load_poses
        poses = load_poses(a.bag_dir)
        if poses is None:
            print(f'{a.bag_dir}: нет trajectory.npz — накопление недоступно.\n'
                  f'  посчитать: python odometry.py {a.bag_dir} --save')
            return
        acc = CorridorAccumulator(poses, history=a.accum or HISTORY)

    # Блок 3.5: межкадровый трекинг находок блока 3. Позы — те же, что для
    # агрегации/накопления; без них трекер работает в системе датчика и сам
    # помечает это в сводке (world_ok=False).
    tracker = None
    if not a.no_track:
        from .accum import load_poses
        from .tracker import ObstacleTracker
        tracker = ObstacleTracker(pt_poses
                                  if pt_poses is not None
                                  else load_poses(a.bag_dir))

    if a.bar:
        from .draw import corridor_bar, corridor_line
        for i in range(0, len(frames), max(1, a.sweep or 1)):
            ex = pagg.extra(i) if pagg is not None else None
            axis, w = analyze(frames[i][0], extra=ex, polyline=a.polyline,
                              bag_key=db, frame_index=i)
            if acc is not None:
                w = acc.update(i, w)
            print(f'[{i:4d}] {corridor_bar(w)} {corridor_line(w, axis)}')
        return

    if a.sweep:
        reach, frame_reach, hl, hr, nax = [], [], [], [], 0
        bridged, holes, unc = [], 0, 0
        ob_hits, ob_gauge, ob_conf, ob_frames = 0, 0, 0, 0
        idx = range(0, len(frames), a.sweep)
        for i in idx:
            ex = pagg.extra(i) if pagg is not None else None
            axis, w = analyze(frames[i][0], extra=ex, polyline=a.polyline,
                              bag_key=db, frame_index=i)
            nax += bool(axis['measured'])
            bridged.append(w['reach_bridged'])
            holes += int((w['status'] == 'hole').sum())
            unc += int(w['uncovered'].sum())
            ob = w.get('obstacles')
            if ob is not None:
                hits, summ = ob
                ing = [h for h in hits if h['in_gauge']]
                ob_hits += summ['hits']
                ob_gauge += len(ing)
                ob_conf += sum(h['confirmed'] for h in ing)
                ob_frames += bool(ing)
                if tracker is not None:
                    # Трекуем только находки В ГАБАРИТЕ: объект вне коридора
                    # (человек B в doubleT_obstacle) треком не становится.
                    tracker.update(i, ing)
            if acc is not None:
                w = acc.update(i, w)
                frame_reach.append(w['reach_frame'])
            reach.append(w['reach'])
            ok = w['status'] == 'ok'
            hl += list(w['half_left'][ok])
            hr += list(w['half_right'][ok])
        reach = np.array(reach)
        print(f'{a.bag_dir}: {len(reach)} кадров'
              + ('  [накопление по одометрии]' if acc is not None else '')
              + ('  [агрегация точек]' if pagg is not None else ''))
        print(f'  ось измерена     : {nax}/{len(reach)}')
        if frame_reach:
            fr = np.array(frame_reach)
            print(f'  без накопления, м: med {np.median(fr):.0f}  '
                  f'p10 {np.percentile(fr, 10):.0f}  min {fr.min():.0f}')
        print(f'  подтверждено, м  : med {np.median(reach):.0f}  '
              f'p10 {np.percentile(reach, 10):.0f}  min {reach.min():.0f}  '
              f'max {reach.max():.0f}')
        br = np.array(bridged)
        print(f'  с мостами, м     : med {np.median(br):.0f}  '
              f'p10 {np.percentile(br, 10):.0f}  min {br.min():.0f}  '
              f'(покадрово, reach_bridged)')
        if holes:
            print(f'  дыры, бинов      : {holes}, из них не покрыты кольцами '
                  f'пола: {unc}')
        print(f'  коридор слева, м : med {np.median(hl):.2f}  min {np.min(hl):.2f}')
        print(f'  коридор справа, м: med {np.median(hr):.2f}  min {np.min(hr):.2f}')
        print(f'  препятствия (блок 3): находок всего {ob_hits}, '
              f'в габарите {ob_gauge} (подтверждённых {ob_conf}), '
              f'кадров с препятствием в габарите {ob_frames}')
        if tracker is not None:
            ts = tracker.summary()
            print(f'  трекинг (блок 3.5): треков {ts["tracks"]}, '
                  f'подтверждённых {ts["confirmed"]}, кадров с подтверждённым '
                  f'препятствием {ts["frames_confirmed"]}'
                  + ('' if ts['world_ok']
                     else '  [нет trajectory.npz: система датчика]'))
            lg = ts['longest']
            if lg is not None:
                print(f'  самый длинный трек: {lg["n"]} кадров '
                      f'({lg["first"]}..{lg["last"]}), последняя дальность '
                      f'{lg["d"]:.0f} м, lat {lg["lat_abs"]:+.2f} м, '
                      f'мир ({lg["world"][0]:+.1f}, {lg["world"][1]:+.1f})'
                      + ('' if lg['confirmed'] else '  [не подтверждён]'))
        return

    pts = frames[a.frame][0]
    ex = pagg.extra(a.frame) if pagg is not None else None
    axis, w = analyze(pts, extra=ex, polyline=a.polyline,
                      bag_key=db, frame_index=a.frame)

    w_acc = None
    if acc is not None:
        # Накопитель осмыслен только на последовательном проходе, поэтому
        # историю перед целевым кадром надо ПРОКРУТИТЬ — и для --plot, и для
        # консольного вывода. Одиночному кадру копить нечего: без прогрева
        # накопленный результат повторил бы одиночный.
        for j in range(max(0, a.frame - acc.history + 1), a.frame):
            ex_j = pagg.extra(j) if pagg is not None else None
            acc.update(j, analyze(frames[j][0], extra=ex_j, polyline=a.polyline,
                                     bag_key=db, frame_index=j)[1])
        w_acc = acc.update(a.frame, w)

    if a.plot is not None:
        from .draw import plot_top
        out = plot_top(pts, w, axis, path=a.plot or None,
                       title=f'{a.bag_dir} f{a.frame}', wall_accum=w_acc)
        if out:
            print(f'сохранено: {out}')
            return

    w_show = w_acc if w_acc is not None else w
    print(f'{a.bag_dir} кадр {a.frame}'
          + ('  [накопление по одометрии]' if w_acc is not None else '')
          + ('  [агрегация точек]' if pagg is not None else ''))
    print(f'  пол z = {w_show["floor"]:.2f} м')
    gauge_txt = (f'колея={axis["gauge"] * 1000:.0f} мм, '
                 if axis['gauge'] is not None else 'колея=— (чужая ветка), ')
    print(f'  ось: lat0={axis["lat0"]:+.2f} м, наклон={axis["slope"]:+.4f}, '
          + gauge_txt + f'срезов={axis["n_obs"]}'
          + ('' if axis['measured'] else '  [НЕ ИЗМЕРЕНА, взята нулевая]'))
    print(f'  габарит подтверждён до {w_show["reach"]:.0f} м '
          f'({w_show["n_ok"]} бинов измерено)'
          + (f'  [одиночный кадр: {w["reach"]:.0f} м]'
             if w_acc is not None else ''))
    print(f'  с мостами через короткие дыры (кадр): {w["reach_bridged"]:.0f} м')
    n_unc = int(w['uncovered'].sum())
    if n_unc:
        fm_unc = w['fwd_mid'][w['uncovered']]
        print(f'  бинов без покрытия кольцами пола: {n_unc} '
              f'({" ".join(f"{v:.0f}" for v in fm_unc)}) — слепые зоны '
              f'сенсора, НЕ "свободно" и НЕ "пусто"')

    st = w.get('structures')
    if st is not None:
        # Остаток необъяснённых точек (он же вход блока 3) по бинам.
        fwd, lat, up = to_frame(pts)
        h = up - w['floor']
        cand = (fwd >= 0) & (fwd < MAX_RANGE) & (h >= H_LO) & (h < H_HI)
        resid = cand & ~st['mask']
        nb = int(MAX_RANGE // BIN) + 1
        cnt = np.bincount(np.floor(fwd[resid] / BIN).astype(int),
                          minlength=nb)
        detail = ' '.join(f'{b * BIN:.0f}:{c}' for b, c in enumerate(cnt) if c)
        print(f'  конструкции: объяснено {st["n_struct"]} ячеек, '
              f'остаток {int(resid.sum())} точек'
              + (f'  (по бинам, м:точки — {detail})' if detail else ''))

    ob = w.get('obstacles')   # покадрово: накопленный коридор их не несёт
    if ob is not None:
        hits, summ = ob
        ing = [h for h in hits if h['in_gauge']]
        tr_by_hit = {}
        if tracker is not None and a.frame > 0:
            # Блок 3.5 для одиночного кадра: прокрутка истории (как прогрев
            # накопителя выше). Глубины MAX_MISS + CONFIRM_FRAMES хватает,
            # чтобы трек подтвердился и пережил допустимые пропуски.
            from .tracker import CONFIRM_FRAMES, MAX_MISS
            for j in range(max(0, a.frame - MAX_MISS - CONFIRM_FRAMES - 1),
                           a.frame):
                ex_j = pagg.extra(j) if pagg is not None else None
                ob_j = analyze(frames[j][0], extra=ex_j, polyline=a.polyline,
                                     bag_key=db, frame_index=j)[1].get('obstacles')
                if ob_j is not None:
                    tracker.update(j, [h for h in ob_j[0] if h['in_gauge']])
            live = tracker.update(a.frame, ing)
            for tr in live:
                for f_j, h_j in tr['hits']:
                    if f_j == a.frame:
                        tr_by_hit[id(h_j)] = tr
        if ing:
            near = min(ing, key=lambda h: h['d'])
            n_uncf = sum(1 for h in ing if not h['confirmed'])
            print(f'  препятствия: {len(ing)} в габарите, ближнее '
                  f'{near["d"]:.0f} м на x={near["lat"]:+.1f}, '
                  f'шxв {near["width"]:.1f}x{near["height"]:.1f} м'
                  + (f'  [{n_uncf} в бине "hole", не подтверждено]'
                     if n_uncf else ''))
            for h in ing:
                tr = tr_by_hit.get(id(h))
                if tr is not None:
                    print(f'    трек #{tr["id"]}: {tr["n"]} кадров, '
                          f'мир ({tr["world"][0]:+.1f}, {tr["world"][1]:+.1f})'
                          f' — {"подтверждён" if tr["confirmed"] else "НЕ подтверждён (однокадровая находка)"}')
        else:
            print(f'  препятствий в габарите нет '
                  f'(подтверждено до {w_show["reach"]:.0f} м)')

    if a.branches:
        # Многопутность/стрелки — отдельный дорогой блок (стены кадра
        # считаются заново в check_wall), поэтому только по флагу и только
        # покадрово, в sweep не включён.
        from .multitrack import detect_divergence, _fmt_branch
        fwd, lat, up = to_frame(pts)
        det = detect_divergence(fwd, lat, up, w['floor'], axis=axis)
        print(f'  ветки пути: diverged={det["diverged"]}, '
              f'веток {len(det["branches"])}')
        for bi, b in enumerate(det['branches']):
            print(_fmt_branch(bi, b))

    print()
    print('   м   статус     стена L   стена R   коридор L  коридор R')
    unc_mask = w_show.get('uncovered')
    for i, fm in enumerate(w_show['fwd_mid']):
        st_i = w_show['status'][i]
        if st_i == 'stop':
            continue
        if unc_mask is not None and unc_mask[i]:
            st_i = 'uncov'     # дыра без покрытия кольцами пола (слепая зона)
        print(f'{fm:6.1f}   {st_i:6s}  {w_show["left"][i]:8.2f}  '
              f'{w_show["right"][i]:8.2f}'
              f'   {w_show["half_left"][i]:8.2f}   '
              f'{w_show["half_right"][i]:8.2f}')
    tail = w_show['fwd_mid'][np.array([s == 'stop' for s in w_show['status']])]
    if len(tail):
        print(f'\nдальше {tail[0] - BIN / 2:.0f} м — не наблюдается '
              f'(данных недостаточно, НЕ "свободно")')



if __name__ == '__main__':
    main()
