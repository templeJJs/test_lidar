"""Кадры, где блок 3 что-то нашёл, свёрнутые в ЭПИЗОДЫ — заготовка разметки.

Зачем. Покадровая выдача блока 3 для разметки не годится: настоящий объект
живёт десятки кадров подряд (человек A в doubleT_obstacle — 201 кадр из 201),
и список находок даёт двести строк про один и тот же объект. Размечать надо
объекты, а не кадры, поэтому здесь находки склеиваются В ТОТ ЖЕ трек, что
использует блок 3.5 (tracker.ObstacleTracker) — тот самый, что уже отсекает
однокадровые транзиенты. Новой логики связывания тут нет и быть не должно:
разметка обязана видеть ровно то, что видит детектор.

Эпизод — один трек: интервал кадров, в которых объект найден, плюс сводка
по его находкам (дальность от/до, поперёк, габариты, число точек). Одна
строка — один объект, который надо разметить или отбросить.

    # что вообще нашлось в бэге (каждый 5-й кадр)
    python -m guard.metrics.episodes for_hackathon/doubleT_obstacle --step 5

    # только то, что живёт >= 10 кадров: короткие мерцания не показывать
    python -m guard.metrics.episodes for_hackathon/doubleT_obstacle --min-frames 10

    # всё, включая находки МИМО габарита (по умолчанию только в габарите)
    python -m guard.metrics.episodes for_hackathon/cloud_with_fake_obj --any-gauge

    # для разметки: список кадров каждого эпизода в JSON
    python -m guard.metrics.episodes for_hackathon/doubleT_obstacle -o ep.json
    python -m guard.metrics.episodes for_hackathon/doubleT_obstacle -o ep.csv

Фильтры (все — ПО ЭПИЗОДУ, а не по кадру):
    --min-frames N   кадров с находкой в эпизоде; главный фильтр мусора
    --confirmed      только подтверждённые трекером (блок 3.5)
    --min-points N   пик n_pts по эпизоду: отсечь совсем редкие сгущения
    --max-range D    эпизоды, чья ближняя дальность дальше D, не показывать

ВАЖНО про --step. Трекер считает пропуски в ОБРАБОТАННЫХ кадрах (MAX_MISS),
а не в кадрах бэга: при --step 5 один пропуск — это 5 кадров бэга, и гейт
связывания растягивается соответственно (в нём стоит ход поезда между
обработанными кадрами). Это штатно, но чем крупнее шаг, тем охотнее склеятся
два разных объекта. Для разметки надёжнее --step 1, для быстрой разведки
бэга — 5-10.
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

# Пакет импортирует visualize_bag из корня проекта — как guard.run.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))


def collect(bag_dir, step=1, beg=0, end=None, in_gauge_only=True,
            polyline=True, progress=None):
    """Прогнать тракт по кадрам бэга -> (эпизоды, статистика прогона).

    Возвращает список эпизодов (см. _episode) и dict со счётчиками прогона:
    сколько кадров обработано, в скольких была находка, были ли позы.

    in_gauge_only — подавать трекеру только находки В ГАБАРИТЕ, как это
    делает run.py --sweep: объект вне коридора треком не становится. С
    False в эпизоды попадут и находки мимо габарита (поле in_gauge у
    эпизода покажет, какие именно).
    """
    from ..run import _frames, analyze
    from ..accum import load_poses
    from ..tracker import ObstacleTracker
    from visualize_bag import find_db3

    frames = _frames(bag_dir)
    db = find_db3(bag_dir)
    poses = load_poses(bag_dir)
    trk = ObstacleTracker(poses)

    end = len(frames) if end is None else min(end, len(frames))
    idx = list(range(max(0, beg), end, max(1, step)))
    n_hit_frames = 0
    for n, i in enumerate(idx):
        axis, w = analyze(frames[i][0], polyline=polyline,
                          bag_key=db, frame_index=i)
        ob = w.get('obstacles')
        hits = [] if not ob else ob[0]
        if in_gauge_only:
            hits = [h for h in hits if h['in_gauge']]
        n_hit_frames += bool(hits)
        trk.update(i, hits)
        if progress is not None and n % 20 == 0:
            progress(n + 1, len(idx))

    stats = dict(frames=len(idx), frames_with_hit=n_hit_frames,
                 step=step, world_ok=trk.world_ok, bag=bag_dir)
    return [_episode(t) for t in trk.tracks], stats


def _episode(tr):
    """Трек -> строка разметки: интервал кадров + сводка по его находкам.

    Дальность берётся от/до, а не средняя: у едущего поезда объект
    приближается, и «дальность эпизода» одним числом — это ложь. Для
    сортировки и фильтра по дальности используется d_near (ближайшая) —
    именно она определяет, насколько эпизод опасен.
    """
    fr = [f for f, _ in tr['hits']]
    hs = [h for _, h in tr['hits']]
    ds = [h['d'] for h in hs]
    return dict(
        id=tr['id'],
        first=int(min(fr)), last=int(max(fr)), n_frames=len(fr),
        frames=[int(f) for f in fr],
        # Длина интервала в кадрах бэга: при --step > 1 она больше n_frames,
        # и путать их нельзя — размечать придётся весь интервал.
        span=int(max(fr) - min(fr) + 1),
        confirmed=bool(tr['confirmed']),
        d_first=float(ds[0]), d_last=float(ds[-1]),
        d_near=float(min(ds)), d_far=float(max(ds)),
        lat_med=float(np.median([h['lat_abs'] for h in hs])),
        width_max=float(max(h['width'] for h in hs)),
        height_max=float(max(h['height'] for h in hs)),
        n_pts_max=int(max(h['n_pts'] for h in hs)),
        in_gauge=bool(any(h['in_gauge'] for h in hs)),
        # Подтверждение БИНА (obstacles.confirmed) — не то же, что
        # подтверждение ТРЕКА выше: бин мог быть 'hole' во всех кадрах.
        bin_confirmed=bool(any(h['confirmed'] for h in hs)),
    )


def select(eps, min_frames=1, confirmed=False, min_points=0, max_range=None):
    """Фильтры по эпизоду. Порядок вывода — по первому кадру: так список
    читается вдоль бэга, как его и размечают."""
    out = [e for e in eps
           if e['n_frames'] >= min_frames
           and e['n_pts_max'] >= min_points
           and (not confirmed or e['confirmed'])
           and (max_range is None or e['d_near'] <= max_range)]
    return sorted(out, key=lambda e: (e['first'], e['d_near']))


def _fmt_frames(e, limit=12):
    """Кадры эпизода строкой. Ряд с ПОСТОЯННЫМ шагом сворачивается в
    'a..b' (или 'a..b/шаг'): при --step 5 сплошного ряда не бывает вовсе,
    а у объекта на 200 кадров список кадров нечитаем и не нужен. Дыры
    показываются как есть — по ним и видно, что объект терялся."""
    fr = e['frames']
    if len(fr) > 2:
        d = set(np.diff(fr))
        if len(d) == 1:
            s = d.pop()
            return f'{fr[0]}..{fr[-1]}' + ('' if s == 1 else f'/{s}')
    if len(fr) > limit:
        return ' '.join(str(f) for f in fr[:limit]) + f' …(+{len(fr) - limit})'
    return ' '.join(str(f) for f in fr)


def report(eps, stats, show_frames=True):
    """Таблица в консоль."""
    print(f'{stats["bag"]}: обработано {stats["frames"]} кадров '
          f'(шаг {stats["step"]}), с находкой {stats["frames_with_hit"]}'
          + ('' if stats['world_ok']
             else '   [нет trajectory.npz: трекинг в системе датчика]'))
    if not eps:
        print('эпизодов нет')
        return
    print(f'{"id":>4s} {"кадры":>13s} {"n":>4s} {"дальн, м":>13s} '
          f'{"lat":>6s} {"шир":>5s} {"выс":>5s} {"точек":>6s} '
          f'{"габ":>4s} {"подт":>5s}')
    for e in eps:
        print(f'{e["id"]:4d} {e["first"]:6d}..{e["last"]:<6d} {e["n_frames"]:4d} '
              f'{e["d_first"]:6.1f}->{e["d_last"]:<6.1f} '
              f'{e["lat_med"]:+6.2f} {e["width_max"]:5.2f} {e["height_max"]:5.2f} '
              f'{e["n_pts_max"]:6d} '
              f'{"ДА" if e["in_gauge"] else "нет":>4s} '
              f'{"да" if e["confirmed"] else "НЕТ":>5s}')
        if show_frames:
            print(f'     кадры: {_fmt_frames(e)}')
    n_conf = sum(e['confirmed'] for e in eps)
    print(f'итого эпизодов {len(eps)}, подтверждённых {n_conf}')


def save(eps, stats, path):
    """JSON или CSV — по расширению. JSON несёт список кадров эпизода (он и
    нужен разметке), CSV — одну строку на эпизод для таблицы."""
    if path.endswith('.csv'):
        cols = ['id', 'first', 'last', 'n_frames', 'span', 'confirmed',
                'bin_confirmed', 'in_gauge', 'd_first', 'd_last', 'd_near',
                'd_far', 'lat_med', 'width_max', 'height_max', 'n_pts_max']
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
            w.writeheader()
            for e in eps:
                w.writerow(e)
    else:
        with open(path, 'w') as f:
            json.dump(dict(stats=stats, episodes=eps), f,
                      ensure_ascii=False, indent=1)
    print(f'сохранено: {path}  ({len(eps)} эпизодов)')


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split('\n\n')[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='\n'.join(__doc__.split('\n')[14:]))
    ap.add_argument('bag_dir', help='папка бэга')
    ap.add_argument('--step', type=int, default=1,
                    help='обрабатывать каждый N-й кадр (по умолчанию 1). '
                         'Крупный шаг быстрее, но охотнее склеивает разные '
                         'объекты в один эпизод — см. описание модуля')
    ap.add_argument('--range-frames', metavar='A:B', default='',
                    help='ограничить прогон кадрами A..B')
    ap.add_argument('--min-frames', type=int, default=2,
                    help='показывать эпизоды длиной от N кадров с находкой '
                         '(по умолчанию 2: однокадровые — транзиенты, см. '
                         'tracker.CONFIRM_FRAMES). 1 — показать всё')
    ap.add_argument('--confirmed', action='store_true',
                    help='только эпизоды, подтверждённые трекером (блок 3.5)')
    ap.add_argument('--min-points', type=int, default=0,
                    help='пик n_pts по эпизоду не меньше N')
    ap.add_argument('--max-range', type=float, default=None,
                    help='только эпизоды, подошедшие ближе D метров')
    ap.add_argument('--any-gauge', action='store_true',
                    help='учитывать и находки МИМО габарита (по умолчанию '
                         'только в габарите, как в run.py --sweep)')
    ap.add_argument('--no-frames', action='store_true',
                    help='не печатать список кадров под каждым эпизодом')
    ap.add_argument('--no-polyline', action='store_true',
                    help='ось блока 1 — прямая вместо ломаной')
    ap.add_argument('-o', '--out', metavar='FILE',
                    help='сохранить в .json (со списком кадров) или .csv')
    a = ap.parse_args()

    beg, end = 0, None
    if a.range_frames:
        parts = (a.range_frames.split(':') + [''])[:2]
        beg = int(parts[0]) if parts[0] else 0
        end = int(parts[1]) if parts[1] else None

    def progress(n, total):
        print(f'\r{n}/{total} кадров ', end='', flush=True)

    eps, stats = collect(a.bag_dir, step=a.step, beg=beg, end=end,
                         in_gauge_only=not a.any_gauge,
                         polyline=not a.no_polyline, progress=progress)
    print('\r' + ' ' * 40 + '\r', end='')

    eps = select(eps, min_frames=a.min_frames, confirmed=a.confirmed,
                 min_points=a.min_points, max_range=a.max_range)
    report(eps, stats, show_frames=not a.no_frames)
    if a.out:
        save(eps, stats, a.out)


if __name__ == '__main__':
    main()
