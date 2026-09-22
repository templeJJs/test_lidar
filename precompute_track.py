"""Посчитать геометрию пути заранее -- чтобы /track не собирал её в запросе.

Зачем (замерено до этого): `/track` нового кадра -- 0.37 с (первый 1.65 с),
`/frame?mode=zones` нового кадра -- 0.39-0.48 с, а воспроизведение просит 10
кадров в секунду (бюджет 0.1 с/кадр). Сервер успевал 2-2.5 кадра/с, очередь
предзагрузки клиента не рассасывалась, ядро было занято сборкой постоянно.
Готовый кадр сервер отдаёт чтением gzip (~10 мс) -- см. `web_viewer.TrackStore`.

Как устроено: скрипт проходит записи и кадры (0, 1, 2, ... -- как при
воспроизведении) и кладёт в хранилище ровно те байты, которые вернул бы /track
при сборке на месте: сериализованный JSON (`TrackGeometry.compute_bytes`), сжатый
gzip. Файл: `<каталог>/<запись>/<кадр>.json.gz`, каталог по умолчанию
`.cache/track` (в .gitignore), он же у сервера (`--track-cache`).

Порядок кадров ВНУТРИ записи важен: у build_track есть тёплая (инкрементальная)
сборка и связь решений соседних кадров (track_geometry: `warm`, LINK_*), поэтому
кадр, посчитанный сразу за пропущенным (уже лежащим) кадром, выйдет не таким,
как в непрерывном проходе. Для полного совпадения с воспроизведением считайте
хранилище одним непрерывным проходом (или `--force`).

    python precompute_track.py for_hackathon                # все записи, 1 поток
    python precompute_track.py for_hackathon --jobs 3       # процесс на запись
    python precompute_track.py for_hackathon --bag roundT_doubleT
    python precompute_track.py for_hackathon --force        # пересчитать всё

Оценка времени: 0.15-0.6 с на кадр, 2488 кадров всех шести записей -- примерно
15 минут в один поток (`--jobs 3` втрое быстрее, память -- по процессу на запись,
внутри build_track свой `BagFrames` с кэшем на 2 кадра). Уже посчитанные кадры по
умолчанию пропускаются: повторный запуск -- это только проверка покрытия.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

from web_viewer import TRACK_CACHE_DIR, BagCatalog, TrackGeometry, TrackStore


def human(size):
    """Размер в удобных единицах (МБ/ГБ)."""
    for unit in ('Б', 'КБ', 'МБ', 'ГБ'):
        if size < 1024 or unit == 'ГБ':
            return f'{size:.1f} {unit}' if unit != 'Б' else f'{size:.0f} Б'
        size /= 1024.0


def precompute_bag(task):
    """Посчитать все кадры ОДНОЙ записи (один процесс -- одна запись).

    Внутри записи кадры идут по порядку и в одном процессе: только так тёплая
    сборка и связь кадров видят то же, что воспроизведение. Кадры, которые уже
    есть в хранилище, не считаются (idempotent); `--force` пересчитывает.
    """
    name, db_path, total, cache_dir, force, every = task
    store = TrackStore(cache_dir)
    # store=None: сборка на месте, кэш в памяти на 2 кадра (больше не нужно --
    # каждый кадр считается один раз), поэтому память на процесс небольшая.
    track = TrackGeometry(name, cache_frames=2, store=None)
    done = skipped = errors = 0
    first_errors = []
    t0 = time.time()
    for frame in range(total):
        if not force and store.has(name, frame):
            skipped += 1
            continue
        try:
            body = track.compute_bytes(name, frame, db_path)
        except Exception as exc:  # noqa: BLE001 -- один плохой кадр не должен
            errors += 1            # останавливать проход по записи
            if len(first_errors) < 5:
                first_errors.append(f'кадр {frame}: {exc!r}')
            continue
        store.put(name, frame, body)
        done += 1
        if done % every == 0:
            spent = time.time() - t0
            print(f'  [{name}] {frame + 1}/{total}: посчитано {done}, '
                  f'пропущено {skipped}, ошибок {errors}, '
                  f'{spent:.0f} с ({spent / max(done, 1):.2f} с/кадр)', flush=True)
    spent = time.time() - t0
    for line in first_errors:
        print(f'  ! [{name}] {line}', flush=True)
    print(f'  [{name}] готово: {total} кадров -- посчитано {done}, '
          f'пропущено {skipped}, ошибок {errors}, {spent:.1f} с', flush=True)
    return {'bag': name, 'frames': total, 'done': done, 'skipped': skipped,
            'errors': errors, 'seconds': spent, 'first_errors': first_errors}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Посчитать геометрию пути (payload /track) для всех кадров '
                    'записей и положить её в хранилище, из которого сервер отдаёт '
                    'её без сборки')
    parser.add_argument('root', nargs='?', default='for_hackathon',
                        help='папка одной записи (внутри *.db3) или корень с '
                             'папками записей (по умолчанию for_hackathon)')
    parser.add_argument('--bag', action='extend', nargs='+', default=None,
                        metavar='NAME',
                        help='считать только эти записи (по умолчанию -- все; '
                             'можно `--bag a b` или `--bag a --bag b`)')
    parser.add_argument('--jobs', type=int, default=1,
                        help='сколько записей считать параллельно (по умолчанию 1: '
                             'один процесс на запись; 3 ускоряет втрое, но и '
                             'память втрое)')
    parser.add_argument('--force', action='store_true',
                        help='пересчитывать кадры, которые уже есть в хранилище')
    parser.add_argument('--cache', default=TRACK_CACHE_DIR,
                        help=f'каталог хранилища (по умолчанию {TRACK_CACHE_DIR}); '
                             'должен совпадать с `--track-cache` сервера')
    parser.add_argument('--progress-every', type=int, default=25,
                        help='печатать прогресс каждые N посчитанных кадров записи '
                             '(по умолчанию 25)')
    args = parser.parse_args(argv)

    if not os.path.isdir(args.root) and not os.path.isfile(args.root):
        raise SystemExit(f'{args.root}: ни папки, ни файла')

    # Каталог записей берём тот же, что у сервера (BagCatalog._scan): имена записей
    # -- имена папок, и по ним же раскладывается хранилище.
    class _Args:
        bag = None
        axis_x = None

    try:
        catalog = BagCatalog(args.root, _Args())
    except FileNotFoundError as exc:
        raise SystemExit(str(exc))

    wanted = args.bag
    names = catalog.names
    if wanted:
        unknown = [n for n in wanted if n not in names]
        if unknown:
            raise SystemExit(f'нет таких записей: {", ".join(unknown)}; доступны: '
                             f'{", ".join(names)}')
        names = [n for n in names if n in wanted]

    tasks = [(name, catalog.paths[name], catalog.frames(name), args.cache,
              bool(args.force), max(1, int(args.progress_every))) for name in names]
    jobs = max(1, min(int(args.jobs), len(tasks)))
    print(f'Записей {len(tasks)}, кадров {sum(t[2] for t in tasks)}, потоков {jobs}; '
          f'хранилище {os.path.abspath(args.cache)}'
          + (', режим --force (пересчёт всего)' if args.force else ''), flush=True)

    t0 = time.time()
    results = []
    if jobs == 1:
        for task in tasks:
            results.append(precompute_bag(task))
    else:
        # Процесс на запись: своя память и свой разбор кадров, общий только диск.
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            for result in pool.map(precompute_bag, tasks):
                results.append(result)
    spent = time.time() - t0

    done = sum(r['done'] for r in results)
    skipped = sum(r['skipped'] for r in results)
    errors = sum(r['errors'] for r in results)
    frames = sum(r['frames'] for r in results)
    store = TrackStore(args.cache)
    size = store.size()
    rate = f'{spent / done:.2f} с на посчитанный кадр' if done else 'считать было нечего'
    print(f'\nИтог: кадров {frames} -- посчитано {done}, пропущено {skipped}, '
          f'ошибок {errors}; время {spent:.1f} с ({rate}); '
          f'в хранилище {store.counts()[0]} кадров по {store.counts()[1]} записям, '
          f'на диске {human(size)}', flush=True)
    if errors:
        for r in results:
            for line in r['first_errors']:
                print(f'  ! [{r["bag"]}] {line}', flush=True)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())