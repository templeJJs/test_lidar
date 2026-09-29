"""Предрасчёт guard-тракта для веб-вьюера (web_viewer.py, маршрут /guard).

Зачем отдельный процесс и готовый JSON: analyze() стоит ~0.4 с на кадр —
на лету из HTTP-обработчика дорого; а импортировать pavel/ из процесса
вьюера нельзя: в корне проекта лежат свои rail_detection.py /
track_geometry.py, и имена модулей пересекаются. Поэтому сервер только
ЧИТАЕТ готовые файлы, а считает их эта команда (запуск из каталога pavel):

    cd pavel
    python -m guard.export_web ../for_hackathon/doubleT_platform
    python -m guard.export_web ../for_hackathon/doubleT_platform --end 10

Результат: pavel/guard_web/<запись>/frame_NNNN.json — коридор бинов, стены,
ось и препятствия кадра в системе guard (fwd=-y, lat=x, up=z — система
лидара). Перевод в систему вьюера (x=lat, y=-fwd, z=up) делает клиент
(web/guardlayer.js), сервер координаты не трогает.

Индексы точек находок (obstacles[i].point_idx) отдаются в системе ПОДГОТОВЛЕННОГО
кадра вьюера (/frame): detect() работает на сыром кадре, а вьюер перед отдачей
обрезает облако (bag_reader.trim: y >= -max_dist, y <= behind) и при превышении
max_points прореживает его шагом. Здесь та же подготовка повторяется
(параметры --viewer-* должны совпадать с запуском web_viewer.py), и сырые
индексы переводятся в индексы подготовленного кадра. Если сработало
прореживание (step > 1), отображения нет — point_idx опускается, и клиент
рисует только боксы. Поле frame_npts — число точек подготовленного кадра:
клиент сверяет его со своим облаком и при расхождении точки не красит.

Имя записи — имя папки бэга: так же его выводит web_viewer (bag_name), и
маршрут /guard?bag=...&frame=N ищет ровно этот файл.

Блок 3.5 (трекер): по ходу предрасчёта находки в габарите скармливаются
ObstacleTracker (как sweep в run.py), и каждая находка в JSON получает поля
track_id / track_n / track_confirmed. Без них веб красил тревогу по in_gauge
одиночного кадра, и однокадровый транзиент (замер tracker.py: пыль живёт
ровно 1 кадр, настоящий объект — каждый) мигал красным. Отдельно каждая
находка получает beyond_reach = not confirmed (уровень находки): трекер
подтверждает и находки за reach, и без флага веб красил их в красную тревогу. При --start>0 трекер
прогревается на MAX_MISS+CONFIRM_FRAMES+1 кадров назад (как run.py для
одиночного кадра); кадры диапазона, уже лежащие в JSON (перезапуск без
--force), трекеру скармливаются из готовых файлов, чтобы трек не рвался на
границе пересчитанного куска. --no-track выключает блок (поля не пишутся,
клиент ведёт себя как со старым кэшем).
"""

import argparse
import json
import math
import os

import numpy as np

from .run import analyze, _frames

# pavel/ — родитель пакета guard.
_PAVEL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _clean(v):
    """numpy -> python, NaN/inf -> None: браузерный JSON NaN не держит."""
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, np.ndarray):
        return _clean(v.tolist())
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return f if math.isfinite(f) else None
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def _viewer_map(pts, max_dist, behind, max_points):
    """Индексы сырого кадра -> индексы кадра, который отдаёт /frame вьюера.

    Повторяет BagViewer._prepare (web_viewer.py): trim по Y и прореживание
    шагом при n > max_points. Возвращает (kept, step): kept — индексы сырого
    кадра, дошедшие до вьюера (уже с учётом прореживания); step — шаг
    прореживания (1 = без него).
    """
    y = np.asarray(pts)[:, 1]
    kept = np.flatnonzero((y >= -max_dist) & (y <= behind))
    step = 1
    if 0 < max_points < len(kept):
        step = -(-len(kept) // max_points)
        kept = kept[::step]
    return kept, step


def frame_payload(bag_name, idx, pts, bag_key=None,
                  viewer_max_dist=200.0, viewer_behind=5.0,
                  viewer_max_points=400000, tracker=None,
                  extra=None):
    """Кадр -> словарь для JSON (система guard, см. заголовок файла).

    Ось блока 1 — ЛОМАНАЯ (polyline=True), как в CLI (run.py по умолчанию):
    веб должен получать ту же ось, что консоль. С прямой осью веб показывал
    физически невозможный излом линии пути на границе ближней зоны (замер
    f876 cloud_with_fake_obj: вторая разность оси бинов 0.77 м на 22.5 м,
    R = 33 м; с полилинией R = 223 м). bag_key/frame_index — ключи тёплой
    сборки полилинии (те же, что передаёт run.py).

    tracker — ObstacleTracker, уже прогретый на прошлых кадрах (или None,
    тогда полей track_* в находках нет). Трекаем только находки В ГАБАРИТЕ —
    тот же отбор, что в run.py: объект вне коридора треком не становится.
    Связь находка -> трек — по identity словаря (тот же приём, что
    run.py:418-430): трекер хранит сами словари находок в tr['hits'].
    extra — агрегированные точки прошлых кадров в системе текущего (тот же
    extra, что sweep в run.py даёт из _PointAggregator): идёт только в оценку
    кромок стен, как в тракте (движущееся в нём не размазывается в
    препятствие — в structures/препятствия extra не попадает).
    """
    axis, wall = analyze(pts, polyline=True, bag_key=bag_key, frame_index=idx,
                         extra=extra)

    kept, step = _viewer_map(pts, viewer_max_dist, viewer_behind,
                             viewer_max_points)

    far = axis.get('far')
    axis_far = None
    if far is not None:
        axis_far = {
            'fwd_mid': far['fwd_mid'],
            'lat': far['lat'],
            'status': list(far['status']),
        }

    # Режимы ведения оси по бинам (блок 1в, wall_rules.build): параллельные
    # массивы к bins.fwd_mid. Биновая сетка оси совпадает с сеткой коридора
    # (обе BIN из walls.py, центры lo + BIN/2), но список оси может быть
    # короче (ось кончается по HOLD_MAX) — там None. После переключения на
    # прямую ('switched' в walls.trace) режимы недействительны: axis_mode там
    # пуст, и его пустота здесь первична над режимом самого Bin.
    axis_bins = axis.get('bins') or []
    by_d = {round(b.d, 3): b for b in axis_bins}
    axis_mode = wall.get('axis_mode')
    b_mode, b_side, b_tight, b_off = [], [], [], []
    for i, f in enumerate(wall['fwd_mid']):
        b = by_d.get(round(float(f), 3))
        if axis_mode is not None and i < len(axis_mode):
            m = axis_mode[i] or None
        else:
            m = b.mode if b is not None else None
        if b is None or not m:
            b_mode.append(None)
            b_side.append(None)
            b_tight.append(None)
            b_off.append(None)
            continue
        b_mode.append(m)
        b_side.append('left' if b.side < 0 else ('right' if b.side > 0 else None))
        b_tight.append(bool(b.tight))
        b_off.append(b.off)

    ob_hits = []
    ob = wall.get('obstacles')
    tr_by_hit = {}
    if tracker is not None:
        # Блок 3.5 по ходу предрасчёта: трекеру — только находки в габарите
        # (отбор run.py), кадры идут последовательно, как в sweep.
        ing = [h for h in (ob[0] if ob is not None else []) if h['in_gauge']]
        for tr in tracker.update(idx, ing):
            for f_j, h_j in tr['hits']:
                if f_j == idx:
                    tr_by_hit[id(h_j)] = tr
    if ob is not None:
        hits, _summary = ob
        for h in hits:
            # x_min/x_max в detect() даны ОТНОСИТЕЛЬНО центра коридора бина;
            # центр = lat_abs - lat, переводим в абсолютную поперечную ось —
            # клиенту не нужно знать про center_bins.
            center = h['lat_abs'] - h['lat']
            hit = {
                'd': h['d'],
                'lat': h['lat_abs'],
                'x_min': h['x_min'] + center,
                'x_max': h['x_max'] + center,
                'width': h['width'],
                'height': h['height'],
                'n_pts': h['n_pts'],
                'in_gauge': h['in_gauge'],
                'confirmed': h['confirmed'],
                # Находка за пределами подтверждённого коридора: флаг — это
                # инверсия confirmed уровня находки (obstacles.py:722:
                # confirmed = бин измерен И не дальше reach), т.е. новых
                # порогов нет. Трекер подтверждает и такие находки чисто по
                # счётчику кадров (tracker.py: confirmed = n >=
                # CONFIRM_FRAMES), обходя правило reach — без этого поля
                # клиент красил их в красную тревогу на 112-125 м при reach
                # 60-90 м (замер по кэшу cloud_with_fake_obj: ~1.6% кадров).
                # Клиент (guardlayer.js) показывает их отдельным янтарным
                # уровнем «за reach»: раннее предупреждение сохраняется
                # (трек на 112 м верифицирован близким проходом через 30
                # кадров), но это не подтверждённая тревога.
                'beyond_reach': not h['confirmed'],
            }
            tr = tr_by_hit.get(id(h))
            if tr is not None:
                # Подтверждение треком (CONFIRM_FRAMES=2, tracker.py):
                # транзиент живёт 1 кадр, настоящий объект — каждый. Поля
                # аддитивны: старый клиент их не читает и работает как раньше.
                hit['track_id'] = tr['id']
                hit['track_n'] = tr['n']
                hit['track_confirmed'] = bool(tr['confirmed'])
            elif tracker is not None:
                # Трекер был, но находку не взял (дедуп двойников DUP_FWD/
                # DUP_LAT, tracker.py) — это НЕ старый формат без полей, а
                # неподтверждённая находка. Без явного False клиент проваливал
                # её в 'hit' и красил красной тревогой (замер f744: «треком
                # подтверждено 0» при красной «⚠ ПРЕПЯТСТВИЕ»).
                hit['track_confirmed'] = False
            # Индексы точек находки — в системе кадра вьюера (см. заголовок).
            # Без прореживания (step == 1) отображение сырых индексов точное.
            raw_idx = h.get('point_idx')
            if raw_idx is not None and step == 1 and len(raw_idx) and len(kept):
                vi = np.searchsorted(kept, np.asarray(raw_idx))
                ok = (vi < len(kept)) & (kept[np.minimum(vi, len(kept) - 1)]
                                         == np.asarray(raw_idx))
                hit['point_idx'] = np.asarray(vi[ok], dtype=np.int64)
            ob_hits.append(hit)

    return {
        'format': 'guard-web/1',
        'bag': bag_name,
        'frame': int(idx),
        'coords': 'guard: fwd=-y, lat=x, up=z (система лидара кадра)',
        'frame_npts': int(len(kept)),   # точек в кадре /frame вьюера (см. шапку)
        'floor': wall['floor'],
        'reach': wall['reach'],
        'reach_bridged': wall['reach_bridged'],
        'n_ok': wall['n_ok'],
        'axis': {
            'lat0': axis['lat0'],
            'slope': axis['slope'],
            'gauge': axis['gauge'],
            'measured': bool(axis['measured']),
        },
        'axis_far': axis_far,
        'bins': {
            'fwd_mid': wall['fwd_mid'],
            # абсолютная поперечина ведомой оси бина: стена = axis + left/right
            'axis': wall['axis'],
            'left': wall['left'],
            'right': wall['right'],
            'half_left': wall['half_left'],
            'half_right': wall['half_right'],
            'status': list(wall['status']),
            'lead': list(wall['lead']),
            'uncovered': wall['uncovered'],
            # режим ведения оси в бине (блок 1в): 'rails'|'center'|'lead'|'hold'
            # или None, где режима нет; side — ведущая сторона LEAD
            # ('left'/'right'), tight — упор в стену, off — отступ стена->путь
            'mode': b_mode,
            'side': b_side,
            'tight': b_tight,
            'off': b_off,
        },
        'obstacles': ob_hits,
    }


def main():
    ap = argparse.ArgumentParser(
        description='Предрасчёт guard-тракта в JSON для веб-вьюера '
                    '(web_viewer.py /guard, web/guardlayer.js)')
    ap.add_argument('bag_dir', help='папка записи (внутри *.db3)')
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--end', type=int, default=0,
                    help='по какой кадр (не включительно); 0 — все кадры')
    ap.add_argument('--step', type=int, default=1)
    ap.add_argument('--out', default=None,
                    help='куда писать (по умолчанию pavel/guard_web/<запись>)')
    ap.add_argument('--force', action='store_true',
                    help='пересчитать кадры, для которых файл уже есть')
    ap.add_argument('--viewer-max-dist', type=float, default=200.0,
                    help='--max-dist web_viewer.py: обрезка кадра вперёд, м '
                         '(нужна для перевода point_idx в индексы /frame)')
    ap.add_argument('--viewer-behind', type=float, default=5.0,
                    help='--behind web_viewer.py: обрезка кадра за спиной, м')
    ap.add_argument('--viewer-max-points', type=int, default=400000,
                    help='--max-points web_viewer.py: если кадр прорежен, '
                         'point_idx не отображаются и в JSON не пишутся')
    ap.add_argument('--no-track', action='store_true',
                    help='выключить блок 3.5 (трекер): поля track_* в JSON '
                         'не пишутся, веб работает как со старым кэшем')
    ap.add_argument('--no-temporal', action='store_true',
                    help='выключить темпоральный фон и накопление остатка '
                         '(блок 3, sweep-режим): кэш станет покадровым, как '
                         'одиночный --frame в run.py')
    a = ap.parse_args()

    bag_dir = a.bag_dir.rstrip('/\\')
    bag_name = os.path.basename(os.path.abspath(bag_dir))
    out_dir = a.out or os.path.join(_PAVEL_DIR, 'guard_web', bag_name)
    os.makedirs(out_dir, exist_ok=True)

    # Агрегация точек прошлых кадров в оценку кромок стен (extra= в analyze),
    # как --accum-pts у run.py: без неё стены в кэше считались слабее, чем в
    # sweep. При отсутствии trajectory.npz — предупреждение, не отказ (та же
    # политика, что в run.py).
    pagg = None
    if not a.no_temporal and a.step == 1:
        from .accum import HISTORY, load_poses
        pt_poses = load_poses(bag_dir)
        if pt_poses is None:
            print(f'{bag_name}: нет trajectory.npz — агрегация точек '
                  f'отключена.', flush=True)
        else:
            from .run import _PointAggregator
            frames = _frames(bag_dir, cache_size=HISTORY + 4)
            pagg = _PointAggregator(frames, pt_poses, HISTORY)
    if pagg is None:
        frames = _frames(bag_dir)
    # Ключ бэга для тёплой сборки полилинии — тот же, что в run.py.
    from visualize_bag import find_db3
    bag_key = find_db3(bag_dir)
    end = a.end if a.end > 0 else len(frames)
    end = min(end, len(frames))

    # Блок 3.5: трекер идёт по кадрам диапазона последовательно (как sweep в
    # run.py). Прогрев перед --start — как у run.py для одиночного кадра:
    # глубины MAX_MISS + CONFIRM_FRAMES + 1 хватает, чтобы трек подтвердился
    # и пережил допустимые пропуски.
    tracker = None
    if not a.no_track:
        from .accum import load_poses
        from .tracker import CONFIRM_FRAMES, MAX_MISS, ObstacleTracker
        tracker = ObstacleTracker(load_poses(bag_dir))
    rng = list(range(a.start, end, max(1, a.step)))
    todo = {i for i in rng
            if a.force or not os.path.isfile(
                os.path.join(out_dir, f'frame_{i:04d}.json'))}
    # Прогрев — от первого РЕАЛЬНО считаемого кадра, а не от --start, и на
    # первого РЕАЛЬНО считаемого кадра, а не от --start: при частичном
    # пересчёте без --force прогрев до a.start оставлял трекер холодным.
    if tracker is not None:
        warm_depth = MAX_MISS + CONFIRM_FRAMES + 1
        first_todo = min(todo) if todo else min(a.start, end)
        warm0 = max(0, first_todo - warm_depth)
        for j in range(warm0, first_todo):
            ob_j = analyze(frames[j][0], polyline=True, bag_key=bag_key,
                           frame_index=j)[1].get('obstacles')
            if ob_j is not None:
                tracker.update(j, [h for h in ob_j[0] if h['in_gauge']])

    print(f'{bag_name}: кадров в записи {len(frames)}, считаем {len(todo)} '
          f'-> {out_dir}', flush=True)
    for n, i in enumerate(rng, 1):
        if i not in todo:
            if tracker is not None:
                # Кадр уже посчитан (перезапуск без --force): скармливаем
                # трекеру находки из готового JSON, чтобы трек не рвался на
                # границе пересчитанного куска. Трекеру нужны только d,
                # lat_abs (= lat в JSON) и n_pts для дедупа двойников.
                try:
                    with open(os.path.join(out_dir, f'frame_{i:04d}.json'),
                              encoding='utf-8') as fh:
                        old = json.load(fh)
                    ing_old = [{'d': h['d'], 'lat_abs': h['lat'],
                                'n_pts': h.get('n_pts', 0)}
                               for h in old.get('obstacles', [])
                               if h.get('in_gauge')]
                except (OSError, ValueError, KeyError):
                    ing_old = []
                tracker.update(i, ing_old)
            continue
        payload = _clean(frame_payload(bag_name, i, frames[i][0],
                                       bag_key=bag_key,
                                       viewer_max_dist=a.viewer_max_dist,
                                       viewer_behind=a.viewer_behind,
                                       viewer_max_points=a.viewer_max_points,
                                       tracker=tracker,
                                       extra=(pagg.extra(i)
                                              if pagg is not None else None)))
        path = os.path.join(out_dir, f'frame_{i:04d}.json')
        tmp = path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as fh:
            json.dump(payload, fh, ensure_ascii=False, separators=(',', ':'))
        os.replace(tmp, path)   # атомарно: сервер не прочитает половину файла
        ing = [h for h in payload['obstacles'] if h['in_gauge']]
        n_trk = sum(1 for h in ing if h.get('track_confirmed'))
        print(f'  [{n}/{len(rng)}] frame_{i:04d}.json  '
              f'reach={payload["reach"]:.0f} м, препятствий в габарите '
              f'{len(ing)}'
              + (f', треком подтверждено {n_trk}' if tracker is not None
                 else ''), flush=True)
    print('готово', flush=True)


if __name__ == '__main__':
    main()
