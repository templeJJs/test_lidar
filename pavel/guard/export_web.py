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
                  viewer_max_points=400000):
    """Кадр -> словарь для JSON (система guard, см. заголовок файла).

    Ось блока 1 — ЛОМАНАЯ (polyline=True), как в CLI (run.py по умолчанию):
    веб должен получать ту же ось, что консоль. С прямой осью веб показывал
    физически невозможный излом линии пути на границе ближней зоны (замер
    f876 cloud_with_fake_obj: вторая разность оси бинов 0.77 м на 22.5 м,
    R = 33 м; с полилинией R = 223 м). bag_key/frame_index — ключи тёплой
    сборки полилинии (те же, что передаёт run.py).
    """
    axis, wall = analyze(pts, polyline=True, bag_key=bag_key, frame_index=idx)

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
            }
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
    a = ap.parse_args()

    bag_dir = a.bag_dir.rstrip('/\\')
    bag_name = os.path.basename(os.path.abspath(bag_dir))
    out_dir = a.out or os.path.join(_PAVEL_DIR, 'guard_web', bag_name)
    os.makedirs(out_dir, exist_ok=True)

    frames = _frames(bag_dir)
    # Ключ бэга для тёплой сборки полилинии — тот же, что в run.py.
    from visualize_bag import find_db3
    bag_key = find_db3(bag_dir)
    end = a.end if a.end > 0 else len(frames)
    end = min(end, len(frames))
    todo = [i for i in range(a.start, end, max(1, a.step))
            if a.force or not os.path.isfile(
                os.path.join(out_dir, f'frame_{i:04d}.json'))]
    print(f'{bag_name}: кадров в записи {len(frames)}, считаем {len(todo)} '
          f'-> {out_dir}', flush=True)
    for n, i in enumerate(todo, 1):
        payload = _clean(frame_payload(bag_name, i, frames[i][0],
                                       bag_key=bag_key,
                                       viewer_max_dist=a.viewer_max_dist,
                                       viewer_behind=a.viewer_behind,
                                       viewer_max_points=a.viewer_max_points))
        path = os.path.join(out_dir, f'frame_{i:04d}.json')
        tmp = path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as fh:
            json.dump(payload, fh, ensure_ascii=False, separators=(',', ':'))
        os.replace(tmp, path)   # атомарно: сервер не прочитает половину файла
        ing = [h for h in payload['obstacles'] if h['in_gauge']]
        print(f'  [{n}/{len(todo)}] frame_{i:04d}.json  '
              f'reach={payload["reach"]:.0f} м, препятствий в габарите '
              f'{len(ing)}', flush=True)
    print('готово', flush=True)


if __name__ == '__main__':
    main()
