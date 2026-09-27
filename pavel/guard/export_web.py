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


def frame_payload(bag_name, idx, pts):
    """Кадр -> словарь для JSON (система guard, см. заголовок файла)."""
    axis, wall = analyze(pts)

    far = axis.get('far')
    axis_far = None
    if far is not None:
        axis_far = {
            'fwd_mid': far['fwd_mid'],
            'lat': far['lat'],
            'status': list(far['status']),
        }

    ob_hits = []
    ob = wall.get('obstacles')
    if ob is not None:
        hits, _summary = ob
        for h in hits:
            # x_min/x_max в detect() даны ОТНОСИТЕЛЬНО центра коридора бина;
            # центр = lat_abs - lat, переводим в абсолютную поперечную ось —
            # клиенту не нужно знать про center_bins.
            center = h['lat_abs'] - h['lat']
            ob_hits.append({
                'd': h['d'],
                'lat': h['lat_abs'],
                'x_min': h['x_min'] + center,
                'x_max': h['x_max'] + center,
                'width': h['width'],
                'height': h['height'],
                'n_pts': h['n_pts'],
                'in_gauge': h['in_gauge'],
                'confirmed': h['confirmed'],
            })

    return {
        'format': 'guard-web/1',
        'bag': bag_name,
        'frame': int(idx),
        'coords': 'guard: fwd=-y, lat=x, up=z (система лидара кадра)',
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
    a = ap.parse_args()

    bag_dir = a.bag_dir.rstrip('/\\')
    bag_name = os.path.basename(os.path.abspath(bag_dir))
    out_dir = a.out or os.path.join(_PAVEL_DIR, 'guard_web', bag_name)
    os.makedirs(out_dir, exist_ok=True)

    frames = _frames(bag_dir)
    end = a.end if a.end > 0 else len(frames)
    end = min(end, len(frames))
    todo = [i for i in range(a.start, end, max(1, a.step))
            if a.force or not os.path.isfile(
                os.path.join(out_dir, f'frame_{i:04d}.json'))]
    print(f'{bag_name}: кадров в записи {len(frames)}, считаем {len(todo)} '
          f'-> {out_dir}', flush=True)
    for n, i in enumerate(todo, 1):
        payload = _clean(frame_payload(bag_name, i, frames[i][0]))
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
