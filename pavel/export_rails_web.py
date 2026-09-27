"""Предрасчёт линий рельсов из карты пути (track_map.npz) для веб-вьюера.

Тот же механизм, что у guard (guard/export_web.py): web_viewer.py не
импортирует pavel/ (в корне проекта свои rail_detection.py/track_geometry.py
— конфликт имён модулей) и ничего не считает на лету, а только отдаёт готовые
JSON маршрутом /rails. Считает их эта команда (запуск из каталога pavel):

    cd pavel
    python export_rails_web.py ../for_hackathon/doubleT_platform

Результат: pavel/rails_web/<запись>/frame_NNNN.json — левая/правая нити
рельсов и осевая линия, спроецированные картой (track_map.map_to_frame) в
систему КАДРА. Координаты — прямо система лидара/вьюера (x поперёк, y назад
— вперёд это -y, z вверх), клиент (web/raillayer.js) рисует их как есть.

Отличие от guard-экспорта: облака кадров здесь не читаются вовсе (карта +
trajectory.npz), поэтому прогон всей записи — секунды, а не минуты.
Нужны track_map.npz (python track_map.py <запись>) и trajectory.npz
(python odometry.py <запись> --save) в папке записи; без них — отказ с
подсказкой, а не молчаливый пустой выход.
"""

import argparse
import json
import math
import os

import numpy as np

from track_map import map_to_frame
from guard.accum import load_poses

_PAVEL_DIR = os.path.dirname(os.path.abspath(__file__))
RAIL_HEIGHT_OVER_GROUND = 0.16   # как в visualize_bag.py: запасная высота головки


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


def frame_payload(bag_name, idx, track, poses, planes, max_range):
    """Кадр -> линии рельсов в системе кадра (система лидара = система вьюера)."""
    proj = map_to_frame(track, poses, idx, max_range=max_range)
    n = len(proj['fwd'])
    plane = planes[idx] if idx < len(planes) else None
    a, b, c = (plane if plane is not None and np.all(np.isfinite(plane))
               else (0.0, 0.0, -1.5))
    if 'rail_dz' in proj:
        dz = np.asarray(proj['rail_dz'], dtype=float)
        dz = np.where(np.isnan(dz), RAIL_HEIGHT_OVER_GROUND, dz)
    else:
        dz = np.full(n, RAIL_HEIGHT_OVER_GROUND)

    def line(xs, ys):
        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)
        zs = a * xs + b * ys + c + dz
        return [[float(x), float(y), float(z)] for x, y, z in zip(xs, ys, zs)]

    lx, ly = proj['left_x'], proj['left_y']
    rx, ry = proj['right_x'], proj['right_y']
    cx = (np.asarray(lx) + np.asarray(rx)) / 2
    cy = (np.asarray(ly) + np.asarray(ry)) / 2
    gauge = np.hypot(np.asarray(lx) - np.asarray(rx),
                     np.asarray(ly) - np.asarray(ry))
    return {
        'format': 'rails-web/1',
        'bag': bag_name,
        'frame': int(idx),
        'coords': 'система лидара кадра (x поперёк, y назад, z вверх) = система вьюера',
        'source': 'track_map.npz',
        'n_nodes': int(n),
        'gauge_med': float(np.median(gauge)) if n else None,
        'left': line(lx, ly),
        'right': line(rx, ry),
        'center': line(cx, cy),
    }


def main():
    ap = argparse.ArgumentParser(
        description='Предрасчёт линий рельсов (track_map) в JSON для '
                    'веб-вьюера (web_viewer.py /rails, web/raillayer.js)')
    ap.add_argument('bag_dir', help='папка записи (внутри *.db3, track_map.npz, trajectory.npz)')
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--end', type=int, default=0,
                    help='по какой кадр (не включительно); 0 — все кадры')
    ap.add_argument('--max-range', type=float, default=150.0,
                    help='дальность отрезка карты вперёд от кадра, м')
    ap.add_argument('--out', default=None,
                    help='куда писать (по умолчанию pavel/rails_web/<запись>)')
    ap.add_argument('--force', action='store_true',
                    help='пересчитать кадры, для которых файл уже есть')
    a = ap.parse_args()

    bag_dir = a.bag_dir.rstrip('/\\')
    bag_name = os.path.basename(os.path.abspath(bag_dir))
    map_path = os.path.join(bag_dir, 'track_map.npz')
    if not os.path.isfile(map_path):
        raise SystemExit(f'{bag_dir}: нет track_map.npz — сначала '
                         f'"python track_map.py {bag_dir}"')
    poses = load_poses(bag_dir)
    if poses is None:
        raise SystemExit(f'{bag_dir}: нет trajectory.npz — сначала '
                         f'"python odometry.py {bag_dir} --save"')
    data = np.load(map_path)
    track = {k: data[k] for k in data.files if k not in ('planes', 'planes_raw')}
    planes = data['planes'] if 'planes' in data.files else None
    if planes is None:
        planes = np.full((len(poses), 3), np.nan)

    out_dir = a.out or os.path.join(_PAVEL_DIR, 'rails_web', bag_name)
    os.makedirs(out_dir, exist_ok=True)

    end = a.end if a.end > 0 else len(poses)
    end = min(end, len(poses))
    todo = [i for i in range(a.start, end)
            if a.force or not os.path.isfile(
                os.path.join(out_dir, f'frame_{i:04d}.json'))]
    print(f'{bag_name}: кадров {len(poses)}, считаем {len(todo)} -> {out_dir}',
          flush=True)
    for n, i in enumerate(todo, 1):
        payload = _clean(frame_payload(bag_name, i, track, poses, planes,
                                       a.max_range))
        path = os.path.join(out_dir, f'frame_{i:04d}.json')
        tmp = path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as fh:
            json.dump(payload, fh, ensure_ascii=False, separators=(',', ':'))
        os.replace(tmp, path)   # атомарно: сервер не прочитает половину файла
        if n % 50 == 0 or n == len(todo):
            print(f'  [{n}/{len(todo)}] frame_{i:04d}.json  '
                  f'узлов {payload["n_nodes"]}', flush=True)
    print('готово', flush=True)


if __name__ == '__main__':
    main()
