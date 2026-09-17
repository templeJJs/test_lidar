"""Сборка синтетического датасета: сенсор + случайная сцена + объекты + запись.

Примеры:
  python -m sim.cli sensor --bag for_hackathon/doubleT_obstacle --out sim/sensor_hesai128.json
  python -m sim.cli dataset --out sim_out --seed 42 --bags 2 --frames 5 \
      --section facts --kind curve --obstacle person:-20:0

Сцена по умолчанию (`--section facts --kind curve`) повторяет один из шести
реальных туннелей, поворачивает и наклоняет его по замеренным диапазонам.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass

import numpy as np

import bag_reader
from sim import objects, scene, writer
from sim.manifest import Manifest, ObjectInfo, SceneInfo
from sim.scanner import add_objects as scan_add_objects
from sim.scanner import scan
from sim.sensor import SensorModel

SENSOR_ARTEFACT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'sensor_hesai128.json')
TOPIC = '/sensing/lidar/hesai128/pointcloud'
MSG_TYPE = 'sensor_msgs/msg/PointCloud2'


@dataclass
class ObstacleSpec:
    cls: str
    y_along_m: float
    lateral_m: float = 0.0
    placement: str = 'in_gauge'
    yaw_deg: float = 0.0

    def to_placement(self) -> 'objects.Placement':
        """`objects.place_objects` ждёт свой `Placement` (там есть `yaw_deg`)."""
        return objects.Placement(cls=self.cls, y_along_m=self.y_along_m,
                                 lateral_m=self.lateral_m, placement=self.placement,
                                 yaw_deg=self.yaw_deg)


def parse_obstacle(text: str) -> ObstacleSpec:
    """`класс:y:смещение[:правило]`, например `person:-20:0` или `suitcase:-30:2.6:by_wall`."""
    parts = (text.split(':') + ['0', 'in_gauge'])[:4]
    return ObstacleSpec(parts[0], float(parts[1]), float(parts[2]), parts[3])


def _visible_frames(frame_xyz: np.ndarray, obj: dict) -> bool:
    """Есть ли у объекта возвраты в кадре: сравниваем с его боксом."""
    lo, hi = obj['bbox_xyz']
    inside = ((frame_xyz[:, 0] >= lo[0] - 0.2) & (frame_xyz[:, 0] <= hi[0] + 0.2)
              & (frame_xyz[:, 1] >= lo[1] - 0.2) & (frame_xyz[:, 1] <= hi[1] + 0.2)
              & (frame_xyz[:, 2] >= lo[2] - 0.2) & (frame_xyz[:, 2] <= hi[2] + 0.2))
    return int(np.count_nonzero(inside)) >= 5


def make_dataset(out_dir: str, seed: int = 1, bags: int = 1, frames: int = 5,
                 section: str = 'facts', kind: str = 'curve', columns: int = None,
                 span_deg: float = None, obstacles: list = None,
                 length_m: float = 200.0, profile: str = None,
                 sensor_path: str = SENSOR_ARTEFACT, hz: float = 10.0) -> dict:
    """Сгенерировать `bags` записей по `frames` кадров.

    `profile` пришпиливает профиль конкретной записи (`section='facts'`): сцена
    воспроизводит ИМЕННО её, без дрожания стен и уклона -- такую запись можно
    сверять с оригиналом через `sim.check_sim`. Без `profile` профиль берётся
    случайной записью, как и требует генератор.

    Возвращает {'bags': [пути], 'sensor': {...}, 'seconds_per_frame': float}.
    """
    obstacles = obstacles or []
    model = (SensorModel.load(sensor_path) if os.path.exists(sensor_path)
             else SensorModel())
    if columns:
        model.azimuth_columns = int(columns)
    if span_deg:
        model.azimuth_span_deg = float(span_deg)

    out_bags, per_frame = [], []
    for b in range(bags):
        params = scene.SceneParams(seed=seed + b, kind=kind, section=section,
                                   length_m=length_m, fact_bag=profile)
        s = scene.build_scene(params)
        # плотность лучей и поле зрения берём из профиля записи: у узких записей сенсор
        # работал с 1320…1478 колонками на 101° (169…189 тыс. точек на кадр), и
        # широкие 2706 колонок на 247° дали бы вдвое больше
        if columns is None and s.params.fact_bag:
            for prof in scene.fact_profiles():
                if prof['bag'] == s.params.fact_bag and prof.get('columns'):
                    model.azimuth_columns = int(prof['columns'])
                    if span_deg is None and prof.get('azimuth_span_deg'):
                        model.azimuth_span_deg = float(prof['azimuth_span_deg'])
                    break
        placed = objects.place_objects(params, s.track,
                                       [o.to_placement() for o in obstacles]) if obstacles else []
        s.objects = [dict(o) for o in placed]
        scan_add_objects(s)          # без этого трассировка не увидит объекты

        frames_out, visible = [], {o['id']: [] for o in placed}
        for i in range(frames):
            t0 = time.perf_counter()
            f = scan(s, model, i, np.random.default_rng(seed + b * 1000 + i))
            per_frame.append(time.perf_counter() - t0)
            frames_out.append(writer.FrameOut(f.xyz, f.intensity, f.ring, f.timestamp))
            for o in placed:
                if _visible_frames(f.xyz, o):
                    visible[o['id']].append(i)

        p = s.params
        bag_dir = os.path.join(out_dir, f'sim_{p.section}_{seed + b:04d}')
        os.makedirs(bag_dir, exist_ok=True)
        db = os.path.join(bag_dir, os.path.basename(bag_dir) + '_0.db3')
        writer.write_bag(frames_out, db, topic=TOPIC, msg_type=MSG_TYPE, hz=hz)

        infos = [ObjectInfo(id=o['id'], cls=o['class'], bbox_xyz=o['bbox_xyz'],
                            y_along_m=o['y_along_m'],
                            offset_from_axis_m=o['offset_from_axis_m'],
                            in_gabarit=o['in_gabarit'],
                            frames_visible=visible[o['id']]) for o in placed]
        Manifest(seed=seed + b, frames=frames, hz=hz,
                 scene=SceneInfo(kind=p.kind, section=p.section, length_m=length_m,
                                 grade_pct=float(p.grade_pct),
                                 tracks=int(p.tracks),
                                 sensor_pose=dict(s.sensor_pose),
                                 tunnel={'width_m': round(abs(p.walls_x[1] - p.walls_x[0]), 3),
                                         'vault_m': round(float(p.vault_m), 3)},
                                 walls_x=[round(float(v), 3) for v in p.walls_x],
                                 track_axis_x=round(float(p.track_axis_x), 3),
                                 curve_radius_m=(None if not np.isfinite(p.curve_radius_m)
                                                 else round(float(p.curve_radius_m), 1)),
                                 curve_sign=int(p.curve_sign),
                                 cant_mm=round(float(p.cant_mm), 2),
                                 fact_bag=p.fact_bag),
                 objects=infos).save(bag_dir)
        out_bags.append(bag_dir)

    return {'bags': out_bags, 'sensor': model.to_dict(),
            'seconds_per_frame': float(np.mean(per_frame)) if per_frame else 0.0}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='симулятор туннеля: сенсор и датасеты')
    sub = ap.add_subparsers(dest='cmd', required=True)

    p_s = sub.add_parser('sensor', help='снять модель сенсора с настоящей записи')
    p_s.add_argument('--bag', required=True)
    p_s.add_argument('--out', default=SENSOR_ARTEFACT)
    p_s.add_argument('--frame', type=int, default=0)

    p_d = sub.add_parser('dataset', help='собрать датасет синтетических записей')
    p_d.add_argument('--out', required=True)
    p_d.add_argument('--seed', type=int, default=1)
    p_d.add_argument('--bags', type=int, default=1)
    p_d.add_argument('--frames', type=int, default=5)
    p_d.add_argument('--section', choices=sorted(scene.SECTIONS), default='facts')
    p_d.add_argument('--kind', choices=('straight', 'curve'), default='curve')
    p_d.add_argument('--columns', type=int, default=None,
                     help='азимутальных колонок на кадр (по умолчанию из модели сенсора)')
    p_d.add_argument('--span', type=float, default=None,
                     help='поле зрения по азимуту, ° (по умолчанию из модели сенсора)')
    p_d.add_argument('--length', type=float, default=200.0,
                     help='длина туннеля в обе стороны, м; у записей данные уходят на 200+ м')
    p_d.add_argument('--profile', default=None,
                     help='пришпилить профиль записи (section=facts), например '
                          'doubleT_obstacle: сцена повторит её точно')
    p_d.add_argument('--obstacle', action='append', default=[],
                     help='класс:y:смещение[:правило], например person:-20:0')

    args = ap.parse_args(argv)
    if args.cmd == 'sensor':
        model = SensorModel.from_bag(bag_reader.find_db3(args.bag), frame=args.frame)
        model.save(args.out)
        print(f'сенсор: {model.rings} колец, {model.azimuth_columns} колонок на '
              f'{model.azimuth_span_deg:.0f}°, шум {model.range_sigma_m:.3f} м -> {args.out}')
        return 0

    t0 = time.perf_counter()
    res = make_dataset(out_dir=args.out, seed=args.seed, bags=args.bags,
                       frames=args.frames, section=args.section, kind=args.kind,
                       columns=args.columns,
                       span_deg=args.span,
                       profile=args.profile,
                       obstacles=[parse_obstacle(t) for t in args.obstacle],
                       length_m=args.length)
    total = time.perf_counter() - t0
    print(f'записей: {len(res["bags"])}, кадров в каждой: {args.frames}, '
          f'{res["seconds_per_frame"]*1000:.0f} мс на кадр, всего {total:.1f} с')
    for path in res['bags']:
        print('  ', path)
    return 0


if __name__ == '__main__':
    sys.exit(main())