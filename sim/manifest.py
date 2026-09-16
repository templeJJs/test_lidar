"""Манифест датасета: сцена, сенсор, объекты с разметкой, кадры.

Разметка -- смысл всего упражнения: без неё синтетику нельзя использовать для
оценки детектора (precision/recall), только «посмотреть глазами».
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field


@dataclass
class SceneInfo:
    kind: str = 'straight'
    section: str = 'wide'
    length_m: float = 200.0
    grade_pct: float = 2.34
    tracks: int = 2
    sensor_pose: dict = field(default_factory=dict)
    tunnel: dict = field(default_factory=dict)


@dataclass
class ObjectInfo:
    id: int
    cls: str
    bbox_xyz: list
    y_along_m: float
    offset_from_axis_m: float
    in_gabarit: bool
    frames_visible: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d['class'] = d.pop('cls')
        return d

    @classmethod
    def from_dict(cls, d: dict) -> 'ObjectInfo':
        d = dict(d)
        d['cls'] = d.pop('class')
        return cls(**d)


@dataclass
class Manifest:
    seed: int
    frames: int
    hz: float
    scene: SceneInfo
    objects: list = field(default_factory=list)
    sensor: str = 'sensor_hesai128.json'

    def to_dict(self) -> dict:
        return {'seed': self.seed, 'frames': self.frames, 'hz': self.hz,
                'scene': asdict(self.scene),
                'objects': [o.to_dict() for o in self.objects],
                'sensor': self.sensor}

    @classmethod
    def from_dict(cls, d: dict) -> 'Manifest':
        return cls(seed=d['seed'], frames=d['frames'], hz=d['hz'],
                   scene=SceneInfo(**d['scene']),
                   objects=[ObjectInfo.from_dict(o) for o in d.get('objects', [])],
                   sensor=d.get('sensor', 'sensor_hesai128.json'))

    def save(self, out_dir: str, name: str = 'manifest.json') -> str:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, name)
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(self.to_dict(), fh, ensure_ascii=False, indent=1)
        return path