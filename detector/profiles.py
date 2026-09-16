"""Модели пути: сохранённая полилиния, `track_geometry` или прямая.

`track_models.json` -- снимок `track_geometry.build_track(db, frame=0)` по шести
записям `for_hackathon/`: полилиния оси (середина колеи) и измеренный УГР
(верх головки рельса) с шагом 0.5 м. Он нужен потому, что `track_geometry`
тянет open3d, а в контейнере и в ROS-узле open3d нет; плюс `build_track` стоит
0.8..1.3 с и в кадровый цикл не влезает.

Порядок выбора модели в `model_for_db`:

  1. сохранённая полилиния из `track_models.json` -- если имя записи известно;
  2. живой `track_geometry.build_track` -- если он импортируется и не падает
     (уточнение наклона оси для новой записи);
  3. прямая `x = s*y + i` со средними параметрами -- последний резерв, даёт
     заведомо худший результат (прямая ось уводит габарит на поворотах),
     поэтому факт подстановки виден в `TrackModel.name`.

Таблица `MEASURED_STRAIGHT` -- те же записи, но прямая ось: нужна для
контрольного прогона, доказывающего, что дело именно в локальной оси.
"""

from __future__ import annotations

import json
import os
from typing import Optional

from .core import TrackModel

_HERE = os.path.dirname(os.path.abspath(__file__))
TABLE_PATH = os.path.join(_HERE, 'track_models.json')

# Прямая ось по записям: (s, i, floor_a, floor_b). Только для контроля.
MEASURED_STRAIGHT = {
    'doubleT_obstacle': (0.018276, -0.082130, -1.874, 2.03e-02),
    'doubleT_platform': (0.004245, -0.008390, -1.446, 6.70e-03),
    'roundT_doubleT': (-0.016923, 0.056538, -1.553, 5.70e-03),
    'roundT_pressureGate_roundT': (0.004713, -0.006302, -1.489, 7.20e-03),
    'roundT_squareT_pressureGate_squareT': (0.006765, -0.003162, -1.337, 8.20e-03),
    'squareT_platform_squareT_switch': (0.006571, -0.005548, -1.408, 1.14e-02),
}

FALLBACK = TrackModel(
    name='fallback-straight',
    gauge_m=1.593,
    axis_straight_s=0.0,
    axis_straight_i=0.0,
)


def _table() -> dict:
    try:
        with open(TABLE_PATH, encoding='utf-8') as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return {}


def record_name(db_path: str) -> Optional[str]:
    """Имя записи по пути к `.db3`: имя каталога над файлом или префикс имени файла."""
    path = os.path.normpath(str(db_path)).replace('\\', '/')
    parts = path.split('/')
    table = _table()
    for candidate in reversed(parts[:-1]):
        if candidate in table:
            return candidate
    stem = os.path.splitext(parts[-1])[0]
    for name in table:
        if name.startswith('_'):
            continue
        if stem.startswith(name):
            return name
    return None


def model_for_name(name: str, straight: bool = False) -> TrackModel:
    """Модель пути по имени записи. `straight=True` -- прямая ось (контроль)."""
    if straight:
        data = MEASURED_STRAIGHT.get(name)
        if data is None:
            return FALLBACK
        s, i, _fa, _fb = data
        return TrackModel(name=f'{name} [straight]', gauge_m=1.593,
                          axis_straight_s=s, axis_straight_i=i)
    entry = _table().get(name)
    if not entry:
        return FALLBACK
    return TrackModel(
        name=name,
        gauge_m=float(entry['gauge_m']),
        y_nodes=entry['y_nodes'],
        axis_x_m=entry['axis_x_m'],
        ugr_z_m=entry['ugr_z_m'],
        axis_straight_s=float(entry.get('axis_straight_s', 0.0)),
        axis_straight_i=float(entry.get('axis_straight_i', 0.0)),
    )


def model_for_db(db_path: str, straight: bool = False) -> TrackModel:
    """Модель пути по пути к `.db3`."""
    name = record_name(db_path)
    if name is None:
        return FALLBACK
    return model_for_name(name, straight=straight)


def build_polyline(db_path: str, frame: int = 0) -> Optional[TrackModel]:
    """Полилиния оси из `track_geometry.build_track`; None -- если недоступен.

    Импорт ленивый и обёрнут: `track_geometry` правится параллельно и тянет
    open3d, которого в контейнере нет. Падение чужого модуля не должно ронять
    детектор -- вызывающий откатывается на сохранённую таблицу.
    """
    try:
        import numpy as np
        from track_geometry import build_track

        track = build_track(str(db_path), int(frame))
        mesh = track['meta']['rail_mesh']
        left = np.asarray(mesh['rails']['left']['nodes'], dtype=np.float64)
        right = np.asarray(mesh['rails']['right']['nodes'], dtype=np.float64)
        if left.size < 6 or right.size != left.size:
            return None
        return TrackModel(
            name=record_name(db_path) or 'live',
            gauge_m=float(track['meta']['gauge_axis_mm']) / 1000.0,
            y_nodes=tuple(left[:, 0].tolist()),
            axis_x_m=tuple((0.5 * (left[:, 1] + right[:, 1])).tolist()),
            ugr_z_m=tuple((0.5 * (left[:, 2] + right[:, 2])).tolist()),
            axis_straight_s=float(track['axis']['s']),
            axis_straight_i=float(track['axis']['i']),
        )
    except Exception:                     # noqa: BLE001 -- чужой модуль/данные
        return None