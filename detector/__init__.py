"""Детектор посторонних объектов в тоннеле метро по облаку точек.

* `detector.core` -- чистая numpy-реализация: свободный объём вокруг локальной
  оси пути, исключение штатного оборудования, поиск компактных занятых
  интервалов и связность занятости в плоскости (y, u), отличающая предмет от
  конструкции, зашедшей в габарит краем. Без ROS, open3d и GUI -- проверяется
  локально.
* `detector.profiles` -- модели пути (полилиния оси + измеренный УГР) из
  `track_models.json`; `track_geometry` подключается только как уточнение.
* `detector.contour` -- дальний контур: отклонение поперечного профиля
  стены/свода и свободная ширина (дальность за пределы измеренной оси).
"""

from .core import (  # noqa: F401
    DetectionResult,
    DetectorConfig,
    Obstacle,
    TrackModel,
    detect,
    equipment_mask,
    occupancy_components,
    split_intervals,
    synthetic_object,
    threshold_points,
    wall_like_object,
)
from .profiles import model_for_db, model_for_name, record_name  # noqa: F401

__all__ = [
    'DetectionResult',
    'DetectorConfig',
    'Obstacle',
    'TrackModel',
    'detect',
    'equipment_mask',
    'occupancy_components',
    'split_intervals',
    'synthetic_object',
    'threshold_points',
    'wall_like_object',
    'model_for_db',
    'model_for_name',
    'record_name',
]