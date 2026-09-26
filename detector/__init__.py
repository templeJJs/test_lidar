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
* `detector.background` -- фоновая модель канала лотка записи (устойчиво
  занятые и доказанно свободные ячейки): расширяет канал за предел
  однокадровой различимости, ядро принимает снимок фона опционально.
  ОПТ-ИН: в `TemporalDetector` выключен по умолчанию -- на едущих пустых
  записях корпуса гейт не съедает бугры (см. `docs/detector-algorithm.md`,
  раздел про фоновую модель).
"""

from .core import (  # noqa: F401
    AxisPose,
    DetectionResult,
    DetectorConfig,
    Obstacle,
    TrackModel,
    axis_pose,
    detect,
    equipment_mask,
    occupancy_components,
    split_intervals,
    synthetic_object,
    threshold_points,
)
from .profiles import model_for_db, model_for_name, record_name  # noqa: F401

__all__ = [
    'AxisPose',
    'DetectionResult',
    'DetectorConfig',
    'Obstacle',
    'TrackModel',
    'axis_pose',
    'detect',
    'equipment_mask',
    'occupancy_components',
    'split_intervals',
    'synthetic_object',
    'threshold_points',
    'model_for_db',
    'model_for_name',
    'record_name',
]