"""Ядро детектора посторонних объектов в тоннеле по одному облаку точек.

Модуль не зависит ни от ROS, ни от open3d, ни от `zones.py`: на вход идёт
`(N, 3)` точек в СИСТЕМЕ ЛИДАРА и модель пути, на выходе -- список препятствий и
диагностика. Поэтому алгоритм одинаков в оффлайне и в ROS-узле, и проверяется
локально (`tests/test_detector_core.py`).

Задача -- «выход габарита»: есть ли что-то внутри свободного объёма поезда.

Координаты и модель пути
------------------------
Туннель уходит в −Y. Всё считается в координатах ОТНОСИТЕЛЬНО ПУТИ:

    u = x - x_оси(y)     -- поперёк пути, 0 = ось колеи
    h = z - z_УГР(y)     -- высота над УГР (уровнем головки рельса)

`u` и `h` берутся из ЛОКАЛЬНОЙ модели пути -- полилинии, измеренной по рельсам
(`track_geometry.build_track`), а не из прямой `x = s*y + i`. Это принципиально:
прямая ось уводит габарит на внешнюю стенку поворота, и на 20..40 м появляются
десятки ложных срабатываний. Полилиния хранится в `detector/track_models.json`
(см. `detector/profiles.py`) -- ROS и контейнер работают без open3d.

Свободный объём
---------------
    |u| <= 1.25 м,   h in [0.30, 3.70] м над УГР
Известное оборудование исключается: ходовые рельсы `|u -+ 0.795| < 0.10` и
кабельный лоток `|u| in [1.25, 1.50] & h in [0.15, 0.60]`.

Почему срабатывание -- не «сколько-то точек в бине»
--------------------------------------------------
Замерено по всему корпусу (6 записей, 2488 кадров): в 1-м бине по y внутри
габарита регулярно оказываются 20..300 точек, разбросанных по высоте на 2..3 м.
Это не объекты, а КРОМКА СТЕНЫ ИЛИ ГЕРМОЗАТВОРА, заходящая в габарит: в
`roundT_*` стена подходит к оси ближе 1.25 м, и её край попадает в объём. Отличить
её от объекта счётом точек нельзя -- счёт у них одного порядка.

Отличие структурное. Кандидат в бине 1 м по y -- интервал по `u` (разрыв >
`u_gap`) с точками, высотой и шириной. Дальше работают три признака, все
проверенные на числах (разбор -- `docs/detector-algorithm.md`, раздел «человек
против стены»):

  1) СВЯЗНОСТЬ ЗАНЯТОСТИ в плоскости (y, u). Точки объёма И полосы
     `|u| <= u_out` кладутся в сетку `wall_cell_y_m` x `wall_cell_u_m`; ячейка
     занята с `wall_min_cell_points` точек; занятые ячейки склеиваются в
     компоненты по 8 соседям. Компонент кандидата, который (а) уходит наружу
     габарита за `|u| = structure_out_m` ИЛИ (б) тянется вдоль пути дольше
     `max_structure_y_m`, -- это стена/платформа, кандидат отбрасывается.
     Связность вместо «интервал доходит до кромки в этом же бине»: замерено на
     `roundT_doubleT`, где стена уходит наружу не в том бине, где её внутренний
     край, и прежнее правило пропускало внутренние обрывки (15 ложных событий
     на кадрах 107..152).
  2) Кластер длиннее `max_cluster_y_m` (4 м) вдоль пути -- стена: запасной
     счётный вариант того же признака (1б), когда сетка компонент не сработала.
  3) Размах высоты больше `max_span_m`: у человека 0.39..0.90 м (замерено на
     46 кадрах `doubleT_obstacle`), у стены в объёме 1.79..3.01 м -- стена
     заходит в габарит наклонной поверхностью и заполняет объём по высоте.

Прежнее правило «интервал обязан замкнуться до `|u| = 1.25 - 0.18`»
(`interior_margin_m`) убрано: человек, идущий по кромке габарита, его не
проходил. `edge_allow_m` (0) остался как жёсткая отсечка: интервал, вышедший за
`|u| = 1.25`, сам по себе не выдаётся.

Проверено на всём корпусе (`python -m validation.fp_per_hour`, 2488 кадров):
0 ложных событий (было 282). Человек в `doubleT_obstacle` находится на 46 из 52
кадров 13..63 -- включая 30..45, где низ объёма срезает силуэт до размаха
0.36..0.39 м (пороги высоты опущены до `min_span_m` = 0.35 и `dense_ratio` =
0.30, и на корпусе это ложных не добавило). На 0/75/100..200 человек стоит ВНЕ
габарита, и детектор молчит -- иначе он выдавался бы на «пустых» записях, где
точно так же выглядит след стены.

Чего детектор НЕ делает
-----------------------
Габарит достоверен только на участке `valid_far_m` (у `doubleT_obstacle` --
60.75 м): полилиния в снимке `track_models.json` продлена до -110 м (рельсы
измерены до -30..-40 м, дальше -- касательное продолжение по наклону хвоста и
кривизне стен/свода), но на длинном продлении ось уезжает, и на записях с
доворотом датчика (`roundT_doubleT`, `roundT_pressureGate_roundT`) участок
пришлось сократить до 20..40 м. Движущиеся объекты не отслеживаются: разметки
нет, каждый кадр обрабатывается независимо. Для дальнего контура (100..200 м)
служит `detector/contour.py`.
"""

from __future__ import annotations



from dataclasses import asdict, dataclass, field, replace
from typing import Optional

import numpy as np

# --- свободный объём и норма тоннеля ---------------------------------------
HALF_WIDTH_M = 1.25          # полуширина габарита от оси пути
H_LOW_M = 0.30               # низ объёма над УГР (головкой рельса)
H_HIGH_M = 3.70              # верх объёма над УГР
U_OUT_M = 1.90               # докуда смотрим по u, чтобы увидеть уход структуры наружу
EDGE_ALLOW_M = 0.0           # допуск на кромку габарита (неопределённость оси), м
U_GAP_M = 0.12               # разрыв по u, разделяющий интервалы
BIN_M = 1.0                  # бин агрегации вдоль пути
MIN_POINTS = 20              # точек в интервале бина
MIN_SPAN_M = 0.35            # размах высот p5..p95 в интервале
MIN_WIDTH_M = 0.25           # размах по u: объект обязан иметь ширину, а не быть щелью
MAX_CLUSTER_Y_M = 4.0        # длина склейки вдоль пути: стена тянется дальше
U_MERGE_M = 0.30             # насколько интервалы по u должны быть близки для склейки
DENSE_MASS = 0.70            # доля точек, обязанная укладываться в окно высоты
DENSE_RATIO = 0.30           # плотная полоса -- не меньше этой доли своего размаха
MIN_RANGE_M = 4.0            # ближе не смотрим: там корпус носителя
MAX_RANGE_M = 150.0          # дальше данных всё равно нет

# --- связность занятости в плоскости (y, u): стена против объекта -----------
# Сетка нужна, чтобы отличить КОМПАКТНЫЙ предмет от ЧАСТИ ПРОТЯЖЁННОЙ
# КОНСТРУКЦИИ, заходящей в габарит краем. Интервал в одном бине этого не даёт:
# стена уходит наружу не в том бине, где виден её внутренний край (замерено на
# `roundT_doubleT`). Ячейка занята с двух точек -- одиночный выброс структуры не
# образует.
WALL_CELL_Y_M = 1.0          # ячейка сетки вдоль пути
WALL_CELL_U_M = 0.12         # ... и поперёк (как разрыв между предметами)
WALL_MIN_CELL_POINTS = 2     # точек в ячейке, чтобы считать её занятой
MAX_STRUCTURE_Y_M = 3.0      # компонент длиннее -- стена/платформа, не предмет
STRUCTURE_OUT_M = 1.45       # компонент за этой |u| -- структура уходит наружу
MAX_SPAN_M = 1.7             # размах выше -- поверхность стены в объёме, не предмет

RAIL_HALF_M = 0.10           # полуширина полосы рельса вокруг |u| = gauge/2
RAIL_H_MAX_M = 0.35          # рельс живёт ниже этой высоты над УГР (головка ~0.17)
TRAY_U_M = (1.25, 1.50)      # кабельный лоток по u
TRAY_H_M = (0.15, 0.60)      # кабельный лоток по высоте

AXIS_EXTENSION_M = 15.0      # продолжение полилинии оси, где она ещё годна
CELL_M = 0.05                # ячейка для диагностики плотности кластера

# Геометрия сканирования -- для нормировки порога по числу колец.
RING_COUNT = 128
ELEV_FOV_DEG = 39.5          # -25.1..+14.4
DEFAULT_RANGE_NORMALIZATION = False
RANGE_REF_M = 10.0


@dataclass(frozen=True)
class TrackModel:
    """Локальная модель пути: полилиния оси + измеренный УГР.

    `y_nodes` -- узлы вдоль пути (по возрастанию, т.е. от дальних к ближним),
    `axis_x_m` -- X середины колеи в узле, `ugr_z_m` -- Z верха головки рельса
    (ось отсчёта высоты `h`). Строится `track_geometry.build_track` и
    сохраняется в `detector/track_models.json`.

    Хранится ещё и прямая `axis_straight` -- только для контроля: прогон на
    ней показывает, сколько ложных срабатываний даёт прямая ось.
    """

    name: str = 'default'
    gauge_m: float = 1.593
    y_nodes: tuple = ()
    axis_x_m: tuple = ()
    ugr_z_m: tuple = ()
    axis_straight_s: float = 0.0
    axis_straight_i: float = 0.0
    axis_extension_m: float = AXIS_EXTENSION_M
    # Дальний конец участка, на котором ось ещё достоверна для габарита (м, < 0).
    # Полилиния может быть продлена дальше (ось определена на всей видимой
    # дальности), но габарит на продлении уезжает, поэтому детектор работает
    # только до `valid_far_m`. 0 -- считать по узлам (`y_nodes - axis_extension_m`).
    valid_far_m: float = 0.0
    # Откуда взялась модель: 'snapshot' (таблица), 'live' (track_geometry),
    # 'straight' (контроль), 'fallback' (прямая по умолчанию). Нужен, чтобы
    # подмена модели была видна в логе, а не молча.
    source: str = ''

    @property
    def has_polyline(self) -> bool:
        return len(self.y_nodes) >= 2

    def _interp(self, y, nodes, values):
        """Линейная интерполяция по узлам + линейное продолжение за края.

        Продолжение берётся по наклону последних `END_SLOPE_NODES` узлов: по двум
        соседним узлам наклон шумный, а ошибка продолжения прямо задаёт уход
        габарита вдали.
        """
        yn = np.asarray(nodes, dtype=np.float64)
        vn = np.asarray(values, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        out = np.interp(y_arr, yn, vn)
        n = yn.size
        k = min(5, n)
        if np.any(y_arr < yn[0]) and n >= 2:
            slope = (vn[k - 1] - vn[0]) / (yn[k - 1] - yn[0])
            lo = y_arr < yn[0]
            out = np.where(lo, vn[0] + slope * (y_arr - yn[0]), out)
        if np.any(y_arr > yn[-1]) and n >= 2:
            slope = (vn[-1] - vn[-k]) / (yn[-1] - yn[-k])
            hi = y_arr > yn[-1]
            out = np.where(hi, vn[-1] + slope * (y_arr - yn[-1]), out)
        return out

    def axis_x(self, y):
        if self.has_polyline:
            return self._interp(y, self.y_nodes, self.axis_x_m)
        y_arr = np.asarray(y, dtype=np.float64)
        return self.axis_straight_s * y_arr + self.axis_straight_i

    def ugr_z(self, y):
        if self.has_polyline:
            return self._interp(y, self.y_nodes, self.ugr_z_m)
        # Запасной путь: УГР как плоскость среднего наклона пола записи.
        y_arr = np.asarray(y, dtype=np.float64)
        return np.full_like(y_arr, -1.15, dtype=np.float64)

    def relative(self, xyz: np.ndarray) -> tuple:
        """(u, h) -- поперёк пути и над УГР для облака (N, 3)."""
        xyz = np.asarray(xyz, dtype=np.float64)
        y = xyz[:, 1]
        u = xyz[:, 0] - self.axis_x(y)
        h = xyz[:, 2] - self.ugr_z(y)
        return u, h

    def valid_range(self, max_range_m: float = MAX_RANGE_M,
                    min_range_m: float = MIN_RANGE_M) -> tuple:
        """(y_far, y_near) -- где ось ещё достоверна, метры по Y (отрицательные).

        Дальний конец: `valid_far_m`, если он задан (снимок продлевает
        полилинию дальше, но габарит на продлении уезжает), иначе измеренные
        узлы минус продолжение. Ближний -- конец измеренных узлов или
        `-min_range_m`, что ближе к датчику.
        """
        if not self.has_polyline:
            return (-max_range_m, -min_range_m)
        if self.valid_far_m < 0.0:
            y_lo = float(self.valid_far_m)
        else:
            y_lo = float(np.min(self.y_nodes)) - float(self.axis_extension_m)
        y_hi = float(max(np.max(self.y_nodes), -min_range_m))
        return (max(y_lo, -max_range_m), min(y_hi, -min_range_m))

    def to_dict(self) -> dict:
        out = asdict(self)
        out['y_nodes'] = [float(v) for v in self.y_nodes]
        out['axis_x_m'] = [float(v) for v in self.axis_x_m]
        out['ugr_z_m'] = [float(v) for v in self.ugr_z_m]
        return out

    @classmethod
    def from_dict(cls, data: dict) -> 'TrackModel':
        data = dict(data)
        for key in ('y_nodes', 'axis_x_m', 'ugr_z_m'):
            if key in data and data[key] is not None:
                data[key] = tuple(float(v) for v in data[key])
        known = {f: data[f] for f in cls.__dataclass_fields__ if f in data}
        return cls(**known)

    @classmethod
    def from_bag(cls, db_path: str, frame: int = 0) -> 'TrackModel':
        """Модель пути: сохранённая полилиния, иначе `track_geometry`, иначе прямая."""
        from .profiles import model_for_db, build_polyline

        built = build_polyline(db_path, frame)
        if built is not None:
            return built
        return model_for_db(db_path)


@dataclass
class DetectorConfig:
    """Пороги детектора выхода из габарита. Все длины -- метры."""

    half_width_m: float = HALF_WIDTH_M
    h_low_m: float = H_LOW_M
    h_high_m: float = H_HIGH_M
    u_out_m: float = U_OUT_M
    edge_allow_m: float = EDGE_ALLOW_M
    u_gap_m: float = U_GAP_M
    bin_m: float = BIN_M
    min_points: int = MIN_POINTS
    min_span_m: float = MIN_SPAN_M
    max_span_m: float = MAX_SPAN_M
    dense_mass: float = DENSE_MASS
    dense_ratio: float = DENSE_RATIO
    min_width_m: float = MIN_WIDTH_M
    max_cluster_y_m: float = MAX_CLUSTER_Y_M
    u_merge_m: float = U_MERGE_M
    # Связность занятости (y, u): стена/платформа против компактного предмета.
    wall_cell_y_m: float = WALL_CELL_Y_M
    wall_cell_u_m: float = WALL_CELL_U_M
    wall_min_cell_points: int = WALL_MIN_CELL_POINTS
    max_structure_y_m: float = MAX_STRUCTURE_Y_M
    structure_out_m: float = STRUCTURE_OUT_M
    min_range_m: float = MIN_RANGE_M
    max_range_m: float = MAX_RANGE_M
    rail_half_m: float = RAIL_HALF_M
    rail_h_max_m: float = RAIL_H_MAX_M
    tray_u_m: tuple = TRAY_U_M
    tray_h_m: tuple = TRAY_H_M
    cell_m: float = CELL_M
    # Нормировка порога на число колец, попадающих в бин (счёт падает как 1/d^2).
    # По умолчанию выключена: в проверенном диапазоне (ось измерена) счётного
    # порога достаточно, а нормировка поднимает чувствительность на дальней
    # границе, где как раз начинаются ложные. Включается вместе с дальним
    # контуром (`contour.py`).
    range_normalization: bool = DEFAULT_RANGE_NORMALIZATION
    range_ref_m: float = RANGE_REF_M
    ring_count: int = RING_COUNT
    elev_fov_deg: float = ELEV_FOV_DEG
    # Ограничить работу участком, где ось измерена (иначе габарит уезжает).
    limit_to_axis_range: bool = True

    def to_dict(self) -> dict:
        out = asdict(self)
        for key in ('tray_u_m', 'tray_h_m'):
            out[key] = list(out[key])
        return out

    @classmethod
    def from_dict(cls, data: dict) -> 'DetectorConfig':
        data = dict(data)
        for key in ('tray_u_m', 'tray_h_m'):
            if key in data and data[key] is not None:
                data[key] = tuple(float(v) for v in data[key])
        known = {f: data[f] for f in cls.__dataclass_fields__ if f in data}
        return cls(**known)


@dataclass
class Obstacle:
    """Найденное препятствие. `u_m` -- поперёк пути, `h_*` -- над УГР."""

    distance_m: float          # дальность до центра (евклидова, из системы лидара)
    y_m: float                 # продольная координата центра
    u_m: float                 # поперечная координата центра относительно оси
    x_m: float                 # то же в абсолютной X
    h_min_m: float             # низ занятого интервала (p5)
    h_max_m: float             # верх занятого интервала (p95)
    span_m: float              # размах высот (p95-p5)
    points: int                # точек в интервале
    cells: int                 # уникальных ячеек 5 см (диагностика плотности)
    confidence: float          # 0..1
    bin_index: int = 0
    u_lo_m: float = 0.0
    u_hi_m: float = 0.0

    def to_dict(self) -> dict:
        out = asdict(self)
        for key, value in out.items():
            if isinstance(value, float):
                out[key] = round(float(value), 3)
        return out


@dataclass
class DetectionResult:
    obstacles: list = field(default_factory=list)
    # Диагностика
    points_total: int = 0
    points_volume: int = 0
    points_candidate: int = 0
    bins_total: int = 0
    bins_blocked: int = 0
    candidates_rejected_wall: int = 0   # интервалов дотянулось до границы габарита
    candidates_rejected_thin: int = 0   # интервалов не набрало высоты
    candidates_rejected_structure: int = 0  # кластер -- часть стены/платформы
    candidates_rejected_tall: int = 0   # кластер заполняет объём по высоте
    max_data_range_m: float = 0.0
    axis_range_m: tuple = (0.0, 0.0)
    blocked: list = field(default_factory=list)

    @property
    def blocked_any(self) -> bool:
        return bool(self.obstacles)

    def nearest(self) -> Optional[Obstacle]:
        return min(self.obstacles, key=lambda o: o.distance_m) if self.obstacles else None

    def to_dict(self) -> dict:
        return {
            'obstacles': [o.to_dict() for o in self.obstacles],
            'points_total': self.points_total,
            'points_volume': self.points_volume,
            'points_candidate': self.points_candidate,
            'bins_total': self.bins_total,
            'bins_blocked': self.bins_blocked,
            'candidates_rejected_wall': self.candidates_rejected_wall,
            'candidates_rejected_thin': self.candidates_rejected_thin,
            'candidates_rejected_structure': self.candidates_rejected_structure,
            'candidates_rejected_tall': self.candidates_rejected_tall,
            'max_data_range_m': round(self.max_data_range_m, 2),
            'axis_range_m': [round(float(v), 2) for v in self.axis_range_m],
            'blocked': self.blocked,
        }


# ---------------------------------------------------------------------------


def equipment_mask(u: np.ndarray, h: np.ndarray, model: TrackModel,
                   cfg: DetectorConfig) -> np.ndarray:
    """Маска штатного оборудования габарита: ходовые рельсы и кабельный лоток.

Рельс исключается ТОЛЬКО ниже `rail_h_max_m`: головка рельса -- 0.17 м над УГР,
и полоса рельса на всю высоту пробивала бы в габарите вертикальную дыру, разрывая
попавший на неё объект на два интервала (замерено тестом: плита при u = +-0.8
переставала находиться).
    """
    u = np.asarray(u, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    half_gauge = model.gauge_m / 2.0
    rails = (np.abs(np.abs(u) - half_gauge) <= cfg.rail_half_m) & (h <= cfg.rail_h_max_m)
    tray = ((np.abs(u) >= cfg.tray_u_m[0]) & (np.abs(u) <= cfg.tray_u_m[1])
            & (h >= cfg.tray_h_m[0]) & (h <= cfg.tray_h_m[1]))
    return rails | tray


def split_intervals(values: np.ndarray, gap: float) -> list:
    """Отсортированные значения -> интервалы [(lo, hi, mask), ...] с разрывом > gap.

    Разрыв по `u` -- это то, что отделяет объект от стены: у стены интервал
    непрерывно уходит наружу габарита, у объекта он замкнут.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return []
    order = np.argsort(values, kind='stable')
    sv = values[order]
    breaks = np.flatnonzero(np.diff(sv) > gap) + 1
    starts = np.r_[0, breaks]
    ends = np.r_[breaks, sv.size]
    out = []
    for a, b in zip(starts, ends):
        out.append((float(sv[a]), float(sv[b - 1]), order[a:b]))
    return out


def threshold_points(distance_m: float, cfg: DetectorConfig) -> int:
    """Порог точек с поправкой на число колец, попадающих в бин.

    Число колец в бине падает с дальностью (бино 1 м по пути), поэтому при
    `range_normalization` порог опускается пропорционально: ожидаемый счёт
    объекта ~ 1/d^2, и порог приводится к опорной дальности `range_ref_m`.
    """
    if not cfg.range_normalization:
        return int(cfg.min_points)
    d = max(float(distance_m), 1.0)
    scale = min(1.0, (cfg.range_ref_m / d) ** 2)
    return max(3, int(round(cfg.min_points * scale)))


def dense_span(values: np.ndarray, mass: float = DENSE_MASS) -> float:
    """Ширина минимального окна, вмещающего долю `mass` значений.

    Размах p5..p95 обманывается редкими выбросами: замерено на
    `roundT_doubleT` -- 43 точки лежат в полосе 0.4 м, а десяток одиночных точек
    растягивает размах до 1.6 м, и бина «препятствие». Настоящий же объект
    занимает по высоте СПЛОШНОЙ диапазон: 80 % его точек не могут уложиться в
    узкую полосу. Поэтому высота объекта меряется этим окном, а не перцентилями.
    """
    values = np.asarray(values, dtype=np.float64)
    n = values.size
    if n == 0:
        return 0.0
    if n == 1:
        return 0.0
    k = max(1, int(np.ceil(mass * n)))
    if k >= n:
        return float(values.max() - values.min())
    sv = np.sort(values)
    widths = sv[k - 1:] - sv[:n - k + 1]
    return float(widths.min())


def height_filled(hv: np.ndarray, cfg: 'DetectorConfig') -> tuple:
    """(размах, плотная полоса, проходит ли) -- высота объекта в бине.

    Размах p5..p95 -- сама высота, плотная полоса -- сколько из неё ЗАПОЛНЕНО
    точками. Абсолютного порога на плотную полосу мало: на 56 м кольца лидара
    расходятся на 0.3 м, и силуэт человека 0.9 м высотой даёт всего 3 кольца --
    замерено 0.37 м плотной полосы при размахе 0.60 (человек в
    `doubleT_obstacle`, кадр 25). Поэтому плотная полоса сравнивается со СВОИМ
    размахом: у настоящего объекта она занимает больше половины, у редкого
    хвоста выбросов -- меньше (43 точки в 0.4 м + десяток по 1.6 м дают 0.25).

    Пороги опущены до `min_span_m` = 0.35 при `dense_ratio` = 0.30 по замеру:
    на кадрах 30..45 силуэт человека в объёме даёт размах 0.36..0.39 м (низ
    объёма срезает его на 0.30 м над УГР), и прежние 0.5/0.5 его теряли. На
    всём корпусе это не дало ни одного ложного: от срезанного силуэта стены
    предмет отличают связность занятости и предел размаха, а не порог высоты.
    """
    hv = np.asarray(hv, dtype=np.float64)
    span = float(np.percentile(hv, 95) - np.percentile(hv, 5))
    dense = dense_span(hv, cfg.dense_mass)
    ok = (span >= cfg.min_span_m
          and dense >= cfg.min_span_m * cfg.dense_ratio
          and dense >= cfg.dense_ratio * span)
    return span, dense, bool(ok)


def _count_cells(u, h, bin_gap, cell_m) -> int:
    """Число уникальных ячеек 5 см в (y, u, h) -- мера плотности кластера."""
    if u.size == 0:
        return 0
    key = (np.floor(np.asarray(bin_gap) / cell_m).astype(np.int64) * 1000003
           + np.floor(u / cell_m).astype(np.int64)) * 100003
    key = key + np.floor(h / cell_m).astype(np.int64)
    return int(np.unique(key).size)


# ---------------------------------------------------------------------------
# Связность занятости в плоскости (y, u): стена против компактного предмета.

_CELL_KEY_M = 4096           # шаг упаковки (iy, iu) в одно число
_CELL_KEY_OFF = 2048         # сдвиг по u: годится для |u| < 2048 ячеек = 245 м


def cell_indices(u, y, cfg) -> tuple:
    """(iy, iu) -- ячейка сетки связности для точек (u, y)."""
    u = np.asarray(u, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    iy = np.floor(-y / cfg.wall_cell_y_m).astype(np.int64)
    iu = np.floor(u / cfg.wall_cell_u_m).astype(np.int64)
    return iy, iu


def occupancy_components(u, y, cfg) -> tuple:
    """Связные компоненты занятой сетки (y, u) -- «это часть конструкции?».

Возвращает `(label_by_cell, stats)`: «ячейка -> номер компонента» и статистику по
компонентам. Компонент помечен `wall`, если он ВЫХОДИТ НАРУЖУ габарита
(`|u| > structure_out_m`) или ТЯНЕТСЯ ВДОЛЬ ПУТИ дольше `max_structure_y_m`:
и то, и другое означает протяжённую конструкцию, край которой заходит в габарит.

Почему сетка, а не интервал в бине. Прежнее правило смотрело один бин: интервал,
дошедший до `|u| = 1.25`, считался стеной. Стена уходит наружу НЕ в том бине, где
виден её внутренний край (замерено на `roundT_doubleT`: внешняя часть -- бины
19..28, внутренний обрывок -- 29..31), поэтому обрывок проходил как «объект».
Связность по 8 соседям склеивает обрывок с наружной частью через соседние бины.

Размеры сетки подобраны по данным: ячейка `wall_cell_u_m` = 0.12 м (как
`u_gap_m`) не склеивает предмет со стеной, стоящей на другой стороне пути;
ячейка вдоль пути 1 м даёт человеку (`doubleT_obstacle`, 46 кадров) компонент
5..14 ячеек и 1..2 м по пути, а стене/платформе (`roundT_squareT_…`,
`doubleT_platform`) -- 45..277 ячеек и 11..46 м, то есть на порядок больше.
Стоимость -- один `np.unique` по точкам объёма и обход ~60..250 занятых ячеек
(замерено на кадре `roundT_squareT_…` 400: 16044 таких точек -> 210 ячеек,
0.6..0.8 мс).
    """
    iy, iu = cell_indices(u, y, cfg)
    if iy.size == 0:
        return {}, []
    key = iy * _CELL_KEY_M + (iu + _CELL_KEY_OFF)
    uniq, counts = np.unique(key, return_counts=True)
    busy = counts >= max(1, int(cfg.wall_min_cell_points))
    cells = [int(v) for v in uniq[busy]]
    if not cells:
        return {}, []
    count_of = {int(k): int(v) for k, v in zip(uniq, counts)}
    index = {c: i for i, c in enumerate(cells)}

    parent = list(range(len(cells)))

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    for c in cells:
        cy, cu = divmod(c, _CELL_KEY_M)
        for dy, du in ((1, 0), (-1, 0), (0, 1), (0, -1),
                       (1, 1), (1, -1), (-1, 1), (-1, -1)):
            other = index.get((cy + dy) * _CELL_KEY_M + cu + du)
            if other is None:
                continue
            ra, rb = find(index[c]), find(other)
            if ra != rb:
                parent[rb] = ra

    groups = {}
    for c in cells:
        groups.setdefault(find(index[c]), []).append(c)

    label_by_cell = {}
    stats = []
    for members in groups.values():
        rows = [m // _CELL_KEY_M for m in members]
        cols = [m % _CELL_KEY_M - _CELL_KEY_OFF for m in members]
        y_extent = (max(rows) - min(rows) + 1) * cfg.wall_cell_y_m
        u_min = min(cols) * cfg.wall_cell_u_m
        u_max = (max(cols) + 1) * cfg.wall_cell_u_m
        wall = bool(y_extent > cfg.max_structure_y_m
                    or u_max > cfg.structure_out_m or u_min < -cfg.structure_out_m)
        label = len(stats)
        stats.append(dict(label=label, cells=len(members),
                          points=sum(count_of[m] for m in members),
                          y_extent=float(y_extent), u_min=float(u_min),
                          u_max=float(u_max), wall=wall))
        for m in members:
            label_by_cell[(m // _CELL_KEY_M, m % _CELL_KEY_M - _CELL_KEY_OFF)] = label
    return label_by_cell, stats


def structure_is_wall(yy, uu, cfg, label_by_cell, stats) -> bool:
    """Принадлежит ли занятость кластера компоненту-конструкции (стене)."""
    if not stats:
        return False
    iy, iu = cell_indices(uu, yy, cfg)
    seen = set()
    for a, b in zip(iy.tolist(), iu.tolist()):
        label = label_by_cell.get((a, b))
        if label is None or label in seen:
            continue
        seen.add(label)
        if stats[label]['wall']:
            return True
    return False


def detect(xyz: np.ndarray,
           model: Optional[TrackModel] = None,
           cfg: Optional[DetectorConfig] = None,
           frame_index: Optional[int] = None) -> DetectionResult:
    """Препятствия в одном облаке точек.

    `xyz` -- (N, 3) точки в системе лидара. Туннель в −Y.
    Точки можно подавать ЛЮБЫЕ, в том числе синтетические: тест с внесённым
    объектом просто конкатенирует массив объекта с кадром.
    """
    model = model or TrackModel()
    cfg = cfg or DetectorConfig()
    xyz = np.asarray(xyz)
    result = DetectionResult()
    if xyz.ndim != 2 or xyz.shape[1] < 3:
        raise ValueError('xyz must be (N, 3)')
    result.points_total = int(xyz.shape[0])
    if xyz.shape[0] == 0:
        return result

    xyz = np.asarray(xyz, dtype=np.float64)
    y_all = xyz[:, 1]
    result.max_data_range_m = float(-np.min(y_all))

    y_far, y_near = model.valid_range(cfg.max_range_m, cfg.min_range_m)
    if not cfg.limit_to_axis_range:
        y_far, y_near = -cfg.max_range_m, -cfg.min_range_m
    result.axis_range_m = (round(y_far, 2), round(y_near, 2))

    in_range = (y_all >= y_far) & (y_all <= y_near)
    if not np.any(in_range):
        return result

    # Грубая отсечка по X и Z ДО интерполяции оси: интерполяция по полилинии на
    # 350 тыс. точек стоит ~22 мс, а после отсечки остаётся 3..20 % точек.
    # Границы берём по крайним узлам модели, поэтому отсечка ничего не теряет.
    if model.has_polyline:
        ax_lo = float(np.min(model.axis_x_m)) - cfg.u_out_m
        ax_hi = float(np.max(model.axis_x_m)) + cfg.u_out_m
        ugr_lo = float(np.min(model.ugr_z_m))
        ugr_hi = float(np.max(model.ugr_z_m))
    else:
        y_sel = y_all[in_range]
        ax_sel = model.axis_x(y_sel)
        ugr_sel = model.ugr_z(y_sel)
        ax_lo = float(np.min(ax_sel)) - cfg.u_out_m
        ax_hi = float(np.max(ax_sel)) + cfg.u_out_m
        ugr_lo = float(np.min(ugr_sel))
        ugr_hi = float(np.max(ugr_sel))

    x_all = xyz[:, 0]
    z_all = xyz[:, 2]
    pre = (in_range
           & (x_all >= ax_lo) & (x_all <= ax_hi)
           & (z_all >= ugr_lo + cfg.h_low_m) & (z_all <= ugr_hi + cfg.h_high_m))
    if not np.any(pre):
        return result

    local = xyz[pre]
    y_all = local[:, 1]
    u, h = model.relative(local)

    volume = ((np.abs(u) <= cfg.half_width_m)
              & (h >= cfg.h_low_m) & (h <= cfg.h_high_m))
    result.points_volume = int(np.count_nonzero(volume))

    # --- бин вдоль пути: номер бина по пройденной дальности
    bin_index_all = np.floor(-y_all / cfg.bin_m).astype(np.int64)
    n_bins = int(np.floor(-y_far / cfg.bin_m)) + 1

    # Точки «объёма плюс полоса наружу»: по ним видно, уходит структура за
    # габарит или заканчивается внутри.
    wide = ((np.abs(u) <= cfg.u_out_m)
            & (h >= cfg.h_low_m) & (h <= cfg.h_high_m)
            & ~equipment_mask(u, h, model, cfg))
    result.points_candidate = int(np.count_nonzero(wide))
    if result.points_candidate == 0:
        result.bins_total = n_bins
        result.blocked = [False] * n_bins
        return result

    sel = np.flatnonzero(wide)
    order = sel[np.argsort(bin_index_all[sel], kind='stable')]
    bins_sorted = bin_index_all[order]
    bounds = np.searchsorted(bins_sorted, np.arange(n_bins + 1))
    blocked = [False] * n_bins
    result.bins_total = n_bins
    # Связность занятости считается ДО склейки: компонент -- свойство всей
    # структуры, а не отдельного кандидата (см. `occupancy_components`).
    label_by_cell, structure_stats = occupancy_components(u[sel], y_all[sel], cfg)

    # Кандидаты собираем по бинам, а решение принимаем ПОСЛЕ склейки: настоящая
    # помеха занимает 1..3 бина, а стена/гермозатвор -- десяток. Отбрасывать
    # «вышедшие за габарит» интервалы сразу нельзя: стена тогда распадается на
    # внутренние куски, и каждый выглядит коротким объектом (замерено на
    # `roundT_doubleT`: 6 бинов подряд, из них 2 внутренних, и они давали
    # ложное срабатывание).
    candidates = []
    for b in range(n_bins):
        a, e = int(bounds[b]), int(bounds[b + 1])
        if e - a < cfg.min_points:
            continue
        idx = order[a:e]
        distance = float(-np.mean(y_all[idx]))
        need = threshold_points(distance, cfg)
        if e - a < need:
            continue
        uu = u[idx]
        hh = h[idx]
        yy = y_all[idx]
        for lo, hi, local in split_intervals(uu, cfg.u_gap_m):
            pts = local.size
            if pts < need:
                continue
            hv = hh[local]
            span, _dense, filled = height_filled(hv, cfg)
            if not filled:
                result.candidates_rejected_thin += 1
                continue
            uv = uu[local]
            width = float(np.percentile(uv, 98) - np.percentile(uv, 2))
            if width < cfg.min_width_m:
                result.candidates_rejected_thin += 1
                continue
            candidates.append((b, uv, hv, yy[local]))
            blocked[b] = True

    result.blocked = blocked
    result.obstacles = _merge_candidates(candidates, model, cfg, result, blocked,
                                         label_by_cell, structure_stats)
    result.bins_blocked = int(sum(blocked))
    result.obstacles.sort(key=lambda o: o.distance_m)
    return result


def _merge_candidates(candidates, model, cfg, result, blocked,
                      label_by_cell=None, structure_stats=None):
    """Склейка кандидатов соседних бинов в одно препятствие.

    Пока разрыв между бинами не больше одного бина, кандидаты -- одна
    структура. Отбраковка кластера:

    * компонент связности занятости (y, u) -- стена/платформа:
      `occupancy_components` помечает компонент `wall`, если он уходит наружу
      габарита или тянется вдоль пути дольше `max_structure_y_m`. Это
      ОСНОВНОЙ дискриминатор, и он обобщает прежнее «интервал доходит до
      кромки»: у стены наружу уходит не тот бин, где виден внутренний край;
    * длина вдоль пути больше `max_cluster_y_m` -- стена: счётный запасной
      вариант того же признака (у стены длина по пути такая же, как у стены,
      а не как у предмета);
    * размах высоты больше `max_span_m` -- стена, зашедшая в габарит наклонной
      поверхностью и заполнившая объём по всей высоте (замерено: у человека
      0.39..0.90 м, у стены 1.79..3.01 м);
    * занятый интервал по `u` ВЫХОДИТ за габарит (`edge_allow_m` = 0) --
      структура продолжается наружу, и внутрь габарита заходит только её край.

    Почему `edge_allow_m` равен нулю. Прежде это был запас (0.18 м) на
    неопределённость оси: интервал обязан был замкнуться до `|u| = 1.07`, а
    человек, идущий по кромке габарита, из-за этого терялся. Теперь «структура
    продолжается наружу» проверяет связность, а не один интервал, и запас не
    нужен: интервал, дошедший до самой `|u| = 1.25`, выдаётся, если он не часть
    конструкции.

    Счётчики отброшенного идут в диагностику, чтобы решение было видно в логе,
    а не молча.
    """
    if not candidates:
        return []
    candidates = sorted(candidates, key=lambda c: c[0])
    # Склейка: соседние бины И перекрытие (или близость) по u. Без проверки по u
    # объект приклеивался к редким точкам дальней стены в соседнем бине, и
    # длинный «стеновой» кластер утаскивал объект за собой (замерено: плита на
    # 15 м не находилась из-за 12-точечных обрывков стены на 13 и 15..19 м).
    clusters = [[candidates[0]]]
    for item in candidates[1:]:
        last = clusters[-1][-1]
        adjacent = item[0] - last[0] <= 1
        near_u = bool(item[1].max() >= last[1].min() - cfg.u_merge_m
                      and last[1].max() >= item[1].min() - cfg.u_merge_m)
        if adjacent and near_u:
            clusters[-1].append(item)
        else:
            clusters.append([item])

    limit = cfg.half_width_m + cfg.edge_allow_m
    obstacles = []
    for cluster in clusters:
        bins = [c[0] for c in cluster]
        # Длина кластера: от начала первого бина до конца последнего берём
        # фактический размах y, а не число бинов -- пропуски внутри допустимы.
        y_all = np.concatenate([c[3] for c in cluster])
        y_extent = float(y_all.max() - y_all.min()) + cfg.bin_m
        if y_extent > cfg.max_cluster_y_m:
            result.candidates_rejected_wall += 1
            for b in bins:
                blocked[b] = False
            continue
        uv = np.concatenate([c[1] for c in cluster])
        if structure_is_wall(y_all, uv, cfg, label_by_cell or {}, structure_stats or []):
            result.candidates_rejected_structure += 1
            for b in bins:
                blocked[b] = False
            continue
        if float(np.abs(uv).max()) > limit:
            result.candidates_rejected_wall += 1
            for b in bins:
                blocked[b] = False
            continue
        hv = np.concatenate([c[2] for c in cluster])
        span, _dense, filled = height_filled(hv, cfg)
        if not filled:
            result.candidates_rejected_thin += 1
            for b in bins:
                blocked[b] = False
            continue
        if span > cfg.max_span_m:
            result.candidates_rejected_tall += 1
            for b in bins:
                blocked[b] = False
            continue
        obstacles.append(_make_obstacle(uv, hv, y_all, model, cfg, bins[0], span))
    return obstacles


def _make_obstacle(u, h, y, model, cfg, bin_index, span) -> Obstacle:
    """Оформление найденного кластера в препятствие.

    Положение по `u` берётся по точкам В ГАБАРИТЕ (`u_m` -- медиана той части
    силуэта, которая реально попадает в свободный объём): у объекта у кромки
    часть точек лежит за `|u| = 1.25`, и медиана всего силуэта уводила бы
    дальность/координату объекта за габарит, к которому он только подходит.
    Высоты, наоборот, мерятся по всему силуэту -- это высота самого предмета.
    """
    u_in = u[np.abs(u) <= cfg.half_width_m]
    if u_in.size == 0:
        u_in = u
    u_c = float(np.median(u_in))
    y_c = float(np.median(y))
    h_lo = float(np.percentile(h, 5))
    h_hi = float(np.percentile(h, 95))
    x_c = u_c + float(model.axis_x(y_c))
    z_c = float(np.median(h)) + float(model.ugr_z(y_c))
    dist = float(np.sqrt(x_c ** 2 + y_c ** 2 + z_c ** 2))
    n = int(u.size)
    # Уверенность: запас по счёту точек, по высоте и по тому, насколько глубоко
    # силуэт заходит в габарит (у кромки запас меньше -- это честно).
    fill = min(1.0, n / (2.0 * max(cfg.min_points, 1)))
    tall = min(1.0, span / 2.0)
    reach = min(1.0, max(0.0, cfg.half_width_m - abs(u_c)) / max(cfg.min_width_m, 1e-3))
    conf = float(np.clip(0.4 * fill + 0.4 * tall + 0.2 * reach, 0.0, 1.0))
    return Obstacle(
        distance_m=dist,
        y_m=y_c,
        u_m=u_c,
        x_m=x_c,
        h_min_m=h_lo,
        h_max_m=h_hi,
        span_m=span,
        points=n,
        cells=_count_cells(u, h, y, cfg.cell_m),
        confidence=round(conf, 3),
        bin_index=int(bin_index),
        u_lo_m=float(u.min()),
        u_hi_m=float(u.max()),
    )


# ---------------------------------------------------------------------------
# Синтетический позитив: во всех записях настоящих объектов нет, поэтому
# детектор проверяется на внесённом объекте.


def synthetic_object(y_m: float, model: TrackModel, u_center_m: float = 0.0,
                     width_m: float = 0.70, height_m: float = 1.60,
                     cell_m: float = 0.06, h_base_m: float = 0.40,
                     thickness_m: float = 0.15) -> np.ndarray:
    """Облако точек поперечного объекта в габарите: (N, 3) в системе лидара.

    Объект -- вертикальная плита, перекрывающая путь: точки ставятся на
    ОБРАЩЁННУЮ К ДАТЧИКУ поверхность (датчик смотрит в −Y, значит видна
    поверхность с большим y). Именно так выглядит настоящее препятствие в
    облаке, поэтому и пороги проверяются на такой геометрии.

    Плотность точек соответствует сканированию: сетка `cell_m` по u и h, что при
    128 кольцах и FOV 100° даёт на 20..40 м десятки-сотни точек -- тот же
    порядок, что у настоящих объектов.
    """
    u_axis = np.arange(u_center_m - width_m / 2.0, u_center_m + width_m / 2.0 + 1e-9, cell_m)
    h_axis = np.arange(h_base_m, h_base_m + height_m + 1e-9, cell_m)
    uu, hh = np.meshgrid(u_axis, h_axis)
    uu = uu.ravel()
    hh = hh.ravel()
    yy = np.full(uu.shape, y_m + thickness_m / 2.0, dtype=np.float64)
    x = uu + model.axis_x(yy)
    z = hh + model.ugr_z(yy)
    return np.column_stack([x, yy, z]).astype(np.float32)


def wall_like_object(y_m: float, model: TrackModel, u_from_m: float = 0.60,
                     cell_m: float = 0.06, height_m: float = 3.0,
                     length_m: float = 6.0, h_base_m: float = 0.30) -> np.ndarray:
    """Протяжённая структура, уходящая за габарит, -- контроль «не объект».

    Плоскость вдоль пути от `u_from_m` до `u_from_m + 2.5` м (заведомо за
    `|u| = 1.25`): детектор обязан её игнорировать.
    """
    y_axis = np.arange(y_m - length_m / 2.0, y_m + length_m / 2.0 + 1e-9, cell_m)
    h_axis = np.arange(h_base_m, h_base_m + height_m + 1e-9, cell_m * 2)
    yy, hh = np.meshgrid(y_axis, h_axis)
    uu = np.full(yy.shape, u_from_m)
    yy = yy.ravel()
    uu = uu.ravel()
    hh = hh.ravel()
    x = uu + model.axis_x(yy)
    z = hh + model.ugr_z(yy)
    return np.column_stack([x, yy, z]).astype(np.float32)