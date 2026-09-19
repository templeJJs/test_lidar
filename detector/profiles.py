"""Модели пути: сохранённая полилиния, `track_geometry` или прямая.

`track_models.json` -- снимок `track_geometry.build_track(db, frame)` по шести
записям `for_hackathon/`: полилиния оси (середина колеи) и измеренный УГР
(верх головки рельса) с шагом 0.5 м, медиана по 9 кадрам записи. Он нужен
потому, что `track_geometry` тянет open3d, а в контейнере и в ROS-узле open3d
нет; плюс `build_track` стоит 0.8..1.3 с и в кадровый цикл не влезает.

Кадры стоят в РАЗНЫХ ПОЗАХ относительно пути (датчик доворачивается), поэтому
перед медианой ось каждого кадра КАНОНИЗУЕТСЯ -- из неё вычитается прямая
`x = shift + slope*y`, найденная по облаку кадра (`detector.core.axis_pose`).
Без этого медиана склеивает «вилку»: на `roundT_doubleT` ось, снятая по разным
кадрам, расходится на 3.66 м на 35 м (из них 2.9 м -- чистый доворот), и
габарит уезжает на метры. Детектор канонизирует облако той же формулой, так что
снимок и облако считаются в одной системе (см. шапку `detector/core.py`).

Ось в снимке продлена до `SNAPSHOT_Y_FAR_M` (см. `extend_polyline`): рельсы
измерены только до -30..-45 м, а человек в `doubleT_obstacle` стоит на 56 м, и
без продления детектор его просто не рассматривает. Наклон продления берётся по
хвосту измеренной полилинии (МНК по последним `TAIL_M` метрам), необязательная
кривизна -- из стен/свода (`rail_mesh.tunnel_axis`). Дальний конец участка, на
котором габарит ещё достоверен, лежит в `TrackModel.valid_far_m` и выбирается
регенератором `detector/build_models.py` по проверке на кадрах записи; в снимке
лежит блок `_meta` с тем, что было использовано (живой `build_track` или прежний
снимок), какие оси кадров вычтены канонизацией, какой разброс оси между кадрами
(до и после канонизации) и какая проверка пройдена.

Порядок выбора модели в `model_for_db`:

  1. сохранённая полилиния из `track_models.json` -- если имя записи известно;
  2. живой `track_geometry.build_track` -- если он импортируется и не падает
     (уточнение наклона оси для новой записи; канонизируется по облаку кадра,
     участок габарита у него консервативный, `LIVE_TRUSTED_EXTENSION_M`, --
     по кадрам он не проверялся);
  3. прямая `x = s*y + i` со средними параметрами -- последний резерв, даёт
     заведомо худший результат (прямая ось уводит габарит на поворотах),
     поэтому факт подстановки виден в `TrackModel.name` и `TrackModel.source`.

Таблица `MEASURED_STRAIGHT` -- те же записи, но прямая ось: нужна для
контрольного прогона, доказывающего, что дело именно в локальной оси.
"""

from __future__ import annotations

from dataclasses import replace
import json
import math
import os
from typing import Optional

from .core import TrackModel

_HERE = os.path.dirname(os.path.abspath(__file__))
TABLE_PATH = os.path.join(_HERE, 'track_models.json')

# Продление оси в снимке и параметры этого продления.
SNAPSHOT_Y_FAR_M = -110.0    # докуда ось продлена в снимке (облако видно до -208)
TAIL_M = 6.0                 # по какой длине хвоста берём наклон продления по X
Z_TAIL_M = 30.0              # ... а по Z окно шире: у живого меша хвост по z зажат
NODE_STEP_M = 0.5            # шаг узлов полилинии (как у build_track)
MAX_TAIL_SLOPE = 0.05        # |наклон| хвоста выше -- это уже выброс полилинии
MIN_MEASURED_M = 8.0         # меньше этого измеренной оси мало для продления
LIVE_TRUSTED_EXTENSION_M = 15.0   # участок габарита у НЕпроверенной живой модели

# Прямая ось по записям: (s, i) -- наклон и сдвиг `x = s*y + i`. Только для
# контроля. Прежние 4-е и 5-е значения кортежа (МНК-пол записи) были мёртвыми:
# оба потребителя (`model_for_name(straight=True)` и ROS-узел `model_node`)
# брали из кортежа только `s` и `i`.
MEASURED_STRAIGHT = {
    'doubleT_obstacle': (0.018276, -0.082130),
    'doubleT_platform': (0.004245, -0.008390),
    'roundT_doubleT': (-0.016923, 0.056538),
    'roundT_pressureGate_roundT': (0.004713, -0.006302),
    'roundT_squareT_pressureGate_squareT': (0.006765, -0.003162),
    'squareT_platform_squareT_switch': (0.006571, -0.005548),
}

FALLBACK = TrackModel(
    name='fallback-straight',
    gauge_m=1.593,
    axis_straight_s=0.0,
    axis_straight_i=0.0,
    source='fallback',
)


def _table() -> dict:
    try:
        with open(TABLE_PATH, encoding='utf-8') as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return {}


def table_meta() -> dict:
    """Блок `_meta` снимка: чем и как он сгенерирован (пусто, если снимка нет)."""
    return dict(_table().get('_meta') or {})


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
    """Модель пути по имени записи. `straight=True` -- прямая ось (контроль).

    Полилиния из снимка -- в КАНОНИЧЕСКОЙ системе (см. шапку модуля): детектор
    приводит к ней облако кадра, поэтому подменять её моделью в системе кадра
    нельзя. `straight=True` возвращает прямую `x = s*y + i` -- контрольный
    прогон, показывающий, сколько ложных даёт прямая ось; канонизацию детектор
    применяет и к ней (иначе облако с доворотом уехало бы ещё и относительно
    этой прямой).
    """
    if straight:
        data = MEASURED_STRAIGHT.get(name)
        if data is None:
            return FALLBACK
        s, i = data
        return TrackModel(name=f'{name} [straight]', gauge_m=1.593,
                          axis_straight_s=s, axis_straight_i=i,
                          source='straight')
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
        axis_extension_m=float(entry.get('axis_extension_m', 0.0)),
        valid_far_m=float(entry.get('valid_far_m', 0.0)),
        source='snapshot',
    )


def model_for_db(db_path: str, straight: bool = False) -> TrackModel:
    """Модель пути по пути к `.db3`."""
    name = record_name(db_path)
    if name is None:
        return FALLBACK
    return model_for_name(name, straight=straight)


# ---------------------------------------------------------------------------
# Полилиния оси из живой геометрии и её продление.


def mid_polyline(left, right, step: float = NODE_STEP_M) -> tuple:
    """Середина колеи по двум нитям: (y, x_axis, z_ugr) на общей сетке узлов.

    `left`/`right` -- узлы `[y, x_нити, z_верх]` (по возрастанию y). Нити
    измеряются независимо, поэтому их наборы узлов НЕ совпадают: в текущем
    `build_track` правая нить доходит до -40.75, левая до -36.75, а их длины
    разные (69 и 77 узлов). Поэтому обе нити интерполируются на общую сетку по
    перекрытию; без этого середина колеи просто не строится, и детектор молча
    откатывался на снимок.
    """
    import numpy as np

    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] < 3 or right.shape[1] < 3:
        raise ValueError('rails must be (N, 3)')
    if left.shape[0] < 2 or right.shape[0] < 2:
        raise ValueError('need at least two nodes per rail')
    left = left[np.argsort(left[:, 0])]
    right = right[np.argsort(right[:, 0])]
    y_lo = max(left[0, 0], right[0, 0])
    y_hi = min(left[-1, 0], right[-1, 0])
    if y_hi - y_lo < MIN_MEASURED_M:
        raise ValueError('rails do not overlap')
    n = max(2, int(round((y_hi - y_lo) / step)) + 1)
    y = np.linspace(y_lo, y_hi, n)
    xl = np.interp(y, left[:, 0], left[:, 1])
    xr = np.interp(y, right[:, 0], right[:, 1])
    zl = np.interp(y, left[:, 0], left[:, 2])
    zr = np.interp(y, right[:, 0], right[:, 2])
    return y, 0.5 * (xl + xr), 0.5 * (zl + zr)


def tail_slope(y, values, tail_m: float = TAIL_M) -> float:
    """Наклон по ДАЛЬНЕМУ хвосту полилинии (`y` по возрастанию, дальний -- y[0]).

    По двум соседним узлам наклон шумный (в снимке `doubleT_obstacle` шаг узла
    перескакивает -0.013..+0.040), а ошибка наклона прямо задаёт уход габарита
    на 20..80 м продления. Поэтому -- МНК по последним `tail_m` метрам.
    """
    import numpy as np

    y = np.asarray(y, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if y.size < 2:
        return 0.0
    order = np.argsort(y)
    y = y[order]
    values = values[order]
    sel = y <= y[0] + tail_m
    if np.count_nonzero(sel) < 2:
        sel = np.zeros_like(y, dtype=bool)
        sel[:min(2, y.size)] = True
    slope = float(np.polyfit(y[sel] - y[0], values[sel], 1)[0])
    if not np.isfinite(slope) or abs(slope) > MAX_TAIL_SLOPE:
        return 0.0
    return slope


def extend_polyline(model: TrackModel, y_far_m: float = SNAPSHOT_Y_FAR_M,
                    curvature: float = 0.0, tail_m: float = TAIL_M,
                    z_tail_m: float = Z_TAIL_M,
                    step: float = NODE_STEP_M) -> TrackModel:
    """Полилиния, продлённая в −Y до `y_far_m` по наклону хвоста и кривизне.

    Ось уходит в −Y как `x = x0 + s*Δ + c2*Δ²`, УГР -- как `z = z0 + sz*Δ`
    (Δ = y − y_начало_измеренного_хвоста, отрицательное). Кривизна `c2`
    (1/м) берётся из стен/свода (`rail_mesh.tunnel_axis`); при нулевой кривизне
    продление прямолинейно-касательное. Ошибка наклона на 60 м продления даёт
    ~0.2 м смещения оси, ошибка кривизны -- ~0.05 м, поэтому наклон считается
    МНК, а не по двум узлам.

    Окна по X и Z разные: хвост по X у живого меша настоящий (наклон рельсов
    замерен до -40 м), а по Z зажат на последних ~5 м (предел ухода меша), и
    короткое окно даёт нулевой наклон там, где пол уходит на 0.014 м/м
    (замерено на `doubleT_obstacle`). Поэтому Z берётся по более широкому окну.
    """
    import numpy as np

    if not model.has_polyline:
        return model
    y = np.asarray(model.y_nodes, dtype=np.float64)
    x = np.asarray(model.axis_x_m, dtype=np.float64)
    z = np.asarray(model.ugr_z_m, dtype=np.float64)
    if float(y.min()) <= y_far_m:
        return model
    sx = tail_slope(y, x, tail_m)
    sz = tail_slope(y, z, z_tail_m)
    d = np.arange(float(y.min()) - step, y_far_m - 1e-9, -step)
    if d.size == 0:
        return model
    delta = d - float(y.min())
    return TrackModel(
        name=model.name,
        gauge_m=model.gauge_m,
        y_nodes=tuple(d.tolist()) + tuple(y.tolist()),
        axis_x_m=tuple((x[0] + sx * delta + curvature * delta * delta).tolist()) + tuple(x.tolist()),
        ugr_z_m=tuple((z[0] + sz * delta).tolist()) + tuple(z.tolist()),
        axis_straight_s=model.axis_straight_s,
        axis_straight_i=model.axis_straight_i,
        axis_extension_m=model.axis_extension_m,
        valid_far_m=model.valid_far_m,
        source=model.source,
    )


def model_from_track(track: dict, name: Optional[str] = None) -> Optional[TrackModel]:
    """Модель пути из выдачи `track_geometry.build_track` (или None, если она пуста).

    Ось -- середина пары нитей (`mid_polyline`), УГР -- средний верх головок.
    Значения наклона прямой оси берутся из `track['axis']`, если они есть: это
    контрольный прогон «прямая ось», которым доказывается польза полилинии.
    """
    import numpy as np

    mesh = (track.get('meta') or {}).get('rail_mesh')
    if not mesh:
        return None
    rails = mesh.get('rails') or {}
    left = rails.get('left', {}).get('nodes')
    right = rails.get('right', {}).get('nodes')
    if left is None or right is None:
        return None
    y, x, z = mid_polyline(left, right)
    axis = track.get('axis') or {}
    gauge_mm = (track.get('meta') or {}).get('gauge_axis_mm')
    return TrackModel(
        name=name or 'live',
        gauge_m=float(gauge_mm) / 1000.0 if gauge_mm else 1.593,
        y_nodes=tuple(float(v) for v in y),
        axis_x_m=tuple(float(v) for v in x),
        ugr_z_m=tuple(float(v) for v in z),
        axis_straight_s=float(axis.get('s', 0.0)),
        axis_straight_i=float(axis.get('i', 0.0)),
        source='live',
    )


def tunnel_curvature(track: dict) -> float:
    """Кривизна продления оси (1/м) из стен/свода: `rail_mesh.tunnel_axis.c2`.

    Значение необязательное: `track_geometry` правится параллельно, и поля может
    не быть. Берём `c2_gated` (только подтверждённые кандидаты), иначе `c2`;
    мусорные и слишком большие значения игнорируем -- ошибка кривизны на 60 м
    продления может увести габарит на метры.
    """
    mesh = (track.get('meta') or {}).get('rail_mesh') or {}
    node = mesh.get('tunnel_axis')
    if not isinstance(node, dict):
        return 0.0
    for key in ('c2_gated', 'c2'):
        try:
            value = float(node.get(key))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and abs(value) < 1e-3:
            return value
    return 0.0


def build_polyline(db_path: str, frame: int = 0) -> Optional[TrackModel]:
    """Полилиния оси из `track_geometry.build_track`; None -- если недоступен.

    Импорт ленивый и обёрнут: `track_geometry` правится параллельно и тянет
    open3d, которого в контейнере нет. Падение чужого модуля не должно ронять
    детектор -- вызывающий откатывается на сохранённую таблицу, а факт подмены
    виден в `TrackModel.source` ('live' против 'snapshot').

    Ось кадра КАНОНИЗУЕТСЯ той же прямой, что в снимке и в детекторе
    (`detector.core.axis_pose` по облаку кадра): модель по одному кадру лежит в
    системе ЭТОГО кадра, а детектор работает в канонической системе -- без
    канонизации живая модель и облако разошлись бы на метры на 30..40 м.
    Облако читается из той же записи (`bag_reader`); если ни `bag_reader`, ни
    кадр недоступны, возвращается None, и вызывающий берёт снимок.

    Живой модели ось продлевается так же, как в снимке (`extend_polyline`), но
    участок габарита берётся КОНСЕРВАТИВНО -- измеренные узлы минус
    `LIVE_TRUSTED_EXTENSION_M`. Запись не проверялась по кадрам (это делает
    `build_models.py`, и результат лежит в снимке), а габарит на продлении
    уезжает тем сильнее, чем длиннее продление.
    """
    try:
        import numpy as np

        import bag_reader

        from .core import axis_pose

        from track_geometry import build_track

        track = build_track(str(db_path), int(frame))
        model = model_from_track(track, name=record_name(db_path) or 'live')
        if model is None:
            return None
        y_meas_far = float(min(model.y_nodes))
        bag = bag_reader.BagFrames(str(db_path), prefetch_ahead=0)
        try:
            cloud = np.asarray(bag[int(frame)].xyz, dtype=np.float64)
        finally:
            bag.close()
        pose = axis_pose(cloud, model)
        if pose.applied:
            model = replace(model, axis_x_m=tuple(
                float(v) - float(pose.line(yn))
                for yn, v in zip(model.y_nodes, model.axis_x_m)))
        model = extend_polyline(model, SNAPSHOT_Y_FAR_M,
                               curvature=tunnel_curvature(track))
        return replace(model, valid_far_m=y_meas_far - LIVE_TRUSTED_EXTENSION_M)
    except Exception:                     # noqa: BLE001 -- чужой модуль/данные
        return None