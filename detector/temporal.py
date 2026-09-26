"""Временной учёт детектора: накопление соседних кадров и подтверждение серией.

Покадровый `detector.core.detect` остаётся основным и рабочим сам по себе; этот
модуль -- ОБЁРТКА, которая добавляет два независимых механизма поверх серии
кадров (10 Гц, подряд):

1) НАКОПИТЕЛЬ (усилитель слабых кандидатов). Кадр `t` обрабатывается вместе с
   точками кадров `t-1..t-K` (K = `window`), перенесёнными в систему кадра `t`:
   продольно -- по интегралу профиля хода (`labeling.motion_between`; сюда
   приходит готовый callable, сам модуль `labeling` не импортирует), поперёк --
   через канонизацию (`axis_pose` из `detector/core.py`: у каждого кадра
   вычитается его ось `x0 + s*y`, точки складываются в канонической системе и
   возвращаются в систему кадра `t` позой кадра `t`).

   Прошлые кадры прореживаются (доля `keep`), а счётные пороги merged-прохода
   (`min_points` и пороги канала лотка) умножаются на ИЗМЕРЕННУЮ плотность
   объединённого облака (точек merged / точек кадра) -- иначе на статике
   счётчики «точек на бин» раздуваются в (1 + K*keep) раз, и порог фактически
   снижается. Формовые правила (размах высоты, связность, длина кластера) от
   плотности не зависят и не трогаются; порог занятости ячейки стены тоже не
   масштабируется (замер на `doubleT_platform`: разрежённая бахрома стены в
   объединении не густеет пропорционально, и порог x3 отрезал ей компоненту --
   стена переставала «доходить наружу» и выдавалась как объект).

   Две дополнительные страховки merged-прохода, обе по замерам на ложных
   сериях первого прогона (до них: 28 лишних событий на пустых записях
   корпуса):

   * КАНАЛ ЛОТКА в merged-проходе выключен (`boost_trough=False`): его пороги
     «плоской плиты» подогнаны впритык, и утроение точек перевалило
     `trough_min_top_m` на пороге гермозатвора (кадр 63
     `roundT_pressureGate_roundT` -- серия 63-65). Лежачих лечит трекер на
     покадровых детекциях лотка (M из N + коастинг), а тела совсем без
     возвышенных точек накоплением не вылечить -- копии тех же точек.
   * ГЕЙТ ПОДДЕРЖКИ (`boost_min_current`): усиленный кандидат обязан иметь
     точки в ТЕКУЩЕМ кадре (максимум из 8 и покадрового счётного порога).
     «Фантомы» -- точки прошлых кадров там, куда текущий кадр смотрит сквозь
     или мимо (верх свода у края достоверности оси на 46-58 м в
     `roundT_doubleT`, серии 156-184) -- имеют 0-10 точек текущего кадра; слабый
     настоящий объект (человек якоря на 56 м) -- 48-84, минимум по его слабым
     кадрам -- 20. Порог 12 лежит в замеренном разрыве.

   Замеры, на которых выбраны параметры (скрипт `.scratch/_tmp_sweep.py`):
   якорь `doubleT_obstacle` (человек на 56 м, 56-84 точки/кадр, покадровый
   разрыв 32-36) -- K=4 с keep=0.5 закрывает кадры 32-33 (на 34-36 силуэт
   срезан низом объёма до полосы h 0.36..0.64: размах 0.27 м меньше
   `min_span_m` = 0.35 у всех кадров сразу, накопление высоты не
   добавляет -- эти кадры закрывает коастинг трека, см. ниже); контроль ложных
   на `roundT_doubleT` кадры 107..152 (там до связности занятости было 15
   ложных событий) -- 0 событий при K=4 keep=0.5 (при keep=0.25 появляется 1
   событие на кадре 109: слишком редкое прореживание ломает структуру); размаз
   стены на едущей записи (ход 1.6-3.3 м между кадрами 100-102): после переноса
   остаток профиля стены 0.043-0.071 м (до переноса 0.057-0.095), то есть
   статика совмещается с субъячеечной точностью и не размазывается.

   Риск «движущийся объект размазывается вдоль пути»: идущий 1 м/кадр за 4
   кадра даёт структуру ~4-5 м вдоль пути -- она ОТБРАСЫВАЕТСЯ правилом
   `max_cluster_y_m` (4 м) как протяжённая, то есть размаз не порождает ложных
   препятствий, а сам движущийся объект по-прежнему находится ПОКАДРОВЫМ
   проходом (накопление -- только усилитель, не замена кадра). Поэтому окно
   не уменьшалось до 2: K=4 нужен якорю (кадр 32 не набирает формы на K<=3),
   а расплаты за него нет.

2) ПОДТВЕРЖДЕНИЕ СЕРИЕЙ (M из N). Детекции обоих проходов складываются в
   треки: состояние (y, u, h) с постоянной скоростью по y (у статики -- ноль;
   ход сенсора между кадрами вычитается тем же интегралом профиля), жадная
   ассоциация по путевым координатам. Покадровые детекции выдаются СРАЗУ
   (`frame_passthrough` по умолчанию: ядро уже имеет 0 ложных на корпусе, а
   обязательная серия резала бы recall ниже покадрового -- замер по датасету:
   person 40+ 0.216 против 0.258, equipment 20-40 0.339 против 0.458, потому
   что объекты с единственным попаданием в отрезке серию не набирают).
   Подтверждение (hits >= M из последних N, по умолчанию 2 из 3) сторожит
   именно то, что ядро НЕ видело: УСИЛЕННЫЕ кандидаты и коастинг. Одиночная
   усиленная вспышка (шум, фантом из прошлых кадров) наружу не выходит.
   Подтверждённый трек переживает до `coast` кадров без детекций (выдаётся
   предсказанная позиция): так закрываются короткие провалы видимости внутри
   серии (кадры 34-36 якоря).

   Задержка подтверждения обрезает НАЧАЛО серии усиленных попаданий (первый
   кадр ждёт второго). Для оффлайн-оценки это компенсируется «задним числом»:
   когда трек подтверждается, его усиленные попадания на предыдущих кадрах
   возвращаются в `TemporalResult.retro` -- по записи серия засчитывается
   целиком.

   Уверенность подтверждённого препятствия: `0.6 * hits/N + 0.4 * conf_детекции`,
   умноженное на стабильность `1 / (1 + std(u) + std(h))` по последним N
   попаданиям -- устойчивый трек с устойчивой геометрией получает ~1, дрожащий
   -- меньше. Кладётся в `Obstacle.confidence`, которым жадно сортирует
   сопоставление `validation/dataset_eval.match_objects`.

Применение по умолчанию -- усилитель: детекции кадра идут в трекер сами по
себе, накопление лишь добавляет попадания там, где кадр слаб. Оба механизма
отключаемы (`window=0` -- без накопления, `m=1, n=1` -- без подтверждения).
"""

from __future__ import annotations

import dataclasses
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .core import (AxisPose, DetectionResult, DetectorConfig, Obstacle,
                   TrackModel, axis_pose, detect, threshold_points)

# --- накопитель -------------------------------------------------------------
ACCUM_WINDOW = 4         # K: прошлых кадров в накоплении (см. замеры в шапке)
ACCUM_KEEP = 0.5         # доля точек прошлого кадра после прореживания
ACCUM_SEED = 20260915    # seed прореживания (воспроизводимость по номеру кадра)
BOOST_MIN_CURRENT = 12   # усиленный кандидат обязан иметь столько точек в ТЕКУЩЕМ
                         # кадре. Замеренный разрыв: «фантомы» из прошлых кадров
                         # (верх свода у края оси, roundT_doubleT 156-184) имеют
                         # поддержку 0-10, слабый настоящий объект (человек якоря
                         # на 56 м) -- 48-84, минимум по его слабым кадрам -- 20.
# --- трекер -----------------------------------------------------------------
TRACK_M = 2              # тревога: не меньше M попаданий ...
TRACK_N = 3              # ... из последних N кадров
TRACK_COAST = 3          # кадров без детекций, которые переживает подтверждённый трек
                         # (замер: разрыв якоря 32-36 -- накопление держит 32-33,
                         # коастинг 3 закрывает 34-36; хвост серии растёт на 1
                         # кадр против coast=2, ложных это не добавляет)
ASSOC_Y_M = 2.0          # ворота ассоциации вдоль пути
ASSOC_U_M = 0.9          # ... поперёк
ASSOC_H_M = 0.8          # ... по высоте
TRACK_MAX_VY = 3.0       # м/кадр: быстрее по записи ничего не движется
VY_SMOOTH = 0.5          # вес нового замера скорости трека


def motion_between_from_profile(profile) -> Callable[[int, int], float]:
    """Callable `dy(f_from, f_to)` по словарю `labeling.motion_profile`.

    Семантика та же, что у `labeling.motion_between`: интеграл профиля хода по
    кадрам между `f_from` и `f_to` (знак -- как у сдвига координаты статичной
    точки: `y_b = y_a + dy`), недостоверные кадры заполняются медианой
    (`m_per_frame`). Продублировано здесь, чтобы `detector` не импортировал
    `labeling` (он тянет `check_motion` и вьюерную обвязку).
    """
    if not isinstance(profile, dict) or not profile.get('profile'):
        return lambda a, b: 0.0
    prof = list(profile['profile'])
    fill = profile.get('m_per_frame')
    if fill is None:
        vals = [v for v in prof if v is not None]
        fill = float(np.median(vals)) if vals else 0.0
    fill = float(fill)
    step = int(profile.get('profile_step') or 1)

    def between(a, b):
        a, b = int(a), int(b)
        total = 0.0
        rng = range(a, b, step) if b >= a else range(a - step, b - step, -step)
        sign = 1.0 if b >= a else -1.0
        for i in rng:
            v = prof[i] if 0 <= i < len(prof) else None
            total += sign * (float(v) if v is not None else fill)
        return total

    return between


def warp_into_frame(xyz, pose_from: AxisPose, pose_to: AxisPose, dy: float):
    """Точки кадра `from` в системе кадра `to`: канонизация + интеграл хода.

    Из X вычитается ось кадра `from` (`axis_pose`), к Y прибавляется ход
    сенсора `dy` (`motion_between(from, to)`), затем прибавляется ось кадра
    `to`. Точка статичной сцены после этого ложится туда же, где её видел бы
    кадр `to` (замер остатка стены после переноса: 0.043-0.071 м на ходе
    1.6-3.3 м/кадр).
    """
    out = np.asarray(xyz, dtype=np.float64).copy()
    if pose_from is not None and pose_from.applied:
        out[:, 0] -= pose_from.line(out[:, 1])
    out[:, 1] += float(dy)
    if pose_to is not None and pose_to.applied:
        out[:, 0] += pose_to.line(out[:, 1])
    return out


def density_scaled_cfg(cfg: DetectorConfig, density: float) -> DetectorConfig:
    """Конфиг merged-прохода: счётные пороги x измеренную плотность облака.

    Статичная сцена в объединённом облаке плотнее в (1 + K*keep) раз; без
    нормировки порог «точек на бин» фактически снижался бы во столько же.
    Геометрические правила (размах высоты, связность, длина кластера) не
    зависят от плотности и не масштабируются.

    `wall_min_cell_points` НЕ масштабируется (замерено на `doubleT_platform`):
    разрежённая бахрома стены (1-3 точки на ячейку) в объединённом облаке не
    густеет пропорционально, и порог x3 отрезал ей компоненту -- стена
    переставала «доходить наружу» и выдавалась как объект.
    """
    f = max(1.0, float(density))
    out = dataclasses.replace(cfg)
    out.min_points = max(1, int(round(cfg.min_points * f)))
    out.trough_min_points = max(1, int(round(cfg.trough_min_points * f)))
    out.trough_min_cluster_points = max(
        1, int(round(cfg.trough_min_cluster_points * f)))
    return out


@dataclass
class Track:
    """Трек препятствия: (y, u, h) с постоянной скоростью по y.

    `y_m` -- в системе ПОСЛЕДНЕГО кадра попадания; при предсказании на кадр
    `t` к нему добавляется ход сенсора `dy(last, t)` (статичный объект едет по
    кадру вместе со сценой) и собственный ход `vy * (t - last)`.
    """

    track_id: int
    y_m: float
    u_m: float
    h_m: float                    # середина занятого интервала по высоте
    vy: float = 0.0               # собственная скорость вдоль пути, м/кадр
    last_frame: int = 0
    last_hit_frame: int = 0
    hits: int = 0
    window: deque = field(default_factory=deque)   # bool по последним N кадрам
    last_obstacle: Optional[Obstacle] = None
    last_source: str = ''           # 'frame' | 'accum' -- проход последнего попадания
    history: dict = field(default_factory=dict)    # frame -> (Obstacle, source)
    recent_u: deque = field(default_factory=deque)
    recent_h: deque = field(default_factory=deque)
    confirmed: bool = False
    coasted: int = 0              # подряд кадров без детекции (после подтверждения)

    def hits_in_window(self, n: int) -> int:
        return int(sum(list(self.window)[-n:]))


@dataclass
class TemporalResult:
    """Результат кадра с временным учётом.

    `obstacles` -- тревоги ЭТОГО кадра (онлайн): детекции подтверждённых
    треков плюс предсказания коастинга. `retro` -- попадания ПРОШЛЫХ кадров
    трека, подтвердившегося только сейчас: оффлайн-оценка засчитывает их тем
    кадрам, онлайн их не видел. `result` -- покадровый DetectionResult,
    `accum_result` -- результат merged-прохода (None, если накопление не
    запускалось).
    """

    obstacles: list = field(default_factory=list)
    retro: dict = field(default_factory=dict)
    result: Optional[DetectionResult] = None
    accum_result: Optional[DetectionResult] = None
    tracks: list = field(default_factory=list)


class TemporalDetector:
    """Обёртка над `detect`: накопление соседних кадров + подтверждение M из N.

    `motion_between` -- callable `(f_from, f_to) -> метры` (ход сенсора; берётся
    из `labeling.motion_profile` через `motion_between_from_profile`). Без него
    продольный перенос нулевой: корректно для стоящей записи, на едущей
    накопление размажет сцену -- профиль обязателен.

    Кадры обязаны идти подряд (10 Гц). Несмежный `frame_index` (разрыв записи,
    новый отрезок разметки) разрывает накопление: прошлые кадры с несмежными
    номерами в объединение не берутся; треки переживают разрыв по воротам
    ассоциации, но дешевле вызвать `reset()`.
    """

    def __init__(self, model: Optional[TrackModel] = None,
                 cfg: Optional[DetectorConfig] = None,
                 motion_between: Optional[Callable[[int, int], float]] = None,
                 window: int = ACCUM_WINDOW, keep: float = ACCUM_KEEP,
                 m: int = TRACK_M, n: int = TRACK_N, coast: int = TRACK_COAST,
                 boost: bool = True, scale_density: bool = True,
                 boost_trough: bool = False,
                 boost_min_current: int = BOOST_MIN_CURRENT,
                 frame_passthrough: bool = True):
        self.model = model or TrackModel()
        self.cfg = cfg or DetectorConfig()
        self.motion_between = motion_between or (lambda a, b: 0.0)
        self.window = max(0, int(window))
        self.keep = min(1.0, max(0.0, float(keep)))
        self.m = max(1, int(m))
        self.n = max(self.m, int(n))
        self.coast = max(0, int(coast))
        self.boost = bool(boost)
        self.scale_density = bool(scale_density)
        # Канал лотка в merged-проходе по умолчанию ВЫКЛЮЧЕН: его пороги
        # «плоской плиты» подогнаны впритык (замер: на кадре 63
        # roundT_pressureGate_roundT объединённое облако перевалило
        # `trough_min_top_m` на пороге гермозатвора, которого покадровый
        # прогон не видит), а лежачие тела без возвышенных точек накоплением
        # всё равно не лечатся -- точки копировать бессмысленно. Трекер
        # (M из N + коастинг) работает и на покадровых детекциях лотка.
        self.boost_trough = bool(boost_trough)
        self.boost_min_current = max(0, int(boost_min_current))
        # Покадровые детекции выдаются сразу (ядро и так 0 ложных на корпусе);
        # подтверждение серией M из N сторожит только УСИЛЕННЫЕ кандидаты и
        # коастинг. Иначе (False) серия обязательна для всего -- но тогда
        # recall по датасету падает ниже покадрового (объекты с одним
        # попаданием в отрезке серию не набирают: замер person 40+ 0.216
        # против 0.258 покадрово).
        self.frame_passthrough = bool(frame_passthrough)
        self._history = deque(maxlen=self.window + 1)  # (frame, xyz64, pose)
        self._tracks = []
        self._next_id = 1
        self._t = None

    def reset(self):
        """Сброс состояния (новая запись / новый отрезок кадров)."""
        self._history.clear()
        self._tracks = []
        self._t = None

    # ------------------------------------------------------------- кадр

    def update(self, xyz, frame_index: Optional[int] = None) -> TemporalResult:
        """Обработать очередной кадр: (тревоги кадра, ретро-попадания)."""
        xyz = np.asarray(xyz)
        if frame_index is None:
            frame_index = 0 if self._t is None else self._t + 1
        t = int(frame_index)
        self._t = t

        result = detect(xyz, model=self.model, cfg=self.cfg)
        xyz64 = np.asarray(xyz, dtype=np.float64)
        pose_t = axis_pose(xyz64, self.model) if self.cfg.canonicalize \
            else AxisPose()

        detections = [(o, 'frame') for o in result.obstacles]
        accum_result = None
        if self.boost and self.window > 0 and self._history:
            accum, density = self._accumulate(xyz, t, pose_t)
            if accum is not None:
                cfg_acc = (density_scaled_cfg(self.cfg, density)
                           if self.scale_density else self.cfg)
                if not self.boost_trough:
                    cfg_acc = dataclasses.replace(cfg_acc, trough_enabled=False)
                accum_result = detect(accum, model=self.model, cfg=cfg_acc)
                support = None
                for o in accum_result.obstacles:
                    # Копия покадровой детекции вторым проходом не нужна.
                    if any(abs(o.y_m - f.y_m) <= ASSOC_Y_M
                           and abs(o.u_m - f.u_m) <= ASSOC_U_M
                           for f, _src in detections):
                        continue
                    # Гейт поддержки: усилитель усиливает СЛАБЫЙ кандидат
                    # текущего кадра, а не изобретает его из прошлых кадров.
                    # У «фантомов» (точки прошлых кадров там, где текущий кадр
                    # смотрит сквозь, -- верх свода у края оси) поддержка
                    # 0-10 точек против 20-84 у слабого человека на 56 м
                    # (замер; порог -- BOOST_MIN_CURRENT).
                    if self.boost_min_current > 0:
                        if support is None:
                            support = self._current_support(xyz64, pose_t)
                        if not self._has_support(o, support):
                            continue
                    detections.append((o, 'accum'))

        retro = {}
        self._feed_tracks(detections, t, pose_t)
        obstacles = []
        for tr in self._tracks:
            if not tr.confirmed and tr.hits_in_window(self.n) >= self.m:
                tr.confirmed = True
                # Ретро-попадания (усиленные, оффлайн-оценке): покадровые
                # детекции и так выданы в свои кадры (frame_passthrough).
                for f, (ob, src) in tr.history.items():
                    if f < t and (src != 'frame' or not self.frame_passthrough):
                        retro.setdefault(f, []).append(ob)
            if tr.last_hit_frame == t and tr.last_obstacle is not None:
                if self.frame_passthrough and tr.last_source == 'frame':
                    # Покадровая детекция выдаётся сразу, как у detect: ядро
                    # уже имеет 0 ложных на корпусе, а подтверждение серией
                    # для неё резало бы recall (замер по датасету: person 40+
                    # 0.216 против 0.258 покадрово -- объекты с единственным
                    # попаданием в отрезке серию не набирают). M из N сторожит
                    # именно УСИЛИТЕЛЬ -- там и живёт риск ложных.
                    obstacles.append(tr.last_obstacle)
                elif tr.confirmed:
                    obstacles.append(self._scored(tr, tr.last_obstacle))
            elif tr.confirmed and 0 < t - tr.last_hit_frame <= self.coast \
                    and tr.last_obstacle is not None:
                obstacles.append(self._scored(tr, self._predicted(tr, t, pose_t)))
        stale = self.n + self.coast + 1    # дольше без попаданий -- трек удаляется
        self._tracks = [tr for tr in self._tracks
                        if t - tr.last_hit_frame <= stale]
        self._history.append((t, np.asarray(xyz, dtype=np.float64), pose_t))
        obstacles.sort(key=lambda o: o.distance_m)
        return TemporalResult(obstacles=obstacles, retro=retro, result=result,
                              accum_result=accum_result,
                              tracks=[self._track_info(tr) for tr in self._tracks])

    # -------------------------------------------------------- накопитель

    def _accumulate(self, xyz, t, pose_t):
        """Объединённое облако кадра t + прошлые кадры в системе кадра t.

        Берутся только СМЕЖНЫЕ номера (t-1, t-2, ...): при разрыве записи
        дальний кадр переносить через неизвестный ход нельзя. Возвращает
        (облако, плотность=точек_merged/точек_кадра) или (None, 1.0).
        """
        parts = [np.asarray(xyz, dtype=np.float64)]
        rng = np.random.default_rng(ACCUM_SEED + t)
        by_frame = {f: (c, p) for f, c, p in self._history}
        used = 0
        for f in range(t - 1, t - self.window - 1, -1):
            got = by_frame.get(f)
            if got is None:
                break
            cloud_f, pose_f = got
            dy = self.motion_between(f, t)
            moved = warp_into_frame(cloud_f, pose_f, pose_t, dy)
            if self.keep < 1.0:
                moved = moved[rng.random(moved.shape[0]) < self.keep]
            parts.append(moved)
            used += 1
        if not used:
            return None, 1.0
        merged = np.vstack(parts)
        density = merged.shape[0] / float(max(parts[0].shape[0], 1))
        return merged, density

    def _current_support(self, xyz64, pose_t):
        """(y, u, h) точек ТЕКУЩЕГО кадра в канонической системе -- для гейта."""
        canon = xyz64
        if pose_t is not None and pose_t.applied:
            canon = xyz64.copy()
            canon[:, 0] -= pose_t.line(canon[:, 1])
        u, h = self.model.relative(canon, canonical=False)
        return canon[:, 1], u, h

    def _has_support(self, ob: Obstacle, support) -> bool:
        """Хватает ли точек текущего кадра в боксе усиленного кандидата."""
        y_cur, u_cur, h_cur = support
        box = ((np.abs(y_cur - ob.y_m) <= 1.5)
               & (u_cur >= ob.u_lo_m - 0.15) & (u_cur <= ob.u_hi_m + 0.15)
               & (h_cur >= ob.h_min_m - 0.15) & (h_cur <= ob.h_max_m + 0.15))
        need = max(self.boost_min_current,
                   threshold_points(float(ob.distance_m), self.cfg))
        return int(np.count_nonzero(box)) >= need

    # ---------------------------------------------------------- трекер

    def _feed_tracks(self, detections, t, pose_t):
        """Жадная ассоциация детекций к трекам по путевым координатам."""
        for tr in self._tracks:
            tr.window.append(False)
            while len(tr.window) > self.n:
                tr.window.popleft()
        free = list(range(len(detections)))
        # Ближайшие пары первыми: при половине десятка треков жадной достаточно
        # (венгерская понадобится, когда объектов станет много).
        pairs = []
        for ti, tr in enumerate(self._tracks):
            y_pred = (tr.y_m + self.motion_between(tr.last_frame, t)
                      + tr.vy * (t - tr.last_frame))
            for di in free:
                ob = detections[di][0]
                dy = abs(ob.y_m - y_pred)
                du = abs(ob.u_m - tr.u_m)
                dh = abs(0.5 * (ob.h_min_m + ob.h_max_m) - tr.h_m)
                if dy <= ASSOC_Y_M and du <= ASSOC_U_M and dh <= ASSOC_H_M:
                    cost = (dy / ASSOC_Y_M) ** 2 + (du / ASSOC_U_M) ** 2 \
                        + (dh / ASSOC_H_M) ** 2
                    pairs.append((cost, ti, di))
        pairs.sort()
        used_t, used_d = set(), set()
        for cost, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            self._hit(self._tracks[ti], detections[di][0], t, detections[di][1])
        for di in free:
            if di in used_d:
                continue
            ob, src = detections[di]
            tr = Track(track_id=self._next_id,
                       y_m=float(ob.y_m), u_m=float(ob.u_m),
                       h_m=0.5 * float(ob.h_min_m + ob.h_max_m))
            self._next_id += 1
            self._tracks.append(tr)
            self._hit(tr, ob, t, src)

    def _hit(self, tr: Track, ob: Obstacle, t: int, source: str = 'frame'):
        dt = t - tr.last_hit_frame if tr.hits else 0
        if tr.hits and dt > 0:
            dy_self = (float(ob.y_m) - tr.y_m
                       - self.motion_between(tr.last_hit_frame, t))
            vy = float(np.clip(dy_self / dt, -TRACK_MAX_VY, TRACK_MAX_VY))
            tr.vy = (1.0 - VY_SMOOTH) * tr.vy + VY_SMOOTH * vy
        tr.y_m = float(ob.y_m)
        tr.u_m = float(ob.u_m)
        tr.h_m = 0.5 * float(ob.h_min_m + ob.h_max_m)
        tr.hits += 1
        tr.last_frame = t
        tr.last_hit_frame = t
        tr.last_obstacle = ob
        tr.last_source = source
        tr.history[t] = (ob, source)
        tr.coasted = 0
        if tr.window:
            tr.window[-1] = True
        else:
            tr.window.append(True)
        while len(tr.window) > self.n:
            tr.window.popleft()
        tr.recent_u.append(tr.u_m)
        tr.recent_h.append(tr.h_m)
        while len(tr.recent_u) > self.n:
            tr.recent_u.popleft()
            tr.recent_h.popleft()

    def _predicted(self, tr: Track, t: int, pose_t: AxisPose) -> Obstacle:
        """Предсказанная позиция подтверждённого трека на кадре без детекции."""
        ob = tr.last_obstacle
        y_new = (tr.y_m + self.motion_between(tr.last_hit_frame, t)
                 + tr.vy * (t - tr.last_hit_frame))
        x_c = tr.u_m + float(self.model.axis_x(np.asarray([y_new]))[0]) \
            + float(pose_t.line(np.asarray([y_new]))[0])
        z_c = tr.h_m + float(self.model.ugr_z(np.asarray([y_new]))[0])
        dist = float(np.sqrt(x_c ** 2 + y_new ** 2 + z_c ** 2))
        out = dataclasses.replace(ob, y_m=float(y_new), x_m=float(x_c),
                                  distance_m=dist)
        tr.coasted += 1
        return out

    def _scored(self, tr: Track, ob: Obstacle) -> Obstacle:
        """Уверенность трека: hits/N, стабильность u/h и уверенность детекции."""
        hits_frac = tr.hits_in_window(self.n) / float(self.n)
        std_u = float(np.std(list(tr.recent_u))) if len(tr.recent_u) > 1 else 0.0
        std_h = float(np.std(list(tr.recent_h))) if len(tr.recent_h) > 1 else 0.0
        stability = 1.0 / (1.0 + std_u + std_h)
        conf = float(np.clip((0.6 * hits_frac + 0.4 * float(ob.confidence))
                             * stability, 0.0, 1.0))
        return dataclasses.replace(ob, confidence=round(conf, 3))

    def _track_info(self, tr: Track) -> dict:
        return {'id': tr.track_id, 'y_m': round(tr.y_m, 2),
                'u_m': round(tr.u_m, 2), 'h_m': round(tr.h_m, 2),
                'vy': round(tr.vy, 3), 'hits': tr.hits,
                'hits_window': tr.hits_in_window(self.n),
                'last_hit_frame': tr.last_hit_frame,
                'confirmed': tr.confirmed}
