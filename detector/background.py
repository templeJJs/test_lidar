"""Фоновая модель канала лотка: что здесь ВСЕГДА занято, а что ДОКАЗАННО пусто.

Однокадровая геометрия канала лотка (возвышение над профилем дна) не отличает
лежащее тело от бугра балласта/плиты дальше ~22 м (`core.TROUGH_MAX_DISTANCE_M`
-- замер по негативам: перцентили признаков совпадают). Различает их ВРЕМЯ:
бугор статичен и занят во всех кадрах записи, тело -- новая занятость в ячейке,
которая раньше была пуста. Этот модуль накапливает по кадрам записи две карты в
КАНОНИЧЕСКИХ координатах (y, u) -- те же ячейки, что у профиля дна
(`trough_y_band_m` x `trough_u_cell_m`):

* `bg_occupied` -- ячейка УСТОЙЧИВО занята: занята в доле `occ_frac` кадров,
  где она вообще наблюдалась (строгая min-занятость «во всех кадрах» не
  работает: на 30 м ячейка бугра занята в 5 из 7 кадров -- редкие возвраты
  выпадают; замерено на `squareT_platform_squareT_switch`);
* `bg_free` -- ячейка БЫЛА ДОКАЗАННО СВОБОДНА: в каком-то кадре она пуста, и
  в том же направлении (тот же столбец по u) есть возврат ДАЛЬШЕ неё -- луч
  прошёл сквозь неё и что-то нашёл. Именно «было доказанно свободно», а не
  «обычно пусто»: медианный фон впитал бы стоящего человека за ~1 с, а свобода,
  засвидетельствованная ДО появления объекта, его не съедает (замер:
  `doubleT_obstacle`, фон по кадрам 100-150, человек на кадре 20 -- все его
  ячейки свободны в фоне, 0 занятых).

Кадры со срабатыванием детектора в фон НЕ идут (`update(..., fired=True)` --
пропуск): иначе объект, который детектор уже видит, за секунды запишется в
«устойчиво занято» и запретит сам себя. Объект, которого детектор НЕ видит
(тело дальше 22 м -- его и лечит этот канал), в фон попадает, но за свои 3-8
кадров эпизода долю `occ_frac` не набирает, а свидетельство свободы остаётся от
кадров до эпизода.

Движение сенсора. Ячейки хранятся в канонической системе ОПОРНОГО кадра
(первого); каждый новый кадр переносится в неё: канонизация (`axis_pose`) +
интеграл профиля хода (`motion_between`, тот же перенос, что в
`detector/temporal.py`). Снимок для кадра `t` (`snapshot`) -- те же ячейки,
сдвинутые по y на ход `ref -> t`, с дилатацией `dilate` ячеек на остаток
канонизации (замерено 0.15-0.27 м -- это 1-2 ячейки по u; по y ячейка 3 м
покрывает дробную часть сдвига).

Структура -- как у `ContourTracker`: stateful-объект записи, ядро
(`core.detect`) остаётся stateless и принимает снимок фона опциональным
аргументом `background` (интерфейс -- `BackgroundView`). Без фона поведение
ядра не меняется: дальше `trough_max_distance_m` канал лотка по-прежнему
молчит.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Set, Tuple

import numpy as np

from .core import (AxisPose, DetectorConfig, TrackModel, axis_pose,
                   floor_baseline, trough_channel_mask)

BG_OCC_FRAC = 0.6      # ячейка занята в такой доле кадров, где наблюдалась, --
                       # «устойчиво занята» (бугор: 5/7; тело за эпизод не набирает)
BG_DILATE_CELLS = 1    # дилатация фона на остаток канонизации (0.15-0.27 м)
BG_CELL_MIN_OCC_POINTS = 2   # точек возвышения, с которых ячейка занята в кадре
BG_MAX_DISTANCE_M = 45.0     # докуда вообще разрешён канал лотка с фоном:
                             # дальше и фон, и текущий кадр слишком редки


@dataclass(frozen=True)
class BackgroundView:
    """Снимок фона в канонической системе ОДНОГО кадра -- то, что ест ядро.

    `occupied`/`free` -- множества ячеек (iy, iu) сетки `cell_y_m` x `cell_u_m`
    (iy = floor(-y / cell_y_m), как в `floor_baseline`). Уже сдвинуты на ход
    сенсора между опорным кадром фона и кадром детекции и продилатированы.
    """

    occupied: frozenset
    free: frozenset
    cell_y_m: float
    cell_u_m: float

    def classify(self, y: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """(в_фоне_занято, в_фоне_свободно) -- булевы маски точек (y, u)."""
        y = np.asarray(y, dtype=np.float64)
        u = np.asarray(u, dtype=np.float64)
        iy = np.floor(-y / self.cell_y_m).astype(np.int64)
        iu = np.floor(u / self.cell_u_m).astype(np.int64)
        occ = np.array([(a, b) in self.occupied for a, b
                        in zip(iy.tolist(), iu.tolist())], dtype=bool)
        free = np.array([(a, b) in self.free for a, b
                         in zip(iy.tolist(), iu.tolist())], dtype=bool)
        return occ, free


class TroughBackground:
    """Накопитель фона канала лотка по кадрам записи.

    `motion_between(f_from, f_to) -> метры` -- интеграл профиля хода (как в
    `detector.temporal`); без него запись считается стоящей. Кадры обязаны идти
    подряд; несмежный номер (разрыв записи) -- `reset()` снаружи.
    """

    def __init__(self, model: Optional[TrackModel] = None,
                 cfg: Optional[DetectorConfig] = None,
                 motion_between: Optional[Callable[[int, int], float]] = None,
                 occ_frac: float = BG_OCC_FRAC, dilate: int = BG_DILATE_CELLS):
        self.model = model or TrackModel()
        self.cfg = cfg or DetectorConfig()
        self.motion_between = motion_between or (lambda a, b: 0.0)
        self.occ_frac = float(occ_frac)
        self.dilate = max(0, int(dilate))
        self._ref_frame: Optional[int] = None
        self._ref_pose: Optional[AxisPose] = None
        # (iy, iu) -> [кадров наблюдалась, кадров занята, кадров доказанно свободна]
        self._cells: Dict[Tuple[int, int], list] = {}
        self.frames_used = 0

    def reset(self):
        self._ref_frame = None
        self._ref_pose = None
        self._cells = {}
        self.frames_used = 0

    # ------------------------------------------------------------ накопление

    def update(self, xyz, frame_index: int = 0, fired: bool = False,
               pose: Optional[AxisPose] = None):
        """Добавить кадр в фон. `fired=True` (детектор сработал) -- пропуск.

        `pose` -- уже вычисленная `axis_pose` кадра (детектор её уже считал);
        без неё считается здесь.
        """
        if fired:
            return
        xyz = np.asarray(xyz, dtype=np.float64)
        if xyz.ndim != 2 or xyz.shape[0] == 0:
            return
        if pose is None:
            pose = axis_pose(xyz, self.model)
        canon = xyz
        if pose.applied:
            canon = xyz.copy()
            canon[:, 0] -= pose.line(canon[:, 1])
        if self._ref_frame is None:
            self._ref_frame = int(frame_index)
            self._ref_pose = pose
        u, h = self.model.relative(canon, canonical=False)
        y = canon[:, 1]
        cfg = self.cfg
        chan = trough_channel_mask(u, h, self.model, cfg)
        # Профиль дна -- по ТОМУ ЖЕ набору точек, что в `core._detect_trough`:
        # всё окно канала по высоте (границы z от глобальных УГР, как в tpre)
        # и |u| до `u_out_m`, включая полосы ниток. Базовая линия по одному
        # каналу (без глубоких точек дна и ниток) встаёт НА сам бугор, и фон
        # его не видит (замер: bump на 22 м в doubleT_obstacle давал n_occ=0).
        ax_lo = float(np.min(self.model.axis_x_m)) - cfg.u_out_m \
            if self.model.has_polyline else -cfg.u_out_m
        ax_hi = float(np.max(self.model.axis_x_m)) + cfg.u_out_m \
            if self.model.has_polyline else cfg.u_out_m
        ugr_lo = float(np.min(self.model.ugr_z_m)) if self.model.has_polyline \
            else float(np.min(h)) - 1.0
        ugr_hi = float(np.max(self.model.ugr_z_m)) if self.model.has_polyline \
            else float(np.max(h)) + 1.0
        # Ограничение дальности -- СЕНСОРНОЕ (в системе ЭТОГО кадра), поэтому
        # применяется ДО переноса в опорную систему (после переноса y уже не
        # дальность от лидара, и фильтр выкидывал бы всё, кроме начала записи).
        y_far, y_near = self.model.valid_range(cfg.max_range_m, cfg.min_range_m)
        in_range = (y >= y_far) & (y <= y_near)
        win = ((canon[:, 0] >= ax_lo) & (canon[:, 0] <= ax_hi)
               & (canon[:, 2] >= ugr_lo + cfg.trough_h_low_m)
               & (canon[:, 2] <= ugr_hi + cfg.trough_h_high_m)
               & in_range)
        chan &= in_range
        if not np.count_nonzero(chan):
            return
        cy, cu = cfg.trough_y_band_m, cfg.trough_u_cell_m
        # Перенос в систему опорного кадра: канонический X уже снят позой,
        # остаётся продольный ход (как warp_into_frame, но только по y).
        dy = 0.0
        if frame_index != self._ref_frame:
            dy = float(self.motion_between(int(frame_index), self._ref_frame))
        cell_base = {}
        if np.count_nonzero(win):
            yw, uw, hw = y[win], u[win], h[win]
            bw = floor_baseline(uw, hw, yw, cfg)
            for a, b, bb in zip(np.floor(-yw / cy).astype(np.int64).tolist(),
                                np.floor(uw / cu).astype(np.int64).tolist(),
                                bw.tolist()):
                if bb == bb and (a, b) not in cell_base:  # не NaN
                    cell_base[(a, b)] = bb
        yc, uc, hc = y[chan], u[chan], h[chan]
        # Базовая линия ищется в ячейках СИСТЕМЫ КАДРА (как в ядре), а в фон
        # ячейки пишутся уже сдвинутыми в опорную систему (iy + ход).
        iy_loc = np.floor(-yc / cy).astype(np.int64)
        iu = np.floor(uc / cu).astype(np.int64)
        base = np.array([cell_base.get((a, b), np.nan)
                         for a, b in zip(iy_loc.tolist(), iu.tolist())])
        elev = ~np.isnan(base) & ((hc - base) >= cfg.trough_elev_m)
        iy = np.floor(-(yc + dy) / cy).astype(np.int64)
        # занятые ячейки кадра: >= BG_CELL_MIN_OCC_POINTS возвышенных точек
        key_e = (iy[elev] * 4096 + (iu[elev] + 2048)).astype(np.int64)
        occ_cells = set()
        if key_e.size:
            uniq, counts = np.unique(key_e, return_counts=True)
            occ_cells = {(int(k) // 4096, int(k) % 4096 - 2048)
                         for k, c in zip(uniq.tolist(), counts.tolist())
                         if c >= BG_CELL_MIN_OCC_POINTS}
        # наблюдаемые ячейки и самый дальний возврат каждого столбца
        obs_cells = set(zip(iy.tolist(), iu.tolist()))
        far_by_iu: Dict[int, int] = {}
        for a, b in obs_cells:
            if a > far_by_iu.get(b, -1):
                far_by_iu[b] = a
        for cell in obs_cells:
            rec = self._cells.setdefault(cell, [0, 0, 0])
            rec[0] += 1
        for cell in occ_cells:
            self._cells[cell][1] += 1
        for a, b in obs_cells:
            if (a, b) in occ_cells:
                continue
            if far_by_iu.get(b, -1) > a:
                self._cells[(a, b)][2] += 1
        self.frames_used += 1

    # --------------------------------------------------------------- снимок

    def snapshot(self, frame_index: int = 0) -> BackgroundView:
        """Фон в канонической системе кадра `frame_index` (сдвиг по ходу)."""
        cfg = self.cfg
        cy, cu = cfg.trough_y_band_m, cfg.trough_u_cell_m
        dy = 0.0
        if self._ref_frame is not None:
            dy = self.motion_between(self._ref_frame, int(frame_index))
        occupied: Set[Tuple[int, int]] = set()
        free: Set[Tuple[int, int]] = set()
        for (a, b), (n_obs, n_occ, n_free) in self._cells.items():
            # Перенос ячейки в систему кадра t: точка, стоящая в опорной
            # системе на y_ref, видна из кадра t на y_ref + dy (как
            # warp_into_frame), поэтому центр ячейки сдвигается на +dy.
            a2 = int(np.floor(((a + 0.5) * cy - dy) / cy))
            if n_occ >= self.occ_frac * max(n_obs, 1):
                occupied.add((a2, b))
            if n_free > 0 and n_occ < self.occ_frac * max(n_obs, 1):
                free.add((a2, b))
        r = self.dilate
        if r:
            occupied = {(a + da, b + db) for a, b in occupied
                        for da in range(-r, r + 1) for db in range(-r, r + 1)}
            free = {(a + da, b + db) for a, b in free
                    for da in range(-r, r + 1) for db in range(-r, r + 1)}
        return BackgroundView(occupied=frozenset(occupied), free=frozenset(free),
                              cell_y_m=float(cy), cell_u_m=float(cu))

    def stats(self) -> dict:
        """Диагностика: сколько ячеек наблюдалось/устойчиво занято/свободно."""
        n_occ = sum(1 for _c, (o, cc, _f) in self._cells.items()
                    if cc >= self.occ_frac * max(o, 1))
        n_free = sum(1 for _c, (_o, _cc, f) in self._cells.items() if f > 0)
        return {'frames_used': self.frames_used, 'cells': len(self._cells),
                'occupied': n_occ, 'free': n_free,
                'ref_frame': self._ref_frame}
