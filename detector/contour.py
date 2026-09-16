"""Дальний контур и свободная ширина: дальность за пределы измеренной оси.

Выход из габарита (`core.py`) работает только там, где полилиния оси измерена
(+15 м продолжения): дальше линейное продолжение на кривой уводит габарит, и
появляются ложные срабатывания (замерено: 19 бинов на 6 записях). Но стены и
свод видны до 100..200 м, и это единственный способ получить дальность, близкую
к требованиям ТЗ.

Здесь два независимых признака, оба НЕ требуют точной оси вдали:

`wall_profile` / `ContourTracker` -- положение стен и свода как функция y.
Абсолютный шаблон невозможен: левая стена по записям гуляет 2.17..2.72 м, а
внутри записи платформа/двери/ниши дают 200..650 мм. Поэтому эталон --
скользящая медиана по последним `window` кадрам той же записи. Срабатывание --
отклонение от неё: 0.15 м по |u| стены, 0.30 м по своду.

`free_width` / `WidthTracker` -- свободная ширина вокруг оси по бинам 1 м.
Замерено: 1.27..1.42 м в узком месте. Срабатывание -- сужение >= 0.3 м
относительно собственной истории.

Контрольные величины для проверки: свод стабилен на 4.18..4.43 м, стена в
`doubleT_obstacle` повторяется с точностью 8..20 мм.

Интенсивность как признак НЕ используется: распределения перекрываются
(рельс 1..2, пол 7..9, стена 8..12, свод 6..11).
"""

from __future__ import annotations

from dataclasses import dataclass, field


import numpy as np

from .core import TrackModel

WALL_BIN_M = 2.0             # бин вдоль пути для профиля стен
WALL_BIN_FAR_M = 5.0         # дальше 60 м бин шире: точек в бине мало
WALL_BIN_SPLIT_M = 60.0
WALL_H_BAND = (0.50, 3.00)   # высоты, по которым ищем стену (выше оборудования)
VAULT_H_BAND = (3.00, 6.00)  # высоты свода
VAULT_U_HALF = 1.20          # свод меряем около оси
U_SEARCH_M = 6.0             # докуда ищем стену по u
WALL_TOL_M = 0.15            # порог отклонения стены
VAULT_TOL_M = 0.30           # порог отклонения свода
WINDOW = 15                  # кадров в скользящей медиане
MIN_POINTS = 40              # точек в бине, иначе положение стены не определено
PERSIST_BINS = 2             # отклонение обязано держаться в стольких бинах подряд

WIDTH_BIN_M = 1.0
WIDTH_CELL_M = 0.10
WIDTH_MIN_POINTS = 6
WIDTH_NARROW_M = 0.30        # сужение относительно истории


@dataclass
class WallProfile:
    """Профиль стен и свода по бинам: y, левая/правая стена (u), свод (h)."""

    y: np.ndarray
    u_left: np.ndarray
    u_right: np.ndarray
    vault_h: np.ndarray
    points: np.ndarray

    def to_dict(self) -> dict:
        return {
            'y': [round(float(v), 2) for v in self.y],
            'u_left': [None if not np.isfinite(v) else round(float(v), 3) for v in self.u_left],
            'u_right': [None if not np.isfinite(v) else round(float(v), 3) for v in self.u_right],
            'vault_h': [None if not np.isfinite(v) else round(float(v), 3) for v in self.vault_h],
            'points': [int(v) for v in self.points],
        }


def wall_profile(xyz: np.ndarray, model: TrackModel, bin_m: float = WALL_BIN_M,
                 y_from: float = -200.0, y_to: float = -2.0,
                 min_points: int = MIN_POINTS,
                 bin_far_m: float = WALL_BIN_FAR_M,
                 split_m: float = WALL_BIN_SPLIT_M) -> WallProfile:
    """Положение стен и свода по бинам вдоль пути.

    Стена в бине -- крайние перцентили (2/98) `u` точек в полосе высот
    `WALL_H_BAND`: берём именно крайние, потому что стена -- это граница
    свободного пространства, а не среднее облако. Свод -- перцентиль 90 `h`
    точек выше `VAULT_H_BAND[0]` около оси (`|u| <= VAULT_U_HALF`): выше свода
    точек нет, поэтому верхняя огибающая и есть его поверхность.

    Бин ШИРЕЕТСЯ с дальностью (`bin_m` до `split_m`, дальше `bin_far_m`): на
    150 м в бине 2 м всего единицы точек, и крайние перцентили прыгают на метры
    от кадра к кадру (замерено: десятки «отклонений» на пустом тоннеле).
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    edges = _adaptive_edges(y_from, y_to, bin_m, split_m, bin_far_m)
    n_bins = edges.size - 1
    profile = WallProfile(
        y=0.5 * (edges[:-1] + edges[1:]),
        u_left=np.full(n_bins, np.nan),
        u_right=np.full(n_bins, np.nan),
        vault_h=np.full(n_bins, np.nan),
        points=np.zeros(n_bins, dtype=np.int64),
    )
    if xyz.shape[0] == 0 or n_bins == 0:
        return profile

    u, h = model.relative(xyz)
    y = xyz[:, 1]
    keep = (y >= y_from) & (y < y_to)
    idx = np.clip(np.searchsorted(edges, y, side='right') - 1, 0, n_bins - 1)

    wall = keep & (h >= WALL_H_BAND[0]) & (h <= WALL_H_BAND[1]) & (np.abs(u) <= U_SEARCH_M)
    vault = keep & (h >= VAULT_H_BAND[0]) & (h <= VAULT_H_BAND[1]) & (np.abs(u) <= VAULT_U_HALF)

    for b in range(n_bins):
        m = wall & (idx == b)
        count = int(np.count_nonzero(m))
        profile.points[b] = count
        if count >= min_points:
            ub = u[m]
            profile.u_left[b] = float(np.percentile(ub, 2))
            profile.u_right[b] = float(np.percentile(ub, 98))
        # Свод меряется НЕЗАВИСИМО от стены: у чисто сводового участка точек в
        # полосе стены нет, и общий порог гасил бы свод вместе со стеной.
        mv = vault & (idx == b)
        if int(np.count_nonzero(mv)) >= max(3, min_points // 2):
            profile.vault_h[b] = float(np.percentile(h[mv], 90))
    return profile


def _adaptive_edges(y_from: float, y_to: float, near_m: float, split_m: float,
                    far_m: float) -> np.ndarray:
    """Границы бинов: `near_m` до `split_m`, дальше `far_m`."""
    edges = [y_from]
    y = y_from
    while y < y_to - 1e-9:
        step = near_m if abs(y) <= split_m else far_m
        y = min(y + step, y_to)
        edges.append(y)
    return np.asarray(edges, dtype=np.float64)


def free_width(xyz: np.ndarray, model: TrackModel, bin_m: float = WIDTH_BIN_M,
               cell_m: float = WIDTH_CELL_M, h_band=WALL_H_BAND,
               min_points: int = WIDTH_MIN_POINTS,
               y_from: float = -200.0, y_to: float = -2.0) -> tuple:
    """Свободная ширина вокруг оси по бинам: (y, half_left, half_right).

    Идём от оси наружу по ячейкам `cell_m`, пока ячейка не занята (>= min_points
    точек в полосе высот). Это «сколько места от оси до первой помехи», а не
    расстояние до стены: в узком месте это 1.27..1.42 м.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    n_bins = max(1, int(np.floor((y_to - y_from) / bin_m)))
    edges = y_from + np.arange(n_bins + 1) * bin_m
    y_centers = 0.5 * (edges[:-1] + edges[1:])
    half_left = np.zeros(n_bins)
    half_right = np.zeros(n_bins)
    if xyz.shape[0] == 0:
        return y_centers, half_left, half_right

    u, h = model.relative(xyz)
    y = xyz[:, 1]
    keep = ((y >= y_from) & (y < y_to)
            & (h >= h_band[0]) & (h <= h_band[1]) & (np.abs(u) <= U_SEARCH_M))
    idx = np.clip(np.searchsorted(edges, y, side='right') - 1, 0, n_bins - 1)
    n_cells = int(U_SEARCH_M / cell_m) + 1
    cell = np.clip((u / cell_m).astype(np.int64) + n_cells, 0, 2 * n_cells)

    counts = np.zeros((n_bins, 2 * n_cells + 1), dtype=np.int64)
    np.add.at(counts, (idx[keep], cell[keep]), 1)
    occupied = counts >= max(int(min_points), 1)

    for b in range(n_bins):
        centre = n_cells
        left = 0
        for c in range(centre - 1, -1, -1):
            if occupied[b, c]:
                break
            left += 1
        right = 0
        for c in range(centre + 1, occupied.shape[1]):
            if occupied[b, c]:
                break
            right += 1
        half_left[b] = left * cell_m
        half_right[b] = right * cell_m
    return y_centers, half_left, half_right


def _nanmedian(stack: np.ndarray) -> np.ndarray:
    """nanmedian по столбцам без предупреждений при полностью пустом столбце."""
    import warnings

    stack = np.asarray(stack, dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        out = np.nanmedian(stack, axis=0)
    return np.asarray(out, dtype=np.float64).reshape(-1)


def _runs(mask: np.ndarray, min_len: int) -> list:
    """Непрерывные отрезки True длиной не меньше `min_len`: [(start, stop), ...]."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return []
    padded = np.r_[False, mask, False]
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    runs = [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2])]
    return [r for r in runs if r[1] - r[0] >= min_len]


@dataclass
class ContourTracker:
    """Скользящая медиана профиля и отклонения от неё.

    Абсолютный шаблон стены невозможен (по записям левая стена 2.17..2.72 м),
    поэтому эталон -- собственная история записи. Первые `window` кадров только
    набирают историю: до её заполнения отклонений не выдаём.
    """

    window: int = WINDOW
    wall_tol_m: float = WALL_TOL_M
    vault_tol_m: float = VAULT_TOL_M
    persist_bins: int = PERSIST_BINS
    _left: list = field(default_factory=list)
    _right: list = field(default_factory=list)
    _vault: list = field(default_factory=list)

    def update(self, profile: WallProfile) -> list:
        """Добавить профиль кадра; вернуть список отклонений (пусто -- норма).

        Отклонение засчитывается, только если держится `persist_bins` соседних
        бинов подряд: одиночный бин на 150 м -- это пара точек, а не смещение
        стены.
        """
        self._left.append(profile.u_left)
        self._right.append(profile.u_right)
        self._vault.append(profile.vault_h)
        for seq in (self._left, self._right, self._vault):
            if len(seq) > self.window:
                del seq[0]
        if len(self._left) < self.window:
            return []

        base_left = _nanmedian(np.vstack(self._left))
        base_right = _nanmedian(np.vstack(self._right))
        base_vault = _nanmedian(np.vstack(self._vault))

        events = []
        for label, values, base, tol in (
                ('wall_left', profile.u_left, base_left, self.wall_tol_m),
                ('wall_right', profile.u_right, base_right, self.wall_tol_m),
                ('vault', profile.vault_h, base_vault, self.vault_tol_m)):
            delta = values - base
            bad = np.isfinite(delta) & (np.abs(delta) >= tol)
            for start, stop in _runs(bad, self.persist_bins):
                for i in range(start, stop):
                    events.append({
                        'kind': label,
                        'y_m': round(float(profile.y[i]), 2),
                        'distance_m': round(abs(float(profile.y[i])), 2),
                        'value_m': round(float(values[i]), 3),
                        'baseline_m': round(float(base[i]), 3),
                        'delta_m': round(float(delta[i]), 3),
                        'bins': stop - start,
                    })
        return events


@dataclass
class WidthTracker:
    """Сужение свободной ширины относительно собственной истории."""

    window: int = WINDOW
    narrow_m: float = WIDTH_NARROW_M
    persist_bins: int = PERSIST_BINS
    _half: list = field(default_factory=list)

    def update(self, half_left: np.ndarray, half_right: np.ndarray,
               y: np.ndarray) -> list:
        total = np.asarray(half_left) + np.asarray(half_right)
        self._half.append(total)
        if len(self._half) > self.window:
            del self._half[0]
        if len(self._half) < self.window:
            return []
        base = _nanmedian(np.vstack(self._half))
        delta = total - base
        bad = np.isfinite(delta) & (delta <= -self.narrow_m)
        events = []
        for start, stop in _runs(bad, self.persist_bins):
            for i in range(start, stop):
                events.append({
                    'kind': 'width_narrow',
                    'y_m': round(float(y[i]), 2),
                    'distance_m': round(abs(float(y[i])), 2),
                    'width_m': round(float(total[i]), 3),
                    'baseline_m': round(float(base[i]), 3),
                    'delta_m': round(float(delta[i]), 3),
                    'bins': stop - start,
                })
        return events


def detect_far(xyz: np.ndarray, model: TrackModel,
               contour: Optional[ContourTracker] = None,
               width: Optional[WidthTracker] = None) -> dict:
    """Дальний контур одним вызовом: профиль, свободная ширина и отклонения.

    Возвращает dict с профилем, шириной и событиями. Трекеры передаются снаружи
    (в ROS-узле живут между сообщениями), чтобы эталон копился по кадрам.
    """
    profile = wall_profile(xyz, model)
    y_w, left, right = free_width(xyz, model)
    events = []
    if contour is not None:
        events += contour.update(profile)
    if width is not None:
        events += width.update(left, right, y_w)
    return {
        'wall': profile,
        'width_y': y_w,
        'half_left_m': left,
        'half_right_m': right,
        'events': events,
    }