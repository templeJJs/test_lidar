"""Сверка «сим похож на реал»: числа синтетики против чисел живой записи.

Синтетика бесполезна, если не похожа на наши записи, поэтому здесь считаются
метрики ровно те, что названы в критерии 3 спеки
(docs/superpowers/specs/2026-09-15-tunnel-simulator-design.md):

  * число точек на кадр;
  * положение пола (a в z = a + b*y) и уклон b;
  * положение стен -- наш `zones.detect_walls`;
  * крайние углы места (перцентили 0.5/99.5);
  * дальность p95 и статистика интенсивности (медиана, p90, доля > 25);
  * сколько колец в кадре и покрытие азимута (плюс L1 гистограмм колец);
  * уровень свода над осью пути (сцена берёт его из замеров своей записи).

Допуски (`TOLERANCES`) -- из спеки: пол +-0.05 м, стены +-0.10 м, медиана
интенсивности +-20 %, точки на кадр +-25 %, свод +-0.40 м. Прочие метрики
печатаются рядом и сверку не роняют: часть из них различается по устройству
(см. ниже).

Сравнивать надо СОПОСТАВИМЫЕ сцены. `section='wide'` задаёт только семейство:
при `kind='curve'` (умолчание CLI) ширина 3.6…9.2 м, центр туннеля, уклон и
сенсор разыгрываются заново (`scene.SceneParams.resolved`), поэтому сверять
такую сцену с одной записью нечестно -- стены окажутся в других местах не из-за
ошибки симулятора. Эталон для `doubleT_obstacle` -- `kind='straight'` +
`section='wide'`: стены -2.62/+6.62 против измеренных -2.65/+6.55, 9.24 м против
9.20 м. Отсюда же следствие для узкой семьи: константа `SECTIONS['narrow']`
(4.30 м) не равна ни одной записи (3.6…5.0 м), поэтому сверка узкой сцены с
одной записью закономерно расходится по стенам -- эталон надо строить по профилю
самой записи, а не по константе семьи. Замеры сверки: широкий эталон проходит
все допуски, узкая сцена против `roundT_doubleT` даёт |Δx| стен 0.42 м при
допуске 0.10 (5.00 м против 4.30 м).

Что расходится ПО УСТРОЙСТВУ, а не по случайности (это не правится допусками):

  * `floor_rms`: пол сцены глаже реального ложа (0.023 против 0.105 у
    `doubleT_obstacle`) -- шероховатость ложа ещё настраивается в `sim/scene.py`;
  * `range_p95_m`: труба сцены закрыта на `length_m` (200 м), а у записей
    геометрия уходит на 208 м, поэтому p95 расходится -- тоже свойство сцены,
    не прибора.

Покрытие азимута больше НЕ расходится: поле зрения прибора берётся из данных
(`sim.sensor.SensorModel.azimuth_span_deg`). Раньше артефакт хранил только ЧИСЛО
колонок и раздавал их на полный оборот, поэтому сцена давала 360° против 247° у
широкой записи. Теперь колонки раздаются на замеренное поле зрения: у
`doubleT_obstacle` это 2709 колонок на 247° (шаг 0.0912°), у `roundT_doubleT` --
1482 колонки на 101° (шаг 0.0682°), то есть шаг колонок у записей тоже совпал с
замеренным (0.0912° и 0.0675°).
"""
from __future__ import annotations

import argparse
import os

import numpy as np

import bag_reader
import zones
from sim import sensor
from sim.dataset_facts import ledge_of, section_profile

# Допуски критерия 3 спеки. Ключ -- ИМЯ РАСХОЖДЕНИЯ из отчёта `compare`, поэтому
# метрику, которую посчитать не удалось (NaN), сверка честно считает провалом.
TOLERANCES = {
    'floor_a_delta_m': 0.05,
    'wall_x_delta_m': 0.10,
    'intensity_median_rel': 0.20,
    'point_count_rel': 0.25,
    # Свод: сцена строит его на замеренной высоте своей записи, поэтому уровень
    # над осью пути обязан совпасть. Форма (арка вместо плоскости) пока не
    # воспроизводится, отсюда и допуск -- 0.40 м, а не 0.05.
    'section_top_axis_delta_m': 0.40,
    # Свод У СТЕН: арка и наклонная плита. Плоский свод на верной высоте проходит
    # по гребню (Δ 0.02…0.06 м), но у стен врёт на 0.3…0.5 м -- эта метрика и
    # проверяет форму.
    # Форма свода по X: худшее расхождение по гладким ячейкам профиля. Плоский свод
    # на верной высоте проходит по гребню, но у стен врёт: замерено 0.32 м на
    # арочной записи и 0.55 м на наклонной, а сцена с профилем даёт 0.04…0.13 м --
    # допуск 0.20 м и ловит плоский свод, и оставляет запас форме.
    'section_top_delta_max_m': 0.20,
    # Полка у стены. Сцена строила ровную стену от ложа до свода, и в метрах за
    # нитью стены у неё не было ни одной точки (метрика = NaN, а «сравнить нечем»
    # -- это провал допуска). Замерено у записей 1.41…1.58 м.
    'ledge_level_delta_m': 0.35,
}

FLOOR_Y_MIN = -40.0          # пол сверяем на видимом участке, а не на всех 208 м
FLOOR_Y_MAX = -2.0
# Полосу по X задаём ОБЯЗАТЕЛЬНО (докстрока `zones.fit_floor`): по всему кадру
# нижнюю огибающую тянут вниз основания стен, и оценщик перестаёт быть тем же,
# которым снят профиль сцены (`sim.dataset_facts.measure_bag`, те же числа).
FLOOR_X_CENTER = 0.0
FLOOR_X_HALF = 2.0
RING_COUNT = 128             # колец у Hesai-128: гистограммы колец сравнимы поэлементно
AZ_BIN_DEG = 0.5             # ячейка азимута, °: крупнее шага колонок записи (0.07…0.13)
WALL_REFINE_BAND_M = 0.15    # окно уточнения X стены вокруг бина детектора
WALL_MIN_POINTS = 100        # меньше точек в окне -- уточнять нечем, берём центр бина
SECTION_AXIS_HALF_M = 1.0    # полоса «над осью пути» для уровня свода
SECTION_SIDE_M = (1.0, 2.5)  # полоса у стен: там арка ниже гребня


# --------------------------------------------------------------------- чтение


def _db3(path: str) -> str:
    """Путь к `.db3` по каталогу записи или самому файлу.

    Внятная ошибка вместо `sqlite3.OperationalError` из `BagFrames`: по опечатке в
    пути сверка должна сказать «нет записи», а не «unable to open database file».
    """
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        return bag_reader.find_db3(path)
    raise FileNotFoundError(f'нет ни каталога записи, ни .db3: {path}')


def _read(path: str, frames: int):
    """Кадры записи одним облаком: (xyz, intensity|None, ring|None, сколько кадров).

    Кадры берём РАВНОМЕРНО по записи (`linspace`), а не первые подряд: у записи
    разметка вдоль пути разная (у `doubleT_obstacle` видимый участок 27 м), и
    первые кадры её не описывают. При `frames=2` это первый и последний кадр.
    """
    fr = bag_reader.BagFrames(_db3(path), cache_size=2, prefetch_ahead=0)
    try:
        n = len(fr)
        if n == 0:
            raise ValueError(f'пустая запись (нет кадров PointCloud2): {path}')
        k = max(int(frames), 1)
        idxs = np.linspace(0, n - 1, min(k, n)).astype(int)
        xyz, inten, ring = [], [], []
        for i in idxs:
            f = fr[i]
            xyz.append(f.xyz[:, :3].astype(np.float64))
            if f.intensity is not None:
                inten.append(np.asarray(f.intensity, dtype=np.float64))
            if f.ring is not None:
                ring.append(np.asarray(f.ring))
    finally:
        fr.close()
    cloud = np.concatenate(xyz)
    if cloud.shape[0] == 0:
        raise ValueError(f'в выбранных кадрах нет ни одной точки: {path}')
    return (cloud,
            np.concatenate(inten) if inten else None,
            np.concatenate(ring) if ring else None,
            int(idxs.size))


# -------------------------------------------------------------------- метрики


def _wall_positions(xyz: np.ndarray, params: zones.ZoneParams, walls: list,
                    band: float = WALL_REFINE_BAND_M,
                    min_points: int = WALL_MIN_POINTS) -> list:
    """Уточнить X стен, найденных `zones.detect_walls`, до точности лучше бина.

    Бин детектора (0.10 м) равен самому допуску на стены, поэтому центр бина как
    координата стены не годится: у `doubleT_obstacle` правая стена попадала в бин
    с центром 6.55, у сцены -- 6.65, и честные 0.07 м расхождения выглядели как
    0.10 м. Здесь берём медиану X точек ВЫШЕ ПОЛА в окне `band` вокруг бина --
    окно узкое, поэтому дрейф стены вдоль пути (у записей 5.65…6.85) в медиану
    не попадает, а квантование бинов пропадает.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    out = []
    if not walls or xyz.shape[0] == 0:
        return [float(v) for v in walls]
    above = ((xyz[:, 2] - zones.floor_profile(xyz[:, 1], params))
             >= params.floor_band)
    xs = xyz[above, 0]
    for centre in walls:
        near = xs[np.abs(xs - float(centre)) <= float(band)]
        out.append(float(np.median(near)) if near.size >= min_points else float(centre))
    return out


def _ring_histogram(ring: np.ndarray) -> np.ndarray:
    """Нормированная гистограмма колец: доля точек на каждое кольцо."""
    size = max(RING_COUNT, int(ring.max()) + 1) if ring.size else RING_COUNT
    counts = np.bincount(np.asarray(ring, dtype=np.intp), minlength=size).astype(np.float64)
    total = counts.sum()
    return counts / total if total else counts


def _section_top_delta(sim: list, real: list, walls: list = None,
                       tol_x_m: float = 0.13, min_points: int = 30,
                       max_step_m: float = 0.50, wall_margin_m: float = 0.40) -> float:
    """Худшее расхождение свода ПО X между сценой и записью.

    Свод сравнивается по ячейкам профиля, и только на ГЛАДКИХ участках. Ячейки с
    крутым перепадом (больше `max_step_m` на 0.25 м) пропускаются: там замер
    неустойчив сам по себе -- перцентиль 99.5 внутри ячейки на крутой ступени
    зависит от того, где именно легли точки. Проверено на синтетике против её
    СОБСТВЕННОГО меша: на гладких участках расхождение 0.02…0.06 м, на ступенях
    (у станции профиль зубчатый: 5.6 → 8.2 → 7.0) до 0.94 м. Медиана по полосе у
    стен для этого не годится тем более: у `doubleT_platform` она даёт 7.77 против
    6.88 при совпадающих по ячейкам профилях.

    Ячейки сцены и записи совмещаются по X с допуском `tol_x_m`: нити стен у них
    чуть разные, поэтому и сетка ячеек смещена на 0.04 м.
    """
    tops = [b for b in (real or ()) if b.get('top') is not None]
    if walls and len(walls) >= 2:
        # ячейки ЗА нитями стен к своду не относятся: у `roundT_squareT_..._squareT`
        # профиль тянется до x = +2.88 при стене на 2.45, и там меряется уже другая
        # конструкция (у сцены за стеной пусто)
        lo = float(min(walls)) + wall_margin_m
        hi = float(max(walls)) - wall_margin_m
        tops = [b for b in tops if lo <= b['x'] <= hi]
    smooth = []
    for i, b in enumerate(tops):
        nxt = tops[i + 1]['top'] if i + 1 < len(tops) else None
        prv = tops[i - 1]['top'] if i > 0 else None
        step = max(abs(b['top'] - nxt) if nxt is not None else 0.0,
                   abs(b['top'] - prv) if prv is not None else 0.0)
        if step <= max_step_m:
            smooth.append(b)
    deltas = []
    for a in sim or ():
        if a.get('top') is None or int(a.get('top_n') or a.get('n') or 0) < min_points:
            continue
        near = [b for b in smooth if abs(b['x'] - a['x']) <= tol_x_m]
        if not near:
            continue
        best = min(near, key=lambda b: abs(b['x'] - a['x']))
        deltas.append(abs(float(a['top']) - float(best['top'])))
    return float(max(deltas)) if len(deltas) >= 3 else float('nan')


def _stats(db_path: str, frames: int = 5) -> dict:
    """Все метрики одной записи. Ключ `ring_share` -- только для расхождения."""
    xyz, intensity, ring, n_frames = _read(db_path, frames)
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    floor_a, floor_b, floor_rms = zones.fit_floor(xyz, y_min=FLOOR_Y_MIN, y_max=FLOOR_Y_MAX,
                                                  x_center=FLOOR_X_CENTER,
                                                  x_half=FLOOR_X_HALF)
    params = zones.ZoneParams(floor_a=floor_a, floor_b=floor_b)
    walls = _wall_positions(xyz, params, zones.detect_walls(xyz, params))
    # форма сечения -- ТЕМ ЖЕ замером, что и профиль записи в `dataset_facts`:
    # иначе «свод» у сцены и у записи окажутся посчитаны по-разному
    section = section_profile(x, (z - (floor_a + floor_b * y)), y, walls)
    tops = [b['top'] for b in section if b.get('top') is not None]
    axis_tops = [b['top'] for b in section
                 if b.get('top') is not None and abs(b['x']) <= SECTION_AXIS_HALF_M]
    side_tops = [b['top'] for b in section
                 if b.get('top') is not None
                 and SECTION_SIDE_M[0] < abs(b['x']) <= SECTION_SIDE_M[1]]
    # полка за нитью стены -- ТЕМ ЖЕ замером, что и в профиле записи
    ledge = ledge_of(x, (z - (floor_a + floor_b * y)), y, walls)

    rng = np.hypot(x, y)
    elev = np.degrees(np.arctan2(z, rng))
    az = np.degrees(np.arctan2(x, -y))
    az_edges = np.arange(-180.0, 180.0 + AZ_BIN_DEG, AZ_BIN_DEG)
    az_occupied = np.histogram(az, bins=az_edges)[0] > 0
    az_forward = az_occupied[(az_edges[:-1] >= -90.0) & (az_edges[:-1] < 90.0)]

    out = {
        'points_per_frame': float(xyz.shape[0] / n_frames),
        'floor_a': float(floor_a),
        'floor_b': float(floor_b),
        'floor_rms': float(floor_rms),
        'wall_x_left': float(min(walls)) if len(walls) >= 2 else float('nan'),
        'wall_x_right': float(max(walls)) if len(walls) >= 2 else float('nan'),
        'walls_x': [round(float(v), 3) for v in walls],
        'elev_min_deg': float(np.percentile(elev, 0.5)),
        'elev_max_deg': float(np.percentile(elev, 99.5)),
        'range_p95_m': float(np.percentile(rng, 95)),
        'y_min_m': float(y.min()),
        'y_max_m': float(y.max()),
        'azimuth_coverage_deg': sensor.azimuth_span_deg(xyz),
        'azimuth_forward_deg': float(AZ_BIN_DEG * int(np.count_nonzero(az_forward))),
        'rings_used': (float(len(np.unique(ring))) if ring is not None else float('nan')),
        # свод: максимум по ячейкам и уровень над осью пути; расхождения печатаются,
        # но допуска пока нет -- у записей свод наклонный или его вовсе не видно
        # в ближней зоне, а сцена строит плоский потолок
        'section_top_m': float(max(tops)) if tops else float('nan'),
        'section_top_axis_m': (float(np.median(axis_tops)) if axis_tops
                               else float('nan')),
        'section_top_side_m': (float(np.median(side_tops)) if side_tops
                               else float('nan')),
        'section': section,
        'ledge_level_m': float(ledge['level_m']) if ledge else float('nan'),
        'ring_share': _ring_histogram(ring) if ring is not None else np.zeros(0),
    }
    if intensity is not None and intensity.size:
        out['intensity_median'] = float(np.median(intensity))
        out['intensity_p90'] = float(np.percentile(intensity, 90))
        out['intensity_share_gt25'] = float(np.mean(intensity > 25.0))
    else:
        out['intensity_median'] = float('nan')
        out['intensity_p90'] = float('nan')
        out['intensity_share_gt25'] = float('nan')
    return out


# ------------------------------------------------------------------ расхождения


def _delta(a: float, b: float) -> float:
    """|a - b|; NaN, если сравнивать нечем (у любой стороны метрики нет)."""
    if not (np.isfinite(a) and np.isfinite(b)):
        return float('nan')
    return float(abs(float(a) - float(b)))


def _rel(a: float, b: float) -> float:
    """Относительное расхождение к реальности; NaN, если реальность нулевая."""
    if not (np.isfinite(a) and np.isfinite(b)) or abs(float(b)) < 1e-9:
        return float('nan')
    return float(abs(float(a) - float(b)) / abs(float(b)))


def compare(sim_bag: str, real_bag: str, frames: int = 5) -> dict:
    """Отчёт сверки: `sim_*`, `real_*` по каждой метрике + расхождения + допуски.

    `passed` = все допуски `TOLERANCES` выполнены. Метрика, которую посчитать не
    удалось (стен нашлось меньше двух, нет интенсивности), допуск НЕ выполняет:
    «сравнить нечем» -- это не «совпало».
    """
    sim = _stats(sim_bag, frames)
    real = _stats(real_bag, frames)
    sim_rings, real_rings = sim.pop('ring_share'), real.pop('ring_share')
    sim_section, real_section = sim.pop('section'), real.pop('section')

    rep = {f'sim_{k}': v for k, v in sim.items()}
    rep.update({f'real_{k}': v for k, v in real.items()})

    rep['point_count_rel'] = _rel(sim['points_per_frame'], real['points_per_frame'])
    rep['floor_a_delta_m'] = _delta(sim['floor_a'], real['floor_a'])
    rep['floor_b_delta'] = _delta(sim['floor_b'], real['floor_b'])
    rep['floor_rms_delta_m'] = _delta(sim['floor_rms'], real['floor_rms'])
    rep['wall_x_left_delta_m'] = _delta(sim['wall_x_left'], real['wall_x_left'])
    rep['wall_x_right_delta_m'] = _delta(sim['wall_x_right'], real['wall_x_right'])
    if np.isfinite(rep['wall_x_left_delta_m']) and np.isfinite(rep['wall_x_right_delta_m']):
        rep['wall_x_delta_m'] = max(rep['wall_x_left_delta_m'],
                                    rep['wall_x_right_delta_m'])
    else:
        rep['wall_x_delta_m'] = float('nan')
    rep['elevation_min_delta_deg'] = _delta(sim['elev_min_deg'], real['elev_min_deg'])
    rep['elevation_max_delta_deg'] = _delta(sim['elev_max_deg'], real['elev_max_deg'])
    rep['range_p95_delta_m'] = _delta(sim['range_p95_m'], real['range_p95_m'])
    rep['range_p95_rel'] = _rel(sim['range_p95_m'], real['range_p95_m'])
    rep['intensity_median_rel'] = _rel(sim['intensity_median'], real['intensity_median'])
    rep['intensity_p90_rel'] = _rel(sim['intensity_p90'], real['intensity_p90'])
    rep['intensity_share_gt25_delta'] = _delta(sim['intensity_share_gt25'],
                                               real['intensity_share_gt25'])
    rep['rings_used_delta'] = _delta(sim['rings_used'], real['rings_used'])
    rep['ring_hist_l1'] = (float(0.5 * np.abs(sim_rings - real_rings).sum())
                           if sim_rings.size and sim_rings.size == real_rings.size
                           else float('nan'))
    rep['azimuth_coverage_delta_deg'] = _delta(sim['azimuth_coverage_deg'],
                                               real['azimuth_coverage_deg'])
    rep['azimuth_forward_delta_deg'] = _delta(sim['azimuth_forward_deg'],
                                              real['azimuth_forward_deg'])
    rep['section_top_delta_m'] = _delta(sim['section_top_m'], real['section_top_m'])
    rep['section_top_axis_delta_m'] = _delta(sim['section_top_axis_m'],
                                             real['section_top_axis_m'])
    rep['section_top_side_delta_m'] = _delta(sim['section_top_side_m'],
                                             real['section_top_side_m'])
    rep['section_top_delta_max_m'] = _section_top_delta(sim_section, real_section,
                                                        real.get('walls_x'))
    rep['ledge_level_delta_m'] = _delta(sim['ledge_level_m'], real['ledge_level_m'])
    rep['y_min_delta_m'] = _delta(sim['y_min_m'], real['y_min_m'])
    rep['y_max_delta_m'] = _delta(sim['y_max_m'], real['y_max_m'])

    rep['tolerances_ok'] = {k: bool(np.isfinite(rep[k]) and rep[k] <= v)
                            for k, v in TOLERANCES.items()}
    rep['passed'] = all(rep['tolerances_ok'].values())
    return rep


# ------------------------------------------------------------------- печать


_ROWS = (
    dict(title='точек на кадр', sim='points_per_frame', fmt='{:,.0f}',
         delta='point_count_rel', dfmt='{:.1%}', tol='point_count_rel', tol_fmt='≤ {:.0%}'),
    dict(title='пол: a (м)', sim='floor_a', fmt='{:+.4f}',
         delta='floor_a_delta_m', dfmt='{:.4f}', tol='floor_a_delta_m',
         tol_fmt='≤ {:.2f}'),
    dict(title='пол: уклон b', sim='floor_b', fmt='{:+.5f}',
         delta='floor_b_delta', dfmt='{:.5f}'),
    dict(title='пол: rms (м)', sim='floor_rms', fmt='{:.4f}',
         delta='floor_rms_delta_m', dfmt='{:.4f}'),
    dict(title='стены: x (м)', sim='walls_x', fmt='{:}',
         delta='wall_x_delta_m', dfmt='{:.3f}', tol='wall_x_delta_m',
         tol_fmt='≤ {:.2f}'),
    dict(title='стены: левая |Δx| (м)', delta='wall_x_left_delta_m', dfmt='{:.3f}'),
    dict(title='стены: правая |Δx| (м)', delta='wall_x_right_delta_m', dfmt='{:.3f}'),
    dict(title='угол места min 0.5 % (°)', sim='elev_min_deg', fmt='{:+.2f}',
         delta='elevation_min_delta_deg', dfmt='{:.2f}'),
    dict(title='угол места max 99.5 % (°)', sim='elev_max_deg', fmt='{:+.2f}',
         delta='elevation_max_delta_deg', dfmt='{:.2f}'),
    dict(title='дальность p95 (м)', sim='range_p95_m', fmt='{:.2f}',
         delta='range_p95_delta_m', dfmt='{:.2f}'),
    dict(title='дальность p95, отн.', delta='range_p95_rel', dfmt='{:.1%}'),
    dict(title='интенсивность: медиана', sim='intensity_median', fmt='{:.2f}',
         delta='intensity_median_rel', dfmt='{:.1%}', tol='intensity_median_rel',
         tol_fmt='≤ {:.0%}'),
    dict(title='интенсивность: p90', sim='intensity_p90', fmt='{:.2f}',
         delta='intensity_p90_rel', dfmt='{:.1%}'),
    dict(title='интенсивность: доля > 25', sim='intensity_share_gt25', fmt='{:.4f}',
         delta='intensity_share_gt25_delta', dfmt='{:.4f}'),
    dict(title='колец в кадре', sim='rings_used', fmt='{:.0f}',
         delta='rings_used_delta', dfmt='{:.0f}'),
    dict(title='гистограмма колец: L1', delta='ring_hist_l1', dfmt='{:.4f}'),
    dict(title='покрытие азимута, полное (°)', sim='azimuth_coverage_deg', fmt='{:.1f}',
         delta='azimuth_coverage_delta_deg', dfmt='{:.1f}'),
    dict(title='покрытие азимута, вперёд (°)', sim='azimuth_forward_deg', fmt='{:.1f}',
         delta='azimuth_forward_delta_deg', dfmt='{:.1f}'),
    dict(title='свод: max (м)', sim='section_top_m', fmt='{:.2f}',
         delta='section_top_delta_m', dfmt='{:.2f}'),
    dict(title='свод: над осью (м)', sim='section_top_axis_m', fmt='{:.2f}',
         delta='section_top_axis_delta_m', dfmt='{:.2f}',
         tol='section_top_axis_delta_m', tol_fmt='≤ {:.2f}'),
    dict(title='свод: у стен (м)', sim='section_top_side_m', fmt='{:.2f}',
         delta='section_top_side_delta_m', dfmt='{:.2f}'),
    dict(title='свод: макс. по X (м)', delta='section_top_delta_max_m', dfmt='{:.2f}',
         tol='section_top_delta_max_m', tol_fmt='≤ {:.2f}'),
    dict(title='полка у стены (м)', sim='ledge_level_m', fmt='{:.2f}',
         delta='ledge_level_delta_m', dfmt='{:.2f}',
         tol='ledge_level_delta_m', tol_fmt='≤ {:.2f}'),
    dict(title='участок по Y: min (м)', sim='y_min_m', fmt='{:+.1f}',
         delta='y_min_delta_m', dfmt='{:.1f}'),
    dict(title='участок по Y: max (м)', sim='y_max_m', fmt='{:+.1f}',
         delta='y_max_delta_m', dfmt='{:.1f}'),
)


def _fmt(value, spec: str = '{:}') -> str:
    """Значение метрики строкой; `—` для None/нечисла (метрику посчитать нечем)."""
    if value is None:
        return '—'
    if isinstance(value, (list, tuple)):
        return '[' + ', '.join(f'{float(v):+.2f}' for v in value) + ']'
    if isinstance(value, (int, float, np.floating)) and not np.isfinite(value):
        return '—'
    try:
        return spec.format(value)
    except (TypeError, ValueError):
        return str(value)


def format_report(rep: dict) -> str:
    """Отчёт таблицей: сим / реал / расхождение (+ допуск, если он есть)."""
    lines = [f"  {'метрика':32s} {'сим':>16s} {'реал':>16s} {'расхождение':>14s}  допуск"]
    for row in _ROWS:
        sim_key, real_key = row.get('sim'), row.get('real')
        sim = rep.get(f'sim_{sim_key}') if sim_key else None
        real = rep.get(f'real_{real_key or sim_key}') if sim_key else None
        cells = [_fmt(sim, row.get('fmt', '{:}')), _fmt(real, row.get('fmt', '{:}')),
                 _fmt(rep.get(row['delta']), row.get('dfmt', '{:}'))]
        tol = row.get('tol')
        tol_text = row.get('tol_fmt', '{:.2f}').format(TOLERANCES[tol]) if tol else ''
        if tol:
            tol_text += '  OK' if rep['tolerances_ok'][tol] else '  ПРЕВЫШЕН'
        lines.append(f"  {row['title']:32s} {cells[0]:>16s} {cells[1]:>16s} "
                     f"{cells[2]:>14s}  {tol_text}".rstrip())
    verdicts = ', '.join(f'{k} ' + ('OK' if v else 'ПРЕВЫШЕН')
                         for k, v in rep['tolerances_ok'].items())
    lines.append(f'допуски: {verdicts}')
    lines.append('итог: ' + ('ПРОШЛА' if rep['passed'] else 'НЕ ПРОШЛА'))
    return '\n'.join(lines)


def main(argv=None) -> int:
    """CLI: 0 -- все допуски выполнены, 1 -- расхождения вышли за допуски."""
    ap = argparse.ArgumentParser(description='сверка синтетики с реальной записью')
    ap.add_argument('--sim', required=True, help='каталог (или .db3) синтетической записи')
    ap.add_argument('--real', required=True, help='каталог (или .db3) реальной записи')
    ap.add_argument('--frames', type=int, default=5, help='сколько кадров на запись')
    args = ap.parse_args(argv)
    try:
        rep = compare(args.sim, args.real, args.frames)
    except (FileNotFoundError, ValueError) as exc:
        print(f'сверка не удалась: {exc}')
        return 1
    print(f'сим:  {args.sim}')
    print(f'реал: {args.real}')
    print(format_report(rep))
    return 0 if rep['passed'] else 1


if __name__ == '__main__':
    import sys
    sys.exit(main())