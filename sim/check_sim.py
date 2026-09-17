"""Сверка «сим похож на реал»: числа синтетики против чисел живой записи.

Синтетика бесполезна, если не похожа на наши записи, поэтому здесь считаются
метрики ровно те, что названы в критерии 3 спеки
(docs/superpowers/specs/2026-09-15-tunnel-simulator-design.md):

  * число точек на кадр;
  * положение пола (a в z = a + b*y) и уклон b;
  * положение стен -- наш `zones.detect_walls`;
  * крайние углы места (перцентили 0.5/99.5);
  * дальность p95 и статистика интенсивности (медиана, p90, доля > 25);
  * сколько колец в кадре и покрытие азимута (плюс L1 гистограмм колец).

Допуски (`TOLERANCES`) -- из спеки: пол +-0.05 м, стены +-0.10 м, медиана
интенсивности +-20 %, точки на кадр +-25 %. Прочие метрики печатаются рядом и
сверку не роняют: часть из них различается по устройству (см. ниже).

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

  * `azimuth_coverage_deg` -- приборная характеристика, и модель сенсора её не
    воспроизводит. Артефакт хранит только ЧИСЛО колонок, а
    `SensorModel.azimuths_deg` раздаёт их на ПОЛНЫЙ оборот, тогда как у записей
    поле зрения по азимуту ограничено: `doubleT_obstacle` занимает ±124°
    (2709 колонок на 248°, шаг 0.0915°), `roundT_doubleT` -- ровно ±50°
    (1482 колонки на 100°, шаг 0.0675°). Поэтому полное покрытие расходится
    (сцена 360° против 247° у широкой записи и 101° у узкой), а совпадает только
    передняя полусфера: `azimuth_forward_deg` 180/180 у обоих. Это долг Task 1
    (модель сенсора), а не сцены;
  * `floor_rms`: пол сцены глаже реального ложа (0.023 против 0.105 у
    `doubleT_obstacle`) -- шероховатость ложа ещё настраивается в `sim/scene.py`;
  * `range_p95_m`: труба сцены закрыта на `length_m` (120 м), а у записей
    геометрия уходит на 208 м, поэтому p95 расходится на ~18 % (22.9 против
    27.8 м) -- тоже свойство сцены, не прибора.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

import bag_reader
import zones

# Допуски критерия 3 спеки. Ключ -- ИМЯ РАСХОЖДЕНИЯ из отчёта `compare`, поэтому
# метрику, которую посчитать не удалось (NaN), сверка честно считает провалом.
TOLERANCES = {
    'floor_a_delta_m': 0.05,
    'wall_x_delta_m': 0.10,
    'intensity_median_rel': 0.20,
    'point_count_rel': 0.25,
}

FLOOR_Y_MIN = -40.0          # пол сверяем на видимом участке, а не на всех 208 м
FLOOR_Y_MAX = -2.0
RING_COUNT = 128             # колец у Hesai-128: гистограммы колец сравнимы поэлементно
AZ_BIN_DEG = 0.5             # ячейка азимута, °: крупнее шага колонок записи (0.07…0.13)
WALL_REFINE_BAND_M = 0.15    # окно уточнения X стены вокруг бина детектора
WALL_MIN_POINTS = 100        # меньше точек в окне -- уточнять нечем, берём центр бина


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


def _stats(db_path: str, frames: int = 5) -> dict:
    """Все метрики одной записи. Ключ `ring_share` -- только для расхождения."""
    xyz, intensity, ring, n_frames = _read(db_path, frames)
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    floor_a, floor_b, floor_rms = zones.fit_floor(xyz, y_min=FLOOR_Y_MIN, y_max=FLOOR_Y_MAX)
    params = zones.ZoneParams(floor_a=floor_a, floor_b=floor_b)
    walls = _wall_positions(xyz, params, zones.detect_walls(xyz, params))

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
        'azimuth_coverage_deg': float(AZ_BIN_DEG * int(np.count_nonzero(az_occupied))),
        'azimuth_forward_deg': float(AZ_BIN_DEG * int(np.count_nonzero(az_forward))),
        'rings_used': (float(len(np.unique(ring))) if ring is not None else float('nan')),
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