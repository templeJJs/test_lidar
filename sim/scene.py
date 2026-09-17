"""Параметрическая сцена туннеля: меши, ось пути, поза сенсора.

Числа взяты из замеров по нашим шести записям (`sim/dataset_facts.json`):
широкий двухпутный туннель 9.2 м (стены −2.65/+6.55), узкие однопутные 3.6…5.0 м,
пол под сенсором 1.36…1.86 м, уклон 0.48…1.94 %, дрейф оси 0.15…0.9 м на видимом
участке (это и есть амплитуда поворотов).

Сечение строится «трубой» по оси туннеля: на каждой станции `y` берётся контур
(пол -- две точки с боковым наклоном, стены, потолок) и соседние контуры сшиваются
в ленту. Благодаря этому туннель умеет поворачивать (`kind='curve'`), а поверхности
остаются точными: при `kind='straight'` и нулевом боковом наклоне пол совпадает с
`floor_z` ровно, стены стоят ровно на замеренных `x`.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np
import open3d as o3d

# Замеры по записям: ширина туннеля, пол под сенсором, уклон %, высота свода
SECTIONS = {
    'wide':   {'width_m': 9.20, 'sensor_z': 1.82, 'grade_pct': 2.34, 'tracks': 2,
               'vault_m': 4.84, 'ledge_m': 1.58, 'ledge_out_m': 0.40},
    'narrow': {'width_m': 4.30, 'sensor_z': 1.32, 'grade_pct': 1.00, 'tracks': 1,
               'vault_m': 4.72, 'ledge_m': 1.50, 'ledge_out_m': 0.50},
    # 'facts' -- профиль туннеля берётся случайно из замеров по всем записям
    # (см. `fact_profiles`), а не из двух усреднённых семей
    'facts':  {'width_m': 5.00, 'sensor_z': 1.50, 'grade_pct': 1.00, 'tracks': 1,
               'vault_m': 4.70, 'ledge_m': 1.50, 'ledge_out_m': 0.50},
}

FACTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset_facts.json')
GAUGE_M = 1.520
TUNNEL_HEIGHT_M = 4.10          # от пола до потолка, если профиль не задал другого
WALL_THICKNESS_M = 0.2          # оставлено для совместимости: стены строятся поверхностью

# Внутренние плоскости стен (x поперёк) для «эталонного» прямого туннеля.
WALL_X_M = {'wide': (-2.62, 6.62), 'narrow': (-2.15, 2.15)}

# Диапазоны для случайного туннеля -- всё из замеров, ничего выдуманного
GRADE_PCT_RANGE = (0.48, 1.94)
SENSOR_Z_RANGE = (1.36, 1.86)
WIDTH_RANGE = (3.60, 9.20)
CENTRE_DRIFT_RANGE = (0.15, 0.90)     # дрейф оси на видимом участке ~30 м
VISIBLE_LENGTH_M = 30.0               # на этой длине замерен дрейф
CANT_MM_RANGE = (0.0, 25.0)           # боковой наклон поверхности (возвышение), мм
FLOOR_ROUGHNESS_M = 0.060            # шероховатость ложа: у реальных записей rms 0.04-0.10
SIM_MEDIAN_AT_SCALE_1 = 9.5          # медиана интенсивности синтетики при масштабе 1 (замерено)
VAULT_AXIS_HALF_M = 1.0              # полоса «над осью пути» для замера свода (как в check_sim)
VAULT_JITTER_M = 0.10                # дрожание свода в случайной сцене
VAULT_MAX_SAMPLES = 41               # больше точек свода, чем ячеек профиля, не нужно
LEDGE_JITTER_M = 0.05                # дрожание полки в случайной сцене
# Высота нишы: у записей в полосе 1.1…1.9 м лежит тонкая поверхность, и
LEDGE_DUCT_H_M = 0.12                # широкая ниша давала бы медиану выше замеренной
TRACK_AXIS_JITTER_M = 0.05            # дрожание оси пути в случайной сцене
INTENSITY_THRESHOLD = 25.0           # порог «ярких» точек: наша метрика доли (check_sim)
INTENSITY_SIGMA_K = 1.0              # коэффициент модели хвоста (замерено 0.90…1.05)
INTENSITY_SIGMA_RANGE = (0.20, 1.60)  # зажим: вне него хвост перестаёт быть похожим
SHARE_RANGE = (1e-4, 0.40)           # доля >25 у записей 0.8…10.8 %, зажим для обратной функции
STATION_STEP_M = 2.0                  # шаг станций трубы: поверхности остаются точными


def _norm_quantile(p: float) -> float:
    """Обратная функция стандартного нормального распределения (Acklam).

    Нужна, чтобы из замеренной доли «ярких» точек получить квантиль: хвост
    логнормального распределения начинается на `ln(порог/медиана) / σ` сигм.
    Точность приближения 1e-9 -- для подбора σ этого с избытком.
    """
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    p = min(max(float(p), 1e-12), 1.0 - 1e-12)
    low, high = 0.02425, 1.0 - 0.02425
    if p < low:
        q = np.sqrt(-2.0 * np.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) \
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p > high:
        q = np.sqrt(-2.0 * np.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) \
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q \
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)


def intensity_sigma_for(median: float, share_gt25: float) -> float:
    """σ разброса яркости под замеренную долю точек выше порога.

    Яркость в сканере -- медиана × логнормальный множитель, поэтому «доля выше
    25» задаётся одной величиной: `ln(25 / медиана) / σ` сигм хвоста. Отсюда
    σ = ln(порог/медиана) / z(1 − доля), где `z` -- обратная функция нормального
    распределения. Коэффициент проверен прогоном по всем шести записям (0.90…1.05),
    и формула воспроизводит долю > 25 у каждой, тогда как прежняя линейная
    подгонка от доли не сходилась: при одинаковой доле у записей с медианой 7 и 11
    нужны разные σ (0.77 против 0.44).
    """
    share = min(max(float(share_gt25), SHARE_RANGE[0]), SHARE_RANGE[1])
    if not np.isfinite(median) or median <= 0.0:
        return float(INTENSITY_SIGMA_RANGE[0])
    z = _norm_quantile(1.0 - share)
    if z <= 0.0:
        return float(INTENSITY_SIGMA_RANGE[1])
    sigma = INTENSITY_SIGMA_K * float(np.log(INTENSITY_THRESHOLD / float(median))) / z
    return float(min(max(sigma, INTENSITY_SIGMA_RANGE[0]), INTENSITY_SIGMA_RANGE[1]))


def vault_from_section(section: list) -> float:
    """Высота свода над ложем по замеренному сечению.

    Берём медиану сводов НАД ОСЬЮ пути (|x| ≤ 1 м) -- это гребень туннеля, и он
    устойчив к одиночным выбросам. Если над осью свода не видно, берём медиану по
    всем ячейкам: у записей со станцией свод поднимается на 8+ м, и «max по
    ячейкам» на такой записи поймал бы случайный выступ.
    """
    tops = [float(b['top']) for b in section or () if b.get('top') is not None]
    if not tops:
        return float(TUNNEL_HEIGHT_M)
    axis = [float(b['top']) for b in section or ()
            if b.get('top') is not None and abs(b['x']) <= VAULT_AXIS_HALF_M]
    return float(np.median(axis)) if axis else float(np.median(tops))


def fact_profiles(path: str = FACTS_PATH) -> list:
    """Профили туннелей, замеренные по ВСЕМ записям (sim/dataset_facts.json).

    Нужны, чтобы случайная сцена опиралась на все датасеты, а не на две
    усреднённые семьи: у нас шесть разных туннелей -- от двухпутного 9.2 м до
    однопутного 3.6 м, с разными уклонами и высотой подвеса сенсора.
    """
    try:
        with open(path, encoding='utf-8') as fh:
            data = json.load(fh)
    except OSError:
        return []
    out = []
    for name, f in (data.get('bags') or {}).items():
        walls = f.get('walls_x') or []
        if len(walls) < 2 or 'floor' not in f:
            continue
        out.append({'bag': name,
                    'walls_x': (float(walls[0]), float(walls[-1])),
                    'grade_pct': float(f['floor']['grade_pct']),
                    'sensor_z': round(-float(f['floor']['a']), 3),
                    'columns': f.get('azimuth_columns'),
                    # поле зрения прибора: у широкой записи 247°, у узкой 101°;
                    # без него сцена светит в хвостовой сектор, которого в записи нет
                    'azimuth_span_deg': f.get('azimuth_span_deg'),
                    # форма сечения (свод и ложе) в ближней зоне: у записей свод
                    # не всегда плоский, и это следующая задача генератора
                    'section': f.get('section') or [],
                    # высота свода: мерится по сечению, а не выдумывается
                    'vault_m': round(vault_from_section(f.get('section')), 3),
                    # полка у стены: сцена без неё не даёт точек за нитью стены
                    'ledge_m': (f.get('ledge') or {}).get('level_m'),
                    'ledge_out_m': (f.get('ledge') or {}).get('out_m'),
                    # ось пути по замеру рельсовых нитей (сцена ставила её в 0)
                    'track_axis_x': (f.get('track') or {}).get('axis_x'),
                    # пропуски возврата: сенсор записи возвращал точку не с каждого
                    # выстрела (замерено 1.0…6.5 %)
                    'dropout': f.get('dropout'),
                    # интенсивность у записей разная (медиана 6…12, доля >25 0.7…10.8 %):
                    # по ней сцена подбирает свой масштаб отражения
                    'intensity_median': (f.get('intensity') or {}).get('median'),
                    'share_gt25': (f.get('intensity') or {}).get('share_gt25')})
    return out


@dataclass
class SceneParams:
    seed: int
    kind: str = 'straight'          # straight | curve (случайный поворот)
    section: str = 'wide'           # wide | narrow | facts
    length_m: float = 200.0
    grade_pct: float = None         # None -> из SECTIONS (straight) или случайно (curve)
    tracks: int = None
    track_axis_x: float = None
    platform: bool = False
    gate: bool = False
    # поля случайного туннеля; None -> подставляются в resolved()
    sensor_z: float = None
    walls_x: tuple = None
    # высота свода над ложем: 4.1 у прямоугольного макета, у записей 4.6…5.1 м,
    # а у записи со станцией 8+ м -- берётся из замеренного сечения
    vault_m: float = None
    # профиль свода по X: список (x, высота над ложем) из замеренного сечения.
    # Пустой список -- свод плоский на `vault_m`
    vault_profile: tuple = None
    # полка у стены: высота над ложем и выступ наружу от нити стены (замерено у
    # всех шести записей: 1.41…1.58 м и 0.40…0.76 м)
    ledge_m: float = None
    ledge_out_m: float = None
    curve_radius_m: float = None
    curve_sign: int = None
    cant_mm: float = None
    # масштаб отражения: подбирается под профиль записи (медиана интенсивности
    # у записей 6…12, а синтетика при масштабе 1 даёт ~9.5)
    intensity_scale: float = None
    # и разброс: у записей хвост ярких точек разной толщины (доля >25 от 0.7 до
    # 10.8 %) -- медиана и хвост у них даже антикоррелированы
    intensity_sigma: float = None
    fact_bag: str = None            # запись, профиль которой повторяет сцена

    def resolved(self) -> 'SceneParams':
        s = SECTIONS[self.section]
        p = SceneParams(**{**self.__dict__})
        rng = np.random.default_rng(int(self.seed))
        randomize = self.kind != 'straight'
        # профиль можно задать явно (`fact_bag`): тогда сцена воспроизводит ИМЕННО
        # эту запись, без дрожания стен и уклона -- иначе сверять сцену с записью
        # нечем (стены уходят на 0.28 м при допуске 0.10)
        pinned = bool(self.fact_bag) and self.section == 'facts'

        # профиль из замеров по всем записям: в случайном режиме берём любую запись
        # и слегка сдвигаем её параметры, чтобы сцены были разными, но похожими
        prof = None
        if self.section == 'facts' and (pinned or randomize):
            profiles = fact_profiles()
            if profiles:
                if pinned:
                    prof = next((q for q in profiles if q['bag'] == self.fact_bag), None)
                    if prof is None:
                        raise ValueError(f'в замерах нет записи {self.fact_bag!r}; есть: '
                                         + ', '.join(q['bag'] for q in profiles))
                else:
                    prof = profiles[int(rng.integers(0, len(profiles)))]
                p.fact_bag = prof['bag']
                median = prof.get('intensity_median')
                if median:
                    # SIM_MEDIAN при масштабе 1 замерена сканером на широкой сцене
                    p.intensity_scale = min(max(float(median) / SIM_MEDIAN_AT_SCALE_1,
                                                0.35), 2.5)
                share = prof.get('share_gt25')
                if median and share is not None:
                    p.intensity_sigma = intensity_sigma_for(float(median), float(share))

        if p.grade_pct is None:
            if pinned:
                p.grade_pct = prof['grade_pct']
            elif prof is not None:
                p.grade_pct = prof['grade_pct'] + float(rng.normal(0.0, 0.15))
                if rng.random() < 0.5:
                    p.grade_pct = -p.grade_pct
            elif randomize:
                p.grade_pct = (float(rng.uniform(*GRADE_PCT_RANGE))
                               * (1 if rng.random() < 0.5 else -1))
            else:
                p.grade_pct = s['grade_pct']
        if p.tracks is None:
            p.tracks = s['tracks']
        if p.track_axis_x is None:
            if prof is not None and prof.get('track_axis_x') is not None:
                p.track_axis_x = float(prof['track_axis_x'])
                if not pinned:
                    p.track_axis_x += float(rng.normal(0.0, TRACK_AXIS_JITTER_M))
            else:
                p.track_axis_x = 0.0
        if p.sensor_z is None:
            if pinned:
                p.sensor_z = prof['sensor_z']
            elif prof is not None:
                p.sensor_z = prof['sensor_z'] + float(rng.normal(0.0, 0.05))
            elif randomize:
                p.sensor_z = float(rng.uniform(*SENSOR_Z_RANGE))
            else:
                p.sensor_z = s['sensor_z']
        if p.walls_x is None:
            if pinned:
                p.walls_x = tuple(prof['walls_x'])
            elif prof is not None:
                wl, wr = prof['walls_x']
                # дрожание небольшое: сцена должна оставаться ТЕМ ЖЕ туннелем
                # (иначе стены расходятся с записью сильнее допуска 0.1 м),
                # разнообразие дают уклон, радиус поворота, наклон и яркость
                width = (wr - wl) * float(rng.uniform(0.97, 1.03))
                centre = 0.5 * (wl + wr) + float(rng.normal(0.0, 0.04))
                p.walls_x = (centre - width / 2.0, centre + width / 2.0)
            elif randomize:
                width = float(rng.uniform(*WIDTH_RANGE))
                centre = float(rng.uniform(-0.2, 0.2))
                p.walls_x = (centre - width / 2.0, centre + width / 2.0)
            else:
                p.walls_x = WALL_X_M[self.section]
        if p.vault_m is None:
            shift = 0.0
            if prof is not None:
                shift = 0.0 if pinned else float(rng.normal(0.0, VAULT_JITTER_M))
                p.vault_m = prof['vault_m'] + shift
                p.vault_profile = tuple(vault_curve_from_section(prof.get('section') or [],
                                                                 p.vault_m, shift))
            else:
                p.vault_m = float(s['vault_m'])
                p.vault_profile = ()
        if p.ledge_m is None:
            if prof is not None and prof.get('ledge_m'):
                p.ledge_m = float(prof['ledge_m'])
                p.ledge_out_m = float(prof.get('ledge_out_m') or s['ledge_out_m'])
                if not pinned:
                    p.ledge_m += float(rng.normal(0.0, LEDGE_JITTER_M))
            else:
                p.ledge_m = float(s['ledge_m'])
                p.ledge_out_m = float(s['ledge_out_m'])
        if p.curve_radius_m is None:
            if randomize:
                drift = float(rng.uniform(*CENTRE_DRIFT_RANGE))
                p.curve_radius_m = (VISIBLE_LENGTH_M ** 2) / (2.0 * drift)
                p.curve_sign = 1 if rng.random() < 0.5 else -1
            else:
                p.curve_radius_m = float('inf')
                p.curve_sign = 1
        if p.curve_sign is None:
            p.curve_sign = 1
        if p.cant_mm is None:
            if randomize:
                p.cant_mm = float(rng.uniform(*CANT_MM_RANGE)) * (1 if rng.random() < 0.5 else -1)
            else:
                p.cant_mm = 0.0
        return p


def _resolved(p: SceneParams) -> SceneParams:
    """Параметры с подставленными случайными полями.

    Нужно потому, что сцену принято описывать коротко (`SceneParams(seed=…)`), а
    `floor_z`/`centre_x` вызывают и те, кто не звал `resolved()`: у неразрешённых
    параметров случайные поля равны None, и без подстановки они падают. Вызов
    детерминирован: генератор создаётся заново от того же зерна.
    """
    if (p.grade_pct is None or p.sensor_z is None or p.walls_x is None
            or p.curve_radius_m is None or p.cant_mm is None or p.vault_m is None
            or p.vault_profile is None or p.ledge_m is None or p.ledge_out_m is None):
        return p.resolved()
    return p


def floor_z(p: SceneParams, y):
    """Уровень пола ПО ОСИ пути в системе сенсора: сенсор на `sensor_z` над полом."""
    p = _resolved(p)
    return -float(p.sensor_z) + p.grade_pct / 100.0 * np.asarray(y, dtype=np.float64)


def centre_x(p: SceneParams, y):
    """Поперечное положение оси туннеля: прямая или дуга окружности.

    Дрейф подобран так, чтобы на `VISIBLE_LENGTH_M` он попадал в замеренный
    диапазон 0.1…0.9 м -- то есть радиусы получаются 500…4500 м, как в метро.
    """
    p = _resolved(p)
    y_arr = np.asarray(y, dtype=np.float64)
    if p.kind == 'straight' or not np.isfinite(p.curve_radius_m):
        return np.full_like(y_arr, float(p.track_axis_x))
    return float(p.track_axis_x) + p.curve_sign * (y_arr ** 2) / (2.0 * p.curve_radius_m)


def cant_slope(p: SceneParams) -> float:
    """Боковой наклон поверхности (возвышение, мм на половину колеи)."""
    p = _resolved(p)
    return float(p.cant_mm) / 1000.0 / max(GAUGE_M / 2.0, 1e-6)


def floor_z_xy(p: SceneParams, x, y):
    """Уровень пола в точке (x, y): плоскость по оси плюс боковой наклон."""
    p = _resolved(p)
    y_arr = np.asarray(y, dtype=np.float64)
    x_arr = np.asarray(x, dtype=np.float64)
    return floor_z(p, y_arr) + cant_slope(p) * (x_arr - centre_x(p, y_arr))


def vault_curve_from_section(section: list, crown: float, shift: float = 0.0) -> list:
    """Профиль свода по X из замеренного сечения: список (x, высота над ложем).

    Ячейки без свода (`top = None`) интерполируются между соседними замеренными,
    а не считаются нулём: у `doubleT_obstacle` такая одна (x ≈ +6.7, свод там
    перекрыт стеной), и ноль нарисовал бы дыру в потолке. За пределами замеренного
    диапазона значения держатся на краю -- свод продолжается вдоль пути.

    `shift` сдвигает весь профиль по высоте (дрожание случайной сцены).
    """
    observed = [(float(b['x']), float(b['top'])) for b in section or ()
                if b.get('top') is not None]
    if len(observed) < 2:
        return []
    ox = np.array([q[0] for q in observed], dtype=np.float64)
    oh = np.array([q[1] for q in observed], dtype=np.float64) + float(shift)
    curve = []
    for b in section:
        x = float(b['x'])
        h = float(b['top']) + float(shift) if b.get('top') is not None else np.nan
        if not np.isfinite(h):
            h = float(np.interp(x, ox, oh))
        curve.append((round(x, 3), round(float(h), 3)))
    if not any(round(h, 2) != round(crown, 2) for _x, h in curve):
        return []               # плоский свод -- профиль не нужен
    return curve


def vault_samples(p: SceneParams) -> np.ndarray:
    """Доли поперёк туннеля для точек свода: 0 -- правая стена, 1 -- левая.

    Точек столько же, сколько ячеек в замеренном профиле (свод у записей бывает
    аркой и наклонной плитой, четырьмя точками его не описать). Без профиля --
    две точки, то есть плоский свод.
    """
    n = len(p.vault_profile or ())
    return np.linspace(0.0, 1.0, min(max(n, 2), VAULT_MAX_SAMPLES))


def vault_height(p: SceneParams, x, y) -> float:
    """Высота свода НАД ПОЛОМ в точке (x, y): по профилю своей записи.

    Профиль замерен в абсолютных X записи, а туннель сцены может уходить в
    поворот, поэтому X приводится к локальной системе оси (`centre_x`): свод
    едет вместе с туннелем, как и стены.
    """
    p = _resolved(p)
    prof = p.vault_profile or ()
    if not prof:
        return float(p.vault_m)
    xs = np.array([q[0] for q in prof], dtype=np.float64)
    hs = np.array([q[1] for q in prof], dtype=np.float64)
    shift = np.asarray(centre_x(p, y), dtype=np.float64) - float(centre_x(p, 0.0))
    return np.interp(np.asarray(x, dtype=np.float64) - shift, xs, hs)


@dataclass
class Scene:
    params: SceneParams
    mesh: o3d.t.geometry.TriangleMesh
    track: dict                       # {'axis_x', 'gauge', 'rails_x'}
    sensor_pose: dict                 # {'x','y','z','pitch_deg'}
    objects: list = field(default_factory=list)

    def raycasting(self) -> o3d.t.geometry.RaycastingScene:
        rc = o3d.t.geometry.RaycastingScene()
        rc.add_triangles(self.mesh)
        return rc


def to_tensor_mesh(part: o3d.geometry.TriangleMesh) -> o3d.t.geometry.TriangleMesh:
    """Меш в тензорный вид: только такие принимает `RaycastingScene`.

    У legacy-меша в установленном open3d 0.19 нет `.to_tensor()`.
    """
    return o3d.t.geometry.TriangleMesh.from_legacy(part)


def stations(p: SceneParams) -> np.ndarray:
    """Станции трубы вдоль оси: шаг мелкий, чтобы поверхности были точными."""
    n = max(3, int(round(2.0 * p.length_m / STATION_STEP_M)) + 1)
    return np.linspace(-p.length_m, p.length_m, n)


def tube_mesh(p: SceneParams) -> o3d.geometry.TriangleMesh:
    """Труба туннеля: контуры на станциях, сшитые в ленту.

    Контур станции: пол слева, пол справа, полка у правой стены, свод справа
    налево (по замеренному профилю, `vault_height`), полка у левой стены. Пол
    берётся из `floor_z_xy` (то есть с боковым наклоном), поэтому при нулевом
    наклоне он ровно совпадает с `floor_z`.

    Полка (`ledge_m`, `ledge_out_m`) -- замеренная особенность ВСЕХ шести записей:
    за нитью стены, которую находит детектор, на высоте 1.41…1.58 м лежит
    поверхность, уходящая наружу на 0.40…0.76 м, а выше стена стоит там же
    (замерено: +0.03…0.28 м от нити). То есть это ниша в стене высотой
    `LEDGE_DUCT_H_M`, а не уступ, расширяющий туннель. Без неё сцена не давала за
    нитью стены ни одной точки, а записи дают там тысячи.

    Свод берётся из `section` своей записи: у `roundT_doubleT` это арка с гребнем
    5.03 м и плечами 4.65 м, у `doubleT_obstacle` -- наклонная плита 4.90 → 4.32 м.
    Без профиля (семьи wide/narrow/facts) свод плоский на замеренной высоте
    `vault_m`.
    """
    p = _resolved(p)
    xl0, xr0 = p.walls_x
    dx0 = float(centre_x(p, 0.0))
    verts, tris = [], []
    ys = stations(p)
    # контур станции: пол слева, пол справа, и дальше свод справа налево -- столько
    # точек, сколько ячеек у замеренного профиля (свод у записей арочный и
    # наклонный, четырьмя точками его не описать)
    ceil_x = vault_samples(p)
    # шероховатость ложа: балласт лежит НА плоскости пола, а не колеблется вокруг
    # неё. Односторонним разбросом воспроизводится нижняя огибающая: наш фит пола
    # (`zones.fit_floor` -- 5-й перцентиль по бинам) у записей попадает ровно на
    # уровень пола, а симметричный шум тянул его на 0.05 м вниз (замерено:
    # floor_a −1.874 против −1.821 у doubleT_obstacle при допуске 0.05).
    rng = np.random.default_rng(int(p.seed) + 7919)
    for y in ys:
        dx = float(centre_x(p, y)) - dx0
        xl, xr = xl0 + dx, xr0 + dx
        zl = float(floor_z_xy(p, xl, y)) + float(rng.uniform(0.0, FLOOR_ROUGHNESS_M))
        zr = float(floor_z_xy(p, xr, y)) + float(rng.uniform(0.0, FLOOR_ROUGHNESS_M))
        # ниша в стене: горизонтальная поверхность за нитью стены на высоте полки.
        # Замерено, что ВЕРХНЯЯ стена остаётся на месте (+0.03…0.28 м от нити), а
        # поверхность уходит наружу на 0.4…0.7 м: это ниша (кабель-канал), а не
        # уступ, расширяющий туннель. Поэтому контур делает выемку и возвращается
        # на ту же нить -- иначе детектор стен у сцены уезжает наружу на 0.4 м и
        # сверка по стенам падает (проверено).
        xl_out, xr_out = xl - p.ledge_out_m, xr + p.ledge_out_m
        hl = float(floor_z_xy(p, xl, y)) + p.ledge_m
        hr = float(floor_z_xy(p, xr, y)) + p.ledge_m
        ring = [(xl, y, zl), (xr, y, zr),
                (xr, y, hr), (xr_out, y, hr),                      # низ нишы
                (xr_out, y, hr + LEDGE_DUCT_H_M), (xr, y, hr + LEDGE_DUCT_H_M)]
        for frac in ceil_x:                                        # свод справа налево
            x = xr + (xl - xr) * frac
            ring.append((x, y, float(floor_z_xy(p, x, y)) + vault_height(p, x, y)))
        ring += [(xl, y, hl + LEDGE_DUCT_H_M), (xl_out, y, hl + LEDGE_DUCT_H_M),
                 (xl_out, y, hl), (xl, y, hl)]                     # ниша слева
        verts.extend(ring)
    m = len(ring)
    for i in range(len(ys) - 1):
        for k in range(m):
            a = i * m + k
            b = i * m + (k + 1) % m
            c = (i + 1) * m + k
            d = (i + 1) * m + (k + 1) % m
            tris.append((a, c, b))
            tris.append((b, c, d))
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(verts, dtype=np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(tris, dtype=np.int32))
    return mesh


def build_scene(params: SceneParams) -> Scene:
    """Собрать сцену: труба туннеля, рельсы и шпалы нашего пути."""
    p = params.resolved()
    parts = [tube_mesh(p)]

    from sim.track_geometry_adapter import add_track
    track = add_track(parts, p)

    mesh = o3d.geometry.TriangleMesh()
    for part in parts:
        mesh += part
    mesh.compute_vertex_normals()
    return Scene(params=p, mesh=to_tensor_mesh(mesh), track=track,
                 sensor_pose={'x': 0.0, 'y': 0.0, 'z': float(p.sensor_z),
                              'pitch_deg': 0.0})