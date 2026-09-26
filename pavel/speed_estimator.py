"""
Оценка скорости поезда по отражателям в облаках точек LiDAR.

Находит яркие объекты (intensity > порога), кластеризует, сопоставляет
между кадрами по XZ-координатам (X и Z не меняются при движении по Y),
вычисляет скорость как медиану сдвигов по Y.

Работает автономно — нужны только два кадра (xyz, intensity).
"""

import numpy as np


def find_reflectors(points, intensity, min_intensity=150, eps=0.5, min_samples=3):
    """Находит отражатели как кластеры ярких точек.

    Параметры:
        points: (N, 3) облако точек
        intensity: (N,) интенсивность
        min_intensity: порог яркости (по умолчанию 150)
        eps: DBSCAN — макс. расстояние между точками в кластере, м
        min_samples: DBSCAN — мин. точек в кластере

    Возвращает:
        list of dict: [{center: (x,y,z), n_points: int, intensity_mean: float}, ...]
    """
    from sklearn.cluster import DBSCAN

    if intensity is None or len(points) == 0:
        return []

    hot = intensity > min_intensity
    if hot.sum() < min_samples:
        return []

    hp = points[hot]
    hi = intensity[hot]

    db = DBSCAN(eps=eps, min_samples=min_samples).fit(hp)
    labels = db.labels_
    clusters = sorted(set(labels) - {-1})

    reflectors = []
    for lbl in clusters:
        mask = labels == lbl
        c = hp[mask]
        reflectors.append({
            'center': c.mean(axis=0),
            'n_points': int(mask.sum()),
            'intensity_mean': float(hi[mask].mean()),
        })

    return reflectors


def match_reflectors(refs_a, refs_b, max_dx=0.3, max_dz=0.3):
    """Сопоставляет отражатели между двумя кадрами по XZ-координатам.

    При движении по Y координаты X и Z статичных объектов не меняются.
    Возвращает список (i, j, dy) — индексы в refs_a/refs_b и сдвиг по Y.

    Параметры:
        max_dx: макс. отклонение по X, м
        max_dz: макс. отклонение по Z, м
    """
    matches = []
    for i, ra in enumerate(refs_a):
        for j, rb in enumerate(refs_b):
            dx = abs(ra['center'][0] - rb['center'][0])
            dz = abs(ra['center'][2] - rb['center'][2])
            if dx < max_dx and dz < max_dz:
                dy = rb['center'][1] - ra['center'][1]
                matches.append((i, j, dy))
    return matches


def estimate_speed(matches, dt, min_speed_kmh=5.0, max_speed_kmh=200.0,
                   consensus_tol_kmh=15.0, min_consensus=2,
                   min_consensus_ratio=0.3):
    """Вычисляет скорость из набора матчей с фильтрацией выбросов.

    Алгоритм:
    1. Переводит сдвиги dY в скорость (км/ч)
    2. Отбрасывает физически невозможные (< min_speed или > max_speed)
    3. Ищет консенсус: группа скоростей, согласованных в ±consensus_tol
    4. Берёт медиану группы с наибольшим числом элементов

    Параметры:
        matches: список (i, j, dy) из match_reflectors
        dt: время между кадрами, с
        min_speed_kmh: мин. реалистичная скорость, км/ч (отсекает шум)
        max_speed_kmh: макс. реалистичная скорость, км/ч
        consensus_tol_kmh: допуск для консенсуса, км/ч
        min_consensus: мин. матчей в консенсусе для доверия результату

    Возвращает:
        dict: {speed_ms: float, speed_kmh: float, direction: int (+1/-1),
               n_matches: int, n_consensus: int, confidence: str} или None
    """
    if not matches or dt <= 0:
        return None

    speeds = []
    for i, j, dy in matches:
        speed_ms = dy / dt
        speed_kmh = abs(speed_ms) * 3.6
        if min_speed_kmh <= speed_kmh <= max_speed_kmh:
            speeds.append((speed_ms, speed_kmh, i, j))

    if len(speeds) < min_consensus:
        return None

    # Консенсус: для каждой скорости считаем сколько других в ±tol
    speed_vals = np.array([s[1] for s in speeds])
    best_group = []
    for k, sv in enumerate(speed_vals):
        group = [m for m, other in enumerate(speed_vals)
                 if abs(other - sv) < consensus_tol_kmh]
        if len(group) > len(best_group):
            best_group = group

    if len(best_group) < min_consensus:
        return None

    # Если consensus слишком мал относительно всех матчей — ненадёжно
    if len(speeds) > 0 and len(best_group) / len(speeds) < min_consensus_ratio:
        return None

    consensus_speeds_ms = np.array([speeds[k][0] for k in best_group])
    median_ms = float(np.median(consensus_speeds_ms))
    median_kmh = abs(median_ms) * 3.6
    std_kmh = float(np.std(np.abs(consensus_speeds_ms) * 3.6))

    if std_kmh < 3.0 and len(best_group) >= 5:
        confidence = 'high'
    elif std_kmh < 8.0 and len(best_group) >= 3:
        confidence = 'medium'
    else:
        confidence = 'low'

    return {
        'speed_ms': abs(median_ms),
        'speed_kmh': median_kmh,
        'dy_per_dt': median_ms,  # со знаком — для аккумуляции
        'direction': 1 if median_ms > 0 else -1,
        'n_matches': len(speeds),
        'n_consensus': len(best_group),
        'std_kmh': std_kmh,
        'confidence': confidence,
    }


def estimate_speed_from_frames(points_a, intensity_a, points_b, intensity_b, dt,
                                min_intensity=150, **kwargs):
    """Скорость между двумя кадрами — всё-в-одном.

    Параметры:
        points_a, intensity_a: первый кадр
        points_b, intensity_b: второй кадр
        dt: время между кадрами, с
        min_intensity: порог яркости для отражателей
        **kwargs: передаются в estimate_speed

    Возвращает:
        dict с результатом или None
    """
    refs_a = find_reflectors(points_a, intensity_a, min_intensity=min_intensity)
    refs_b = find_reflectors(points_b, intensity_b, min_intensity=min_intensity)

    if len(refs_a) < 2 or len(refs_b) < 2:
        return None

    matches = match_reflectors(refs_a, refs_b)
    return estimate_speed(matches, dt, **kwargs)


def estimate_speed_multi(frames, indices, timestamps, min_intensity=150,
                          n_pairs=3, **kwargs):
    """Скорость по нескольким парам кадров — robustнее.

    Берёт n_pairs последовательных пар вокруг текущего кадра,
    возвращает медиану скоростей.

    Параметры:
        frames: BagFrames или список (xyz, intensity, timestamp)
        indices: индексы кадров для анализа (e.g. [idx-3, idx-2, idx-1, idx])
        timestamps: массив таймстемпов (наносекунды)
        n_pairs: сколько пар анализировать
    """
    speeds = []

    for k in range(len(indices) - 1):
        i, j = indices[k], indices[k + 1]
        pts_i, int_i, _ = frames[i]
        pts_j, int_j, _ = frames[j]
        dt = (timestamps[j] - timestamps[i]) / 1e9

        if dt <= 0:
            continue

        result = estimate_speed_from_frames(
            pts_i, int_i, pts_j, int_j, dt,
            min_intensity=min_intensity, **kwargs)

        if result is not None:
            speeds.append(result['speed_ms'])

    if not speeds:
        return None

    median_ms = float(np.median(speeds))
    return {
        'speed_ms': abs(median_ms),
        'speed_kmh': abs(median_ms) * 3.6,
        'direction': 1 if median_ms > 0 else -1,
        'n_estimates': len(speeds),
        'std_kmh': float(np.std(np.abs(np.array(speeds)) * 3.6)),
    }


# --- CLI для тестирования ---
if __name__ == '__main__':
    import argparse
    import sys
    import os

    sys.path.insert(0, os.path.dirname(__file__))
    from visualize_bag import BagFrames, find_db3

    parser = argparse.ArgumentParser(description='Estimate train speed from LiDAR reflectors')
    parser.add_argument('bag_dir', help='папка с bag-файлом')
    parser.add_argument('--start', type=int, default=0, help='начальный кадр')
    parser.add_argument('--n-frames', type=int, default=20, help='сколько кадров анализировать')
    parser.add_argument('--min-intensity', type=int, default=150, help='порог яркости')
    parser.add_argument('--step', type=int, default=1,
                        help='шаг между кадрами для сопоставления (1=соседние, 2=через один)')
    args = parser.parse_args()

    db_path = find_db3(args.bag_dir)
    frames = BagFrames(db_path)
    total = len(frames)
    print(f'Bag: {db_path}, {total} кадров')

    start = max(0, args.start)
    end = min(total, start + args.n_frames)
    step = args.step

    print(f'\nАнализ кадров {start}..{end-1}, step={step}:')
    print(f'{"Frame":>6} {"→":>2} {"Frame":>6} {"dt,ms":>7} {"Refl_A":>7} {"Refl_B":>7} '
          f'{"Matches":>8} {"Consens":>8} {"Speed":>10} {"Conf":>6}')
    print('-' * 75)

    all_speeds = []
    for i in range(start, end - step, 1):
        j = i + step
        if j >= end:
            break

        pts_i, int_i, ts_i = frames[i]
        pts_j, int_j, ts_j = frames[j]
        dt = (ts_j - ts_i) / 1e9

        refs_i = find_reflectors(pts_i, int_i, min_intensity=args.min_intensity)
        refs_j = find_reflectors(pts_j, int_j, min_intensity=args.min_intensity)
        matches = match_reflectors(refs_i, refs_j)
        result = estimate_speed(matches, dt)

        speed_str = '—'
        conf_str = '—'
        n_cons = 0
        if result:
            speed_str = f'{result["speed_kmh"]:.1f} km/h'
            conf_str = result['confidence']
            n_cons = result['n_consensus']
            all_speeds.append(result['speed_kmh'])

        print(f'{i:6d} {"→":>2} {j:6d} {dt*1000:7.0f} {len(refs_i):7d} {len(refs_j):7d} '
              f'{len(matches):8d} {n_cons:8d} {speed_str:>10} {conf_str:>6}')

    if all_speeds:
        s = np.array(all_speeds)
        print(f'\n=== Итого ===')
        print(f'  Медиана: {np.median(s):.1f} km/h')
        print(f'  Среднее: {s.mean():.1f} km/h')
        print(f'  СКО:     {s.std():.1f} km/h')
        print(f'  Мин:     {s.min():.1f} km/h')
        print(f'  Макс:    {s.max():.1f} km/h')
        print(f'  Оценок:  {len(s)} / {end - start - step}')
    else:
        print('\nСкорость не определена — мало отражателей или нет консенсуса')
