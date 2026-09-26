"""
Офлайн-разметка рельсов: глобальные полиномы рельсов.

Подход:
1. Предрассчитать cumulative_dy[i] для всех кадров
2. На каждом кадре детектить рельсы вблизи (0-10м)
3. Рельсовые точки (центры рельсов) сдвинуть по Y в глобальную СК
4. Фитануть два полинома: left_x(global_y), right_x(global_y)
5. При визуализации: для каждой точки кадра посчитать global_y,
   подставить в полином → X рельса → маска по близости

Результат — гладкая линия от первого до последнего кадра, без скачков.

Выход:
- rail_polyfit.npz: left_coeffs, right_coeffs, cumulative_dy, ground_planes
"""

import argparse
import os
import time

import numpy as np

from visualize_bag import BagFrames, find_db3
from rail_detection import find_rail_lines, apply_rail_lines, _estimate_ground_plane


def _precompute_speeds(frames, min_intensity=50, step=3):
    """Предрассчитываем dy для ВСЕХ пар кадров."""
    from speed_estimator import estimate_speed_from_frames

    total = len(frames)
    dys = np.zeros(total - 1)

    print(f'Предрассчёт скорости ({total - 1} пар)...')
    t0 = time.time()

    for i in range(total - 1):
        pts_a, int_a, ts_a = frames[i]
        estimates = []

        for s in [1, step, step * 2]:
            j = i + s
            if j >= total:
                continue
            pts_b, int_b, ts_b = frames[j]
            dt = (ts_b - ts_a) / 1e9
            if dt <= 0 or dt > 5.0:
                continue

            result = estimate_speed_from_frames(
                pts_a, int_a, pts_b, int_b, dt,
                min_intensity=min_intensity,
                min_speed_kmh=0.0, min_consensus=2)

            if result is not None:
                dy_per_frame = result['speed_ms'] * result['direction'] * (
                    (frames.timestamps[i + 1] - ts_a) / 1e9)
                estimates.append(dy_per_frame)

        if estimates:
            dys[i] = float(np.median(estimates))

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f'  {i + 1}/{total - 1}  {elapsed:.1f}s')

    # Сглаживание нулей между ненулевыми
    for i in range(1, len(dys) - 1):
        if dys[i] == 0 and dys[i - 1] != 0 and dys[i + 1] != 0:
            dys[i] = (dys[i - 1] + dys[i + 1]) / 2

    elapsed = time.time() - t0
    nonzero = (dys != 0).sum()
    print(f'  Готово за {elapsed:.1f}s: {nonzero}/{len(dys)} ненулевых')

    return dys


def run_rail_mapper(bag_dir, detect_range=10.0,
                    min_intensity=50, own_x_range=1.5,
                    speed_step=3, poly_degree=5):
    """Строим глобальные полиномы рельсов."""
    db_path = find_db3(bag_dir)
    frames = BagFrames(db_path, cache_size=16)
    total = len(frames)
    print(f'Bag: {db_path}, {total} кадров')

    # 1. Скорости
    dys = _precompute_speeds(frames, min_intensity=min_intensity, step=speed_step)

    cumulative_dy = np.zeros(total)
    for i in range(1, total):
        cumulative_dy[i] = cumulative_dy[i - 1] + dys[i - 1]

    # 2. Собираем центры рельсов в глобальной СК
    print('\nСбор центров рельсов...')
    t0 = time.time()

    left_gy = []   # global Y
    left_gx = []   # X левого рельса
    right_gy = []
    right_gx = []
    ground_planes = []  # (a, b, c) для каждого кадра
    n_detected = 0

    for i in range(total):
        pts, intensity, ts = frames[i]

        rail_lines, ground_plane = find_rail_lines(
            pts, intensity,
            own_x_range=own_x_range,
            max_y_distance=detect_range)

        if not rail_lines or ground_plane is None or len(rail_lines) < 2:
            ground_planes.append(ground_planes[-1] if ground_planes else (0, 0, -1.5))
            continue

        ground_planes.append(ground_plane)
        n_detected += 1

        # rail_lines отсортированы по X (левый, правый)
        s_left, i_left = rail_lines[0]
        s_right, i_right = rail_lines[1]

        # Генерируем точки вдоль рельсов (каждые 0.5м в локальном Y)
        # Y лидара отрицательный (вперёд по ходу = -Y)
        local_ys = np.arange(-detect_range, 0, 0.5)
        for ly in local_ys:
            gy = ly + cumulative_dy[i]
            lx = s_left * ly + i_left
            rx = s_right * ly + i_right
            left_gy.append(gy)
            left_gx.append(lx)
            right_gy.append(gy)
            right_gx.append(rx)

        if (i + 1) % 50 == 0 or i == total - 1:
            elapsed = time.time() - t0
            print(f'  [{i:4d}] detected={n_detected}  points={len(left_gy)}  {elapsed:.1f}s')

    if n_detected < 3:
        print('Слишком мало кадров с рельсами')
        return

    left_gy = np.array(left_gy)
    left_gx = np.array(left_gx)
    right_gy = np.array(right_gy)
    right_gx = np.array(right_gx)

    print(f'\nВсего точек: left={len(left_gy)}, right={len(right_gy)}')
    print(f'Global Y range: {left_gy.min():.1f} .. {left_gy.max():.1f}')

    # 3. Робастный фит: кусочно-линейный через узлы (knots)
    # Полином на длинном Y дрейфует. Вместо этого — узлы каждые knot_step метров,
    # в каждом окне медиана X. Между узлами — линейная интерполяция.
    knot_step = 5.0  # узел каждые 5 метров
    gy_min = min(left_gy.min(), right_gy.min())
    gy_max = max(left_gy.max(), right_gy.max())
    knot_ys = np.arange(gy_min, gy_max + knot_step, knot_step)

    def fit_knots(gy, gx, knot_ys, window=7.5):
        """Медианный X в скользящем окне вокруг каждого узла."""
        knot_xs = []
        valid_ys = []
        for ky in knot_ys:
            mask = np.abs(gy - ky) < window
            if mask.sum() >= 3:
                # MAD-фильтрация выбросов
                x_vals = gx[mask]
                med = np.median(x_vals)
                mad = np.median(np.abs(x_vals - med))
                if mad > 0:
                    good = np.abs(x_vals - med) < 3 * 1.4826 * mad
                    if good.sum() >= 2:
                        x_vals = x_vals[good]
                knot_xs.append(float(np.median(x_vals)))
                valid_ys.append(ky)
        return np.array(valid_ys), np.array(knot_xs)

    left_knot_ys, left_knot_xs = fit_knots(left_gy, left_gx, knot_ys)
    right_knot_ys, right_knot_xs = fit_knots(right_gy, right_gx, knot_ys)

    print(f'Узлы: left={len(left_knot_ys)}, right={len(right_knot_ys)}')

    if len(left_knot_ys) < 2 or len(right_knot_ys) < 2:
        print('Слишком мало узлов')
        return

    # 4. Сохраняем
    out_path = os.path.join(bag_dir, 'rail_polyfit.npz')
    np.savez(out_path,
             left_knot_ys=left_knot_ys,
             left_knot_xs=left_knot_xs,
             right_knot_ys=right_knot_ys,
             right_knot_xs=right_knot_xs,
             cumulative_dy=cumulative_dy,
             ground_planes=np.array(ground_planes))

    wall = time.time() - t0
    print(f'\nГотово за {wall:.1f}с')
    print(f'  {n_detected}/{total} кадров с рельсами')
    print(f'  -> {out_path}')
    print(f'\nДля визуализации:')
    print(f'  python visualize_bag.py {bag_dir} --rail-map')


def main():
    parser = argparse.ArgumentParser(
        description='Офлайн-разметка рельсов: глобальные кривые')
    parser.add_argument('bag_dir', help='папка с bag-файлом')
    parser.add_argument('--detect-range', type=float, default=10.0,
                        help='дальность детекции рельсов, м (по умолчанию 10)')
    parser.add_argument('--min-intensity', type=int, default=50,
                        help='порог яркости для отражателей (по умолчанию 50)')
    parser.add_argument('--own-x-range', type=float, default=1.5,
                        help='макс. смещение центра колеи от лидара по X, м')
    parser.add_argument('--speed-step', type=int, default=3,
                        help='шаг для оценки скорости (по умолчанию 3)')
    args = parser.parse_args()

    run_rail_mapper(
        args.bag_dir,
        detect_range=args.detect_range,
        min_intensity=args.min_intensity,
        own_x_range=args.own_x_range,
        speed_step=args.speed_step,
    )


if __name__ == '__main__':
    main()
