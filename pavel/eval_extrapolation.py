"""
Замер: насколько точно можно предсказать траекторию пути вперёд.

Вопрос, от которого зависит архитектура детектора: можно ли построить
габаритный коридор на 100-200 м чистой геометрией, или нужна модель,
«додумывающая» путь по контексту (нейросеть)?

Идея замера: траектория, пройденная поездом, — это и есть осевая линия пути,
измеренная точно. Значит, для кадра i истинный путь вперёд известен из поз
кадров i+1, i+2, ... Сравниваем с тем, что даёт экстраполяция по короткому
участку наблюдений, и смотрим на ошибку в метрах на разных дальностях.

Три модели экстраполяции, от простой к сложной:
  straight  — путь идёт прямо (нулевая кривизна)
  arc       — постоянная кривизна (дуга окружности)
  clothoid  — кривизна линейна по длине (переходная кривая, как в путейном деле)

Ошибка меряется как боковое отклонение (по X в системе кадра i) — именно оно
определяет, попадёт ли объект в габарит.
"""

import argparse
import os

import numpy as np

from odometry import transform_to_local

# Дальности, на которых оцениваем ошибку. Кейс требует уверенной детекции
# на 100-300 м, поэтому смотрим весь диапазон.
RANGES = (25.0, 50.0, 100.0, 150.0, 200.0, 300.0)

# Полуширина габарита поезда — масштаб, с которым сравниваем ошибку.
# Ошибка сильно меньше него означает, что коридор строится надёжно.
HALF_GAUGE = 1.7


def _true_path_local(poses, i, max_range):
    """Истинный путь впереди кадра i, в локальной системе этого кадра.

    Берём позиции будущих кадров (центр лидара = точка на осевой линии пути)
    и переводим их в систему кадра i. Возвращает (s, x) — пройденный вперёд
    путь и боковое смещение.
    """
    fut = poses[i:]
    local = transform_to_local(fut[:, :2], poses[i])
    # Ось Y лидара смотрит назад: вперёд по ходу движения — это -Y.
    s = -local[:, 1]
    x = local[:, 0]
    keep = (s >= 0) & (s <= max_range)
    return s[keep], x[keep]


def _fit_and_predict(s_obs, x_obs, s_pred, model):
    """Строит модель по наблюдённому участку и предсказывает боковое смещение.

    Все три модели выражены как полиномы по пройденному пути s:
      straight: x = a0 + a1·s          (кривизна 0)
      arc:      x = a0 + a1·s + a2·s²  (кривизна const, 2·a2)
      clothoid: x = a0 + ... + a3·s³   (кривизна линейна по s)
    Для малых углов это точные представления соответствующих кривых.
    """
    deg = {'straight': 1, 'arc': 2, 'clothoid': 3}[model]
    if len(s_obs) < deg + 1:
        return None
    coeffs = np.polyfit(s_obs, x_obs, deg)
    return np.polyval(coeffs, s_pred)


def evaluate(bag_dir, obs_window=30.0, models=('straight', 'arc', 'clothoid'),
             min_speed=0.3):
    """Считает ошибку экстраполяции по всем кадрам бэга.

    obs_window — длина участка перед поездом, по которому строится модель.
    Это тот участок, где рельсы детектируются надёжно.

    min_speed — кадры на стоянке пропускаем: там впереди нет пройденного
    пути, замерять нечего.
    """
    traj = np.load(os.path.join(bag_dir, 'trajectory.npz'))
    poses, arc = traj['poses'], traj['arc']
    total = len(poses)

    errors = {m: {r: [] for r in RANGES} for m in models}
    n_used = 0

    for i in range(total):
        # Нужен и участок наблюдения, и «будущее» достаточной длины
        if arc[-1] - arc[i] < min(RANGES):
            break
        if i + 1 < total and arc[i + 1] - arc[i] < min_speed:
            continue

        s_true, x_true = _true_path_local(poses, i, max(RANGES))
        if len(s_true) < 6:
            continue

        # Наблюдаемый участок — ближние obs_window метров того же пути.
        # Это стоит за детекцией рельсов на 0-30 м: там она надёжна.
        obs = s_true <= obs_window
        if obs.sum() < 4:
            continue
        s_obs, x_obs = s_true[obs], x_true[obs]

        n_used += 1
        for m in models:
            for r in RANGES:
                if s_true[-1] < r:
                    continue
                x_ref = np.interp(r, s_true, x_true)
                pred = _fit_and_predict(s_obs, x_obs, np.array([r]), m)
                if pred is not None:
                    errors[m][r].append(abs(pred[0] - x_ref))

    return errors, n_used


def _fmt_row(label, vals):
    return label + ''.join(f'{v:>12}' for v in vals)


def report(bag_name, errors, n_used):
    print(f'\n=== {bag_name}  ({n_used} кадров) ===')
    header = _fmt_row(f'{"модель":<10}', [f'{int(r)}м' for r in RANGES])
    print(header)
    print('-' * len(header))

    for m, per_range in errors.items():
        med, p95 = [], []
        for r in RANGES:
            e = per_range[r]
            med.append(f'{np.median(e):.2f}' if e else '—')
            p95.append(f'{np.percentile(e, 95):.2f}' if e else '—')
        print(_fmt_row(f'{m:<10}', med))
        print(_fmt_row(f'{"  p95":<10}', p95))


def main():
    parser = argparse.ArgumentParser(
        description='Точность геометрической экстраполяции пути вперёд')
    parser.add_argument('bag_dirs', nargs='+')
    parser.add_argument('--obs-window', type=float, default=30.0,
                        help='длина наблюдаемого участка, м (по умолчанию 30)')
    args = parser.parse_args()

    pooled = {m: {r: [] for r in RANGES}
              for m in ('straight', 'arc', 'clothoid')}
    pooled_n = 0

    for bag_dir in args.bag_dirs:
        if not os.path.exists(os.path.join(bag_dir, 'trajectory.npz')):
            print(f'{bag_dir}: нет trajectory.npz — сначала odometry.py --save')
            continue
        errors, n_used = evaluate(bag_dir, obs_window=args.obs_window)
        report(os.path.basename(bag_dir.rstrip('/')), errors, n_used)
        pooled_n += n_used
        for m in pooled:
            for r in RANGES:
                pooled[m][r].extend(errors[m][r])

    if pooled_n:
        print(f'\n{"#" * 60}')
        report(f'ВСЕ БЭГИ (окно наблюдения {args.obs_window:.0f} м)',
               pooled, pooled_n)
        print(f'\nПолуширина габарита для сравнения: {HALF_GAUGE} м')


if __name__ == '__main__':
    main()
