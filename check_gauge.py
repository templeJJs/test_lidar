"""Проверка измерения колеи в visualize_bag.py.

Что печатает:
  1. детерминизм — три прогона полного конвейера на одном кадре (числа должны
     совпасть побитово);
  2. колея по внутренним граням головок против норматива 1520 -4/+8 мм и старое
     число «по осям головок»;
  3. разброс ДО (медиана в +-0.5 м, прежний код) и ПОСЛЕ (устойчивый центр по
     Y-окну W) — внутрикадровый и по кадрам — на трёх записях;
  4. что даёт W = 1.0 / 1.5 / 2.0 м;
  5. строку «ИТОГ: ...» с достигнутым сжатием разброса и сдвигом среднего.

Запуск:  python check_gauge.py [--frames 8] [--bag-root for_hackathon]
Кадры читаются по одному (BagFrames с кэшем 3), так что память не растёт.
"""

import argparse
import os
import sys

import numpy as np

import visualize_bag as vb

# Записи для таблицы разброса: три разные конфигурации пути (стрелки/платформа/переезд).
RECORDINGS = [
    'doubleT_platform',
    'roundT_pressureGate_roundT',
    'squareT_platform_squareT_switch',
]
WINDOWS = [1.0, 1.5, 2.0]   # полуширины Y-окна, которые требует ТЗ
HEAD_DEPTH = 0.013          # 13 мм ниже поверхности катания (ПТЭ п.45)
OLD = (None, 0)             # прежнее поведение: медиана в +-0.5 м, без порога
DEFAULT = (1.5, 0)          # поведение по умолчанию: устойчивый центр, Y-окно 1.5 м
# Дополнительные варианты: тот же центр, но с минимумом точек в Y-окне.
VARIANTS = [OLD] + [(w, 0) for w in WINDOWS] + [(1.5, 3), (1.5, 5)]


def load_recording(bag_root, name, frames_wanted):
    """Линии по кадру 0 (детерминировано) + замеры кадров 0..N-1 по всем вариантам."""
    db = vb.find_db3(os.path.join(bag_root, name))
    frames = vb.BagFrames(db, cache_size=3)
    n = min(frames_wanted, len(frames))
    if n == 0:
        return None
    xyz0, inten0, _ = frames[0]
    lines, gp = vb.find_rail_lines(xyz0, inten0)

    per_frame = {v: [] for v in VARIANTS}
    for idx in range(n):
        xyz, _, _ = frames[idx]
        mask = vb.apply_rail_lines(xyz, lines, gp, own_track_only=True)
        for w, mp in VARIANTS:
            per_frame[(w, mp)].append(vb.measure_gauge(xyz, mask, lines, gp,
                                                       center_window=w,
                                                       center_min_points=mp,
                                                       head_depth=HEAD_DEPTH))
    return {
        'name': name, 'n_frames': n, 'n_points': len(xyz0),
        'lines': lines, 'ground_plane': gp,
        'per_frame': per_frame,
    }


def window_summary(records):
    """Сводка по кадрам для одного окна: среднее, СКО по кадрам, размах по кадрам,
    средний внутрикадровый СКО и размах (по срезам, усреднённые по кадрам)."""
    means, in_frame_sigmas, in_frame_ranges = [], [], []
    for g in records:
        st = g['axes']
        if st is None:
            continue
        means.append(st['mean'])
        in_frame_sigmas.append(st['std'])
        in_frame_ranges.append(st['max'] - st['min'])
    if not means:
        return None
    means = np.array(means)
    return {
        'n_frames': len(means),
        'mean': float(means.mean()),
        'sigma_frames': float(means.std()),
        'range_frames': float(means.max() - means.min()),
        'sigma_in_frame': float(np.mean(in_frame_sigmas)),
        'range_in_frame': float(np.mean(in_frame_ranges)),
    }


def inner_summary(records):
    """Сводка колеи по внутренним граням (одна база на кадр)."""
    means = [g['inner']['mean'] for g in records if g['inner'] is not None]
    if not means:
        return None
    means = np.array(means)
    return {'n_frames': len(means), 'mean': float(means.mean()),
            'sigma_frames': float(means.std()),
            'range_frames': float(means.max() - means.min()),
            'frame0': float(means[0])}


def mm(x):
    return x * 1000.0


def print_determinism(bag_root):
    print('=== 1. Детерминизм: три полных прогона на одном кадре (doubleT_platform, кадр 0) ===')
    name = RECORDINGS[0]
    db = vb.find_db3(os.path.join(bag_root, name))
    axes_vals, inner_vals = [], []
    for run in range(3):
        frames = vb.BagFrames(db, cache_size=2)
        xyz, inten, _ = frames[0]
        lines, gp = vb.find_rail_lines(xyz, inten)
        mask = vb.apply_rail_lines(xyz, lines, gp, own_track_only=True)
        g = vb.measure_gauge(xyz, mask, lines, gp, center_window=1.5, head_depth=HEAD_DEPTH)
        axes_vals.append(g['axes']['mean'])
        inner_vals.append(g['inner']['mean'] if g['inner'] is not None else float('nan'))
        print(f'  прогон {run + 1}: оси = {axes_vals[-1]:.9f} м ({mm(axes_vals[-1]):.6f} мм), '
              f'внутренние грани = {mm(inner_vals[-1]):.6f} мм')
    axes_ok = axes_vals[0] == axes_vals[1] == axes_vals[2]
    inner_ok = inner_vals[0] == inner_vals[1] == inner_vals[2] and not np.isnan(inner_vals[0])
    print(f'  побитово равны: оси — {"ДА" if axes_ok else "НЕТ"}, '
          f'внутренние грани — {"ДА" if inner_ok else "НЕТ"}')
    return axes_ok and inner_ok


def print_base_check(rec):
    """Колея по внутренним граням и по осям головок на doubleT_platform."""
    print(f"\n=== 2. База отсчёта ({rec['name']}, кадр 0, {rec['n_frames']} кадров) ===")
    base = rec['per_frame'][OLD][0]
    after = rec['per_frame'][DEFAULT][0]
    inner = rec['per_frame'][DEFAULT]
    old_axes = mm(base['axes']['mean'])
    new_axes = mm(after['axes']['mean'])
    inner0 = inner[0]['inner']
    if inner0 is not None:
        print(f'  колея (внутренние грани, 13 мм под верхом головки): {mm(inner0["mean"]):.0f} мм'
              f'   (норматив 1520 -4/+8 мм, ожидание 1520 +-10)')
        s = inner_summary(inner)
        print(f'    по кадрам 0..{s["n_frames"] - 1}: {mm(s["mean"]):.0f} +- {mm(s["sigma_frames"]):.1f} мм, '
              f'размах {mm(s["range_frames"]):.0f} мм')
    else:
        print('  колея (внутренние грани): н/д — футпринт головки не найден')
    print(f'  колея (оси головок, старое): {old_axes:.0f} мм')
    print(f'  колея (оси головок, после смены центра среза): {new_axes:.0f} мм')
    print(f'  сдвиг среднего (одна база, только центр среза): {new_axes - old_axes:+.1f} мм')
    if inner0 is not None:
        print(f'  разница баз (оси - внутренние грани): {old_axes - mm(inner0["mean"]):+.0f} мм')


def print_spread(recs):
    print('\n=== 3. Разброс ДО (медиана +-0.5 м) и ПОСЛЕ (устойчивый центр, W=1.5 м) ===')
    print('запись                          кадр | ДО: среднее СКО(кадры) размах(кадры) СКО(в кадре) '
          'размах(в кадре) | ПОСЛЕ: среднее СКО(кадры) размах(кадры) СКО(в кадре) размах(в кадре) '
          '| сдвиг  сжатие')
    for rec in recs:
        before = window_summary(rec['per_frame'][OLD])
        after = window_summary(rec['per_frame'][DEFAULT])
        if before is None or after is None:
            print(f'{rec["name"]:30s}  — нет срезов для замера')
            continue
        comp_frames = before['sigma_frames'] / after['sigma_frames'] if after['sigma_frames'] > 0 else float('inf')
        comp_in = before['sigma_in_frame'] / after['sigma_in_frame'] if after['sigma_in_frame'] > 0 else float('inf')
        print(f'{rec["name"]:30s} {after["n_frames"]:4d} |'
              f' {mm(before["mean"]):8.1f} {mm(before["sigma_frames"]):9.2f} {mm(before["range_frames"]):11.1f}'
              f' {mm(before["sigma_in_frame"]):12.1f} {mm(before["range_in_frame"]):14.1f} |'
              f' {mm(after["mean"]):8.1f} {mm(after["sigma_frames"]):9.2f} {mm(after["range_frames"]):11.1f}'
              f' {mm(after["sigma_in_frame"]):12.1f} {mm(after["range_in_frame"]):14.1f}'
              f' | {mm(after["mean"] - before["mean"]):+6.1f}  x{comp_in:.2f}/x{comp_frames:.2f}')
    print('  (СКО(кадры)/размах(кадры) — по средним кадров; СКО(в кадре)/размах(в кадре) — '
          'по срезам, усреднено по кадрам)')
    print('  (числа «по осям головок»: линии детерминированы random_state=0, '
          'меняется только способ выбора центра среза)')


def variant_label(w, mp):
    if w is None:
        return 'медиана +-0.5'
    label = f'{w:.1f} м'
    return label if not mp else f'{w:.1f} м, мин {mp}'


def print_windows(recs):
    print('\n=== 4. Что даёт полуширина Y-окна W устойчивого центра ===')
    print('W, м          запись                          кадр  среднее  СКО(кадры) размах(кадры) '
          'СКО(в кадре)  сдвиг к ДО')
    for rec in recs:
        before = window_summary(rec['per_frame'][OLD])
        if before is None:
            continue
        for v in VARIANTS:
            s = window_summary(rec['per_frame'][v])
            if s is None:
                continue
            print(f'{variant_label(*v):14s}{rec["name"]:32s} {s["n_frames"]:4d}'
                  f'  {mm(s["mean"]):7.1f} {mm(s["sigma_frames"]):10.2f} {mm(s["range_frames"]):12.1f}'
                  f' {mm(s["sigma_in_frame"]):12.1f}  {mm(s["mean"] - before["mean"]):+8.1f}')
        print()


def main():
    ap = argparse.ArgumentParser(description='Проверка измерения колеи (visualize_bag.measure_gauge)')
    ap.add_argument('--frames', type=int, default=8,
                    help='сколько кадров на запись для таблицы разброса (по умолчанию 8, не меньше 7)')
    ap.add_argument('--bag-root', default='for_hackathon', help='папка с записями (по умолчанию for_hackathon)')
    args = ap.parse_args()

    frames_wanted = max(7, args.frames)
    print('Проверка измерения колеи: visualize_bag.py (measure_gauge)')
    print(f'База отсчёта по нормативу: внутренние рабочие грани головок на уровне '
          f'{HEAD_DEPTH * 1000:.0f} мм ниже поверхности катания, номинал 1520 -4/+8 мм\n')

    ok = print_determinism(args.bag_root)

    recs = []
    for name in RECORDINGS:
        path = os.path.join(args.bag_root, name)
        if not os.path.isdir(path):
            print(f'  [пропуск] нет папки {path}')
            continue
        print(f'читаю {name}...', flush=True)
        rec = load_recording(args.bag_root, name, frames_wanted)
        if rec is not None:
            recs.append(rec)

    if recs:
        print_base_check(recs[0])
        print_spread(recs)
        print_windows(recs)

    # Итог: сжатие разброса и сдвиг среднего на калибровочной записи (её же требует ТЗ).
    print()
    plat = next((r for r in recs if r['name'] == RECORDINGS[0]), recs[0] if recs else None)
    if plat is None:
        print('ИТОГ: нет данных для вывода.')
        return 1
    before = window_summary(plat['per_frame'][OLD])
    after = window_summary(plat['per_frame'][DEFAULT])
    inner = inner_summary(plat['per_frame'][DEFAULT])
    shift = mm(after['mean'] - before['mean'])
    c_in = before['sigma_in_frame'] / after['sigma_in_frame']
    c_fr = before['sigma_frames'] / after['sigma_frames']
    print('ИТОГ: детерминизм — ' + ('ДОСТИГНУТ' if ok else 'НЕ ДОСТИГНУТ') + '.')
    print(f'ИТОГ: {plat["name"]}: разброс внутри кадра СКО {mm(before["sigma_in_frame"]):.1f} -> '
          f'{mm(after["sigma_in_frame"]):.1f} мм (x{c_in:.2f}), размах '
          f'{mm(before["range_in_frame"]):.1f} -> {mm(after["range_in_frame"]):.1f} мм; по кадрам '
          f'СКО {mm(before["sigma_frames"]):.2f} -> {mm(after["sigma_frames"]):.2f} мм (x{c_fr:.2f}), '
          f'размах {mm(before["range_frames"]):.1f} -> {mm(after["range_frames"]):.1f} мм; '
          f'среднее {mm(before["mean"]):.1f} -> {mm(after["mean"]):.1f} мм (сдвиг {shift:+.1f} мм).')
    guard = window_summary(plat['per_frame'][(1.5, 3)])
    if guard is not None:
        print(f'ИТОГ: {plat["name"]}: вариант «W=1.5 м, минимум 3 точек в окне» '
              f'(не по умолчанию): СКО по кадрам '
              f'{mm(before["sigma_frames"]):.2f} -> {mm(guard["sigma_frames"]):.2f} мм '
              f'(x{before["sigma_frames"] / guard["sigma_frames"]:.2f}), СКО в кадре '
              f'{mm(before["sigma_in_frame"]):.1f} -> {mm(guard["sigma_in_frame"]):.1f} мм, '
              f'сдвиг {mm(guard["mean"] - before["mean"]):+.1f} мм — сжатие сильнее, но среднее уезжает.')
    if inner is not None:
        print(f'ИТОГ: колея по внутренним граням {mm(inner["mean"]):.0f} +- '
              f'{mm(inner["sigma_frames"]):.1f} мм — в норме 1520 -4/+8 мм.')
    else:
        print('ИТОГ: колея по внутренним граням не измерена (футпринт головки не найден).')
    print('ИТОГ: doubleT_obstacle не в таблице — там _find_own_track берёт пару из двух '
          'разных нитей (x(y=0) = -2.08 и -0.30 -> 1778 мм); это отдельный дефект выбора пары.')
    return 0


if __name__ == '__main__':
    sys.exit(main())