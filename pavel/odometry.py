"""
SE(2)-одометрия поезда по облакам точек LiDAR.

Прежний speed_estimator оценивал только dY (сдвиг вдоль пути). Этого хватает
на прямой, но на повороте кадр i повёрнут относительно кадра 0 на угол theta_i,
и локальный X точки — координата в повёрнутой системе. Складывать такие X в одну
карту нельзя: медиана по точкам из разных систем систематически спрямляет поворот.

Здесь оценивается полное плоское преобразование (dx, dy, dtheta) между кадрами,
из него интегрируется траектория поз (x, y, theta) в глобальной системе кадра 0.

Оси лидара: Y — вдоль пути (вперёд), X — поперёк, Z — вверх.
Плоское движение: поезд не летает, крен/тангаж игнорируем.
"""

import numpy as np

from speed_estimator import find_reflectors


def _match_pairs(refs_a, refs_b, max_dx=0.4, max_dz=0.4, max_dy=25.0):
    """Сопоставляет отражатели между кадрами, возвращает пары точек в XY.

    Матчинг по XZ: при движении вдоль Y статичный объект почти не меняет
    X и Z, поэтому они работают как «подпись» объекта. Z в само преобразование
    не входит (движение плоское), поэтому наружу отдаём только XY.

    Неоднозначность разрешается ВЗАИМНО БЛИЖАЙШИМ соседом, а не отбрасыванием.
    Прежде пара отбрасывалась целиком, если у отражателя было больше одного
    кандидата, — а в тоннеле отражателей много и они похожи, так что кандидатов
    почти всегда несколько. В итоге из 7-11 найденных отражателей до Кабша
    доходило 0-2 пары при пороге min_pairs=3, и оценка движения проваливалась:
    прямые оценки были лишь на 62% кадров, остальное интерполировалось.
    Замер на roundT_pressureGate_roundT: кадров с >=3 парами 173 -> 262 из 267.

    Возвращает (A, B) — массивы (M, 2) соответствующих точек XY.
    """
    if len(refs_a) < 2 or len(refs_b) < 2:
        return np.empty((0, 2)), np.empty((0, 2))

    ca = np.array([r['center'] for r in refs_a])
    cb = np.array([r['center'] for r in refs_b])

    # Кандидаты: близость по X и Z, разумный сдвиг по Y
    dx = np.abs(ca[:, None, 0] - cb[None, :, 0])
    dz = np.abs(ca[:, None, 2] - cb[None, :, 2])
    dy = cb[None, :, 1] - ca[:, None, 1]
    ok = (dx < max_dx) & (dz < max_dz) & (np.abs(dy) < max_dy)

    # Взаимно ближайшие по XZ-подписи: i выбирает лучший j, j выбирает тот же i.
    # Остальную грязь снимет отбраковка по остатку в estimate_motion.
    cost = np.where(ok, np.hypot(dx, dz), np.inf)
    best_b = cost.argmin(axis=1)
    best_a = cost.argmin(axis=0)
    rows = np.arange(len(ca))
    keep = np.isfinite(cost[rows, best_b]) & (best_a[best_b] == rows)
    ii = rows[keep]
    jj = best_b[keep]

    return ca[ii][:, [0, 1]], cb[jj][:, [0, 1]]


def _kabsch_2d(A, B):
    """Жёсткое 2D-преобразование, переводящее A в B (метод Кабша).

    Находит R (поворот) и t (сдвиг), минимизирующие |R·A + t - B|².
    Отражения исключены: det(R) принудительно +1, иначе при вырожденной
    геометрии решение может «вывернуть» сцену зеркально.

    Возвращает (dx, dy, dtheta) — преобразование из системы A в систему B,
    то есть движение СЦЕНЫ относительно сенсора.
    """
    ca = A.mean(axis=0)
    cb = B.mean(axis=0)
    H = (A - ca).T @ (B - cb)

    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, d]) @ U.T

    t = cb - R @ ca
    theta = float(np.arctan2(R[1, 0], R[0, 0]))
    return float(t[0]), float(t[1]), theta


def estimate_motion(refs_a, refs_b, max_dtheta=0.12, inlier_tol=0.35,
                    min_pairs=3, n_iter=3):
    """Оценивает (dx, dy, dtheta) между двумя кадрами по их отражателям.

    Кабш чувствителен к выбросам — одна ошибочная пара разворачивает решение.
    Поэтому после первой оценки выбросы отбрасываются по остатку и фит
    повторяется (IRLS-подобная схема, дешевле RANSAC и здесь достаточна,
    т.к. взаимно-однозначный матчинг уже отсеял большую часть мусора).

    max_dtheta — физический предел поворота за кадр (~0.12 рад ≈ 7°): при
    10 Гц и радиусе кривой метро 300 м реальный поворот на порядок меньше,
    так что превышение означает развал оценки, а не крутой поворот.

    Возвращает dict или None, если надёжной оценки нет.
    """
    A, B = _match_pairs(refs_a, refs_b)
    if len(A) < min_pairs:
        return None

    idx = np.arange(len(A))
    dx = dy = theta = 0.0

    for _ in range(n_iter):
        if len(idx) < min_pairs:
            return None
        dx, dy, theta = _kabsch_2d(A[idx], B[idx])

        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        pred = A @ R.T + np.array([dx, dy])
        resid = np.linalg.norm(pred - B, axis=1)

        new_idx = np.nonzero(resid < inlier_tol)[0]
        if len(new_idx) == len(idx):
            break
        idx = new_idx

    if len(idx) < min_pairs or abs(theta) > max_dtheta:
        return None

    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    resid = np.linalg.norm(A[idx] @ R.T + np.array([dx, dy]) - B[idx], axis=1)

    # Кабш измерил движение сцены относительно сенсора; движение сенсора —
    # обратное преобразование. Ось Y лидара направлена назад (вперёд = -Y),
    # поэтому у едущего вперёд поезда dy получается отрицательным.
    mdx, mdy, mdth = _invert((dx, dy, theta))

    return {
        'dx': mdx,
        'dy': mdy,
        'dtheta': mdth,
        'n_pairs': int(len(idx)),
        'rms': float(np.sqrt((resid ** 2).mean())),
    }


def estimate_motion_from_frames(pts_a, int_a, pts_b, int_b, min_intensity=50,
                                **kwargs):
    """estimate_motion прямо по двум облакам точек."""
    refs_a = find_reflectors(pts_a, int_a, min_intensity=min_intensity)
    refs_b = find_reflectors(pts_b, int_b, min_intensity=min_intensity)
    return estimate_motion(refs_a, refs_b, **kwargs)


def _compose(pose, motion):
    """Накладывает относительное движение на позу: pose ⊕ motion (SE(2))."""
    x, y, th = pose
    dx, dy, dth = motion
    c, s = np.cos(th), np.sin(th)
    return (x + c * dx - s * dy,
            y + s * dx + c * dy,
            th + dth)


def _invert(motion):
    """Обратное преобразование SE(2)."""
    dx, dy, dth = motion
    c, s = np.cos(-dth), np.sin(-dth)
    return (-(c * dx - s * dy), -(s * dx + c * dy), -dth)


def _estimate_weight(m, rms_scale=0.03, min_pairs=3):
    """Насколько можно доверять оценке движения: вес для слияния баз.

    Кабш по трём-четырём парам формально проходит все проверки, но угол в нём
    определяется почти произвольно: лишние пары — это избыточность, которая
    только и делает поворот наблюдаемым. Поэтому вес растёт с числом пар СВЕРХ
    минимума (n - min_pairs + 1), а не с самим числом пар: между 3 и 4 парами
    разница вдвое, между 10 и 11 — почти никакой, что и отражает реальность.

    Остаток фита входит как 1/(1 + (rms/rms_scale)^2) — по-гауссовски, как
    обратная дисперсия. rms_scale = 0.03 м взят по самим данным: у здоровых
    оценок rms 0.02-0.04 м (замер на roundT_doubleT, кадры 157-168), так что
    поправка там около единицы и веса решает число пар, а развалившийся фит
    с rms 0.11 давится в ~8 раз.

    Замер на roundT_doubleT f160: 4 пары, rms 0.106 -> вес 0.15 против 4.4 у
    соседнего f158 (7 пар, rms 0.023), то есть в 30 раз меньше.
    """
    n_extra = max(float(m['n_pairs']) - min_pairs + 1.0, 0.0)
    rms = float(m['rms'])
    return n_extra / (1.0 + (rms / rms_scale) ** 2)


def _fuse_candidates(candidates, weights):
    """Сводит оценки движения от разных баз в одну с учётом их веса.

    Прежде здесь была np.median по кандидатам — она игнорирует, что оценки
    неравноточны: фит по 4 шумным парам весил столько же, сколько по 7
    чистым. На кадрах, где надёжных баз одна-две, медиана могла встать прямо
    на слабую оценку (roundT_doubleT f160: -1.34°/кадр против -0.2..-0.4° у
    соседей, отсюда излом карты 10° на s=260).

    Взвешенное среднее по dx/dy; угол усредняется через вектор (cos, sin),
    чтобы не ломаться на переходе ±pi.
    """
    C = np.asarray(candidates, dtype=float)
    w = np.asarray(weights, dtype=float)

    # Вырожденный случай (все веса нулевые) — считаем оценки равноправными,
    # поведение как у прежней медианы.
    if not np.isfinite(w).all() or w.sum() <= 0:
        return np.median(C, axis=0)

    w = w / w.sum()
    dx = float(C[:, 0] @ w)
    dy = float(C[:, 1] @ w)
    th = float(np.arctan2(np.sin(C[:, 2]) @ w, np.cos(C[:, 2]) @ w))
    return np.array([dx, dy, th])


def compute_trajectory(frames, min_intensity=50, steps=(1, 2, 3), verbose=True,
                       progress_every=50):
    """Строит траекторию поз (x, y, theta) для всех кадров бэга.

    Движение между соседними кадрами оценивается по нескольким базам
    (кадр i→i+1, i→i+2, i→i+3): дальняя база точнее по углу (больше плечо,
    меньше влияние шума центров кластеров), ближняя надёжнее по матчингу.
    Оценки с дальних баз приводятся к шагу одного кадра через композицию
    уже известных движений и усредняются по медиане.

    Кадры без оценки (мало отражателей, стоянка) получают нулевое движение,
    а затем заполняются интерполяцией между соседними валидными.

    Возвращает dict с:
        poses:  (N, 3) — x, y, theta в системе кадра 0
        motions:(N-1, 3) — dx, dy, dtheta между соседними кадрами
        valid:  (N-1,) bool — была ли прямая оценка
        arc:    (N,) — пройденный путь (длина дуги), м
    """
    import time

    total = len(frames)
    if total < 2:
        return None

    max_step = max(steps)

    if verbose:
        print(f'Одометрия SE(2): {total} кадров...')
    t0 = time.time()

    # Отражатели считаем один раз на кадр — самая дорогая часть (DBSCAN)
    refs = []
    for i in range(total):
        pts, intensity, _ = frames[i]
        refs.append(find_reflectors(pts, intensity, min_intensity=min_intensity))
        if verbose and progress_every and (i + 1) % progress_every == 0:
            print(f'  отражатели {i + 1}/{total}  {time.time() - t0:.1f}s')

    # Прямые оценки для всех используемых баз.
    # Кроме самого движения храним, НА ЧЁМ оно построено: число пар и остаток
    # фита. Без этого все оценки кадра неразличимы, и слабая (мало пар, большой
    # остаток) весит наравне с надёжной — см. _estimate_weight.
    raw = {}
    for s in steps:
        for i in range(total - s):
            m = estimate_motion(refs[i], refs[i + s])
            if m is not None:
                raw[(i, s)] = ((m['dx'], m['dy'], m['dtheta']),
                               _estimate_weight(m))

    motions = np.zeros((total - 1, 3))
    valid = np.zeros(total - 1, dtype=bool)
    # Совокупный вес оценки кадра — на чём она держится. Нужен, чтобы отличить
    # одинокий развалившийся фит от честной оценки на повороте.
    conf = np.zeros(total - 1)
    n_rejected = 0
    n_dx_fixed = 0

    # Сначала базовые оценки шагом 1
    for i in range(total - 1):
        if (i, 1) in raw:
            motions[i] = raw[(i, 1)][0]
            valid[i] = True

    # Уточнение дальними базами: из i→i+s вычитаем уже известные i+1→i+s
    for i in range(total - 1):
        candidates, weights = [], []
        if valid[i]:
            candidates.append(motions[i])
            weights.append(raw[(i, 1)][1] if (i, 1) in raw else 1.0)
        for s in steps:
            if s == 1 or (i, s) not in raw:
                continue
            if not valid[i + 1:i + s].all():
                continue
            tail = (0.0, 0.0, 0.0)
            for k in range(i + 1, i + s):
                tail = _compose(tail, motions[k])
            # motion(i→i+1) = motion(i→i+s) ⊕ inverse(tail)
            mot, w = raw[(i, s)]
            candidates.append(np.array(_compose(mot, _invert(tail))))
            # Вычитание хвоста переносит в оценку и ошибки самого хвоста,
            # поэтому длинная база доверия не прибавляет: делим вес на s.
            weights.append(w / s)
        if candidates:
            motions[i] = _fuse_candidates(candidates, weights)
            valid[i] = True
            conf[i] = float(np.sum(weights))

    # Одинокая слабая оценка, спорящая с соседями по УГЛУ, — это развал фита,
    # а не поворот. Вес её уже ловит (_estimate_weight), но вес работает
    # только ВНУТРИ кадра, между кандидатами: если база всего одна,
    # сравнивать не с чем, и оценка проходит как есть. Так на roundT_doubleT
    # кадр 160 (4 пары, rms 0.106, вес 0.15, единственный кандидат) дал
    # -1.34°/кадр против -0.18..-0.40° у соседей и излом карты ~10° на s=260.
    #
    # Отбраковываем только когда оценка И расходится с соседями, И слабее их
    # по весу. Сильная оценка проверку переживает — настоящий поворот
    # стрелки так не потерять.
    #
    # Снос dx лечится ЗАМЕНОЙ, а не отбраковкой, и это принципиально. Снятый
    # кадр отдаётся интерполяции целиком, вместе со здоровыми dy и dtheta, и
    # там, где прямых оценок мало, дыра в цепочке вредит больше исходного
    # выброса: на roundT_doubleT отбраковка кадров 174-175 перед остановкой
    # растила излом s=306 с 4.9° до 5.7°, а участок s=242..306 — с 3.5° до
    # 5.2°. Замена трогает только испорченное число, длина цепочки та же.
    #
    # Выброс сноса — это тот же развал фита, что и по углу, но угловая
    # проверка его не видит: у roundT_doubleT кадра 89 dtheta почти нулевой
    # при dx=+0.132 м, у roundT_squareT_pressureGate_squareT кадра 497 —
    # dtheta -0.021° при dx=+0.226 (4 пары, rms 0.197, вес 0.05, база s=2
    # единственная). Поезд на рельсах вбок не съезжает: типичный |dx| здесь
    # 0.01-0.02 м, и всё сверх того — ошибка оценки, а не движение.
    if valid.sum() >= 5:
        half = 3
        for i in np.nonzero(valid)[0]:
            lo, hi = max(0, i - half), min(total - 1, i + half + 1)
            near = [k for k in range(lo, hi) if k != i and valid[k]]
            if len(near) < 3:
                continue
            weak = conf[i] < 0.5 * np.median(conf[near])

            med_th = np.median(motions[near, 2])
            # Порог в 3 раза выше типичного разброса соседей, но не ниже
            # 0.35° — иначе на идеально ровном участке отсеются и здоровые.
            spread = np.median(np.abs(motions[near, 2] - med_th))
            if abs(motions[i, 2] - med_th) > max(3.0 * spread, np.radians(0.35)):
                # Снимаем, только если опереться на неё нечем: слабее соседей.
                if weak:
                    valid[i] = False
                    n_rejected += 1
                    continue

            # Снос проверяется только на ходу: на стоянке (doubleT_obstacle —
            # 11 м за весь бэг, 0.046 м/кадр) dx это уже не движение, а шум
            # того же порядка, что сам шаг, и критерий вырождается.
            ds = np.median(np.hypot(motions[near, 0], motions[near, 1]))
            if ds <= 0.5:
                continue
            med_dx = np.median(motions[near, 0])
            spread_dx = np.median(np.abs(motions[near, 0] - med_dx))
            if abs(motions[i, 0] - med_dx) <= max(3.0 * spread_dx, 0.05):
                continue
            if weak:
                motions[i, 0] = med_dx
                n_dx_fixed += 1

    # Пробелы — линейная интерполяция между валидными соседями
    if valid.any():
        idx = np.arange(total - 1)
        for c in range(3):
            motions[~valid, c] = np.interp(idx[~valid], idx[valid], motions[valid, c])
    elif verbose:
        print('  ВНИМАНИЕ: ни одной валидной оценки движения')

    # Интегрирование в глобальные позы
    poses = np.zeros((total, 3))
    for i in range(1, total):
        poses[i] = _compose(poses[i - 1], motions[i - 1])

    arc = np.zeros(total)
    arc[1:] = np.cumsum(np.hypot(motions[:, 0], motions[:, 1]))

    if verbose:
        print(f'  готово за {time.time() - t0:.1f}s: '
              f'{valid.sum()}/{total - 1} прямых оценок '
              f'(снято выбросов поворота: {n_rejected}, '
              f'заменён снос: {n_dx_fixed}), '
              f'путь {arc[-1]:.1f} м, '
              f'курс {np.degrees(poses[-1, 2]):+.1f}°')

    return {'poses': poses, 'motions': motions, 'valid': valid, 'arc': arc}


def transform_to_global(pts, pose):
    """Переводит точки кадра в глобальную систему (кадр 0).

    Именно этого не хватало прежней карте: она сдвигала только Y, а X
    оставляла в локальной (повёрнутой) системе кадра.

    pts: (N, 2) или (N, 3). Z не меняется — движение плоское.
    """
    x, y, th = pose
    c, s = np.cos(th), np.sin(th)
    out = pts.copy().astype(float)
    lx, ly = pts[:, 0], pts[:, 1]
    out[:, 0] = x + c * lx - s * ly
    out[:, 1] = y + s * lx + c * ly
    return out


def transform_to_local(pts, pose):
    """Переводит точки из глобальной системы в систему кадра с данной позой."""
    x, y, th = pose
    c, s = np.cos(-th), np.sin(-th)
    out = pts.copy().astype(float)
    gx, gy = pts[:, 0] - x, pts[:, 1] - y
    out[:, 0] = c * gx - s * gy
    out[:, 1] = s * gx + c * gy
    return out


if __name__ == '__main__':
    import argparse
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from visualize_bag import BagFrames, find_db3

    parser = argparse.ArgumentParser(description='SE(2)-одометрия по лидару')
    parser.add_argument('bag_dir')
    parser.add_argument('--min-intensity', type=int, default=50)
    parser.add_argument('--save', action='store_true',
                        help='сохранить trajectory.npz в папку бэга')
    args = parser.parse_args()

    frames = BagFrames(find_db3(args.bag_dir), cache_size=8)
    traj = compute_trajectory(frames, min_intensity=args.min_intensity)

    poses = traj['poses']
    print(f'\nТраектория ({len(poses)} кадров):')
    print(f'{"frame":>6} {"arc,m":>9} {"x,m":>9} {"y,m":>9} {"heading,deg":>12}')
    step = max(1, len(poses) // 25)
    for i in range(0, len(poses), step):
        print(f'{i:6d} {traj["arc"][i]:9.1f} {poses[i, 0]:9.2f} '
              f'{poses[i, 1]:9.2f} {np.degrees(poses[i, 2]):12.2f}')

    if args.save:
        out = os.path.join(args.bag_dir, 'trajectory.npz')
        np.savez(out, **traj)
        print(f'\n-> {out}')
