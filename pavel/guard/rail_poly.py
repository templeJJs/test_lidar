"""Блок 1п: ось пути как ЛОМАНАЯ по двум нитям рельсов.

Зачем отдельно от `track.py`. Блок 1 сознательно строит ПРЯМУЮ (lat = lat0 +
slope*fwd) и держит наклон в SLOPE_LIM = 0.004 — на прямом участке это верно,
а на кривой заглушка-прямая промахивается втрое больше полугабарита и
обесценивает всё, что от неё считается. Рельсы при этом извиваются, и в
проекте уже есть детекция, которая ведёт каждую нить ПОЛИЛИНИЕЙ — узлами через
0.5 м, каждый следующий в окне от предсказания по предыдущим
(`rail_geometry` над `test/track_geometry.py`, флаг `--polyline` в
visualize_bag.py). Этот модуль берёт её результат и строит ось как ломаную по
СЕРЕДИНЕ между нитями.

Ничего в блоке 1 не меняется: `straight_axis` остаётся путём по умолчанию,
ломаная включается флагом (`run.analyze(polyline=True)`), чтобы две версии
можно было сравнивать на одних кадрах.

Система координат. `rail_geometry` работает в осях ЛИДАРА (x вправо, y назад),
guard — в тройке `fwd = -y`, `lat = x` (см. geom.py). Перевод один: fwd = -y,
lat = x, и дальше по пакету идут только fwd/lat.

Контракт выдачи — тот же dict, что у `straight_axis` (gauge, n_obs, obs_fwd,
obs_lat, measured), поэтому потребители блока 1 работают без правок. Сверх
него:

  knots_f / knots_l — узлы САМОЙ оси (ломаная), по ним считает `axis_at`;
  rails             — обе нити как (fwd, lat), для картинки;
  lat0 / slope      — линейная аппроксимация БЛИЖНЕЙ зоны (та же NEAR_FWD, что
                      у блока 1). Нужна только для обратной совместимости тех
                      мест, что читают lat0/slope напрямую (multitrack,
                      wall_rules, run.py при печати); ось при наличии узлов
                      считается по узлам, а не по этой прямой.
"""

import numpy as np

from .track import (GAUGE_NOM, NEAR_FWD, OWN_LAT, SLOPE_LIM,  # noqa: F401
                    straight_axis)

# Ось строится только там, где ИЗМЕРЕНЫ обе нити: середина между нитями — это
# и есть ось, а одна нить без второй положения оси не задаёт (отступ полколеи
# в какую сторону — неизвестно).
MIN_KNOTS = 4            # меньше узлов — не ломаная, а пара точек
_ENABLED = False         # rail_geometry.enable() ставится один раз за процесс
# Отсечка по дальности узлов не ставится: полилиния сама останавливается там,
# где кончаются точки (стена, разрежение), и дотянутые узлы `rail_geometry` в
# pairs не отдаёт (_MEASURED_ORIGINS там же).


def _fail(reason):
    """Отказ: контракт straight_axis с measured=False и пустыми узлами."""
    return dict(lat0=0.0, slope=0.0, gauge=GAUGE_NOM, n_obs=0,
                obs_fwd=np.zeros(0), obs_lat=np.zeros(0), measured=False,
                knots_f=np.zeros(0), knots_l=np.zeros(0), rails=None,
                poly=False, poly_reason=reason)


def poly_axis(pts, fwd, lat, up, floor, bag_key=None, frame_index=None):
    """Ось пути ломаной по двум нитям полилинии.

    pts — облако в осях ЛИДАРА (то, что пришло в analyze): полилинейная
    детекция принимает именно его, а не готовую тройку fwd/lat/up.
    fwd/lat/up/floor — та же тройка, что у блока 1; нужна для отката на
    прямую ось, когда нитей в кадре нет.

    bag_key/frame_index — ключи тёплой сборки `rail_geometry` (затравка от
    предыдущего кадра). Без них каждый кадр считается холодным — работает, но
    хуже; plot_axis их передаёт.
    """
    try:
        import rail_geometry
    except ImportError:
        return _fail('нет rail_geometry')

    # Тот же вход, что у `visualize_bag.py --polyline`: он зовёт enable()
    # перед первым кадром. Сама find_rail_lines ниже — уже полилинейная и без
    # enable, так что для guard это ничего не меняет; вызов стоит, чтобы тракт
    # был идентичен, если кто-то из общих модулей (rail_detection) окажется
    # втянут по пути. Ставится один раз за процесс, не на кадр.
    global _ENABLED
    if not _ENABLED:
        try:
            rail_geometry.enable(also_visualize=False)
        except Exception:                       # не критично: детекция и так та
            pass
        _ENABLED = True

    try:
        _lines, _ground, pairs = rail_geometry.find_rail_lines(
            pts, return_pairs=True, bag_key=bag_key, frame_index=frame_index)
    except Exception as e:                      # детекция не должна ронять кадр
        return _fail(f'детекция отказала: {type(e).__name__}')

    pairs = np.asarray(pairs, dtype=float)
    if len(pairs) < MIN_KNOTS:
        return _fail(f'узлов {len(pairs)} < {MIN_KNOTS}')

    # pairs: (y, x_left, x_right, z_left, z_right) в осях лидара. fwd = -y.
    f = -pairs[:, 0]
    l_left, l_right = pairs[:, 1], pairs[:, 2]
    keep = f > 0                                # позади поезда оси нет
    if keep.sum() < MIN_KNOTS:
        return _fail('узлов впереди мало')
    f, l_left, l_right = f[keep], l_left[keep], l_right[keep]

    o = np.argsort(f)
    f, l_left, l_right = f[o], l_left[o], l_right[o]

    centre = (l_left + l_right) / 2.0
    gauge = np.abs(l_right - l_left)

    # Своя ось от лидара не уезжает — поезд на ней стоит. Проверка по БЛИЖНЕЙ
    # зоне (дальше путь имеет право уходить вбок, в этом весь смысл ломаной),
    # тот же порог OWN_LAT, что в блоке 1.
    near = f <= NEAR_FWD[1]
    if near.sum() >= 2 and abs(float(np.median(centre[near]))) > OWN_LAT:
        return _fail('центр нитей дальше OWN_LAT — чужой путь')

    # lat0/slope — линейная аппроксимация ближней зоны, только для обратной
    # совместимости. Клампа SLOPE_LIM здесь НЕТ: он относится к прямой оси как
    # к модели пути, а здесь прямая — не модель, а грубая сводка ближней зоны.
    nf = f[near] if near.sum() >= 2 else f
    nl = centre[near] if near.sum() >= 2 else centre
    slope, lat0 = np.polyfit(nf, nl, 1) if len(nf) >= 2 else (0.0, float(nl[0]))

    rails = ((f, l_left), (f, l_right))
    return dict(lat0=float(lat0), slope=float(slope),
                gauge=float(np.median(gauge)), n_obs=int(len(f)),
                obs_fwd=f, obs_lat=centre, measured=True,
                knots_f=f, knots_l=centre, rails=rails,
                poly=True, poly_reason='')


def axis_or_straight(pts, fwd, lat, up, floor, bag_key=None, frame_index=None):
    """Ломаная ось, а где её нет — прямая блока 1 (бит в бит как раньше)."""
    ax = poly_axis(pts, fwd, lat, up, floor, bag_key=bag_key,
                   frame_index=frame_index)
    if ax['measured']:
        return ax
    st = straight_axis(fwd, lat, up, floor)
    st['poly'] = False
    st['poly_reason'] = ax['poly_reason']
    st['rails'] = None
    return st
