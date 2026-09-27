"""Визуализация коридора для visualize_bag.py.

Строит Open3D LineSet по результату guard.walls.trace: две линии габарита
(то, что должно быть свободно) и две линии найденных стен. Цвет несёт
статус бина, потому что именно он решает, можно ли верить картинке:

    зелёный  — габарит подтверждён измерением стен в этом бине
    жёлтый   — бин ведётся предсказанием (дыра между кольцами лидара):
               это "не наблюдается", а НЕ "свободно"
    серый    — стены; показаны, чтобы видеть, за что зацепилось ведение

Обратно в систему кадра: fwd = -y, lat = x, up = z (см. guard/geom.py).
"""

import numpy as np

try:
    import open3d as o3d
except ImportError:      # визуализация опциональна
    o3d = None

from .walls import GAUGE_H_LO, GAUGE_H_HI

OK = (0.10, 0.75, 0.20)      # подтверждено
HOLE = (0.95, 0.75, 0.05)    # ведётся предсказанием
WALL = (0.55, 0.55, 0.60)    # стена
AXIS = (0.10, 0.45, 0.95)    # ось пути


def _polyline(pts_xyz, color, lines, points, colors):
    """Добавляет ломаную в накопители геометрии."""
    base = len(points)
    points.extend(pts_xyz)
    for k in range(len(pts_xyz) - 1):
        lines.append([base + k, base + k + 1])
        colors.append(color)


def corridor_lineset(wall, axis, show_walls=True, show_axis=True):
    """LineSet габарита и стен. None, если вести нечего."""
    if o3d is None:
        return None
    sel = np.array([s != 'stop' for s in wall['status']])
    if sel.sum() < 2:
        return None

    fwd = wall['fwd_mid'][sel]
    floor = wall['floor']
    z_lo, z_hi = floor + GAUGE_H_LO, floor + GAUGE_H_HI
    status = wall['status'][sel]

    points, lines, colors = [], [], []

    # Габарит: вертикальные стойки на границе свободного коридора + прогоны
    # по низу и верху. Рисуется посегментно, чтобы каждый бин нёс свой цвет.
    ax = wall['axis'][sel]
    for side, half in ((-1, wall['half_left'][sel]), (+1, wall['half_right'][sel])):
        x = ax + side * half
        for zc in (z_lo, z_hi):
            for k in range(len(fwd) - 1):
                c = OK if (status[k] == 'ok' and status[k + 1] == 'ok') else HOLE
                base = len(points)
                points.append([x[k], -fwd[k], zc])
                points.append([x[k + 1], -fwd[k + 1], zc])
                lines.append([base, base + 1])
                colors.append(c)
        # Стойки через бин: показывают высоту габарита, не зашумляя картинку.
        for k in range(0, len(fwd), 2):
            base = len(points)
            points.append([x[k], -fwd[k], z_lo])
            points.append([x[k], -fwd[k], z_hi])
            lines.append([base, base + 1])
            colors.append(OK if status[k] == 'ok' else HOLE)

    if show_walls:
        for key in ('left', 'right'):
            v = wall[key][sel]
            ok = ~np.isnan(v)
            if ok.sum() >= 2:
                # left/right хранятся ОТНОСИТЕЛЬНО ведомой оси бина
                # (walls.trace: вынос от guided-оси), поэтому в абсолютный X
                # их возвращает прибавление оси бина.
                _polyline([[ax[k] + v[k], -fwd[k], z_lo + 0.05]
                           for k in np.flatnonzero(ok)],
                          WALL, lines, points, colors)

    if show_axis:
        _polyline([[ax[k], -fwd[k], floor + 0.35] for k in range(len(fwd))],
                  AXIS, lines, points, colors)

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    ls.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
    ls.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
    return ls


# --- консольная строка ----------------------------------------------------

_BLOCK = '█'


def _nearest_in_gauge(wall):
    """Ближняя находка блока 3 в габарите или None.

    run.analyze кладёт в коридор ключ 'obstacles' = (hits, summary) из
    obstacles.detect; накопленный коридор (accum) этого ключа не несёт.
    """
    ob = wall.get('obstacles')
    if not ob:
        return None
    ing = [h for h in ob[0] if h.get('in_gauge')]
    return min(ing, key=lambda h: h['d']) if ing else None


def corridor_bar(wall, width=60, max_range=115.0):
    """Полоса статуса по дальности для консоли: видно, докуда верим.

    width — максимум символов в полосе: дальний хвост обрезается, чтобы
    строка на кадр помещалась в терминал. Красный '!' в конце — есть
    препятствие в габарите (блок 3).
    """
    n = len(wall['fwd_mid'])
    out = []
    for i in range(n):
        if wall['fwd_mid'][i] > max_range or len(out) >= width:
            break
        s = wall['status'][i]
        if s == 'ok':
            out.append('\033[32m' + _BLOCK + '\033[0m')
        elif s == 'hole':
            out.append('\033[33m' + _BLOCK + '\033[0m')
        else:
            out.append('\033[90m' + '·' + '\033[0m')
    if _nearest_in_gauge(wall) is not None:
        out.append('\033[31m!\033[0m')
    return ''.join(out)


def corridor_line(wall, axis):
    """Одна строка на кадр: дальность подтверждения + узкое место.

    Если блок 3 нашёл препятствие в габарите, в конец добавляется маркер
    с дальностью ближней находки.
    """
    hit = _nearest_in_gauge(wall)
    mark = f'   ! препятствие {hit["d"]:.0f} м' if hit is not None else ''
    ok = wall['status'] == 'ok'
    if not ok.any():
        return (f'подтверждено 0 м  [ось {"изм" if axis["measured"] else "НЕТ"}]'
                + mark)
    hl, hr = wall['half_left'][ok], wall['half_right'][ok]
    i = int(np.argmin(np.minimum(hl, hr)))
    narrow = min(hl[i], hr[i])
    return (f'подтверждено {wall["reach"]:5.0f} м   '
            f'коридор {hl.min():.2f}/{hr.min():.2f} м   '
            f'узко {narrow:.2f} м на {wall["fwd_mid"][ok][i]:.0f} м   '
            f'ось {"изм" if axis["measured"] else "НЕТ"}' + mark)


# --- вид сверху (matplotlib) ----------------------------------------------

def _panel(ax_, pts, wall, max_range, title):
    """Рисует один вид сверху на готовой оси matplotlib.

    Вынесено из plot_top, чтобы можно было положить рядом две версии одного
    кадра — без накопления и с ним, — и сравнивать их на общих пределах.
    """
    from .geom import to_frame
    from .walls import GAUGE_H_LO, GAUGE_H_HI

    fwd, lat, up = to_frame(pts)
    floor = wall['floor']
    h = up - floor
    # Только то, что попадает в высоту поезда: пол и свод на виде сверху
    # только мешают.
    m = (fwd > 0) & (fwd < max_range) & (h > GAUGE_H_LO) & (h < GAUGE_H_HI)

    ax_.scatter(fwd[m], lat[m], s=0.6, c='0.75', linewidths=0, label='точки')

    sel = np.array([s != 'stop' for s in wall['status']])
    f = wall['fwd_mid'][sel]
    axc = wall['axis'][sel]
    st = wall['status'][sel]
    ok = st == 'ok'

    ax_.plot(f, axc, color=AXIS, lw=1.6, label='ось пути')
    for key, lbl in (('left', 'стена'), ('right', None)):
        # left/right — вынос ОТ ведомой оси бина (walls.trace), в поперечную
        # координату картинки их возвращает прибавление оси бина.
        v = wall[key][sel]
        ax_.plot(f, axc + v, color=WALL, lw=1.0, ls='--', label=lbl)

    lo = axc - wall['half_left'][sel]
    hi = axc + wall['half_right'][sel]
    # Заливка посегментно: коридор непрерывен, но цвет несёт статус бина.
    # np.nan-маской заливать нельзя — она рвёт полосу на острова и картинка
    # врёт, будто между бинами коридора нет.
    shown = set()
    for k in range(len(f) - 1):
        col = OK if (ok[k] and ok[k + 1]) else HOLE
        lbl = None
        key = 'ok' if col is OK else 'hole'
        if key not in shown:
            lbl = 'габарит подтверждён' if col is OK else 'не наблюдается'
            shown.add(key)
        ax_.fill_between(f[k:k + 2], lo[k:k + 2], hi[k:k + 2],
                         color=col, alpha=0.30, lw=0, label=lbl)
        ax_.plot(f[k:k + 2], lo[k:k + 2], color=col, lw=1.4)
        ax_.plot(f[k:k + 2], hi[k:k + 2], color=col, lw=1.4)

    ax_.set_xlabel('дальность вперёд, м')
    ax_.set_ylabel('поперёк пути, м')
    ax_.set_title(title or 'Габаритный коридор (вид сверху)')
    ax_.set_xlim(0, max_range)
    ax_.set_ylim(-6, 6)
    # После set_ylim: иначе подпись встаёт по старым пределам и уезжает за оси.
    ax_.axvline(wall['reach'], color='k', ls=':', lw=1.2)
    ax_.annotate(f'подтверждено {wall["reach"]:.0f} м',
                 xy=(wall['reach'], 5.7), xytext=(-6, 0),
                 textcoords='offset points', ha='right', va='top', fontsize=9)
    ax_.grid(alpha=0.25)
    ax_.legend(loc='upper left', fontsize=8, ncol=3)


def plot_top(pts, wall, axis, path=None, max_range=115.0, title='',
             wall_accum=None):
    """Вид сверху: облако, ось пути, стены и габаритный коридор.

    Показывает то, чего не видно в 3D-просмотрщике: форму коридора целиком и
    как он ложится на реальные точки. path=None — показать окно.

    wall_accum — накопленный коридор того же кадра (guard.accum). Если он
    передан, рисуются ДВЕ панели на общих пределах: сверху этот кадр сам по
    себе, снизу он же с накоплением. Сравнивать имеет смысл только так:
    разница в стенах видна лишь на одном и том же кадре.

    axis — результат analyze()[0]; из него берётся только пометка «ось
    изм/НЕТ» в заголовок (как в corridor_line): без неё на картинке не видно,
    что ось взята нулевой и коридору верить нельзя.
    """
    import matplotlib
    if path:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    tag = f'ось {"изм" if axis["measured"] else "НЕТ"}'
    if wall_accum is None:
        fig, ax_ = plt.subplots(figsize=(13, 6))
        _panel(ax_, pts, wall, max_range,
               f'{title or "Габаритный коридор (вид сверху)"}   [{tag}]')
    else:
        fig, axs = plt.subplots(2, 1, figsize=(13, 11), sharex=True, sharey=True)
        base = f'{title or "Габаритный коридор (вид сверху)"}   [{tag}]'
        nb = int((wall['status'] == 'ok').sum())
        na = int((wall_accum['status'] == 'ok').sum())
        _panel(axs[0], pts, wall, max_range,
               f'{base} — БЕЗ накопления: стен {nb}, '
               f'подтверждено {wall["reach"]:.0f} м')
        _panel(axs[1], pts, wall_accum, max_range,
               f'{base} — с накоплением ({wall_accum["n_frames"]} кадров): '
               f'стен {na}, подтверждено {wall_accum["reach"]:.0f} м')
        # Нижняя панель несёт голоса: сколько кадров подтвердили каждый бин.
        # Без этого не видно, чем накопление отличается от одиночного кадра.
        v = wall_accum.get('votes')
        if v is not None:
            sel = np.array([s != 'stop' for s in wall_accum['status']])
            for fm, nv in zip(wall_accum['fwd_mid'][sel], v[sel]):
                axs[1].annotate(str(int(nv)), xy=(fm, -5.6), ha='center',
                                va='bottom', fontsize=6, color='0.35')
            axs[1].annotate('голосов:', xy=(0, -5.6), ha='left', va='bottom',
                            fontsize=6, color='0.35')
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130)
        plt.close(fig)
        return path
    plt.show()

