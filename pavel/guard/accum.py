"""Межкадровое накопление коридора по одометрии.

Зачем. Каждый кадр guard.walls.trace считает коридор с нуля, и подтверждённая
дальность скачет: на doubleT_platform соседние кадры дают 115 и 85 м на одном
и том же участке пути. Причина не в тоннеле, а в дырах между кольцами лидара —
бин, пустой в этом кадре, был измерен в предыдущем, когда поезд стоял на 1 м
позади. Эту информацию мы просто выбрасывали.

Что накапливается. НЕ точки, а уже посчитанные бины коридора. Копить точки
(как accumulate_with_speed) для габарита неверно: движущийся объект
размазывается в полосу вдоль пути, а он-то нам и нужен. Бин же несёт вывод
"стена здесь", и у неподвижной стены выводы разных кадров совпадают.

Как переносится. Поза кадра берётся из trajectory.npz (odometry.py, кэш есть
на всех шести бэгах). Бин, измеренный в кадре i на дальности d, в кадре j
лежит на d - (пройденное между i и j). Замер по doubleT_platform: за 4 кадра
боковой снос <= 0.04 м, поворот <= 0.06 deg — на порядок меньше бина 5 м,
поэтому на прямой достаточно сдвига вдоль пути. На кривых кадр со сносом
или поворотом за допуском (полбина, см. MAX_TURN) в голосовании не участвует.

Голосование медианой, а не средним: один сорвавшийся кадр не должен утянуть
границу. Бин считается подтверждённым, если за него отдали голос минимум
MIN_VOTES кадров — это и есть защита от случайного отклика.
"""

import numpy as np

from .walls import BIN, GAUGE_HALF, MAX_RANGE

# Сколько прошлых кадров участвует. 6 кадров при 1 м/кадр — это 6 м пути:
# столько, чтобы закрыть дыру между кольцами, но не столько, чтобы накопить
# ошибку одометрии или протащить устаревшую сцену.
HISTORY = 6

# Минимум голосов за бин. 1 голос — это тот же одиночный кадр, что и был;
# 2 голоса уже требуют, чтобы стену увидели с двух разных точек пути.
MIN_VOTES = 2

# Дальше этого назад бин не тащим: он ушёл под поезд и не проверяем.
MIN_KEEP = 0.0

# Отбраковка голосов кадров, съехавших вбок/по курсу (кривые). Пороги НЕ
# подгонка, а допуск самой системы отсчёта: боковой снос BIN/2 перекладывает
# голос в соседний бин (BIN уже в walls.py); поворот, дающий тот же снос
# BIN/2 на дальнем конце MAX_RANGE, — чистая геометрия тех же констант:
# arctan(2.5/115) = 0.0217 рад ~= 1.2 deg. Замер по кривым бэгов: 0.3-0.5 deg
# за 6 кадров (снос 0.3-0.9 м на 60-100 м) — под порогом, режутся только
# реально съехавшие кадры.
MAX_TURN = np.arctan2(BIN / 2, MAX_RANGE)


def _shift_along_path(poses, i_from, i_to):
    """Сколько метров ПРОШЁЛ поезд от кадра i_from до кадра i_to.

    Положительное значение = i_to впереди i_from, то есть бины старого кадра
    в системе нового лежат ближе на эту величину.

    Знак здесь критичен и однажды уже был перепутан: если вернуть положение
    старого кадра относительно нового (оно отрицательное), накопитель сдвинет
    бины НЕ в ту сторону — назад вместо вперёд, — и голоса лягут мимо своих
    бинов. Проверка: соседние кадры doubleT_platform дают около +1.0 м.

    Возвращает (пройдено_вперёд, снос_вбок, поворот) или None, если поз нет.
    """
    if poses is None or i_from >= len(poses) or i_to >= len(poses):
        return None
    x0, y0, t0 = poses[i_from]
    x1, y1, t1 = poses[i_to]
    # Разность в системе СТАРОГО кадра: куда уехал новый относительно него.
    c, s = np.cos(-t0), np.sin(-t0)
    gx, gy = x1 - x0, y1 - y0
    lx, ly = c * gx - s * gy, s * gx + c * gy
    # fwd = -y (см. guard/geom.py).
    return (-ly, lx, t1 - t0)


class CorridorAccumulator:
    """Копит бины коридора по кадрам и выдаёт сглаженный результат.

    Использование:
        acc = CorridorAccumulator(poses)
        wall = acc.update(frame_idx, wall)   # wall из guard.walls.trace
    """

    def __init__(self, poses=None, history=HISTORY, min_votes=MIN_VOTES):
        self.poses = poses
        self.history = history
        self.min_votes = min_votes
        self._past = []          # [(idx, fwd_mid, left, right, status)]
        self._span = None        # сдвиг БЛИЖАЙШЕГО голосующего прошлого кадра, м

    def _shift_span(self):
        """Ближняя зона, куда голоса прошлых кадров дотянуться не могли, м.

        Это сдвиг БЛИЖАЙШЕГО голосующего кадра истории, а не самого старого:
        его бины уехали назад на эту величину, и ближе начала его сетки
        голосовать некому. Раньше здесь стоял сдвиг старейшего кадра: при
        --sweep 10 это 50 м, и бины до 50 м проходили с одним голосом, а при
        --sweep 60 накопитель вырождался в одиночный кадр.
        """
        return self._span if self._span is not None else 0.0

    def update(self, idx, wall):
        """Добавляет кадр и возвращает накопленный коридор.

        Исходный wall не меняется — возвращается копия с полями:
          left/right/half_* — по медиане голосов
          votes   — сколько кадров подтвердили бин
          status  — 'ok' там, где голосов хватило
          reach   — дальность по накопленным подтверждённым бинам
          reach_frame — то же по одному текущему кадру (для сравнения)
          n_frames — сколько кадров сейчас в истории (для подписи на графике)
        """
        # Измеренный бин — это 'ok' ИЛИ 'gauge' (одна стена измерена, вторая
        # ведётся клампом габарита): walls.trace считает оба наблюдаемыми
        # (см. reach там же), иначе бины-'gauge' не голосуют вовсе и
        # накопленный reach падает НИЖЕ одиночного кадра (замер после смены
        # словаря статусов в walls: med 55 м против 75 м без накопления).
        ok_now = (wall['status'] == 'ok') | (wall['status'] == 'gauge')
        self._past.append((idx, wall['fwd_mid'].copy(),
                           wall['left'].copy(), wall['right'].copy(), ok_now))
        if len(self._past) > self.history:
            self._past.pop(0)

        n = len(wall['fwd_mid'])
        votes_l = [[] for _ in range(n)]
        votes_r = [[] for _ in range(n)]

        self._span = None
        n_voters = 0
        for p_idx, p_fwd, p_left, p_right, p_ok in self._past:
            if p_idx == idx:
                shift = 0.0
            else:
                mv = _shift_along_path(self.poses, p_idx, idx)
                if mv is None:
                    continue
                shift, lat_drift, turn = mv
                # Кадр, съехавший вбок или по курсу, голосует мимо своего
                # бина (кривые): такой голос отбраковываем, см. MAX_TURN.
                if abs(lat_drift) > BIN / 2 or abs(turn) > MAX_TURN:
                    continue
                if self._span is None or shift < self._span:
                    self._span = shift
            n_voters += 1
            # Бин из прошлого кадра в системе текущего лежит ближе на shift.
            for k in np.flatnonzero(p_ok):
                d = p_fwd[k] - shift
                if d < MIN_KEEP or d > MAX_RANGE:
                    continue
                # Не floor, а ближайший центр бина: при дробном сдвиге голос
                # бина k не дребезжит между k и k-1 на границе. Свои бины
                # (shift=0) попадают точно в себя: центр бина k лежит на
                # k*BIN + BIN/2, и (k*BIN + BIN/2)/BIN - 0.5 = k ровно.
                j = int(np.round(d / BIN - 0.5))
                if 0 <= j < n:
                    votes_l[j].append(p_left[k])
                    votes_r[j].append(p_right[k])

        out = dict(wall)
        out['left'] = np.full(n, np.nan)
        out['right'] = np.full(n, np.nan)
        out['votes'] = np.zeros(n, dtype=int)
        status = np.array(['stop'] * n, dtype=object)

        # Пока кадров меньше, чем нужно голосов, требовать их нельзя: иначе
        # на старте (и после разрыва одометрии) накопитель отдаёт reach = 0
        # там, где одиночный кадр честно видел 95 м. Считаем только реально
        # голосующие кадры: кадры без позы пропущены выше и голосов не дают.
        need = min(self.min_votes, n_voters)

        for j in range(n):
            v = len(votes_l[j])
            out['votes'][j] = v
            if v == 0:
                continue
            # Медиана, а не среднее: один сорвавшийся кадр не тянет границу.
            out['left'][j] = float(np.median(votes_l[j]))
            out['right'][j] = float(np.median(votes_r[j]))
            # Ближние бины физически не могут набрать голосов: прошлые кадры
            # стояли ПОЗАДИ, и их бины сюда не дотягиваются. Замер: бин 0-5 м
            # всегда получал ровно 1 голос, из-за чего непрерывный reach
            # обрывался на нуле в 35 кадрах из 69. Зона прямо под поездом
            # измерена надёжнее всего — здесь хватает голоса текущего кадра.
            near = out['fwd_mid'][j] <= self._shift_span()
            status[j] = 'ok' if (v >= need or (near and ok_now[j])) else 'hole'

        # reach — НЕПРЕРЫВНАЯ подтверждённая дальность, а не последний
        # зелёный бин. Иначе подтверждённый бин за дырой поднимает reach, и
        # выдача заявляет знание о том, что лежит за неизвестным участком.
        reach = 0.0
        for j in range(n):
            if status[j] != 'ok':
                break
            reach = wall['fwd_mid'][j] + BIN / 2

        # Хвост после последнего подтверждённого бина — не "свободно",
        # а "не наблюдается": там ведение остановлено.
        out['status'] = status
        out['reach'] = reach
        out['reach_frame'] = wall['reach']
        out['n_ok'] = int((status == 'ok').sum())
        out['n_frames'] = len(self._past)
        out['half_left'] = np.minimum(np.abs(out['left']), GAUGE_HALF)
        out['half_right'] = np.minimum(out['right'], GAUGE_HALF)
        return out


def load_poses(bag_dir):
    """Позы из trajectory.npz, если он посчитан. Иначе None."""
    import os
    p = os.path.join(bag_dir, 'trajectory.npz')
    if not os.path.exists(p):
        return None
    try:
        return np.load(p)['poses']
    except (KeyError, ValueError, OSError):
        return None
