"""Блок 3.5. Межкадровый трекинг препятствий: отсечение однокадровых
транзиентов без потери настоящих находок.

Зачем. Блок 3 (obstacles.py) покадров: однокадровый транзиент (пыль,
случайное сгущение остатка на дальности) неотличим от настоящего дальнего
объекта. Известный подтверждённый ложный: roundT_squareT_pressureGate_squareT
f280 — 95 м, 19 точек, ровно 1 кадр (замер _tmp_trk_meas3: на кадрах 250-310
это единственная находка в габарите, ни кадра рядом). Настоящий объект живёт
десятки кадров: человек A в doubleT_obstacle виден 201 кадр из 201
(замер _tmp_trk_meas1).

Как. Каждая находка переводится в мировые координаты полным SE(2) по позам
trajectory.npz (конвенция та же, что в accum._shift_along_path и
run._PointAggregator: p_world = R(theta) @ p_local + t, локальная система
лидара x=lat, y=-fwd; проверка знака поворота — _tmp_trk_meas2: на кривой
cloud_with_fake_obj верный знак собирает статичный вставленный объект в один
компактный мировой кластер, неверный размазывает). Связывание — ближайший
трек в пределах допуска; трек с числом кадров >= CONFIRM_FRAMES помечается
confirmed. Находки НЕ удаляются: однокадровая находка остаётся в выдаче
блока 3, трекер лишь добавляет ей уровень подтверждения (сторона ошибки:
лучше показать неподтверждённое, чем скрыть настоящее).

Допуск связывания — анизотропный, в системе ТЕКУЩЕГО кадра:

* поперёк пути: GATE_LAT0 + WALK_STEP * пропущено_кадров. WALK_STEP из
  физики: пешеход <= 2 м/с при 10 Гц = 0.2 м/кадр (эталон: человек A ходит
  0.1-0.15 м/кадр, замер _tmp_trk_meas1 med 0.074 м/кадр поперёк). GATE_LAT0
  — ошибка переноса координат и шум центроида находки: одометрия даёт
  сантиметры на кадр (замер accum: за 4 кадра снос <= 0.04 м, поворот <=
  0.06 deg; на 100 м дальности поворот 0.06 deg — это 0.10 м), шум медианы
  lat у человека на 55 м — сантиметры (плавный ход 0.07-0.15 м/кадр без
  дрожания). Запас до 0.3 м покрывает и кривые, и дальние находки.
* вдоль пути: GATE_FWD0 + WALK_STEP * пропущено_кадров, где GATE_FWD0 =
  OBST_WIN + GATE_LAT0. Поле d находки — ЦЕНТР окна поиска (окно OBST_WIN =
  5 м, шаг OBST_STEP = 2.5 м), а не центроид кластера: истинное положение
  объекта в пределах полуокна, поэтому у стоящего объекта сообщённый центр
  между кадрами может прыгнуть на целое окно (замер: у человека на границе
  окон f120->f130 doubleT_obstacle d: 57.5 -> 52.5 м, поперёк тот же 0.1 м —
  с базой OBST_WIN/2 трек рвался).

Подтверждение CONFIRM_FRAMES = 2: транзиент живёт 1 кадр (замер f280 выше;
человек A — каждый кадр), поэтому второй кадр уже разделяет классы. Платим
за это один кадр задержки подтверждения настоящего препятствия — приёмлемо:
находка первого кадра всё равно показывается, но как неподтверждённая.

Смерть трека — MAX_MISS обработанных кадров без находки. Объект может
выпадать на дыру между кольцами лидара (бин 'hole'/'uncovered' в walls) или
за край обработанной дальности. Замер серий пропусков (_tmp_trk_meas1 и
_tmp_trk_meas2): человек A на стоящем поезде — 0 пропусков за 201 кадр;
вставленный объект на едущем поезде (cloud_with_fake_obj, sweep 10) — см.
комментарий у MAX_MISS.

Без trajectory.npz мировой перенос невозможен — трекер работает в системе
датчика (эквивалентно стоящему поезду) и честно предупреждает поле
world_ok=False: на стоящем бэге (doubleT_obstacle, снос 0.013 м/кадр) это
корректно, на едущем трекинг деградирует до покадрового.
"""

import numpy as np

from .obstacles import OBST_STEP, OBST_WIN

# --- допуск связывания (вывод — в докстринге модуля) -------------------------
WALK_STEP = 0.2          # м/кадр: пешеход <= 2 м/с при 10 Гц
GATE_LAT0 = 0.3          # м поперёк: одометрия (см/кадр) + поворот на
                         # дальности (0.10 м на 100 м) + шум центроида (см)
GATE_FWD0 = OBST_WIN + GATE_LAT0      # м вдоль: d — центр окна 5 м с шагом
                         # 2.5 м; окно шире шага, поэтому у объекта на границе
                         # окон сообщённый центр может прыгнуть на OBST_WIN
                         # за кадр (замер: f120->f130 doubleT_obstacle,
                         # 57.5 -> 52.5 при неподвижном человеке — рвало трек)

# --- подтверждение и смерть ---------------------------------------------------
CONFIRM_FRAMES = 2       # кадров с находкой для confirmed: транзиент живёт
                         # ровно 1 кадр (замер roundT_sqT f250-310), настоящий
                         # объект — каждый кадр; второй кадр разделяет классы
MAX_MISS = 3             # обработанных кадров без находки до смерти трека.
                         # Замер серий пропусков (_tmp_trk_meas1/2): человек A
                         # — 0 пропусков за 201 кадр; вставленный объект
                         # (cloud_with_fake_obj, sweep 10) в зоне уверенной
                         # видимости — максимум 1 обработанный кадр подряд
                         # (дыра кольца). 3 — запас над дырой. Длинные провалы
                         # на 90-110 м (до 4 кадров подряд, f10->f80) — не
                         # дыры, а зона предельной видимости (объект едва
                         # набирает OBST_MIN_PTS); её трекер не мостует —
                         # сторона ошибки: не протаскивать устаревшее.

# Двойные находки одной цели в ОДНОМ кадре. Окна блока 3 перекрываются
# (OBST_WIN 5 м с шагом OBST_STEP 2.5 м), и широкая цель может дать две
# находки, которые _merge_hits не слил (у него допуск MERGE_DX=0.6 поперёк).
# Замер (cloud_with_fake_obj f130): пара d=70.0/72.5 (= OBST_STEP вдоль),
# Δlat=0.75, обе n=52 — края одной цели в соседних окнах. Без дедупликации
# вторая находка порождала трек-паразит, который на следующих кадрах крал
# связывания у основного (объект распадался на 2 подтверждённых трека по
# 6-8 кадров вместо одного на 11). Допуск поперёк — ширина цели (у человека
# до 0.8 м, замеры obstacles.py) плюс запас. Дедупликация НЕ удаляет находку
# из выдачи блока 3 — только не даёт ей создать второй трек. Побочный случай
# (новый объект, впервые виденный вплотную к существующему треку, стартует
# кадром позже) принят: два объекта ближе 1 м поперёк и 2.5 м вдоль одновре-
# менно в габарите — вне замеренной реальности.
DUP_LAT = 1.0            # м поперёк
DUP_FWD = OBST_STEP      # м вдоль: перекрытие окон блока 3 ровно OBST_STEP


class ObstacleTracker:
    """Связывает находки блока 3 между кадрами в мировых координатах.

    Использование:
        trk = ObstacleTracker(poses)      # poses из accum.load_poses или None
        tracks = trk.update(frame_idx, hits)   # hits — из obstacles.detect
    update возвращает список ЖИВЫХ треков (словарей) после этого кадра;
    полный список (с умершими) — self.tracks.
    """

    def __init__(self, poses=None):
        self.poses = poses
        self.world_ok = poses is not None
        self.tracks = []          # все треки, живые и умершие
        self._next_id = 1
        self.frames_confirmed = 0  # кадров, где была находка подтверждённого
                                   # трека

    def _pose(self, i):
        if self.poses is None or i >= len(self.poses):
            return 0.0, 0.0, 0.0
        x, y, t = (float(v) for v in self.poses[i])
        return x, y, t

    def _to_world(self, hit, i):
        """Находка кадра i -> мир: p = R(theta) (lat_abs, -d) + t."""
        x, y, t = self._pose(i)
        c, s = np.cos(t), np.sin(t)
        lx, ly = hit['lat_abs'], -hit['d']
        return np.array([c * lx - s * ly + x, s * lx + c * ly + y])

    def _to_frame(self, w, i):
        """Мир -> система кадра i: (fwd, lat)."""
        x, y, t = self._pose(i)
        c, s = np.cos(-t), np.sin(-t)
        gx, gy = w[0] - x, w[1] - y
        lx, ly = c * gx - s * gy, s * gx + c * gy
        return -ly, lx                     # fwd=-y, lat=x

    def update(self, frame_idx, hits):
        """Добавить находки кадра; вернуть живые треки.

        hits — находки obstacles.detect (словари с 'd', 'lat_abs'). Список не
        меняется; трекеру нужны только положение и флаги для сводки.
        """
        # Двойники одной цели из перекрывающихся окон блока 3 (см. DUP_LAT):
        # оставляем находку с большим числом точек. Это не фильтрация выдачи
        # блока 3 — только защита от трека-паразита.
        if len(hits) > 1:
            ded = []
            for h in sorted(hits, key=lambda a: -a.get('n_pts', 0)):
                if any(abs(h['d'] - k['d']) <= DUP_FWD
                       and abs(h['lat_abs'] - k['lat_abs']) <= DUP_LAT
                       for k in ded):
                    continue
                ded.append(h)
            hits = ded
        # Связывание жадное по нормированному расстоянию: находок и треков
        # в кадре единицы, венгерский алгоритм избыточен.
        cand = []   # (d_norm, трек, находка, мир)
        for tr in self.tracks:
            if not tr['alive']:
                continue
            gap = frame_idx - tr['last']
            gl = GATE_LAT0 + WALK_STEP * gap
            gf = GATE_FWD0 + WALK_STEP * gap
            f_t, l_t = self._to_frame(tr['world'], frame_idx)
            for h in hits:
                dn = np.hypot((h['d'] - f_t) / gf, (h['lat_abs'] - l_t) / gl)
                if dn <= 1.0:
                    cand.append((dn, tr, h))
        cand.sort(key=lambda c: c[0])
        used_tr, used_h = set(), set()
        for dn, tr, h in cand:
            if id(tr) in used_tr or id(h) in used_h:
                continue
            used_tr.add(id(tr)); used_h.add(id(h))
            tr['n'] += 1
            tr['last'] = frame_idx
            tr['miss'] = 0
            tr['world'] = self._to_world(h, frame_idx)
            tr['d'] = h['d']
            tr['lat_abs'] = h['lat_abs']
            tr['confirmed'] = tr['n'] >= CONFIRM_FRAMES
            tr['hits'].append((frame_idx, h))
        for h in hits:
            if id(h) in used_h:
                continue
            self.tracks.append(dict(
                id=self._next_id, n=1, first=frame_idx, last=frame_idx,
                miss=0, world=self._to_world(h, frame_idx), d=h['d'],
                lat_abs=h['lat_abs'], confirmed=False, alive=True,
                hits=[(frame_idx, h)]))
            self._next_id += 1
        # Смерть по пропускам (в обработанных кадрах, не в кадрах бэга:
        # трекер видит только то, что ему скармливают).
        for tr in self.tracks:
            if not tr['alive'] or tr['last'] == frame_idx:
                continue
            tr['miss'] += 1
            if tr['miss'] > MAX_MISS:
                tr['alive'] = False
        if any(tr['alive'] and tr['confirmed'] for tr in self.tracks):
            self.frames_confirmed += 1
        return [tr for tr in self.tracks if tr['alive']]

    def summary(self):
        """Сводка для печати: счётчики и самый длинный трек."""
        conf = [t for t in self.tracks if t['confirmed']]
        longest = max(self.tracks, key=lambda t: t['n'], default=None)
        return dict(tracks=len(self.tracks), confirmed=len(conf),
                    frames_confirmed=self.frames_confirmed, longest=longest,
                    world_ok=self.world_ok)
