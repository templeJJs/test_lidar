"""Единый контракт формул клиент/сервер: эталонные векторы для обеих сторон.

Проблема: формулы позы, траектории и хода записи написаны ДВАЖДЫ -- на сервере
(`labeling.py`) и на клиенте (`web/labellayer.js`, `web/app.js`). Численная сверка
была «сдерживанием тестами» (.scratch/pose_fixture.json собирался вручную):
при правке одной стороны вторая расходилась молча.

Здесь -- единый источник: каждый вектор -- это ВХОД + ВЫХОД, посчитанный вызовом
СЕРВЕРНОЙ функции (источник истины). Файл `contracts/formula_vectors.json`
генерируется отсюда (`python formula_contract.py`), а проверяют его обе стороны:

* `tests/test_formula_contract.py` -- JSON на диске обязан совпадать с
  регенерацией из текущего `labeling.py` байт-в-байт: правка серверной формулы
  без пересчёта векторов роняет батарею сразу, а не в рассинхроне с клиентом;
* `.scratch/formula_contract_test.mjs` -- клиентские функции (настоящий код
  `web/labellayer.js` импортом, функции `web/app.js` вырезкой по имени, как в
  `.scratch/motion_client_test.mjs`) гоняются по тем же входам и обязаны дать
  те же выходы в пределах допусков (`tolerance` в файле).

Допуски: позиции и сдвиги, считаемые в float64 на обеих сторонах, -- 1e-6 м
(точность замеров хода в проекте -- 1e-3 м, так что это с запасом на три
порядка); вершины позы после постановки -- 1e-5 м, потому что клиент хранит
их в Float32Array (буфер GPU): eps float32 на координатах ~20 м -- уже 2e-6 м.
Индексы моментов и имена поз -- точно.

Покрытые пары (сервер <- клиент):
    pose_moment        <- poseMoment            (момент клипа по времени)
    pose_vertices      <- poseVerticesAt        (склейка моментов)
    _rotation          <- poseRotation          (R = Rx(roll)Ry(pitch)Rz(yaw))
    place_pose         <- placePose             (постановка меша на опору)
    pose_rule          <- poseInScene           (поза в сцене: правило + валидация)
    trajectory_at      <- trajectoryAt          (точка траектории по времени)
    trajectory_normalize <- trajectoryDistance/trajectorySpeed (длина и скорость)
    motion_at          <- labelMotionAt         (ход на кадре из профиля)
    motion_between     <- labelMotionBetween    (интеграл профиля между кадрами)
    _instance_center   <- trajectoryBaseAtFrame + labelBaseAt/labelCenterAt
    (save, t_f)        <- labelPoseTime         (время позы от опорного кадра)

Входы -- синтетические и детерминированные (никаких записей и мешей: формулам
нужны только числа), поэтому файл пересчитывается за миллисекунды и не зависит
от содержимого `assets/` и `for_hackathon/`.
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

import labeling

ROOT = labeling.root_dir()
CONTRACT_PATH = os.path.join(ROOT, 'contracts', 'formula_vectors.json')
CONTRACT_VERSION = 1
HZ = 10.0                       # частота кадров в векторах (labeling.HZ_DEFAULT)

# Допуски сверки (см. docstring модуля).
TOLERANCE = {
    'position_m': 1e-6,         # float64-формулы: центры, траектория, ход
    'float32_m': 1e-5,          # вершины позы: клиент хранит их во Float32Array
    'blend': 1e-9,              # доля склейки моментов
    'time_s': 1e-9,
    'matrix': 1e-12,            # матрица поворота (чистый float64 обеих сторон)
}

# Синтетический клип позы для векторов pose_moment/pose_vertices: формулам нужны
# только число моментов, времена, период и режим -- настоящие меши не нужны.
_CLIP_SPECS = [
    # (play, times, period)
    ('loop', [0.0, 0.112, 0.224, 0.336, 0.448, 0.56], 0.672),
    ('one_shot', [0.0, 0.3, 0.6, 0.9, 1.2, 1.5], 1.5),
    ('static', [0.0], 0.0),
]
# Моменты времени для опроса: внутри цикла, границы, за концом, отрицательные.
_POSE_TS = [0.0, 0.05, 0.335, 0.671, 0.672, 0.7, -0.1, 2.7, 1.5, -0.5]

# Вершины синтетического клипа склейки (T=4 момента, N=3 вершины): значения
# разнесены, чтобы склейка соседних моментов была видна числом.
_BLEND_VERTS = [
    [[0.00, 0.00, 0.00], [0.40, -0.10, 1.20], [-0.30, 0.20, 0.60]],
    [[0.05, 0.02, 0.01], [0.44, -0.06, 1.24], [-0.26, 0.24, 0.62]],
    [[0.10, 0.05, 0.03], [0.50, 0.00, 1.30], [-0.20, 0.30, 0.70]],
    [[0.16, 0.09, 0.06], [0.58, 0.08, 1.38], [-0.12, 0.38, 0.79]],
]
_BLEND_TIMES = [0.0, 0.4, 0.8, 1.2]


def _fake_clip(times, period, play, verts=None):
    """Клип в формате `labeling.pose_clip`: только то, что читают формулы."""
    t = np.asarray(times, dtype=np.float64)
    if verts is None:
        V = np.zeros((t.size, 1, 3), dtype=np.float32)
    else:
        V = np.asarray(verts, dtype=np.float32).reshape(t.size, -1, 3)
    return {'V': V, 'times': t, 'period': float(period), 'play': str(play)}


def _f3(v):
    return [float(v[0]), float(v[1]), float(v[2])]


# ------------------------------------------------------------- pose_moment ---

def _pose_moment_cases():
    cases = []
    for play, times, period in _CLIP_SPECS:
        clip = _fake_clip(times, period, play)
        for t in _POSE_TS:
            i, w, t_in = labeling.pose_moment(clip, t)
            cases.append({
                'times': [float(v) for v in times], 'period_s': float(period),
                'play': play, 't': float(t),
                'expect': {'index': int(i), 'blend': float(w),
                           'time_in_clip': float(t_in)},
            })
    return cases


# ----------------------------------------------------------- pose_vertices ---

def _pose_vertices_cases():
    cases = []
    for play, period in (('loop', 1.6), ('one_shot', 1.2)):
        clip = _fake_clip(_BLEND_TIMES, period, play, verts=_BLEND_VERTS)
        for t in (0.0, 0.13, 0.4, 0.55, 1.19, 1.2, 1.6, 1.77, -0.3):
            verts, i, w, t_in = labeling.pose_vertices(clip, t)
            cases.append({
                'vertices': [[float(x) for x in row] for moment in _BLEND_VERTS
                             for row in moment],
                'times': list(_BLEND_TIMES), 'period_s': float(period),
                'play': play, 't': float(t),
                'expect': {'index': int(i), 'blend': float(w),
                           'time_in_clip': float(t_in),
                           'vertices': [float(v) for v in
                                        np.asarray(verts, dtype=np.float32).reshape(-1)]},
            })
    return cases


# --------------------------------------------------------------- _rotation ---

_ROTATION_ARGS = [
    (0.0, 0.0, 0.0), (90.0, 0.0, 0.0), (0.0, 90.0, 0.0), (0.0, 0.0, 90.0),
    (33.0, -12.0, 7.0), (-45.0, 30.0, -60.0), (180.0, 5.0, -3.0),
    (-170.0, -80.0, 15.0),
]


def _rotation_cases():
    cases = []
    for yaw, pitch, roll in _ROTATION_ARGS:
        m = labeling._rotation(yaw, pitch, roll)
        cases.append({'yaw': yaw, 'pitch': pitch, 'roll': roll,
                      'expect': [float(v) for v in np.asarray(m).reshape(-1)]})
    return cases


# -------------------------------------------------------------- place_pose ---

_PLACE_VERTS = [
    [[0.00, 0.00, 0.00], [0.42, -0.12, 1.71], [-0.35, 0.22, 0.85],
     [0.10, -0.40, 0.30], [-0.15, 0.35, 1.55]],
]
_PLACE_ARGS = [
    # (center_xy, base_z, yaw, pitch, roll)
    ((1.5, -20.0), -1.2, 0.0, 0.0, 0.0),
    ((-0.8, -35.5), -2.1, 33.0, 0.0, 0.0),
    ((0.0, -8.0), -0.9, -45.0, 12.0, -7.0),
    ((2.2, -55.7), -1.9, 170.0, -5.0, 3.0),
]


def _place_pose_cases():
    cases = []
    for verts in _PLACE_VERTS:
        arr = np.asarray(verts, dtype=np.float32)
        for (cx, cy), base_z, yaw, pitch, roll in _PLACE_ARGS:
            placed, lift, anchor = labeling.place_pose(
                arr, [cx, cy, 0.0], base_z, yaw, pitch, roll)
            cases.append({
                'vertices': [[float(v) for v in row] for row in verts],
                'center': [float(cx), float(cy)], 'base_z': float(base_z),
                'yaw': float(yaw), 'pitch': float(pitch), 'roll': float(roll),
                'expect': {
                    'positions': [float(v) for v in
                                  np.asarray(placed, dtype=np.float64).reshape(-1)],
                    'lift': float(lift),
                    'anchor': _f3(anchor),
                },
            })
    return cases


# --------------------------------------------------------------- pose_rule ---

def _pose_rule_cases():
    poses = ['standing', 'walking', 'running', 'lying', 'fallen', 'unknown',
             'crouch', '', None, 'WALKING', 'Standing', 'lying-down']
    cases = []
    for pose in poses:
        for has_traj in (False, True):
            cases.append({
                'pose': pose, 'has_trajectory': bool(has_traj),
                'expect': labeling.pose_rule('person', pose, has_traj),
            })
    return cases


# ----------------------------------------------------------- trajectory_at ---

_TRAJS = [
    {'start': [0.5, -10.0, -1.9], 'end': [1.5, -4.0, -1.6], 'seconds': 3.0},
    {'start': [-2.0, -25.0, -0.8], 'end': [-2.0, -25.0, -0.8], 'seconds': 5.0},
    {'start': [1.0, -5.0, -1.0], 'end': [3.0, -15.0, -1.4], 'seconds': 0.0},
    {'start': [0.0, 0.0, 0.0], 'end': [1.0, -2.0, 0.5], 'seconds': 1.7},
    {'start': [-1.25, -40.0, -2.05], 'end': [0.75, -30.0, -1.55], 'seconds': 12.4},
]


def _trajectory_at_cases():
    cases = []
    for traj in _TRAJS:
        for t in (-1.0, 0.0, 0.5, 1.5, 3.0, 4.5):
            got = labeling.trajectory_at(traj, t)
            cases.append({'traj': traj, 't': float(t), 'expect': _f3(got)})
    return cases


def _trajectory_normalize_cases():
    """Длина и скорость траектории (trajectoryDistance/trajectorySpeed в клиенте)."""
    cases = []
    for traj in _TRAJS:
        norm = labeling.trajectory_normalize(traj)
        if norm is None:
            continue
        cases.append({
            'traj': traj,
            'expect': {'distance_m': float(norm['distance_m']),
                       'speed_mps': float(norm['speed_mps']),
                       'speed_kmh': float(norm['speed_kmh'])},
        })
    return cases


# ------------------------------------------------------------------ motion ---

# Профили хода для векторов: с дырами (None -- кадр недостоверен), с шагом 2,
# и вовсе без профиля (старое одно число). None в JSON -- null.
_MOTIONS = {
    'holes': {'profile': [0.5, None, 0.7, 0.2, None, 0.9],
              'm_per_frame': 0.55, 'profile_step': 1},
    'step2': {'profile': [0.4, None, 0.8, None, 0.6],
              'm_per_frame': 0.6, 'profile_step': 2},
    'scalar': {'m_per_frame': 0.3},
    'still': {'profile': [0.0, 0.0, None, 0.0], 'm_per_frame': 0.0,
              'profile_step': 1},
}
_MOTION_PAIRS = [(0, 5), (5, 0), (1, 4), (3, 10), (-2, 2), (2, 2)]


def _motion_between_cases():
    cases = []
    for name, motion in _MOTIONS.items():
        default = float(motion['m_per_frame'])
        for a, b in _MOTION_PAIRS:
            got = labeling.motion_between(motion, a, b, default=default)
            cases.append({'motion': name, 'from': int(a), 'to': int(b),
                          'default': default, 'expect': float(got)})
    return cases


def _motion_at_cases():
    cases = []
    for name, motion in _MOTIONS.items():
        for f in (-1, 0, 1, 2, 3, 5, 9):
            got = labeling.motion_at(motion, f)
            cases.append({'motion': name, 'frame': int(f),
                          'expect': (None if got is None else float(got))})
    return cases


# -------------------------------------------------------- _instance_center ---

_INSTANCE_MOTION = {
    'profile': [0.5, None, 0.7, 0.2, None, 0.9, 0.4, 0.4, None, 0.3, 0.3],
    'm_per_frame': 0.45, 'profile_step': 1,
}
_INSTANCE_CENTER = [1.0, -20.0, -0.325]     # z = z_ref + высота/2 (size[1]=1.75)
_INSTANCE_SIZE = [0.6, 1.75, 0.4]
_INSTANCE_TRAJ = {'start': [0.5, -10.0, -1.9], 'end': [1.5, -4.0, -1.6],
                  'seconds': 3.0}


def _instance_center_cases():
    """Центр объекта на кадре: labelBaseAt/labelCenterAt против _instance_center.

    `z_ref` на клиенте -- основание (center.z - size[1]/2); у сервера основание
    без траектории не двигается, поэтому z_ref = center.z - size[1]/2.
    """
    center = _INSTANCE_CENTER
    size = _INSTANCE_SIZE
    z_ref = center[2] - 0.5 * size[1]
    ref = 4
    cases = []
    for name, traj, motion in (
            ('plain', None, None),
            ('profile', None, _INSTANCE_MOTION),
            ('traj', _INSTANCE_TRAJ, None),
            ('traj_profile', _INSTANCE_TRAJ, _INSTANCE_MOTION)):
        for f in (0, 4, 7, 10, 14):
            got = labeling._instance_center(center, ref, f, 0.45, traj,
                                            hz=HZ, size=size, motion=motion)
            cases.append({
                'kind': name,
                'center': list(center), 'size': list(size), 'z_ref': float(z_ref),
                'ref_frame': ref, 'frame': int(f), 'speed': 0.45, 'hz': float(HZ),
                'traj': traj, 'motion': (_INSTANCE_MOTION if motion is not None
                                         else None),
                'expect': _f3(got),
            })
    return cases


# --------------------------------------------------------------- pose_time ---

def _pose_time_cases():
    """Время позы от опорного кадра: (f - ref)/hz, как в `labeling.save`."""
    cases = []
    for ref, frame in ((10, 10), (10, 25), (25, 10), (0, 7)):
        t = (int(frame) - int(ref)) / float(HZ)
        cases.append({'ref_frame': int(ref), 'frame': int(frame), 'hz': float(HZ),
                      'expect': float(t)})
    return cases


# -------------------------------------------------------------------- сбор ---

def vectors():
    """Все эталонные векторы: вход + выход, посчитанный серверными функциями."""
    return {
        'contract': 'formula-vectors',
        'version': CONTRACT_VERSION,
        'source': 'formula_contract.py (вызывает labeling.py -- источник истины)',
        'tolerance': dict(TOLERANCE),
        # Именованные профили хода: векторы motion_*/instance_center ссылаются
        # на них по имени; клиенту для прогона нужен сам профиль.
        'motions': {name: dict(m) for name, m in _MOTIONS.items()},
        'formulas': {
            'pose_moment': {
                'server': 'labeling.pose_moment', 'client': 'labellayer.poseMoment',
                'cases': _pose_moment_cases()},
            'pose_vertices': {
                'server': 'labeling.pose_vertices',
                'client': 'labellayer.poseVerticesAt',
                'cases': _pose_vertices_cases()},
            'pose_rotation': {
                'server': 'labeling._rotation', 'client': 'labellayer.poseRotation',
                'cases': _rotation_cases()},
            'place_pose': {
                'server': 'labeling.place_pose', 'client': 'labellayer.placePose',
                'cases': _place_pose_cases()},
            'pose_rule': {
                'server': 'labeling.pose_rule', 'client': 'labellayer.poseInScene',
                'cases': _pose_rule_cases()},
            'trajectory_at': {
                'server': 'labeling.trajectory_at',
                'client': 'labellayer.trajectoryAt',
                'cases': _trajectory_at_cases()},
            'trajectory_normalize': {
                'server': 'labeling.trajectory_normalize',
                'client': 'labellayer.trajectoryDistance/trajectorySpeed',
                'cases': _trajectory_normalize_cases()},
            'motion_at': {
                'server': 'labeling.motion_at', 'client': 'app.labelMotionAt',
                'cases': _motion_at_cases()},
            'motion_between': {
                'server': 'labeling.motion_between',
                'client': 'app.labelMotionBetween',
                'cases': _motion_between_cases()},
            'instance_center': {
                'server': 'labeling._instance_center',
                'client': 'app.labelBaseAt/labelCenterAt + labellayer.trajectoryBaseAtFrame',
                'cases': _instance_center_cases()},
            'pose_time': {
                'server': 'labeling.save (t_f = (f - ref)/hz)',
                'client': 'app.labelPoseTime',
                'cases': _pose_time_cases()},
        },
    }


def dumps(data):
    """Каноничный текст файла контракта (детерминированный, без дат)."""
    return json.dumps(data, ensure_ascii=False, indent=1, sort_keys=True) + '\n'


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    text = dumps(vectors())
    if '--check' in argv:
        try:
            with open(CONTRACT_PATH, encoding='utf-8') as fh:
                current = fh.read()
        except OSError:
            print(f'НЕТ ФАЙЛА: {CONTRACT_PATH} -- запустите: python formula_contract.py')
            return 1
        if current != text:
            print(f'РАСХОДИТСЯ: {CONTRACT_PATH} -- пересчитайте: python formula_contract.py')
            return 1
        print(f'ok: {os.path.relpath(CONTRACT_PATH, ROOT)} совпадает с labeling.py')
        return 0
    os.makedirs(os.path.dirname(CONTRACT_PATH), exist_ok=True)
    with open(CONTRACT_PATH, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write(text)
    n = sum(len(f['cases']) for f in vectors()['formulas'].values())
    print(f'записано: {os.path.relpath(CONTRACT_PATH, ROOT)} '
          f'({len(vectors()["formulas"])} формул, {n} векторов)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
