"""Собрать обучающий датасет: программная разметка объектов на записях.

Разметка для обучения синтезируется БЕЗ ручных кликов: раскладка объектов на
каждой записи (класс, поза, дальность, поперечное смещение, кадры, траектория)
задаётся СЕДОМ и собирается в чистых функциях (`plan_record`), а запись идёт
теми же вызовами, что у инструмента разметки (`labeling.propose` -- опора и
поворот, `labeling.preview` -- статус детектора, `labeling.save` -- карточки,
npz, index.jsonl). Выход -- ОТДЕЛЬНЫЙ каталог `dataset/` (в .gitignore):
рукописная разметка заказчика в `labels/` не трогается.

Что кладётся в датасет
----------------------
* `dataset/labels/` -- синтетика в формате `labels/` v2: люди в позах
  standing/walking/running/lying/fallen на дальностях 5/10/20/40 м, поперёк
  пути u в -1.2..+1.2 м, часть -- с траекториями (вдоль/поперёк, 2-8 с,
  0.5-2 м/с); оборудование -- столб/ящик примитивом `sim/objects.py`;
* `dataset/anchor_person/` -- РЕАЛЬНЫЙ человек из `doubleT_obstacle`
  (кадры 13..63): GT -- ИЗМЕРЕННЫЕ точки кластера (`points_source=measured`),
  см. `labeling._measured_map`;
* `dataset/manifest.json` -- статистика по классам/позам/дальностям/
  траекториям, негативы, сплит train/val ПО ЗАПИСАМ, seed, ревизии.

Негативы -- кадры, где нет разметки И детектор молчит (список в манифесте,
карточек нет). Кадры 10..66 `doubleT_obstacle` (там стоит реальный человек) из
раскладки и негативов исключены.

Ограничение по памяти: `BagFrames` держит ~2 ГБ на запись, поэтому воркеров
немного (`--workers`, по умолчанию 2). Раскладка детерминирована: одинаковый
seed -- одинаковые объекты (прогон повторно пропускает готовые записи,
`--force` пересчитывает).

    python tools/build_dataset.py                       # все 6 записей
    python tools/build_dataset.py --records doubleT_obstacle roundT_doubleT
    python tools/build_dataset.py --workers 3 --objects-per-record 70

Оценка времени (замерено на этой машине): ход-профиль ~0.12 с/кадр записи,
трассировка ~0.16 с на кадр объекта, propose+preview ~0.3 с на объект;
2488 кадров шести записей -- примерно 15-25 мин одним воркером, с двумя-тремя
-- кратно быстрее.
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import labeling  # noqa: E402
import numpy as np  # noqa: E402
from bag_reader import BagFrames, find_db3  # noqa: E402

RECORDS = (
    'doubleT_obstacle',
    'doubleT_platform',
    'roundT_doubleT',
    'roundT_pressureGate_roundT',
    'roundT_squareT_pressureGate_squareT',
    'squareT_platform_squareT_switch',
)

# Реальный человек в doubleT_obstacle (кадры 10..66): сюда синтетику НЕ ставим
# и негативы оттуда НЕ берём; его собственная разметка -- якорь (ANCHOR_FRAMES).
REAL_RECORD = 'doubleT_obstacle'
REAL_FRAMES = frozenset(range(10, 67))
ANCHOR_FRAMES = tuple(range(13, 64))

DISTANCES = (5.0, 10.0, 20.0, 40.0)
U_RANGE = (-1.2, 1.2)
POSES_STATIC = ('standing', 'lying', 'fallen')
POSES_MOVING = ('walking', 'running')
TRAJ_CHANCE = 0.35            # доля людей с траекторией
TRAJ_SECONDS = (2.0, 8.0)
TRAJ_SPEED = (0.5, 2.0)
TRAJ_U_LIMIT = 2.0            # поперечная траектория не дальше от оси
EQUIPMENT_CHANCE = 0.20
FRAMES_MIN, FRAMES_MAX = 3, 8
FRAMES_NEAR_MAX = 4           # на 5 м -- короче: сенсор наезжает быстро
FRAME_START_MIN = 8           # самые первые кадры записи пропускаем

PERSON_SIZE = {
    'standing': (0.55, 1.75, 0.55),
    'walking': (0.60, 1.75, 0.60),
    'running': (0.65, 1.75, 0.65),
    'lying': (0.60, 0.45, 2.00),
    'fallen': (0.60, 0.50, 1.90),
}
EQUIPMENT_SIZE = {'column': (0.30, 2.00, 0.30), 'box': (0.80, 0.80, 0.80)}

# Сплит ПО ЗАПИСАМ (кадры одной записи -- одна сцена; перемешивать их между
# train и val -- утечка сцены). val = 2 из 6: doubleT_obstacle (запись СТОИТ,
# там реальный человек -- якорь сверки с реальностью) + roundT_doubleT (едет
# по кривой); train -- платформа, ворота, кривая с воротами, стрелка. Так в
# val попадают обе крайности (стенд и ход), и он достаточно велик, чтобы
# recall по дальностям/позам был осмысленным (записей всего шесть).
VAL_RECORDS = frozenset(('doubleT_obstacle', 'roundT_doubleT'))
SPLIT_WHY = ('по записям, иначе утечка сцены: val = doubleT_obstacle (стоит, '
             'реальный человек -- якорь) + roundT_doubleT (едет по кривой); '
             'train = платформа, ворота, кривая+ворота, стрелка')

NEGATIVE_WANT = 60            # кандидатов в негативы на запись (до фильтра)
Y_BEHIND_LIMIT = -1.5         # ближе этого к сенсору объект в кадре не виден
Y_FAR_LIMIT = -75.0


def split_records(records=RECORDS):
    """Сплит train/val по записям: {'train': [...], 'val': [...], 'why': ...}."""
    val = [r for r in records if r in VAL_RECORDS]
    train = [r for r in records if r not in VAL_RECORDS]
    return {'train': train, 'val': val, 'why': SPLIT_WHY}


def plan_record(record, total, seed, n_objects):
    """Раскладка объектов записи: ЧИСТАЯ функция (тот же seed -- та же раскладка).

    Возвращает список спецификаций: id, класс, поза, дальность (5/10/20/40 м),
    поперечное смещение u, размер, кадры (3-8 подряд), траектория (часть людей:
    вдоль/поперёк, 2-8 с, 0.5-2 м/с) или None. Кадры doubleT_obstacle обходят
    REAL_FRAMES (10..66): там стоит реальный человек. walking/running бывают
    только с траекторией (без неё поза в сцене играет standing), стоячие позы
    (standing/lying/fallen) -- только без.
    """
    rng = random.Random('%s|%s' % (seed, record))
    lo_frame = (max(REAL_FRAMES) + 1) if record == REAL_RECORD else FRAME_START_MIN
    specs = []
    for i in range(int(n_objects)):
        equip = rng.random() < EQUIPMENT_CHANCE
        dist = rng.choice(DISTANCES)
        u = round(rng.uniform(*U_RANGE), 3)
        n_frames = (rng.randint(FRAMES_MIN, FRAMES_NEAR_MAX) if dist <= DISTANCES[0] + 1e-9
                    else rng.randint(FRAMES_MIN, FRAMES_MAX))
        hi = int(total) - n_frames - 1
        if hi < lo_frame:
            continue                                  # запись короче раскладки
        f0 = rng.randint(lo_frame, hi)
        frames = list(range(f0, f0 + n_frames))
        if equip:
            kind = rng.choice(('column', 'box'))
            specs.append({
                'id': 'gen-%04d' % i, 'class': 'equipment', 'pose': 'unknown',
                'equipment_kind': kind, 'size': list(EQUIPMENT_SIZE[kind]),
                'distance_m': float(dist), 'u_m': u, 'ref_frame': f0,
                'frames': frames, 'trajectory': None,
            })
            continue
        traj = None
        if rng.random() < TRAJ_CHANCE:
            seconds = round(rng.uniform(*TRAJ_SECONDS), 2)
            speed = round(rng.uniform(*TRAJ_SPEED), 2)
            traj = {'direction': rng.choice(('along', 'across')),
                    'sign': rng.choice((-1, 1)),
                    'seconds': seconds, 'speed_mps': speed}
        pose = rng.choice(POSES_MOVING if traj else POSES_STATIC)
        specs.append({
            'id': 'gen-%04d' % i, 'class': 'person', 'pose': pose,
            'size': list(PERSON_SIZE[pose]),
            'distance_m': float(dist), 'u_m': u, 'ref_frame': f0,
            'frames': frames, 'trajectory': traj,
        })
    return specs


def negative_candidates(total, used, forbidden=(), want=NEGATIVE_WANT):
    """Кандидаты в негативы: кадры без разметки и вне запретных зон.

    Шаг по записи -- чтобы равномерно покрыть её (~`want` кандидатов); если
    раскладка заняла почти весь шаг (кандидатов меньше четверти), шаг мельчит.
    Молчание детектора на кандидатах проверяет воркер (`_silent_negatives`):
    здесь только геометрия -- негатив обязан не пересекаться с размеченными
    кадрами.
    """
    used = {int(f) for f in used}
    bad = {int(f) for f in forbidden}

    def pick(stride):
        return [f for f in range(0, int(total), stride)
                if f not in used and f not in bad]

    stride = max(1, int(total) // max(int(want), 1))
    out = pick(stride)
    while stride > 1 and len(out) < max(4, int(want) // 4):
        stride = max(1, stride // 2)
        out = pick(stride)
    return out


def _silent_negatives(model, cfg, candidates, xyz_at):
    """Кандидаты, на которых детектор молчит: проверенно пустые кадры."""
    from detector.core import detect  # noqa: PLC0415

    silent, errors = [], []
    for f in candidates:
        try:
            xyz, _inten, _ring = xyz_at(int(f))
            result = detect(np.asarray(xyz, dtype=np.float64), model=model, cfg=cfg)
        except Exception:                              # noqa: BLE001 -- битый кадр не негатив
            errors.append(int(f))
            continue
        if not (result.obstacles or []):
            silent.append(int(f))
    return silent, errors


# ------------------------------------------------------------------- воркер

def _hz_of(frames):
    n = len(frames)
    if n > 1:
        return (frames.timestamp(n - 1) - frames.timestamp(0)) / 1e9 / (n - 1)
    return labeling.HZ_DEFAULT


def _xyz_at(frames):
    import numpy as np  # noqa: PLC0415

    def xyz_at(f):
        got = frames[int(f)]
        return (np.asarray(got.xyz, dtype=np.float64),
                np.asarray(got.intensity, dtype=np.float64),
                np.asarray(got.ring, dtype=np.int64))

    return xyz_at


def _spec_to_object(spec, xyz, model, motion, hz, db_path=''):
    """Спецификация -> объект для labeling.save: опора, поворот, траектория.

    Опора -- ИЗМЕРЕННАЯ поверхность под местом объекта (`_surface_z`), поворот
    бокса -- по оси пути кадра. Траектория переводится в две точки основания:
    вдоль пути (±Y) или поперёк (±X, не дальше TRAJ_U_LIMIT от оси).
    """
    pf = labeling.path_frame(xyz, model)
    y = -float(spec['distance_m'])
    u = float(spec['u_m'])
    surf, _info = labeling._surface_z(xyz, pf, y, u)
    z_ref = (float(pf.floor_z(np.asarray([y]))[0]) if surf is None else float(surf))
    x0 = float(pf.axis_x(np.asarray([y]))[0]) + u
    size = [float(v) for v in spec['size']]
    center = [x0, y, z_ref + 0.5 * size[1]]
    traj = None
    t = spec.get('trajectory')
    if t:
        base = [x0, y, z_ref]
        if t['direction'] == 'along':
            end = [x0, y + t['sign'] * t['speed_mps'] * t['seconds'], z_ref]
        else:
            reach = t['speed_mps'] * t['seconds']
            room = max(0.5, TRAJ_U_LIMIT - abs(u))
            end = [x0 + t['sign'] * min(reach, room), y, z_ref]
        traj = {'start': base, 'end': end, 'seconds': float(t['seconds']),
                'source': 'synth'}
    obj = {
        'id': spec['id'], 'class': spec['class'], 'pose': spec['pose'],
        'ref_frame': int(spec['ref_frame']), 'center': center, 'size': size,
        'yaw': math.degrees(math.atan2(-pf.slope(y), 1.0)),
        'pitch': math.degrees(math.atan2(pf.grade(y), 1.0)),
        'roll': 0.0, 'frames': list(spec['frames']), 'automation': 'cluster',
        'speed_m_per_frame': 0.0, 'speed_source': 'measured', 'z_ref': z_ref,
        'db_path': db_path,
    }
    if traj:
        obj['trajectory'] = traj
    return obj, pf, z_ref


def build_record(task):
    """Разметить ОДНУ запись (один процесс -- одна запись): план -> save.

    Шаги: ход-профиль по кадрам -> раскладка -> опора/поворот по кадру ->
    preview на опорном кадре (статус детектора -> automation) -> один вызов
    `labeling.save` на все объекты записи -> негативы (детектор молчит).
    Объекты, чья проекция на последний кадр уводит их за сенсор или дальше
    75 м, отбрасываются ДО записи (честно считаются в `dropped`).
    """
    record, db_path, total, out_root, seed, n_objects, force = task
    labels_dir = os.path.join(out_root, 'labels')
    t_start = time.time()
    frames = BagFrames(db_path, cache_size=16, prefetch_ahead=0)
    xyz_at = _xyz_at(frames)
    hz = _hz_of(frames)
    model = labeling._model_for(db_path)
    plan = plan_record(record, total, seed, n_objects)

    have = {row.get('id') for row in
            labeling._read_index(os.path.join(labels_dir, 'index.jsonl'))
            if row.get('record') == record}
    want_ids = {spec['id'] for spec in plan}
    # Скип: все ЗАПИСАННЫЕ объекты записи входят в план (часть плана отбрасы-
    # вается проекцией, их id в индексе нет -- поэтому сравнение в эту сторону).
    # Раскладка детерминирована: те же seed и объём -- тот же набор выживших.
    if not force and have and have <= want_ids:
        frames.close()
        print('  [%s] уже размечена (%d объектов) -- пропуск (сид совпадает)'
              % (record, len(want_ids)), flush=True)
        return _stats_from_rows(record, labels_dir, plan, want_ids, 0.0,
                                out_root=out_root)

    print('  [%s] кадров %d, план %d объектов; ход-профиль...'
          % (record, total, len(plan)), flush=True)
    t0 = time.time()
    motion = labeling.motion_profile(db_path, xyz_at=xyz_at, total=total, hz=hz)
    motion_s = time.time() - t0
    print('  [%s] ход %.3f м/кадр (%.1f с); раскладка и preview...'
          % (record, motion['m_per_frame'], motion_s), flush=True)

    objs, dropped = [], []
    confirmed = 0
    t0 = time.time()
    for k, spec in enumerate(plan):
        try:
            xyz, inten, ring = xyz_at(spec['ref_frame'])
            obj, _pf, z_ref = _spec_to_object(spec, xyz, model, motion, hz,
                                              db_path=db_path)
            # Проекция на последний кадр: объект обязан оставаться в поле
            # зрения (не за сенсором и не дальше Y_FAR_LIMIT).
            traj_n = labeling.trajectory_normalize(obj.get('trajectory'))
            last = labeling._instance_center(
                obj['center'], obj['ref_frame'], spec['frames'][-1], 0.0, traj_n,
                hz=hz, size=obj['size'], motion=motion)
            if not (Y_FAR_LIMIT < last[1] < Y_BEHIND_LIMIT):
                dropped.append({'id': spec['id'], 'y_last': round(last[1], 2)})
                continue
            pv = labeling.preview(
                obj['class'], obj['pose'], obj['center'], obj['size'],
                yaw=obj['yaw'], pitch=obj['pitch'], xyz=xyz,
                intensity=inten, ring=ring, model=model,
                pose_time_s=0.0, base_z=z_ref)
            if pv['confirmed']:
                obj['automation'] = 'detector'
                confirmed += 1
            else:
                obj['automation'] = 'cluster'
            objs.append(obj)
        except Exception as exc:                       # noqa: BLE001 -- один объект не рушит запись
            dropped.append({'id': spec['id'], 'error': '%s: %s' % (type(exc).__name__, exc)})
        if (k + 1) % 20 == 0:
            print('  [%s] %d/%d объектов собрано (%.1f с)'
                  % (record, k + 1, len(plan), time.time() - t0), flush=True)
    if not objs and plan:
        print('  ! [%s] НИ ОДИН объект не собран (ошибок %d, отброшено %d) -- '
              'запись не размечена' % (record, len(dropped), len(dropped)),
              flush=True)
    place_s = time.time() - t0

    t0 = time.time()
    report = labeling.save(record, objs, xyz_at, out_dir=labels_dir,
                           hz=hz, motion=motion)
    save_s = time.time() - t0
    written_ids = [w['id'] for w in report['written']]
    traced_frames = sum(len(o['frames']) for o in objs)
    print('  [%s] save: %d объектов, %d кадров трассировки, %.1f с (%.2f с/кадр), '
          'ошибок %d' % (record, len(written_ids), traced_frames, save_s,
                         save_s / max(traced_frames, 1), len(report['errors'])),
          flush=True)
    for err in report['errors'][:5]:
        print('  ! [%s] %s: %s' % (record, err['object'], err['error']), flush=True)

    # Негативы: кадры без разметки, где детектор молчит.
    used = {f for o in objs for f in o['frames']}
    forbidden = set(REAL_FRAMES) if record == REAL_RECORD else set()
    cands = negative_candidates(total, used, forbidden)
    t0 = time.time()
    from detector import DetectorConfig  # noqa: PLC0415

    silent, neg_errors = _silent_negatives(model, DetectorConfig(),
                                           cands, xyz_at)
    neg_s = time.time() - t0
    print('  [%s] негативы: %d из %d кандидатов молчат (%.1f с)'
          % (record, len(silent), len(cands), neg_s), flush=True)
    frames.close()

    # Негативы и статистика записи -- на диск: манифест пересобирается и при
    # пропуске записи (повторный прогон), и числа не теряются.
    neg_dir = os.path.join(out_root, 'negatives')
    os.makedirs(neg_dir, exist_ok=True)
    with open(os.path.join(neg_dir, record + '.json'), 'w', encoding='utf-8') as fh:
        json.dump({'negatives': silent, 'candidates': len(cands),
                   'errors': neg_errors, 'excluded_frames': sorted(forbidden)},
                  fh, ensure_ascii=False, indent=1)
    result = {
        'record': record, 'objects': len(written_ids), 'planned': len(plan),
        'dropped': dropped, 'errors': [e['error'] for e in report['errors']],
        'confirmed': confirmed,
        'frames_traced': traced_frames, 'points': sum(w['n_points'] for w in
                                                      report['written']),
        'negatives': silent, 'negative_candidates': len(cands),
        'negative_errors': neg_errors,
        'hz': hz, 'motion_m_per_frame': motion['m_per_frame'],
        'seconds': {'motion': round(motion_s, 1), 'place': round(place_s, 1),
                    'save': round(save_s, 1), 'negatives': round(neg_s, 1),
                    'total': round(time.time() - t_start, 1)},
        'sec_per_traced_frame': round(save_s / max(traced_frames, 1), 3),
    }
    stats_dir = os.path.join(out_root, 'records')
    os.makedirs(stats_dir, exist_ok=True)
    with open(os.path.join(stats_dir, record + '.json'), 'w', encoding='utf-8') as fh:
        json.dump(result, fh, ensure_ascii=False, indent=1)
    return result


def _stats_from_rows(record, labels_dir, plan, want_ids, total_s, out_root=None):
    """Статистика уже записанной записи (пропуск при повторном прогоне).

    Числа (время, скорость, счётчики) берутся из сохранённой воркером
    статистики `records/<record>.json`; кадры/объекты пересчитываются по
    индексу -- это гарантированно то, что лежит на диске.
    """
    objs = labeling._read_index(os.path.join(labels_dir, 'index.jsonl'))
    rows = [r for r in objs if r.get('record') == record and r.get('id') in want_ids]
    used = {int(f) for r in rows for f in (r.get('frames') or [])}
    frames_traced = sum(len(r.get('frames') or []) for r in rows)
    root = out_root or os.path.dirname(labels_dir)
    result = {
        'record': record, 'objects': len(rows), 'planned': len(plan),
        'dropped': [], 'errors': [], 'confirmed': None,
        'frames_traced': frames_traced, 'frames_distinct': len(used),
        'points': sum(int(r.get('n_points') or 0) for r in rows),
        'negatives': [], 'negative_candidates': 0, 'negative_errors': [],
        'hz': None, 'motion_m_per_frame': None,
        'seconds': {'total': total_s}, 'sec_per_traced_frame': None,
        'skipped': True,
    }
    for rel in (('negatives', record + '.json'), ('records', record + '.json')):
        path = os.path.join(root, rel[0], rel[1])
        if not os.path.isfile(path):
            continue
        with open(path, encoding='utf-8') as fh:
            data = json.load(fh)
        if rel[0] == 'negatives':
            result['negatives'] = [int(f) for f in data.get('negatives') or []]
            result['negative_candidates'] = int(data.get('candidates') or 0)
        else:
            for key in ('dropped', 'errors', 'confirmed', 'hz',
                        'motion_m_per_frame', 'seconds',
                        'sec_per_traced_frame'):
                if data.get(key) is not None:
                    result[key] = data[key]
    return result


# ------------------------------------------------------------------- якорь

def build_anchor(out_root, db_path, force=False):
    """Разметить РЕАЛЬНОГО человека из doubleT_obstacle (кадры 13..63).

    Правило кластера (задокументировано и в манифесте датасета):
    1) на кадре гоняется детектор; семя кластера -- точка ЦЕНТРА его лучшего
       препятствия в путевых координатах (y, u, середина h_min..h_max); в этих
       записях детектор на кадрах якоря других препятствий не выдаёт
       (validation/fp_per_hour: все события -- этот человек);
    2) если детектор молчит -- семя центроид кластера ПРЕДЫДУЩЕГО кадра
       (запись стоит, человек на месте; кадр честно помечается miss, если и
       этого нет);
    3) кластер -- связные воксели вокруг семени (`labeling._cluster_around`,
       тот же, что у инструмента разметки), при неудаче -- ближайший в плане
       (`_cluster_plan`), затем шар (`_cluster_ball`);
    4) кластер дальше 2 м от семени отбрасывается (кадр -- miss).
    GT карточки -- сами точки кластера (points_source=measured), центр на
    кадре -- бокс по кластеру. Поза -- по высоте кластера (>= 0.9 м -- standing,
    иначе fallen).
    """
    anchor_dir = os.path.join(out_root, 'anchor_person')
    have = {row.get('id') for row in
            labeling._read_index(os.path.join(anchor_dir, 'index.jsonl'))}
    if not force and 'real-person' in have:
        print('  [anchor] уже размечен -- пропуск', flush=True)
        return {'skipped': True}

    frames = BagFrames(db_path, cache_size=8, prefetch_ahead=0)
    xyz_at = _xyz_at(frames)
    model = labeling._model_for(db_path)
    from detector import DetectorConfig  # noqa: PLC0415
    from detector.core import detect  # noqa: PLC0415

    cfg = DetectorConfig()
    measured, misses = {}, []
    det_frames, det_silent = [], []
    prev_click = None
    points_per_frame = {}
    for f in ANCHOR_FRAMES:
        try:
            xyz, _inten, ring = xyz_at(f)
        except Exception as exc:                       # noqa: BLE001 -- кадр может не читаться
            misses.append({'frame': int(f), 'reason': 'кадр не читается: %s' % exc})
            continue
        pf = labeling.path_frame(xyz, model)
        result = detect(xyz, model=model, cfg=cfg)
        obstacles = sorted(result.obstacles or [], key=lambda o: -o.confidence)
        seed_kind = 'detector'
        if obstacles:
            det_frames.append(int(f))
            ob = obstacles[0].to_dict()
            h_mid = 0.5 * (float(ob['h_min_m']) + float(ob['h_max_m']))
            click = pf.to_xyz(np.asarray([float(ob['y_m'])]),
                              float(ob['u_m']), h_mid)[0].tolist()
            click = [float(v) for v in click]
        elif prev_click is not None:
            det_silent.append(int(f))
            seed_kind = 'prev-frame'
            click = prev_click
        else:
            det_silent.append(int(f))
            misses.append({'frame': int(f), 'reason': 'детектор молчит, прошлого кластера нет'})
            continue
        got = labeling._cluster_around(xyz, click, pf)
        if got is None or got[0].size < labeling.MEASURED_MIN_POINTS:
            got = labeling._cluster_plan(xyz, click, pf)
            seed_kind += '+plan'
        if got is None or got[0].size < labeling.MEASURED_MIN_POINTS:
            got = labeling._cluster_ball(xyz, click, pf)
            seed_kind += '+ball'
        if got is None or got[0].size < labeling.MEASURED_MIN_POINTS:
            misses.append({'frame': int(f),
                           'reason': 'кластер не набрался (семя: %s)' % seed_kind})
            continue
        idx, _terms = got
        centroid = np.asarray(xyz[idx]).mean(axis=0)
        if math.dist(centroid.tolist()[:2], click[:2]) > 2.0:
            misses.append({'frame': int(f),
                           'reason': 'кластер дальше 2 м от семени (%s)' % seed_kind})
            continue
        measured[int(f)] = idx.tolist()
        points_per_frame[int(f)] = int(idx.size)
        prev_click = centroid.tolist()
    frames.close()

    if not measured:
        return {'error': 'кластеры не построились ни на одном кадре', 'misses': misses}

    frames = BagFrames(db_path, cache_size=8, prefetch_ahead=0)
    xyz_at = _xyz_at(frames)
    ref = min(measured)
    xyz_ref, _inten, ring_ref = xyz_at(ref)
    pf_ref = labeling.path_frame(xyz_ref, model)
    tr0 = labeling._measured_frame_gt(xyz_ref, ring_ref,
                                      np.asarray(measured[ref]).astype('int64'), model)
    height = float(tr0['box']['extents']['h_floor'][1] - tr0['box']['extents']['h_floor'][0])
    pose = 'standing' if height >= 0.9 else 'fallen'
    y_c = float(tr0['center'][1])
    obj = {
        'id': 'real-person', 'class': 'person', 'pose': pose,
        'ref_frame': ref, 'center': [float(v) for v in tr0['center']],
        'size': [float(v) for v in tr0['box']['size']],
        'yaw': math.degrees(math.atan2(-pf_ref.slope(y_c), 1.0)),
        'pitch': math.degrees(math.atan2(pf_ref.grade(y_c), 1.0)),
        'roll': 0.0, 'frames': sorted(measured), 'automation': 'cluster',
        'speed_m_per_frame': 0.0, 'speed_source': 'measured',
        'points_source': 'measured', 'measured': measured,
        'z_ref': float(tr0['box']['base_z']), 'db_path': db_path,
    }
    report = labeling.save('doubleT_obstacle', [obj], xyz_at,
                           out_dir=anchor_dir, hz=_hz_of(frames), motion=None)
    frames.close()
    counts = points_per_frame
    return {
        'record': 'doubleT_obstacle', 'id': 'real-person',
        'frames_labeled': sorted(measured), 'misses': misses,
        'detector_frames': det_frames, 'detector_silent': det_silent,
        'points_total': int(sum(counts.values())),
        'points_avg': round(sum(counts.values()) / float(len(counts)), 1),
        'points_min': int(min(counts.values())), 'points_max': int(max(counts.values())),
        'pose': pose, 'cluster_height_m': round(height, 3),
        'save_errors': [e['error'] for e in report['errors']],
        'rule': ('семя -- центр лучшего препятствия детектора (u/y/h кадра); '
                 'детектор молчит -- центроид кластера прошлого кадра; кластер -- '
                 'связные воксели labeling._cluster_around (+plan, +ball), '
                 'дальше 2 м от семени -- miss'),
    }


# ------------------------------------------------------------------- манифест

def _dir_size(path):
    total = 0
    for base, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(base, name))
            except OSError:
                pass
    return total


TRAJ_ROW_FIELDS = (('start', 4), ('end', 4), ('seconds', 3),
                   ('distance_m', 4), ('speed_mps', 4), ('speed_kmh', 3))


def rebuild_labels_index(labels_dir):
    """Перестроить index.jsonl и manifest ПО КАРТОЧКАМ (устраняет гонку воркеров).

    `labeling.save` пишет index.jsonl целиком (читает-дописывает), поэтому при
    параллельной разметке нескольких записей последний воркер затирает строки
    остальных. Карточки при этом целы -- индекс собирается заново из них:
    строки те же, что пишет save (поля, округления траектории, сортировка).
    """
    labels_dir = os.path.abspath(labels_dir)
    rows = []
    if not os.path.isdir(labels_dir):
        return []
    for record in sorted(os.listdir(labels_dir)):
        rec_dir = os.path.join(labels_dir, record)
        if not os.path.isdir(rec_dir):
            continue
        for fn in sorted(os.listdir(rec_dir)):
            if not fn.endswith('.json'):
                continue
            try:
                with open(os.path.join(rec_dir, fn), encoding='utf-8') as fh:
                    card = json.load(fh)
            except (OSError, ValueError):
                continue
            row = {k: card[k] for k in ('id', 'record', 'class', 'pose',
                                        'pose_scene', 'frames', 'n_points',
                                        'automation')
                   if k in card}
            row['format'] = int((card.get('format') or {}).get('version') or 1)
            if card.get('points_source'):
                row['points_source'] = str(card['points_source'])
            traj = card.get('trajectory')
            if isinstance(traj, dict) and 'start' in traj:
                out = {}
                for key, digits in TRAJ_ROW_FIELDS:
                    if key not in traj:
                        continue
                    val = traj[key]
                    out[key] = ([round(float(v), digits) for v in val]
                                if isinstance(val, (list, tuple))
                                else round(float(val), digits))
                row['trajectory'] = out
            rows.append(row)
    rows.sort(key=labeling._id_sort_key)
    if rows:
        labeling._atomic_write(
            os.path.join(labels_dir, 'index.jsonl'),
            ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows)
            .encode('utf-8'))
        labeling._atomic_write(
            os.path.join(labels_dir, 'manifest.json'),
            json.dumps(labeling._manifest(rows, labeling.revision()),
                       ensure_ascii=False, indent=1).encode('utf-8'))
    return rows


def _breakdown(specs_by_id):
    by_class, by_pose, by_dist, by_traj = {}, {}, {}, {}
    for spec in specs_by_id.values():
        by_class[spec['class']] = by_class.get(spec['class'], 0) + 1
        by_pose[spec['pose']] = by_pose.get(spec['pose'], 0) + 1
        key = '%g м' % spec['distance_m']
        by_dist[key] = by_dist.get(key, 0) + 1
        kind = ('traj' if spec.get('trajectory') else 'static')
        by_traj[kind] = by_traj.get(kind, 0) + 1
    return {'by_class': dict(sorted(by_class.items())),
            'by_pose': dict(sorted(by_pose.items())),
            'by_distance': dict(sorted(by_dist.items())),
            'by_trajectory': dict(sorted(by_traj.items()))}


def write_manifest(out_root, stats, seed, args):
    """Манифест датасета: статистика, негативы, сплит, seed, ревизии."""
    labels_dir = os.path.join(out_root, 'labels')
    rows = labeling._read_index(os.path.join(labels_dir, 'index.jsonl'))
    # Ключ (запись, id): id генерации повторяются в записях (gen-0001 есть в
    # каждой), без записи в ключе статистика съезжается в одну запись.
    plan_specs = {}
    for record, result in stats['records'].items():
        for spec in plan_record(record, result.get('total') or 0, seed,
                                result.get('n_objects_arg') or 0):
            plan_specs[(record, spec['id'])] = spec
    written = {(r.get('record'), r['id']): r for r in rows}
    specs = {k: v for k, v in plan_specs.items() if k in written}
    manifest = {
        'dataset': 'train/val по записям for_hackathon, синтез labeling.save',
        'created': datetime.datetime.now().isoformat(timespec='seconds'),
        'tool': 'tools/build_dataset.py',
        'seed': seed,
        'labels_dir': 'dataset/labels',
        'anchor_dir': 'dataset/anchor_person',
        'format': dict(labeling.FORMAT_INFO),
        'revision': labeling.revision(),
        'args': dict(vars(args)),
        'split': split_records(args.records if args.records else list(RECORDS)),
        'totals': {
            'objects': len(rows),
            'frames': sum(len(r.get('frames') or []) for r in rows),
            'points': sum(int(r.get('n_points') or 0) for r in rows),
            'negatives': sum(len(s.get('negatives') or []) for s in
                             stats['records'].values()),
            'records': len(stats['records']),
            'seconds': round(sum((s.get('seconds') or {}).get('total') or 0.0
                                 for s in stats['records'].values()), 1),
        },
        'statistics': _breakdown(specs),
        'records': {
            name: {
                'objects': s.get('objects'), 'planned': s.get('planned'),
                'frames_traced': s.get('frames_traced'),
                'points': s.get('points'), 'negatives': s.get('negatives'),
                'dropped': s.get('dropped'), 'errors': s.get('errors'),
                'automation_detector': s.get('confirmed'),
                'motion_m_per_frame': s.get('motion_m_per_frame'),
                'seconds': s.get('seconds'),
                'sec_per_traced_frame': s.get('sec_per_traced_frame'),
                'skipped': bool(s.get('skipped')),
            } for name, s in stats['records'].items()
        },
        'anchor_person': stats.get('anchor'),
        'negatives_rule': ('кадр без разметки, детектор молчит (detector.core.detect, '
                           'DetectorConfig по умолчанию); кандидаты -- равномерный шаг по '
                           'записи; кадры реального человека (10..66 doubleT_obstacle) '
                           'исключены и из раскладки, и из негативов'),
        'disk_bytes': _dir_size(out_root),
    }
    path = os.path.join(out_root, 'manifest.json')
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=1)
    return path


def _record_frames(db_path):
    frames = BagFrames(db_path, cache_size=1, prefetch_ahead=0)
    n = len(frames)
    frames.close()
    return n


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Собрать датасет: синтетическая разметка + якорь реального '
                    'человека + негативы + манифест со сплитом')
    parser.add_argument('--seed', type=int, default=20260915,
                        help='сид раскладки (один сид -- одна раскладка)')
    parser.add_argument('--out', default='dataset',
                        help='корень датасета (по умолчанию dataset/; в .gitignore)')
    parser.add_argument('--records', nargs='*', default=None,
                        help='записи (по умолчанию все 6)')
    parser.add_argument('--objects-per-record', type=int, default=60,
                        help='объектов в плане на запись (по умолчанию 60)')
    parser.add_argument('--workers', type=int, default=2,
                        help='параллельных процессов (по умолчанию 2: BagFrames '
                             'держит ~2 ГБ на запись)')
    parser.add_argument('--no-anchor', action='store_true',
                        help='не строить якорь реального человека')
    parser.add_argument('--anchor-only', action='store_true',
                        help='только якорь реального человека')
    parser.add_argument('--force', action='store_true',
                        help='пересчитать, даже если объекты уже записаны')
    args = parser.parse_args(argv)

    out_root = os.path.abspath(os.path.join(ROOT, args.out))
    records = [r for r in (args.records or list(RECORDS)) if r in RECORDS]
    unknown = [r for r in (args.records or []) if r not in RECORDS]
    if unknown:
        raise SystemExit('нет таких записей: %s; доступны: %s'
                         % (', '.join(unknown), ', '.join(RECORDS)))

    totals = {}
    for record in records:
        db = find_db3(os.path.join(ROOT, 'for_hackathon', record))
        if db is None:
            raise SystemExit('нет .db3 у записи %s' % record)
        totals[record] = _record_frames(db)
    print('Датасет: записей %d, кадров %d, объектов в плане %d на запись, сид %d; '
          'выход %s' % (len(records), sum(totals.values()), args.objects_per_record,
                        args.seed, out_root), flush=True)

    stats = {'records': {}}
    t0 = time.time()
    tasks = [(record, find_db3(os.path.join(ROOT, 'for_hackathon', record)),
              totals[record], out_root, args.seed, args.objects_per_record,
              bool(args.force))
             for record in records]
    if not args.anchor_only:
        jobs = max(1, min(int(args.workers), len(tasks)))
        if jobs == 1:
            for task in tasks:
                stats['records'][task[0]] = build_record(task)
        else:
            with ProcessPoolExecutor(max_workers=jobs) as pool:
                for task, result in zip(tasks, pool.map(build_record, tasks)):
                    stats['records'][task[0]] = result
        for record in records:
            stats['records'][record]['total'] = totals[record]
            stats['records'][record]['n_objects_arg'] = args.objects_per_record
        # Воркеры писали общий index.jsonl параллельно -- индекс и манифест
        # формата пересобираются по карточкам (гонка устранена в главном).
        rows = rebuild_labels_index(os.path.join(out_root, 'labels'))
        print('  index: %d объектов в %d записях' % (len(rows), len(records)),
              flush=True)

    if not args.no_anchor:
        print('  [anchor] реальный человек doubleT_obstacle, кадры %d..%d...'
              % (ANCHOR_FRAMES[0], ANCHOR_FRAMES[-1]), flush=True)
        t1 = time.time()
        anchor_db = find_db3(os.path.join(ROOT, 'for_hackathon', REAL_RECORD))
        stats['anchor'] = build_anchor(out_root, anchor_db, force=args.force)
        rebuild_labels_index(os.path.join(out_root, 'anchor_person'))
        print('  [anchor] готов за %.1f с' % (time.time() - t1), flush=True)

    manifest_path = write_manifest(out_root, stats, args.seed, args)
    spent = time.time() - t0
    objects = sum(s.get('objects') or 0 for s in stats['records'].values())
    frames_traced = sum(s.get('frames_traced') or 0 for s in stats['records'].values())
    negatives = sum(len(s.get('negatives') or []) for s in stats['records'].values())
    print('\nИтог: записей %d, объектов %d, размеченных кадров %d, негативов %d; '
          'время %.1f с; датасет %.1f МБ; манифест %s'
          % (len(stats['records']), objects, frames_traced, negatives, spent,
             _dir_size(out_root) / 2**20, manifest_path), flush=True)
    if records and not args.anchor_only and objects == 0:
        print('ОШИБКА: объектов не записано ни одного -- смотрите dropped/errors '
              'по записям выше', flush=True)
        return 1
    if records and not args.anchor_only:
        sec_frame = [s['sec_per_traced_frame'] for s in stats['records'].values()
                     if s.get('sec_per_traced_frame')]
        if sec_frame:
            print('скорость: %.2f-%.2f с на кадр трассировки; оценка на все 6 записей '
                  'в одним воркером: %.0f мин (медиана %.2f с/кадр x ~2200 кадров '
                  '+ ход-профиль ~0.12 с/кадр x 2488)'
                  % (min(sec_frame), max(sec_frame),
                     (float(sum(sec_frame)) / len(sec_frame) * 2200
                      + 0.12 * 2488) / 60.0,
                     sorted(sec_frame)[len(sec_frame) // 2]), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
