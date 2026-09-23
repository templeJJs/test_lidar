"""Чтение разметки `labels/`: облако кадра + объекты + маски, по любой паре (запись, кадр).

Формат `labels/` писался `labeling.save` (вьюер, режим «Разметка»), но его НИКТО
не читал: ни читателя, ни экспорта «кадр+метки», ни прогона детектора по
разметке не было. Этот модуль -- читатель. Два формата (см. `labeling.FORMAT_INFO`):

* версия 1 (карточка без `format`): `points` в .npz -- трассировка ТОЛЬКО
  опорного кадра, тень фона -- число `counts.points_removed`, маски нет;
* версия 2: в .npz рядом с точками лежит пер-кадровый GT -- ключи лучей
  (`gt_ray_keys`), дальности (`gt_t_hit`) и кратность (`gt_mult`) по кадрам
  (`frames`, границы -- `gt_offsets`). По нему восстанавливаются и точки
  объекта, и тень на КАЖДОМ кадре из `card['frames']`.

API
---
    reader = LabelsReader('labels', 'for_hackathon')
    reader.rows                                  # строки index.jsonl
    obj = reader.object('doubleT_obstacle-0001') # LabeledObject
    obj.center_at(15)                            # [x, y, z] | None
    obj.points_at(xyz, ring, 15)                 # (N, 3) точки в системе кадра
    obj.shadow_mask(xyz, ring, 15)               # bool (M,) | None (v1)
    scene = reader.frame('doubleT_obstacle', 15) # всё сразу (см. FrameGT ниже)

    objects, hz = labels_io.load_record('labels', 'doubleT_obstacle')
    # тела в форме POST /label/save -- раунд-трип: открыть свою же разметку
    # (GET /label/load в web_viewer.py) и продолжить править её в панели

`scene = reader.frame(record, frame)` возвращает dict:
    record, frame                -- что просили;
    xyz, intensity, ring         -- облако кадра из .db3 (bag_reader.BagFrames);
    objects                      -- список объектов, размеченных НА ЭТОМ кадре:
        id, record, class, pose, pose_scene, format_version,
        center                   -- позиция объекта в системе ЭТОГО кадра;
        instance                 -- словарь instance из карточки (может быть None,
                                    если кадр вне card['frames']);
        points (N, 3)            -- точки объекта в системе кадра: v2 -- traced
                                    (пер-кадровый GT), v1 -- локальные точки из
                                    .npz, повёрнутые и перенесённые в центр;
        reflectance (N,) | None  -- интенсивность лучей объекта (v2) или опорного
                                    кадра (v1);
        rays_hit, n_points, points_removed -- счётчики трассировки кадра (v2);
        source                   -- 'per-frame-trace' (v2) | 'ref-frame-transform' (v1);
    shadow_mask (M,)             -- union маска точек кадра, закрытых объектами
                                    (None, если ни у одного объекта её нет);
    background_mask (M,)         -- ~shadow_mask: точки, которые остаются фоном.

Позиция объекта на кадре (`center_at`) берётся из `instances[]` карточки, а для
кадра ВНЕ `frames` восстанавливается ПРАВИЛОМ из карточки -- тем же расчётом,
что писал `save` (`labeling._instance_center`): траектория интерполируется по
времени (`trajectory.rule`) и к ней добавляется ход сенсора -- ИНТЕГРАЛОМ
профиля хода между кадрами (`track.rule` / `rule_profile`, `motion_between`),
а не линейной формулой v*(f-ref).

Формула перевода локальных точек в систему кадра (v1 и «points» в .npz):
`points @ R.T + center`, где `R = Rx(roll) @ Ry(pitch) @ Rz(yaw)` из instance
(та же матрица, которой поворачивали меш). Проверена кодом: совпадает с трассировкой
`save` с точностью до float32 (тесты/test_labels_io.py).

Облака читаются через `bag_reader.BagFrames` с кэшем на запись: запись держится
открытой, пока живёт читатель (метод `close()`).
"""
from __future__ import annotations

import json
import os

import numpy as np

import labeling
from bag_reader import BagFrames, find_db3
from validation import synth

DEFAULT_LABELS_DIR = 'labels'
DEFAULT_BAGS_DIR = 'for_hackathon'


def _root():
    return os.path.dirname(os.path.abspath(labeling.__file__))


def read_index(labels_dir=DEFAULT_LABELS_DIR):
    """Строки index.jsonl (пустой список, если файла нет)."""
    path = os.path.join(_root(), labels_dir) if not os.path.isabs(labels_dir) else labels_dir
    rows = []
    index_path = os.path.join(path, 'index.jsonl')
    if os.path.isfile(index_path):
        with open(index_path, encoding='utf-8') as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except ValueError:
                        continue
    return rows


def read_card(labels_dir, record, obj_id):
    """Карточка объекта (dict) или None, если файла нет."""
    path = os.path.join(_root(), labels_dir) if not os.path.isabs(labels_dir) else labels_dir
    card_path = os.path.join(path, record, f'{obj_id}.json')
    if not os.path.isfile(card_path):
        return None
    with open(card_path, encoding='utf-8') as fh:
        return json.load(fh)


def _npz_keys(z):
    """Ключи .npz (NpzFile или пустой dict, если файла нет)."""
    if hasattr(z, 'files'):
        return list(z.files)
    if hasattr(z, 'keys'):
        return list(z.keys())
    return []


def _motion_from_card(card):
    """Профиль хода из карточки -- dict для `labeling.motion_between`.

    Карточка хранит профиль округлённым до 1 мм (`motion_profile_m_per_frame`),
    поэтому центры на кадрах вне `frames` восстанавливаются с точностью до
    округления профиля (доли миллиметра на кадр).
    """
    track = card.get('track') or {}
    prof = track.get('motion_profile_m_per_frame')
    if not prof:
        return None
    return {
        'profile': list(prof),
        'profile_step': int(track.get('motion_profile_step') or 1),
        'm_per_frame': (track.get('measured_m_per_frame')
                        if track.get('measured_m_per_frame') is not None
                        else track.get('speed_m_per_frame')),
    }


def _traj_for_rule(card):
    """Траектория из карточки для `_instance_center` (None, если её нет)."""
    traj = card.get('trajectory')
    if not isinstance(traj, dict) or 'start' not in traj or 'end' not in traj:
        return None
    return traj


class LabeledObject:
    """Один объект разметки: строка индекса, карточка и .npz.

    Ленивый: карточка и .npz читаются при первом обращении, поэтому `LabelsReader`
    может держать все объекты, а память -- только у востребованных.
    """

    def __init__(self, row, labels_dir=DEFAULT_LABELS_DIR):
        self.row = dict(row)
        self.labels_dir = labels_dir
        self.id = str(row['id'])
        self.record = str(row['record'])
        self.obj_class = str(row.get('class') or 'other')
        self.frames = [int(f) for f in (row.get('frames') or [])]
        self._card = None
        self._npz = None

    # ------------------------------------------------------------ данные

    @property
    def card(self):
        if self._card is None:
            self._card = read_card(self.labels_dir, self.record, self.id) or {}
        return self._card

    @property
    def npz(self):
        if self._npz is None:
            path = os.path.join(_root(), self.labels_dir) if not os.path.isabs(self.labels_dir) \
                else self.labels_dir
            npz_path = os.path.join(path, self.record, f'{self.id}.npz')
            if os.path.isfile(npz_path):
                self._npz = np.load(npz_path, allow_pickle=False)
            else:
                self._npz = {}
        return self._npz

    @property
    def format_version(self):
        """Версия схемы: `format` карточки (нет -- 1, старые карточки)."""
        got = self.card.get('format')
        if isinstance(got, dict):
            return int(got.get('version') or 1)
        if isinstance(got, (int, float)):
            return int(got)
        return int(self.row.get('format') or 1)

    @property
    def points(self):
        """Локальные точки опорного кадра (N, 3) float64 (пустые, если .npz нет)."""
        z = self.npz
        if 'points' not in _npz_keys(z):
            return np.zeros((0, 3), dtype=np.float64)
        return np.asarray(z['points'], dtype=np.float64).reshape(-1, 3)

    @property
    def reflectance(self):
        z = self.npz
        if 'reflectance' not in _npz_keys(z):
            return None
        return np.asarray(z['reflectance'], dtype=np.float64).reshape(-1)

    def instance(self, frame):
        """Instance карточки для кадра (или None, если кадр вне frames)."""
        for inst in self.card.get('instances') or []:
            if int(inst.get('frame', -1)) == int(frame):
                return inst
        return None

    # ----------------------------------------------------- позиция на кадре

    def ref_frame(self):
        """Опорный кадр объекта (track.reference_frame или первый из frames)."""
        track = self.card.get('track') or {}
        if track.get('reference_frame') is not None:
            return int(track['reference_frame'])
        return self.frames[0] if self.frames else 0

    def center_at(self, frame):
        """Центр объекта на кадре (система лидара ЭТОГО кадра) или None.

        Кадр из `frames` -- центр из `instances[]` (как записал `save`). Кадр вне
        `frames` -- тот же расчёт по правилу карточки: траектория по времени +
        ход сенсора ИНТЕГРАЛОМ профиля (`labeling._instance_center`). Правило
        верно и для кадров из `frames` -- расхождение только от округления
        профиля в карточке (мм-доли на кадр).
        """
        inst = self.instance(frame)
        if inst is not None and inst.get('center'):
            return [float(v) for v in inst['center']]
        card = self.card
        track = card.get('track') or {}
        ref = self.ref_frame()
        inst_ref = self.instance(ref)
        center = (inst_ref or {}).get('center') or (card.get('instances') or [{}])[0].get('center')
        if not center:
            return None
        speed = float(track.get('speed_m_per_frame') or 0.0)
        hz = float(track.get('hz') or labeling.HZ_DEFAULT)
        motion = _motion_from_card(card)
        traj = _traj_for_rule(card)
        size = card.get('size') or [0.6, 1.0, 1.0]
        got = labeling._instance_center([float(v) for v in center], ref, int(frame),
                                        speed, traj, hz=hz, size=size, motion=motion)
        return [float(v) for v in got]

    # ----------------------------------------------- пер-кадровый GT (v2)

    def has_frame_gt(self):
        """Есть ли пер-кадровый GT (формат 2) с кадрами, покрывающими `frames`."""
        z = self.npz
        if not all(k in _npz_keys(z) for k in ('frames', 'gt_offsets', 'gt_ray_keys',
                                               'gt_t_hit', 'gt_mult')):
            return False
        frames = np.asarray(z['frames']).reshape(-1)
        return set(int(f) for f in frames) >= set(self.frames)

    def _frame_gt_slice(self, frame):
        z = self.npz
        frames = np.asarray(z['frames']).reshape(-1)
        where = np.flatnonzero(frames == int(frame))
        if where.size != 1:
            return None
        i = int(where[0])
        offsets = np.asarray(z['gt_offsets']).reshape(-1)
        lo, hi = int(offsets[i]), int(offsets[i + 1])
        return {
            'ray_keys': np.asarray(z['gt_ray_keys']).reshape(-1)[lo:hi],
            't_hit': np.asarray(z['gt_t_hit']).reshape(-1)[lo:hi],
            'mult': np.asarray(z['gt_mult']).reshape(-1)[lo:hi],
        }

    def frame_gt(self, xyz, ring, frame, intensity=None):
        """Пер-кадровый GT на кадре: точки объекта, отражение и МАСКА тени.

        Восстановление по кадру (то же, что считала трассировка `save`):
        представитель луча -- ближайший возврат кадра с этим ключом; точка
        объекта = направление представителя * `t_hit`, повторённая `mult` раз
        (кратность кадра); тень -- все точки кадра, чей ключ луча совпал с
        попавшим в объект. Требует формат 2; для формата 1 возвращает None.

        Ответ: dict `points` (N, 3), `reflectance` (N,) | None, `shadow_mask`
        (M,) bool, `rays_hit`, `n_points`, `points_removed`, `ray_keys`.
        """
        gt = self._frame_gt_slice(frame)
        if gt is None:
            return None
        xyz = np.asarray(xyz, dtype=np.float64)
        keys_all = synth.ray_keys(xyz, None if ring is None
                                  else np.asarray(ring, dtype=np.int64))
        hit_keys = np.asarray(gt['ray_keys'], dtype=np.int64)
        t_hit = np.asarray(gt['t_hit'], dtype=np.float64).reshape(-1)
        mult = np.asarray(gt['mult'], dtype=np.int64).reshape(-1)
        shadow = np.isin(keys_all, hit_keys) if hit_keys.size else np.zeros(xyz.shape[0], bool)
        out = {
            'ray_keys': hit_keys,
            'shadow_mask': shadow,
            'points_removed': int(np.count_nonzero(shadow)),
            'rays_hit': int(hit_keys.size),
            'n_points': int(np.sum(mult)),
            'points': np.zeros((0, 3), dtype=np.float64),
            'reflectance': None,
        }
        if hit_keys.size == 0:
            return out
        dirs, _ranges, _mult, rep = labeling.frame_rays(xyz, ring)
        keys_rep = keys_all[rep]                 # репрезентативные лучи, по ключу
        pos = np.searchsorted(np.sort(keys_rep), hit_keys)
        pos = np.clip(pos, 0, keys_rep.size - 1)
        found = np.sort(keys_rep)[pos] == hit_keys
        if not found.all():
            raise ValueError(f'{self.id}: кадр {frame} не содержит '
                             f'{int(np.count_nonzero(~found))} лучей из GT -- '
                             'облако кадра не совпадает с трассированным')
        order = np.argsort(keys_rep, kind='stable')
        idx = order[pos]
        d = dirs[idx]
        pts = np.repeat(d, mult, axis=0) * np.repeat(t_hit, mult)[:, None]
        out['points'] = pts
        if intensity is not None:
            vals = np.asarray(intensity, dtype=np.float64)[idx]
            out['reflectance'] = np.repeat(vals, mult)
        return out

    # ------------------------------------------------- точки объекта на кадре

    def points_at(self, xyz, ring, frame, intensity=None):
        """Точки объекта в системе кадра: dict `points`, `reflectance`, `source`.

        Формат 2 -- трассировка ЭТОГО кадра (пер-кадровый GT). Формат 1 -- точки
        опорного кадра, повёрнутые и перенесённые в центр кадра (формула из
        `labels/README.md`: `points @ R.T + center`).
        """
        if self.has_frame_gt():
            got = self.frame_gt(xyz, ring, frame, intensity=intensity)
            if got is not None:
                got['source'] = 'per-frame-trace'
                return got
        inst = self.instance(frame) or {}
        yaw = float(inst.get('yaw', 0.0))
        pitch = float(inst.get('pitch', 0.0))
        roll = float(inst.get('roll', 0.0))
        center = self.center_at(frame)
        if center is None:
            return {'points': np.zeros((0, 3)), 'reflectance': None,
                    'source': 'ref-frame-transform'}
        rot = labeling._rotation(yaw, pitch, roll)
        pts = self.points @ rot.T + np.asarray(center, dtype=np.float64)
        return {'points': pts, 'reflectance': self.reflectance,
                'source': 'ref-frame-transform'}

    def shadow_mask(self, xyz, ring, frame):
        """Маска точек кадра, закрытых объектом (bool, M,) или None (формат 1)."""
        if not self.has_frame_gt():
            return None
        got = self.frame_gt(xyz, ring, frame)
        return None if got is None else got['shadow_mask']

    # ------------------------------------------------------ раунд-трип разметки

    def save_body(self):
        """Объект в форме тела POST /label/save (та же, что шлёт web/app.js).

        Раунд-трип разметки: инструмент обязан уметь ОТКРЫТЬ свою же сохранённую
        разметку -- вернуть объекты в панель, чтобы разметчик продолжил работу
        или поправил ошибку. Форма -- РОВНО та, которой клиент сохранял
        (`labelSave`): `id, class, pose, pose_scene, ref_frame, frames, center,
        size, yaw, pitch, roll, z_ref, trajectory, speed_m_per_frame,
        speed_source, automation, pose_time_s`. Источники -- строка индекса,
        карточка и её track-блок: `center` -- центр опорного кадра из
        `instances[]` (то самое значение, из которого `save` его и посчитал),
        `speed_*` -- из `track`, `pose_time_s` -- из `mesh`.

        Дополнительно (клиенту для восстановления панели, `save` их игнорирует):
        `floor_z` -- опора объекта `base_z_m` (ближайшее к `surface.floor_z_m`
        из предложения: сам `save` пол не пишет), `speed_kmh`, `n_points`,
        `points_source`/`format` -- если есть в карточке.

        Старые карточки (формат 1) читаются тем же кодом: у них есть
        `instances[]`, `track` и `mesh`; отсутствующее подставляется значениями
        по умолчанию, маршрут на них не падает.
        """
        row, card = self.row, self.card
        track = card.get('track') or {}
        ref = self.ref_frame()
        inst = self.instance(ref)
        if inst is None:
            inst = (card.get('instances') or [{}])[0]
        center = inst.get('center')
        size = (card.get('size') or inst.get('size')
                or list(labeling.MANUAL_SIZE))
        traj = card.get('trajectory') if isinstance(card.get('trajectory'), dict) else None
        trajectory = None
        if traj and 'start' in traj and 'end' in traj:
            trajectory = {
                'start': [float(v) for v in traj['start']],
                'end': [float(v) for v in traj['end']],
                'seconds': float(traj.get('seconds') or 0.0),
                'source': str(traj.get('source') or 'manual'),
            }
        base_z = card.get('base_z_m')
        out = {
            'id': self.id,
            'class': self.obj_class,
            'pose': str(row.get('pose') or card.get('pose') or 'unknown'),
            'pose_scene': str(row.get('pose_scene') or card.get('pose_scene')
                              or row.get('pose') or card.get('pose') or 'unknown'),
            'ref_frame': int(ref),
            'frames': list(self.frames),
            'center': ([float(v) for v in center]
                       if isinstance(center, (list, tuple)) and len(center) == 3
                       else None),
            'size': [float(v) for v in size],
            'yaw': float(inst.get('yaw', 0.0)),
            'pitch': float(inst.get('pitch', 0.0)),
            'roll': float(inst.get('roll', 0.0)),
            'z_ref': (None if base_z is None else float(base_z)),
            'trajectory': trajectory,
            'speed_m_per_frame': float(track.get('speed_m_per_frame') or 0.0),
            'speed_source': str(track.get('source') or 'measured'),
            'automation': str(row.get('automation') or card.get('automation')
                              or 'manual'),
            'pose_time_s': float((card.get('mesh') or {}).get('pose_time_s') or 0.0),
            # --- дополнительное: панели нужно, save не читает ---
            'floor_z': (None if base_z is None else float(base_z)),
            'speed_kmh': (None if track.get('speed_kmh') is None
                          else float(track.get('speed_kmh'))),
            'n_points': int(card.get('n_points') or row.get('n_points') or 0),
            'format': int(self.format_version),
        }
        if card.get('points_source'):
            out['points_source'] = str(card['points_source'])
        return out


class LabelsReader:
    """Читатель разметки: индекс, карточки, .npz и облака кадров записей.

    Облако кадра -- из `.db3` записи (`bag_reader.BagFrames`, кэш на запись,
    записи ищутся в `bags_dir`). Объекты -- из `labels_dir`. Оба пути
    относительные -- от корня проекта.
    """

    def __init__(self, labels_dir=DEFAULT_LABELS_DIR, bags_dir=DEFAULT_BAGS_DIR):
        self.labels_dir = labels_dir
        self.bags_dir = bags_dir
        self._rows = None
        self._objects = {}
        self._bags = {}

    def close(self):
        """Закрыть открытые записи (облака .db3)."""
        for frames in self._bags.values():
            frames.close()
        self._bags = {}

    # ------------------------------------------------------------- индекс

    @property
    def rows(self):
        if self._rows is None:
            self._rows = read_index(self.labels_dir)
        return self._rows

    def objects(self, record=None):
        """LabeledObject по строкам индекса (все или одной записи).

        Ищется по строке С ЗАПИСЬЮ, а не по id: id бывает уникален только
        внутри записи (генерация датасета даёт каждой записи свои gen-XXXX),
        и поиск по одному id склеивал бы объекты разных записей.
        """
        out = []
        for row in self.rows:
            if record is not None and row.get('record') != record:
                continue
            key = (str(row.get('record')), str(row['id']))
            if key not in self._objects:
                self._objects[key] = LabeledObject(row, self.labels_dir)
            out.append(self._objects[key])
        return out

    def object(self, obj_id, record=None):
        """Объект по id (кэш; KeyError, если id нет в index.jsonl).

        `record` сужает поиск до одной записи: без него берётся первая строка
        с таким id -- при совпадающих id разных записей это чужой объект.
        """
        key = (str(record) if record is not None else None, str(obj_id))
        if key not in self._objects:
            row = next((r for r in self.rows
                        if r.get('id') == obj_id
                        and (record is None or r.get('record') == record)), None)
            if row is None:
                raise KeyError(f'нет объекта {obj_id!r} (запись {record!r}) '
                               'в index.jsonl')
            self._objects[(str(row.get('record')), str(obj_id))] = \
                LabeledObject(row, self.labels_dir)
            key = (str(row.get('record')), str(obj_id))
        return self._objects[key]

    # ------------------------------------------------------------- облако

    def db_path(self, record):
        """.db3 записи (for_hackathon/<record>/<первый .db3>)."""
        root = os.path.join(_root(), self.bags_dir) if not os.path.isabs(self.bags_dir) \
            else self.bags_dir
        return find_db3(os.path.join(root, record))

    def frame_cloud(self, record, frame):
        """Облако кадра: (xyz (N, 3) float64, intensity, ring)."""
        if record not in self._bags:
            self._bags[record] = BagFrames(self.db_path(record), cache_size=2,
                                            prefetch_ahead=0)
        got = self._bags[record][int(frame)]
        xyz = np.asarray(got.xyz, dtype=np.float64)
        inten = (None if got.intensity is None
                 else np.asarray(got.intensity, dtype=np.float64))
        ring = (None if got.ring is None
                else np.asarray(got.ring, dtype=np.int64))
        return xyz, inten, ring

    # ------------------------------------------------------------- сборка

    def frame(self, record, frame, include_unlabeled_frames=True):
        """Кадр + вся разметка на нём (см. докстринг модуля).

        `include_unlabeled_frames=False` -- только объекты, у которых кадр входит
        в их `frames` (кадры разметки). При True добавляются и объекты записи,
        чей кадр ВНЕ их `frames`: позиция восстанавливается правилом карточки,
        точки -- как в `LabeledObject.points_at` (v2 на неразмеченном кадре
        вернёт точки опорного кадра в новом центре).
        """
        xyz, inten, ring = self.frame_cloud(record, frame)
        objects = []
        shadow_union = np.zeros(xyz.shape[0], dtype=bool)
        have_shadow = False
        for obj in self.objects(record):
            inst = obj.instance(frame)
            if inst is None and not include_unlabeled_frames:
                continue
            got = obj.points_at(xyz, ring, frame, intensity=inten)
            mask = obj.shadow_mask(xyz, ring, frame)
            if mask is not None:
                shadow_union |= mask
                have_shadow = True
            objects.append({
                'id': obj.id, 'record': record, 'class': obj.obj_class,
                'pose': obj.row.get('pose'), 'pose_scene': obj.row.get('pose_scene'),
                'format_version': obj.format_version,
                'frame_labeled': inst is not None,
                'center': obj.center_at(frame),
                'instance': inst,
                'points': np.asarray(got['points'], dtype=np.float64),
                'reflectance': got['reflectance'],
                'source': got['source'],
                'rays_hit': (int(got['rays_hit']) if 'rays_hit' in got else None),
                'n_points': int(np.asarray(got['points']).shape[0]),
                'points_removed': (int(mask.sum()) if mask is not None else None),
                'shadow_mask': mask,
            })
        return {
            'record': record, 'frame': int(frame),
            'xyz': xyz, 'intensity': inten, 'ring': ring,
            'objects': objects,
            'shadow_mask': (shadow_union if have_shadow else None),
            'background_mask': (~shadow_union if have_shadow
                                else np.ones(xyz.shape[0], dtype=bool)),
        }


def load_record(labels_dir, record, default_hz=None):
    """Разметка записи для ОТКРЫТИЯ своей же сохранённой работы (GET /label/load).

    Возвращает `(objects, hz)`: `objects` -- список тел в форме POST /label/save
    (см. `LabeledObject.save_body`), ровно по строкам index.jsonl ЭТОЙ записи;
    записи без разметки (или отсутствующий каталог) -- пустой список. `hz` --
    частота из track-блока первой карточки (её же писал `save`), иначе
    `default_hz`.

    Карточка битая (нет центра в `instances[]`) не роняет всю выдачу: объект
    пропускается, остальные отдаются.
    """
    objects = []
    hz = None
    for row in read_index(labels_dir):
        if str(row.get('record')) != str(record):
            continue
        try:
            obj = LabeledObject(row, labels_dir)
            body = obj.save_body()
        except (KeyError, TypeError, ValueError):
            continue
        if body.get('center') is None:
            continue
        got = (obj.card.get('track') or {}).get('hz')
        if hz is None and got:
            hz = float(got)
        objects.append(body)
    if hz is None:
        hz = (float(default_hz) if default_hz is not None else labeling.HZ_DEFAULT)
    return objects, hz
