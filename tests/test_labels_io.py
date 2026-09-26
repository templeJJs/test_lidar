"""Тесты читателя labels/ и round-trip «save -> чтение».

Round-trip: объект ставится рабочим путём инструмента (propose -> preview ->
save) во временный каталог .scratch, читается `labels_io` и сверяется с
НЕЗАВИСИМОЙ трассировкой тех же кадров: точки, маска тени, центры, счётчики.
Отдельно -- обратная совместимость: из карточки выбрасываются поля версии 2
(format, пер-кадровый GT), и читатель обязан работать по правилам версии 1
(точки опорного кадра по формуле README, тени нет).

Раунд-трип ОТКРЫТИЯ (TestLabelLoad): save -> labels_io.load_record (это и есть
GET /label/load) -> прочитанное тело снова в save -- карточки, строки индекса и
.nzp совпадают (diff == 0), форма тела -- та же, что шлёт labelSave в клиенте.
"""
import json
import os
import shutil
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import labeling  # noqa: E402
import labels_io  # noqa: E402
from validation import dataset_eval  # noqa: E402
from validation import synth  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL_DB = os.path.join(ROOT, 'for_hackathon', 'doubleT_obstacle',
                       'doubleT_obstacle_0.db3')
DEMO_CARD = os.path.join(ROOT, 'labels', 'doubleT_obstacle',
                         'doubleT_obstacle-0001.json')
OUT_DIR = os.path.join(ROOT, '.scratch', 'labels_io_test')
HZ = 9.856791946774774


def _bag_frames():
    from bag_reader import BagFrames  # noqa: PLC0415

    return BagFrames(REAL_DB, cache_size=8, prefetch_ahead=0)


def _xyz_at(frames):
    def xyz_at(f):
        got = frames[int(f)]
        return (np.asarray(got.xyz, dtype=np.float64),
                np.asarray(got.intensity, dtype=np.float64),
                np.asarray(got.ring, dtype=np.int64))

    return xyz_at


class TestRoundTrip(unittest.TestCase):
    """save -> labels_io: точки, маски, центры, счётчики, формат."""

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(REAL_DB):
            raise unittest.SkipTest('bag not present')
        shutil.rmtree(OUT_DIR, ignore_errors=True)
        cls.frames = _bag_frames()
        cls.xyz_at = _xyz_at(cls.frames)
        cls.model = labeling._model_for(REAL_DB)
        xyz = cls.xyz_at(13)[0]

        # Объект 1: человек стоит на месте реального человека записи (демо-объект).
        card = None
        if os.path.isfile(DEMO_CARD):
            with open(DEMO_CARD, encoding='utf-8') as fh:
                card = json.load(fh)
        if card:
            center = [float(v) for v in card['instances'][0]['center']]
            size = [float(v) for v in card['size']]
            yaw = float(card['instances'][0]['yaw'])
        else:
            center, size, yaw = [-0.74, -55.74, -1.5], [0.55, 1.75, 0.55], 0.0
        cls.obj_standing = {
            'id': 'rt-standing', 'class': 'person', 'pose': 'standing',
            'ref_frame': 13, 'center': center, 'size': size,
            'yaw': yaw, 'pitch': 0.0, 'roll': 0.0,
            'frames': [13, 14, 15], 'automation': 'manual',
            'speed_m_per_frame': 0.0, 'speed_source': 'measured',
        }

        # Объект 2: идущий с траекторией на 20 м (propose -- опора и поворот).
        pf = labeling.path_frame(xyz, cls.model)
        y = -20.0
        click = [float(pf.axis_x(np.asarray([y]))[0]) + 0.8, y, -1.0]
        got = labeling.propose(xyz, click, model=cls.model)
        z_ref = float(got['z_ref'])
        size2 = [0.55, 1.75, 0.55]
        center2 = [click[0], y, z_ref + 0.5 * size2[1]]
        cls.obj_walking = {
            'id': 'rt-walking', 'class': 'person', 'pose': 'walking',
            'ref_frame': 13, 'center': center2, 'size': size2,
            'yaw': float(got['yaw']), 'pitch': float(got['pitch']), 'roll': 0.0,
            'frames': [13, 14, 15], 'automation': 'manual',
            'speed_m_per_frame': 0.0, 'speed_source': 'measured',
            'z_ref': z_ref,
            'trajectory': {'start': [center2[0], y, z_ref],
                           'end': [center2[0], y + 2.0, z_ref],
                           'seconds': 2.0, 'source': 'manual'},
        }
        report = labeling.save('doubleT_obstacle',
                               [cls.obj_standing, cls.obj_walking],
                               cls.xyz_at, out_dir=OUT_DIR, hz=HZ, motion=None)
        assert not report['errors'], report['errors']
        cls.report = report
        cls.reader = labels_io.LabelsReader(OUT_DIR, 'for_hackathon')

    @classmethod
    def tearDownClass(cls):
        cls.reader.close()
        cls.frames.close()
        shutil.rmtree(OUT_DIR, ignore_errors=True)

    # ------------------------------------------------------------ формат

    def test_format_fields_everywhere(self):
        self.assertEqual(self.reader.rows[0].get('format'), labeling.FORMAT_VERSION)
        for row in self.reader.rows:
            card = self.reader.object(row['id']).card
            self.assertEqual(card['format']['version'], 2)
            self.assertTrue(card['format']['per_frame_gt'])
        with open(os.path.join(OUT_DIR, 'manifest.json'), encoding='utf-8') as fh:
            manifest = json.load(fh)
        self.assertEqual(manifest['format']['version'], labeling.FORMAT_VERSION)

    def test_reader_version_of_objects(self):
        for obj_id in ('rt-standing', 'rt-walking'):
            obj = self.reader.object(obj_id)
            self.assertEqual(obj.format_version, 2)
            self.assertTrue(obj.has_frame_gt())

    def test_npz_has_gt_arrays(self):
        obj = self.reader.object('rt-standing')
        keys = labels_io._npz_keys(obj.npz)
        for key in ('frames', 'gt_offsets', 'gt_ray_keys', 'gt_t_hit', 'gt_mult'):
            self.assertIn(key, keys)
        self.assertEqual(list(np.asarray(obj.npz['frames'])), [13, 14, 15])
        offsets = np.asarray(obj.npz['gt_offsets'])
        self.assertEqual(offsets.shape, (4,))
        total = int(np.asarray(obj.npz['gt_ray_keys']).size)
        self.assertEqual(int(offsets[-1]), total)

    # --------------------------------------------------- точки и маски

    def test_points_at_ref_frame_equals_npz_transform(self):
        """Формула README: points @ R.T + center -- она же points_at на опорном."""
        obj = self.reader.object('rt-standing')
        xyz, _inten, ring = self.reader.frame_cloud('doubleT_obstacle', 13)
        got = obj.points_at(xyz, ring, 13)
        inst = obj.instance(13)
        rot = labeling._rotation(inst['yaw'], inst['pitch'], inst['roll'])
        manual = obj.points @ rot.T + np.asarray(inst['center'])
        np.testing.assert_allclose(got['points'], manual, atol=1e-6)
        self.assertEqual(got['source'], 'per-frame-trace')

    def test_per_frame_points_match_independent_trace(self):
        """Читатель восстанавливает трассировку save ПОБИТОВО (лучами, не точками)."""
        for obj_id, obj_spec in (('rt-standing', self.obj_standing),
                                 ('rt-walking', self.obj_walking)):
            obj = self.reader.object(obj_id)
            for f in (13, 14, 15):
                xyz, inten, ring = self.reader.frame_cloud('doubleT_obstacle', f)
                gt = obj.frame_gt(xyz, ring, f, intensity=inten)
                t = (f - 13) / HZ
                center_f = labeling._instance_center(
                    obj_spec['center'], 13, f, obj_spec['speed_m_per_frame'],
                    labeling.trajectory_normalize(obj_spec.get('trajectory')),
                    hz=HZ, size=obj_spec['size'], motion=None)
                mesh, _info = labeling.object_geometry(
                    'person', labeling.pose_rule('person', obj_spec['pose'],
                                                 bool(obj_spec.get('trajectory'))),
                    obj_spec['size'], center_f, obj_spec['yaw'],
                    obj_spec['pitch'], obj_spec['roll'],
                    pose_time=t, base_z=obj_spec.get('z_ref'))
                tr = labeling.trace_object(xyz, ring, mesh)
                self.assertEqual(gt['points'].shape, tr['points'].shape,
                                 '%s кадр %d' % (obj_id, f))
                np.testing.assert_allclose(gt['points'], tr['points'], atol=1e-9,
                                           err_msg='%s кадр %d' % (obj_id, f))
                np.testing.assert_array_equal(gt['shadow_mask'], tr['removed'])
                self.assertEqual(gt['rays_hit'], tr['rays_hit'])
                self.assertEqual(gt['points_removed'], tr['points_removed'])
                if gt['reflectance'] is not None:
                    self.assertEqual(gt['reflectance'].shape, gt['points'].shape[0:1])

    def test_instance_counts_match_gt(self):
        obj = self.reader.object('rt-standing')
        xyz, _inten, ring = self.reader.frame_cloud('doubleT_obstacle', 14)
        gt = obj.frame_gt(xyz, ring, 14)
        inst = obj.instance(14)
        self.assertEqual(inst['n_points'], gt['n_points'])
        self.assertEqual(inst['rays_hit'], gt['rays_hit'])
        self.assertEqual(inst['points_removed'], gt['points_removed'])

    def test_shadow_mask_is_subset_of_frame(self):
        obj = self.reader.object('rt-standing')
        xyz, _inten, ring = self.reader.frame_cloud('doubleT_obstacle', 15)
        mask = obj.shadow_mask(xyz, ring, 15)
        self.assertEqual(mask.shape, (xyz.shape[0],))
        self.assertGreater(int(mask.sum()), 0)
        # Тень -- точки ЗА объектом: на каждом луче из GT ближайший возврат кадра
        # дальше поверхности объекта (это и решало видимость при трассировке).
        gt = obj._frame_gt_slice(15)
        dirs, ranges, _mult, rep = labeling.frame_rays(xyz, ring)
        keys_all = synth.ray_keys(xyz, None if ring is None
                                  else np.asarray(ring, dtype=np.int64))
        keys_rep = keys_all[rep]
        order = np.argsort(keys_rep, kind='stable')
        sorted_keys = keys_rep[order]
        hit_keys = np.asarray(gt['ray_keys'], dtype=np.int64)
        t_hit = np.asarray(gt['t_hit'], dtype=np.float64)
        pos = np.searchsorted(sorted_keys, hit_keys)
        rep_ranges = ranges[order[pos]]
        self.assertTrue(np.all(rep_ranges > t_hit - 1e-3),
                        'луч попал в объект -- поверхность ближе возврата кадра')

    # --------------------------------------------------------- центры

    def test_centers_from_instances(self):
        for obj_id in ('rt-standing', 'rt-walking'):
            obj = self.reader.object(obj_id)
            for f in (13, 14, 15):
                inst = obj.instance(f)
                self.assertIsNotNone(inst)
                self.assertEqual(obj.center_at(f), [float(v) for v in inst['center']])

    def test_center_rule_outside_frames(self):
        """Кадр вне frames: позиция -- ПРАВИЛОМ карточки (траектория по времени)."""
        obj = self.reader.object('rt-walking')
        card = obj.card
        traj = card['trajectory']
        # После seconds объект на КОНЕЧНОЙ точке: t = (40-13)/9.86 = 2.74 > 2 c.
        end = np.asarray(traj['end'], dtype=np.float64)
        half = 0.5 * float(card['size'][1])
        got = obj.center_at(40)
        self.assertAlmostEqual(got[0], float(end[0]), places=6)
        self.assertAlmostEqual(got[1], float(end[1]), places=6)
        self.assertAlmostEqual(got[2], float(end[2] + half), places=6)
        # Между кадрами -- интерполяция по времени, а не по номеру кадра.
        mid = obj.center_at(14)
        t = 1.0 / HZ
        frac = min(t / traj['seconds'], 1.0)
        expect_y = traj['start'][1] + (traj['end'][1] - traj['start'][1]) * frac
        self.assertAlmostEqual(mid[1], float(expect_y), places=6)

    # ------------------------------------------------------ сборка кадра

    def test_frame_scene(self):
        scene = self.reader.frame('doubleT_obstacle', 14,
                                  include_unlabeled_frames=False)
        self.assertEqual(scene['record'], 'doubleT_obstacle')
        self.assertEqual(scene['frame'], 14)
        self.assertEqual(len(scene['objects']), 2)
        ids = sorted(o['id'] for o in scene['objects'])
        self.assertEqual(ids, ['rt-standing', 'rt-walking'])
        shadow = scene['shadow_mask']
        self.assertIsNotNone(shadow)
        self.assertEqual(int(scene['background_mask'].sum()),
                         scene['xyz'].shape[0] - int(shadow.sum()))
        total_removed = sum(int(o['points_removed']) for o in scene['objects'])
        self.assertEqual(int(shadow.sum()), total_removed,
                         'тень кадра -- union теней объектов (объекты не пересекаются)')

    def test_frame_scene_excludes_unlabeled(self):
        scene = self.reader.frame('doubleT_obstacle', 40,
                                  include_unlabeled_frames=False)
        self.assertEqual(scene['objects'], [])
        self.assertIsNone(scene['shadow_mask'])
        self.assertTrue(bool(scene['background_mask'].all()))

    # ------------------------------------------------- обратная совместимость

    def test_v1_card_is_read_by_fallback(self):
        """Карточка без полей v2 читается по правилам v1 (формула README, без тени)."""
        # Копия разметки без полей v2 -- старая карточка, не трогаем рабочую.
        v1_dir = os.path.join(ROOT, '.scratch', 'labels_io_v1')
        shutil.rmtree(v1_dir, ignore_errors=True)
        os.makedirs(os.path.join(v1_dir, 'doubleT_obstacle'))
        card_path = os.path.join(v1_dir, 'doubleT_obstacle', 'rt-standing.json')
        npz_path = os.path.join(v1_dir, 'doubleT_obstacle', 'rt-standing.npz')
        index_path = os.path.join(v1_dir, 'index.jsonl')
        card = self.reader.object('rt-standing').card
        card_v1 = {k: v for k, v in card.items() if k != 'format'}
        card_v1['instances'] = [
            {k: v for k, v in inst.items()
             if k not in ('rays_hit', 'n_points', 'points_removed')}
            for inst in card['instances']]
        with open(card_path, 'w', encoding='utf-8') as fh:
            json.dump(card_v1, fh, ensure_ascii=False, indent=1)
        rows = [dict(r) for r in self.reader.rows]
        for row in rows:
            row.pop('format', None)
        with open(index_path, 'w', encoding='utf-8') as fh:
            fh.write(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
        with np.load(os.path.join(OUT_DIR, 'doubleT_obstacle', 'rt-standing.npz')) as z:
            payload = {'points': z['points']}
            if 'reflectance' in labels_io._npz_keys(z):
                payload['reflectance'] = z['reflectance']
        np.savez_compressed(npz_path, **payload)

        reader = labels_io.LabelsReader(v1_dir, 'for_hackathon')
        try:
            obj = reader.object('rt-standing')
            self.assertEqual(obj.format_version, 1)
            self.assertFalse(obj.has_frame_gt())
            xyz, _inten, ring = reader.frame_cloud('doubleT_obstacle', 15)
            got = obj.points_at(xyz, ring, 15)
            self.assertEqual(got['source'], 'ref-frame-transform')
            self.assertEqual(got['points'].shape, obj.points.shape)
            inst = obj.instance(15)
            rot = labeling._rotation(inst['yaw'], inst['pitch'], inst['roll'])
            manual = obj.points @ rot.T + np.asarray(inst['center'])
            np.testing.assert_allclose(got['points'], manual, atol=1e-9)
            self.assertIsNone(obj.shadow_mask(xyz, ring, 15))
        finally:
            reader.close()
            shutil.rmtree(v1_dir, ignore_errors=True)


class TestLabelLoad(unittest.TestCase):
    """Раунд-трип ОТКРЫТИЯ разметки: save -> load_record -> тело /label/save.

    Инструмент обязан уметь открыть свою же сохранённую разметку (GET
    /label/load в web_viewer.py -- тонкая обёртка над labels_io.load_record):
    вернуть объекты в той форме, которой их сохранял клиент, чтобы разметчик
    продолжил работу или поправил ошибку. Проверяется, что прочитанное тело
    годится для ПОВТОРНОГО save без изменений: те же id, frames, карточки
    (кроме времени записи), те же точки в .npz.
    """

    RECORD = 'doubleT_obstacle'
    DIR1 = os.path.join(ROOT, '.scratch', 'label_load_rt1')
    DIR2 = os.path.join(ROOT, '.scratch', 'label_load_rt2')
    V1_DIR = os.path.join(ROOT, '.scratch', 'label_load_v1')

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(REAL_DB):
            raise unittest.SkipTest('bag not present')
        for d in (cls.DIR1, cls.DIR2, cls.V1_DIR):
            shutil.rmtree(d, ignore_errors=True)
        cls.frames = _bag_frames()
        cls.xyz_at = _xyz_at(cls.frames)
        card = None
        if os.path.isfile(DEMO_CARD):
            with open(DEMO_CARD, encoding='utf-8') as fh:
                card = json.load(fh)
        center = ([float(v) for v in card['instances'][0]['center']]
                  if card else [-0.74, -55.74, -1.5])
        size = [float(v) for v in (card['size'] if card else [0.55, 1.75, 0.55])]
        yaw = float(card['instances'][0]['yaw']) if card else 0.0
        # Тело -- то же, что строит labelSave в web/app.js: все ключи, которыми
        # клиент сохраняет объект. Центр у объекта с траекторией -- как его видит
        # сцена на опорном кадре (labelCenterAt): НАЧАЛО траектории + полвысоты.
        z_ref = -2.1
        cls.body = [{
            'id': 'rt-load', 'class': 'person', 'pose': 'standing',
            'pose_scene': 'standing', 'pose_time_s': 0.0,
            'ref_frame': 13, 'frames': [13, 14, 15],
            'center': [center[0], center[1], z_ref + size[1] / 2.0],
            'size': size,
            'yaw': yaw, 'pitch': 0.0, 'roll': 0.0,
            'z_ref': z_ref, 'speed_m_per_frame': 0.0, 'speed_source': 'measured',
            'automation': 'manual',
            'trajectory': {'start': [center[0], center[1], z_ref],
                           'end': [center[0], center[1] + 1.5, z_ref],
                           'seconds': 3.0, 'source': 'manual'},
        }]
        report = labeling.save(cls.RECORD, cls.body, cls.xyz_at,
                               out_dir=cls.DIR1, hz=HZ, motion=None)
        assert not report['errors'], report['errors']
        cls.loaded, cls.hz = labels_io.load_record(cls.DIR1, cls.RECORD,
                                                   default_hz=HZ)
        report2 = labeling.save(cls.RECORD, cls.loaded, cls.xyz_at,
                                out_dir=cls.DIR2, hz=cls.hz, motion=None)
        assert not report2['errors'], report2['errors']

    @classmethod
    def tearDownClass(cls):
        cls.frames.close()
        for d in (cls.DIR1, cls.DIR2, cls.V1_DIR):
            shutil.rmtree(d, ignore_errors=True)

    @staticmethod
    def _card(path):
        with open(path, encoding='utf-8') as fh:
            return json.load(fh)

    def test_load_shape_is_save_body(self):
        """Форма ответа -- РОВНО тело /label/save (те же ключи, что шлёт клиент)."""
        self.assertEqual(len(self.loaded), 1)
        obj = self.loaded[0]
        for key in ('id', 'class', 'pose', 'pose_scene', 'ref_frame', 'frames',
                    'center', 'size', 'yaw', 'pitch', 'roll', 'z_ref',
                    'trajectory', 'speed_m_per_frame', 'speed_source',
                    'automation', 'pose_time_s'):
            self.assertIn(key, obj)
        self.assertEqual(obj['id'], 'rt-load')
        self.assertEqual(obj['class'], 'person')
        self.assertEqual(obj['pose_scene'], 'standing')
        self.assertEqual(obj['ref_frame'], 13)
        self.assertEqual(obj['frames'], [13, 14, 15])
        self.assertEqual(obj['automation'], 'manual')
        self.assertEqual(obj['speed_source'], 'measured')
        self.assertEqual(obj['speed_m_per_frame'], 0.0)
        self.assertEqual(obj['z_ref'], -2.1)
        self.assertEqual(obj['trajectory']['seconds'], 3.0)
        self.assertEqual(obj['trajectory']['start'][1], self.body[0]['center'][1])
        # Дополнительные сведения для панели: пол, скорость, счётчик, версия.
        self.assertEqual(obj['floor_z'], -2.1)
        self.assertEqual(obj['format'], labeling.FORMAT_VERSION)
        # центр -- центр ОПОРНОГО кадра из instances[] карточки (у клиента с
        # траекторией это начало траектории + полвысоты -- тот же расчёт)
        card = self._card(os.path.join(self.DIR1, self.RECORD, 'rt-load.json'))
        self.assertEqual(obj['center'], [float(v) for v
                                         in card['instances'][0]['center']])
        self.assertEqual(obj['center'], self.body[0]['center'])

    def test_hz_from_card_track(self):
        self.assertAlmostEqual(self.hz, HZ, places=6)

    def test_load_other_record_is_empty(self):
        objects, hz = labels_io.load_record(self.DIR1, 'roundT_doubleT',
                                            default_hz=7.0)
        self.assertEqual(objects, [])
        self.assertEqual(hz, 7.0)      # карточек нет -- дефолт вызывающего

    def test_load_missing_dir_is_empty(self):
        objects, hz = labels_io.load_record(
            os.path.join(ROOT, '.scratch', 'label_load_absent'), self.RECORD)
        self.assertEqual(objects, [])
        self.assertAlmostEqual(hz, labeling.HZ_DEFAULT, places=6)

    def test_roundtrip_ids_and_frames(self):
        """Повторный save по прочитанному телу: те же id и frames, дубликата нет."""
        for d in (self.DIR1, self.DIR2):
            objects, _hz = labels_io.load_record(d, self.RECORD)
            self.assertEqual([o['id'] for o in objects], ['rt-load'])
            self.assertEqual(objects[0]['frames'], [13, 14, 15])

    def test_roundtrip_cards_identical(self):
        """Карточки совпадают (кроме времени записи и ревизии) -- diff == 0."""
        skip = ('created', 'revision')
        c1 = self._card(os.path.join(self.DIR1, self.RECORD, 'rt-load.json'))
        c2 = self._card(os.path.join(self.DIR2, self.RECORD, 'rt-load.json'))
        self.assertEqual({k: v for k, v in c1.items() if k not in skip},
                         {k: v for k, v in c2.items() if k not in skip})

    def test_roundtrip_rows_identical(self):
        def rows(d):
            with open(os.path.join(d, 'index.jsonl'), encoding='utf-8') as fh:
                return [json.loads(line) for line in fh if line.strip()]
        self.assertEqual(rows(self.DIR1), rows(self.DIR2))

    def test_roundtrip_npz_identical(self):
        """Точки и пер-кадровый GT повторного save совпадают с первым."""
        with np.load(os.path.join(self.DIR1, self.RECORD, 'rt-load.npz')) as z1, \
                np.load(os.path.join(self.DIR2, self.RECORD, 'rt-load.npz')) as z2:
            for key in ('points', 'reflectance', 'frames', 'gt_offsets',
                        'gt_ray_keys', 'gt_t_hit', 'gt_mult'):
                np.testing.assert_array_equal(z1[key], z2[key],
                                              err_msg=key)

    def test_load_v1_card_does_not_crash(self):
        """Старая карточка (формат 1, без v2-полей) читается -- «отдать что есть»."""
        os.makedirs(os.path.join(self.V1_DIR, self.RECORD))
        card = self._card(os.path.join(self.DIR1, self.RECORD, 'rt-load.json'))
        card_v1 = {k: v for k, v in card.items() if k != 'format'}
        card_v1['instances'] = [
            {k: v for k, v in inst.items()
             if k not in ('rays_hit', 'n_points', 'points_removed')}
            for inst in card['instances']]
        with open(os.path.join(self.V1_DIR, self.RECORD, 'rt-load.json'), 'w',
                  encoding='utf-8') as fh:
            json.dump(card_v1, fh, ensure_ascii=False, indent=1)
        with open(os.path.join(self.V1_DIR, 'index.jsonl'), 'w',
                  encoding='utf-8') as fh:
            row = {'id': 'rt-load', 'record': self.RECORD, 'class': 'person',
                   'pose': 'standing', 'frames': [13, 14, 15], 'n_points': 1}
            fh.write(json.dumps(row, ensure_ascii=False) + '\n')
        objects, hz = labels_io.load_record(self.V1_DIR, self.RECORD)
        self.assertEqual(len(objects), 1)
        obj = objects[0]
        self.assertEqual(obj['id'], 'rt-load')
        self.assertEqual(obj['center'], [float(v) for v
                                         in card_v1['instances'][0]['center']])
        self.assertEqual(obj['pose_scene'], 'standing')   # из карточки
        self.assertEqual(obj['format'], 1)                # карточка без `format`
        self.assertIsNone(obj.get('points_source'))
        self.assertAlmostEqual(hz, HZ, places=6)          # track.hz карточки



    """Сопоставление и вёдра дальности -- на синтетике, без записей."""

    class _Frame:
        def axis_x(self, y):
            return np.zeros_like(np.asarray(y, dtype=np.float64))

    def _obstacle(self, y, u, conf=0.9, dist=None):
        return {'y_m': y, 'u_m': u, 'confidence': conf,
                'distance_m': dist if dist is not None else abs(y)}

    def _scene(self, *objects):
        return {'objects': [dict(o, frame_labeled=True) for o in objects],
                'shadow_mask': None}

    def test_bucket_edges(self):
        self.assertEqual(dataset_eval.bucket_of(5.0), '0-10')
        self.assertEqual(dataset_eval.bucket_of(10.0), '0-10')
        self.assertEqual(dataset_eval.bucket_of(10.001), '10-20')
        self.assertEqual(dataset_eval.bucket_of(20.0), '10-20')
        self.assertEqual(dataset_eval.bucket_of(55.7), '40+')
        self.assertIsNone(dataset_eval.bucket_of(0.0))
        self.assertIsNone(dataset_eval.bucket_of(None))

    def test_match_within_tolerance(self):
        scene = self._scene({'id': 'a', 'class': 'person', 'center': [0.0, -25.0, -1.0],
                             'instance': {'size': [0.55, 1.75, 0.55]}})
        obstacles = [self._obstacle(-25.3, 0.2)]
        pairs, unmatched = dataset_eval.match_objects(scene, obstacles, self._Frame())
        self.assertEqual(len(pairs), 1)
        self.assertEqual(unmatched, [])

    def test_match_rejects_far_detection(self):
        scene = self._scene({'id': 'a', 'class': 'person', 'center': [0.0, -25.0, -1.0],
                             'instance': {'size': [0.55, 1.75, 0.55]}})
        obstacles = [self._obstacle(-40.0, 0.2)]
        pairs, unmatched = dataset_eval.match_objects(scene, obstacles, self._Frame())
        self.assertEqual(pairs, [])
        self.assertEqual(len(unmatched), 1)

    def test_greedy_matching_by_confidence(self):
        scene = self._scene(
            {'id': 'a', 'class': 'person', 'center': [0.0, -25.0, -1.0],
             'instance': {'size': [0.55, 1.75, 0.55]}},
            {'id': 'b', 'class': 'person', 'center': [0.0, -26.0, -1.0],
             'instance': {'size': [0.55, 1.75, 0.55]}},
        )
        obstacles = [self._obstacle(-25.4, 0.0, conf=0.5),
                     self._obstacle(-25.6, 0.0, conf=0.9)]
        pairs, unmatched = dataset_eval.match_objects(scene, obstacles, self._Frame())
        self.assertEqual(len(pairs), 2)
        by_id = {t['obj']['id']: d for t, d in pairs}
        # confident (-25.6) ближе к объекту b (-26.0), слабый (-25.4) -- к a.
        self.assertAlmostEqual(by_id['b']['y_m'], -25.6, places=9)
        self.assertAlmostEqual(by_id['a']['y_m'], -25.4, places=9)
        self.assertEqual(unmatched, [])

    def test_report_rows_ordered_and_rated(self):
        stats = {}
        dataset_eval._bump(stats, 'person', '10-20', gt=2, matched=1, det=1, det_ok=1)
        dataset_eval._bump(stats, 'all', '10-20', gt=2, matched=1, det=1, det_ok=1)
        dataset_eval._bump(stats, 'all', '0-10', gt=4, matched=4, det=5, det_ok=4)
        rows = dataset_eval._report_rows(stats)
        self.assertEqual(rows[0]['class'], 'all')
        self.assertEqual([r['bucket'] for r in rows], ['0-10', '10-20', '10-20'])
        self.assertAlmostEqual(rows[1]['recall'], 0.5)
        self.assertAlmostEqual(rows[0]['precision'], 0.8)


class TestEvalOnLabels(unittest.TestCase):
    """Прогон eval по round-trip-разметке: структурные проверки результата."""

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(REAL_DB):
            raise unittest.SkipTest('bag not present')
        shutil.rmtree(OUT_DIR, ignore_errors=True)
        cls.frames = _bag_frames()
        xyz_at = _xyz_at(cls.frames)
        card = None
        if os.path.isfile(DEMO_CARD):
            with open(DEMO_CARD, encoding='utf-8') as fh:
                card = json.load(fh)
        center = ([float(v) for v in card['instances'][0]['center']]
                  if card else [-0.74, -55.74, -1.5])
        size = [float(v) for v in card['size']] if card else [0.55, 1.75, 0.55]
        obj = {
            'id': 'rt-standing', 'class': 'person', 'pose': 'standing',
            'ref_frame': 13, 'center': center, 'size': size,
            'yaw': float(card['instances'][0]['yaw']) if card else 0.0,
            'pitch': 0.0, 'roll': 0.0,
            'frames': [13, 14, 15], 'automation': 'manual',
            'speed_m_per_frame': 0.0, 'speed_source': 'measured',
        }
        report = labeling.save('doubleT_obstacle', [obj], xyz_at,
                               out_dir=OUT_DIR, hz=HZ, motion=None)
        assert not report['errors'], report['errors']
        cls.rows, cls.meta = dataset_eval.eval_labels(
            OUT_DIR, 'for_hackathon', verbose=False)

    @classmethod
    def tearDownClass(cls):
        cls.frames.close()
        shutil.rmtree(OUT_DIR, ignore_errors=True)

    def test_scenes_and_gt_counts(self):
        self.assertEqual(self.meta['scenes'], 3)
        self.assertEqual(self.meta['objects'], 1)
        gt_total = sum(r['gt'] for r in self.rows if r['class'] != 'all')
        self.assertEqual(gt_total, 3)

    def test_recall_within_unit(self):
        for row in self.rows:
            if row['gt']:
                self.assertGreaterEqual(row['recall'], 0.0)
                self.assertLessEqual(row['recall'], 1.0)
            if row['det']:
                self.assertGreaterEqual(row['precision'], 0.0)
                self.assertLessEqual(row['precision'], 1.0)
        # Объект стоит на месте реального человека записи: детектор обязан его
        # видеть (смотри validation/out/dataset_eval.csv по labels/).
        person = [r for r in self.rows if r['class'] == 'person' and r['gt']]
        self.assertTrue(person)
        for row in person:
            self.assertEqual(row['matched'], row['gt'])


class TestDemoCard(unittest.TestCase):
    """Демо-карточка labels/ после перегенерации: формат v2, старое сохранено."""

    @classmethod
    def setUpClass(cls):
        if not (os.path.isfile(DEMO_CARD) and os.path.exists(REAL_DB)):
            raise unittest.SkipTest('labels/ или bag отсутствуют')
        cls.reader = labels_io.LabelsReader('labels', 'for_hackathon')

    @classmethod
    def tearDownClass(cls):
        cls.reader.close()

    def test_card_fields(self):
        card = self.reader.object('doubleT_obstacle-0001').card
        self.assertEqual(card['format']['version'], 2)
        self.assertEqual(card['frames'], [13, 14, 15, 16, 17])
        self.assertEqual(card['n_points'], 104)
        self.assertEqual(card['pose_scene'], 'standing')
        self.assertEqual(card['mesh']['source'], 'primitive')
        self.assertEqual(card['counts']['rays_hit'], 52)
        for inst in card['instances']:
            for key in ('rays_hit', 'n_points', 'points_removed'):
                self.assertIn(key, inst)

    def test_frame_gt_reads(self):
        obj = self.reader.object('doubleT_obstacle-0001')
        self.assertTrue(obj.has_frame_gt())
        xyz, _inten, ring = self.reader.frame_cloud('doubleT_obstacle', 15)
        gt = obj.frame_gt(xyz, ring, 15)
        inst = obj.instance(15)
        self.assertEqual(gt['n_points'], inst['n_points'])
        self.assertEqual(gt['points_removed'], inst['points_removed'])
        self.assertEqual(int(gt['shadow_mask'].sum()), inst['points_removed'])

    def test_ray_key_semantics(self):
        """Ключ луча из GT -- тот же, что считает synth.ray_keys по кадру."""
        obj = self.reader.object('doubleT_obstacle-0001')
        xyz, _inten, ring = self.reader.frame_cloud('doubleT_obstacle', 13)
        gt = obj.frame_gt(xyz, ring, 13)
        keys = synth.ray_keys(xyz, ring)
        self.assertTrue(np.isin(gt['ray_keys'], keys).all())


class TestMutualOcclusion(unittest.TestCase):
    """Взаимная окклюзия объектов: задний закрыт передним, как в физике лидара.

    Синтетический кадр (свой, записи не нужны): кольца 0..127 с elevation
    -20..+5°, азимут -8..+8° шагом 0.1°, фон на 40 м, по 2 возврата на луч --
    как у Hesai 128. Два человека-примитива на оси: 5 м и 10 м за ним.
    """

    OUT = os.path.join(ROOT, '.scratch', 'labels_io_occlusion')
    SIZE = [0.55, 1.75, 0.55]
    FLOOR_Z = -1.5

    @classmethod
    def setUpClass(cls):
        pts, rings = [], []
        elev = np.radians(np.linspace(-20.0, 5.0, 128))
        az = np.radians(np.arange(-80, 81) * 0.1)
        for k, e in enumerate(elev):
            d = np.column_stack([np.sin(az) * np.cos(e),
                                 -np.cos(az) * np.cos(e),
                                 np.full(az.shape, np.sin(e))])
            p = d * 40.0
            pts.append(p)
            pts.append(p)                  # 2 возврата на луч, как в записях
            rings.append(np.full((2 * az.size,), k, dtype=np.int64))
        cls.xyz = np.vstack(pts)
        cls.ring = np.concatenate(rings)
        cls.inten = np.full(cls.xyz.shape[0], 100.0)
        cls.xyz_at = staticmethod(lambda f: (cls.xyz, cls.inten, cls.ring))

    @classmethod
    def mesh_at(cls, y, lateral=0.0):
        center = [lateral, y, cls.FLOOR_Z + 0.5 * cls.SIZE[1]]
        mesh, _info = labeling.object_geometry(
            'person', 'standing', cls.SIZE, center, mesh_source='primitive')
        return mesh

    def test_rear_object_is_occluded_by_front(self):
        """Задний (10 м) за передним (5 м) по оси теряет почти весь силуэт."""
        front = self.mesh_at(-5.0)
        rear = self.mesh_at(-10.0)
        alone = labeling.trace_object(self.xyz, self.ring, rear)
        occl = labeling.trace_object(self.xyz, self.ring, rear,
                                     occluders=[front])
        self.assertGreater(alone['n_points'], 100)
        self.assertLess(occl['n_points'], alone['n_points'])
        # Передний вдвое ближе и той же ширины: его силуэт накрывает задний
        # целиком, остаются единичные лучи на кромке (замерено 24 из 2992).
        lost = 1.0 - occl['n_points'] / float(alone['n_points'])
        self.assertGreater(lost, 0.90, 'задний объект закрыт передним')

    def test_front_object_unaffected_by_rear(self):
        """Передний не теряет ничего от заднего: чужой меш ДАЛЬШЕ -- не окклюдер."""
        front = self.mesh_at(-5.0)
        rear = self.mesh_at(-10.0)
        alone = labeling.trace_object(self.xyz, self.ring, front)
        occl = labeling.trace_object(self.xyz, self.ring, front,
                                     occluders=[rear])
        np.testing.assert_allclose(occl['points'], alone['points'], atol=1e-9)
        np.testing.assert_array_equal(occl['removed'], alone['removed'])
        self.assertEqual(occl['rays_hit'], alone['rays_hit'])

    def test_occluder_off_axis_changes_nothing(self):
        """Окклюдер рядом с линией визирования не режет задний объект."""
        rear = self.mesh_at(-10.0)
        aside = self.mesh_at(-5.0, lateral=1.5)
        alone = labeling.trace_object(self.xyz, self.ring, rear)
        occl = labeling.trace_object(self.xyz, self.ring, rear,
                                     occluders=[aside])
        np.testing.assert_allclose(occl['points'], alone['points'], atol=1e-9)
        self.assertEqual(occl['n_points'], alone['n_points'])

    def test_save_applies_mutual_occlusion(self):
        """save() двух объектов на одном кадре: задний записан уже усечённым."""
        shutil.rmtree(self.OUT, ignore_errors=True)

        def obj(oid, y):
            return {'id': oid, 'class': 'person', 'pose': 'standing',
                    'ref_frame': 0, 'frames': [0],
                    'center': [0.0, y, self.FLOOR_Z + 0.5 * self.SIZE[1]],
                    'size': self.SIZE, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0,
                    'z_ref': self.FLOOR_Z, 'speed_m_per_frame': 0.0,
                    'speed_source': 'manual', 'automation': 'manual',
                    'mesh_source': 'primitive'}

        try:
            pair = labeling.save('synth_occ', [obj('front', -5.0),
                                               obj('rear', -10.0)],
                                 self.xyz_at, out_dir=self.OUT, hz=10.0,
                                 motion=None)
            self.assertEqual(pair['errors'], [])
            reader = labels_io.LabelsReader(self.OUT, 'for_hackathon')
            try:
                n_pair = reader.object('rear').card['n_points']
            finally:
                reader.close()

            shutil.rmtree(self.OUT, ignore_errors=True)
            solo = labeling.save('synth_occ', [obj('rear', -10.0)],
                                 self.xyz_at, out_dir=self.OUT, hz=10.0,
                                 motion=None)
            self.assertEqual(solo['errors'], [])
            reader = labels_io.LabelsReader(self.OUT, 'for_hackathon')
            try:
                n_solo = reader.object('rear').card['n_points']
            finally:
                reader.close()
            self.assertGreater(n_solo, 100)
            self.assertLess(n_pair, 0.1 * n_solo,
                            'в паре задний объект должен потерять силуэт')
        finally:
            shutil.rmtree(self.OUT, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
