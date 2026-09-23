"""Тесты билдера датасета (tools/build_dataset.py) и measured-карточек.

ЧТО проверяется:
* детерминизм раскладки: один seed -- одна и та же раскладка (чистая функция
  plan_record, без данных записей);
* кадры реального человека (10..66 doubleT_obstacle) не попадают ни в раскладку,
  ни в негативы; негативы не пересекаются с размеченными кадрами;
* сплит train/val ПО ЗАПИСАМ: записи не пересекаются, val содержит
  doubleT_obstacle (якорь реального человека);
* round-trip points_source='measured': save пишет GT по ИЗМЕРЕННЫМ точкам,
  labels_io читает его как формат 2 и восстанавливает РОВНО те же точки
  (лучи: направление представителя * дальность точки);
* measured без полного покрытия frames -- ошибка в отчёте save, карточки нет.
"""
import json
import os
import shutil
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, 'tools')
for path in (ROOT, TOOLS):
    if path not in sys.path:
        sys.path.insert(0, path)

import labeling  # noqa: E402
import labels_io  # noqa: E402
import build_dataset  # noqa: E402

REAL_DB = os.path.join(ROOT, 'for_hackathon', 'doubleT_obstacle',
                       'doubleT_obstacle_0.db3')
OUT_DIR = os.path.join(ROOT, '.scratch', 'build_dataset_test')
# «Кластер» для measured-тестов: точки ЛЕВОЙ стены у оси на кадрах без человека
# (кадры 100..102 doubleT_obstacle; у оси на u=-2.2 точки есть, на u=+0.8 -- нет).
MEAS_U = -2.2
MEAS_Y = -20.0


def _xyz_at(frames):
    def xyz_at(f):
        got = frames[int(f)]
        return (np.asarray(got.xyz, dtype=np.float64),
                np.asarray(got.intensity, dtype=np.float64),
                np.asarray(got.ring, dtype=np.int64))

    return xyz_at


def _wall_cluster(xyz, model, y_c=MEAS_Y, u_c=MEAS_U):
    """Индексы точек стены: облако -> индексы (правила те же, что у якоря)."""
    pf = labeling.path_frame(xyz, model)
    y, u, h = pf.terms(xyz)
    return np.flatnonzero((np.abs(y - y_c) < 0.6) & (np.abs(u - u_c) < 0.6)
                          & (h > 0.25) & (h < 1.9))


class TestPlan(unittest.TestCase):
    """Раскладка: детерминизм и инварианты (без данных записей)."""

    def test_deterministic_same_seed(self):
        a = build_dataset.plan_record('roundT_doubleT', 252, 42, 40)
        b = build_dataset.plan_record('roundT_doubleT', 252, 42, 40)
        self.assertEqual(json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True))

    def test_different_seed_changes_layout(self):
        a = build_dataset.plan_record('roundT_doubleT', 252, 1, 40)
        b = build_dataset.plan_record('roundT_doubleT', 252, 2, 40)
        self.assertNotEqual(json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True))

    def test_plan_invariants(self):
        for record, total in (('doubleT_obstacle', 201), ('squareT_platform_squareT_switch', 877)):
            specs = build_dataset.plan_record(record, total, 7, 120)
            self.assertGreaterEqual(len(specs), 100)
            ids = [s['id'] for s in specs]
            self.assertEqual(len(ids), len(set(ids)))
            for s in specs:
                self.assertIn(s['class'], labeling.CLASSES)
                self.assertIn(s['distance_m'], build_dataset.DISTANCES)
                self.assertTrue(-1.2 <= s['u_m'] <= 1.2)
                self.assertTrue(3 <= len(s['frames']) <= 8)
                self.assertEqual(s['frames'], list(range(s['ref_frame'],
                                                         s['ref_frame'] + len(s['frames']))))
                self.assertTrue(0 <= s['ref_frame'])
                self.assertLessEqual(s['frames'][-1], total - 1)
                if s['class'] == 'person':
                    if s['trajectory']:
                        self.assertIn(s['pose'], build_dataset.POSES_MOVING)
                        t = s['trajectory']
                        self.assertTrue(2.0 <= t['seconds'] <= 8.0)
                        self.assertTrue(0.5 <= t['speed_mps'] <= 2.0)
                        self.assertIn(t['direction'], ('along', 'across'))
                    else:
                        self.assertIn(s['pose'], build_dataset.POSES_STATIC)
                else:
                    self.assertIsNone(s['trajectory'])
                    self.assertIn(s['equipment_kind'], build_dataset.EQUIPMENT_SIZE)

    def test_plan_avoids_real_person_frames(self):
        specs = build_dataset.plan_record('doubleT_obstacle', 201, 3, 80)
        forbidden = set(build_dataset.REAL_FRAMES)
        for s in specs:
            self.assertFalse(forbidden.intersection(s['frames']),
                             'кадры %s пересекают 10..66' % s['frames'])


class TestSplit(unittest.TestCase):
    """Сплит train/val по записям: без пересечений, val содержит якорь."""

    def test_split_disjoint_by_records(self):
        split = build_dataset.split_records()
        self.assertEqual(len(split['val']), 2)
        self.assertEqual(sorted(split['val'] + split['train']),
                         sorted(build_dataset.RECORDS))
        self.assertFalse(set(split['val']) & set(split['train']))
        self.assertIn('doubleT_obstacle', split['val'])

    def test_split_subset_of_requested_records(self):
        records = ['doubleT_obstacle', 'doubleT_platform', 'roundT_doubleT']
        split = build_dataset.split_records(records)
        self.assertEqual(sorted(split['val'] + split['train']), sorted(records))
        self.assertEqual(split['val'], ['doubleT_obstacle', 'roundT_doubleT'])


class TestNegatives(unittest.TestCase):
    """Негативы не пересекаются с разметкой и запретными кадрами."""

    def test_negative_candidates_disjoint(self):
        used = {20, 21, 22, 100, 101}
        forbidden = set(build_dataset.REAL_FRAMES)
        cands = build_dataset.negative_candidates(201, used, forbidden, want=60)
        self.assertTrue(cands)
        self.assertFalse(set(cands) & used)
        self.assertFalse(set(cands) & forbidden)
        self.assertTrue(all(0 <= f < 201 for f in cands))
        # Кандидаты покрывают запись равномерно, а не первые 60 кадров.
        self.assertGreater(cands[-1], 100)

    def test_negative_candidates_all_used_returns_empty(self):
        self.assertEqual(build_dataset.negative_candidates(10, set(range(10))), [])


class TestMeasuredRoundTrip(unittest.TestCase):
    """points_source='measured': save -> labels_io, точки восстановлены точно."""

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(REAL_DB):
            raise unittest.SkipTest('bag not present')
        shutil.rmtree(OUT_DIR, ignore_errors=True)
        from bag_reader import BagFrames  # noqa: PLC0415

        cls.frames = BagFrames(REAL_DB, cache_size=8, prefetch_ahead=0)
        cls.xyz_at = _xyz_at(cls.frames)
        cls.model = labeling._model_for(REAL_DB)
        # «Кластер» -- точки стены у оси на кадрах без человека (кадр 100):
        # правила те же, что у якоря (облако -> индексы), источник неважен.
        cls.measured = {}
        for f in (100, 101, 102):
            xyz, _inten, _ring = cls.xyz_at(f)
            idx = _wall_cluster(xyz, cls.model)
            assert idx.size >= labeling.MEASURED_MIN_POINTS, idx.size
            cls.measured[int(f)] = idx.tolist()
        xyz_ref = cls.xyz_at(100)[0]
        tr0 = labeling._measured_frame_gt(xyz_ref, cls.xyz_at(100)[2],
                                          np.asarray(cls.measured[100]), cls.model)
        cls.obj = {
            'id': 'meas-0001', 'class': 'person', 'pose': 'standing',
            'ref_frame': 100, 'center': [float(v) for v in tr0['center']],
            'size': [float(v) for v in tr0['box']['size']],
            'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0,
            'frames': [100, 101, 102], 'automation': 'cluster',
            'speed_m_per_frame': 0.0, 'speed_source': 'measured',
            'points_source': 'measured', 'measured': cls.measured,
            'z_ref': float(tr0['box']['base_z']), 'db_path': REAL_DB,
        }
        cls.report = labeling.save('doubleT_obstacle', [cls.obj], cls.xyz_at,
                                   out_dir=OUT_DIR, hz=10.0, motion=None)
        assert not cls.report['errors'], cls.report['errors']
        cls.reader = labels_io.LabelsReader(OUT_DIR, 'for_hackathon')

    @classmethod
    def tearDownClass(cls):
        cls.reader.close()
        cls.frames.close()
        shutil.rmtree(OUT_DIR, ignore_errors=True)

    def test_card_and_row_mark_measured(self):
        obj = self.reader.object('meas-0001')
        self.assertEqual(obj.card['points_source'], 'measured')
        self.assertEqual(obj.card['mesh']['source'], 'measured')
        self.assertEqual(obj.card['primitive']['source'], 'measured')
        self.assertEqual(obj.row['points_source'], 'measured')
        self.assertEqual(obj.format_version, 2)
        self.assertTrue(obj.has_frame_gt())

    def test_frame_gt_points_equal_measured(self):
        """Лучи восстановлены из кадра: ровно измеренные точки (не трассировка)."""
        obj = self.reader.object('meas-0001')
        for f in (100, 101, 102):
            xyz, inten, ring = self.reader.frame_cloud('doubleT_obstacle', f)
            gt = obj.frame_gt(xyz, ring, f, intensity=inten)
            want = xyz[np.asarray(self.measured[f])]
            self.assertEqual(gt['points'].shape, want.shape, 'кадр %d' % f)
            np.testing.assert_allclose(gt['points'], want, atol=1e-4,
                                       err_msg='кадр %d' % f)
            self.assertEqual(gt['n_points'], want.shape[0])
            self.assertEqual(gt['reflectance'].shape, want.shape[0:1])
            # Тень -- все точки кадра на лучах кластера (включая сами точки).
            self.assertEqual(gt['points_removed'], int(gt['shadow_mask'].sum()))
            self.assertGreaterEqual(gt['points_removed'], want.shape[0])

    def test_centers_are_per_frame_measured(self):
        """Центры на кадрах -- боксы по кластерам, а не правило проекции."""
        obj = self.reader.object('meas-0001')
        for f in (100, 101, 102):
            xyz, _inten, ring = self.reader.frame_cloud('doubleT_obstacle', f)
            tr = labeling._measured_frame_gt(xyz, ring,
                                             np.asarray(self.measured[f]),
                                             self.model)
            inst = obj.instance(f)
            self.assertIsNotNone(inst)
            np.testing.assert_allclose(inst['center'], tr['center'], atol=1e-6)

    def test_npz_gt_is_per_point(self):
        """Ключи лучей в GT -- ПО ТОЧКЕ (mult=1): двойные возвраты одного луча."""
        obj = self.reader.object('meas-0001')
        z = obj.npz
        keys = np.asarray(z['gt_ray_keys']).reshape(-1)
        mult = np.asarray(z['gt_mult']).reshape(-1)
        self.assertEqual(int(mult.sum()), keys.size)
        self.assertTrue((mult == 1).all())
        total = int(np.asarray(z['gt_offsets'])[-1])
        self.assertEqual(total, keys.size)

    def test_scene_assembles_measured_object(self):
        scene = self.reader.frame('doubleT_obstacle', 101,
                                  include_unlabeled_frames=False)
        self.assertEqual(len(scene['objects']), 1)
        obj = scene['objects'][0]
        self.assertEqual(obj['source'], 'per-frame-trace')
        want = np.asarray(scene['xyz'])[np.asarray(self.measured[101])]
        np.testing.assert_allclose(obj['points'], want, atol=1e-4)
        self.assertIsNotNone(scene['shadow_mask'])


class TestMeasuredValidation(unittest.TestCase):
    """Ошибки measured: покрытие кадров, пустые кластеры, чужой points_source."""

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(REAL_DB):
            raise unittest.SkipTest('bag not present')
        shutil.rmtree(OUT_DIR, ignore_errors=True)
        from bag_reader import BagFrames  # noqa: PLC0415

        cls.frames = BagFrames(REAL_DB, cache_size=4, prefetch_ahead=0)
        cls.xyz_at = _xyz_at(cls.frames)

    @classmethod
    def tearDownClass(cls):
        cls.frames.close()
        shutil.rmtree(OUT_DIR, ignore_errors=True)

    def _obj(self, **override):
        obj = {
            'id': 'bad-0001', 'class': 'person', 'pose': 'standing',
            'ref_frame': 100, 'center': [0.0, -20.0, -1.0], 'size': [0.5, 1.7, 0.5],
            'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0, 'frames': [100, 101],
            'automation': 'manual', 'speed_m_per_frame': 0.0,
            'speed_source': 'measured', 'db_path': REAL_DB,
        }
        obj.update(override)
        return obj

    def test_measured_must_cover_frames(self):
        report = labeling.save('doubleT_obstacle',
                               [self._obj(points_source='measured',
                                          measured={100: list(range(20))})],
                               _xyz_at(self.frames), out_dir=OUT_DIR, hz=10.0)
        self.assertEqual(len(report['errors']), 1)
        self.assertIn('кадр', report['errors'][0]['error'])
        self.assertEqual(report['written'], [])
        self.assertFalse(os.path.exists(os.path.join(OUT_DIR, 'doubleT_obstacle',
                                                     'bad-0001.json')))

    def test_measured_rejects_tiny_cluster(self):
        report = labeling.save('doubleT_obstacle',
                               [self._obj(points_source='measured',
                                          measured={100: [1, 2, 3], 101: [1, 2, 3]})],
                               _xyz_at(self.frames), out_dir=OUT_DIR, hz=10.0)
        self.assertEqual(len(report['errors']), 1)
        self.assertIn('кластер', report['errors'][0]['error'])

    def test_unknown_points_source_rejected(self):
        report = labeling.save('doubleT_obstacle',
                               [self._obj(points_source='magic')],
                               _xyz_at(self.frames), out_dir=OUT_DIR, hz=10.0)
        self.assertEqual(len(report['errors']), 1)
        self.assertIn('points_source', report['errors'][0]['error'])

    def test_trace_objects_have_no_measured_fields(self):
        """Обычная (трассированная) карточка не получает полей measured."""
        obj = self._obj(points_source='trace')
        report = labeling.save('doubleT_obstacle', [obj], _xyz_at(self.frames),
                               out_dir=OUT_DIR, hz=10.0)
        self.assertEqual(report['errors'], [])
        card = json.load(open(os.path.join(OUT_DIR, 'doubleT_obstacle',
                                           'bad-0001.json'), encoding='utf-8'))
        self.assertNotIn('points_source', card)
        row = next(r for r in labeling._read_index(
            os.path.join(OUT_DIR, 'index.jsonl')) if r['id'] == 'bad-0001')
        self.assertNotIn('points_source', row)


class TestReaderAmbiguousIds(unittest.TestCase):
    """id, повторяющийся в записях: читатель обязан различать их по записи.

    Регрессия: LabelsReader.object(id) искал только по id, и объекты датасета
    (у каждой записи свои gen-XXXX) склеивались -- eval сверял лучи ЧУЖОЙ записи.
    """

    def test_same_id_different_records(self):
        labels = os.path.join(OUT_DIR, 'ambiguous')
        shutil.rmtree(labels, ignore_errors=True)
        os.makedirs(os.path.join(labels, 'rec_a'))
        os.makedirs(os.path.join(labels, 'rec_b'))
        rows = [
            {'id': 'gen-0001', 'record': 'rec_a', 'class': 'person',
             'pose': 'standing', 'frames': [10, 11], 'n_points': 5},
            {'id': 'gen-0001', 'record': 'rec_b', 'class': 'equipment',
             'pose': 'unknown', 'frames': [20, 21], 'n_points': 7},
        ]
        with open(os.path.join(labels, 'index.jsonl'), 'w', encoding='utf-8') as fh:
            fh.write(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
        try:
            reader = labels_io.LabelsReader(labels, 'for_hackathon')
            a = reader.objects('rec_a')
            b = reader.objects('rec_b')
            self.assertEqual([o.id for o in a], ['gen-0001'])
            self.assertEqual(a[0].frames, [10, 11])
            self.assertEqual(a[0].obj_class, 'person')
            self.assertEqual(b[0].frames, [20, 21])
            self.assertEqual(b[0].obj_class, 'equipment')
            self.assertIsNot(a[0], b[0])
            got = reader.object('gen-0001', 'rec_b')
            self.assertEqual(got.frames, [20, 21])
        finally:
            reader.close()
            shutil.rmtree(labels, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
