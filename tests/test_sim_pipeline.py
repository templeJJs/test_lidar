"""Сквозные тесты симулятора: датасет собирается, наш конвейер его читает.

Это главная приёмка (критерии 1-4 спеки): синтетика бесполезна, если наш же код
(`bag_reader`, `zones`, `web_viewer`) на ней не работает, а объект не ловится
там, где поставлен.
"""
import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bag_reader  # noqa: E402
import zones  # noqa: E402
from sim import cli  # noqa: E402


def read_cloud(bag_dir: str) -> np.ndarray:
    fr = bag_reader.BagFrames(bag_reader.find_db3(bag_dir), cache_size=2, prefetch_ahead=0)
    try:
        return np.concatenate([fr[i].xyz[:, :3].astype(np.float64) for i in range(len(fr))])
    finally:
        fr.close()


class TestDatasetBuild(unittest.TestCase):
    def setUp(self):
        self.out = tempfile.mkdtemp(prefix='simds_')

    def tearDown(self):
        shutil.rmtree(self.out, ignore_errors=True)

    def test_dataset_has_bag_manifest_and_obstacle_markup(self):
        res = cli.make_dataset(out_dir=self.out, seed=42, bags=1, frames=3,
                               section='wide', kind='straight', columns=200,
                               obstacles=[cli.ObstacleSpec('person', -20.0, 0.0)])
        bag_dir = res['bags'][0]
        self.assertTrue(os.path.isdir(bag_dir))
        manifest = json.load(open(os.path.join(bag_dir, 'manifest.json'), encoding='utf-8'))
        self.assertEqual(manifest['seed'], 42)
        self.assertEqual(manifest['frames'], 3)
        self.assertEqual(manifest['objects'][0]['class'], 'person')
        self.assertAlmostEqual(manifest['objects'][0]['y_along_m'], -20.0, delta=0.01,
                               msg='разметка должна совпадать с местом установки')
        self.assertTrue(manifest['objects'][0]['in_gabarit'])
        self.assertEqual(len(manifest['scene']['walls_x']), 2)

        fr = bag_reader.BagFrames(bag_reader.find_db3(bag_dir), cache_size=2, prefetch_ahead=0)
        try:
            self.assertEqual(len(fr), 3)
            self.assertIsNotNone(fr[0].ring)
            self.assertIsNotNone(fr[0].intensity)
            self.assertEqual(int(fr[0].ring.max()) + 1, 128)
        finally:
            fr.close()

    def test_generated_frame_looks_like_a_tunnel(self):
        """Пол и стены находятся нашим детектором на местах, заданных сценой."""
        res = cli.make_dataset(out_dir=self.out, seed=7, bags=1, frames=2,
                               section='wide', kind='straight', columns=300,
                               length_m=60.0)
        bag_dir = res['bags'][0]
        manifest = json.load(open(os.path.join(bag_dir, 'manifest.json'), encoding='utf-8'))
        cloud = read_cloud(bag_dir)

        a, b, _rms = zones.fit_floor(cloud, y_min=-40.0, y_max=-2.0)
        sensor_z = manifest['scene']['sensor_pose']['z']
        self.assertAlmostEqual(-a, sensor_z, delta=0.15, msg='пол должен быть под сенсором')
        walls = zones.detect_walls(cloud, zones.ZoneParams(floor_a=a, floor_b=b))
        expect = manifest['scene']['walls_x']
        self.assertEqual(len(walls), 2)
        self.assertAlmostEqual(min(walls), min(expect), delta=0.25)
        self.assertAlmostEqual(max(walls), max(expect), delta=0.25)


class TestObstacleSemantics(unittest.TestCase):
    def setUp(self):
        self.out = tempfile.mkdtemp(prefix='sims_')

    def tearDown(self):
        shutil.rmtree(self.out, ignore_errors=True)

    def test_object_is_found_at_its_own_y(self):
        res = cli.make_dataset(out_dir=self.out, seed=5, bags=1, frames=2,
                               section='wide', kind='straight', columns=300,
                               length_m=60.0,
                               obstacles=[cli.ObstacleSpec('person', -20.0, 0.0)])
        bag_dir = res['bags'][0]
        manifest = json.load(open(os.path.join(bag_dir, 'manifest.json'), encoding='utf-8'))
        cloud = read_cloud(bag_dir)
        a, b, _ = zones.fit_floor(cloud, y_min=-40.0, y_max=-2.0)
        walls = zones.detect_walls(cloud, zones.ZoneParams(floor_a=a, floor_b=b))
        # ось пути берём из манифеста: середина между стенами -- НЕ ось пути
        # (в широком туннеле два пути, и сенсор стоит над нашим, а не в центре)
        axis = float(manifest['scene']['track_axis_x'])
        p = zones.ZoneParams(floor_a=a, floor_b=b, axis_x=axis, wall_x=tuple(walls))
        t = zones.tunnel_blocked(cloud, p, -40.0, 40.0)
        spans = t['spans']
        hit = [s for s in spans if s[0] - 0.5 <= -20.0 <= s[1] + 0.5]
        self.assertTrue(hit, f'человек на y=-20 обязан попасть в коридор, участки {spans}')
        self.assertLess(sum(b_ - a_ for a_, b_ in spans), 8.0,
                        'пустой туннель должен быть почти свободен')

        object_frames = manifest['objects'][0]['frames_visible']
        self.assertTrue(object_frames, 'объект обязан давать возвраты хотя бы в одном кадре')


if __name__ == '__main__':
    unittest.main()