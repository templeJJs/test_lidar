"""Тесты геометрии сцены: стены/пол на измеренных местах, трассировка попадает."""
import os
import sys
import unittest

import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim import scene  # noqa: E402


class TestStraightTunnel(unittest.TestCase):
    def setUp(self):
        self.p = scene.SceneParams(seed=1, kind='straight', section='wide',
                                   length_m=60.0, grade_pct=2.34, tracks=2)
        self.s = scene.build_scene(self.p)

    def test_mesh_is_built(self):
        self.assertIsInstance(self.s.mesh, o3d.t.geometry.TriangleMesh)
        self.assertGreater(int(self.s.mesh.vertex.positions.shape[0]), 100)

    def test_casts_a_ray_into_the_floor_at_expected_height(self):
        rc = self.s.raycasting()
        origin = np.array([[0.0, -10.0, 2.0]], dtype=np.float32)
        direction = np.array([[0.0, 0.0, -1.0]], dtype=np.float32)
        rays = np.concatenate([origin, direction], axis=1)
        res = rc.cast_rays(o3d.core.Tensor(rays))
        hit_z = float(res['t_hit'].numpy()[0]) * -1.0 + 2.0
        # пол: a + b*y, при a=-1.82 (сенсор 1.82 м над полом) в y=-10 уровень ~-2.05
        self.assertAlmostEqual(hit_z, aux_floor(self.p, -10.0), delta=0.05)

    def test_side_rays_hit_the_walls_at_measured_x(self):
        rc = self.s.raycasting()
        for sign, expect in ((-1, -2.62), (1, 6.62)):
            rays = np.array([[0.0, -10.0, 0.5, float(sign), 0.0, 0.0]], dtype=np.float32)
            res = rc.cast_rays(o3d.core.Tensor(rays))
            hit = float(res['t_hit'].numpy()[0]) * sign
            self.assertAlmostEqual(hit, expect, delta=0.15)

    def test_sensor_pose_follows_the_section(self):
        self.assertAlmostEqual(self.s.sensor_pose['z'], 1.82, delta=0.05)
        narrow = scene.build_scene(scene.SceneParams(seed=1, kind='straight',
                                                     section='narrow', length_m=60.0))
        self.assertLess(narrow.sensor_pose['z'], 1.82)


def aux_floor(p: scene.SceneParams, y: float) -> float:
    return scene.floor_z(p, y)


if __name__ == '__main__':
    unittest.main()