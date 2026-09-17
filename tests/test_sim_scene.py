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
        # пол: a + b*y, при a=-1.82 (сенсор 1.82 м над полом) в y=-10 уровень ~-2.05.
        # Допуск 0.10 м -- потому что ложе намеренно шероховатое (FLOOR_ROUGHNESS_M),
        # как балласт в реальных записях; гладкая плита давала бы 1e-7.
        self.assertAlmostEqual(hit_z, aux_floor(self.p, -10.0), delta=0.10)

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


class TestTrack(unittest.TestCase):
    def setUp(self):
        self.p = scene.SceneParams(seed=1, kind='straight', section='wide', length_m=40.0)
        self.s = scene.build_scene(self.p)
        self.rc = self.s.raycasting()

    def test_two_running_rails_at_gauge(self):
        lx, rx = self.s.track['rails_x']
        self.assertAlmostEqual(rx - lx, 1.520, delta=0.01)

    def test_rail_head_is_16_cm_above_the_floor(self):
        y = -10.0
        x = self.s.track['rails_x'][0]
        rays = np.array([[x, y, 1.0, 0.0, 0.0, -1.0]], dtype=np.float32)
        res = self.rc.cast_rays(o3d.core.Tensor(rays))
        hit_z = 1.0 - float(res['t_hit'].numpy()[0])
        self.assertAlmostEqual(hit_z, scene.floor_z(self.p, y) + 0.16, delta=0.03)

    def test_profile_comes_from_track_geometry_module(self):
        import track_geometry
        verts, sources = track_geometry.rail_profile_mm(73.0)
        self.assertGreater(len(verts), 8)
        self.assertIn('gost R65', sources)


class TestVaultShape(unittest.TestCase):
    """Свод повторяет замеренный профиль своей записи, а не плоскость."""

    Y = -15.0

    def _ceiling(self, s, x: float) -> float:
        rc = s.raycasting()
        rays = np.array([[x, self.Y, -1.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        hit = float(rc.cast_rays(o3d.core.Tensor(rays))['t_hit'].numpy()[0])
        return -1.0 + hit

    def test_arch_recording_gives_an_arched_vault(self):
        """У `roundT_doubleT` свод арочный: у стен ниже, над осью выше.

        Профиль замера: гребень 5.03 м над осью, плечи 4.65…4.77 м. Плоский свод
        сцену бы не отличил -- а именно это и было расхождением.
        """
        p = scene.SceneParams(seed=1, kind='straight', section='facts',
                              fact_bag='roundT_doubleT', length_m=60.0).resolved()
        s = scene.build_scene(p)
        self.assertGreater(len(p.vault_profile or ()), 2, 'профиль свода не подставился')
        crown = self._ceiling(s, 0.0) - scene.floor_z(p, self.Y)
        shoulder = self._ceiling(s, 2.0) - scene.floor_z(p, self.Y)
        self.assertAlmostEqual(crown, 5.03, delta=0.15, msg='гребень по замеру')
        self.assertAlmostEqual(shoulder, 4.65, delta=0.25, msg='плечо по замеру')
        self.assertGreater(crown - shoulder, 0.15, 'арка должна быть ниже у стен')

    def test_sloped_recording_gives_a_sloped_vault(self):
        """У двухпутной записи свод -- наклонная плита: 4.90 м слева, 4.32 справа."""
        p = scene.SceneParams(seed=1, kind='straight', section='facts',
                              fact_bag='doubleT_obstacle', length_m=60.0).resolved()
        s = scene.build_scene(p)
        left = self._ceiling(s, -2.0) - scene.floor_z(p, self.Y)
        right = self._ceiling(s, 5.0) - scene.floor_z(p, self.Y)
        self.assertAlmostEqual(left, 4.88, delta=0.20)
        self.assertAlmostEqual(right, 4.30, delta=0.25)
        self.assertGreater(left - right, 0.3, 'свод записи наклонён')

    def test_vault_follows_the_tunnel_in_a_turn(self):
        """Свод едет вместе с туннелем: профиль привязан к оси, а не к мировому X."""
        p = scene.SceneParams(seed=4, kind='curve', section='facts',
                              fact_bag='roundT_doubleT', length_m=120.0).resolved()
        s = scene.build_scene(p)
        for y in (-40.0, -5.0):
            axis = float(scene.centre_x(p, y))
            crown = self._ceiling(s, axis) - scene.floor_z(p, y)
            self.assertAlmostEqual(crown, p.vault_m, delta=0.25,
                                   msg=f'гребень должен быть над осью и на повороте (y={y})')

    def test_family_without_profile_has_a_flat_vault(self):
        p = scene.SceneParams(seed=2, kind='straight', section='wide',
                              length_m=60.0, grade_pct=2.34).resolved()
        s = scene.build_scene(p)
        self.assertFalse(p.vault_profile, 'у семейства профиль свода не задан')
        self.assertAlmostEqual(p.vault_m, 4.84, delta=1e-6)
        self.assertAlmostEqual(self._ceiling(s, -1.0), self._ceiling(s, 5.0), delta=0.05)




    def test_wall_has_a_duct_at_the_measured_level(self):
            """За нитью стены на высоте полки есть ниша: луч уходит наружу на выступ.

            Так сцена перестала быть «ровной стеной от ложа до свода»: у записей за
            нитью стены лежит поверхность на 1.41…1.58 м, и она уходит наружу на
            0.40…0.76 м, а стена выше стоит на месте.
            """
            p = scene.SceneParams(seed=1, kind='straight', section='facts',
                                  fact_bag='roundT_doubleT', length_m=60.0).resolved()
            s = scene.build_scene(p)
            rc = s.raycasting()
            wall_x = max(p.walls_x)
            base = float(scene.floor_z(p, self.Y))
            for dz, expect in ((0.05, wall_x + p.ledge_out_m),      # внутри ниши
                               (0.30, wall_x)):                      # выше ниши
                z = base + p.ledge_m + dz
                rays = np.array([[0.0, self.Y, z, 1.0, 0.0, 0.0]], dtype=np.float32)
                hit = float(rc.cast_rays(o3d.core.Tensor(rays))['t_hit'].numpy()[0])
                self.assertAlmostEqual(hit, expect, delta=0.12,
                                       msg=f'при z = {z:.2f} стена должна быть на {expect:.2f}')



def aux_floor(p: scene.SceneParams, y: float) -> float:
    return scene.floor_z(p, y)


if __name__ == '__main__':
    unittest.main()