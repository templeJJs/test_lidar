"""Тесты случайного туннеля: повороты, наклоны и воспроизводимость по зерну.

Требование цели: генерируемый туннель должен иметь случайные повороты и наклоны
поверхности в разные стороны, но оставаться похожим на подземный. Поэтому все
случайные величины берутся из диапазонов, замеренных по нашим шести записям
(`sim/dataset_facts.json`), а не выдумываются:

  * ширина туннеля   -- 3.6 .. 9.2 м (узкая и широкая семьи)
  * уклон пола       -- 0.48 .. 1.94 %
  * высота сенсора   -- 1.36 .. 1.86 м над полом
  * дрейф оси        -- 0.15 .. 0.9 м на видимом участке (это амплитуда поворота)

Эти тесты описывают, как сцена должна себя вести; до реализации кривых они падают.
"""
import os
import sys
import unittest

import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json  # noqa: E402

from sim import scene  # noqa: E402

FACTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     'sim', 'dataset_facts.json')


def facts_ranges() -> dict:
    with open(FACTS, encoding='utf-8') as fh:
        return json.load(fh)['generator_ranges']


class TestRandomTunnel(unittest.TestCase):
    def test_kind_curve_bends_the_tunnel_axis(self):
        p = scene.SceneParams(seed=4242, kind='curve', section='wide', length_m=80.0)
        s = scene.build_scene(p)
        centres = [scene.centre_x(p, y) for y in (-30.0, -15.0, 0.0)]
        drift = max(centres) - min(centres)
        lo, hi = facts_ranges()['centre_drift_m']
        self.assertGreater(drift, 0.05, 'поворот должен смещать ось')
        self.assertLessEqual(drift, max(hi * 3.0, 2.0),
                             f'дрейф оси {drift:.2f} м выходит за замеренный диапазон {lo}..{hi} м')
        self.assertIsInstance(s.mesh, o3d.t.geometry.TriangleMesh)

    def test_straight_tunnel_keeps_the_axis_where_it_was(self):
        p = scene.SceneParams(seed=1, kind='straight', section='wide', length_m=80.0)
        scene.build_scene(p)
        self.assertAlmostEqual(scene.centre_x(p, -30.0), scene.centre_x(p, 0.0), places=6)

    def test_walls_follow_the_bent_axis(self):
        p = scene.SceneParams(seed=7, kind='curve', section='narrow', length_m=60.0)
        s = scene.build_scene(p)
        rc = s.raycasting()
        for y in (-30.0, -5.0):
            cx = float(scene.centre_x(s.params, y))
            # ширину берём у САМОЙ сцены: в случайном туннеле она не равна дефолту
            half = abs(s.params.walls_x[0] - s.params.walls_x[1]) / 2.0
            rays = np.array([[cx, y, 0.5, -1.0, 0.0, 0.0],
                             [cx, y, 0.5, 1.0, 0.0, 0.0]], dtype=np.float32)
            res = rc.cast_rays(o3d.core.Tensor(rays))
            hits = res['t_hit'].numpy()
            self.assertTrue(np.isfinite(hits).all(), f'луч в стену на y={y} не попал')
            self.assertAlmostEqual(float(hits[0]), half, delta=0.25)
            self.assertAlmostEqual(float(hits[1]), half, delta=0.25)

    def test_grade_and_width_are_drawn_from_measured_ranges(self):
        rng = facts_ranges()
        grades, widths = [], []
        for seed in range(1, 21):
            p = scene.SceneParams(seed=seed, kind='curve', length_m=60.0).resolved()
            grades.append(abs(p.grade_pct))
            walls = scene.WALL_X_M[p.section]
            widths.append(abs(walls[1] - walls[0]))
        self.assertGreaterEqual(min(grades), rng['grade_pct'][0] - 1e-6)
        self.assertLessEqual(max(grades), rng['grade_pct'][1] + 1e-6)
        lo, hi = rng['tunnel_width_m']
        self.assertGreaterEqual(min(widths), lo - 1e-6)
        self.assertLessEqual(max(widths), hi + 0.15,
                             'ширина может отличаться от замера на полшага, но не больше')

    def test_seed_reproduces_the_same_geometry(self):
        a = scene.build_scene(scene.SceneParams(seed=99, kind='curve', length_m=60.0))
        b = scene.build_scene(scene.SceneParams(seed=99, kind='curve', length_m=60.0))
        c = scene.build_scene(scene.SceneParams(seed=100, kind='curve', length_m=60.0))
        pa = a.mesh.vertex.positions.numpy()
        pb = b.mesh.vertex.positions.numpy()
        pc = c.mesh.vertex.positions.numpy()
        np.testing.assert_allclose(pa, pb, atol=1e-9, err_msg='одно зерно -- одна геометрия')
        self.assertFalse(np.allclose(pa, pc), 'разные зёрна должны давать разную геометрию')


class TestFactsProfiles(unittest.TestCase):
    def test_scenes_are_built_from_all_measured_bags(self):
        """Режим section='facts' берёт профиль из замеров по всем записям."""
        from sim import scene as sc
        profiles = sc.fact_profiles()
        self.assertGreaterEqual(len(profiles), 5, 'профилей должно быть по числу записей')

        widths, grades, heights = [], [], []
        for seed in range(1, 31):
            p = sc.SceneParams(seed=seed, kind='curve', section='facts',
                               length_m=60.0).resolved()
            widths.append(abs(p.walls_x[1] - p.walls_x[0]))
            grades.append(p.grade_pct)
            heights.append(p.sensor_z)

        measured = [abs(pr['walls_x'][1] - pr['walls_x'][0]) for pr in profiles]
        self.assertGreaterEqual(min(widths), min(measured) * 0.8,
                                'ширина не должна выпадать из замеренной семьи')
        self.assertLessEqual(max(widths), max(measured) * 1.2)
        # разные зёрна берут РАЗНЫЕ записи: у наших шести туннелей ширина 3.6…9.2 м,
        # и округление до метра оставило бы всего три ведра (4, 5 и 9 м), поэтому
        # различие меряем парой «профиль + ширина»
        chosen = {(p_i.fact_bag, round(abs(p_i.walls_x[1] - p_i.walls_x[0]), 2))
                  for p_i in (sc.SceneParams(seed=s_i, kind='curve', section='facts',
                                             length_m=60.0).resolved()
                              for s_i in range(1, 31))}
        self.assertGreaterEqual(len({b for b, _ in chosen}), 5,
                                'за 30 зёрен должны выпасть почти все шесть записей')
        self.assertGreater(len(chosen), 5,
                           'сцены должны получаться разными, а не одной на все зёрна')
        lo, hi = sc.GRADE_PCT_RANGE
        for g in grades:
            self.assertLessEqual(abs(g), max(hi * 1.5, 3.0))
        self.assertGreaterEqual(min(heights), sc.SENSOR_Z_RANGE[0] - 0.2)
        self.assertLessEqual(max(heights), sc.SENSOR_Z_RANGE[1] + 0.2)

    def test_profile_can_be_pinned_to_one_recording(self):
        """`fact_bag` пришпиливает профиль: сцена повторяет запись ТОЧНО.

        Без этой возможности сверять сцену с записью нечем (стены уходят за
        допуск 0.10 м), а генератор всё равно берёт профиль случайно.
        """
        from sim import scene as sc
        profiles = {pr['bag']: pr for pr in sc.fact_profiles()}
        for name, prof in profiles.items():
            p = sc.SceneParams(seed=1, kind='curve', section='facts', length_m=60.0,
                               fact_bag=name).resolved()
            self.assertEqual(p.fact_bag, name)
            self.assertEqual(tuple(p.walls_x), tuple(prof['walls_x']),
                             'стены пришпиленного профиля -- ровно замеренные')
            self.assertEqual(p.grade_pct, prof['grade_pct'])
            self.assertEqual(p.sensor_z, prof['sensor_z'])

    def test_unknown_profile_is_reported(self):
        from sim import scene as sc
        with self.assertRaises(ValueError):
            sc.SceneParams(seed=1, kind='curve', section='facts',
                           fact_bag='нет_такой_записи').resolved()


if __name__ == '__main__':
    unittest.main()