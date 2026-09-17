"""Тесты библиотеки объектов: меши, материалы, флаг «в габарите»."""
import os
import sys
import unittest

import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim import objects, scene  # noqa: E402


class TestObjectLibrary(unittest.TestCase):
    def test_person_and_suitcase_are_parameterized(self):
        self.assertIn('person', objects.OBJECT_LIBRARY)
        self.assertIn('suitcase', objects.OBJECT_LIBRARY)
        person = objects.OBJECT_LIBRARY['person'](height_m=1.75)
        self.assertAlmostEqual(person.size_m[2], 1.75, delta=0.01)
        self.assertIsNotNone(person.mesh)
        self.assertGreater(person.reflectivity, 0.0)

    def test_placement_on_the_rail_and_in_the_gauge(self):
        sp = scene.SceneParams(seed=7, kind='straight', section='wide', length_m=60.0)
        s = scene.build_scene(sp)
        specs = [objects.Placement('person', y_along_m=-30.0, lateral_m=0.0,
                                   placement='in_gauge')]
        objs = objects.place_objects(sp, s.track, specs)
        self.assertEqual(len(objs), 1)
        o = objs[0]
        self.assertAlmostEqual(o['y_along_m'], -30.0, delta=0.01)
        self.assertTrue(o['in_gabarit'], 'человек в колее -- внутри габарита')

    def test_object_outside_the_gabarit_is_flagged(self):
        sp = scene.SceneParams(seed=7, kind='straight', section='wide', length_m=60.0)
        s = scene.build_scene(sp)
        specs = [objects.Placement('suitcase', y_along_m=-20.0, lateral_m=3.2,
                                   placement='by_wall')]
        objs = objects.place_objects(sp, s.track, specs)
        self.assertFalse(objs[0]['in_gabarit'], 'чемодан в 3.2 м от пути -- вне габарита')

    def test_object_carries_legacy_meshes_and_an_honest_box(self):
        """`_meshes` -- тот самый служебный ключ, по которому сцена берёт меши объекта.

        Заодно проверяем, что объявленный размер совпадает с реальным боксом меша:
        по `bbox_xyz` считается «виден ли объект в кадре», и врать он не должен.
        """
        for name in ('person', 'suitcase'):
            om = objects.OBJECT_LIBRARY[name]()
            self.assertIsInstance(om.mesh, o3d.geometry.TriangleMesh)
            self.assertFalse(hasattr(om.mesh, 'to_tensor'), 'меш должен быть legacy')
            extent = np.asarray(om.mesh.get_axis_aligned_bounding_box().get_extent())
            self.assertTrue(np.allclose(extent, om.size_m, atol=0.01),
                            f'{name}: меш {extent} не совпал с size_m {om.size_m}')

        sp = scene.SceneParams(seed=7, kind='straight', section='wide', length_m=60.0)
        s = scene.build_scene(sp)
        specs = [objects.Placement('suitcase', y_along_m=-20.0, lateral_m=0.0,
                                   placement='in_gauge')]
        o = objects.place_objects(sp, s.track, specs)[0]
        self.assertEqual(len(o['_meshes']), 1)
        mesh = o['_meshes'][0]
        lo, hi = o['bbox_xyz']
        extent = np.asarray(mesh.get_axis_aligned_bounding_box().get_extent())
        self.assertTrue(np.allclose([hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]],
                                    extent, atol=0.01),
                        'бокс объекта должен совпасть с мешем, который ставится в сцену')
        self.assertAlmostEqual(float(np.asarray(mesh.get_axis_aligned_bounding_box().get_min_bound())[1]),
                               lo[1], delta=0.01)


if __name__ == '__main__':
    unittest.main()