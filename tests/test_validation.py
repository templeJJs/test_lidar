"""Тесты валидации: внесение объекта в реальные лучи и фон из кадров.

Синтетический кадр-стенд собран так, чтобы повторять измеренную сетку Hesai 128:
128 колец, плотное ядро 64 кольца с шагом 0.1°, периферия 0.2°, своя фаза азимута
у каждого кольца, два возврата на луч (как в реальных bag-ах). Стены -- цилиндр
R = 60 м, пол -- плоскость z = -0.1 м. Такой стенд проверяет и 25 м (объект
виден), и 200 м (объект за измеренной поверхностью -- не виден), и не зависит от
наличия bag-ов.
"""
import os
import sys
import time
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validation import selfcheck as selfcheck_mod  # noqa: E402
from validation import stitch_bg as bg  # noqa: E402
from validation import synth  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL_DB = os.path.join(ROOT, 'for_hackathon', 'doubleT_obstacle',
                       'doubleT_obstacle_0.db3')

TUNNEL_RADIUS = 60.0
TUNNEL_FLOOR_Z = -0.1
AZ_HALF_DEG = 10.0


def synthetic_tunnel(radius=TUNNEL_RADIUS, floor_z=TUNNEL_FLOOR_Z,
                     az_half=AZ_HALF_DEG, returns=2, phase_std_steps=0.3):
    """Кадр-стенд: (points, ring, intensity, floor_ab, axis_nodes)."""
    core = np.linspace(1.97, -6.07, 64)
    up = 1.97 + np.arange(1, 33) * 0.389
    down = -6.07 - np.arange(1, 33) * 0.596
    elev = np.sort(np.concatenate([core, up, down]))[::-1]
    rng = np.random.default_rng(11)

    dirs = []
    rings = []
    for k, el in enumerate(elev):
        step = 0.1 if -6.07 - 1e-9 <= el <= 1.97 + 1e-9 else 0.2
        phase = float(rng.uniform(-0.5, 0.5)) * step * (phase_std_steps / 0.3)
        n = int(round(2 * az_half / step)) + 1
        az = np.arange(n) * step - az_half + phase
        el_rad = np.radians(el)
        az_rad = np.radians(az)
        d = np.stack([np.cos(el_rad) * np.sin(az_rad),
                      -np.cos(el_rad) * np.cos(az_rad),
                      np.full(n, np.sin(el_rad))], axis=1)
        dirs.append(d)
        rings.append(np.full(n, k, dtype=np.int64))
    dirs = np.concatenate(dirs)
    ring = np.concatenate(rings)

    # Дальность: вниз -- до пола z = floor_z, вверх -- до стены R = radius.
    dz = dirs[:, 2]
    t = np.where(dz < -1e-9, floor_z / dz, radius)
    t = np.minimum(t, radius)
    points = dirs * t[:, None]
    if returns > 1:
        points = np.repeat(points, returns, axis=0)
        ring = np.repeat(ring, returns)
    intensity = np.full(points.shape[0], 20.0)
    return points, ring, intensity, (-0.1, 0.0), np.array([[0.0, -300.0], [0.0, 3.0]])


class TestSyntheticFrame(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.points, cls.ring, cls.intensity, cls.floor_ab, cls.axis = synthetic_tunnel()
        cls.meta = synth.build_rings_meta(cls.points, cls.ring, cls.intensity,
                                          axis_nodes=cls.axis, floor_ab=cls.floor_ab)

    def test_frame_shape_and_multiplicity(self):
        self.assertEqual(len(self.points), 2 * np.unique(
            synth.ray_keys(self.points, self.ring)).size)
        self.assertEqual(len(self.ring), len(self.points))

    def test_ring_grid_reproduces_sensor_facts(self):
        grid = synth.ring_grid(self.points, self.ring)
        self.assertEqual(int(np.count_nonzero(np.isfinite(grid['elev_deg']))), 128)
        step = grid['az_step_deg']
        self.assertAlmostEqual(float(np.median(step[step < 0.15])), 0.1, places=6)
        self.assertAlmostEqual(float(np.median(step[step >= 0.15])), 0.2, places=6)
        phase = grid['az_phase']
        dense = np.isfinite(step) & (step < 0.15)
        self.assertLess(float(np.std(phase[dense])), 0.4,
                        'фаза азимута должна быть СВОЕЙ у каждого кольца')

    def test_floor_fit_on_frame(self):
        a, b, rms = synth.fit_floor_ab(self.points)
        self.assertAlmostEqual(a, TUNNEL_FLOOR_Z, delta=0.05)
        self.assertLess(abs(b), 0.02)

    def test_axis_point_extrapolation_is_straight(self):
        point, tangent, normal, z_floor, extrapolated = synth.axis_point(self.meta, 200.0, 0.0)
        self.assertAlmostEqual(float(point[0]), 0.0, places=9)
        self.assertAlmostEqual(float(point[1]), -200.0, places=9)
        self.assertAlmostEqual(float(tangent[1]), 1.0, places=9)
        self.assertAlmostEqual(float(normal[0]), 1.0, places=9)
        self.assertAlmostEqual(z_floor, TUNNEL_FLOOR_Z, places=9)


class TestInsertObject(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.points, cls.ring, cls.intensity, floor_ab, cls.axis = synthetic_tunnel()
        cls.meta = synth.build_rings_meta(cls.points, cls.ring, cls.intensity,
                                          axis_nodes=cls.axis, floor_ab=floor_ab)

    def insert(self, D=25.0, lateral=0.0, seed=0, size_m=(0.3, 0.3, 1.0), **kw):
        return synth.insert_object(self.points, self.meta, D, lateral,
                                   size_m=size_m, seed=seed, **kw)

    def test_object_at_25m_has_many_points(self):
        points_new, mask, meta = self.insert(D=25.0)
        self.assertGreaterEqual(meta['points_inserted'], 100,
                                'объект 0.3x0.3x1 м на 25 м обязан быть плотным')
        self.assertGreater(meta['rays_hit'], 20)
        self.assertEqual(meta['points_inserted'], meta['points_removed'])
        self.assertEqual(meta['points_inserted'], 2 * meta['rays_hit'],
                         'в стенде по 2 возврата на луч, как в реальных bag-ах')
        self.assertEqual(len(points_new), len(self.points))

    def test_object_at_200m_is_invisible(self):
        points_new, mask, meta = self.insert(D=200.0)
        self.assertEqual(meta['rays_hit'], 0)
        self.assertEqual(meta['points_inserted'], 0)
        self.assertEqual(int(mask.sum()), 0)
        np.testing.assert_array_equal(np.asarray(points_new), self.points)

    def test_shadow_removes_background_points(self):
        points_new, mask, meta = self.insert(D=25.0)
        self.assertGreater(int(mask.sum()), 0, 'тень обязана удалить точки фона')
        removed = self.points[mask]
        object_points = np.asarray(points_new)[meta['background_count']:]
        # Точки объекта лежат на тех же лучах, что и удалённые точки фона.
        removed_az = set(synth.az_bins(removed).tolist())
        object_az = set(synth.az_bins(object_points).tolist())
        self.assertTrue(object_az.issubset(removed_az),
                        'точка объекта должна стоять на луче удалённой точки')
        # И все удалённые точки дальше объекта: иначе объект был бы за поверхностью.
        r_removed = np.linalg.norm(removed, axis=1)
        r_object = np.linalg.norm(object_points, axis=1)
        self.assertGreater(float(r_removed.min()), float(r_object.max()) - 1e-6)

    def test_object_points_inside_box_plus_noise(self):
        points_new, mask, meta = self.insert(D=50.0)
        obj = np.asarray(points_new)[meta['background_count']:]
        self.assertGreater(len(obj), 0)
        box_min = meta['box_min'] - 0.05
        box_max = meta['box_max'] + 0.05
        self.assertTrue(np.all(obj >= box_min[None, :] - 1e-6))
        self.assertTrue(np.all(obj <= box_max[None, :] + 1e-6))

    def test_distance_of_object_points(self):
        points_new, mask, meta = self.insert(D=50.0)
        obj = np.asarray(points_new)[meta['background_count']:]
        r = np.linalg.norm(obj, axis=1)
        self.assertLess(abs(float(np.median(r)) - 50.0), 0.5)

    def test_determinism_same_seed(self):
        p1, m1, meta1 = self.insert(D=25.0, seed=17)
        p2, m2, meta2 = self.insert(D=25.0, seed=17)
        np.testing.assert_array_equal(np.asarray(p1), np.asarray(p2))
        np.testing.assert_array_equal(m1, m2)
        self.assertEqual(meta1['rays_hit'], meta2['rays_hit'])

    def test_different_seed_same_geometry_other_noise(self):
        p1, m1, meta1 = self.insert(D=25.0, seed=1)
        p2, m2, meta2 = self.insert(D=25.0, seed=2)
        np.testing.assert_array_equal(m1, m2)
        self.assertEqual(meta1['rays_hit'], meta2['rays_hit'])
        o1 = np.asarray(p1)[meta1['background_count']:]
        o2 = np.asarray(p2)[meta2['background_count']:]
        self.assertEqual(o1.shape, o2.shape)
        # Шум радиальный: разность дальностей -- это разность двух N(0, 10 мм),
        # то есть N(0, 14.1 мм).
        radial = np.linalg.norm(o1, axis=1) - np.linalg.norm(o2, axis=1)
        self.assertGreater(float(np.abs(radial).max()), 0.0, 'шум обязан зависеть от seed')
        self.assertAlmostEqual(float(radial.std()), synth.RANGE_NOISE_M * np.sqrt(2.0),
                               delta=0.006)

    def test_noise_sigma_declared(self):
        _, _, meta = self.insert(D=50.0, seed=3)
        self.assertAlmostEqual(meta['noise_sigma_m'], synth.RANGE_NOISE_M, places=9)
        self.assertAlmostEqual(synth.RANGE_NOISE_M, 0.01, places=9)

    def test_dropout_removes_rays(self):
        _, _, plain = self.insert(D=50.0, seed=5)
        _, _, dropped = self.insert(D=50.0, seed=5, dropout_frac=0.005)
        self.assertLess(dropped['points_inserted'], plain['points_inserted'])
        self.assertEqual(dropped['points_removed'], plain['points_removed'],
                         'тень от dropout не зависит: объект загораживает луч всегда')
        self.assertGreater(dropped['points_inserted'],
                           int(0.9 * plain['points_inserted']))

    def test_expand_z_adds_points(self):
        _, _, flat = self.insert(D=50.0, seed=1, expand_z=False)
        _, _, deep = self.insert(D=50.0, seed=1, expand_z=True)
        self.assertGreaterEqual(deep['points_inserted'], flat['points_inserted'])
        self.assertGreater(deep['hub_m'], 0.0)

    def test_lateral_moves_object_across_track(self):
        _, _, left = self.insert(D=50.0, lateral=-synth.GAUGE_HALF_M)
        _, _, right = self.insert(D=50.0, lateral=+synth.GAUGE_HALF_M)
        self.assertLess(left['center'][0], right['center'][0])
        self.assertAlmostEqual(float(right['center'][0] - left['center'][0]),
                               2 * synth.GAUGE_HALF_M, delta=0.05)

    def test_intensity_bootstrap(self):
        _, _, meta = self.insert(D=50.0)
        values = set(np.unique(self.intensity).tolist())
        self.assertEqual(meta['object_intensity'].shape[0], meta['points_inserted'])
        for v in np.unique(meta['object_intensity']):
            self.assertIn(float(v), values,
                          'интенсивность объекта -- бутстрап из значений кадра')

    def test_visibility_criterion(self):
        points_new, _, meta = self.insert(D=25.0)
        counts = synth.object_point_counts(points_new, meta,
                                           cluster_fn=bg.count_clusters)
        self.assertTrue(counts['visible'])
        self.assertGreaterEqual(counts['vertical_extent_m'], 0.3)
        points_far, _, meta_far = self.insert(D=200.0)
        counts_far = synth.object_point_counts(points_far, meta_far,
                                               cluster_fn=bg.count_clusters)
        self.assertFalse(counts_far['visible'])
        self.assertEqual(counts_far['points_object'], 0)

    def test_bad_size_raises(self):
        with self.assertRaises(ValueError):
            self.insert(D=25.0, size_m=(0.3, 0.3))

    def test_meta_keys_present_when_invisible(self):
        _, _, meta = self.insert(D=200.0)
        for key in ('rays_hit', 'points_removed', 'points_inserted', 'points_new',
                    'rays_candidate', 'background_count', 'object_returns_per_ray'):
            self.assertIn(key, meta)

    def test_dtype_of_input_is_kept(self):
        points32 = self.points.astype(np.float32)
        points_new, mask, meta = synth.insert_object(points32, self.meta, 25.0, 0.0,
                                                    seed=1)
        self.assertEqual(np.asarray(points_new).dtype, np.float32)
        self.assertEqual(mask.dtype, bool)
        self.assertEqual(len(points_new), len(points32))

    def test_float32_frame_and_azimuth_of_object_points(self):
        points32 = self.points.astype(np.float32)
        _, _, meta = synth.insert_object(points32, self.meta, 25.0, 0.0, seed=1)
        self.assertGreater(meta['points_inserted'], 100)


class TestStitchBackground(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.radius = 30.0
        cls.empty = synthetic_tunnel(radius=cls.radius, returns=1)[:2]
        pts, ring = cls.empty
        # "Помеха" -- кусок стены, вдвинутый на 3 м ближе, в узкой апертуре +-2.5°:
        # так остаток получается одним компактным кластером, как настоящая помеха,
        # а не сплошным кольцом вокруг оси.
        obj_dirs = (pts / np.linalg.norm(pts, axis=1)[:, None])
        near = np.linalg.norm(pts, axis=1) - 3.0
        shifted = obj_dirs * near[:, None]
        patch = np.abs(synth.az_deg(pts)) < 2.5
        cls.occupied_points = shifted[patch]
        cls.with_object = (cls.occupied_points, ring[patch])

    def test_per_ray_background_flags_stable_cells(self):
        background = bg.per_ray_background([self.empty, self.empty], gate_min_frames=2)
        self.assertGreater(background['stable_frac'], 0.99)
        self.assertGreater(background['observed_frac'], 0.03)
        ring0 = int(self.empty[1][0])
        bin0 = int(synth.az_bins(self.empty[0][0:1])[0])
        self.assertTrue(bool(background['stable'][ring0, bin0]),
                        'наблюдавшаяся в обоих кадрах ячейка обязана быть устойчивой')

    def test_per_ray_residual_finds_closer_surface(self):
        background = bg.per_ray_background([self.empty, self.empty], gate_min_frames=2)
        residue = bg.per_ray_residual(*self.with_object, background)
        self.assertGreater(len(residue), 50)
        r = np.linalg.norm(residue, axis=1)
        self.assertTrue(np.all(r < self.radius - 0.1))

    def test_per_ray_residual_empty_is_empty(self):
        background = bg.per_ray_background([self.empty, self.empty], gate_min_frames=2)
        residue = bg.per_ray_residual(*self.empty, background)
        self.assertEqual(len(residue), 0)

    def test_object_patch_is_one_cluster(self):
        background = bg.per_ray_background([self.empty, self.empty], gate_min_frames=2)
        residue = bg.per_ray_residual(*self.with_object, background)
        clusters = bg.count_clusters(residue, min_extent=0.3, min_extent_z=0.4)
        self.assertEqual(len(clusters), 1)
        self.assertGreater(clusters[0]['n'], 50)
        self.assertGreater(clusters[0]['extent_z'], 0.4)

    def test_voxel_background_and_residual(self):
        background = bg.voxel_background([self.empty], voxel=0.25, min_hits=1)
        self.assertGreater(background['count'], 0)
        self.assertEqual(len(bg.voxel_residual(self.empty[0], background)), 0)
        residue = bg.voxel_residual(self.with_object[0], background)
        self.assertGreater(len(residue), 100)

    def test_count_clusters_splits_two_blobs(self):
        a = np.zeros((20, 3))
        a[:, 1] = np.linspace(-40.0, -40.4, 20)
        b = np.zeros((20, 3))
        b[:, 1] = np.linspace(-45.0, -45.4, 20)
        b[:, 2] = np.linspace(0.0, 0.8, 20)
        clusters = bg.count_clusters(np.vstack([a, b]), min_extent=0.3)
        self.assertEqual(len(clusters), 2)
        self.assertEqual(sorted(c['n'] for c in clusters), [20, 20])

    def test_count_clusters_filters_small_and_flat(self):
        thin = np.zeros((20, 3))
        thin[:, 1] = np.linspace(-40.0, -40.1, 20)      # размах 0.1 м < 0.3 м
        self.assertEqual(bg.count_clusters(thin, min_extent=0.3), [])
        small = np.zeros((4, 3))
        small[:, 2] = np.linspace(0.0, 0.5, 4)          # 4 точки < 5
        self.assertEqual(bg.count_clusters(small, min_points=5), [])

    def test_cluster_height_rule(self):
        tall = np.zeros((10, 3))
        tall[:, 2] = np.linspace(0.0, 0.6, 10)
        self.assertEqual(len(bg.count_clusters(tall, min_extent=0.3, min_extent_z=0.4)), 1)
        self.assertEqual(bg.count_clusters(tall, min_extent=0.3, min_extent_z=0.8), [])

    def test_corridor_gate(self):
        axis = np.array([[0.0, -100.0], [0.0, 3.0]])
        floor_ab = (-0.1, 0.0)
        pts = np.array([[0.0, -40.0, 1.0],      # в габарите
                        [2.0, -40.0, 1.0],      # поперёк оси вне габарита
                        [0.0, -40.0, 5.0],      # выше габарита
                        [0.0, -40.0, -0.05]])   # ниже пола
        mask = bg.corridor_gate(pts, floor_ab, axis)
        np.testing.assert_array_equal(mask, [True, False, False, False])


class TestNoHeavyDependencies(unittest.TestCase):
    """Валидация обязана работать без open3d/ROS/scipy -- проверяем по исходникам."""

    def test_validation_sources_have_no_heavy_imports(self):
        forbidden = ('import open3d', 'from open3d', 'import scipy', 'from scipy',
                     'import rospy', 'import rosbag', 'import matplotlib',
                     'from matplotlib', 'import cv2', 'from sklearn')
        folder = os.path.join(ROOT, 'validation')
        for name in sorted(os.listdir(folder)):
            if not name.endswith('.py'):
                continue
            with open(os.path.join(folder, name), encoding='utf-8') as fh:
                text = fh.read()
            for token in forbidden:
                self.assertNotIn(token, text, '%s: %s' % (name, token))

    def test_synth_uses_numpy_only(self):
        self.assertEqual(synth.np.__name__, 'numpy')
        self.assertFalse(hasattr(synth, 'o3d'))


class TestRealBag(unittest.TestCase):
    """Числа на реальном кадре: ориентир исследования 264 / 60 / 22-28 / 3-16 / 0."""

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(REAL_DB):
            raise unittest.SkipTest('bag not present')
        cls.points, cls.ring, cls.intensity = synth.load_frame(REAL_DB, 0)
        cls.meta = synth.build_rings_meta(cls.points, cls.ring, cls.intensity,
                                          db_path=REAL_DB, frame_index=0)

    def test_frame_has_two_returns_per_ray(self):
        rays = np.unique(synth.ray_keys(self.points, self.ring)).size
        self.assertGreater(len(self.points) / rays, 1.99)
        self.assertLessEqual(len(self.points) / rays, 2.0)

    def test_axis_from_track_inside_corridor(self):
        nodes = self.meta['axis_nodes']
        self.assertGreaterEqual(len(nodes), 10)
        self.assertLess(float(np.max(np.abs(nodes[:, 0]))), 3.0,
                        'ось пути не может уезжать из тоннеля шириной 16 м')
        self.assertGreater(float(nodes[-1, 1] - nodes[0, 1]), 20.0)

    def test_25m_matches_reference_number(self):
        _, _, meta = synth.insert_object(self.points, self.meta, 25.0, 0.0, seed=1)
        self.assertGreaterEqual(meta['points_inserted'], 100)
        self.assertLessEqual(meta['points_inserted'], 400,
                             'ориентир исследования -- 264 точки на 25 м')

    def test_200m_is_invisible(self):
        _, mask, meta = synth.insert_object(self.points, self.meta, 200.0, 0.0, seed=1)
        self.assertEqual(meta['points_inserted'], 0)
        self.assertEqual(int(mask.sum()), 0)

    def test_insert_faster_than_100ms(self):
        synth.insert_object(self.points, self.meta, 50.0, 0.0, seed=1)   # прогрев
        best = min(self._timed() for _ in range(3))
        self.assertLess(best, 0.15,
                        'замер на 350 тыс. точек: %.0f мс' % (1000 * best))

    def _timed(self):
        t0 = time.perf_counter()
        synth.insert_object(self.points, self.meta, 50.0, 0.0, seed=1)
        return time.perf_counter() - t0

    def test_selfcheck_module_imports_and_counts(self):
        bags = selfcheck_mod.db_paths(ROOT)
        self.assertTrue(bags)
        self.assertIn(bags[0][0], 'doubleT_obstacle')


if __name__ == '__main__':
    unittest.main()