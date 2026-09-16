"""Unit tests for the testable core of zones.py (no GUI required)."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import zones  # noqa: E402


def probe(x, y, z):
    """One isolated point as a (1, 3) float32 array, like a bag frame."""
    return np.array([[x, y, z]], dtype=np.float32)


def params_with_walls():
    return zones.ZoneParams(wall_x=(6.6,))


def zone_cases(p):
    """One unambiguous probe per zone (y = 0 so floor_profile(0) = floor_a)."""
    floor0 = p.floor_a
    return {
        'rail': (2.36, 0.0, floor0 + 0.16),
        'bed': (5.0, 0.0, floor0 + 0.10),
        'ceiling': (5.0, 0.0, 2.5),
        'wall': (6.6, 0.0, floor0 + 1.0),
        'middle': (5.0, 0.0, floor0 + 1.0),
    }


class TestFitFloor(unittest.TestCase):
    def test_recovers_synthetic_slope(self):
        rng = np.random.default_rng(0)
        y = rng.uniform(-60.0, -2.0, 60000)
        z = 1.0 - 0.03 * y + rng.uniform(-0.02, 0.02, y.size)
        xyz = np.column_stack([np.zeros_like(y), y, z]).astype(np.float32)
        a, b, rms = zones.fit_floor(xyz)
        self.assertAlmostEqual(a, 1.0, delta=0.06)
        self.assertAlmostEqual(b, -0.03, delta=0.005)
        self.assertLess(rms, 0.03)

    def test_fewer_than_two_bins_falls_back_to_flat(self):
        xyz = np.array([[0.0, -300.0, -1.0], [0.0, -301.0, -0.8]], dtype=np.float32)
        a, b, rms = zones.fit_floor(xyz, y_min=-60.0, y_max=-2.0)
        self.assertAlmostEqual(a, -0.9, delta=1e-6)
        self.assertEqual(b, 0.0)
        self.assertEqual(rms, 0.0)


class TestFloorProfile(unittest.TestCase):
    def test_matches_a_plus_b_y(self):
        p = zones.ZoneParams(floor_a=1.0, floor_b=-0.03)
        y = np.array([-60.0, -30.0, -2.0, 0.0])
        np.testing.assert_allclose(zones.floor_profile(y, p), 1.0 - 0.03 * y)


class TestClassify(unittest.TestCase):
    def setUp(self):
        self.p = params_with_walls()

    def _label(self, x, y, z):
        return int(zones.classify(probe(x, y, z), self.p)[0])

    def test_one_probe_per_zone(self):
        for name, (x, y, z) in zone_cases(self.p).items():
            self.assertEqual(zones.ZONE_NAMES[self._label(x, y, z)], name)

    def test_rail_beats_bed(self):
        p = self.p
        x = p.axis_x + p.gauge / 2.0
        z = p.floor_a + 0.16
        self.assertEqual(zones.ZONE_NAMES[self._label(x, 0.0, z)], 'rail')

    def test_just_above_floor_is_bed_not_middle(self):
        z = self.p.floor_a + 0.29
        self.assertEqual(zones.ZONE_NAMES[self._label(5.0, 0.0, z)], 'bed')

    def test_high_point_is_ceiling(self):
        self.assertEqual(zones.ZONE_NAMES[self._label(5.0, 0.0, 2.5)], 'ceiling')

    def test_wall_point(self):
        z = self.p.floor_a + 1.0
        self.assertEqual(zones.ZONE_NAMES[self._label(6.6, 0.0, z)], 'wall')

    def test_middle_point(self):
        z = self.p.floor_a + 1.0
        self.assertEqual(zones.ZONE_NAMES[self._label(5.0, 0.0, z)], 'middle')

    def test_empty_returns_empty_uint8(self):
        labels = zones.classify(np.zeros((0, 3), dtype=np.float32), self.p)
        self.assertEqual(labels.shape, (0,))
        self.assertEqual(labels.dtype, np.uint8)


class TestColors(unittest.TestCase):
    def test_five_colors_pairwise_distinct(self):
        table = np.array([zones.ZONE_COLORS[n] for n in zones.ZONE_NAMES])
        self.assertEqual(len({tuple(c) for c in table}), 5)

    def test_zone_colors_reproduces_table(self):
        p = params_with_walls()
        cases = zone_cases(p)
        xyz = np.array(list(cases.values()), dtype=np.float64)
        colors = zones.zone_colors(xyz, p)
        self.assertEqual(colors.dtype, np.float64)
        self.assertEqual(colors.shape, (len(cases), 3))
        for i, name in enumerate(cases):
            np.testing.assert_allclose(colors[i], zones.ZONE_COLORS[name])

    def test_empty_shape_and_dtype(self):
        colors = zones.zone_colors(np.zeros((0, 3), dtype=np.float32), params_with_walls())
        self.assertEqual(colors.shape, (0, 3))
        self.assertEqual(colors.dtype, np.float64)


class TestCounts(unittest.TestCase):
    def test_counts_sum_and_keys(self):
        p = params_with_walls()
        cases = zone_cases(p)
        xyz = np.array(list(cases.values()) + [(5.0, 0.0, p.floor_a + 1.1)], dtype=np.float64)
        counts = zones.classify_counts(xyz, p)
        self.assertEqual(set(counts), set(zones.ZONE_NAMES))
        self.assertEqual(sum(counts.values()), len(xyz))
        for value in counts.values():
            self.assertIsInstance(value, int)

    def test_empty_counts_are_all_zero(self):
        counts = zones.classify_counts(np.zeros((0, 3), dtype=np.float32), params_with_walls())
        self.assertEqual(set(counts), set(zones.ZONE_NAMES))
        self.assertEqual(set(counts.values()), {0})


class TestDetectWalls(unittest.TestCase):
    def test_two_synthetic_walls(self):
        p = zones.ZoneParams()
        rng = np.random.default_rng(1)
        y = rng.uniform(-50.0, -5.0, 40000)
        zrel = rng.uniform(0.5, 2.0, y.size)
        z = p.floor_a + p.floor_b * y + zrel
        x = np.where(rng.random(y.size) < 0.5, -2.7, 6.6) + rng.normal(0.0, 0.03, y.size)
        xyz = np.column_stack([x, y, z])
        walls = zones.detect_walls(xyz, p)
        self.assertEqual(len(walls), 2)
        self.assertAlmostEqual(walls[0], -2.7, delta=0.3)
        self.assertAlmostEqual(walls[1], 6.6, delta=0.3)

    def test_no_points_above_bed_returns_empty(self):
        p = zones.ZoneParams()
        y = np.linspace(-30.0, -5.0, 1000)
        z = p.floor_a + p.floor_b * y + 0.05
        xyz = np.column_stack([np.full(y.size, 3.0), y, z])
        self.assertEqual(zones.detect_walls(xyz, p), [])


class TestRailXPositions(unittest.TestCase):
    def test_two_running_rails_when_contact_disabled(self):
        p = zones.ZoneParams(contact_rail=False)
        rails = zones.rail_x_positions(p)
        self.assertEqual(len(rails), 2)
        half = p.gauge / 2.0
        self.assertAlmostEqual(rails[0][0], p.axis_x - half, places=6)
        self.assertAlmostEqual(rails[1][0], p.axis_x + half, places=6)
        for _x, height in rails:
            self.assertAlmostEqual(height, zones.RAIL_HEIGHT)

    def test_three_rails_with_contact(self):
        p = zones.ZoneParams(contact_rail=True, contact_side=+1)
        rails = zones.rail_x_positions(p)
        self.assertEqual(len(rails), 3)
        half = p.gauge / 2.0
        self.assertAlmostEqual(rails[0][0], p.axis_x - half, places=6)
        self.assertAlmostEqual(rails[1][0], p.axis_x + half, places=6)
        self.assertAlmostEqual(rails[2][0], p.axis_x + p.contact_offset, places=6)
        self.assertAlmostEqual(rails[2][1], p.contact_height)


class TestContactRailClassify(unittest.TestCase):
    """Контактный рельс: своя полоса вокруг СВОЕЙ высоты, а не высота ходового.

    По норме (ПТЭ метрополитенов п. 3.26) он лежит слева по ходу, в 1450 мм от
    оси, рабочая поверхность 160 мм над УГР, то есть `contact_height =
    RAIL_HEIGHT + 0.16 = 0.32` м над плоскостью пола. Смещение в тестах берём
    своим (1.30): важно, что признак живёт на своей высоте и своей полуширине.
    """

    def _params(self, **kw):
        base = dict(wall_x=(6.6,), contact_offset=1.30, contact_side=+1)
        base.update(kw)
        return zones.ZoneParams(**base)

    def test_contact_point_is_rail(self):
        p = self._params()
        x = p.axis_x + p.contact_offset
        z = p.floor_a + p.contact_height
        self.assertEqual(zones.ZONE_NAMES[int(zones.classify(probe(x, 0.0, z), p)[0])], 'rail')

    def test_contact_point_stops_being_rail_when_disabled(self):
        """Та же точка без контактного рельса -- уже не рельс.

        Было `assertEqual(..., 'bed')`: при старой высоте 0.20 м точка попадала
        под floor_band 0.30. По нормативу высота 0.32 м над полом, поэтому
        проверяем сам факт «не рельс», а не соседнюю зону.
        """
        on = self._params()
        off = self._params(contact_rail=False)
        x = on.axis_x + on.contact_offset
        z = on.floor_a + on.contact_height
        self.assertEqual(zones.ZONE_NAMES[int(zones.classify(probe(x, 0.0, z), on)[0])],
                         'rail')
        self.assertNotEqual(zones.ZONE_NAMES[int(zones.classify(probe(x, 0.0, z), off)[0])],
                            'rail')

    def test_negative_side_mirrors_contact_rail(self):
        p = self._params(contact_side=-1)
        rails = zones.rail_x_positions(p)
        self.assertAlmostEqual(rails[2][0], p.axis_x - p.contact_offset, places=6)
        x = p.axis_x - p.contact_offset
        z = p.floor_a + p.contact_height
        self.assertEqual(zones.ZONE_NAMES[int(zones.classify(probe(x, 0.0, z), p)[0])], 'rail')


class TestRailLines(unittest.TestCase):
    def test_two_rails_geometry_and_color(self):
        p = zones.ZoneParams(contact_rail=False)
        ls = zones.make_rail_lines(p, y_from=-80.0, y_to=0.0, step=1.0)
        pts = np.asarray(ls.points)
        lines = np.asarray(ls.lines)
        self.assertEqual(pts.dtype, np.float64)
        self.assertEqual(lines.shape, (2, 2), 'exactly two lines')

        n = pts.shape[0] // 2
        self.assertEqual(pts.shape[0], 2 * n)
        self.assertAlmostEqual(float(pts[0, 1]), -80.0)
        self.assertAlmostEqual(float(pts[n - 1, 1]), 0.0)

        left_x = p.axis_x - p.gauge / 2.0
        right_x = p.axis_x + p.gauge / 2.0
        self.assertTrue(np.allclose(pts[:n, 0], left_x))
        self.assertTrue(np.allclose(pts[n:, 0], right_x))

        for block in (pts[:n], pts[n:]):
            expected_z = p.floor_a + p.floor_b * block[:, 1] + 0.16
            np.testing.assert_allclose(block[:, 2], expected_z)

        colors = np.asarray(ls.colors)
        self.assertEqual(colors.shape, (2, 3))
        self.assertTrue(np.allclose(colors, zones.ZONE_COLORS['rail']))

    def test_three_rails_geometry_includes_contact(self):
        p = zones.ZoneParams()
        ls = zones.make_rail_lines(p, y_from=-20.0, y_to=0.0, step=2.0)
        pts = np.asarray(ls.points)
        n = zones._y_nodes(-20.0, 0.0, 2.0).size
        rails = zones.rail_x_positions(p)
        self.assertEqual(pts.shape[0], len(rails) * n)

        xs = np.unique(np.round(pts[:, 0], 6))
        self.assertEqual(xs.size, 3, 'three distinct rail X positions')

        contact = pts[2 * n:3 * n]
        # Сторона -- часть модели (`contact_side`), а не знак «плюс» по умолчанию:
        # по норме контактный рельс уложен слева по ходу, то есть при -1 от оси.
        self.assertTrue(np.allclose(contact[:, 0],
                                    p.axis_x + p.contact_side * p.contact_offset))
        expected_z = zones.floor_profile(contact[:, 1], p) + p.contact_height
        np.testing.assert_allclose(contact[:, 2], expected_z)
        self.assertEqual(len(rails), 3)


class TestContactRailDefaults(unittest.TestCase):
    """Дефолты приведены к нормативу (расхождения из спеки, раздел «Установленные
    факты»): 160 мм над УГР, 1450 мм от оси, слева по ходу."""

    def test_offset_height_and_side(self):
        p = zones.ZoneParams()
        self.assertAlmostEqual(p.contact_offset, 1.45, places=6)
        self.assertEqual(p.contact_side, -1)
        self.assertAlmostEqual(p.contact_height, zones.RAIL_HEIGHT + 0.16, places=6)
        self.assertAlmostEqual(p.contact_height, 0.32, places=6)

    def test_default_contact_rail_sits_left_of_the_axis(self):
        p = zones.ZoneParams(axis_x=1.60)
        rails = zones.rail_x_positions(p)
        self.assertEqual(len(rails), 3)
        self.assertAlmostEqual(rails[2][0], 1.60 - 1.45, places=6)
        self.assertAlmostEqual(rails[2][1], 0.32, places=6)

    def test_contact_band_does_not_overlap_the_running_rail_band(self):
        p = zones.ZoneParams()
        gap = (abs(p.contact_offset - p.gauge / 2.0)
               - p.rail_half_width - p.contact_half_width)
        self.assertGreater(gap, 0.0, 'контактный рельс не должен сливаться с ходовым')


if __name__ == '__main__':
    unittest.main()

class TestColorsFromLabels(unittest.TestCase):
    def test_matches_zone_colors_and_table(self):
        pts = np.array([
            [1.60, -10.0, -1.70],   # near rails / bed region
            [1.60, -10.0, 2.50],    # high -> ceiling
            [-2.70, -10.0, 1.00],   # wall if wall_x set
            [4.00, -20.0, 0.50],    # high above floor, below ceiling
        ], dtype=np.float64)
        params = zones.ZoneParams(floor_a=-1.886, floor_b=0.021585, wall_x=(-2.70, 6.60))
        labels = zones.classify(pts, params)
        via_labels = zones.colors_from_labels(labels)
        via_xyz = zones.zone_colors(pts, params)
        np.testing.assert_allclose(via_labels, via_xyz)
        for i, lab in enumerate(labels):
            np.testing.assert_allclose(
                via_labels[i], zones.ZONE_COLORS[zones.ZONE_NAMES[int(lab)]])

    def test_empty(self):
        out = zones.colors_from_labels(np.zeros(0, dtype=np.uint8))
        self.assertEqual(out.shape, (0, 3))
        self.assertEqual(out.dtype, np.float64)

class TestCorridorBlocked(unittest.TestCase):
    def _params(self, **kw):
        base = dict(floor_a=-1.886, floor_b=0.021585, axis_x=1.60,
                    gauge=1.520, wall_x=(-2.70, 6.60))
        base.update(kw)
        return zones.ZoneParams(**base)

    def _cloud(self, obstacle_y_range=None):
        """Flat bed along y in (-40, 0) plus an optional vertical structure patch."""
        ys = np.arange(-40.0, 0.0, 0.25)
        parts = []
        for y in ys:
            floor = -1.886 + 0.021585 * y
            for x in np.arange(0.9, 2.3, 0.1):      # bed points, below floor_band
                parts.append((x, y, floor + 0.05))
        if obstacle_y_range is not None:
            for y in np.arange(*obstacle_y_range, 0.25):
                floor = -1.886 + 0.021585 * y
                for x in np.arange(1.0, 2.2, 0.05):  # vertical structure 0.3..2.5 m up
                    for zz in np.arange(0.4, 2.5, 0.2):
                        parts.append((x, y, floor + zz))
        return np.array(parts, dtype=np.float64)

    def test_bed_alone_is_not_blocked(self):
        params = self._params()
        ys, blocked = zones.corridor_blocked(self._cloud(), params, -40.0, 0.0,
                                             bin_width=1.0)
        self.assertEqual(blocked.shape, ys.shape)
        self.assertFalse(bool(np.any(blocked)), "bed points are below floor_band")

    def test_structure_blocks_only_its_own_y(self):
        params = self._params()
        ys, blocked = zones.corridor_blocked(self._cloud((-16.0, -14.0)), params,
                                             -40.0, 0.0, bin_width=1.0, min_points=20)
        self.assertTrue(bool(np.any(blocked)))
        hit = blocked[np.argmin(np.abs(ys + 15.0))]
        self.assertTrue(bool(hit), "y ~ -15 must be blocked")
        clear = blocked[np.argmin(np.abs(ys + 30.0))]
        self.assertFalse(bool(clear), "y ~ -30 must be clear")

    def test_axis_moved_away_clears_the_corridor(self):
        params = self._params(axis_x=4.0)
        ys, blocked = zones.corridor_blocked(self._cloud((-16.0, -14.0)), params,
                                             -40.0, 0.0, bin_width=1.0, min_points=20)
        self.assertFalse(bool(np.any(blocked)),
                         "structure at x=1..2 must not block a corridor centred at 4")

    def test_empty_cloud_is_all_clear(self):
        params = self._params()
        ys, blocked = zones.corridor_blocked(np.zeros((0, 3)), params, -40.0, 0.0)
        self.assertEqual(blocked.shape, ys.shape)
        self.assertFalse(bool(np.any(blocked)))

    def test_spans_merge_consecutive_bins(self):
        ys = np.arange(-9.0, 1.0, 1.0)
        blocked = np.array([False, True, True, False, True, False, False, False,
                            False, False])
        spans = zones.blocked_spans(ys, blocked)
        self.assertEqual(len(spans), 2)
        self.assertAlmostEqual(spans[0][0], -8.5)
        self.assertAlmostEqual(spans[0][1], -6.5)
        self.assertAlmostEqual(spans[1][0], -5.5)
        self.assertAlmostEqual(spans[1][1], -4.5)

    def test_spans_empty_when_nothing_blocked(self):
        ys = np.arange(-5.0, 1.0, 1.0)
        self.assertEqual(zones.blocked_spans(ys, np.zeros(ys.size, dtype=bool)), [])

    def _thin_surface_cloud(self, y_range=(-16.0, -14.0)):
        """Тонкая поверхность вдоль края коридора: 5 см по высоте, много точек.

        Так выглядит кромка балласта, кабельный лоток или стенка рельса --
        замерено на doubleT_obstacle: 0.0-0.19 м по высоте.
        """
        parts = []
        for y in np.arange(*y_range, 0.25):
            floor = -1.886 + 0.021585 * y
            for x in np.arange(1.0, 2.2, 0.05):
                for zz in (0.40, 0.42, 0.45):
                    parts.append((x, y, floor + zz))
        return np.array(parts, dtype=np.float64)

    def test_thin_surface_does_not_block(self):
        params = self._params()
        cloud = self._thin_surface_cloud()
        _, by_count = zones.corridor_blocked(cloud, params, -40.0, 0.0,
                                             bin_width=1.0, min_points=20,
                                             min_height=0.0)
        self.assertTrue(bool(np.any(by_count)),
                        'без требования высоты поверхность считается препятствием')
        _, by_height = zones.corridor_blocked(cloud, params, -40.0, 0.0,
                                              bin_width=1.0, min_points=20)
        self.assertFalse(bool(np.any(by_height)),
                         'тонкая поверхность не должна перекрывать путь')

    def test_tall_structure_still_blocks(self):
        params = self._params()
        ys, blocked = zones.corridor_blocked(self._cloud((-16.0, -14.0)), params,
                                             -40.0, 0.0, bin_width=1.0,
                                             min_points=20, min_height=0.5)
        self.assertTrue(bool(blocked[np.argmin(np.abs(ys + 15.0))]),
                        'объект высотой 2 м обязан остаться препятствием')

    def test_bin_span_separates_thin_from_tall(self):
        edges = np.array([-1.5, -0.5, 0.5, 1.5])
        thin = np.array([0.40, 0.42, 0.45, 0.41, 0.43, 0.44])
        tall = np.array([0.40, 0.90, 1.40, 1.90, 2.40, 2.50])
        spans = zones._bin_span(np.concatenate([thin, tall]),
                                np.concatenate([np.full(thin.size, -1.0),
                                                np.full(tall.size, 0.0)]), edges)
        self.assertLess(spans[0], 0.1, 'тонкая поверхность: размах в сантиметрах')
        self.assertGreater(spans[1], 1.0, 'высокий объект: размах в метрах')
        self.assertEqual(spans[2], 0.0, 'пустой бин: размах 0')
        self.assertEqual(zones._bin_span(np.zeros(0), np.zeros(0), edges).shape,
                         (3,))


class TestRailBlockedRendering(unittest.TestCase):
    PARAMS = zones.ZoneParams(axis_x=1.60, gauge=1.520)

    def test_render_is_split_and_uses_stop_colour(self):
        ys = zones._y_nodes(-10.0, 0.0, 1.0)
        blocked = np.zeros(ys.size, dtype=bool)
        blocked[np.argmin(np.abs(ys + 5.0))] = True
        lineset = zones.make_rail_lines(self.PARAMS, y_from=-10.0, y_to=0.0,
                                       step=1.0, blocked=blocked)
        lines = np.asarray(lineset.lines)
        colors = np.asarray(lineset.colors)
        n_rails = len(zones.rail_x_positions(self.PARAMS))
        self.assertEqual(n_rails, 3)
        self.assertEqual(len(lines), n_rails * (ys.size - 1))
        self.assertEqual(colors.shape, (len(lines), 3))
        stop = np.array(zones.BLOCKED_COLOR)
        rail = np.array(zones.ZONE_COLORS['rail'])
        self.assertTrue(any(np.allclose(c, stop) for c in colors),
                        "blocked segment must use BLOCKED_COLOR")
        self.assertTrue(any(np.allclose(c, rail) for c in colors))
        self.assertGreater(int(np.sum(np.all(np.isclose(colors, stop), axis=1))), 0)

    def test_mismatched_blocked_length_raises(self):
        with self.assertRaises(ValueError):
            zones.make_rail_lines(self.PARAMS, y_from=-10.0, y_to=0.0, step=1.0,
                                  blocked=np.zeros(3, dtype=bool))

    def test_rail_line_points_lie_on_gauge(self):
        p = zones.ZoneParams(axis_x=1.60, gauge=1.520, contact_rail=False)
        lineset = zones.make_rail_lines(p, y_from=-20.0, y_to=0.0, step=2.0)
        pts = np.asarray(lineset.points)
        xs = np.unique(np.round(pts[:, 0], 6))
        self.assertEqual(xs.size, 2)
        self.assertAlmostEqual(xs[0], 1.60 - 0.76, places=6)
        self.assertAlmostEqual(xs[1], 1.60 + 0.76, places=6)
        zs = zones.floor_profile(pts[:, 1], p) + zones.RAIL_HEIGHT
        np.testing.assert_allclose(pts[:, 2], zs)

class TestFindAxis(unittest.TestCase):
    PARAMS = zones.ZoneParams(floor_a=-1.886, floor_b=0.021585,
                              wall_x=(-2.70, 6.60))

    def _cloud_with_pillar(self, pillar_x=(1.30, 1.80), y_span=(-30.0, -10.0)):
        ys = np.arange(-40.0, 0.0, 0.25)
        parts = []
        for y in ys:
            floor = -1.886 + 0.021585 * y
            for x in np.arange(-2.0, 6.0, 0.1):          # flat bed
                parts.append((x, y, floor + 0.05))
        for y in np.arange(y_span[0], y_span[1], 0.25):  # the pillar
            floor = -1.886 + 0.021585 * y
            for x in np.arange(pillar_x[0], pillar_x[1], 0.05):
                for zz in np.arange(0.4, 2.5, 0.2):
                    parts.append((x, y, floor + zz))
        return np.array(parts, dtype=np.float64)

    def test_axis_avoids_the_pillar(self):
        cloud = self._cloud_with_pillar()
        axis = zones.find_axis(cloud, self.PARAMS, -40.0, 0.0, bin_width=1.0) 
        # gauge/2 + clearance = 0.76 + 0.30 = 1.06 -> corridor must clear 1.30..1.80
        self.assertTrue(axis + 1.06 <= 1.30 or axis - 1.06 >= 1.80,
                        f"axis {axis} puts the corridor through the pillar")

    def test_wall_midpoint_would_fail_the_constraint(self):
        """The old heuristic (midpoint between walls) is inside the blocked set."""
        cloud = self._cloud_with_pillar()
        midpoint = 0.5 * (-2.70 + 6.60)
        corridor = (midpoint - 1.06, midpoint + 1.06)
        overlaps = corridor[0] < 1.80 and corridor[1] > 1.30
        self.assertTrue(overlaps,
                        "sanity: the midpoint heuristic must indeed be blocked here")

    def test_free_axis_has_full_free_length(self):
        cloud = self._cloud_with_pillar()
        axes, free, n_bins = zones.free_corridor_profile(
            cloud, self.PARAMS, -40.0, 0.0, bin_width=1.0)
        axis = zones.find_axis(cloud, self.PARAMS, -40.0, 0.0, bin_width=1.0)
        idx = int(np.argmin(np.abs(axes - axis)))
        self.assertEqual(int(free[idx]), n_bins, "chosen axis must be free all along Y")

    def test_empty_cloud_returns_current_axis(self):
        axis = zones.find_axis(np.zeros((0, 3)), self.PARAMS, -40.0, 0.0)
        self.assertAlmostEqual(axis, self.PARAMS.axis_x, places=6)

    def test_returns_centre_of_the_free_plateau(self):
        cloud = self._cloud_with_pillar()
        axes, free, _ = zones.free_corridor_profile(cloud, self.PARAMS, -40.0, 0.0)
        axis = zones.find_axis(cloud, self.PARAMS, -40.0, 0.0)
        idx = int(np.argmin(np.abs(axes - axis)))
        # must not sit on the very edge of the free region
        self.assertTrue(free[idx - 1] == free[idx] or free[idx + 1] == free[idx]
                        if 0 < idx < len(free) - 1 else True)


# ---------------------------------------------------------------------------
# Линия хода и туннель безопасности
# (docs/superpowers/specs/2026-09-15-path-and-safety-tunnel-design.md, раздел
# «Проверка»). Синтетика повторяет doubleT_obstacle: пол наклонён, стены
# x = -2.70 и 6.60, шаг узлов 1 м, дальность 40 м.

FLOOR_A = -1.886
FLOOR_B = 0.021585


def tunnel_params(**kw):
    """Параметры с прямыми стенами -- как на реальном bag'е."""
    base = dict(floor_a=FLOOR_A, floor_b=FLOOR_B, axis_x=1.60, gauge=1.520,
                wall_x=(-2.70, 6.60))
    base.update(kw)
    return zones.ZoneParams(**base)


def cloud_block(x_range, y_range, zrel_range, step_x=0.05, step_y=0.25,
                step_z=0.20):
    """Плотный параллелепипед: вертикальная структура (стена, столб, груз)."""
    parts = []
    for y in np.arange(y_range[0], y_range[1], step_y):
        floor = FLOOR_A + FLOOR_B * y
        for x in np.arange(x_range[0], x_range[1], step_x):
            for zz in np.arange(zrel_range[0], zrel_range[1], step_z):
                parts.append((x, y, floor + zz))
    return np.array(parts, dtype=np.float64)


def cloud_bed(y_range=(-40.0, 0.0), x_range=(-2.6, 6.5)):
    """Ровное ложе: точки ниже floor_band, поэтому препятствием не считаются."""
    parts = []
    for y in np.arange(y_range[0], y_range[1], 0.25):
        floor = FLOOR_A + FLOOR_B * y
        for x in np.arange(x_range[0], x_range[1], 0.1):
            parts.append((x, y, floor + 0.05))
    return np.array(parts, dtype=np.float64)


def rail_model(s=0.0, i=1.0):
    """Заглушка `track_geometry.build_track`: ось x = s*y + i."""
    return lambda db_path, frame: {'s': s, 'i': i}


class TestPathVariants(unittest.TestCase):
    def _cloud_with_pillar(self):
        """Ложе + стены + столб на x 1.30..1.80, y -30..-10 (как на реальных данных)."""
        return np.vstack([
            cloud_bed(),
            cloud_block((-2.70, -2.65), (-40.0, 0.0), (0.5, 4.0), 0.01, 0.25),
            cloud_block((6.55, 6.60), (-40.0, 0.0), (0.5, 4.0), 0.01, 0.25),
            cloud_block((1.30, 1.80), (-30.0, -10.0), (0.4, 2.5), 0.05, 0.25),
        ])

    def test_block_shape_and_sources(self):
        path = zones.path_variants(self._cloud_with_pillar(), tunnel_params(),
                                   -40.0, 0.0, rail_model=rail_model())
        self.assertAlmostEqual(path['step'], zones.PATH_STEP)
        self.assertAlmostEqual(path['z_offset'], zones.RAIL_HEIGHT)
        self.assertEqual(set(path['variants']), {'corridor', 'rails', 'axis'})
        for name, source in (('corridor', 'corridor-centre'),
                             ('rails', 'track_geometry'),
                             ('axis', 'panel-axis')):
            variant = path['variants'][name]
            self.assertEqual(variant['source'], source)
            self.assertTrue(variant['available'], name)
            self.assertEqual(variant['y'].shape, variant['x'].shape)
            self.assertEqual(variant['y'].shape, variant['z'].shape)

    def test_nodes_only_where_there_is_data(self):
        """Облако кончается на y = -25 -- узлов дальше быть не должно."""
        cloud = cloud_bed(y_range=(-40.0, -25.0))
        path = zones.path_variants(cloud, tunnel_params(), -40.0, 0.0,
                                   rail_model=rail_model())
        for variant in path['variants'].values():
            self.assertTrue(variant['available'])
            self.assertGreaterEqual(float(variant['y'].min()), -40.0)
            self.assertLessEqual(float(variant['y'].max()), -25.0)
        self.assertLess(len(path['variants']['axis']['y']), 41)

    def test_corridor_line_stays_inside_walls_and_free_corridor(self):
        p = tunnel_params()
        cloud = self._cloud_with_pillar()
        corridor = zones.path_variants(cloud, p, -40.0, 0.0,
                                       rail_model=rail_model())['variants']['corridor']
        self.assertTrue(corridor['available'])
        lo, hi = min(p.wall_x), max(p.wall_x)
        self.assertTrue(np.all((corridor['x'] > lo) & (corridor['x'] < hi)),
                        'линия не должна выходить за стены')

        x, y, z = cloud[:, 0], cloud[:, 1], cloud[:, 2]
        zrel = z - zones.floor_profile(y, p)
        half = p.gauge / 2.0 + 0.30
        for yy, xx in zip(corridor['y'], corridor['x']):
            hit = ((np.abs(y - yy) <= 0.5) & (np.abs(x - xx) <= half)
                   & (zrel > p.floor_band) & (zrel < zones.TRAIN_TOP))
            self.assertEqual(int(np.count_nonzero(hit)), 0,
                             f'препятствие в коридоре вокруг линии на y = {yy}')

    def test_corridor_goes_around_the_pillar(self):
        """Столб на x 1.30..1.80 сдвигает линию, сглаживание его не съедает."""
        p = tunnel_params()
        corridor = zones.path_variants(self._cloud_with_pillar(), p, -40.0, 0.0,
                                       rail_model=rail_model())['variants']['corridor']
        self.assertGreater(float(corridor['x'].max() - corridor['x'].min()), 0.5)
        for yy, xx in zip(corridor['y'], corridor['x']):
            if -25.0 <= yy <= -15.0:
                self.assertGreater(xx, 1.80, f'на y = {yy} линия стоит в столбе')

    def test_axis_variant_is_a_straight_line_at_the_panel_axis(self):
        p = tunnel_params(axis_x=3.31)
        path = zones.path_variants(cloud_bed(), p, -40.0, 0.0,
                                   rail_model=rail_model())
        axis = path['variants']['axis']
        self.assertTrue(np.allclose(axis['x'], 3.31))
        np.testing.assert_allclose(axis['z'],
                                   zones.floor_profile(axis['y'], p) + zones.RAIL_HEIGHT)

    def test_rails_variant_follows_the_model(self):
        path = zones.path_variants(cloud_bed(), tunnel_params(), -40.0, 0.0,
                                   rail_model=rail_model(s=0.02, i=-0.08))
        rails = path['variants']['rails']
        np.testing.assert_allclose(rails['x'], 0.02 * rails['y'] - 0.08)

    def test_rails_variant_unavailable_when_track_geometry_fails(self):
        def boom(db_path, frame):
            raise RuntimeError('нет модели оси')

        path = zones.path_variants(cloud_bed(), tunnel_params(), -40.0, 0.0,
                                   rail_model=boom)
        rails = path['variants']['rails']
        self.assertFalse(rails['available'])
        self.assertIn('нет модели оси', rails['error'])
        self.assertEqual(rails['y'].size, 0)
        self.assertEqual(rails['x'].size, 0)
        self.assertTrue(path['variants']['corridor']['available'])
        self.assertTrue(path['variants']['axis']['available'])

    def test_rails_variant_unavailable_without_a_model_at_all(self):
        """db_path не передан -- вариант помечен unavailable, а не падает."""
        path = zones.path_variants(cloud_bed(), tunnel_params(), -40.0, 0.0)
        rails = path['variants']['rails']
        self.assertFalse(rails['available'])
        self.assertTrue(rails['error'])

    def test_empty_cloud_gives_unavailable_variants(self):
        path = zones.path_variants(np.zeros((0, 3)), tunnel_params(), -40.0, 0.0)
        for name, variant in path['variants'].items():
            self.assertFalse(variant['available'], name)
            self.assertEqual(variant['y'].size, 0)
            self.assertEqual(variant['z'].size, 0)
            self.assertTrue(variant['error'])

    def test_no_free_interval_marks_corridor_unavailable(self):
        """Полоса забита сплошной вертикальной структурой -- свободных ячеек нет."""
        solid = cloud_block((-2.75, 6.65), (-40.0, 0.0), (0.4, 2.0), 0.05, 0.25)
        path = zones.path_variants(solid, tunnel_params(), -40.0, 0.0,
                                   rail_model=rail_model())
        corridor = path['variants']['corridor']
        self.assertFalse(corridor['available'])
        self.assertIn('свободного интервала', corridor['error'])
        self.assertEqual(corridor['y'].size, 0)
        self.assertTrue(path['variants']['axis']['available'])


class TestTunnelBlocked(unittest.TestCase):
    def test_stray_points_in_platform_band_do_not_clear_an_obstacle(self):
        """Пара случайных точек в полосе платформы не гасит настоящую помеху.

        Раньше бин помечался платформой от одной точки: замерено на
        doubleT_obstacle -- бины с 177 и 1143 точками гасли из-за 25 и 19 точек
        в полосе платформы.
        """
        p = tunnel_params()
        wide = 1.60                       # полоса платформы (1.40..1.55) входит в сечение
        obstacle = cloud_block((p.axis_x - 0.30, p.axis_x + 0.30), (-15.0, -14.0),
                               (0.40, 1.60), 0.05, 0.10, 0.20)
        stray = cloud_block((p.axis_x + 1.42, p.axis_x + 1.50), (-15.0, -14.5),
                            (1.16, 1.30), 0.03, 0.10, 0.05)
        t = zones.tunnel_blocked(np.vstack([obstacle, stray]), p, -20.0, 0.0,
                                 half_width=wide, min_height=0.5)
        i = int(np.argmin(np.abs(t['y'] - (-14.5))))
        self.assertFalse(bool(t['platform'][i]), 'это не платформа')
        self.assertTrue(bool(t['blocked'][i]), 'помеха обязана остаться')

    def test_platform_edge_that_dominates_the_bin_is_not_an_obstacle(self):
        p = tunnel_params()
        wide = 1.60
        edge = cloud_block((p.axis_x + 1.40, p.axis_x + 1.56), (-20.0, -10.0),
                           (1.16, 1.36), 0.02, 0.10, 0.05)
        t = zones.tunnel_blocked(edge, p, -20.0, 0.0, half_width=wide, min_height=0.2)
        i = int(np.argmin(np.abs(t['y'] - (-15.0))))
        self.assertTrue(bool(t['platform'][i]), 'кромка распознана как платформа')
        self.assertFalse(bool(t['blocked'][i]), 'платформа помехой не считается')
        t2 = zones.tunnel_blocked(edge, p, -20.0, 0.0, half_width=wide,
                                  min_height=0.2, exclude_platform=False)
        self.assertTrue(bool(t2['blocked'][i]), 'без исключения та же геометрия -- помеха')

    def test_section_heights_are_counted_above_the_rail_head(self):
        t = zones.tunnel_blocked(np.zeros((0, 3)), tunnel_params(), -40.0, 0.0)
        self.assertAlmostEqual(t['half_width'], 1.36)
        self.assertAlmostEqual(t['z_low'], 0.10)
        self.assertAlmostEqual(t['z_high'], 3.70)
        self.assertAlmostEqual(t['rail_height'], zones.RAIL_HEIGHT)
        self.assertAlmostEqual(t['z_low_zrel'], 0.10 + zones.RAIL_HEIGHT)
        self.assertAlmostEqual(t['z_high_zrel'], 3.70 + zones.RAIL_HEIGHT)

    def test_running_rail_head_does_not_block(self):
        """Полоса в диапазоне высот рельса (0.27..0.35 над полом) -- не помеха."""
        p = tunnel_params()
        half = p.gauge / 2.0
        cloud = np.vstack([
            cloud_block((p.axis_x + half - 0.05, p.axis_x + half + 0.05),
                        (-20.0, 0.0), (0.27, 0.35), 0.01, 0.1, 0.02),
            cloud_block((p.axis_x - half - 0.05, p.axis_x - half + 0.05),
                        (-20.0, 0.0), (0.27, 0.35), 0.01, 0.1, 0.02),
        ])
        clear = zones.tunnel_blocked(cloud, p, -20.0, 0.0, min_height=0.05)
        self.assertEqual(clear['bins_blocked'], 0)
        self.assertEqual(int(clear['counts'].max()), 0, 'рельсы исключены из объёма')
        # без исключения рельсов те же точки -- препятствие
        no_excl = tunnel_params(rail_high=-0.5)
        hit = zones.tunnel_blocked(cloud, no_excl, -20.0, 0.0, min_height=0.05)
        self.assertGreater(hit['bins_blocked'], 0)

    def test_contact_rail_does_not_block_inside_a_widened_section(self):
        p = tunnel_params()
        contact_x = p.axis_x + p.contact_side * p.contact_offset
        cloud = cloud_block((contact_x - 0.05, contact_x + 0.05), (-20.0, 0.0),
                            (0.27, 0.50), 0.01, 0.1, 0.05)
        # В штатном габарите (1.36) контактный рельс вообще вне сечения: 1.45 > 1.36
        self.assertGreater(abs(p.contact_offset), zones.TUNNEL_HALF_WIDTH)
        wide = 1.60
        clear = zones.tunnel_blocked(cloud, p, -20.0, 0.0, half_width=wide,
                                     min_height=0.15)
        self.assertEqual(clear['bins_blocked'], 0)
        self.assertEqual(int(clear['counts'].max()), 0)
        hit = zones.tunnel_blocked(cloud, p, -20.0, 0.0, half_width=wide,
                                   min_height=0.15, exclude_contact=False)
        self.assertGreater(hit['bins_blocked'], 0)

    def test_platform_edge_exempts_its_bin(self):
        """Край платформы (1.40..1.55 от оси, 1.0..1.2 над УГР) -- не помеха."""
        p = tunnel_params()
        obstacle = cloud_block((1.90, 2.30), (-15.0, -14.0), (0.5, 2.0), 0.05, 0.25)
        platform = cloud_block((p.axis_x + zones.PLATFORM_FROM,
                                p.axis_x + zones.PLATFORM_TO),
                               (-14.8, -14.6), (1.16, 1.37), 0.01, 0.05, 0.05)
        cloud = np.vstack([obstacle, platform])
        exempt = zones.tunnel_blocked(cloud, p, -15.0, -14.0)
        self.assertTrue(exempt['platform'][0], 'бин с краем платформы помечается')
        self.assertEqual(exempt['bins_blocked'], 1,
                         'бин платформы не помеха, соседний бин -- помеха')
        strict = zones.tunnel_blocked(cloud, p, -15.0, -14.0, exclude_platform=False)
        self.assertEqual(strict['bins_blocked'], 2)

    def test_obstacle_is_caught_only_when_the_axis_covers_it(self):
        """Сечение ходит за осью панели -- как /set?axis= во вьюере."""
        target = cloud_block((4.90, 5.25), (-38.6, -37.4), (0.30, 1.40),
                             0.05, 0.1, 0.1)
        near = zones.tunnel_blocked(target, tunnel_params(axis_x=4.05),
                                    -40.0, -36.0)
        far = zones.tunnel_blocked(target, tunnel_params(axis_x=3.31),
                                   -40.0, -36.0)
        self.assertEqual(near['bins_total'], 5)
        self.assertGreater(near['bins_blocked'], 0)
        self.assertTrue(any(a <= -38.0 <= b for a, b in near['spans']),
                        'участок вокруг y = -38 обязан быть занят')
        self.assertEqual(far['bins_blocked'], 0, 'ось 3.31 объёмом не достаёт до цели')

    def test_empty_cloud_has_no_blocked_bins(self):
        t = zones.tunnel_blocked(np.zeros((0, 3)), tunnel_params(), -40.0, 0.0)
        self.assertEqual(t['bins_total'], 41)
        self.assertEqual(t['bins_blocked'], 0)
        self.assertEqual(t['spans'], [])
        self.assertEqual(t['y'].shape, (41,))
        self.assertEqual(t['blocked'].shape, (41,))

    def test_points_outside_the_analysis_range_do_not_count(self):
        """Облако за пределами [y_from, y_to) не должно занимать крайние бины."""
        far = cloud_block((0.5, 2.5), (-200.0, -100.0), (0.5, 2.0), 0.05, 1.0)
        t = zones.tunnel_blocked(far, tunnel_params(), -40.0, 0.0)
        self.assertEqual(t['bins_blocked'], 0)
        self.assertEqual(int(t['counts'].max()), 0)
