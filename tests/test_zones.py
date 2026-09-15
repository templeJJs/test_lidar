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
    """Контактный рельс не должен сливаться с ходовым: при contact_offset по
    умолчанию (0.70) он геометрически попадает в полосу ходового (0.76 +- 0.12),
    поэтому для изоляции признака берём вынос за пределы этой полосы."""

    def _params(self, **kw):
        base = dict(wall_x=(6.6,), contact_offset=1.30)
        base.update(kw)
        return zones.ZoneParams(**base)

    def test_contact_point_is_rail(self):
        p = self._params()
        x = p.axis_x + p.contact_offset
        z = p.floor_a + p.contact_height
        self.assertEqual(zones.ZONE_NAMES[int(zones.classify(probe(x, 0.0, z), p)[0])], 'rail')

    def test_contact_point_is_bed_when_disabled(self):
        p = self._params(contact_rail=False)
        x = p.axis_x + p.contact_offset
        z = p.floor_a + p.contact_height        # residual 0.20 < floor_band 0.30
        self.assertEqual(zones.ZONE_NAMES[int(zones.classify(probe(x, 0.0, z), p)[0])], 'bed')

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
        self.assertTrue(np.allclose(contact[:, 0], p.axis_x + p.contact_offset))
        expected_z = zones.floor_profile(contact[:, 1], p) + p.contact_height
        np.testing.assert_allclose(contact[:, 2], expected_z)


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
