"""Unit tests for the testable core of bag_reader.py (parsing, cache, colours)."""
import os
import sqlite3
import struct
import sys
import time
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bag_reader  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BAG = os.path.join(ROOT, 'for_hackathon', 'doubleT_obstacle')
DB = os.path.join(BAG, 'doubleT_obstacle_0.db3')


def build_cdr(points, field_names=('x', 'y', 'z', 'intensity')):
    """Build a minimal little-endian CDR PointCloud2 blob with float32 fields."""
    bo = '<'
    out = bytearray(b'\x00\x01\x00\x00')
    out += struct.pack(bo + 'II', 0, 0)
    fid = b'lidar\x00'
    out += struct.pack(bo + 'I', len(fid)) + fid
    out += b'\x00' * (-len(out) % 4)
    out += struct.pack(bo + 'II', 1, len(points))
    out += struct.pack(bo + 'I', len(field_names))
    for i, nm in enumerate(field_names):
        name = nm.encode() + b'\x00'
        out += struct.pack(bo + 'I', len(name)) + name
        out += b'\x00' * (-len(out) % 4)
        out += struct.pack(bo + 'I', i * 4)
        out += struct.pack(bo + 'B', 7)
        out += b'\x00' * (-len(out) % 4)
        out += struct.pack(bo + 'I', 1)
    out += struct.pack(bo + 'B', 0)
    out += b'\x00' * (-len(out) % 4)
    arr = np.asarray(points, dtype='<f4')
    step = 4 * len(field_names)
    out += struct.pack(bo + 'III', step, step * len(points), arr.nbytes)
    out += arr.tobytes()
    return bytes(out)


class TestParser(unittest.TestCase):
    def test_synthetic_roundtrip(self):
        pts = [(1.0, 2.0, 3.0, 10.0), (-4.5, -60.0, 0.5, 200.0), (0.0, 0.0, 0.0, 0.0)]
        xyz, intensity = bag_reader.parse_pointcloud2_cdr(build_cdr(pts))
        self.assertEqual(xyz.shape, (2, 3), 'all-zero point must be filtered out')
        self.assertEqual(xyz.dtype, np.float32)
        np.testing.assert_allclose(xyz[0], [1.0, 2.0, 3.0], rtol=1e-6)
        np.testing.assert_allclose(xyz[1], [-4.5, -60.0, 0.5], rtol=1e-6)
        self.assertEqual(intensity.dtype, np.float32)
        np.testing.assert_allclose(intensity, [10.0, 200.0], rtol=1e-6)

    def test_synthetic_without_intensity_field(self):
        pts = [(5.0, -1.0, 0.0), (6.0, -2.0, 0.0)]
        xyz, intensity = bag_reader.parse_pointcloud2_cdr(
            build_cdr(pts, field_names=('x', 'y', 'z')))
        self.assertEqual(xyz.shape, (2, 3))
        self.assertIsNone(intensity, 'no intensity field -> None')

    def test_real_bag_matches_known_facts(self):
        if not os.path.exists(DB):
            self.skipTest('bag not present')
        conn = sqlite3.connect(DB)
        blob = conn.execute('SELECT data FROM messages ORDER BY timestamp ASC LIMIT 1').fetchone()[0]
        conn.close()
        xyz, intensity = bag_reader.parse_pointcloud2_cdr(blob)
        self.assertEqual(len(xyz), 346808)
        self.assertAlmostEqual(float(xyz[:, 0].min()), -8.67, places=1)
        self.assertAlmostEqual(float(xyz[:, 0].max()), 7.07, places=1)
        self.assertLess(float(xyz[:, 1].min()), -200.0)
        self.assertLess(float(xyz[:, 2].min()), -6.0)
        self.assertLess(float(xyz[:, 2].max()), 3.0)
        self.assertAlmostEqual(float(intensity.max()), 255.0, places=3)


class TestBagFrames(unittest.TestCase):
    def setUp(self):
        if not os.path.exists(DB):
            self.skipTest('bag not present')

    def test_len_and_topics(self):
        fr = bag_reader.BagFrames(DB, cache_size=2, prefetch_ahead=0)
        self.assertEqual(len(fr), 201)
        self.assertIn('/sensing/lidar/hesai128/pointcloud', fr.topic_names)

    def test_cache_avoids_reparse(self):
        fr = bag_reader.BagFrames(DB, cache_size=2, prefetch_ahead=0)
        fr[0]
        first = fr.parse_count
        fr[0]
        self.assertEqual(fr.parse_count, first, 'second access must hit the cache')

    def test_frame_carries_ring(self):
        fr = bag_reader.BagFrames(DB, cache_size=2, prefetch_ahead=0)
        frame = fr[0]
        self.assertEqual(len(frame.xyz), 346808)
        self.assertIsNotNone(frame.ring)
        self.assertEqual(len(frame.ring), len(frame.xyz))
        self.assertEqual(frame.ring.dtype, __import__('numpy').uint16)
        self.assertEqual(int(frame.ring.max()), 127)

    def test_ring_cache_evicts(self):
        fr = bag_reader.BagFrames(DB, cache_size=2, prefetch_ahead=0)
        for i in range(4):
            fr[i]
        self.assertLessEqual(len(fr._cache), 2)

    def test_timestamps_sorted_ns_and_span(self):
        fr = bag_reader.BagFrames(DB, cache_size=1, prefetch_ahead=0)
        self.assertTrue(np.all(np.diff(fr.timestamps) > 0))
        span_s = (fr.timestamps[-1] - fr.timestamps[0]) / 1e9
        self.assertAlmostEqual(span_s, 20.4, delta=0.5)

    def test_prefetch_fills_ahead(self):
        fr = bag_reader.BagFrames(DB, cache_size=8, prefetch_ahead=3)
        fr.want(0)
        time.sleep(4.0)
        self.assertGreaterEqual(len(fr._cache), 2, 'prefetch must load frames ahead')
        fr.close()


class TestColors(unittest.TestCase):
    def setUp(self):
        self.xyz = np.array([[0.0, 0.0, -7.0], [0.0, 0.0, 0.0], [0.0, 0.0, 3.0],
                             [1.0, -40.0, 0.0]], dtype=np.float32)
        self.inten = np.array([0.0, 0.0, 1.0, 255.0], dtype=np.float32)
        self.ring = np.array([0, 64, 127, 12], dtype=np.uint16)

    def test_shape_dtype_range_all_modes(self):
        for mode in bag_reader.MODES:
            c = bag_reader.frame_colors(self.xyz, self.inten, mode=mode,
                                         max_dist=80.0, ring=self.ring)
            self.assertEqual(c.shape, (4, 3), mode)
            self.assertEqual(c.dtype, np.float64, mode)
            self.assertTrue(np.all(c >= 0.0) and np.all(c <= 1.0), mode)
            self.assertGreater(float(c.max()), 0.5,
                               f'{mode}: colors must be bright, not 0.05-0.2')

    def test_height_mode_distinguishes_floor_and_ceiling(self):
        c = bag_reader.frame_colors(self.xyz, self.inten, mode='height')
        self.assertNotEqual(tuple(c[0]), tuple(c[2]))

    def test_range_mode_varies_with_distance(self):
        near = np.array([[0.0, -1.0, 0.0]], dtype=np.float32)
        far = np.array([[0.0, -79.0, 0.0]], dtype=np.float32)
        cn = bag_reader.frame_colors(near, None, mode='range', max_dist=80.0)
        cf = bag_reader.frame_colors(far, None, mode='range', max_dist=80.0)
        self.assertNotEqual(tuple(cn[0]), tuple(cf[0]))

    def test_intensity_mode_uses_percentiles_not_max(self):
        i = np.concatenate([np.zeros(990, dtype=np.float32),
                            np.linspace(1, 255, 10).astype(np.float32)])
        xyz = np.zeros((1000, 3), dtype=np.float32)
        c = bag_reader.frame_colors(xyz, i, mode='intensity')
        self.assertGreater(float(c.max()), 0.5,
                           'percentile normalisation must not be eaten by max=255')

    def test_missing_field_raises(self):
        with self.assertRaises(ValueError):
            bag_reader.frame_colors(self.xyz, None, mode='intensity')
        with self.assertRaises(ValueError):
            bag_reader.frame_colors(self.xyz, self.inten, mode='ring', ring=None)

    def test_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            bag_reader.frame_colors(self.xyz, self.inten, mode='nope')

    def test_empty_input(self):
        c = bag_reader.frame_colors(np.zeros((0, 3), dtype=np.float32), None, mode='height')
        self.assertEqual(c.shape, (0, 3))
        self.assertEqual(c.dtype, np.float64)


class TestTrim(unittest.TestCase):
    def test_trim_range_keeps_corridor_only(self):
        xyz = np.array([[0.0, -250.0, 0.0], [0.0, -80.0, 0.0], [0.0, 0.0, 0.0],
                        [0.0, 5.0, 0.0], [0.0, 40.0, 0.0]], dtype=np.float32)
        mask = bag_reader.trim_range(xyz, max_dist=80.0, behind=5.0)
        np.testing.assert_array_equal(mask, [False, True, True, True, False])

    def test_trim_applies_to_intensity_and_ring(self):
        xyz = np.array([[0.0, -100.0, 0.0], [0.0, -1.0, 0.0]], dtype=np.float32)
        inten = np.array([1.0, 2.0], dtype=np.float32)
        ring = np.array([7, 8], dtype=np.uint16)
        out_xyz, out_inten, out_ring = bag_reader.trim(
            xyz, inten, ring, max_dist=80.0, behind=5.0)
        self.assertEqual(len(out_xyz), 1)
        np.testing.assert_allclose(out_inten, [2.0])
        np.testing.assert_array_equal(out_ring, [8])


if __name__ == '__main__':
    unittest.main()