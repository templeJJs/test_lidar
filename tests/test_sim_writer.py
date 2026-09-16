"""Round-trip: записанный bag читается нашим же bag_reader без расхождений."""
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bag_reader  # noqa: E402
from sim import writer  # noqa: E402

TOPIC = '/sensing/lidar/hesai128/pointcloud'
TYPE = 'sensor_msgs/msg/PointCloud2'


def frame(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    return writer.FrameOut(
        xyz=rng.normal(0.0, 5.0, size=(n, 3)).astype(np.float32),
        intensity=rng.integers(0, 256, size=n).astype(np.float32),
        ring=rng.integers(0, 128, size=n).astype(np.uint16),
        timestamp=0.1)


class TestWriter(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix='simtest_')
        self.db = os.path.join(self.dir, 'sim_0.db3')

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_roundtrip_through_bag_reader(self):
        frames = [frame(seed=i) for i in range(3)]
        writer.write_bag(frames, self.db, topic=TOPIC, msg_type=TYPE, hz=10.0)
        fr = bag_reader.BagFrames(self.db, cache_size=2, prefetch_ahead=0)
        try:
            self.assertEqual(len(fr), 3)
            self.assertIn(TOPIC, fr.topic_names)
            back = fr[1]
            np.testing.assert_allclose(back.xyz, frames[1].xyz, rtol=1e-6)
            np.testing.assert_allclose(back.intensity, frames[1].intensity, rtol=1e-6)
            np.testing.assert_array_equal(back.ring, frames[1].ring)
        finally:
            fr.close()

    def test_timestamps_ascend_from_a_fixed_start(self):
        frames = [frame(seed=i) for i in range(4)]
        writer.write_bag(frames, self.db, topic=TOPIC, msg_type=TYPE, hz=10.0,
                         start_ns=1_700_000_000_000_000_000)
        fr = bag_reader.BagFrames(self.db, cache_size=1, prefetch_ahead=0)
        try:
            ts = fr.timestamps
            self.assertTrue(bool(np.all(np.diff(ts) > 0)))
            self.assertEqual(int(ts[0]), 1_700_000_000_000_000_000)
        finally:
            fr.close()

    def test_same_seed_gives_identical_file(self):
        a = os.path.join(self.dir, 'a.db3')
        b = os.path.join(self.dir, 'b.db3')
        writer.write_bag([frame(seed=5)], a, topic=TOPIC, msg_type=TYPE, hz=10.0,
                         start_ns=1)
        writer.write_bag([frame(seed=5)], b, topic=TOPIC, msg_type=TYPE, hz=10.0,
                         start_ns=1)
        with open(a, 'rb') as fa, open(b, 'rb') as fb:
            self.assertEqual(fa.read(), fb.read())


if __name__ == '__main__':
    unittest.main()