"""Тесты модели сенсора: она извлекается из настоящей записи и воспроизводима."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bag_reader  # noqa: E402
from sim import sensor  # noqa: E402

BAG = 'for_hackathon/doubleT_obstacle'


class TestSensorFromBag(unittest.TestCase):
    def setUp(self):
        if not os.path.exists(BAG):
            self.skipTest('bag не распакован')
        self.model = sensor.SensorModel.from_bag(bag_reader.find_db3(BAG))

    def test_has_128_rings_with_measured_elevations(self):
        self.assertEqual(self.model.rings, 128)
        self.assertAlmostEqual(self.model.elevations_deg[0], 14.40, delta=0.15)
        self.assertAlmostEqual(self.model.elevations_deg[64], -3.02, delta=0.15)
        self.assertAlmostEqual(self.model.elevations_deg[127], -25.12, delta=0.15)

    def test_vertical_step_is_non_uniform_and_kept_from_data(self):
        step = float(np.median(np.abs(np.diff(self.model.elevations_deg))))
        self.assertAlmostEqual(step, 0.168, delta=0.02)
        spread = float(np.max(np.abs(np.diff(self.model.elevations_deg))))
        self.assertGreater(spread, 0.5, 'шаг по вертикали неравномерный, это надо сохранить')

    def test_azimuth_columns_match_the_reference_frame(self):
        self.assertAlmostEqual(self.model.azimuth_columns, 2709, delta=60)

    def test_intensity_stats_and_noise_are_measured(self):
        self.assertAlmostEqual(self.model.intensity['median'], 11.0, delta=2.0)
        self.assertGreater(self.model.range_sigma_m, 0.005)
        self.assertLess(self.model.range_sigma_m, 0.10)

    def test_roundtrip_through_json(self):
        d = self.model.to_dict()
        again = sensor.SensorModel.from_dict(d)
        np.testing.assert_allclose(again.elevations_deg, self.model.elevations_deg)
        self.assertEqual(again.azimuth_columns, self.model.azimuth_columns)


if __name__ == '__main__':
    unittest.main()