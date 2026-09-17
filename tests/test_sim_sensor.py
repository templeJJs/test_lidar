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
        """Колонок столько, чтобы ВОЗВРАТОВ вышло как у записи.

        `azimuth_columns` -- слоты азимутальной сетки, а не возвраты: у записи
        6.5 % выстрелов точки не вернули, поэтому слотов 2894, а возвратов
        2894 × (1 − 0.065) ≈ 2709 -- ровно столько, сколько в кадре записи.
        """
        self.assertGreater(self.model.dropout, 0.01, 'у записи есть пропуски возврата')
        returns = self.model.azimuth_columns * (1.0 - self.model.dropout)
        self.assertAlmostEqual(returns, 2709, delta=60)

    def test_dropout_is_measured_not_assumed(self):
        """Пропуски возврата -- приборная характеристика, и она из записи (6.4 %)."""
        self.assertGreater(self.model.dropout, 0.02)
        self.assertLess(self.model.dropout, 0.12)

    def test_azimuth_span_is_the_visible_sector(self):
        """Поле зрения по азимуту -- приборная характеристика, и она из записи.

        У `doubleT_obstacle` возвраты занимают ±124°, то есть 247° из оборота:
        хвостовой сектор 113° не снимается вовсе. Модель обязана это хранить,
        иначе колонки разложатся на полный оборот и синтетика «увидит» назад.
        """
        self.assertAlmostEqual(self.model.azimuth_span_deg, 247.0, delta=1.0)

    def test_columns_are_spread_over_the_visible_sector(self):
        """Шаг колонок: слоты на 247° -- 0.085°, у записи шаг колонок 0.0912°."""
        a = self.model.azimuths_deg()
        self.assertEqual(a.shape[0], self.model.azimuth_columns)
        self.assertAlmostEqual(float(a[0]), -123.5, delta=0.5, msg='азимут 0 -- вперёд')
        self.assertAlmostEqual(float(a[-1]), 123.5, delta=1.0)
        step = float(a[1] - a[0])
        self.assertAlmostEqual(step, 0.0912, delta=0.02)

    def test_intensity_stats_and_noise_are_measured(self):
        self.assertAlmostEqual(self.model.intensity['median'], 11.0, delta=2.0)
        self.assertGreater(self.model.range_sigma_m, 0.005)
        self.assertLess(self.model.range_sigma_m, 0.10)

    def test_roundtrip_through_json(self):
        d = self.model.to_dict()
        again = sensor.SensorModel.from_dict(d)
        np.testing.assert_allclose(again.elevations_deg, self.model.elevations_deg)
        self.assertEqual(again.azimuth_columns, self.model.azimuth_columns)
        self.assertEqual(again.azimuth_span_deg, self.model.azimuth_span_deg)


if __name__ == '__main__':
    unittest.main()