"""Тесты яркости: сцена подстраивает масштаб отражения под свою запись.

У наших записей интенсивность разная: медиана 6…12, доля точек выше 25 -- от
0.7 % (узкие тоннели) до 10.8 % (широкий). Синтетика без подстройки давала 7…8 %
на всех сценах, то есть врала про узкие туннели. Теперь масштаб берётся из
профиля записи, по которому строится сцена.
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim import scene, scanner  # noqa: E402
from sim.sensor import SensorModel  # noqa: E402

SENSOR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      'sim', 'sensor_hesai128.json')


def small_scan(intensity_scale: float, columns: int = 160, seed: int = 3):
    model = SensorModel.load(SENSOR)
    model.azimuth_columns = columns
    p = scene.SceneParams(seed=seed, kind='straight', section='wide', length_m=40.0)
    p = p.resolved()
    p.intensity_scale = intensity_scale
    s = scene.build_scene(p)
    return scanner.scan(s, model, 0, np.random.default_rng(seed))


class TestIntensityScale(unittest.TestCase):
    def test_profiles_carry_measured_intensity(self):
        profiles = scene.fact_profiles()
        self.assertTrue(profiles)
        for prof in profiles:
            self.assertIn('intensity_median', prof)
            self.assertIn('share_gt25', prof)

    def test_scene_takes_scale_from_its_profile(self):
        scales = []
        for seed in range(1, 31):
            p = scene.SceneParams(seed=seed, kind='curve', section='facts',
                                  length_m=40.0).resolved()
            scales.append(float(p.intensity_scale))
        self.assertGreater(len(set(round(s, 3) for s in scales)), 3,
                           'масштаб должен зависеть от записи, а не быть константой')
        self.assertGreater(min(scales), 0.2)
        self.assertLess(max(scales), 4.0, 'масштаб не должен улетать')

    def test_brighter_scale_makes_brighter_cloud(self):
        dim = small_scan(0.6)
        bright = small_scan(1.8)
        med_dim = float(np.median(dim.intensity))
        med_bright = float(np.median(bright.intensity))
        self.assertGreater(med_bright, med_dim * 1.8,
                           f'масштаб 1.8 против 0.6 должен давать ярче: {med_bright:.1f} против {med_dim:.1f}')
        share_dim = float(np.mean(dim.intensity > 25))
        share_bright = float(np.mean(bright.intensity > 25))
        self.assertGreater(share_bright, share_dim,
                           'доля ярких точек тоже должна расти')

    def test_scale_is_deterministic_for_a_seed(self):
        a = scene.SceneParams(seed=11, kind='curve', section='facts').resolved()
        b = scene.SceneParams(seed=11, kind='curve', section='facts').resolved()
        self.assertAlmostEqual(float(a.intensity_scale), float(b.intensity_scale), places=9)


if __name__ == '__main__':
    unittest.main()