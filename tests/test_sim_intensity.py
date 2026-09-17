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


class TestIntensityTail(unittest.TestCase):
    """Хвост яркости: σ подбирается так, чтобы воспроизвести долю точек выше 25.

    Прежняя линейная подгонка от доли не сходилась: при ОДНОЙ И ТОЙ ЖЕ доле у
    записей с медианой 7 и 11 нужны разные σ (0.77 против 0.44), а у записи с
    медианой 7 и долей 4.3 % синтетика давала 0.3 % против 4.3 %. Здесь проверяется
    модель `σ = ln(25 / медиана) / z(1 − доля)` -- на доле и медиане СВОЕЙ записи.
    """

    MAX_REL_ERROR = 0.35        # замеренные расхождения 9…21 % (см. sim/README.md)

    def test_normal_quantiles_are_exact_enough(self):
        self.assertAlmostEqual(scene._norm_quantile(0.5), 0.0, places=6)
        self.assertAlmostEqual(scene._norm_quantile(0.975), 1.95996, places=4)
        self.assertAlmostEqual(scene._norm_quantile(0.9), 1.28155, places=4)
        self.assertAlmostEqual(scene._norm_quantile(0.025), -1.95996, places=4)

    def test_sigma_grows_with_share_and_falls_with_median(self):
        """Смысл модели: толще хвост -- больше σ, ярче сцена -- меньше σ."""
        self.assertGreater(scene.intensity_sigma_for(11.0, 0.10),
                           scene.intensity_sigma_for(11.0, 0.02))
        self.assertGreater(scene.intensity_sigma_for(7.0, 0.05),
                           scene.intensity_sigma_for(11.0, 0.05))
        lo, hi = scene.INTENSITY_SIGMA_RANGE
        for median, share in ((11.0, 0.5), (7.0, 1e-6), (0.0, 0.05)):
            self.assertGreaterEqual(scene.intensity_sigma_for(median, share), lo)
            self.assertLessEqual(scene.intensity_sigma_for(median, share), hi)

    def test_share_reproduces_the_recording_it_came_from(self):
        """Каждая запись: синтетика даёт ту же долю точек выше 25, что и запись.

        Колонки и поле зрения берём из профиля (как `sim.cli`): при поле зрения
        247° на узкой сцене лучи уходят в стены, а стены ярче ложа, и доля >25
        вырастает вдвое -- проверяли бы не хвост, а ошибку в поле зрения.
        """
        model = SensorModel.load(SENSOR)
        checked = 0
        for prof in scene.fact_profiles():
            target = float(prof['share_gt25'])
            model.azimuth_columns = int(prof['columns'])
            model.azimuth_span_deg = float(prof['azimuth_span_deg'])
            p = scene.SceneParams(seed=5, kind='straight', section='facts',
                                  length_m=50.0, fact_bag=prof['bag']).resolved()
            s = scene.build_scene(p)
            f = scanner.scan(s, model, 0, np.random.default_rng(4))
            share = float(np.mean(f.intensity > 25.0))
            rel = abs(share - target) / target if target else abs(share - target)
            self.assertLess(rel, self.MAX_REL_ERROR,
                            f'{prof["bag"]}: доля >25 {share:.4f} против {target:.4f} '
                            f'в записи (σ {p.intensity_sigma:.3f})')
            checked += 1
        self.assertGreaterEqual(checked, 5, 'проверяем по всем записям, а не по одной')


class TestIntensityScale(unittest.TestCase):
    def test_profiles_carry_measured_intensity(self):
        profiles = scene.fact_profiles()
        self.assertTrue(profiles)
        for prof in profiles:
            self.assertIn('intensity_median', prof)
            self.assertIn('share_gt25', prof)

    def test_scene_takes_scale_from_its_profile(self):
        """Масштаб -- ровно от медианы той записи, чей профиль взяла сцена.

        Проверяем не «разные числа», а связь: у записей медианы 7, 9 и 11, и
        масштаб равен `медиана / SIM_MEDIAN_AT_SCALE_1` (в пределах зажима), иначе
        синтетика врала бы про яркость узких туннелей.
        """
        medians = {pr['bag']: float(pr['intensity_median'])
                   for pr in scene.fact_profiles()}
        self.assertGreaterEqual(len(set(medians.values())), 3,
                                'записи должны различаться по яркости')
        scales = []
        for seed in range(1, 31):
            p = scene.SceneParams(seed=seed, kind='curve', section='facts',
                                  length_m=40.0).resolved()
            want = min(max(medians[p.fact_bag] / scene.SIM_MEDIAN_AT_SCALE_1, 0.35), 2.5)
            self.assertAlmostEqual(float(p.intensity_scale), want, places=9,
                                   msg=f'масштаб не от записи {p.fact_bag}')
            scales.append(float(p.intensity_scale))
        self.assertGreater(len(set(round(s, 3) for s in scales)), 2,
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