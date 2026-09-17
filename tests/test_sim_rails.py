"""Тесты поиска рельсовых нитей: эталон -- синтетика, где положение рельсов известно.

Сцена ставит рельсы сама (`track_geometry_adapter.add_track`), поэтому у детектора
есть эталон: две нити на `ось ± колея/2`, на 0.16 м над ложем. На записях эталона
нет, и первой попытке (пики X в полосе высоты головки) верить было нельзя: она
давала восемь «нитей» на 2.5 м при стенах 9.2 м. Признак, который работает, --
головка ВЫСТУПАЕТ над своим ложем.
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim import cli, rails, scene  # noqa: E402
from bag_reader import BagFrames  # noqa: E402

GAUGE_M = scene.GAUGE_M
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def synthetic_clouds(out_dir: str, kind: str = 'straight', length_m: float = 60.0,
                     frames: int = 3, profile: str = None):
    """Облака синтетической сцены и её параметры (эталон для детектора)."""
    res = cli.make_dataset(out_dir=out_dir, seed=5, bags=1, frames=frames,
                           section='facts' if profile else 'wide', kind=kind,
                           length_m=length_m, profile=profile)
    bag = BagFrames(sorted(__import__('glob').glob(os.path.join(res['bags'][0], '*.db3')))[0])
    try:
        clouds = [bag[i].xyz.astype(np.float64) for i in range(len(bag))]
    finally:
        bag.close()
    return clouds, res['bags'][0]


class TestRailLinesOnSynthetic(unittest.TestCase):
    OUT = os.path.join(ROOT, '.scratch', 'rails_gt')

    @classmethod
    def setUpClass(cls):
        cls.clouds, cls.bag = synthetic_clouds(cls.OUT)
        cls.lines = rails.rail_lines(cls.clouds[0], walls=[-2.62, 6.62])

    def test_finds_the_two_rails_of_our_track(self):
        xs = sorted(l['x0'] for l in self.lines if l['points'] >= 100)
        self.assertGreaterEqual(len(xs), 2, f'нитей мало: {xs}')
        for want in (-GAUGE_M / 2, GAUGE_M / 2):
            near = min(xs, key=lambda x: abs(x - want))
            self.assertAlmostEqual(near, want, delta=0.08,
                                   msg=f'рельс на {want:+.2f} м найден на {near:+.2f}')

    def test_measured_axis_and_gauge_match_the_scene(self):
        axis = rails.track_axis(self.clouds, walls=[-2.62, 6.62])
        self.assertTrue(axis, 'пара нитей на расстоянии колеи не найдена')
        self.assertAlmostEqual(axis['axis_x'], 0.0, delta=0.08, msg='ось пути')
        self.assertAlmostEqual(axis['gauge_m'], GAUGE_M, delta=0.10, msg='колея')

    def test_steeper_criterion_than_height_band_alone(self):
        """Вибрация: убери признак «выступает» -- и нитей станет вдвое больше.

        Это ровно то, на чём провалилась первая попытка: полоса высоты головки
        есть у любого бугра ложа, и линии из них находятся не хуже рельсов.
        """
        strict = len(self.lines)
        old = rails.RAIL_ABOVE_BED_M
        rails.RAIL_ABOVE_BED_M = -1.0            # признак выключен
        try:
            loose = rails.rail_lines(self.clouds[0], walls=[-2.62, 6.62])
        finally:
            rails.RAIL_ABOVE_BED_M = old
        self.assertGreater(len(loose), strict,
                           'без признака «выступает над ложем» линий должно быть больше')


class TestRailLinesOnCurve(unittest.TestCase):
    OUT = os.path.join(ROOT, '.scratch', 'rails_gt_curve')

    def test_axis_is_found_on_a_curve_too(self):
        """На повороте нити наклонены, и трекинг с наклоном их удерживает.

        Наклон нити на повороте равен `y / R`: радиус у нас 500…3000 м, то есть на
        семи метрах вперёд это 0.002…0.014 -- почти нуль, поэтому проверяем не
        величину наклона, а то, что пара на расстоянии колеи находится и ось
        попадает туда, где её поставила сцена.
        """
        p = scene.SceneParams(seed=3, kind='curve', section='wide',
                              length_m=120.0).resolved()
        cloud, _bag = synthetic_clouds(self.OUT, kind='curve', length_m=120.0)
        axis = rails.track_axis(cloud, walls=[-2.62, 6.62])
        self.assertTrue(axis, 'на повороте нити не найдены')
        self.assertAlmostEqual(axis['axis_x'], float(p.track_axis_x), delta=0.30,
                               msg='ось на повороте')
        self.assertAlmostEqual(axis['gauge_m'], GAUGE_M, delta=0.15)
        lines = rails.rail_lines(cloud[0], walls=[-2.62, 6.62])
        strong = [l for l in lines if l['points'] >= 100]
        self.assertGreaterEqual(len(strong), 2, 'на повороте нитей мало')
        self.assertLess(max(abs(l['slope']) for l in strong), 0.10,
                        'наклон нити на наших радиусах не превышает 0.1')


if __name__ == '__main__':
    unittest.main()