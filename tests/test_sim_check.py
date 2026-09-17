"""Сверка «сим похож на реал» (критерий 3 спеки, Task 9 плана).

Синтетику сверяем с ОДНОЙ записью широкого семейства: `kind='straight'` +
`section='wide'` повторяет `doubleT_obstacle` (стены -2.62/+6.62 против измеренных
-2.65/+6.55). При `kind='curve'` (умолчание) `section='wide'` задаёт только
семейство -- ширина 3.6…9.2 м и центр туннеля разыгрываются заново, и сверять
такую сцену с одним эталоном нечестно.

Колонки сенсора НЕ трогаем (`columns=None`) -- плотность берётся из профиля
записи (2709): с колонками по умолчанию синтетика даёт 343 тыс. точек на кадр
против 347 тыс. у `doubleT_obstacle`, то есть расхождение 1 % вместо 100 %.

Сверка узкой сцены с одной узкой записью здесь НЕ проверяется на допуски, хотя
план предлагает `roundT_doubleT`: константа `SECTIONS['narrow']` (4.30 м) не
равна ни одной записи узкой семьи (3.6…5.0 м), и расхождение по стенам выходит за
допуск закономерно (5.00 м против 4.30 м -> |Δx| 0.42 м). Поэтому второй тест
использует узкую запись как заведомо ЧУЖУЮ сцену: так проверяется код возврата 1.
"""
import contextlib
import io
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim import check_sim, cli  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL_WIDE = os.path.join(ROOT, 'for_hackathon', 'doubleT_obstacle')
REAL_NARROW = os.path.join(ROOT, 'for_hackathon', 'roundT_doubleT')
FRAMES = 2


class TestCheckSim(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out = tempfile.mkdtemp(prefix='simchk_')
        res = cli.make_dataset(out_dir=cls.out, seed=7, bags=1, frames=FRAMES,
                               section='wide', kind='straight')
        cls.sim_bag = res['bags'][0]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.out, ignore_errors=True)

    @unittest.skipUnless(os.path.isdir(REAL_WIDE), 'нет эталона doubleT_obstacle')
    def test_metrics_match_reference_within_tolerance(self):
        rep = check_sim.compare(sim_bag=self.sim_bag, real_bag=REAL_WIDE, frames=FRAMES)
        print('\n' + check_sim.format_report(rep))

        for key in ('sim_points_per_frame', 'real_points_per_frame',
                    'sim_floor_a', 'real_floor_a', 'floor_a_delta_m',
                    'sim_wall_x_left', 'sim_wall_x_right', 'wall_x_delta_m',
                    'point_count_rel', 'intensity_median_rel',
                    'elevation_max_delta_deg', 'sim_rings_used', 'real_rings_used',
                    'ring_hist_l1', 'sim_azimuth_coverage_deg',
                    'real_azimuth_coverage_deg', 'azimuth_forward_delta_deg',
                    'tolerances_ok', 'passed'):
            self.assertIn(key, rep, f'в отчёте нет метрики {key}')

        self.assertLess(rep['floor_a_delta_m'], 0.05, 'пол должен совпасть по уровню')
        self.assertLess(rep['wall_x_delta_m'], 0.10, 'стены должны совпасть по X')
        self.assertLess(rep['point_count_rel'], 0.25, 'точек на кадр')
        self.assertLess(rep['intensity_median_rel'], 0.20, 'медиана интенсивности')
        self.assertEqual(sorted(rep['tolerances_ok']), sorted(check_sim.TOLERANCES))
        self.assertTrue(rep['passed'], f'сверка не прошла: {rep["tolerances_ok"]}')

        # Кольца и передняя полусфера у сцены и записи обязаны совпадать: модель
        # сенсора снята с записи, а в закрытом туннеле каждый луч вперёд во что-то
        # да попадает. Полное покрытие азимута НЕ проверяем -- оно расходится по
        # устройству (сцена 360.0° против 247.0° у записи): артефакт сенсора хранит
        # только число колонок и раздаёт их на полный оборот, тогда как у записей
        # поле зрения ограничено (±124° у широкой, ровно ±50° у узкой).
        self.assertEqual(rep['sim_rings_used'], rep['real_rings_used'])
        self.assertLess(rep['azimuth_forward_delta_deg'], 5.0,
                        'передняя полусфера должна быть покрыта одинаково')

    @unittest.skipUnless(os.path.isdir(REAL_WIDE), 'нет эталона doubleT_obstacle')
    def test_cli_exit_code_is_zero_when_close(self):
        """Сопоставимая сцена -- код 0 (иначе сверку нельзя ставить в CI)."""
        with contextlib.redirect_stdout(io.StringIO()) as out:
            code = check_sim.main(['--sim', self.sim_bag, '--real', REAL_WIDE,
                                   '--frames', str(FRAMES)])
        self.assertEqual(code, 0)
        self.assertIn('ПРОШЛА', out.getvalue())

    @unittest.skipUnless(os.path.isdir(REAL_NARROW), 'нет эталона roundT_doubleT')
    def test_cli_exit_code_is_one_on_wrong_family(self):
        """Узкая однопутная запись -- ДРУГАЯ сцена: сверка обязана упасть.

        Без этого теста «код 1» не проверен вовсе: допуски могли бы проходить
        всегда, и сверка молчала бы о любой сцене.
        """
        with contextlib.redirect_stdout(io.StringIO()):
            code = check_sim.main(['--sim', self.sim_bag, '--real', REAL_NARROW,
                                   '--frames', str(FRAMES)])
        self.assertEqual(code, 1)
        rep = check_sim.compare(sim_bag=self.sim_bag, real_bag=REAL_NARROW, frames=FRAMES)
        self.assertFalse(rep['passed'])
        self.assertFalse(rep['tolerances_ok']['wall_x_delta_m'])

    def test_missing_bag_is_reported_not_traced(self):
        """Опечатка в пути -- отчёт с кодом 1, а не трейсбек sqlite."""
        with contextlib.redirect_stdout(io.StringIO()) as out:
            code = check_sim.main(['--sim', self.sim_bag,
                                   '--real', os.path.join(self.out, 'нет_такой_записи'),
                                   '--frames', '1'])
        self.assertEqual(code, 1)
        self.assertIn('сверка не удалась', out.getvalue())


if __name__ == '__main__':
    unittest.main()