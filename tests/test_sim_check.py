"""Сверка «сим похож на реал» (критерий 3 спеки, Task 9 плана).

Сверять надо СОПОСТАВИМЫЕ сцены, и здесь два способа это сделать:

  * `kind='straight'` + `section='wide'` повторяет семейство `doubleT_obstacle`
    (стены -2.62/+6.62 против измеренных -2.65/+6.55) -- эталон «почти тот самый
    туннель», и он проходит все допуски;
  * `section='facts'` + `profile='doubleT_obstacle'` -- ТОЧНО та запись: стены,
    уклон, высота сенсора, колонки и поле зрения берутся из её замеров, поэтому
    сходятся и стены, и покрытие азимута (`TestPinnedProfileCompare`).

Случайную сцену (`section='facts'` + `kind='curve'` без профиля) с одной записью
сверять нечестно: ширина, центр туннеля, уклон и яркость разыгрываются заново, и
стены окажутся в других местах не из-за ошибки симулятора. Именно поэтому узкая
запись используется здесь как заведомо ЧУЖАЯ сцена: так проверяется код возврата 1.

Колонки и поле зрения сенсора НЕ трогаем: и то и другое берётся из профиля записи,
и расхождение по точкам на кадр выходит 0.2 % вместо 100 % (с колонками по
умолчанию на полный оборот синтетика «видела» назад и давала 360° против 247°).
"""
import contextlib
import io
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim import check_sim, cli, scene  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL_WIDE = os.path.join(ROOT, 'for_hackathon', 'doubleT_obstacle')
REAL_NARROW = os.path.join(ROOT, 'for_hackathon', 'roundT_doubleT')
WIDE = 'doubleT_obstacle'
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
        """Семейство `wide` сверяется с записью по геометрии, но не по форме свода."""
        rep = check_sim.compare(sim_bag=self.sim_bag, real_bag=REAL_WIDE, frames=FRAMES)
        print('\n' + check_sim.format_report(rep))

        for key in ('sim_points_per_frame', 'real_points_per_frame',
                    'sim_floor_a', 'real_floor_a', 'floor_a_delta_m',
                    'sim_wall_x_left', 'sim_wall_x_right', 'wall_x_delta_m',
                    'point_count_rel', 'intensity_median_rel',
                    'elevation_max_delta_deg', 'sim_rings_used', 'real_rings_used',
                    'ring_hist_l1', 'sim_azimuth_coverage_deg',
                    'real_azimuth_coverage_deg', 'azimuth_forward_delta_deg',
                    'sim_section_top_m', 'real_section_top_m', 'section_top_delta_m',
                    'section_top_axis_delta_m',
                    'tolerances_ok', 'passed'):
            self.assertIn(key, rep, f'в отчёте нет метрики {key}')

        self.assertLess(rep['floor_a_delta_m'], 0.05, 'пол должен совпасть по уровню')
        self.assertLess(rep['wall_x_delta_m'], 0.10, 'стены должны совпасть по X')
        self.assertLess(rep['point_count_rel'], 0.25, 'точек на кадр')
        self.assertLess(rep['intensity_median_rel'], 0.20, 'медиана интенсивности')
        self.assertLess(rep['section_top_axis_delta_m'], 0.40, 'уровень свода')
        self.assertEqual(sorted(rep['tolerances_ok']), sorted(check_sim.TOLERANCES))

        # А вот ФОРМА свода у семейства `wide` сойтись и не должна: семейство --
        # идеализация с плоским сводом, а у записи он наклонный (4.90 м слева,
        # 4.32 м справа). Эталон для формы -- пришпиленный профиль
        # (`TestPinnedProfileCompare`), и там метрика проходит на всех шести записях.
        self.assertGreater(rep['section_top_delta_max_m'], 0.20,
                           'плоский свод семейства не должен выдавать себя за наклонный')
        self.assertFalse(rep['tolerances_ok']['section_top_delta_max_m'])
        self.assertFalse(rep['passed'])

        # Кольца, полное и переднее покрытие азимута у сцены и записи обязаны
        # совпадать: модель сенсора снята с записи -- и колонки (2709), и поле
        # зрения (247°). Раньше артефакт хранил только число колонок и раздавал их
        # на полный оборот, из-за чего полное покрытие расходилось 360° против 247°.
        self.assertEqual(rep['sim_rings_used'], rep['real_rings_used'])
        self.assertLess(rep['azimuth_forward_delta_deg'], 5.0,
                        'передняя полусфера должна быть покрыта одинаково')
        self.assertLess(rep['azimuth_coverage_delta_deg'], 5.0,
                        'поле зрения по азимуту -- приборная характеристика, и она из записи')

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


class TestPinnedProfileCompare(unittest.TestCase):
    """Пришпиленный профиль: сцена обязана сойтись с ЗАПИСЬЮ, а не с семьёй.

    `section='facts'` + `profile='doubleT_obstacle'` -- это тот же туннель с тем же
    полем зрения прибора, поэтому здесь проверяются ВСЕ допуски, включая полное
    покрытие азимута. Так сверка и должна работать: раньше сравнить было нечего,
    потому что случайная сцена брала ширину и центр из чужого туннеля.
    """

    @classmethod
    def setUpClass(cls):
        cls.out = tempfile.mkdtemp(prefix='simpin_')
        res = cli.make_dataset(out_dir=cls.out, seed=77, bags=1, frames=FRAMES,
                               section='facts', kind='straight',
                               profile='doubleT_obstacle')
        cls.sim_bag = res['bags'][0]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.out, ignore_errors=True)

    @unittest.skipUnless(os.path.isdir(REAL_WIDE), 'нет эталона doubleT_obstacle')
    def test_pinned_scene_reproduces_its_recording(self):
        rep = check_sim.compare(sim_bag=self.sim_bag, real_bag=REAL_WIDE, frames=FRAMES)
        self.assertTrue(rep['passed'], f'сверка не прошла: {rep["tolerances_ok"]}')
        self.assertLess(rep['wall_x_delta_m'], 0.05, 'стены повторяются точно')
        self.assertLess(rep['azimuth_coverage_delta_deg'], 5.0,
                        'поле зрения -- то же самое: 247°')
        self.assertLess(rep['point_count_rel'], 0.05, 'колонок столько же')


class TestAllPinnedProfiles(unittest.TestCase):
    """Все ШЕСТЬ записей: модель обязана воспроизводить каждую, а не две семьи.

    Цель требует модель «на основе всех датасетов», поэтому здесь сцена строится
    по профилю каждой записи и сверяется с ней. Это же и регрессия: таблица в
    `sim/README.md` получена этим прогоном.
    """

    # Кадров столько же, сколько в замере профиля (`dataset_facts --frames 5`):
    # профиль -- среднее по этим кадрам, а геометрия у части записей заметно гуляет
    # по длине (платформа, напорные ворота, стрелка -- их и видно в именах). Другое
    # подмножество кадров даёт другое среднее, и это свойство ЗАПИСИ, а не сцены:
    # при `--frames 3` расходятся 4 записи из 6 (пол до 0.02 м, стены до 0.04 м).
    FRAMES_ALL = 5
    # Медиана яркости -- единственный допуск, который у части записей близок к
    # границе (13 % при допуске 20 %), поэтому ждём именно прохождения, а не
    # точного совпадения: форму хвоста интенсивности ещё калибровать.
    MAX_INTENSITY_REL = 0.20

    @classmethod
    def setUpClass(cls):
        cls.profiles = [p['bag'] for p in scene.fact_profiles()]
        cls.out = tempfile.mkdtemp(prefix='simall_')

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.out, ignore_errors=True)

    def setUp(self):
        if not self.profiles:
            self.skipTest('нет sim/dataset_facts.json с профилями')

    def test_each_recording_is_reproduced_within_tolerances(self):
        checked, failed = [], []
        for bag in self.profiles:
            real = os.path.join(ROOT, 'for_hackathon', bag)
            if not os.path.isdir(real):
                continue
            res = cli.make_dataset(out_dir=self.out, seed=77, bags=1,
                                   frames=self.FRAMES_ALL, section='facts',
                                   kind='straight', profile=bag)
            rep = check_sim.compare(sim_bag=res['bags'][0], real_bag=real,
                                    frames=self.FRAMES_ALL)
            checked.append(bag)
            if not rep['passed']:
                failed.append((bag, {k: v for k, v in rep['tolerances_ok'].items()
                                     if not v}))
            else:
                self.assertLess(rep['point_count_rel'], 0.05,
                                f'{bag}: плотность лучей должна совпасть')
                self.assertLess(rep['azimuth_coverage_delta_deg'], 5.0,
                                f'{bag}: поле зрения по азимуту из записи')
                self.assertLess(rep['intensity_median_rel'], self.MAX_INTENSITY_REL,
                                f'{bag}: медиана яркости')
        self.assertEqual(len(checked), len(self.profiles),
                         'сверены должны быть все профили, что есть в замерах')
        self.assertGreaterEqual(len(checked), 5, 'профилей у нас шесть')
        self.assertFalse(failed, f'не прошли допуски: {failed}')

    @unittest.skipUnless(os.path.isdir(REAL_WIDE), 'нет эталона doubleT_obstacle')
    def test_cli_exit_code_is_zero_for_a_pinned_profile(self):
        """Сопоставимая сцена -- код 0 (иначе сверку нельзя ставить в CI).

        Эталон -- пришпиленный профиль, а не семейство `wide`: у семейства свод
        плоский, и метрика формы законно падает (см. `TestCheckSim`).
        """
        res = cli.make_dataset(out_dir=self.out, seed=11, bags=1, frames=3,
                               section='facts', kind='straight', profile=WIDE)
        with contextlib.redirect_stdout(io.StringIO()) as out:
            code = check_sim.main(['--sim', res['bags'][0], '--real', REAL_WIDE,
                                   '--frames', '3'])
        self.assertEqual(code, 0)
        self.assertIn('ПРОШЛА', out.getvalue())


if __name__ == '__main__':
    unittest.main()