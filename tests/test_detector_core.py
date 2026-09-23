"""Тесты ядра детектора габарита (`detector/core.py`) и дальнего контура.

Обычный `unittest`, без ROS и open3d. Две группы:

* СИНТЕТИКА -- управляемая сцена: отдельно рельс, корыто, ложе, лоток, стена,
  свод и внесённый объект. Проверяется, что ловится ТОЛЬКО объект;
* РЕАЛЬНЫЕ КАДРЫ -- по одному кадру на запись `for_hackathon/`: настоящих
  объектов в этих записях нет, поэтому правильный ответ -- ноль препятствий.
  Отдельно проверяется боковой разделитель `doubleT_obstacle` (та самая стена,
  которая при прямой оси давала ложный класс срабатываний).
"""

import os
import sys
import time
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detector import DetectorConfig, TrackModel, detect  # noqa: E402
from detector.core import dense_span, split_intervals, threshold_points  # noqa: E402
from detector.contour import (  # noqa: E402
    ContourTracker,
    WallProfile,
    WidthTracker,
    free_width,
    wall_profile,
)
from detector.profiles import model_for_name, record_name  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'for_hackathon')

RECORDS = (
    'doubleT_obstacle',
    'doubleT_platform',
    'roundT_doubleT',
    'roundT_pressureGate_roundT',
    'roundT_squareT_pressureGate_squareT',
    'squareT_platform_squareT_switch',
)


def make_model(axis_i=0.0, ugr=-1.20, gauge=1.596, y_lo=-60.0, y_hi=-2.0,
               step=1.0) -> TrackModel:
    """Прямой путь для синтетики: ось `axis_i`, УГР `ugr` на всём участке."""
    ys = np.arange(y_lo, y_hi + 1e-9, step)
    return TrackModel(
        name='synthetic',
        gauge_m=gauge,
        y_nodes=tuple(float(v) for v in ys),
        axis_x_m=tuple([float(axis_i)] * ys.size),
        ugr_z_m=tuple([float(ugr)] * ys.size),
    )


def to_xyz(model, u, h, y):
    """(u, h, y) -> точки в системе лидара."""
    u = np.asarray(u, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = u + np.asarray(model.axis_x(y), dtype=np.float64)
    z = h + np.asarray(model.ugr_z(y), dtype=np.float64)
    return np.column_stack([x, y, z]).astype(np.float32)


def strip(model, u_lo, u_hi, h_lo, h_hi, y_lo=-50.0, y_hi=-4.0, du=0.05, dy=0.25):
    """Плитка точек в прямоугольнике (u x h x y)."""
    y_axis = np.arange(y_lo, y_hi + 1e-9, dy)
    u_axis = np.arange(u_lo, u_hi + 1e-9, du)
    h_axis = np.arange(h_lo, h_hi + 1e-9, du)
    uu, hh, yy = np.meshgrid(u_axis, h_axis, y_axis)
    return to_xyz(model, uu.ravel(), hh.ravel(), yy.ravel())


class TestSyntheticNegative(unittest.TestCase):
    """Норма тоннеля не должна считаться препятствием."""

    def setUp(self):
        self.model = make_model()
        self.cfg = DetectorConfig()

    def check_empty(self, cloud, label):
        result = detect(cloud, model=self.model, cfg=self.cfg)
        self.assertEqual(
            result.obstacles, [],
            f'{label}: ложное срабатывание {[o.to_dict() for o in result.obstacles]}')

    def test_rails(self):
        cloud = np.vstack([strip(self.model, s * 0.798 - 0.07, s * 0.798 + 0.07,
                                 -0.10, 0.10, dy=0.1) for s in (-1.0, 1.0)])
        self.check_empty(cloud, 'ходовые рельсы')

    def test_bed_and_floor(self):
        self.check_empty(strip(self.model, -0.90, 0.90, 0.10, 0.40), 'корыто')
        self.check_empty(strip(self.model, -1.70, 1.70, -0.20, 0.25), 'ложе/пол')

    def test_cable_tray(self):
        cloud = np.vstack([strip(self.model, s * 1.25, s * 1.50, 0.15, 0.60)
                           for s in (-1.0, 1.0)])
        self.check_empty(cloud, 'кабельный лоток')

    def test_vault_above_volume(self):
        self.check_empty(strip(self.model, -1.20, 1.20, 3.90, 4.60), 'свод выше объёма')

    def test_wall_outside_gauge(self):
        self.check_empty(strip(self.model, 1.40, 2.50, 0.40, 3.20, du=0.05),
                         'стена за габаритом')

    def test_side_separator_whose_edge_enters_the_gauge(self):
        """Стена, КРАЙ которой заходит внутрь габарита, -- не препятствие.

        Это профиль бокового разделителя `doubleT_obstacle`: занятый интервал по
        u не заканчивается внутри габарита, а уходит наружу. Именно на нём
        ломались счётные критерии.
        """
        cloud = strip(self.model, 0.55, 2.60, 0.30, 3.40, du=0.05, dy=0.25)
        self.check_empty(cloud, 'разделитель, заходящий в габарит краем')

    def test_structure_continuing_outside_neighbouring_bins(self):
        """Внутренний обрывок стены не выдаётся: связность важнее одного бина.

        Профиль `roundT_doubleT` (замерено: ложные события на кадрах 107..152):
        наружная часть стены -- УЗКАЯ полоса вне габарита (ширина 0.15 м, ниже
        `min_width_m`), кандидатом она не становится, и склейка по y на ней
        обрывается; внутренний же обрывок (u 0.90..1.24, размах 0.9 м -- как у
        человека) выглядит компактным предметом. Связность занятости в (y, u)
        видит, что обрывок -- часть той же структуры, и отбрасывает его.

        Контроль: со снятым структурным правилом (`max_structure_y_m` и
        `structure_out_m` в бесконечность) тот же кадр даёт ложное срабатывание
        -- значит, тест проверяет именно это правило, а не что-то ещё.
        """
        outside = strip(self.model, 1.30, 1.45, 0.30, 0.70,
                        y_lo=-26.0, y_hi=-13.0, du=0.05, dy=0.25)
        inside = strip(self.model, 0.90, 1.24, 0.30, 1.20,
                       y_lo=-12.0, y_hi=-10.0, du=0.05, dy=0.25)
        cloud = np.vstack([outside, inside])
        off = DetectorConfig(max_structure_y_m=1e9, structure_out_m=1e9)
        old = detect(cloud, model=self.model, cfg=off)
        self.assertTrue(old.obstacles,
                        'контроль не сработал: со снятым структурным правилом '
                        'обрывок обязан выдаваться, иначе тест ничего не проверяет')
        self.check_empty(cloud, 'обрывок стены, связанный с наружной частью')

    def test_wall_filling_volume_height_is_ignored(self):
        """Стена, заполнившая объём по высоте, -- не предмет (размах высоты).

        Замерено на всём корпусе: у человека размах 0.65..0.90 м, у стены,
        зашедшей в габарит наклонной поверхностью, -- 1.8..3.0 м.
        """
        cloud = strip(self.model, -0.60, 0.60, 0.40, 3.40,
                      y_lo=-16.0, y_hi=-14.0, du=0.05, dy=0.25)
        off = DetectorConfig(max_span_m=1e9)
        self.assertTrue(detect(cloud, model=self.model, cfg=off).obstacles,
                        'контроль не сработал: без предела размаха стена не выдаётся')
        self.check_empty(cloud, 'стена на всю высоту объёма')

    def test_long_wall_along_path(self):
        """Протяжённая вдоль пути структура отбрасывается по длине кластера."""
        cloud = strip(self.model, -0.40, 0.40, 0.40, 2.60, y_lo=-45.0, y_hi=-5.0,
                      du=0.05, dy=0.25)
        self.check_empty(cloud, 'стена вдоль пути по центру габарита')

    def test_empty_and_zero_input(self):
        self.assertEqual(detect(np.zeros((0, 3), dtype=np.float32),
                                model=self.model).obstacles, [])
        self.assertEqual(detect(np.zeros((10, 3), dtype=np.float32),
                                model=self.model).obstacles, [])


class TestSyntheticPositive(unittest.TestCase):
    """Внесённый объект обязан находиться, с верной дальностью."""

    def setUp(self):
        self.model = make_model()
        self.cfg = DetectorConfig()

    def test_object_detected_at_several_ranges(self):
        from detector.core import synthetic_object

        for y_obj in (-10.0, -20.0, -35.0, -50.0):
            with self.subTest(y=y_obj):
                obj = synthetic_object(y_obj, self.model)
                result = detect(obj, model=self.model, cfg=self.cfg)
                self.assertTrue(result.obstacles, f'объект на {y_obj} м не найден')
                found = result.nearest()
                self.assertAlmostEqual(found.y_m, y_obj, delta=1.5)
                self.assertAlmostEqual(found.distance_m, abs(y_obj), delta=2.5)
                self.assertGreaterEqual(found.span_m, self.cfg.min_span_m)

    def test_object_off_axis(self):
        from detector.core import synthetic_object

        for u_center in (-0.6, -0.3, 0.0, 0.3, 0.6):
            with self.subTest(u=u_center):
                obj = synthetic_object(-25.0, self.model, u_center_m=u_center)
                result = detect(obj, model=self.model, cfg=self.cfg)
                self.assertTrue(result.obstacles, f'объект при u={u_center} не найден')
                self.assertAlmostEqual(result.nearest().u_m, u_center, delta=0.2)

    def test_object_at_gauge_edge_is_reported(self):
        """Объект у самой кромки габарита выдаётся.

        Прежнее правило требовало, чтобы занятый интервал замыкался до
        `|u| = 1.25 - 0.18`: человек, идущий по кромке габарита на 56 м
        (`doubleT_obstacle`, интервал до -1.24 м), отбраковывался вместе со
        следом стены. Различает их длина кластера вдоль пути: у стены она
        десятки метров, у предмета -- метры. Поэтому теперь внутри габарита
        объект выдаётся, даже если доходит до самой границы.
        """
        from detector.core import synthetic_object

        obj = synthetic_object(-25.0, self.model, u_center_m=0.80)
        result = detect(obj, model=self.model, cfg=self.cfg)
        self.assertTrue(result.obstacles, 'объект у кромки габарита не найден')
        found = result.nearest()
        self.assertAlmostEqual(found.u_m, 0.80, delta=0.2)
        self.assertLessEqual(abs(found.u_m), self.cfg.half_width_m)

    def test_person_sized_object_is_reported(self):
        """Объект с силуэтом человека (0.9 м) выдаётся.

        Отсечки «не предмет» (`max_span_m`, связность занятости) проверяются на
        том, что настоящий предмет они не задевают: у человека в
        `doubleT_obstacle` размах 0.65..0.90 м, у внесённого объекта -- 0.90 м.
        """
        from detector.core import synthetic_object

        obj = synthetic_object(-15.0, self.model, height_m=0.90)
        result = detect(obj, model=self.model, cfg=self.cfg)
        self.assertTrue(result.obstacles, 'объект высотой 0.9 м не найден')
        found = result.nearest()
        self.assertLess(found.span_m, self.cfg.max_span_m)
        self.assertAlmostEqual(found.y_m, -15.0, delta=1.5)

    def test_object_outside_gauge_is_not_reported(self):
        """Объект ЗА габаритом не выдаётся: внутрь занятый интервал не заходит.

        Ровно так выглядит след стены -- интервал начинается за `|u| = 1.25` и
        наружу продолжается. По одним точкам такой интервал от предмета не
        отличить, поэтому за габаритом не выдаём ничего.
        """
        from detector.core import synthetic_object

        obj = synthetic_object(-25.0, self.model, u_center_m=1.75)
        result = detect(obj, model=self.model, cfg=self.cfg)
        self.assertEqual(result.obstacles, [])

    def test_object_detected_against_real_frame(self):
        """Объект виден на фоне реального кадра, а не только на пустом облаке."""
        from detector.core import synthetic_object

        cloud = self._real_frame('roundT_doubleT')
        if cloud is None:
            self.skipTest('нет данных for_hackathon/')
        model = model_for_name('roundT_doubleT')
        for y_obj in (-15.0, -25.0):
            with self.subTest(y=y_obj):
                obj = synthetic_object(y_obj, model)
                result = detect(np.vstack([cloud, obj]), model=model)
                near = [o for o in result.obstacles if abs(o.y_m - y_obj) < 3.0]
                self.assertTrue(near, f'объект на {y_obj} м не найден на реальном кадре')

    def _real_frame(self, name):
        path = os.path.join(DATA, name, f'{name}_0.db3')
        if not os.path.exists(path):
            return None
        import bag_reader

        frames = bag_reader.BagFrames(path, prefetch_ahead=0)
        try:
            return frames[0].xyz
        finally:
            frames.close()


class TestTroughChannel(unittest.TestCase):
    """Канал лотка: лежащие ниже основного объёма объекты.

    Освободить место -- окно h in [-0.45, +0.05] с маской u-полос ниток:
    тело лежащего (h примерно -0.37..+0.11) в основной объём (низ +0.30) не
    попадает, и без канала recall по позам lying/fallen был 0.00..0.29.
    """

    def setUp(self):
        self.model = make_model()
        self.cfg = DetectorConfig()

    def scene(self, *parts):
        return np.vstack(parts)

    def floor_and_rails(self):
        """Дно лотка (-0.25) и нитки рельсов -- норма, не препятствие."""
        floor = strip(self.model, -0.65, 0.65, -0.26, -0.24, du=0.02, dy=0.25)
        rails = np.vstack([strip(self.model, s * 0.798 - 0.07, s * 0.798 + 0.07,
                                 -0.10, 0.16, du=0.02, dy=0.25) for s in (-1.0, 1.0)])
        return floor, rails

    def test_floor_and_rails_are_empty(self):
        floor, rails = self.floor_and_rails()
        result = detect(self.scene(floor, rails), model=self.model, cfg=self.cfg)
        self.assertEqual(result.obstacles, [])
        self.assertGreater(result.points_trough, 0, 'канал лотка не смотрит в окно')

    def test_lying_body_in_trough_is_reported(self):
        """Тело, лежащее в лотке (верх ниже +0.30), выдаётся каналом лотка."""
        floor, rails = self.floor_and_rails()
        body = strip(self.model, 0.10, 0.90, -0.25, 0.05,
                     y_lo=-21.0, y_hi=-20.0, du=0.03, dy=0.10)
        result = detect(self.scene(floor, rails, body), model=self.model, cfg=self.cfg)
        self.assertEqual(len(result.obstacles), 1,
                         f'ожидался 1 объект: {[o.to_dict() for o in result.obstacles]}')
        found = result.nearest()
        self.assertAlmostEqual(found.y_m, -20.5, delta=0.5)
        self.assertAlmostEqual(found.u_m, 0.50, delta=0.15)
        self.assertLessEqual(found.h_max_m, 0.06, 'верх объекта лотка выше окна канала')

    def test_body_across_rail_line_is_reported(self):
        """Тело поперёк пути совпадает с ниткой по u -- видны куски вне полосы."""
        floor, rails = self.floor_and_rails()
        body = strip(self.model, 0.30, 1.20, -0.25, 0.05,
                     y_lo=-21.0, y_hi=-20.0, du=0.03, dy=0.10)
        result = detect(self.scene(floor, rails, body), model=self.model, cfg=self.cfg)
        self.assertTrue(result.obstacles,
                        'тело поперёк нитки не найдено (маска съела оба куска?)')

    def test_beam_crossing_the_gauge_is_not_reported(self):
        """Балка на уровне дна через всю колею -- конструкция, не предмет.

        Профиль порога гермозатвора: нижняя балка лежит на уровне дна поперёк
        колеи и рельсами режется на три куска; правило «занятость за |u| > 0.90
        с обеих сторон» обязано её снять (замер: 25 ложных кадров без правила).
        """
        floor, rails = self.floor_and_rails()
        beam = strip(self.model, -1.25, 1.25, -0.25, -0.10,
                     y_lo=-21.2, y_hi=-20.8, du=0.03, dy=0.10)
        result = detect(self.scene(floor, rails, beam), model=self.model, cfg=self.cfg)
        self.assertEqual(result.obstacles, [])

    def test_standing_object_is_not_duplicated(self):
        """Стоящий объект достаётся основному объёму, лоток его не дублирует."""
        floor, rails = self.floor_and_rails()
        standing = strip(self.model, -0.30, 0.30, -0.25, 1.50,
                         y_lo=-21.0, y_hi=-20.0, du=0.05, dy=0.25)
        result = detect(self.scene(floor, rails, standing), model=self.model, cfg=self.cfg)
        self.assertEqual(len(result.obstacles), 1,
                         f'объект задублирован: {[o.to_dict() for o in result.obstacles]}')
        self.assertGreaterEqual(result.obstacles[0].h_min_m, 0.30,
                                'объект выдан каналом лотка, а не объёмом')

    def test_trough_disabled_keeps_old_behavior(self):
        """Выключенный канал -- прежнее поведение: лежащее тело не выдаётся."""
        floor, rails = self.floor_and_rails()
        body = strip(self.model, 0.10, 0.90, -0.25, 0.05,
                     y_lo=-21.0, y_hi=-20.0, du=0.03, dy=0.10)
        cfg = DetectorConfig(trough_enabled=False)
        result = detect(self.scene(floor, rails, body), model=self.model, cfg=cfg)
        self.assertEqual(result.obstacles, [])
        self.assertEqual(result.points_trough, 0)

    def test_floor_baseline_survives_the_body(self):
        """Опора дна (нижний квантиль ячейки (y, u)) не поднимается телом.

        Тело добавляет точки СВЕРХУ дна: квантиль остаётся на дне, и возвышение
        тела видно. Медиану тело вытесняло -- возвышение обнулялось (замер).
        """
        from detector.core import floor_baseline, trough_channel_mask

        floor, rails = self.floor_and_rails()
        body = strip(self.model, 0.10, 0.90, -0.25, 0.05,
                     y_lo=-21.0, y_hi=-20.0, du=0.03, dy=0.10)
        cloud = self.scene(floor, rails, body)
        u, h = self.model.relative(np.asarray(cloud, dtype=np.float64))
        m = trough_channel_mask(u, h, self.model, self.cfg)
        base = floor_baseline(u[m], h[m], np.asarray(cloud, dtype=np.float64)[m, 1],
                              self.cfg)
        ok = ~np.isnan(base)
        self.assertGreater(np.percentile(base[ok], 75), -0.28,
                           'опора дна уехала вниз')
        self.assertLess(np.percentile(base[ok], 75), -0.22,
                        'опору дна подняло телом (медиана вместо квантиля)')


class TestCanonicalization(unittest.TestCase):
    """Доворот датчика: ось кадра приводится к канонической системе модели.

    Замерено на `roundT_doubleT`: ось, измеренная по рельсам, уезжает в системе
    кадра на 0.9 м на 10 м и на 4 м на 40 м от кадра к кадру -- датчик
    доворачивается между кадрами. Снимок собран по НЕСКОЛЬКИМ кадрам, поэтому
    без канонизации габарит считается по чужой оси, и участок приходится
    укорачивать (было −45.75 против −63.75 на прямых записях).

    Здесь проверяется сама операция: облако кадра с доворотом `slope` и
    КАНОНИЧЕСКАЯ модель (ось `x = 0`), объект внесён на ось кадра. Без
    канонизации объект уезжает по `u` на `slope*y` (1.25 м на 25 м) и теряется.
    """

    SLOPE = 0.05          # доворот, рад: 0.05*25 = 1.25 м на 25 м
    Y_OBJ = -25.0

    def setUp(self):
        self.model = make_model()
        self.cfg = DetectorConfig()

    def frame_frame(self, cloud):
        """Облако в системе КАДРА: ось кадра наклонена на `SLOPE`."""
        out = np.array(cloud, dtype=np.float64)
        out[:, 0] += self.SLOPE * out[:, 1]
        return out

    def rails(self):
        """Полосы рельсов в канонической системе.

        Сетка по `u` МЕЛЬЧЕ полосы (0.01 против `CANON_RAIL_BAND_M` = 0.04):
        при шаге 0.05 точки полосы стоят столбцами через 0.05 м, и сдвиг на
        такой шаг переводит столбец в столбец -- по счёту это то же самое, то
        есть облако вырождается в гребёнку с неоднозначным сдвигом (замерено:
        на такой гребёнке сетка промахивалась на 0.05 м и уточнение
        отбраковывалось). Настоящие рельсы -- сплошная поверхность, поэтому и в
        тесте полоса обязана быть плотной.
        """
        return np.vstack([strip(self.model, s * 0.798 - 0.07, s * 0.798 + 0.07,
                                -0.10, 0.10, y_lo=-30.0, y_hi=-4.0, du=0.01, dy=0.2)
                          for s in (-1.0, 1.0)])

    def test_pose_is_taken_from_the_cloud(self):
        """Наклон кадра меряется по полосам рельсов облака, а не по модели."""
        from detector.core import axis_pose

        pose = axis_pose(self.frame_frame(self.rails()), self.model)
        self.assertTrue(pose.applied, 'полосы рельсов не найдены')
        self.assertAlmostEqual(pose.slope, self.SLOPE, delta=0.002)
        self.assertAlmostEqual(pose.shift_m, 0.0, delta=0.03)

    def test_object_on_axis_is_found_in_rotated_frame(self):
        from detector.core import synthetic_object

        cloud = self.frame_frame(np.vstack(
            [self.rails(), synthetic_object(self.Y_OBJ, self.model)]))
        result = detect(cloud, model=self.model, cfg=self.cfg)
        near = [o for o in result.obstacles if abs(o.y_m - self.Y_OBJ) < 3.0]
        self.assertTrue(near, f'объект на оси кадра не найден: '
                              f'{[o.to_dict() for o in result.obstacles]}')
        self.assertAlmostEqual(near[0].u_m, 0.0, delta=0.2)
        self.assertTrue(result.pose['applied'])

    def test_without_canonicalization_the_object_is_lost(self):
        """Контроль: на выключенной канонизации тот же кадр объект теряет.

        Значит, тест проверяет именно канонизацию, а не что-то ещё: на 25 м
        доворот уводит объект на 1.25 м, интервал выходит за габарит и
        отбраковывается как продолжение структуры наружу.
        """
        from detector.core import synthetic_object

        cloud = self.frame_frame(np.vstack(
            [self.rails(), synthetic_object(self.Y_OBJ, self.model)]))
        result = detect(cloud, model=self.model,
                        cfg=DetectorConfig(canonicalize=False))
        near = [o for o in result.obstacles if abs(o.y_m - self.Y_OBJ) < 3.0]
        self.assertEqual(near, [], f'без канонизации объект не должен находиться: '
                                   f'{[o.to_dict() for o in near]}')

    def test_no_rails_means_cloud_is_not_touched(self):
        """Без полос рельсов облако не сдвигается: синтетика не меняется."""
        from detector.core import axis_pose, synthetic_object

        obj = synthetic_object(-25.0, self.model, u_center_m=0.80)
        pose = axis_pose(obj, self.model)
        self.assertFalse(pose.applied, 'поза взялась из облака без рельсов')
        result = detect(obj, model=self.model, cfg=self.cfg)
        self.assertTrue(result.obstacles, 'объект у кромки потерян')
        self.assertAlmostEqual(result.nearest().u_m, 0.80, delta=0.2)
        self.assertFalse(result.pose['applied'])


class TestHelpers(unittest.TestCase):
    def test_split_intervals(self):
        values = np.array([0.0, 0.01, 0.02, 0.5, 0.51, 2.0])
        parts = split_intervals(values, 0.12)
        self.assertEqual(len(parts), 3)
        self.assertAlmostEqual(parts[0][0], 0.0)
        self.assertAlmostEqual(parts[0][1], 0.02)
        self.assertAlmostEqual(parts[2][0], 2.0)

    def test_dense_span_ignores_sparse_tail(self):
        dense = np.linspace(0.0, 1.5, 100)
        sparse = np.r_[dense, np.linspace(2.0, 4.0, 10)]
        self.assertGreater(dense_span(dense), 1.0)
        self.assertLess(dense_span(sparse), 1.7,
                        'редкий хвост не должен раздувать размах')

    def test_threshold_points_normalization(self):
        cfg = DetectorConfig(range_normalization=False)
        self.assertEqual(threshold_points(100.0, cfg), cfg.min_points)
        cfg2 = DetectorConfig(range_normalization=True, range_ref_m=10.0)
        self.assertLess(threshold_points(100.0, cfg2), cfg2.min_points)
        self.assertEqual(threshold_points(5.0, cfg2), cfg2.min_points)

    def test_model_valid_range_uses_extension(self):
        model = make_model(y_lo=-30.0, y_hi=-2.0)
        far, near = model.valid_range()
        self.assertAlmostEqual(far, -30.0 - model.axis_extension_m, places=6)
        self.assertAlmostEqual(near, -4.0, places=6)

    def test_model_interpolates_axis(self):
        model = TrackModel(name='t', gauge_m=1.6,
                           y_nodes=(-20.0, -10.0, 0.0),
                           axis_x_m=(0.0, 0.5, 1.0),
                           ugr_z_m=(-1.0, -1.0, -1.0))
        np.testing.assert_allclose(model.axis_x(np.array([-15.0, 0.0, 10.0])),
                                   [0.25, 1.0, 1.5], atol=1e-9)
        self.assertAlmostEqual(float(model.axis_x(np.array([-30.0]))[0]), -0.5, places=9)


class TestFarContour(unittest.TestCase):
    def test_free_width_on_synthetic_walls(self):
        model = make_model()
        cloud = np.vstack([
            strip(model, -3.20, -2.00, 0.40, 3.00, du=0.05, dy=0.30),
            strip(model, 2.00, 3.20, 0.40, 3.00, du=0.05, dy=0.30),
        ])
        y, left, right = free_width(cloud, model)
        wide = (left + right) > 3.0
        self.assertTrue(wide.any(), 'свободная ширина не измерена')

    def test_vault_tracker_fires_on_shift(self):
        model = make_model()
        tracker = ContourTracker(window=3, persist_bins=1)
        base = strip(model, -1.00, 1.00, 4.10, 4.40, dy=0.5)
        for _ in range(3):
            tracker.update(wall_profile(base, model))
        low = strip(model, -1.00, 1.00, 3.40, 3.70, dy=0.5)
        events = tracker.update(wall_profile(low, model))
        self.assertTrue(any(e['kind'] == 'vault' for e in events),
                        f'смещение свода не поймано: {events}')

    def test_width_tracker_fires_on_narrowing(self):
        y = np.arange(-20.0, -2.0, 1.0)
        tracker = WidthTracker(window=3, persist_bins=1, narrow_m=0.3)
        for _ in range(3):
            tracker.update(np.full(y.size, 1.4), np.full(y.size, 1.4), y)
        events = tracker.update(np.full(y.size, 1.0), np.full(y.size, 0.9), y)
        self.assertTrue(any(e['kind'] == 'width_narrow' for e in events))

    def test_no_events_before_history_is_full(self):
        model = make_model()
        tracker = ContourTracker(window=5)
        events = tracker.update(wall_profile(strip(model, -1.0, 1.0, 4.0, 4.4), model))
        self.assertEqual(events, [])


class TestRealRecords(unittest.TestCase):
    """На реальных записях настоящих объектов нет: правильный ответ -- ноль."""

    def frame(self, name, index):
        path = os.path.join(DATA, name, f'{name}_0.db3')
        if not os.path.exists(path):
            self.skipTest(f'нет записи {name}')
        import bag_reader

        frames = bag_reader.BagFrames(path, prefetch_ahead=0)
        try:
            return frames[index].xyz
        finally:
            frames.close()

    def test_no_false_positives_one_frame_per_record(self):
        rows = []
        for name in RECORDS:
            with self.subTest(record=name):
                cloud = self.frame(name, 100)
                model = model_for_name(name)
                result = detect(cloud, model=model)
                rows.append((name, result.bins_blocked, [o.to_dict() for o in result.obstacles]))
                self.assertEqual(result.obstacles, [], f'{name}: {rows[-1][2]}')

    def test_doubleT_obstacle_side_separator_ignored(self):
        """Тот самый разделитель, из-за которого при прямой оси были срабатывания."""
        cloud = self.frame('doubleT_obstacle', 100)
        model = model_for_name('doubleT_obstacle')
        local = detect(cloud, model=model)
        straight = detect(cloud, model=model_for_name('doubleT_obstacle', straight=True))
        self.assertEqual(local.obstacles, [],
                         f'локальная ось дала срабатывание: {[o.to_dict() for o in local.obstacles]}')
        # контроль: на прямой оси ось уводит габарит, и появляется другая картина
        self.assertEqual(straight.axis_range_m, (-150.0, -4.0))

    def test_kernel_time_budget(self):
        """Ядро на самом большом кадре (347 тыс. точек) укладывается в бюджет."""
        cloud = self.frame('doubleT_obstacle', 0)
        model = model_for_name('doubleT_obstacle')
        detect(cloud, model=model)                     # разогрев numpy
        start = time.perf_counter()
        detect(cloud, model=model)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self.assertLess(elapsed_ms, 100.0,
                        f'ядро {elapsed_ms:.1f} мс на {len(cloud)} точек > бюджета 100 мс')

    def test_profiles_cover_all_records(self):
        for name in RECORDS:
            with self.subTest(record=name):
                model = model_for_name(name)
                self.assertTrue(model.has_polyline, f'{name}: нет полилинии оси')
                self.assertGreaterEqual(len(model.y_nodes), 20)
                self.assertAlmostEqual(model.gauge_m, 1.59, delta=0.02)

    def test_record_name_recognised(self):
        path = os.path.join(DATA, 'roundT_doubleT', 'roundT_doubleT_0.db3')
        self.assertEqual(record_name(path), 'roundT_doubleT')


if __name__ == '__main__':
    unittest.main()