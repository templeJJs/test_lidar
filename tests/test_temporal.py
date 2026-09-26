"""Тесты временного учёта детектора (`detector/temporal.py`).

Синтетика -- как у `tests/test_detector_core.py`: прямая модель пути, плитки
точек, внесённый объект. Проверяются: перенос кадра (канонизация + ход),
интеграл профиля хода, накопление как усилитель слабого кандидата, трекер
M из N (подтверждение, гашение одиночной вспышки, коастинг, retro), оценка
собственной скорости трека и масштабирование счётных порогов по плотности.

Отдельный класс -- якорь на реальной записи (пропускается, если бага нет):
человек на 56 м в `doubleT_obstacle`, покадровый разрыв 32-36 обязан
закрыться накоплением + коастингом.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detector import DetectorConfig, TrackModel, detect  # noqa: E402
from detector.core import AxisPose  # noqa: E402
from detector.temporal import (  # noqa: E402
    TemporalDetector,
    density_scaled_cfg,
    motion_between_from_profile,
    warp_into_frame,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANCHOR_DB = os.path.join(ROOT, 'for_hackathon', 'doubleT_obstacle',
                         'doubleT_obstacle_0.db3')
ANCHOR_LABELS = os.path.join(ROOT, 'dataset', 'anchor_person')


def make_model(axis_i=0.0, ugr=-1.20, gauge=1.596, y_lo=-80.0, y_hi=-2.0,
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


def scene_floor(model):
    """Фон нормы: ложе/пол (препятствий быть не должно).

    Пол лежит НИЖЕ окна высоты головки рельса (h -0.14..0.10): иначе плотная
    равномерная плита синтетики даёт ложные «полосы рельсов», и канонизация
    сдвигает сцену на десятки сантиметров (замерено: на плите h -0.20..0.25
    axis_pose находил shift 0.30 м, и объект на 50 м уезжал за габарит).
    """
    return strip(model, -1.70, 1.70, -0.45, -0.20)


def sparse_object(model, y_m, seed):
    """Объект на 50 м, нарочно разреженный: 12 точек на кадр, каждый кадр --
    в своих кольцах (кадр `s` бьёт чуть выше/ниже, как реальный сканер).

    Один кадр форму не набирает (все точки в 2-3 ячейках высоты из 4 нужных
    формовому гейту, размах 0.14 м), а 4-5 кадров вместе дают 24-36 точек,
    опору по высоте и размах 0.6+ -- проверка накопителя как усилителя.
    12 точек/кадр -- ровно гейт поддержки `boost_min_current`: усиленный
    кандидат обязан быть виден (хоть слабо) и в текущем кадре.
    """
    h = np.linspace(0.40, 0.54, 12) + 0.12 * seed
    rng = np.random.default_rng(seed)
    u = -0.20 + 0.4 * rng.random(12)
    y = y_m - 0.3 + 0.1 * rng.random(12)     # середина бина 1 м, не его край
    return to_xyz(model, u, h, y)


class TestWarp(unittest.TestCase):
    """Перенос точек между кадрами: канонизация + продольный ход."""

    def setUp(self):
        self.model = make_model()

    def test_roundtrip(self):
        rng = np.random.default_rng(1)
        xyz = rng.normal(size=(100, 3)) * [1.0, 20.0, 1.0] + [0.0, -30.0, 0.0]
        pose_a = AxisPose(shift_m=0.05, slope=0.01, applied=True)
        pose_b = AxisPose(shift_m=-0.08, slope=-0.02, applied=True)
        moved = warp_into_frame(xyz, pose_a, pose_b, dy=1.5)
        back = warp_into_frame(moved, pose_b, pose_a, dy=-1.5)
        np.testing.assert_allclose(back, xyz, atol=1e-9)

    def test_pure_longitudinal_shift(self):
        xyz = to_xyz(self.model, 0.3, 1.0, -20.0)
        moved = warp_into_frame(xyz, AxisPose(), AxisPose(), dy=2.0)
        self.assertAlmostEqual(float(moved[0, 1]), -18.0, places=9)
        self.assertAlmostEqual(float(moved[0, 0]), float(xyz[0, 0]), places=9)

    def test_motion_between_integrates_profile(self):
        profile = {'profile': [0.5, None, 1.0], 'profile_step': 1,
                   'm_per_frame': 0.25}
        between = motion_between_from_profile(profile)
        self.assertAlmostEqual(between(0, 3), 0.5 + 0.25 + 1.0, places=9)
        self.assertAlmostEqual(between(3, 0), -(0.5 + 0.25 + 1.0), places=9)
        self.assertAlmostEqual(between(1, 1), 0.0, places=9)

    def test_motion_between_without_profile_is_zero(self):
        between = motion_between_from_profile(None)
        self.assertEqual(between(0, 5), 0.0)
        between = motion_between_from_profile({'profile': []})
        self.assertEqual(between(0, 5), 0.0)


class TestDensityScaling(unittest.TestCase):
    """Счётные пороги merged-прохода масштабируются на плотность облака."""

    def test_counts_scale_and_floor(self):
        cfg = DetectorConfig()
        out = density_scaled_cfg(cfg, 3.0)
        self.assertEqual(out.min_points, 60)
        self.assertEqual(out.trough_min_points, cfg.trough_min_points * 3)
        # Порог занятости ячейки стены НЕ масштабируется: бахрома стены в
        # объединении не густеет пропорционально (замер на doubleT_platform).
        self.assertEqual(out.wall_min_cell_points, cfg.wall_min_cell_points)
        same = density_scaled_cfg(cfg, 0.5)
        self.assertEqual(same.min_points, cfg.min_points,
                         'плотность ниже кадровой пороги не опускает')
        self.assertEqual(cfg.min_points, 20, 'исходный конфиг не тронут')


class TestTracker(unittest.TestCase):
    """Подтверждение серией M из N, коастинг, retro, скорость трека."""

    def setUp(self):
        self.model = make_model()
        self.cfg = DetectorConfig()
        self.floor = scene_floor(self.model)

    def _run(self, td, frames):
        """Прогнать кадры: {номер кадра: TemporalResult}."""
        out = {}
        for i, cloud in enumerate(frames):
            out[i] = td.update(cloud, frame_index=i)
        return out

    def test_confirms_on_second_frame_and_retro_first(self):
        obj = strip(self.model, -0.2, 0.3, 0.5, 1.5, y_lo=-25.5, y_hi=-25.0)
        frames = [np.vstack([self.floor, obj]) for _ in range(4)]
        td = TemporalDetector(self.model, self.cfg, window=0,
                              frame_passthrough=False)
        out = self._run(td, frames)
        self.assertEqual(out[0].obstacles, [], 'первый кадр ждёт подтверждения')
        self.assertTrue(out[1].obstacles, 'второй кадр -- тревога (2 из 3)')
        self.assertIn(0, out[1].retro, 'первый кадр засчитан задним числом')
        self.assertTrue(out[2].obstacles and out[3].obstacles)

    def test_single_flash_never_alarms(self):
        obj = strip(self.model, -0.2, 0.3, 0.5, 1.5, y_lo=-25.5, y_hi=-25.0)
        frames = [np.vstack([self.floor, obj])] + [self.floor] * 5
        td = TemporalDetector(self.model, self.cfg, window=0,
                              frame_passthrough=False)
        out = self._run(td, frames)
        for i in range(6):
            self.assertEqual(out[i].obstacles, [],
                             'одиночная вспышка (кадр 0) тревоги не даёт')

    def test_frame_passthrough_default_alarms_immediately(self):
        """По умолчанию покадровая детекция выдаётся сразу, без серии."""
        obj = strip(self.model, -0.2, 0.3, 0.5, 1.5, y_lo=-25.5, y_hi=-25.0)
        frames = [np.vstack([self.floor, obj])] + [self.floor] * 2
        td = TemporalDetector(self.model, self.cfg, window=0)
        out = self._run(td, frames)
        self.assertTrue(out[0].obstacles,
                        'покадровый detect не должен ждать подтверждения')
        self.assertEqual(out[1].obstacles, [],
                         'неподтверждённый трек не коастит')

    def test_coast_two_frames_then_silence(self):
        obj = strip(self.model, -0.2, 0.3, 0.5, 1.5, y_lo=-25.5, y_hi=-25.0)
        frames = [np.vstack([self.floor, obj])] * 4 + [self.floor] * 4
        td = TemporalDetector(self.model, self.cfg, window=0, coast=2)
        out = self._run(td, frames)
        self.assertTrue(out[4].obstacles and out[5].obstacles,
                        'коастинг: 2 кадра без детекции трек переживает')
        self.assertEqual(out[6].obstacles, [], 'третий пропуск -- молчание')
        self.assertEqual(out[7].obstacles, [])

    def test_track_velocity_of_moving_object(self):
        frames = []
        for i in range(6):
            obj = strip(self.model, -0.2, 0.3, 0.5, 1.5,
                        y_lo=-30.5 + 0.5 * i, y_hi=-30.0 + 0.5 * i)
            frames.append(np.vstack([self.floor, obj]))
        td = TemporalDetector(self.model, self.cfg, window=0)
        out = self._run(td, frames)
        self.assertTrue(out[5].obstacles)
        info = [t for t in out[5].tracks if t['confirmed']]
        self.assertTrue(info)
        self.assertGreater(info[0]['vy'], 0.3,
                           'идущий на 0.5 м/кадр объект: трек обязан '
                           'измерить его скорость')

    def test_sensor_motion_compensated_in_association(self):
        """Едущий сенсор: статичный объект едет по кадру, трек стоит в мире."""
        frames = []
        for i in range(6):
            obj = strip(self.model, -0.2, 0.3, 0.5, 1.5,
                        y_lo=-30.5 + 1.0 * i, y_hi=-30.0 + 1.0 * i)
            frames.append(np.vstack([self.floor, obj]))
        td = TemporalDetector(self.model, self.cfg, window=0,
                              motion_between=lambda a, b: 1.0 * (b - a))
        out = self._run(td, frames)
        self.assertTrue(out[5].obstacles)
        info = [t for t in out[5].tracks if t['confirmed']]
        self.assertTrue(info)
        self.assertAlmostEqual(info[0]['vy'], 0.0, delta=0.1,
                               msg='ход сенсора вычтен: собственная скорость '
                                   'статичного объекта -- ноль')

    def test_confidence_in_unit_interval(self):
        obj = strip(self.model, -0.2, 0.3, 0.5, 1.5, y_lo=-25.5, y_hi=-25.0)
        frames = [np.vstack([self.floor, obj]) for _ in range(4)]
        td = TemporalDetector(self.model, self.cfg, window=0)
        out = self._run(td, frames)
        for res in out.values():
            for ob in res.obstacles:
                self.assertGreaterEqual(ob.confidence, 0.0)
                self.assertLessEqual(ob.confidence, 1.0)


class TestAccumulator(unittest.TestCase):
    """Накопление как усилитель: слабый кадр + соседние = детекция."""

    def setUp(self):
        self.model = make_model()
        self.cfg = DetectorConfig()
        self.floor = scene_floor(self.model)

    def test_sparse_object_detected_only_with_history(self):
        frames = [np.vstack([self.floor, sparse_object(self.model, -50.0, s)])
                  for s in range(6)]
        lonely = detect(frames[3], model=self.model, cfg=self.cfg)
        self.assertEqual(lonely.obstacles, [],
                         '4 точки на 50 м один кадр видеть не обязан')
        td = TemporalDetector(self.model, self.cfg, window=4, keep=0.5)
        seen = [td.update(f, frame_index=i) for i, f in enumerate(frames)]
        hits = [r for r in seen
                if any(45.0 <= o.distance_m <= 55.0 for o in r.obstacles)]
        self.assertTrue(hits, 'накопление 4 кадров слабый объект находит')

    def test_empty_scene_stays_silent(self):
        frames = [self.floor] * 6
        td = TemporalDetector(self.model, self.cfg, window=4, keep=0.5)
        for i, f in enumerate(frames):
            res = td.update(f, frame_index=i)
            self.assertEqual(res.obstacles, [],
                             'кадр %d: на пустой сцене накопление молчит' % i)

    def test_first_frame_has_no_accumulation(self):
        td = TemporalDetector(self.model, self.cfg, window=4)
        res = td.update(self.floor, frame_index=0)
        self.assertIsNone(res.accum_result)

    def test_frame_gap_breaks_accumulation(self):
        """Несмежные номера кадров в объединение не берутся."""
        frames = [np.vstack([self.floor, sparse_object(self.model, -50.0, s)])
                  for s in range(8)]
        td = TemporalDetector(self.model, self.cfg, window=4, keep=0.5)
        seen = []
        for i, f in enumerate(frames):
            num = i if i < 4 else i + 100      # разрыв между 3 и 104
            seen.append(td.update(f, frame_index=num))
        hits = [r for j, r in enumerate(seen) if j >= 4
                and any(45.0 <= o.distance_m <= 55.0 for o in r.obstacles)]
        self.assertFalse(hits, 'после разрыва истории накопление начинается '
                               'заново и 4 кадров не хватает')


@unittest.skipUnless(os.path.exists(ANCHOR_DB)
                     and os.path.isdir(ANCHOR_LABELS),
                     'нет записи doubleT_obstacle или разметки якоря')
class TestAnchorRealBag(unittest.TestCase):
    """Якорь: человек на 56 м, серия 13..63; разрыв 32-36 закрыт трекером.

    Покадровый детектор видит человека на 46 из 51 кадра (пропуск 32-36);
    с накоплением и коастингом серия обязана быть целиком.
    """

    @classmethod
    def setUpClass(cls):
        import json

        import bag_reader
        import labels_io
        from detector.profiles import model_for_db
        from validation.dataset_eval import motion_fn_cached

        cls.model = model_for_db(ANCHOR_DB)
        # staticmethod: функция в атрибуте класса иначе биндится как метод и
        # получает лишний self при вызове из TemporalDetector.
        cls.motion_fn = staticmethod(motion_fn_cached(ANCHOR_DB))
        cls.frames = bag_reader.BagFrames(ANCHOR_DB, cache_size=8,
                                          prefetch_ahead=0)
        reader = labels_io.LabelsReader(ANCHOR_LABELS, 'for_hackathon')
        scene = reader.frame('doubleT_obstacle', 20,
                             include_unlabeled_frames=False)
        center = scene['objects'][0]['center']
        # u человека -- поперёк пути (минус ось кадра), как у Obstacle.u_m.
        import labeling
        pf = labeling.path_frame(np.asarray(scene['xyz']), cls.model)
        y_c = float(center[1])
        cls.person = (float(center[0]) - float(pf.axis_x(np.asarray([y_c]))[0]),
                      y_c)
        reader.close()

    @classmethod
    def tearDownClass(cls):
        cls.frames.close()

    def test_series_has_no_gap(self):
        td = TemporalDetector(self.model, motion_between=self.motion_fn)
        alarms, retro = set(), set()
        for f in range(13, 64):
            res = td.update(self.frames[f].xyz, frame_index=f)
            if any(abs(o.y_m - self.person[1]) <= 2.0
                   and abs(o.u_m - self.person[0]) <= 1.5
                   for o in res.obstacles):
                alarms.add(f)
            for rf, robs in res.retro.items():
                for o in robs:
                    if (abs(o.y_m - self.person[1]) <= 2.0
                            and abs(o.u_m - self.person[0]) <= 1.5):
                        retro.add(rf)
        covered = alarms | retro
        self.assertGreaterEqual(len(covered), 51,
                                'серия 13..63 обязана быть целиком: '
                                'покрыто %d/51 (пропуски: %s)'
                                % (len(covered),
                                   sorted(set(range(13, 64)) - covered)))


if __name__ == '__main__':
    unittest.main()
