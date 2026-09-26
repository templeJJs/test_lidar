"""Тесты сканирования: облако похоже на настоящее по геометрии и статистике."""
import os
import sys
import unittest

import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim import objects, scanner, scene  # noqa: E402
from sim.sensor import SensorModel  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, 'sim', 'sensor_hesai128.json')
COLUMNS = 200          # редкие колонки: полные 2709 -- дело CLI, а не юнит-теста


def make(seed=3, obstacles=(), length_m=60.0, section='wide'):
    """Сцена-труба плюс сенсор из настоящей записи, но с редкими колонками.

    Углы места берём из артефакта, а не из умолчаний `SensorModel`: там таблица
    пустая, и проверять диапазон колец было бы не на чем.
    """
    p = scene.SceneParams(seed=seed, kind='straight', section=section, length_m=length_m)
    s = scene.build_scene(p)
    s.objects = objects.place_objects(p, s.track, list(obstacles))
    scanner.add_objects(s)
    sensor = SensorModel.load(MODEL_PATH)
    sensor.azimuth_columns = COLUMNS
    sensor.range_sigma_m = 0.01
    # тестам нужен единый малый шум на всех поверхностях: поверхностные sigma
    # артефакта (0.02/0.09) иначе перекрывают `range_sigma_m`
    sensor.range_sigma_surface_m = None
    sensor.range_sigma_ground_m = None
    return p, s, sensor


def blocker_in_front(y_at: float = -20.15) -> o3d.geometry.TriangleMesh:
    """Плита на всё сечение туннеля: перекрывает путь целиком, щелей нет.

    Шире стен (x -4..8 при стенах -2.62/6.62) и глубже подошвы (z -3.0..4.0 при
    полу -1.8..-3.2 на длине трубы) -- иначе луч прошёл бы мимо преграды, и тест
    на затенение ничего не проверял.
    """
    box = o3d.geometry.TriangleMesh.create_box(12.0, 0.2, 7.0)
    box.translate((-4.0, y_at - 0.1, -3.0))
    return box


class TestScanner(unittest.TestCase):
    def test_point_count_is_rings_times_columns_minus_dropouts(self):
        p, s, sensor = make()
        f = scanner.scan(s, sensor, frame_idx=0, rng=np.random.default_rng(0))
        self.assertEqual(f.ring.shape[0], f.xyz.shape[0])
        self.assertEqual(f.intensity.shape[0], f.xyz.shape[0])
        self.assertLessEqual(f.xyz.shape[0], 128 * COLUMNS)
        self.assertGreater(f.xyz.shape[0], 128 * COLUMNS * 0.90)

    def test_rings_span_the_measured_elevation_range(self):
        p, s, sensor = make()
        f = scanner.scan(s, sensor, frame_idx=0, rng=np.random.default_rng(0))
        r = np.hypot(f.xyz[:, 0], f.xyz[:, 1])
        elev = np.degrees(np.arctan2(f.xyz[:, 2], r))
        self.assertAlmostEqual(float(elev.max()), 14.40, delta=0.5)
        self.assertAlmostEqual(float(elev.min()), -25.12, delta=0.5)

    def test_an_object_in_front_changes_the_cloud(self):
        p, s, sensor = make()
        empty = scanner.scan(s, sensor, 0, np.random.default_rng(1))
        p2, s2, sensor2 = make(obstacles=[objects.Placement('person', -20.0, 0.0, 'in_gauge')])
        with_obj = scanner.scan(s2, sensor2, 0, np.random.default_rng(1))
        near = with_obj.xyz[np.hypot(with_obj.xyz[:, 0], with_obj.xyz[:, 1]) < 22.0]
        self.assertGreater(near.shape[0], 100, 'объект в 20 м должен дать возвраты')
        self.assertNotEqual(empty.xyz.shape[0], with_obj.xyz.shape[0])

    def test_a_blocker_in_front_hides_everything_behind_it(self):
        """Затенение честное, и «вперёд» -- минус Y: за преградой возвратов нет.

        Без преграды облако тянется на всю длину трубы (60 м), с преградой на
        20 м за ней не должно быть ничего. Оборот ставим полный: проверяем
        геометрию тени, а не поле зрения прибора (оно -- отдельный тест).
        """
        p, s, sensor = make()
        sensor.azimuth_span_deg = 360.0
        empty = scanner.scan(s, sensor, 0, np.random.default_rng(7))
        self.assertLess(float(empty.xyz[:, 1].min()), -30.0, 'в пустом туннеле видно далеко')

        # в open3d 0.19 тензорные меши не складываются, поэтому преграда
        # склеивается с мешом сцены в legacy-виде и переводится в тензорный целиком
        legacy = o3d.t.geometry.TriangleMesh.to_legacy(s.mesh) + blocker_in_front()
        s.mesh = scene.to_tensor_mesh(legacy)
        blocked = scanner.scan(s, sensor, 0, np.random.default_rng(7))
        self.assertEqual(int(np.count_nonzero(blocked.xyz[:, 1] < -20.2)), 0,
                         'за преградой возвратов быть не должно')
        self.assertGreater(int(np.count_nonzero(blocked.xyz[:, 1] > 20.0)), 100,
                           'за сенсором туннель продолжается и виден')

    def test_the_tail_sector_is_not_scanned(self):
        """Поле зрения прибора: у широкой записи 247°, хвостовой сектор пуст.

        Точки собственного вагона при съёмке отфильтрованы, поэтому синтетика не
        должна светить назад: азимут возвратов обязан уложиться в поле зрения
        модели, снятое с записи.
        """
        p, s, sensor = make()
        self.assertAlmostEqual(sensor.azimuth_span_deg, 247.0, delta=1.0,
                               msg='поле зрения берётся из артефакта сенсора')
        f = scanner.scan(s, sensor, 0, np.random.default_rng(3))
        az = np.degrees(np.arctan2(f.xyz[:, 0], -f.xyz[:, 1]))
        half = sensor.azimuth_span_deg / 2.0 + 1.0
        self.assertLessEqual(float(np.abs(az).max()), half,
                             'возвраты не должны выходить за поле зрения')
        self.assertGreater(float(np.abs(az).max()), half - 5.0,
                           'до края поля зрения лучи всё-таки доходят')

    def test_dropout_removes_about_half_of_the_returns(self):
        p, s, sensor = make()
        sensor.dropout = 0.5
        f = scanner.scan(s, sensor, 0, np.random.default_rng(5))
        self.assertGreater(f.xyz.shape[0], 128 * COLUMNS * 0.40)
        self.assertLess(f.xyz.shape[0], 128 * COLUMNS * 0.60)

    def test_hits_beyond_max_range_are_dropped(self):
        p, s, sensor = make()
        f = scanner.scan(s, sensor, 0, np.random.default_rng(6), max_range_m=15.0)
        self.assertGreater(f.xyz.shape[0], 1000)
        self.assertLessEqual(float(np.linalg.norm(f.xyz, axis=1).max()), 15.05)

    def test_timestamp_grows_with_frame_index(self):
        p, s, sensor = make()
        f0 = scanner.scan(s, sensor, 0, np.random.default_rng(8))
        f3 = scanner.scan(s, sensor, 3, np.random.default_rng(8))
        self.assertGreater(f3.timestamp, f0.timestamp)

    def test_object_meshes_are_the_legacy_ones_the_scene_traces(self):
        p, s, sensor = make(obstacles=[objects.Placement('person', -20.0, 0.0, 'in_gauge')])
        parts = scanner.object_meshes(s.objects[0])
        self.assertEqual(len(parts), 1)
        self.assertIsInstance(parts[0], o3d.geometry.TriangleMesh)

    def test_intensity_is_in_byte_range_and_has_high_tail(self):
        p, s, sensor = make()
        f = scanner.scan(s, sensor, 0, np.random.default_rng(2))
        self.assertGreaterEqual(int(f.intensity.min()), 0)
        self.assertLessEqual(int(f.intensity.max()), 255)
        self.assertGreater(float(np.percentile(f.intensity, 90)), 5.0)

    def test_intensity_statistics_are_in_the_measured_range(self):
        """Медиана, p90 и доля >25 -- в тех же рамках, что у настоящих записей.

        Рамки -- объединение по шести записям: медиана 7..12, p90 14..26, доля >25
        от 0.66 % (`roundT_squareT_pressureGate_squareT`) до 10.8 %
        (`doubleT_obstacle`). Широкая двухпутная сцена -- ближайший к
        `doubleT_obstacle` случай по форме распределения (самый большой p90 и
        самая большая доля ярких точек), и рамки теста -- те же, что у задачи
        целиком: медиана 6..11, p90 20..30, доля >25 в 0.7..10.8 %.
        """
        p, s, sensor = make()
        f = scanner.scan(s, sensor, 0, np.random.default_rng(4))
        median = float(np.median(f.intensity))
        p90 = float(np.percentile(f.intensity, 90))
        share_gt25 = float(np.mean(f.intensity > 25))
        self.assertGreaterEqual(median, 6.0)
        self.assertLessEqual(median, 11.0)
        self.assertGreaterEqual(p90, 20.0)
        self.assertLessEqual(p90, 30.0)
        self.assertGreaterEqual(share_gt25, 0.007)
        self.assertLessEqual(share_gt25, 0.108)

    def test_intensity_falls_with_range_and_is_brightest_on_the_walls(self):
        """Модель действительно от поверхности и дальности, а не от константы."""
        p, s, sensor = make()
        f = scanner.scan(s, sensor, 0, np.random.default_rng(9))
        r = np.linalg.norm(f.xyz, axis=1)
        near = f.intensity[(r > 2.0) & (r < 6.0)]
        far = f.intensity[(r > 40.0) & (r < 80.0)]
        self.assertGreater(float(np.median(near)), float(np.median(far)))

        params = s.params
        floor_z = scene.floor_z_xy(params, f.xyz[:, 0], f.xyz[:, 1])
        ballast = f.intensity[f.xyz[:, 2] <= floor_z + 0.05]
        walls = f.intensity[np.abs(f.xyz[:, 0] - params.walls_x[1]) < 0.2]
        self.assertGreater(float(np.median(walls)), float(np.median(ballast)),
                           'облицовка стены отражает ярче балласта')


if __name__ == '__main__':
    unittest.main()