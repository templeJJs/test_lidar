"""Узел детектора: PointCloud2 -> препятствия + телеметрия.

Тонкий слой над `detector.core.detect` (модуль лежит в `/opt/lidar`, путь
монтируется как есть). Узел:

* подписан на ОБА топика из записей ТЗ -- `/lidar_points` (frame `hesai_lidar`)
  и `/sensing/lidar/hesai128/pointcloud` (frame `lidar_livox`);
* QoS подписки -- RELIABLE, depth 1, VOLATILE. Это НЕ произвольно: в записях
  издатель объявлен надёжным, и `SensorDataQoS()` (BEST_EFFORT) даёт молча ноль
  данных -- самая дорогая ошибка в этом датасете;
* горячий путь без копий: `np.frombuffer` -> view (N, point_step) -> колонки
  x, y, z по смещениям из `msg.fields`;
* публикует `/tunnel/obstacles` (MarkerArray: CUBE + TEXT_VIEW_FACING с
  дистанцией) и `/tunnel/status` (JSON одной строкой).

Модель пути приходит из `model_node` по `/tunnel/model` (TRANSIENT_LOCAL), чтобы
детектор не пересчитывал её на каждом сообщении.
"""

import json
import time
from collections import deque

import numpy as np
import rclpy
from geometry_msgs.msg import Point
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

from detector import DetectorConfig, TrackModel, detect

DEFAULT_TOPICS = (
    '/lidar_points',
    '/sensing/lidar/hesai128/pointcloud',
)

# Пара «последний экземпляр, надёжно»: ровно то, что объявлено в bag-файлах.
BAG_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    durability=DurabilityPolicy.VOLATILE,
)

MODEL_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


def xyz_from_msg(msg: PointCloud2) -> np.ndarray:
    """(N, 3) float32 точки из PointCloud2 без промежуточных копий.

    Колонки берутся по смещениям из `msg.fields`, поэтому разбор не зависит от
    порядка полей и работает и на 26-байтовой точке Hesai, и на других раскладках.
    """
    if msg.width == 0 or msg.height == 0 or msg.point_step == 0:
        return np.zeros((0, 3), dtype=np.float32)
    offsets = {f.name: f.offset for f in msg.fields}
    missing = [name for name in ('x', 'y', 'z') if name not in offsets]
    if missing:
        raise ValueError(f'PointCloud2 has no fields {missing}')

    raw = np.frombuffer(msg.data, dtype=np.uint8)
    count = min(int(msg.width) * int(msg.height), raw.size // int(msg.point_step))
    rows = raw[:count * int(msg.point_step)].reshape(count, int(msg.point_step))
    step = int(msg.point_step)
    quad = np.empty((count, 3), dtype=np.float32)
    for i, name in enumerate(('x', 'y', 'z')):
        off = int(offsets[name])
        quad[:, i] = rows[:, off:off + 4].copy().view(np.float32).reshape(count)
    finite = np.isfinite(quad).all(axis=1) & (quad != 0).any(axis=1)
    return np.ascontiguousarray(quad[finite])


class DetectorNode(Node):
    def __init__(self):
        super().__init__('tunnel_detector')
        self.declare_parameter('topics', list(DEFAULT_TOPICS))
        self.declare_parameter('frames', 20)
        self.declare_parameter('max_range_m', DetectorConfig().max_range_m)
        self.declare_parameter('status_period_s', 1.0)

        topics = list(self.get_parameter('topics').value)
        self.cfg = DetectorConfig(max_range_m=float(self.get_parameter('max_range_m').value))
        self.model = None
        self._recv = deque(maxlen=int(self.get_parameter('frames').value))
        self._proc = deque(maxlen=int(self.get_parameter('frames').value))
        self._latency = deque(maxlen=int(self.get_parameter('frames').value))
        self._last_status = time.monotonic()
        self._detections = 0

        self.marker_pub = self.create_publisher(MarkerArray, '/tunnel/obstacles', 1)
        self.points_pub = self.create_publisher(PointCloud2, '/tunnel/obstacle_points', 1)
        self.status_pub = self.create_publisher(String, '/tunnel/status', 1)
        self.create_subscription(String, '/tunnel/model', self._on_model, MODEL_QOS)
        self._subs = [
            self.create_subscription(PointCloud2, topic, self._on_cloud, BAG_QOS)
            for topic in topics
        ]
        self.get_logger().info(
            f'tunnel_detector: подписка на {topics}, QoS RELIABLE depth=1 VOLATILE')

    # ------------------------------------------------------------------
    def _on_model(self, msg):
        try:
            self.model = TrackModel.from_dict(json.loads(msg.data))
            self.get_logger().info(f'модель пути получена: {self.model.name}')
        except (ValueError, TypeError) as exc:
            self.get_logger().warn(f'модель пути не разобрана: {exc}')

    def _on_cloud(self, msg: PointCloud2):
        now = time.monotonic()
        self._recv.append(now)
        try:
            xyz = xyz_from_msg(msg)
        except ValueError as exc:
            self.get_logger().error(str(exc))
            return

        model = self.model or TrackModel(name='fallback-straight')
        if self.model is None:
            self.get_logger().warn(
                'модель пути ещё не пришла: габарит считается от прямой оси, '
                'на поворотах это даёт ложные срабатывания',
                once=True)

        started = time.monotonic()
        result = detect(xyz, model=model, cfg=self.cfg)
        elapsed = time.monotonic() - started

        self._proc.append(time.monotonic())
        self._latency.append(elapsed)
        self._detections += len(result.obstacles)

        self._publish_markers(result, msg)
        self._publish_obstacle_points(xyz, result, msg)
        self._publish_status(result, msg, elapsed)

    def _publish_obstacle_points(self, xyz, result, msg):
        """Красное облако: точки кадра, попавшие в найденные препятствия.

        Точки восстанавливаются по геометрии препятствия (полоса `u` и диапазон
        `y`), а не хранятся в ядре: так `detect()` остаётся чистым и не таскает
        индексы точек ради визуализации.
        """
        cloud = PointCloud2()
        cloud.header.frame_id = msg.header.frame_id or 'map'
        cloud.header.stamp = msg.header.stamp
        if result.obstacles and xyz.size:
            u = xyz[:, 0] - self.model.axis_x(xyz[:, 1]) if self.model is not None \
                else xyz[:, 0]
            y = xyz[:, 1]
            mask = np.zeros(xyz.shape[0], dtype=bool)
            for obstacle in result.obstacles:
                mask |= ((y >= obstacle.y_min_m - 0.1) & (y <= obstacle.y_max_m + 0.1)
                         & (u >= obstacle.u_lo_m - 0.05) & (u <= obstacle.u_hi_m + 0.05))
            selected = np.ascontiguousarray(xyz[mask].astype(np.float32))
        else:
            selected = np.zeros((0, 3), dtype=np.float32)

        cloud.height = 1
        cloud.width = int(selected.shape[0])
        cloud.is_bigendian = False
        cloud.is_dense = True
        cloud.point_step = 12
        cloud.row_step = 12 * cloud.width
        cloud.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        cloud.data = selected.tobytes()
        self.points_pub.publish(cloud)

    # ------------------------------------------------------------------
    def _publish_markers(self, result, msg):
        array = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        array.markers.append(clear)

        for i, obstacle in enumerate(result.obstacles):
            cube = Marker()
            cube.header.frame_id = msg.header.frame_id or 'map'
            cube.header.stamp = msg.header.stamp
            cube.ns = 'tunnel_obstacles'
            cube.id = i * 2
            cube.type = Marker.CUBE
            cube.action = Marker.ADD
            cube.pose.position.x = float(obstacle.x_m)
            cube.pose.position.y = float(obstacle.y_m)
            cube.pose.position.z = float(0.5 * (obstacle.h_min_m + obstacle.h_max_m)
                                        + self._ugr_at(obstacle.y_m))
            cube.pose.orientation.w = 1.0
            cube.scale.x = float(max(obstacle.x_spread_m, 0.3))
            cube.scale.y = float(max(obstacle.y_max_m - obstacle.y_min_m, 0.3))
            cube.scale.z = float(max(obstacle.span_m, 0.3))
            cube.color.r = 1.0
            cube.color.g = 0.15
            cube.color.b = 0.10
            cube.color.a = 0.55
            array.markers.append(cube)

            text = Marker()
            text.header.frame_id = msg.header.frame_id or 'map'
            text.header.stamp = msg.header.stamp
            text.ns = 'tunnel_obstacles'
            text.id = i * 2 + 1
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position = Point(x=float(obstacle.x_m), y=float(obstacle.y_m),
                                       z=float(obstacle.h_max_m + self._ugr_at(obstacle.y_m) + 0.4))
            text.pose.orientation.w = 1.0
            text.scale.z = 0.6
            text.color.r = text.color.g = text.color.b = 1.0
            text.color.a = 1.0
            text.text = f'{obstacle.distance_m:.1f} м'
            array.markers.append(text)

        self.marker_pub.publish(array)

    def _ugr_at(self, y_m: float) -> float:
        if self.model is None or not self.model.has_polyline:
            return -1.15
        return float(self.model.ugr_z(np.asarray([y_m]))[0])

    def _rate(self, stamps):
        if len(stamps) < 2:
            return 0.0
        span = stamps[-1] - stamps[0]
        return float((len(stamps) - 1) / span) if span > 1e-6 else 0.0

    def _publish_status(self, result, msg, elapsed):
        now = time.monotonic()
        if now - self._last_status < float(self.get_parameter('status_period_s').value):
            return
        self._last_status = now
        payload = {
            'recv_fps': round(self._rate(self._recv), 2),
            'proc_fps': round(self._rate(self._proc), 2),
            'latency_ms': round(float(np.median(self._latency)) * 1000.0, 2) if self._latency else 0.0,
            'points': int(result.points_total),
            'cells': int(result.cells_total),
            'bins_blocked': int(result.bins_blocked),
            'bins_total': int(result.bins_total),
            'max_data_range_m': round(float(result.max_data_range_m), 2),
            'axis_range_m': [float(v) for v in result.axis_range_m],
            'blocked': bool(result.blocked_any),
            'detections': self._detections,
            'frame_id': msg.header.frame_id,
            'obstacles': [o.to_dict() for o in result.obstacles],
        }
        self.status_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))
        self._detections = 0


def main(args=None):
    rclpy.init(args=args)
    node = DetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()