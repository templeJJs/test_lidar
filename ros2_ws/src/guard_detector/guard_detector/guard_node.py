"""Узел габаритного контроля: PointCloud2 -> guard.run.analyze -> публикации.

Тонкий ROS-слой над трактом `guard` (лежит в `/opt/lidar/pavel`, путь в
PYTHONPATH образа). Узел:

* подписан на топик лидара (параметр `topic`, по умолчанию `/lidar_points`;
  в записи doubleT_obstacle — `/sensing/lidar/hesai128/pointcloud`);
* QoS подписки — RELIABLE, KEEP_LAST depth 1, VOLATILE. Это НЕ произвольно:
  в записях ТЗ издатель объявлен надёжным, и BEST_EFFORT-подписка даёт молча
  ноль данных (см. ros2_ws/README.md, главная ловушка датасета);
* на каждое облако: guard.cloud.points_from_msg -> analyze -> маркеры
  находок (`/guard/obstacles`), линии габаритного коридора
  (`/guard/corridor`), статус JSON (`/guard/status`);
* латентность analyze печатается в лог на каждом кадре и идёт в статус.

Система координат лидара: x вправо, y НАЗАД, z вверх (guard внутри ведёт
fwd=-y, lat=x, up=z); маркеры кладутся в frame_id сообщения с этой же
конвенцией: x = lat_abs находки, y = -дальность, z = пол + высота/2.
"""

import json
import time
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

from guard.cloud import points_from_msg
from guard.obstacles import OBST_WIN
from guard.run import analyze
from guard.walls import GAUGE_H_LO, GAUGE_H_HI

# Пара «надёжно, последний экземпляр»: ровно то, что объявлено издателями
# в bag-записях (reliability: 1), плюс depth=1 — нода realtime, старые кадры
# накапливать нельзя.
LIDAR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    durability=DurabilityPolicy.VOLATILE,
)

# Минимальный размер маркера — только чтобы находка была видна в RViz,
# на логику не влияет.
MARKER_MIN = 0.3


class GuardNode(Node):
    def __init__(self):
        super().__init__('guard_detector')
        self.declare_parameter('topic', '/lidar_points')
        self.declare_parameter('frames', 20)
        self.declare_parameter('publish_corridor', True)

        topic = str(self.get_parameter('topic').value)
        n = int(self.get_parameter('frames').value)
        self._recv = deque(maxlen=n)
        self._proc = deque(maxlen=n)
        self._latency = deque(maxlen=n)

        self.obstacles_pub = self.create_publisher(MarkerArray, '/guard/obstacles', 1)
        self.corridor_pub = self.create_publisher(MarkerArray, '/guard/corridor', 1)
        self.status_pub = self.create_publisher(String, '/guard/status', 1)
        self.create_subscription(PointCloud2, topic, self._on_cloud, LIDAR_QOS)
        self.get_logger().info(
            f'guard_detector: подписка на {topic}, QoS RELIABLE KEEP_LAST depth=1')

    # ------------------------------------------------------------------
    def _on_cloud(self, msg: PointCloud2):
        self._recv.append(time.monotonic())
        try:
            xyz, _intensity = points_from_msg(msg)
        except ValueError as exc:
            self.get_logger().error(str(exc))
            return

        t0 = time.monotonic()
        axis, wall = analyze(xyz.astype(np.float64))
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        self._proc.append(time.monotonic())
        self._latency.append(elapsed_ms)

        hits, summ = wall.get('obstacles', ([], {}))
        ing = [h for h in hits if h['in_gauge']]
        self.get_logger().info(
            f'кадр: {elapsed_ms:.0f} мс, точек {xyz.shape[0]}, '
            f'reach {wall["reach"]:.0f} м, находок {summ.get("hits", 0)}, '
            f'в габарите {len(ing)}'
            + (', ближняя %.0f м lat %+.2f' % (min(h["d"] for h in ing),
                                               min(ing, key=lambda h: h["d"])["lat"])
               if ing else ''))

        self._publish_obstacles(hits, wall, msg)
        if bool(self.get_parameter('publish_corridor').value):
            self._publish_corridor(wall, msg)
        self._publish_status(xyz, axis, wall, summ, msg, elapsed_ms)

    # ------------------------------------------------------------------
    def _publish_obstacles(self, hits, wall, msg):
        array = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        array.markers.append(clear)

        floor = float(wall['floor'])
        for i, h in enumerate(hits):
            cube = Marker()
            cube.header.frame_id = msg.header.frame_id or 'lidar'
            cube.header.stamp = msg.header.stamp
            cube.ns = 'guard_obstacles'
            cube.id = i * 2
            cube.type = Marker.CUBE
            cube.action = Marker.ADD
            cube.pose.position.x = float(h['lat_abs'])   # x вправо
            cube.pose.position.y = float(-h['d'])        # y назад: fwd = -y
            cube.pose.position.z = floor + 0.5 * float(h['height'])
            cube.pose.orientation.w = 1.0
            cube.scale.x = float(max(h['width'], MARKER_MIN))
            cube.scale.y = float(OBST_WIN)               # окно поиска по дальности
            cube.scale.z = float(max(h['height'], MARKER_MIN))
            if h['in_gauge'] and h['confirmed']:
                cube.color.r, cube.color.g, cube.color.b = 1.0, 0.10, 0.05
            elif h['in_gauge']:
                cube.color.r, cube.color.g, cube.color.b = 1.0, 0.55, 0.05
            else:
                cube.color.r, cube.color.g, cube.color.b = 0.55, 0.55, 0.60
            cube.color.a = 0.7
            array.markers.append(cube)

            text = Marker()
            text.header = cube.header
            text.ns = 'guard_obstacles'
            text.id = i * 2 + 1
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = float(h['lat_abs'])
            text.pose.position.y = float(-h['d'])
            text.pose.position.z = floor + float(h['height']) + 0.4
            text.pose.orientation.w = 1.0
            text.scale.z = 0.5
            text.color.r = text.color.g = text.color.b = 1.0
            text.color.a = 1.0
            text.text = (f"{h['d']:.0f} м, lat {h['lat']:+.2f}, "
                         f"{h['width']:.1f}x{h['height']:.1f} м, "
                         f"in_gauge={h['in_gauge']}, confirmed={h['confirmed']}")
            array.markers.append(text)

        self.obstacles_pub.publish(array)

    def _publish_corridor(self, wall, msg):
        """Линии габаритного коридора: x = axis ± half на двух высотах.

        Та же геометрия, что в guard.draw.corridor_lineset, только маркерами:
        сегмент зелёный, если оба его бина 'ok' (подтверждены измерением),
        жёлтый — если ведутся предсказанием ('hole' — это "не наблюдается",
        а не "свободно").
        """
        sel = np.array([s != 'stop' for s in wall['status']])
        if sel.sum() < 2:
            self.corridor_pub.publish(MarkerArray())
            return
        fwd = wall['fwd_mid'][sel]
        ax = wall['axis'][sel]
        status = wall['status'][sel]
        floor = float(wall['floor'])
        z_lo, z_hi = floor + GAUGE_H_LO, floor + GAUGE_H_HI

        pts, colors = [], []
        ok_c = (0.10, 0.75, 0.20, 1.0)
        hole_c = (0.95, 0.75, 0.05, 1.0)

        def seg(xa, ya, za, xb, yb, zb, c):
            pts.append((xa, ya, za))
            pts.append((xb, yb, zb))
            colors.extend((c, c))

        for side, half in ((-1.0, wall['half_left'][sel]),
                           (1.0, wall['half_right'][sel])):
            x = ax + side * half
            for k in range(len(fwd) - 1):
                c = ok_c if (status[k] == 'ok' and status[k + 1] == 'ok') else hole_c
                for zc in (z_lo, z_hi):
                    seg(x[k], -fwd[k], zc, x[k + 1], -fwd[k + 1], zc, c)
            for k in range(0, len(fwd), 2):
                c = ok_c if status[k] == 'ok' else hole_c
                seg(x[k], -fwd[k], z_lo, x[k], -fwd[k], z_hi, c)

        from geometry_msgs.msg import Point
        from std_msgs.msg import ColorRGBA
        marker = Marker()
        marker.header.frame_id = msg.header.frame_id or 'lidar'
        marker.header.stamp = msg.header.stamp
        marker.ns = 'guard_corridor'
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 0.08
        marker.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2]))
                         for p in pts]
        marker.colors = [ColorRGBA(r=c[0], g=c[1], b=c[2], a=c[3]) for c in colors]
        self.corridor_pub.publish(MarkerArray(markers=[marker]))

    # ------------------------------------------------------------------
    def _rate(self, stamps):
        if len(stamps) < 2:
            return 0.0
        span = stamps[-1] - stamps[0]
        return float((len(stamps) - 1) / span) if span > 1e-6 else 0.0

    def _publish_status(self, xyz, axis, wall, summ, msg, elapsed_ms):
        ok = wall['status'] == 'ok'
        hl = wall['half_left'][ok]
        hr = wall['half_right'][ok]
        payload = {
            'frame_id': msg.header.frame_id,
            'stamp': float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9,
            'points': int(xyz.shape[0]),
            'recv_fps': round(self._rate(self._recv), 2),
            'proc_fps': round(self._rate(self._proc), 2),
            'latency_ms': round(float(np.median(self._latency)), 1),
            'latency_last_ms': round(elapsed_ms, 1),
            'axis_measured': bool(axis['measured']),
            'reach_m': round(float(wall['reach']), 1),
            'reach_bridged_m': round(float(wall['reach_bridged']), 1),
            'corridor': {
                'half_left_med_m': round(float(np.median(hl)), 2) if len(hl) else None,
                'half_left_min_m': round(float(np.min(hl)), 2) if len(hl) else None,
                'half_right_med_m': round(float(np.median(hr)), 2) if len(hr) else None,
                'half_right_min_m': round(float(np.min(hr)), 2) if len(hr) else None,
            },
            'obstacles': {
                'hits': int(summ.get('hits', 0)),
                'in_gauge': int(summ.get('hits_in_gauge', 0)),
                'in_gauge_confirmed': int(summ.get('hits_in_gauge_confirmed', 0)),
                'detect_ms': round(float(summ.get('detect_ms', 0.0)), 1),
            },
        }
        self.status_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))


def main(args=None):
    rclpy.init(args=args)
    node = GuardNode()
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
