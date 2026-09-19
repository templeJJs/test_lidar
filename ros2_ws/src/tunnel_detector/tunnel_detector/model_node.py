"""Узел модели пути: полилиния оси + УГР -> `/tunnel/model` (TRANSIENT_LOCAL).

Детектору нужны `u = x − x_оси(y)` и `h = z − z_УГР(y)`. Считать модель на каждом
кадре нельзя: `track_geometry.build_track` стоит 0.8..1.3 с. Поэтому модель
считается один раз и публикуется как JSON латч-топиком (TRANSIENT_LOCAL), так что
детектор получает её, даже если стартовал позже.

Три источника, по порядку:

1. параметр `record` -- сохранённая полилиния из `detector/track_models.json`
   (снимок `build_track` по записям `for_hackathon/`). Это рабочий режим демо:
   заранее известно, какая запись проигрывается;
2. живой `track_geometry.build_track` -- если он импортируется (нужен open3d).
   Даёт модель для записи, которой нет в таблице;
3. прямая ось -- последний резерв, с предупреждением в лог: на поворотах она
   уводит габарит и даёт ложные срабатывания (замерено: 13 бинов против 0).

Параметр `record` не определяется по облаку автоматически: надёжно отличить
запись по одному кадру нельзя, а угадывание дало бы молча неверную ось.
"""

import json

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from detector import TrackModel
from detector.profiles import MEASURED_STRAIGHT, model_for_name

MODEL_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


class ModelNode(Node):
    def __init__(self):
        super().__init__('tunnel_model')
        self.declare_parameter('record', '')
        self.declare_parameter('publish_period_s', 10.0)
        self.publisher = self.create_publisher(String, '/tunnel/model', MODEL_QOS)
        self.model = self._build_model()
        self.publish()
        self.create_timer(float(self.get_parameter('publish_period_s').value), self.publish)

    def _build_model(self) -> TrackModel:
        record = str(self.get_parameter('record').value).strip()
        if not record:
            self.get_logger().warn(
                'параметр record не задан: публикую прямую ось. Для демо укажите '
                'record:=<имя записи> (например roundT_doubleT)')
            return TrackModel(name='fallback-straight', gauge_m=1.593)
        model = model_for_name(record)
        if not model.has_polyline:
            straight = MEASURED_STRAIGHT.get(record)
            if straight is None:
                self.get_logger().warn(f'нет модели для {record!r}: публикую прямую ось')
            else:
                s, i = straight
                model = TrackModel(name=f'{record} [straight]', gauge_m=1.593,
                                   axis_straight_s=s, axis_straight_i=i)
        self.get_logger().info(
            f'модель пути {model.name}: узлов {len(model.y_nodes)}, '
            f'колея {model.gauge_m:.3f} м')
        return model

    def publish(self):
        self.publisher.publish(String(data=json.dumps(self.model.to_dict(),
                                                      ensure_ascii=False)))


def main(args=None):
    rclpy.init(args=args)
    node = ModelNode()
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