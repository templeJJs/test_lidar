"""Запуск: узел guard_detector и (опционально) ros2 bag play.

    ros2 launch guard_detector guard.launch.py
    ros2 launch guard_detector guard.launch.py \
        topic:=/sensing/lidar/hesai128/pointcloud bag:=/data/doubleT_obstacle

Топик лидара — параметр `topic` (по умолчанию /lidar_points). В записи
doubleT_obstacle облака идут в /sensing/lidar/hesai128/pointcloud — его надо
передать явно.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _setup(context, *args, **kwargs):
    topic = LaunchConfiguration('topic').perform(context)
    bag = LaunchConfiguration('bag').perform(context)
    start_offset = LaunchConfiguration('start_offset').perform(context)

    actions = [
        Node(
            package='guard_detector',
            executable='guard_node',
            name='guard_detector',
            parameters=[{'topic': topic}],
            output='screen',
        ),
    ]
    if bag:
        # --disable-keyboard-controls: из-под launch stdin закрыт, и без этого
        # флага плеер rosbag2 завершается сразу после старта.
        # --read-ahead-queue-size: дефолтные 1000 сообщений при облаках по
        # ~24 МБ — это десятки ГБ ОЗУ; 4 кадра достаточно для 10 Гц.
        cmd = ['ros2', 'bag', 'play', bag, '--disable-keyboard-controls',
               '--read-ahead-queue-size', '4']
        if start_offset:
            cmd += ['--start-offset', start_offset]
        actions.append(ExecuteProcess(cmd=cmd, output='screen'))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('topic', default_value='/lidar_points',
                              description='топик sensor_msgs/PointCloud2 лидара'),
        DeclareLaunchArgument('bag', default_value='',
                              description='каталог rosbag2 для воспроизведения'),
        DeclareLaunchArgument('start_offset', default_value='',
                              description='сдвиг старта воспроизведения, сек'),
        OpaqueFunction(function=_setup),
    ])
