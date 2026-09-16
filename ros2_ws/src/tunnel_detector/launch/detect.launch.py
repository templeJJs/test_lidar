"""Запуск: узел модели пути, детектор, статические TF и (опционально) ros2 bag play.

    ros2 launch tunnel_detector detect.launch.py record:=roundT_doubleT
    ros2 launch tunnel_detector detect.launch.py record:=roundT_doubleT \
        bag:=/data/roundT_doubleT rviz:=true

`record` обязателен для качества: без него узел модели публикует прямую ось, и на
поворотах появляются ложные срабатывания (замерено: 13 бинов против 0 на
локальной оси).

Статические TF ставятся из обоих frame_id, встречающихся в записях:
`hesai_lidar` (пять записей, топик `/lidar_points`) и `lidar_livox`
(`doubleT_obstacle`, топик `/sensing/lidar/hesai128/pointcloud`). Точное взаимное
положение датчиков неизвестно -- обе оси совмещаются с `map` поворотом на 180°
вокруг Z (лидар смотрит в −Y, RViz удобнее с осью X вперёд); при появлении
калибровки числа надо заменить.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _static_tf(frame_id):
    return Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name=f'static_tf_{frame_id}',
        arguments=['--x', '0', '--y', '0', '--z', '0',
                   '--roll', '0', '--pitch', '0', '--yaw', '3.14159265',
                   '--frame-id', 'map', '--child-frame-id', frame_id],
        output='log',
    )


def _setup(context, *args, **kwargs):
    record = LaunchConfiguration('record').perform(context)
    bag = LaunchConfiguration('bag').perform(context)
    use_rviz = LaunchConfiguration('rviz').perform(context).lower() in ('1', 'true', 'yes')
    topics = LaunchConfiguration('topics').perform(context)
    share = get_package_share_directory('tunnel_detector')

    actions = [
        _static_tf('hesai_lidar'),
        _static_tf('lidar_livox'),
        Node(
            package='tunnel_detector',
            executable='model_node',
            name='tunnel_model',
            parameters=[{'record': record}],
            output='screen',
        ),
        Node(
            package='tunnel_detector',
            executable='detector_node',
            name='tunnel_detector',
            parameters=[{'topics': [t for t in topics.split(',') if t]}],
            output='screen',
        ),
    ]
    if bag:
        actions.append(ExecuteProcess(
            cmd=['ros2', 'bag', 'play', bag, '--clock'],
            output='screen',
        ))
    if use_rviz:
        actions.append(Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', os.path.join(share, 'rviz', 'tunnel.rviz')],
            output='log',
        ))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('record', default_value='',
                              description='имя записи из detector/track_models.json'),
        DeclareLaunchArgument('bag', default_value='',
                              description='каталог rosbag2 для воспроизведения'),
        DeclareLaunchArgument('rviz', default_value='false'),
        DeclareLaunchArgument(
            'topics',
            default_value='/lidar_points,/sensing/lidar/hesai128/pointcloud',
            description='топики PointCloud2 через запятую'),
        OpaqueFunction(function=_setup),
    ])