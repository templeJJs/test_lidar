"""Пакет ROS 2 `tunnel_detector`: узел детектора, узел модели пути и запуск.

Сам алгоритм лежит вне пакета, в `/opt/lidar/detector` (монтируется как есть),
поэтому ROS-слой тонкий: разобрать PointCloud2, вызвать `detector.detect`,
опубликовать маркеры и статус.
"""

import os
from glob import glob

from setuptools import setup

package_name = 'tunnel_detector'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Tunnel detector team',
    maintainer_email='hackathon@example.com',
    description='Детектор препятствий в тоннеле по 3D-лидару',
    license='MIT',
    entry_points={
        'console_scripts': [
            'detector_node = tunnel_detector.detector_node:main',
            'model_node = tunnel_detector.model_node:main',
        ],
    },
)