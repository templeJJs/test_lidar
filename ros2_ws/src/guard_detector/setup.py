"""Пакет ROS 2 `guard_detector`: узел габаритного контроля и запуск.

Сам тракт лежит вне пакета, в `/opt/lidar/pavel/guard` (монтируется как есть,
путь в PYTHONPATH образа), поэтому ROS-слой тонкий: разобрать PointCloud2,
вызвать `guard.run.analyze`, опубликовать находки, коридор и статус.
"""

import os
from glob import glob

from setuptools import setup

package_name = 'guard_detector'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Tunnel guard team',
    maintainer_email='hackathon@example.com',
    description='Габаритный контроль тоннеля по 3D-лидару (тракт guard)',
    license='MIT',
    entry_points={
        'console_scripts': [
            'guard_node = guard_detector.guard_node:main',
        ],
    },
)
