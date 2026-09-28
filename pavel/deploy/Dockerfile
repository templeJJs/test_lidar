# Поставка: Ubuntu 22.04 + ROS 2 Humble, нода габаритного контроля (guard).
#
# Сборка (контекст — КОРЕНЬ репозитория, нужны pavel/ и ros2_ws/):
#   docker build -f pavel/deploy/Dockerfile -t guard_detector .
#
# Запуск (бэг монтируется с хоста):
#   docker run --rm -v /abs/path/for_hackathon/doubleT_obstacle:/data/doubleT_obstacle \
#       guard_detector ros2 launch guard_detector guard.launch.py \
#       topic:=/sensing/lidar/hesai128/pointcloud bag:=/data/doubleT_obstacle
#
# Подробности — pavel/deploy/README.md.

FROM ros:humble

# ROS-пакеты сообщений и rosbag2 в ros:humble (ros-core) НЕ входят — ставим явно.
# open3d нарочно НЕ ставится: тракт guard работает на numpy+scipy
# (см. pavel/guard/cloud.py).
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-numpy \
    python3-scipy \
    python3-colcon-common-extensions \
    ros-humble-rclpy \
    ros-humble-sensor-msgs \
    ros-humble-std-msgs \
    ros-humble-geometry-msgs \
    ros-humble-visualization-msgs \
    ros-humble-ros2bag \
    ros-humble-rosbag2-storage-default-plugins \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/lidar

# Тракт габарита (без visualize_bag.py/open3d: ноде нужен только guard/).
COPY pavel/guard /opt/lidar/pavel/guard
ENV PYTHONPATH=/opt/lidar/pavel

# ROS 2 пакет-обёртка (setup.bash — bash-сценарий, поэтому не /bin/sh).
SHELL ["/bin/bash", "-c"]
COPY ros2_ws/src/guard_detector /opt/lidar/ros2_ws/src/guard_detector
RUN source /opt/ros/humble/setup.bash \
    && cd /opt/lidar/ros2_ws \
    && colcon build --packages-select guard_detector

ENTRYPOINT ["/bin/bash", "-c", \
    "source /opt/ros/humble/setup.bash && source /opt/lidar/ros2_ws/install/setup.bash && exec \"$@\"", "--"]
CMD ["ros2", "launch", "guard_detector", "guard.launch.py"]
