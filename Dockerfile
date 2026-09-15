FROM ros:humble

RUN apt-get update && apt-get install -y \
    ros-humble-rviz2 \
    ros-humble-rosbag2-storage-default-plugins \
    ros-humble-rmw-cyclonedds-cpp \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /data

ENTRYPOINT ["/bin/bash", "-c", "source /opt/ros/humble/setup.bash && exec \"$@\"", "--"]
CMD ["bash"]
