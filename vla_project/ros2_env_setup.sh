#!/bin/bash
if [ ! -d "/home/ros2-dev/mnt/ws/.venv" ]; then
    echo "Virtual environment not found!"
    exit 1
fi
export PYTHONPATH=/home/ros2-dev/mnt/ws/.venv/lib/python3.10/site-packages:$PYTHONPATH

if [ -f "/home/ros2-dev/mnt/ws/vla_project/install/setup.bash" ]; then
    source /home/ros2-dev/mnt/ws/vla_project/install/setup.bash
fi

echo "Environment setup complete. PYTHONPATH and ROS2 environment configured."