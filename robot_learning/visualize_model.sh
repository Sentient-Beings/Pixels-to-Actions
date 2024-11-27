#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_NAME=$(basename "$SCRIPT_DIR")

cd /opt/mujoco/mujoco-3.2.5/bin/

# below change the xml file to the one you want to visualize 
./simulate "../../../../home/ros2-dev/mnt/ws/$PROJECT_NAME/model/ur5e.xml"