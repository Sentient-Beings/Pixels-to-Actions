FROM ros2_base:humble

USER root

RUN apt-get update -y && \
    apt-get install -qq -y --no-install-recommends \
    unzip \
    wget -y && \
    rm -rf /var/lib/apt/lists/*

USER root
RUN mkdir -p /opt/mujoco && chown ros2-dev:ros2-dev /opt/mujoco

USER ros2-dev
WORKDIR /opt/mujoco
RUN wget https://github.com/deepmind/mujoco/releases/download/3.2.5/mujoco-3.2.5-linux-x86_64.tar.gz && \
    tar -xvzf mujoco-3.2.5-linux-x86_64.tar.gz && \
    rm mujoco-3.2.5-linux-x86_64.tar.gz

ENV LD_LIBRARY_PATH=/opt/mujoco/mujoco-3.2.5/bin:$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/mujoco/mujoco-3.2.5/lib

WORKDIR /home/ros2-dev/mnt/ws

RUN pip install mujoco
# TODO: Add the following into post create container scritps
# Add rosdep installation
# RUN rosdep update

# RUN apt-get update && \
#     rosdep install -r --from-paths . --ignore-src --rosdistro humble -y

RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc && \
    echo "source /usr/share/gazebo/setup.sh" >> ~/.bashrc
