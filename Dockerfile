# FROM osrf/ros:humble-desktop

# ENV DEBIAN_FRONTEND=noninteractive

FROM ros:humble

ENV ROS_DISTRO=humble
ENV DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-c"]

# System + ROS build tools
RUN apt-get update && apt-get install -y \
    git \
    python3-pip \
    python3-colcon-common-extensions \
    python3-numpy \
    ros-humble-rclpy \
    ros-humble-rosidl-runtime-py \
    ros-humble-rosbag2-py \
    ros-humble-rosbag2-storage-mcap \
    ros-humble-builtin-interfaces \    
    ros-humble-std-msgs \    
    ros-humble-geometry-msgs \
    ros-humble-sensor-msgs \
    && rm -rf /var/lib/apt/lists/*

# Wxtra Python deps used elsewhere in repo
RUN pip3 install --no-cache-dir \
    scipy \    
    matplotlib \    
    tqdm

# Workspace
WORKDIR /ws
RUN mkdir -p /ws/src

# 1) Copy ESKF repo from local (instead of git clone)
COPY . /ws/src/ESKF_rov_tracking

# 2) blueboat_interfaces (ROS package)
# Replace branch if needed.
RUN cd /ws/src && git clone https://turlab.itk.ntnu.no/turlab/blueboat_interfaces.git
RUN cd /ws/src/blueboat_interfaces && git switch microAmp/fgo_rov_tracking

# # 2) ESKF repo
# RUN cd /ws/src && git clone https://github.com/ahaanesen/ESKF_rov_tracking.git

# Build interfaces first (and anything else in ws)
RUN source /opt/ros/humble/setup.bash && \
    cd /ws && colcon build --symlink-install


# Auto-source ROS + workspace on shell start
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc && \
    echo "source /ws/install/setup.bash" >> /root/.bashrc

WORKDIR /ws/src/ESKF_rov_tracking
CMD ["/bin/bash"]