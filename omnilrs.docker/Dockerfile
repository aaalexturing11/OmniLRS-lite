# Official Isaac Sim 6.2 image. It ships Warp 1.17, which resolves
# cuDeviceGetUuid on the R580 / CUDA 13 driver where Isaac Sim 5.0's Warp 1.7.1
# aborts and Kit then segfaults. Do not install another Warp build or pin
# NumPy: either one replaces the build Isaac Sim tested.
FROM nvcr.io/nvidia/isaac-sim:6.2.0

# The published image drops to uid 1234. apt needs root. This derived image
# stays root so the bind-mounted repository is writable without host ACLs.
USER root

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMNI_KIT_ACCEPT_EULA=YES \
    ACCEPT_EULA=Y \
    OMNILRS_ROOT=/workspace/omnilrs \
    OMNILRS_MEMORY_PROFILE=auto \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

RUN echo 'Etc/UTC' > /etc/timezone && \
    ln -sf /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    apt-get update && \
    apt-get install -q -y --no-install-recommends \
        tzdata \
        apt-utils \
        dirmngr \
        gnupg2 \
        curl \
        ca-certificates \
        libglib2.0-0 \
        tmux \
        vim \
        nano \
        build-essential \
        pkg-config \
        gdal-bin \
        libgdal-dev \
        g++ \
        wget \
    && rm -rf /var/lib/apt/lists/*

# System ROS is only the external CLI (ros2 topic list). Humble on 22.04,
# Jazzy on 24.04. It is not put on LD_LIBRARY_PATH and ROS_DISTRO is not
# exported: Isaac Sim 6.2 then keeps its bundled ROS libraries, which match
# its Python 3.12. Sourcing the apt ROS inside the simulator process is the
# Python 3.10 mismatch the 6.2 ROS guide tells you to avoid.
RUN . /etc/os-release && \
    if [ "$VERSION_ID" = "24.04" ]; then DISTRO=jazzy; else DISTRO=humble; fi && \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
        -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu ${UBUNTU_CODENAME} main" \
        > /etc/apt/sources.list.d/ros2.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        ros-${DISTRO}-ros-core \
        ros-${DISTRO}-vision-msgs \
        ros-${DISTRO}-tf2-msgs && \
    printf '%s\n' "$DISTRO" > /etc/omnilrs-ros-distro && \
    rm -rf /var/lib/apt/lists/*

# only-if-needed refuses to downgrade Isaac Sim's NumPy, Warp, or CUDA wheels.
# coverage>=7.6.1 supplies coverage.types.Tracer. Numba 0.62 subclasses it
# at import time and crashes on the older coverage shipped with Isaac Sim.
RUN /isaac-sim/python.sh -m pip install --upgrade-strategy only-if-needed \
        "coverage>=7.6.1" \
        omegaconf \
        hydra-core \
        skyfield \
        zfpy \
        numba \
        gdown \
        em \
        lark \
        pandas \
        av \
        yamcs-client \
        "eclipse-zenoh==1.10.1" \
        "asyncio_for_robotics==1.5.0" \
        msgspec \
    && /isaac-sim/python.sh -m pip install --upgrade-strategy only-if-needed --no-deps opencv-python

RUN printf '%s\n' \
        "alias isaacpy='/isaac-sim/python.sh'" \
        "alias isaacapp='/isaac-sim/runheadless.sh'" \
        "alias omnilrs-ros='unset LD_LIBRARY_PATH; source /opt/ros/$(cat /etc/omnilrs-ros-distro)/setup.bash'" \
        >> /root/.bashrc

COPY ./entrypoint.sh /
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
