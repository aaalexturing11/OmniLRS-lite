#!/bin/bash
set -e

# Isaac Sim 6.2 selects the ROS libraries shipped inside the image when
# ROS_DISTRO is unset. Those libraries match Python 3.12 and the Warp build
# that works on current NVIDIA drivers. An apt-installed ROS on
# LD_LIBRARY_PATH is a different Python and is not loaded here.
unset ROS_DISTRO
cd /workspace/omnilrs
exec "$@"
