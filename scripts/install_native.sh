#!/bin/bash
# Extra Python packages for an Isaac Sim 6.2 install that is already on disk.
# The simulator's own Warp, NumPy and CUDA wheels stay as Isaac shipped them.

set -euo pipefail

git submodule init
git submodule update

if [ -n "${ISAACSIM_ROOT:-}" ] && [ -x "${ISAACSIM_ROOT}/python.sh" ]; then
    ISAAC_PY="${ISAACSIM_ROOT}/python.sh"
elif [ -x /isaac-sim/python.sh ]; then
    ISAAC_PY=/isaac-sim/python.sh
else
    echo "Set ISAACSIM_ROOT to the Isaac Sim 6.2 directory that contains python.sh" >&2
    exit 1
fi

"${ISAAC_PY}" -m pip install --upgrade-strategy only-if-needed \
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
    msgspec
"${ISAAC_PY}" -m pip install --upgrade-strategy only-if-needed --no-deps opencv-python
