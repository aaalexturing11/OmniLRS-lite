#!/bin/bash
# Build first: bash omnilrs.docker/build_docker.sh
# Default command is headless Lunalab on the Isaac Sim 6.2 image.
# A shell is: bash omnilrs.docker/run_docker.sh bash
set -euo pipefail

ROOT="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
CACHE="${HOME}/docker/isaac-sim"
mkdir -p \
    "${CACHE}/cache/main" \
    "${CACHE}/cache/computecache" \
    "${CACHE}/cache/kit" \
    "${CACHE}/config" \
    "${CACHE}/data" \
    "${CACHE}/logs" \
    "${CACHE}/pkg" \
    /tmp/images_streaming \
    /tmp/images_oncommand \
    /tmp/images_apxs \
    /tmp/images_depth \
    /tmp/images_monitoring \
    /tmp/images_lander

if [ "$#" -eq 0 ]; then
    set -- /isaac-sim/python.sh run.py environment=lunalab mode=ROS2 rendering.renderer.headless=true
fi

exec docker run --name isaac-sim-omnilrs-container --gpus all --rm \
    -e ACCEPT_EULA=Y \
    -e PRIVACY_CONSENT=Y \
    -e OMNI_KIT_ACCEPT_EULA=YES \
    -e OMNILRS_MEMORY_PROFILE=auto \
    -e OMNILRS_ROOT=/workspace/omnilrs \
    --network=host \
    --ipc=host \
    -v "${ROOT}:/workspace/omnilrs" \
    -v "${CACHE}/cache/kit:/isaac-sim/kit/cache:rw" \
    -v "${CACHE}/cache/main:/isaac-sim/.cache:rw" \
    -v "${CACHE}/cache/computecache:/isaac-sim/.nv/ComputeCache:rw" \
    -v "${CACHE}/logs:/isaac-sim/.nvidia-omniverse/logs:rw" \
    -v "${CACHE}/config:/isaac-sim/.nvidia-omniverse/config:rw" \
    -v "${CACHE}/data:/isaac-sim/.local/share/ov/data:rw" \
    -v "${CACHE}/pkg:/isaac-sim/.local/share/ov/pkg:rw" \
    -v /tmp/images_streaming:/tmp/images_streaming:rw \
    -v /tmp/images_oncommand:/tmp/images_oncommand:rw \
    -v /tmp/images_apxs:/tmp/images_apxs:rw \
    -v /tmp/images_depth:/tmp/images_depth:rw \
    -v /tmp/images_monitoring:/tmp/images_monitoring:rw \
    -v /tmp/images_lander:/tmp/images_lander:rw \
    -w /workspace/omnilrs \
    isaac-sim-omnilrs:latest \
    "$@"
