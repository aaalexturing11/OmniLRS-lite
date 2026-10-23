#!/usr/bin/env bash
# Prepend the Isaac Sim 6.2 bundled ROS libraries when they are present.
# Isaac Sim 5.0 used exts/isaacsim.ros2.bridge/<distro>/lib under Python 3.11.
# Isaac Sim 6.2 uses exts/isaacsim.ros2.core/<distro>/lib under Python 3.12.
# A missing directory is skipped so an old path cannot shadow the driver.

# An unset ROS_DISTRO is the Isaac Sim 6.2 default: the launcher adds the
# bundled libraries itself. Prepending a distro here would freeze that choice.
if [ -z "${CONDA_PREFIX:-}" ] || [ -z "${ROS_DISTRO:-}" ]; then
    exit 0
fi

distro="${ROS_DISTRO}"
other="jazzy"
if [ "$distro" = "jazzy" ]; then
    other="humble"
fi

shopt -s nullglob
candidates=()
for py in python3.12 python3.11; do
    for name in isaacsim.ros2.core isaacsim.ros2.bridge; do
        candidates+=("${CONDA_PREFIX}/lib/${py}/site-packages/isaacsim/exts/${name}/${distro}/lib")
    done
done
for py in python3.12 python3.11; do
    for name in isaacsim.ros2.core isaacsim.ros2.bridge; do
        candidates+=("${CONDA_PREFIX}/lib/${py}/site-packages/isaacsim/exts/${name}/${other}/lib")
    done
done
shopt -u nullglob

for bridge_lib in "${candidates[@]}"; do
    if [ -d "$bridge_lib" ]; then
        case ":${LD_LIBRARY_PATH:-}:" in
            *":${bridge_lib}:"*) ;;
            *)
                if [ -n "${LD_LIBRARY_PATH:-}" ]; then
                    export LD_LIBRARY_PATH="${bridge_lib}:${LD_LIBRARY_PATH}"
                else
                    export LD_LIBRARY_PATH="${bridge_lib}"
                fi
                ;;
        esac
        break
    fi
done
