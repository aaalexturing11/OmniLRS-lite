"""Locks the Isaac Sim 6.2 runtime choice. These checks do not start Kit.

The CUDA failure on Isaac Sim 5.0 is Warp 1.7.1 calling cuDeviceGetUuid against
a driver that Isaac Sim 6.2 already runs. The image must be the official 6.2
tag, and the Docker build must not replace that image's Warp or NumPy.
"""

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def _text(relative: str) -> str:
    return (_ROOT / relative).read_text(encoding="utf-8")


def test_docker_image_is_official_isaac_sim_62():
    dockerfile = _text("omnilrs.docker/Dockerfile")
    assert "FROM nvcr.io/nvidia/isaac-sim:6.2.0" in dockerfile
    assert "isaac-sim:5.0.0" not in dockerfile
    assert "warp-lang" not in dockerfile
    assert "numpy==" not in dockerfile
    assert "isaacsim.ros2.bridge/humble/lib" not in dockerfile
    assert "/isaac-sim/python.sh" in dockerfile


def test_docker_entrypoint_does_not_force_system_ros():
    entry = _text("omnilrs.docker/entrypoint.sh")
    assert "unset ROS_DISTRO" in entry
    assert "source /opt/ros" not in entry


def test_run_script_starts_headless_lunalab_on_62_mounts():
    script = _text("omnilrs.docker/run_docker.sh")
    assert "rendering.renderer.headless=true" in script
    assert "environment=lunalab" in script
    assert "/isaac-sim/.cache" in script
    assert "OMNILRS_MEMORY_PROFILE=auto" in script
    assert "isaac-sim:5.0.0" not in script


def test_pixi_requests_isaac_sim_62_without_a_numpy_pin():
    pixi = _text("pixi.toml")
    assert 'version = "==6.2.0.0"' in pixi
    assert "5.0.0" not in pixi
    assert 'numpy = "==1.26.0"' not in pixi
    assert 'python = "3.12.*"' in pixi
    assert 'ROS_DISTRO = "humble"' not in pixi


def test_project_does_not_require_a_warp_replacement():
    project = _text("pyproject.toml")
    dependencies = project.split("[project.optional-dependencies]", 1)[0]
    assert "warp-lang" not in dependencies
    assert "warp-lang>=1.8.1" in project


def test_ros_activation_prefers_the_62_core_library():
    script = _text("scripts/activate_isaac_ros.sh")
    assert "isaacsim.ros2.core" in script
    assert "python3.12" in script
    assert '[ -z "${ROS_DISTRO:-}" ]' in script
    loop = "for name in isaacsim.ros2.core isaacsim.ros2.bridge"
    assert loop in script
