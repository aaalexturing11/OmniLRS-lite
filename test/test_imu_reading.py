"""IMU reading mapping for Isaac Sim 6.2. Does not start Kit.

The integration check at the bottom runs only with OMNILRS_RUN_ISAAC=1, on the
Python that already starts Isaac Sim 6.2:

    export OMNILRS_RUN_ISAAC=1
    export ISAAC_PY=/isaac-sim/python.sh
    "$ISAAC_PY" -m pytest test/test_imu_reading.py -q
"""

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.robots.imu_reading import readings_from_imu

_ROOT = Path(__file__).resolve().parents[1]


def _reading(**overrides):
    values = dict(
        is_valid=True,
        linear_acceleration_x=0.1,
        linear_acceleration_y=0.2,
        linear_acceleration_z=9.81,
        angular_velocity_x=0.01,
        angular_velocity_y=-0.02,
        angular_velocity_z=0.03,
        orientation_w=1.0,
        orientation_x=0.0,
        orientation_y=0.0,
        orientation_z=0.0,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_identity_quaternion_is_zero_euler():
    acceleration, angular_velocity, orientation = readings_from_imu(_reading(), "/robot/imu")
    assert acceleration == {"ax": 0.1, "ay": 0.2, "az": 9.81}
    assert angular_velocity == {"gx": 0.01, "gy": -0.02, "gz": 0.03}
    assert orientation["roll"] == pytest.approx(0.0, abs=1e-6)
    assert orientation["pitch"] == pytest.approx(0.0, abs=1e-6)
    assert orientation["yaw"] == pytest.approx(0.0, abs=1e-6)


def test_yaw_uses_the_wxyz_components_as_xyzw():
    half = 0.7071067811865476
    _, _, orientation = readings_from_imu(
        _reading(orientation_w=half, orientation_x=0.0, orientation_y=0.0, orientation_z=half),
        "/robot/imu",
    )
    assert orientation["roll"] == pytest.approx(0.0, abs=1e-4)
    assert orientation["pitch"] == pytest.approx(0.0, abs=1e-4)
    assert orientation["yaw"] == pytest.approx(90.0, abs=1e-4)


def test_invalid_reading_is_not_published_as_a_measurement():
    acceleration, angular_velocity, orientation = readings_from_imu(_reading(is_valid=False), "/robot/imu")
    assert acceleration == {"ax": 0.0, "ay": 0.0, "az": 0.0}
    assert angular_velocity == {"gx": 0.0, "gy": 0.0, "gz": 0.0}
    assert orientation == {"roll": 0.0, "pitch": 0.0, "yaw": 0.0}


def test_robot_wraps_the_existing_prim_with_the_62_runtime():
    source = (_ROOT / "src" / "robots" / "robot.py").read_text(encoding="utf-8")
    assert "from isaacsim.sensors.experimental.physics import IMUSensor" in source
    assert "isaacsim.sensors.physics" not in source
    assert "acquire_imu_sensor_interface" not in source
    assert "lin_acc_x" not in source
    assert "IMUSensor(self._imu_sensor_path)" in source
    assert "get_sensor_reading(read_gravity=True)" in source
    assert "use_latest_data" not in source


@pytest.mark.skipif(os.environ.get("OMNILRS_RUN_ISAAC") != "1", reason="Isaac Sim is not launched from this checkout")
def test_isaac62_imu_runtime_exposes_the_new_reading():
    python = os.environ.get("ISAAC_PY", sys.executable)
    script = """
from isaacsim.sensors.experimental.physics import IMUSensor
import inspect
signature = inspect.signature(IMUSensor.get_sensor_reading)
assert "read_gravity" in signature.parameters
assert "use_latest_data" not in signature.parameters
print("imu-api-ok")
"""
    completed = subprocess.run(
        [python, "-c", script],
        cwd=_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    output = completed.stdout + "\n" + completed.stderr
    assert completed.returncode == 0, output
    assert "imu-api-ok" in output
