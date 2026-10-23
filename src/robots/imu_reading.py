"""Map an Isaac Sim 6.2 IMU reading onto the telemetry dicts OmniLRS already publishes.

The function does not import Isaac. ``IMUSensor.get_sensor_reading()`` returns
``linear_acceleration_*``, ``angular_velocity_*`` and ``orientation_w/x/y/z``.
Orientation is a world-frame quaternion stored w, x, y, z. The existing Euler
conversion takes x, y, z, w.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R

_ZERO_ACCELERATION = {"ax": 0.0, "ay": 0.0, "az": 0.0}
_ZERO_ANGULAR_VELOCITY = {"gx": 0.0, "gy": 0.0, "gz": 0.0}
_ZERO_ORIENTATION = {"roll": 0.0, "pitch": 0.0, "yaw": 0.0}


def _xyzw_to_xyz_degrees(orientation_xyzw):
    """Same conversion as ``transform_orientation_from_xyzw_into_xyz``."""

    q_xyzw = np.array(orientation_xyzw)
    e_zyx_from_wxyz = R.from_quat(q_xyzw).as_euler("zyx", degrees=True)
    return np.array([e_zyx_from_wxyz[2], e_zyx_from_wxyz[1], e_zyx_from_wxyz[0]])


def readings_from_imu(reading, sensor_path: str):
    """Copy one IMU sample into acceleration, angular velocity and roll/pitch/yaw.

    ``read_gravity=True`` on the caller keeps specific force, the same quantity
    the Isaac Sim 5 interface returned. An invalid reading is the shared
    placeholder from Isaac Sim 6.2, so it is reported as zeros.
    """

    if not bool(getattr(reading, "is_valid", True)):
        print(
            f"[WARN] IMU reading is not valid at path {sensor_path}. "
            "Sensor is probably not initialized yet."
        )
        return _ZERO_ACCELERATION.copy(), _ZERO_ANGULAR_VELOCITY.copy(), _ZERO_ORIENTATION.copy()

    linear_acceleration = {
        "ax": float(reading.linear_acceleration_x),
        "ay": float(reading.linear_acceleration_y),
        "az": float(reading.linear_acceleration_z),
    }
    angular_velocity = {
        "gx": float(reading.angular_velocity_x),
        "gy": float(reading.angular_velocity_y),
        "gz": float(reading.angular_velocity_z),
    }

    raw_orientation = np.array(
        [
            reading.orientation_x,
            reading.orientation_y,
            reading.orientation_z,
            reading.orientation_w,
        ],
        dtype=float,
    )

    if raw_orientation.shape != (4,):
        print(f"[WARN] Invalid IMU orientation shape: {raw_orientation}")
        xyz_orientation = np.zeros(3)
    elif np.linalg.norm(raw_orientation) < 1e-8:
        print(
            f"[WARN] IMU returned zero quaternion at path {sensor_path}: {raw_orientation}. "
            "Sensor is probably not initialized yet."
        )
        xyz_orientation = np.zeros(3)
    else:
        raw_orientation = raw_orientation / np.linalg.norm(raw_orientation)
        xyz_orientation = _xyzw_to_xyz_degrees(raw_orientation)

    orientation = {
        "roll": -float(xyz_orientation[0]),
        "pitch": -float(xyz_orientation[1]),
        "yaw": float(xyz_orientation[2]),
    }
    return linear_acceleration, angular_velocity, orientation
