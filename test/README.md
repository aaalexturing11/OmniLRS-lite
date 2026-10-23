# Model Unit Tests

Standalone tests for the physics models (power, thermal, radio, camera compression).
They exercise each model in isolation and produce diagnostic plots.

## Setup

```bash
# from the repository root (OmniLRS/)
python3.11 -m venv test/venv
source test/venv/bin/activate
pip install -r test/requirements.txt
```

> **Note:** Python 3.11+ is required for these model tests (`StrEnum` support).
> Isaac Sim 6.2 itself runs on Python 3.12.

## Run

All commands assume the venv is active and are executed from the repository root:

```bash
python -m src.subsystems.robot_physics_models.radio_model
python -m src.subsystems.robot_physics_models.thermal_model
python -m src.subsystems.robot_physics_models.power_model
python -m src.subsystems.robot_physics_models.camera_compression_model
```

Each script will print a message and save a `.png` plot in the current directory.

The camera compression test streams the 640x480 rover frames in `test/inputs/camera_compression/` through the model for six bitrate / packet-loss / keyframe scenarios. For each scenario it writes the processed frames and a 1 fps video to `test/outputs/camera_compression/<scenario>/`, and prints the bandwidth and packet-loss counts.