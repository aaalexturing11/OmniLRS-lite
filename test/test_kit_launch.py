"""Launch policy, warmup counts and subsystem models that do not need Kit.

These tests do not start Isaac Sim. They lock the decisions that keep
sensors, PhysX and the steady-state render loop in place.
"""

import ast
from pathlib import Path

import numpy as np
import pytest

from src.runtime.kit_launch import REQUIRED_EXTENSIONS, UI_EXTENSIONS, build_kit_launch
from src.runtime.sim_warmup import step_warmup
from src.subsystems.robot_physics_models.radio_model import BEST_RSSI, REFERENCE_DISTANCE, WORST_RSSI, RadioModel
from src.subsystems.robot_physics_models.thermal_model import TIME_CONSTANT, ThermalModel

_GIB = 1024**3
_ROOT = Path(__file__).resolve().parents[1]

_KIT_ENV = (
    "EXP_PATH",
    "OMNILRS_EXPERIENCE",
    "OMNILRS_KEEP_WINDOW",
    "OMNILRS_DISABLE_VIEWPORT",
    "OMNILRS_DISABLE_UI_EXTENSIONS",
    "OMNILRS_WARMUP_RENDERED",
    "OMNILRS_WARMUP_PHYSICS",
    "OMNILRS_MEMORY_PROFILE",
)


@pytest.fixture(autouse=True)
def _clear_kit_env(monkeypatch):
    for name in _KIT_ENV:
        monkeypatch.delenv(name, raising=False)


def _cfg(
    total,
    available,
    profile,
    requested="auto",
    mode="ROS2",
    headless=False,
    width=1280,
    height=720,
    gpus=None,
):
    return {
        "mode": {"name": mode},
        "rendering": {
            "renderer": {
                "headless": headless,
                "width": width,
                "height": height,
                "samples_per_pixel_per_frame": 32,
                "max_bounces": 6,
                "renderer": "RayTracedLighting",
            }
        },
        "_memory": {
            "name": profile,
            "requested": requested,
            "total_bytes": total,
            "available_bytes": available,
            "gpu_memory_mib": gpus,
        },
    }


def _method_source(relative: str, class_name: str, method: str) -> str:
    path = _ROOT / relative
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method:
                    segment = ast.get_source_segment(text, item)
                    assert segment is not None
                    return segment
    raise AssertionError(f"{class_name}.{method} not found in {relative}")


def _imported_modules(relative: str) -> list[str]:
    tree = ast.parse((_ROOT / relative).read_text(encoding="utf-8"))
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


def test_low_installed_ram_runs_headless_without_dropping_sensors():
    cfg = _cfg(8 * _GIB, 3 * _GIB, "low", gpus=[4096])
    plan = build_kit_launch(cfg)
    launch = plan["launch_updates"]
    assert plan["forced_headless"] is True
    assert cfg["rendering"]["renderer"]["headless"] is True
    assert launch["headless"] is True
    assert launch["disable_viewport_updates"] is True
    assert launch["hide_ui"] is True
    assert launch["limit_cpu_threads"] == 4
    assert launch["max_gpu_count"] == 1
    assert "window_width" not in launch
    assert plan["warmup_physics_frames"] == 98
    assert plan["warmup_rendered_frames"] == 2
    assert plan["skip_replicator_on_close"] is True
    assert plan["experience"] == ""
    assert "samples_per_pixel_per_frame" not in launch
    assert "max_bounces" not in launch
    assert "physics_dt" not in launch
    disabled = " ".join(launch.get("extra_args", []))
    for name in REQUIRED_EXTENSIONS:
        assert name not in disabled


def test_keep_window_on_a_small_host_leaves_the_viewport_enabled(monkeypatch):
    monkeypatch.setenv("OMNILRS_KEEP_WINDOW", "1")
    plan = build_kit_launch(_cfg(8 * _GIB, 3 * _GIB, "low", gpus=[4096]))
    launch = plan["launch_updates"]
    assert plan["forced_headless"] is False
    assert "headless" not in launch
    assert "disable_viewport_updates" not in launch
    assert launch["window_width"] == 1280
    assert launch["window_height"] == 720
    assert plan["warmup_physics_frames"] == 98
    assert plan["warmup_rendered_frames"] == 2


def test_explicit_high_on_a_small_host_keeps_the_window_and_the_warmup():
    plan = build_kit_launch(_cfg(8 * _GIB, 2 * _GIB, "high", requested="high", width=None, height=None, gpus=[8192]))
    launch = plan["launch_updates"]
    assert "headless" not in launch
    assert "limit_cpu_threads" not in launch
    assert "disable_viewport_updates" not in launch
    assert launch["max_gpu_count"] == 1
    assert plan["warmup_rendered_frames"] == 100
    assert plan["warmup_physics_frames"] == 0


def test_large_host_keeps_the_historical_startup():
    plan = build_kit_launch(
        _cfg(64 * _GIB, 40 * _GIB, "high", requested="auto", width=None, height=None, gpus=[24576, 24576])
    )
    launch = plan["launch_updates"]
    assert launch == {}
    assert plan["forced_headless"] is False
    assert plan["warmup_rendered_frames"] == 100
    assert plan["warmup_physics_frames"] == 0
    assert plan["skip_replicator_on_close"] is True


def test_sdg_on_a_small_host_keeps_replicator_and_rendered_warmup():
    plan = build_kit_launch(_cfg(8 * _GIB, 3 * _GIB, "low", mode="SDG", gpus=[4096]))
    launch = plan["launch_updates"]
    assert "headless" not in launch
    assert "disable_viewport_updates" not in launch
    assert plan["skip_replicator_on_close"] is False
    assert plan["warmup_rendered_frames"] == 100
    assert plan["warmup_physics_frames"] == 0
    assert launch["limit_cpu_threads"] == 4


def test_yaml_headless_on_a_small_host_skips_the_unused_viewport():
    plan = build_kit_launch(_cfg(8 * _GIB, 3 * _GIB, "low", headless=True, gpus=[4096]))
    assert plan["forced_headless"] is False
    assert plan["launch_updates"]["disable_viewport_updates"] is True


def test_viewport_disable_on_a_large_host_is_opt_in(monkeypatch):
    monkeypatch.setenv("OMNILRS_DISABLE_VIEWPORT", "1")
    quiet = build_kit_launch(_cfg(64 * _GIB, 40 * _GIB, "high", headless=False, width=None, height=None))
    assert "disable_viewport_updates" not in quiet["launch_updates"]
    headless = build_kit_launch(_cfg(64 * _GIB, 40 * _GIB, "high", headless=True, width=None, height=None))
    assert headless["launch_updates"]["disable_viewport_updates"] is True


def test_ui_extensions_are_opt_in_and_cannot_include_sensors(monkeypatch):
    monkeypatch.setenv("OMNILRS_DISABLE_UI_EXTENSIONS", "1")
    plan = build_kit_launch(_cfg(8 * _GIB, 3 * _GIB, "low", gpus=[4096]))
    extra = plan["launch_updates"]["extra_args"]
    disabled = extra[1::2]
    assert extra[0::2] == ["--disable"] * len(UI_EXTENSIONS)
    assert set(disabled) == set(UI_EXTENSIONS)
    assert set(disabled).isdisjoint(REQUIRED_EXTENSIONS)


def test_experience_uses_base_python_kit_when_the_file_exists(tmp_path, monkeypatch):
    kit = tmp_path / "isaacsim.exp.base.python.kit"
    kit.write_text("dependencies = []\n", encoding="utf-8")
    (tmp_path / "isaacsim.exp.full.kit").write_text("dependencies = []\n", encoding="utf-8")
    monkeypatch.setenv("EXP_PATH", str(tmp_path))
    plan = build_kit_launch(_cfg(64 * _GIB, 40 * _GIB, "high", width=None, height=None))
    assert plan["experience"] == str(kit)

    monkeypatch.delenv("EXP_PATH")
    monkeypatch.setenv("OMNILRS_EXPERIENCE", str(tmp_path / "isaacsim.exp.full.kit"))
    override = build_kit_launch(_cfg(64 * _GIB, 40 * _GIB, "high", width=None, height=None))
    assert override["experience"].endswith("isaacsim.exp.full.kit")


def test_missing_base_kit_leaves_the_experience_empty(tmp_path, monkeypatch):
    (tmp_path / "isaacsim.exp.full.kit").write_text("dependencies = []\n", encoding="utf-8")
    monkeypatch.setenv("EXP_PATH", str(tmp_path))
    plan = build_kit_launch(_cfg(64 * _GIB, 40 * _GIB, "high", width=None, height=None))
    assert plan["experience"] == ""


def test_warmup_steps_physics_before_the_rendered_frames():
    class World:
        def __init__(self):
            self.calls = []

        def step(self, render):
            self.calls.append(render)

        def reset(self):
            self.calls.append("reset")

    world = World()
    step_warmup(world, {"_kit": {"warmup_rendered_frames": 2, "warmup_physics_frames": 3}})
    assert world.calls == [False, False, False, True, True, "reset"]

    plain = World()
    step_warmup(plain, {}, reset=False)
    assert plain.calls == [True] * 100


def test_run_loops_still_render_every_step():
    ros2 = _method_source(
        "src/environments_wrappers/ros2/simulation_manager_ros2.py",
        "ROS2_SimulationManager",
        "run_simulation",
    )
    yamcs = _method_source(
        "src/environments_wrappers/yamcs/simulation_manager_yamcs.py",
        "Yamcs_SimulationManager",
        "run_simulation",
    )
    sdg = _method_source(
        "src/environments_wrappers/sdg/simulation_manager_sdg.py",
        "SDG_SimulationManager",
        "run_simulation",
    )
    zenoh = _method_source(
        "src/environments_wrappers/zenoh/simulation_manager_zenoh.py",
        "Zenoh_SimulationManager",
        "run_simulation",
    )
    assert "self.world.step(render=True)" in ros2
    assert "render=False" not in ros2
    assert "self.world.step(render=True)" in yamcs
    assert "render=False" not in yamcs
    assert "self.world.step(render=True)" in sdg
    assert "self.simulation_app.update()" in zenoh

    ros2_init = _method_source(
        "src/environments_wrappers/ros2/simulation_manager_ros2.py",
        "ROS2_SimulationManager",
        "__init__",
    )
    assert ros2_init.count("step_warmup") == 2
    sdg_init = _method_source(
        "src/environments_wrappers/sdg/simulation_manager_sdg.py",
        "SDG_SimulationManager",
        "__init__",
    )
    assert "reset=False" in sdg_init


def test_subsystem_models_do_not_import_kit():
    decoupled = [
        "src/stellar/stellar_engine.py",
        "src/subsystems/robot_physics_models/radio_model.py",
        "src/subsystems/robot_physics_models/power_model.py",
        "src/subsystems/robot_physics_models/thermal_model.py",
        "src/subsystems/robot_physics_models/obc_metrics_model.py",
        "src/subsystems/robot_physics_models/camera_compression_model.py",
    ]
    for relative in decoupled:
        for name in _imported_modules(relative):
            assert not name.startswith(("omni", "isaacsim", "pxr")), relative
    mixin = _imported_modules("src/environments/stellar_engine_env_mixin.py")
    assert any(name == "pxr" or name.startswith("pxr.") for name in mixin)


def test_radio_rssi_follows_the_quadratic_falloff():
    model = RadioModel()
    model.set_inputs((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    assert model._calculate_rssi() == pytest.approx(BEST_RSSI)
    model.set_inputs((0.0, 0.0, 0.0), (REFERENCE_DISTANCE, 0.0, 0.0))
    assert model._calculate_rssi() == pytest.approx(WORST_RSSI)
    model.set_inputs((0.0, 0.0, 0.0), (REFERENCE_DISTANCE / 2.0, 0.0, 0.0))
    expected = BEST_RSSI + (WORST_RSSI - BEST_RSSI) * 0.25
    assert model._calculate_rssi() == pytest.approx(expected)


def test_thermal_faces_track_the_sun_without_kit():
    model = ThermalModel()
    sun = np.array((0.0, 0.0, 1.0))
    model.set_inputs(sun, 0.0)
    factors = model._compute_view_factors(model._sun_direction)
    assert factors["+Z"] == pytest.approx(1.0)
    assert factors["-Z"] == pytest.approx(0.0)
    assert factors["+X"] == pytest.approx(0.0)
    model.compute(TIME_CONSTANT)
    assert model._node_temps["+Z"] == pytest.approx(model._target_temperature(1.0))
    assert model._node_temps["-Z"] == pytest.approx(model._target_temperature(0.0))
    assert model._node_temps["+Y"] == pytest.approx(model._target_temperature(0.0))


def test_stellar_engine_computes_a_finite_sun_position():
    pytest.importorskip("skyfield")
    pytest.importorskip("scipy")
    from src.configurations.stellar_engine_confs import StellarEngineConf
    from src.stellar.stellar_engine import StellarEngine

    engine = StellarEngine(StellarEngineConf())
    engine.set_lat_lon(-89.5, 0.0)
    # de421, default date 2024-05-11 16:30 UTC, skyfield 1.55.
    altitude, azimuth, distance = engine.get_alt_az("sun")
    assert altitude == pytest.approx(-1.3348060645844735, rel=1e-6)
    assert azimuth == pytest.approx(140.1270022764254, rel=1e-6)
    assert distance == pytest.approx(150849434263.81134, rel=1e-6)
    earth_alt, earth_az, earth_dist = engine.get_alt_az("earth")
    assert earth_alt == pytest.approx(6.928977184654222, rel=1e-6)
    assert earth_az == pytest.approx(6.160473688170155, rel=1e-6)
    assert earth_dist == pytest.approx(383662407.39391404, rel=1e-6)
