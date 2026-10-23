"""Checks for the memory-budget and terrain-buffer changes.

These tests do not start Isaac Sim. They lock the topology and the DEM shift
so a later edit cannot silently change the mesh or drop height samples.
"""

import os
from pathlib import Path

import numpy as np

from src.runtime.memory_budget import (
    apply_memory_budget,
    classify_profile,
    estimate_hr_dem_bytes,
    process_memory,
    renderer_runtime_flags,
)
from src.terrain_management.array_utils import load_transposed_elevation
from src.terrain_management.large_scale_terrain.dem_shift import shift_heightfield
from src.terrain_management.terrain_grid import build_terrain_grid

_GIB = 1024**3


def _reference_shift(dem, x_shift, y_shift):
    out = np.zeros_like(dem)
    height, width = dem.shape
    if x_shift == 0 and y_shift == 0:
        return dem.copy()
    if abs(x_shift) >= height or abs(y_shift) >= width:
        return out
    x_src = slice(0, height - x_shift) if x_shift >= 0 else slice(-x_shift, height)
    x_dst = slice(x_shift, height) if x_shift >= 0 else slice(0, height + x_shift)
    y_src = slice(0, width - y_shift) if y_shift >= 0 else slice(-y_shift, width)
    y_dst = slice(y_shift, width) if y_shift >= 0 else slice(0, width + y_shift)
    out[x_dst, y_dst] = dem[x_src, y_src]
    return out


def _naive_grid(sim_length, sim_width, grid_size):
    """The previous Python-list builder, kept here only as an oracle."""

    vertices = []
    indices = []
    uvs = []
    for y in range(sim_length):
        for x in range(sim_width):
            vertices.append((float(x) * grid_size, float(y) * grid_size, 0.0))
            if x > 0 and y > 0:
                stride = sim_width
                indices.extend(
                    [
                        y * stride + x - stride,
                        y * stride + x,
                        y * stride + x - stride - 1,
                        y * stride + x,
                        y * stride + x - 1,
                        y * stride + x - stride - 1,
                    ]
                )
                uvs.extend([(x, y - 1), (x, y), (x - 1, y - 1), (x, y), (x - 1, y), (x - 1, y - 1)])
    return (
        np.array(vertices, dtype=np.float32),
        np.array(indices, dtype=np.int32),
        np.array(uvs, dtype=np.float32) * grid_size,
    )


def test_terrain_grid_matches_the_previous_topology():
    grid = build_terrain_grid(6, 5, 0.02)
    vertices, indices, uvs = _naive_grid(6, 5, 0.02)
    assert grid.vertices.shape == (6 * 5, 3)
    assert grid.vertices.dtype == np.float32
    assert grid.indices.dtype == np.int32
    assert np.array_equal(grid.vertices, vertices)
    assert np.array_equal(grid.indices, indices)
    assert np.array_equal(grid.uvs, uvs.astype(np.float32))
    assert grid.face_vertex_counts.shape == (len(indices) // 3,)
    assert np.all(grid.face_vertex_counts == 3)


def test_heightfield_shift_matches_an_out_of_place_copy():
    source = np.arange(80, dtype=np.float32).reshape(8, 10)
    for x_shift, y_shift in ((0, 0), (2, 3), (-2, 4), (3, -1), (-4, -2), (8, 1), (2, -10)):
        moved = source.copy()
        shift_heightfield(moved, x_shift, y_shift)
        expected = _reference_shift(source, x_shift, y_shift)
        assert np.array_equal(moved, expected)


def test_transposed_elevation_is_float32_and_matches_the_historical_layout(tmp_path):
    source = np.arange(12, dtype=np.float64).reshape(3, 4)
    path = tmp_path / "dem.npy"
    np.save(path, source)
    loaded = load_transposed_elevation(str(path))
    expected = np.ascontiguousarray(np.flip(source.T, axis=1), dtype=np.float32)
    assert loaded.dtype == np.float32
    assert loaded.flags.c_contiguous
    assert np.array_equal(loaded, expected)


def test_low_profile_shrinks_the_window_without_dropping_below_rock_span(monkeypatch):
    monkeypatch.setenv("OMNILRS_MEMORY_PROFILE", "auto")
    cfg = _sample_cfg()
    record = apply_memory_budget(cfg, memory={"total": 8 * _GIB, "available": 3 * _GIB}, gpus=[4096])
    terrain = cfg["environment"]["large_scale_terrain"]
    assert record["name"] == "low"
    assert terrain["hr_dem_num_blocks"] == 2
    assert terrain["num_workers_craters"] == 2
    assert terrain["geo_cm_coarse_acceleration_mode"] == "hybrid"
    assert cfg["rendering"]["renderer"]["samples_per_pixel_per_frame"] == 1
    assert cfg["rendering"]["renderer"]["multi_gpu"] is False
    assert estimate_hr_dem_bytes(4, 50, 0.025) > estimate_hr_dem_bytes(2, 50, 0.025)


def test_64gb_ram_with_8gb_vram_caps_the_viewport_only(monkeypatch):
    monkeypatch.setenv("OMNILRS_MEMORY_PROFILE", "auto")
    cfg = _sample_cfg()
    record = apply_memory_budget(cfg, memory={"total": 64 * _GIB, "available": 40 * _GIB}, gpus=[8192])
    renderer = cfg["rendering"]["renderer"]
    terrain = cfg["environment"]["large_scale_terrain"]
    assert record["name"] == "high"
    assert record["vram_class"] == "8gb"
    assert renderer["samples_per_pixel_per_frame"] == 1
    assert renderer["max_bounces"] == 6
    assert renderer["max_specular_transmission_bounces"] == 6
    assert renderer["width"] == 1280
    assert renderer["height"] == 720
    assert renderer["multi_gpu"] is False
    assert terrain["hr_dem_num_blocks"] == 4
    assert terrain["num_workers_craters"] == 8
    assert terrain["geo_cm_coarse_acceleration_mode"] == "hybrid"
    assert cfg["physics"]["physics_scene"] == {}
    assert renderer_runtime_flags(record) == {"texture_streaming": True, "eco_mode": False}


def test_explicit_high_keeps_viewport_samples_on_an_8gb_gpu(monkeypatch):
    monkeypatch.setenv("OMNILRS_MEMORY_PROFILE", "high")
    cfg = _sample_cfg()
    record = apply_memory_budget(cfg, memory={"total": 64 * _GIB, "available": 40 * _GIB}, gpus=[8192])
    assert record["vram_class"] == "8gb"
    assert cfg["rendering"]["renderer"]["samples_per_pixel_per_frame"] == 32
    assert "width" not in cfg["rendering"]["renderer"]
    assert renderer_runtime_flags(record)["texture_streaming"] is False
    assert cfg["environment"]["large_scale_terrain"]["geo_cm_coarse_acceleration_mode"] == "hybrid"


def test_vram_between_8_and_12gb_keeps_samples_and_streams_textures(monkeypatch):
    monkeypatch.setenv("OMNILRS_MEMORY_PROFILE", "auto")
    cfg = _sample_cfg()
    record = apply_memory_budget(cfg, memory={"total": 64 * _GIB, "available": 40 * _GIB}, gpus=[10240])
    assert record["name"] == "high"
    assert record["vram_class"] == "limited"
    assert cfg["rendering"]["renderer"]["samples_per_pixel_per_frame"] == 32
    assert cfg["environment"]["large_scale_terrain"]["hr_dem_num_blocks"] == 4
    assert cfg["environment"]["large_scale_terrain"]["geo_cm_coarse_acceleration_mode"] == "hybrid"
    assert renderer_runtime_flags(record)["texture_streaming"] is True
    assert renderer_runtime_flags(record)["eco_mode"] is False


def test_rgba_and_depth_share_one_camera_product():
    text = (Path(__file__).resolve().parents[1] / "src" / "robots" / "robot.py").read_text(encoding="utf-8")
    body = text[text.index("def _initialize_cameras") : text.index("def get_rgba_camera_view")]
    assert body.count("Camera(") == 1
    assert "_depth_cameras[res] = camera" in body
    assert "_cameras[res] = camera" in body


def test_high_profile_keeps_configured_quality(monkeypatch):
    monkeypatch.setenv("OMNILRS_MEMORY_PROFILE", "auto")
    cfg = _sample_cfg()
    record = apply_memory_budget(cfg, memory={"total": 64 * _GIB, "available": 40 * _GIB}, gpus=[24576, 24576])
    terrain = cfg["environment"]["large_scale_terrain"]
    assert record["name"] == "high"
    assert terrain["hr_dem_num_blocks"] == 4
    assert terrain["num_workers_craters"] == 8
    assert cfg["rendering"]["renderer"]["samples_per_pixel_per_frame"] == 32
    assert cfg["rendering"]["renderer"]["multi_gpu"] is True
    assert record["adjustments"] == []


def test_explicit_profile_is_not_overridden_by_free_ram(monkeypatch):
    monkeypatch.setenv("OMNILRS_MEMORY_PROFILE", "high")
    assert classify_profile(64 * _GIB, 1 * _GIB, "high") == "high"
    assert classify_profile(8 * _GIB, 6 * _GIB, "off") == "off"
    assert classify_profile(64 * _GIB, 1 * _GIB, "auto") == "low"


def test_single_gpu_disables_multi_gpu_on_a_large_host(monkeypatch):
    monkeypatch.delenv("OMNILRS_MEMORY_PROFILE", raising=False)
    cfg = _sample_cfg()
    apply_memory_budget(cfg, memory={"total": 64 * _GIB, "available": 40 * _GIB}, gpus=[16384])
    assert cfg["rendering"]["renderer"]["multi_gpu"] is False
    assert cfg["environment"]["large_scale_terrain"]["hr_dem_num_blocks"] == 4


def test_process_memory_is_separate_from_system_ram():
    info = process_memory()
    if os.name != "nt":
        return
    assert info["working_set"] > 0
    assert info["commit"] > 0


def _sample_cfg():
    return {
        "rendering": {
            "renderer": {
                "renderer": "RayTracedLighting",
                "samples_per_pixel_per_frame": 32,
                "max_bounces": 6,
                "max_specular_transmission_bounces": 6,
                "max_volume_bounces": 4,
                "multi_gpu": True,
            }
        },
        "environment": {
            "name": "LargeScale",
            "large_scale_terrain": {
                "hr_dem_num_blocks": 4,
                "block_size": 50,
                "hr_dem_resolution": 0.025,
                "num_workers_craters": 8,
                "num_workers_interpolation": 1,
                "input_queue_size": 400,
                "output_queue_size": 30,
                "geo_cm_coarse_acceleration_mode": "gpu",
                "rock_gen_cfgs": [{"block_span": 2}, {"block_span": 1}],
            },
        },
        "physics": {"physics_scene": {}},
    }

