"""Startup stepping shared by ROS 2, Zenoh, Yamcs and SDG.

The counts come from ``cfg["_kit"]``. Missing config keeps the historical
100 rendered steps. Physics-only steps advance the same solver; they do not
change ``physics_dt``. After warmup, each manager's run loop still renders.
"""

from __future__ import annotations


def warmup_counts(cfg) -> tuple[int, int]:
    kit = {}
    if isinstance(cfg, dict):
        raw = cfg.get("_kit")
        if isinstance(raw, dict):
            kit = raw
    rendered = int(kit.get("warmup_rendered_frames", 100))
    physics = int(kit.get("warmup_physics_frames", 0))
    return rendered, physics


def step_warmup(world, cfg, reset: bool = True) -> None:
    rendered, physics = warmup_counts(cfg)
    for _ in range(physics):
        world.step(render=False)
    for _ in range(rendered):
        world.step(render=True)
    if reset:
        world.reset()
