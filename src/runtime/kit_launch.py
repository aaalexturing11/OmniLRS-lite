"""Isaac Sim 5.0 launch policy. This module does not import Kit.

Standalone ``SimulationApp`` already loads ``isaacsim.exp.base.python.kit``.
Selecting ``isaacsim.exp.full.kit`` would pull in the editor and raise memory.
The policy below only adds launch keys Kit already understands.

PhysX timestep, solver, CCD, gravity and the steady-state ``render=True``
loop are left alone. Fabric pose updates stay enabled because OmniLRS reads
articulation and prim transforms. Sensor and camera extensions are never
passed to ``--disable``.
"""

from __future__ import annotations

import logging
import os

from src.runtime.memory_budget import classify_profile, host_memory

logger = logging.getLogger("omnilrs.kit")

_GIB = 1024**3

# Present so a later edit cannot pass these names to ``--disable``.
REQUIRED_EXTENSIONS = (
    "isaacsim.core.api",
    "isaacsim.sensors.camera",
    "isaacsim.sensors.rtx",
    "omni.physx",
    "omni.hydra.rtx",
    "omni.usd",
)

# UI shells. They are not the viewport render product, PhysX or the sensors.
# Applied only when OMNILRS_DISABLE_UI_EXTENSIONS=1, because a hard dependency
# inside base.python.kit was not verified against a running Kit 107.3.
UI_EXTENSIONS = (
    "omni.kit.window.toolbar",
    "omni.kit.window.status_bar",
    "omni.kit.window.console",
    "omni.kit.widget.cache_indicator",
    "isaacsim.gui.menu",
    "omni.kit.viewport.menubar.camera",
    "omni.kit.viewport.menubar.display",
    "omni.kit.viewport.menubar.render",
    "omni.kit.viewport.menubar.settings",
)

# 98 physics steps plus 2 rendered steps keep the historical 100 physics
# updates. The run loop still renders every frame after this.
LOW_WARMUP_PHYSICS = 98
LOW_WARMUP_RENDERED = 2
DEFAULT_WARMUP_RENDERED = 100

_FORBIDDEN_LAUNCH_KEYS = (
    "samples_per_pixel_per_frame",
    "max_bounces",
    "max_specular_transmission_bounces",
    "max_volume_bounces",
    "physics_dt",
    "rendering_dt",
    "use_gpu_pipeline",
)
_FORBIDDEN_ARG_FRAGMENTS = (
    "fabricUpdate",
    "skipMaterialLoading",
    "minFrameRate",
    "isaacsim.exp.full.kit",
)


def installed_class(total_bytes: int) -> str:
    """Class from installed RAM only. Free RAM does not change this label."""

    if total_bytes < 12 * _GIB:
        return "low"
    if total_bytes < 24 * _GIB:
        return "balanced"
    return "high"


def resolve_experience(env: dict | None = None) -> str:
    """Absolute base-kit path when it exists on disk, otherwise an empty string.

    An empty string keeps ``SimulationApp``'s own default, which for Isaac Sim
    5.0 standalone is ``isaacsim.exp.base.python.kit``. ``EXP_PATH`` is never
    resolved to ``isaacsim.exp.full.kit``.
    """

    env = os.environ if env is None else env
    override = env.get("OMNILRS_EXPERIENCE", "").strip()
    if override:
        return override if os.path.isfile(override) else ""
    root = env.get("EXP_PATH", "").strip()
    if not root:
        return ""
    path = os.path.join(root, "isaacsim.exp.base.python.kit")
    if os.path.basename(path) != "isaacsim.exp.base.python.kit":
        return ""
    return path if os.path.isfile(path) else ""


def _field(obj, name: str, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    getter = getattr(obj, "get", None)
    if callable(getter):
        try:
            return getter(name, default)
        except Exception:
            return default
    return getattr(obj, name, default)


def _enabled(env: dict, name: str, default: str = "0") -> bool:
    return str(env.get(name, default)).strip().lower() in ("1", "true", "yes", "on")


def _optional_int(env: dict, name: str):
    raw = str(env.get(name, "")).strip()
    if not raw:
        return None
    return int(raw)


def _memory_record(cfg: dict) -> dict:
    record = cfg.get("_memory") if isinstance(cfg, dict) else None
    if isinstance(record, dict) and "total_bytes" in record:
        return record
    memory = host_memory()
    requested = os.environ.get("OMNILRS_MEMORY_PROFILE", "auto")
    return {
        "name": classify_profile(memory["total"], memory["available"], requested),
        "requested": requested,
        "total_bytes": memory["total"],
        "available_bytes": memory["available"],
        "gpu_memory_mib": None,
    }


def _set_headless(renderer, value: bool) -> None:
    if isinstance(renderer, dict):
        renderer["headless"] = value
    elif renderer is not None and hasattr(renderer, "headless"):
        renderer.headless = value


def _guard(launch_updates: dict) -> None:
    overlap = set(launch_updates) & set(_FORBIDDEN_LAUNCH_KEYS)
    if overlap:
        raise RuntimeError(f"kit launch must not override {sorted(overlap)}")
    extra = launch_updates.get("extra_args") or []
    disabled = []
    tokens = iter(extra)
    for token in tokens:
        if token == "--disable":
            disabled.append(next(tokens))
    required = set(disabled) & set(REQUIRED_EXTENSIONS)
    if required:
        raise RuntimeError(f"refusing to disable {sorted(required)}")
    blob = " ".join(str(token) for token in extra)
    for fragment in _FORBIDDEN_ARG_FRAGMENTS:
        if fragment in blob:
            raise RuntimeError(f"refusing kit argument containing {fragment}")


def build_kit_launch(cfg: dict, env: dict | None = None) -> dict:
    """Return the experience path and the SimulationApp keys for this host.

    The returned ``launch_updates`` is merged on top of ``RendererConf``.
    Sample counts, bounces and the physics scene are not in that dict.
    """

    env = os.environ if env is None else env
    record = _memory_record(cfg)
    requested = record.get("requested", "auto")
    total = int(record["total_bytes"])
    available = int(record.get("available_bytes", total))
    profile = record.get("name")
    if profile not in ("off", "low", "balanced", "high"):
        profile = classify_profile(total, available, requested)
    host_class = installed_class(total)
    preserve = requested in ("off", "high") or profile in ("off", "high")

    mode = _field(cfg.get("mode") if isinstance(cfg, dict) else None, "name", "")
    renderer = cfg.get("rendering", {}).get("renderer") if isinstance(cfg, dict) else None
    yaml_headless = bool(_field(renderer, "headless", False))
    width = _field(renderer, "width", None)
    height = _field(renderer, "height", None)
    sdg = mode == "SDG"

    launch: dict = {}
    adjustments: list[str] = []
    extra_args: list[str] = []

    experience = resolve_experience(env)
    if experience:
        adjustments.append(f"experience {experience}")
        if experience.endswith("isaacsim.exp.full.kit"):
            adjustments.append("OMNILRS_EXPERIENCE points at the full kit, which uses more RAM than base.python")
    else:
        adjustments.append("experience empty; SimulationApp keeps isaacsim.exp.base.python.kit")

    gpus = record.get("gpu_memory_mib")
    single_gpu = isinstance(gpus, list) and len(gpus) == 1
    if single_gpu or (host_class == "low" and not preserve and not sdg):
        launch["max_gpu_count"] = 1
        adjustments.append("max_gpu_count 1")

    if not preserve and (host_class == "low" or profile == "low"):
        launch["limit_cpu_threads"] = 4 if host_class == "low" else 8
        adjustments.append(f"limit_cpu_threads {launch['limit_cpu_threads']}")
    elif not preserve and (host_class == "balanced" or profile == "balanced"):
        launch["limit_cpu_threads"] = 8
        adjustments.append("limit_cpu_threads 8")

    keep_window = _enabled(env, "OMNILRS_KEEP_WINDOW")
    forced_headless = False
    if host_class == "low" and not preserve and not sdg and not keep_window and not yaml_headless:
        forced_headless = True
        launch["headless"] = True
        _set_headless(renderer, True)
        adjustments.append(
            "installed RAM is under 12 GiB: headless=True. OMNILRS_KEEP_WINDOW=1 keeps the interactive window"
        )

    effective_headless = yaml_headless or forced_headless
    disable_viewport = False
    if not sdg and effective_headless and (
        (host_class == "low" and not preserve) or _enabled(env, "OMNILRS_DISABLE_VIEWPORT")
    ):
        disable_viewport = True
        launch["hide_ui"] = True
        launch["disable_viewport_updates"] = True
        adjustments.append(
            "disable_viewport_updates=True while headless. Camera and RTX sensor render products stay enabled"
        )

    if (
        not sdg
        and effective_headless
        and _enabled(env, "OMNILRS_DISABLE_UI_EXTENSIONS")
    ):
        for name in UI_EXTENSIONS:
            extra_args.extend(["--disable", name])
        adjustments.append("UI extensions listed in OMNILRS_DISABLE_UI_EXTENSIONS are passed to --disable")

    if extra_args:
        launch["extra_args"] = extra_args

    if not effective_headless and width and height:
        launch["window_width"] = int(width)
        launch["window_height"] = int(height)
        adjustments.append(f"window {int(width)}x{int(height)} matches the renderer resolution")

    if sdg or preserve or host_class != "low":
        rendered = DEFAULT_WARMUP_RENDERED
        physics = 0
    else:
        rendered = LOW_WARMUP_RENDERED
        physics = LOW_WARMUP_PHYSICS
        adjustments.append(
            "startup warmup is 98 physics steps and 2 rendered steps. "
            "The run loop still calls world.step(render=True) every frame"
        )

    rendered_override = _optional_int(env, "OMNILRS_WARMUP_RENDERED")
    physics_override = _optional_int(env, "OMNILRS_WARMUP_PHYSICS")
    if rendered_override is not None:
        rendered = rendered_override
        adjustments.append(f"OMNILRS_WARMUP_RENDERED {rendered}")
    if physics_override is not None:
        physics = physics_override
        adjustments.append(f"OMNILRS_WARMUP_PHYSICS {physics}")

    plan = {
        "experience": experience,
        "launch_updates": launch,
        "adjustments": adjustments,
        "warmup_rendered_frames": int(rendered),
        "warmup_physics_frames": int(physics),
        "skip_replicator_on_close": not sdg,
        "installed_class": host_class,
        "profile": profile,
        "forced_headless": forced_headless,
        "disable_viewport_updates": disable_viewport,
    }
    _guard(launch)

    message = "Isaac Sim launch (%s, installed class %s): %s" % (
        profile,
        host_class,
        "; ".join(adjustments),
    )
    if forced_headless or disable_viewport or physics:
        logger.warning(message)
    else:
        logger.info(message)
    return plan
