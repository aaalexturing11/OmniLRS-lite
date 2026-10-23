"""Host-memory profile for OmniLRS.

The profile scales caches, worker processes and renderer sample counts.
It does not remove sensors, physics, deformation, rocks, craters or clipmaps.
``OMNILRS_MEMORY_PROFILE=off`` keeps every value from the yaml.
``OMNILRS_MEMORY_PROFILE=high`` does the same, except it disables multi-GPU
when the machine has a single GPU.
"""

from __future__ import annotations

import ctypes
import gc
import logging
import os
import subprocess
from typing import Optional

logger = logging.getLogger("omnilrs.memory")

_GIB = 1024**3
_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
)


def host_memory() -> dict:
    """Return total and available physical RAM in bytes.

    Available memory is what the operating system can give a new allocation
    without paging. It is not the working set of this process.
    """

    if os.name == "nt":
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError("GlobalMemoryStatusEx failed")
        return {"total": int(status.ullTotalPhys), "available": int(status.ullAvailPhys)}

    total = available = None
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            info = {}
            for line in handle:
                key, _, rest = line.partition(":")
                info[key] = int(rest.strip().split()[0]) * 1024
        total = info.get("MemTotal")
        available = info.get("MemAvailable", info.get("MemFree"))
    except OSError:
        page = os.sysconf("SC_PAGE_SIZE")
        total = int(os.sysconf("SC_PHYS_PAGES")) * page
        available = int(os.sysconf("SC_AVPHYS_PAGES")) * page
    return {"total": int(total), "available": int(available)}


def process_memory() -> dict:
    """Return this process working set and commit charge, in bytes.

    The working set is physical RAM currently charged to the process.
    The commit charge includes pages that may be sitting in the page file.
    Neither number is the memory used by the whole machine.
    """

    if os.name == "nt":
        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.K32GetProcessMemoryInfo.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
            ctypes.c_ulong,
        ]
        kernel.K32GetProcessMemoryInfo.restype = ctypes.c_int
        if not kernel.K32GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return {}
        return {
            "working_set": int(counters.WorkingSetSize),
            "commit": int(counters.PagefileUsage),
        }

    try:
        with open("/proc/self/status", "r", encoding="utf-8") as handle:
            info = {}
            for line in handle:
                key, _, rest = line.partition(":")
                parts = rest.strip().split()
                if parts and parts[0].isdigit():
                    info[key] = int(parts[0]) * 1024
        return {"working_set": info.get("VmRSS"), "commit": info.get("VmSize")}
    except OSError:
        return {}


def gpu_memory_mib() -> Optional[list]:
    """Total VRAM per GPU, in MiB. None when nvidia-smi is unavailable."""

    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            text=True,
            timeout=3,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    values = []
    for line in output.splitlines():
        token = line.strip().split()[0] if line.strip() else ""
        if token.isdigit():
            values.append(int(token))
    return values or None


def classify_profile(total_bytes: int, available_bytes: int, requested: str) -> str:
    """Pick low, balanced or high from installed RAM, then step down if little is free."""

    if requested in ("off", "low", "balanced", "high"):
        return requested
    if total_bytes < 12 * _GIB:
        profile = "low"
    elif total_bytes < 24 * _GIB:
        profile = "balanced"
    else:
        profile = "high"
    if available_bytes < 2 * _GIB:
        return "low"
    if available_bytes < 4 * _GIB and profile == "high":
        return "balanced"
    return profile


def estimate_hr_dem_bytes(num_blocks: int, block_size: float, resolution: float) -> int:
    """Bytes of the resident float32 high-resolution window, padding block included."""

    side = int((int(num_blocks) * 2 + 3) * float(block_size) / float(resolution))
    return side * side * 4


def apply_early_thread_limits() -> None:
    """Limit BLAS and Numba thread pools before NumPy is imported.

    Each worker process inherits these variables. On a memory-constrained host,
    the default of one pool per core multiplies stack memory by the crater workers.
    Existing values in the environment are left untouched.
    """

    requested = os.environ.get("OMNILRS_MEMORY_PROFILE", "auto")
    if requested in ("off", "high"):
        return
    memory = host_memory()
    profile = classify_profile(memory["total"], memory["available"], requested)
    if profile in ("off", "high"):
        return
    threads = "2" if profile == "low" else "4"
    for name in _THREAD_VARS:
        os.environ.setdefault(name, threads)


def _cap_int(value, cap: int, minimum: int) -> int:
    return max(minimum, min(int(value), cap))


def _max_block_span(terrain: dict) -> int:
    spans = []
    for rock_cfg in terrain.get("rock_gen_cfgs") or []:
        if isinstance(rock_cfg, dict) and rock_cfg.get("block_span") is not None:
            spans.append(int(rock_cfg["block_span"]))
    return max(spans) if spans else 1


def apply_memory_budget(cfg: dict, memory: Optional[dict] = None, gpus: Optional[list] = None) -> dict:
    """Mutate a resolved config dict and record what changed.

    Call this after OmegaConf resolution and before the dataclass factory.
    The record is stored at ``cfg["_memory"]`` for the renderer startup path.
    """

    memory = memory if memory is not None else host_memory()
    if gpus is None:
        gpus = gpu_memory_mib()
    requested = os.environ.get("OMNILRS_MEMORY_PROFILE", "auto")
    profile = classify_profile(memory["total"], memory["available"], requested)
    adjustments = []

    if profile != "off":
        _apply_renderer_budget(cfg, profile, adjustments)
        _apply_gpu_budget(cfg, profile, gpus, adjustments)
        _apply_terrain_budget(cfg, profile, gpus, adjustments)
        _apply_physics_budget(cfg, profile, adjustments)

    record = {
        "name": profile,
        "requested": requested,
        "total_bytes": memory["total"],
        "available_bytes": memory["available"],
        "gpu_memory_mib": gpus,
        "adjustments": adjustments,
    }
    cfg["_memory"] = record
    _log_record(record)
    return record


def _apply_renderer_budget(cfg: dict, profile: str, adjustments: list) -> None:
    rendering = cfg.get("rendering")
    if not isinstance(rendering, dict):
        return
    renderer = rendering.get("renderer")
    if not isinstance(renderer, dict) or profile == "high":
        return

    kind = renderer.get("renderer", "RayTracedLighting")
    if profile == "low" and kind == "RayTracedLighting":
        caps = {
            "samples_per_pixel_per_frame": 1,
            "max_bounces": 3,
            "max_specular_transmission_bounces": 2,
            "max_volume_bounces": 1,
        }
    elif profile == "low":
        caps = {
            "samples_per_pixel_per_frame": 8,
            "max_bounces": 4,
            "max_specular_transmission_bounces": 3,
            "max_volume_bounces": 2,
        }
    elif kind == "RayTracedLighting":
        caps = {
            "samples_per_pixel_per_frame": 8,
            "max_bounces": 4,
            "max_specular_transmission_bounces": 4,
            "max_volume_bounces": 2,
        }
    else:
        caps = {
            "samples_per_pixel_per_frame": 16,
            "max_bounces": 4,
            "max_specular_transmission_bounces": 4,
            "max_volume_bounces": 2,
        }

    for key, cap in caps.items():
        if key not in renderer:
            continue
        updated = _cap_int(renderer[key], cap, 1)
        if updated != int(renderer[key]):
            adjustments.append(f"rendering.renderer.{key} {renderer[key]} -> {updated}")
            renderer[key] = updated

    if profile == "low":
        if renderer.get("width") is None:
            renderer["width"] = 1280
            adjustments.append("rendering.renderer.width unset -> 1280")
        if renderer.get("height") is None:
            renderer["height"] = 720
            adjustments.append("rendering.renderer.height unset -> 720")


def _apply_gpu_budget(cfg: dict, profile: str, gpus: Optional[list], adjustments: list) -> None:
    rendering = cfg.get("rendering")
    if not isinstance(rendering, dict):
        return
    renderer = rendering.get("renderer")
    if not isinstance(renderer, dict) or "multi_gpu" not in renderer:
        return
    single_gpu = gpus is not None and len(gpus) < 2
    if renderer["multi_gpu"] and (profile == "low" or single_gpu):
        renderer["multi_gpu"] = False
        reason = "single GPU" if single_gpu else "low-memory profile"
        adjustments.append(f"rendering.renderer.multi_gpu True -> False ({reason})")


def _apply_terrain_budget(cfg: dict, profile: str, gpus: Optional[list], adjustments: list) -> None:
    environment = cfg.get("environment")
    if not isinstance(environment, dict):
        return
    terrain = environment.get("large_scale_terrain")
    if not isinstance(terrain, dict):
        return

    if profile == "low" and "hr_dem_num_blocks" in terrain:
        floor = _max_block_span(terrain)
        updated = max(floor, min(int(terrain["hr_dem_num_blocks"]), 2))
        if updated != int(terrain["hr_dem_num_blocks"]):
            before = estimate_hr_dem_bytes(
                terrain["hr_dem_num_blocks"], terrain.get("block_size", 50), terrain.get("hr_dem_resolution", 0.025)
            )
            after = estimate_hr_dem_bytes(
                updated, terrain.get("block_size", 50), terrain.get("hr_dem_resolution", 0.025)
            )
            adjustments.append(
                f"large_scale_terrain.hr_dem_num_blocks {terrain['hr_dem_num_blocks']} -> {updated} "
                f"(resident window {before / _GIB:.2f} GiB -> {after / _GIB:.2f} GiB, same meters-per-pixel)"
            )
            terrain["hr_dem_num_blocks"] = updated

    worker_cap = 2 if profile == "low" else 4 if profile == "balanced" else None
    if worker_cap is not None and "num_workers_craters" in terrain:
        updated = _cap_int(terrain["num_workers_craters"], worker_cap, 1)
        if updated != int(terrain["num_workers_craters"]):
            adjustments.append(f"large_scale_terrain.num_workers_craters {terrain['num_workers_craters']} -> {updated}")
            terrain["num_workers_craters"] = updated
    if profile == "low" and "num_workers_interpolation" in terrain:
        updated = _cap_int(terrain["num_workers_interpolation"], 1, 1)
        if updated != int(terrain["num_workers_interpolation"]):
            adjustments.append(
                f"large_scale_terrain.num_workers_interpolation {terrain['num_workers_interpolation']} -> {updated}"
            )
            terrain["num_workers_interpolation"] = updated

    queue_caps = {"input_queue_size": 32 if profile == "low" else 64, "output_queue_size": 4 if profile == "low" else 8}
    if profile in ("low", "balanced"):
        for key, cap in queue_caps.items():
            if key not in terrain:
                continue
            updated = _cap_int(terrain[key], cap, 1)
            if updated != int(terrain[key]):
                adjustments.append(f"large_scale_terrain.{key} {terrain[key]} -> {updated}")
                terrain[key] = updated

    low_vram = gpus is not None and min(gpus) < 12288
    if (profile == "low" or low_vram) and terrain.get("geo_cm_coarse_acceleration_mode") == "gpu":
        terrain["geo_cm_coarse_acceleration_mode"] = "hybrid"
        adjustments.append(
            "large_scale_terrain.geo_cm_coarse_acceleration_mode gpu -> hybrid "
            "(the low-resolution DEM stays in system RAM and is not copied to VRAM)"
        )


def _apply_physics_budget(cfg: dict, profile: str, adjustments: list) -> None:
    physics = cfg.get("physics")
    if not isinstance(physics, dict) or profile == "high":
        return
    scene = physics.get("physics_scene")
    if not isinstance(scene, dict):
        return
    cap = 2 if profile == "low" else 4
    current = scene.get("worker_thread_count")
    updated = cap if current is None else _cap_int(current, cap, 1)
    if current != updated:
        scene["worker_thread_count"] = updated
        adjustments.append(f"physics.physics_scene.worker_thread_count {current} -> {updated}")


def _log_record(record: dict) -> None:
    total = record["total_bytes"] / _GIB
    available = record["available_bytes"] / _GIB
    message = (
        f"OmniLRS memory profile '{record['name']}' "
        f"(requested {record['requested']}). "
        f"System RAM {total:.2f} GiB total, {available:.2f} GiB available. "
        f"GPU VRAM MiB: {record['gpu_memory_mib']}."
    )
    if record["adjustments"]:
        logger.warning("%s Adjustments: %s", message, "; ".join(record["adjustments"]))
    else:
        logger.info("%s No quality caps applied.", message)


def log_process_memory(stage: str) -> None:
    """Log process working set beside system RAM. Both numbers are measured."""

    system = host_memory()
    process = process_memory()
    working = process.get("working_set")
    commit = process.get("commit")
    logger.info(
        "Memory at %s: system total %.2f GiB, system available %.2f GiB, "
        "process working set %s, process commit %s.",
        stage,
        system["total"] / _GIB,
        system["available"] / _GIB,
        "n/a" if working is None else f"{working / _GIB:.2f} GiB",
        "n/a" if commit is None else f"{commit / _GIB:.2f} GiB",
    )


def release_unused_memory() -> None:
    """Collect cyclic garbage left by scene construction.

    This does not trim the Windows working set. Forcing that trim only moves
    live pages into the page file and makes the next frame fault them back.
    """

    gc.collect()
    gc.collect()
