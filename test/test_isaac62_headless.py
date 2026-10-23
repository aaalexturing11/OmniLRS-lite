"""Headless Lunalab on the machine where Isaac Sim 6.2 already runs.

This file does nothing unless OMNILRS_RUN_ISAAC=1. A skipped result is not a
passing Isaac run: cuDeviceGetUuid and the segmentation fault only show up
when Kit starts.

On the host that already runs Isaac Sim 6.2 headless:

    export OMNILRS_RUN_ISAAC=1
    export ISAAC_PY=/isaac-sim/python.sh
    "$ISAAC_PY" -m pytest test/test_isaac62_headless.py -q
"""

import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_FATAL = (
    "cuDeviceGetUuid",
    "CUDA error 36",
    "Segmentation fault",
    "ModuleNotFoundError",
    "Error executing job",
)
_STARTED = (
    "ArticulationTelemetry initialized",
    "app ready",
    "mesh update took",
)


@pytest.mark.skipif(os.environ.get("OMNILRS_RUN_ISAAC") != "1", reason="Isaac Sim is not launched from this checkout")
def test_headless_lunalab_starts_without_warp_cuda_fault():
    python = os.environ.get("ISAAC_PY", sys.executable)
    env = os.environ.copy()
    env["OMNILRS_MEMORY_PROFILE"] = env.get("OMNILRS_MEMORY_PROFILE", "auto")
    env.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    env.pop("ROS_DISTRO", None)
    timeout = int(os.environ.get("OMNILRS_ISAAC_TIMEOUT", "900"))
    process = subprocess.Popen(
        [
            python,
            "run.py",
            "environment=lunalab",
            "mode=ROS2",
            "rendering.renderer.headless=true",
        ],
        cwd=_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    lines: queue.Queue[str | None] = queue.Queue()

    def _read() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            lines.put(line)
        lines.put(None)

    threading.Thread(target=_read, daemon=True).start()
    collected: list[str] = []
    started = False
    deadline = time.monotonic() + timeout
    try:
        while time.monotonic() < deadline:
            try:
                line = lines.get(timeout=1)
            except queue.Empty:
                if process.poll() is not None:
                    break
                continue
            if line is None:
                break
            collected.append(line)
            text = "".join(collected)
            for marker in _FATAL:
                assert marker not in text
            if any(marker in text for marker in _STARTED):
                started = True
                break
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=30)
    assert started, "".join(collected[-80:])
