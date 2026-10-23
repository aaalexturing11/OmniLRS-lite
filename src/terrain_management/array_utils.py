"""Small array helpers shared by the terrain loaders."""

from __future__ import annotations

import numpy as np


def close_npy(array: np.ndarray) -> None:
    """Close the memmap behind an array loaded with ``mmap_mode``."""

    mapped = getattr(array, "_mmap", None)
    if mapped is not None:
        mapped.close()


def as_float32_block(array: np.ndarray) -> np.ndarray:
    """Return a C-contiguous float32 copy of just this array, not its base.

    Queueing a view can keep the parent allocation alive until the consumer
    finishes. Callers that hand blocks to worker processes need the compact form.
    """

    if array.dtype == np.float32 and array.flags.c_contiguous:
        return array
    return np.ascontiguousarray(array, dtype=np.float32)


def load_transposed_elevation(path: str) -> np.ndarray:
    """Load a DEM as MapManager does: transpose, then flip the last axis.

    The file is read in column chunks so a float64 source is not resident at
    the same time as the float32 result. Elevations in meters survive the
    narrowing: float32 resolves about a millimeter at a kilometer of relief,
    and the clipmap kernels already consume float32.
    """

    source = np.load(path, mmap_mode="r")
    try:
        height, width = source.shape
        out = np.empty((width, height), dtype=np.float32)
        chunk = 64
        for start in range(0, width, chunk):
            stop = min(start + chunk, width)
            block = np.asarray(source[::-1, start:stop])
            out[start:stop, :] = block.T
        return out
    finally:
        close_npy(source)
