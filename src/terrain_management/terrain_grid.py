"""Vectorized terrain mesh topology.

The previous builder appended one Python tuple per vertex and one Python int
per index. On a 40 m yard at 2 cm that is four million tuples and about
24 million boxed integers, which peaks well above a gigabyte before NumPy
ever sees the mesh. The arrays below are the same topology.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


class TerrainGrid(NamedTuple):
    vertices: np.ndarray
    indices: np.ndarray
    uvs: np.ndarray
    face_vertex_counts: np.ndarray


def build_terrain_grid(sim_length: int, sim_width: int, grid_size: float) -> TerrainGrid:
    """Build the yard mesh used by ``TerrainManager.buildGrid``.

    Vertex (x, y) is stored at index ``y * sim_width + x``. Each interior cell
    emits two triangles, in the same corner order as the original nested loops.
    UVs are those corner coordinates in meters, face-varying.
    """

    # Multiply in float64, then narrow once. That matches the old builder, which
    # stored Python floats and cast the finished list to float32.
    x = (np.arange(sim_width, dtype=np.float64) * float(grid_size)).astype(np.float32)
    y = (np.arange(sim_length, dtype=np.float64) * float(grid_size)).astype(np.float32)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    vertices = np.stack((xx, yy, np.zeros_like(xx)), axis=-1).reshape(-1, 3)

    if sim_length < 2 or sim_width < 2:
        empty_i = np.empty((0,), dtype=np.int32)
        empty_uv = np.empty((0, 2), dtype=np.float32)
        return TerrainGrid(vertices, empty_i, empty_uv, np.empty((0,), dtype=np.int32))

    ys = np.arange(1, sim_length, dtype=np.int32)
    xs = np.arange(1, sim_width, dtype=np.int32)
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
    grid_y = grid_y.reshape(-1)
    grid_x = grid_x.reshape(-1)

    def index(x_coord: np.ndarray, y_coord: np.ndarray) -> np.ndarray:
        return y_coord * sim_width + x_coord

    corners_x = (grid_x, grid_x, grid_x - 1, grid_x, grid_x - 1, grid_x - 1)
    corners_y = (grid_y - 1, grid_y, grid_y - 1, grid_y, grid_y, grid_y - 1)
    indices = np.stack([index(cx, cy) for cx, cy in zip(corners_x, corners_y)], axis=1).reshape(-1)
    indices = np.ascontiguousarray(indices, dtype=np.int32)

    uv_corners = np.stack(
        [np.stack((cx, cy), axis=1) for cx, cy in zip(corners_x, corners_y)],
        axis=1,
    ).reshape(-1, 2)
    # The old code cast the integer corners to float32 and then multiplied by a
    # Python float, which promoted the product to float64. USD stores float2,
    # so the resident copy is that product narrowed back to float32.
    uvs = np.ascontiguousarray(uv_corners, dtype=np.float32) * float(grid_size)
    uvs = np.ascontiguousarray(uvs, dtype=np.float32)
    face_vertex_counts = np.full(indices.shape[0] // 3, 3, dtype=np.int32)
    return TerrainGrid(vertices, indices, uvs, face_vertex_counts)
