"""In-place shift of the resident high-resolution DEM.

``np.roll`` would allocate a second copy of the window. With the default
settings that window is (4 * 2 + 3) * 50 / 0.025 = 22000 pixels on a side,
1.80 GiB of float32. Shifting by rows reuses the same buffer.
"""

from __future__ import annotations

import numpy as np


def shift_heightfield(dem: np.ndarray, x_shift: int, y_shift: int) -> None:
    """Move ``dem`` contents by ``(x_shift, y_shift)`` pixels and clear the vacated edge.

    Positive shifts move data toward larger indices. A shift that empties the
    array fills it with zeros. The array is modified in place.
    """

    height, width = dem.shape
    if x_shift == 0 and y_shift == 0:
        return
    if abs(x_shift) >= height or abs(y_shift) >= width:
        dem.fill(0)
        return

    if x_shift > 0:
        for row in range(height - 1, x_shift - 1, -1):
            dem[row, :] = dem[row - x_shift, :]
        dem[:x_shift, :] = 0
    elif x_shift < 0:
        step = -x_shift
        for row in range(height - step):
            dem[row, :] = dem[row + step, :]
        dem[height - step :, :] = 0

    if y_shift > 0:
        row_buffer = np.empty(width, dtype=dem.dtype)
        for row in range(height):
            row_buffer[:] = dem[row]
            dem[row, y_shift:] = row_buffer[:-y_shift]
            dem[row, :y_shift] = 0
    elif y_shift < 0:
        step = -y_shift
        row_buffer = np.empty(width, dtype=dem.dtype)
        for row in range(height):
            row_buffer[:] = dem[row]
            dem[row, :-step] = row_buffer[step:]
            dem[row, -step:] = 0
