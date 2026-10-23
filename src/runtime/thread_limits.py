"""Apply BLAS/Numba thread caps as soon as this module is imported."""

from src.runtime.memory_budget import apply_early_thread_limits

apply_early_thread_limits()
