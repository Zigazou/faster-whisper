"""Utilities for dividing audio into overlapping windows."""

from __future__ import annotations

from collections.abc import Iterator


def chunk_windows(
    total_samples: int,
    chunk_samples: int,
    overlap_samples: int,
) -> Iterator[tuple[int, int, int, int]]:
    """Yield ``(window_start, window_end, core_start, core_end)`` offsets."""
    if chunk_samples <= 0:
        raise ValueError("chunk_samples must be positive")
    if overlap_samples < 0:
        raise ValueError("overlap_samples cannot be negative")

    core_start = 0
    while core_start < total_samples:
        core_end = min(core_start + chunk_samples, total_samples)
        yield (
            max(0, core_start - overlap_samples),
            min(total_samples, core_end + overlap_samples),
            core_start,
            core_end,
        )
        core_start = core_end

