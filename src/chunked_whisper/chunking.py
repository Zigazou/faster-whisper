"""Utilities for dividing audio into overlapping windows."""

from __future__ import annotations

from collections.abc import Iterator


def chunk_windows(
    total_samples: int,
    chunk_samples: int,
    overlap_samples: int,
) -> Iterator[tuple[int, int, int, int]]:
    """Yield overlapping audio windows with their core region offsets.

    Args:
        total_samples: Total number of samples in the source audio stream.
        chunk_samples: Number of samples in each non-overlapping core window.
        overlap_samples: Number of samples to extend each window on both sides.

    Yields:
        A tuple of ``(window_start, window_end, core_start, core_end)`` sample
        offsets for each chunk.

    Raises:
        ValueError: If ``chunk_samples`` is not positive.
        ValueError: If ``overlap_samples`` is negative.
    """
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

