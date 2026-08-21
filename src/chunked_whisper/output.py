"""Transcription output formatting and persistence."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Literal

from .models import SegmentResult, TranscriptionResult

OutputFormat = Literal["all", "txt", "srt", "json"]


def format_timestamp(seconds: float, decimal_marker: str = ",") -> str:
    """Format seconds as an SRT-compatible timestamp."""
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return (
        f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}"
        f"{decimal_marker}{milliseconds:03d}"
    )


def render_srt(segments: Sequence[SegmentResult]) -> str:
    """Render transcription segments as SubRip text."""
    blocks = [
        (
            f"{index}\n{format_timestamp(segment.start)} --> "
            f"{format_timestamp(segment.end)}\n{segment.text}"
        )
        for index, segment in enumerate(segments, start=1)
    ]
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def write_outputs(
    result: TranscriptionResult,
    audio_path: Path,
    output_dir: Path | None,
    output_format: OutputFormat,
) -> list[Path]:
    """Write the requested transcription formats and return their paths."""
    directory = output_dir if output_dir is not None else audio_path.parent
    directory.mkdir(parents=True, exist_ok=True)
    base = directory / audio_path.stem
    selected_formats = (
        ("txt", "srt", "json") if output_format == "all" else (output_format,)
    )
    written: list[Path] = []

    for selected_format in selected_formats:
        output_path = base.with_suffix(f".{selected_format}")
        if output_path.resolve() == audio_path.resolve():
            raise ValueError(
                f"refusing to overwrite the input file with output: {output_path}"
            )
        if selected_format == "txt":
            content = result.text + ("\n" if result.text else "")
        elif selected_format == "srt":
            content = render_srt(result.segments)
        else:
            payload = asdict(result)
            payload["text"] = result.text
            content = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        output_path.write_text(content, encoding="utf-8")
        written.append(output_path)

    return written

