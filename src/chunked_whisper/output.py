"""Transcription output formatting and persistence."""

from __future__ import annotations

from json import dumps
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Literal

from .models import SegmentResult, TranscriptionResult

OutputFormat = Literal["all", "txt", "srt", "json"]


def _speaker_prefix(segment: SegmentResult) -> str:
    """Return a speaker label prefix suitable for rendered output.

    Args:
        segment: Segment whose speaker label is rendered.

    Returns:
        A bracketed speaker prefix such as ``[SPEAKER_00] `` when present.
    """
    return f"[{segment.speaker}] " if segment.speaker else ""


def render_txt(segments: Sequence[SegmentResult]) -> str:
    """Render plain text, adding one line per diarized speaker turn.

    Args:
        segments: Sequence of transcript segments to render.

    Returns:
        Plain-text output with either one line per segment or a single text
        block when no speaker labels are present.
    """
    if any(segment.speaker for segment in segments):
        lines = [
            f"{_speaker_prefix(segment)}{segment.text}"
            for segment in segments
        ]

        return "\n".join(lines) + ("\n" if lines else "")

    text = " ".join(segment.text.strip() for segment in segments).strip()

    return text + ("\n" if text else "")


def format_timestamp(seconds: float, decimal_marker: str = ",") -> str:
    """Format seconds as an SRT-compatible timestamp.

    Args:
        seconds: Time value in seconds.
        decimal_marker: Separator inserted before the millisecond field.

    Returns:
        A timestamp string formatted as ``HH:MM:SS,mmm`` by default.
    """
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)

    return (
        f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}"
        f"{decimal_marker}{milliseconds:03d}"
    )


def render_srt(segments: Sequence[SegmentResult]) -> str:
    """Render transcription segments as SubRip text.

    Args:
        segments: Sequence of segments to convert to SRT blocks.

    Returns:
        A complete SRT document with one block per segment.
    """
    blocks = [
        (
            f"{index}\n{format_timestamp(segment.start)} --> "
            f"{format_timestamp(segment.end)}\n"
            f"{_speaker_prefix(segment)}{segment.text}"
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
    """Write the requested transcription formats and return their paths.

    Args:
        result: The transcription result to serialize.
        audio_path: Original input audio path used to derive output names.
        output_dir: Directory in which output files should be written.
        output_format: Which output format(s) to emit.

    Returns:
        A list of paths for each file that was written.

    Raises:
        ValueError: If an output path would overwrite the input file.
    """
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
            content = render_txt(result.segments)
        elif selected_format == "srt":
            content = render_srt(result.segments)
        else:
            payload = asdict(result)
            payload["text"] = result.text
            content = dumps(payload, ensure_ascii=False, indent=2) + "\n"

        output_path.write_text(content, encoding="utf-8")
        written.append(output_path)

    return written
