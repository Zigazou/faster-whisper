#!/usr/bin/env python3
"""Transcribe audio files in overlapping chunks with Faster Whisper."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio


SAMPLE_RATE = 16_000
LOGGER = logging.getLogger("transcribe")


@dataclass
class WordResult:
    start: float
    end: float
    word: str
    probability: float


@dataclass
class SegmentResult:
    start: float
    end: float
    text: str
    words: list[WordResult] = field(default_factory=list)


@dataclass
class TranscriptionResult:
    source: str
    language: str
    language_probability: float
    duration: float
    model: str
    segments: list[SegmentResult]

    @property
    def text(self) -> str:
        return " ".join(segment.text.strip() for segment in self.segments).strip()


def chunk_windows(
    total_samples: int,
    chunk_samples: int,
    overlap_samples: int,
) -> Iterable[tuple[int, int, int, int]]:
    """Yield (window_start, window_end, core_start, core_end) sample offsets."""
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


def _shift_segment(segment: Any, offset: float) -> SegmentResult:
    words = [
        WordResult(
            start=max(0.0, float(word.start) + offset),
            end=max(0.0, float(word.end) + offset),
            word=word.word,
            probability=float(word.probability),
        )
        for word in (segment.words or [])
    ]
    return SegmentResult(
        start=max(0.0, float(segment.start) + offset),
        end=max(0.0, float(segment.end) + offset),
        text=segment.text.strip(),
        words=words,
    )


def transcribe_audio(
    model: WhisperModel,
    audio_path: Path,
    model_name: str,
    chunk_duration: float,
    overlap: float,
    language: str | None,
    beam_size: int,
    vad_filter: bool,
    word_timestamps: bool,
) -> TranscriptionResult:
    LOGGER.info("Decoding %s", audio_path)
    audio = decode_audio(str(audio_path), sampling_rate=SAMPLE_RATE)
    total_samples = len(audio)
    duration = total_samples / SAMPLE_RATE
    if total_samples == 0:
        raise ValueError("the audio stream is empty")

    chunk_samples = max(1, round(chunk_duration * SAMPLE_RATE))
    overlap_samples = round(overlap * SAMPLE_RATE)
    windows = list(chunk_windows(total_samples, chunk_samples, overlap_samples))
    resolved_language = language
    language_probability = 1.0 if language else 0.0
    results: list[SegmentResult] = []

    for index, (window_start, window_end, core_start, core_end) in enumerate(
        windows, start=1
    ):
        LOGGER.info(
            "Transcribing chunk %d/%d (%.1f-%.1f seconds)",
            index,
            len(windows),
            window_start / SAMPLE_RATE,
            window_end / SAMPLE_RATE,
        )
        segments, info = model.transcribe(
            audio[window_start:window_end],
            language=resolved_language,
            beam_size=beam_size,
            vad_filter=vad_filter,
            word_timestamps=word_timestamps,
            condition_on_previous_text=True,
        )

        # Detect once, then keep the language stable across all subsequent chunks.
        if resolved_language is None:
            resolved_language = info.language
            language_probability = float(info.language_probability)

        window_offset = window_start / SAMPLE_RATE
        core_start_seconds = core_start / SAMPLE_RATE
        core_end_seconds = core_end / SAMPLE_RATE
        is_last = core_end == total_samples

        for segment in segments:
            shifted = _shift_segment(segment, window_offset)
            midpoint = (shifted.start + shifted.end) / 2
            belongs_to_core = midpoint >= core_start_seconds and (
                midpoint < core_end_seconds
                or (is_last and midpoint <= core_end_seconds)
            )
            if belongs_to_core and shifted.text:
                shifted.start = min(shifted.start, duration)
                shifted.end = min(shifted.end, duration)
                results.append(shifted)

    return TranscriptionResult(
        source=str(audio_path),
        language=resolved_language or "unknown",
        language_probability=language_probability,
        duration=duration,
        model=model_name,
        segments=results,
    )


def format_timestamp(seconds: float, decimal_marker: str = ",") -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return (
        f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}"
        f"{decimal_marker}{milliseconds:03d}"
    )


def render_srt(segments: Sequence[SegmentResult]) -> str:
    blocks = []
    for index, segment in enumerate(segments, start=1):
        blocks.append(
            f"{index}\n{format_timestamp(segment.start)} --> "
            f"{format_timestamp(segment.end)}\n{segment.text}"
        )
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def write_outputs(
    result: TranscriptionResult,
    audio_path: Path,
    output_dir: Path | None,
    output_format: str,
) -> list[Path]:
    directory = output_dir if output_dir is not None else audio_path.parent
    directory.mkdir(parents=True, exist_ok=True)
    base = directory / audio_path.stem
    selected_formats = ("txt", "srt", "json") if output_format == "all" else (
        output_format,
    )
    written = []

    for selected_format in selected_formats:
        output_path = base.with_suffix(f".{selected_format}")
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


def positive_float(value: str) -> float:
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def non_negative_float(value: str) -> float:
    number = float(value)
    if number < 0:
        raise argparse.ArgumentTypeError("cannot be negative")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transcribe audio files on a GPU using overlapping chunks."
    )
    parser.add_argument("audio", nargs="+", type=Path, help="audio file(s) to process")
    parser.add_argument(
        "--model",
        default="large-v3",
        help="model name or local model directory (default: large-v3)",
    )
    parser.add_argument(
        "--device", default="cuda", help="CTranslate2 device (default: cuda)"
    )
    parser.add_argument(
        "--compute-type",
        default="int8_float16",
        help="CTranslate2 compute type (default: int8_float16)",
    )
    parser.add_argument(
        "--chunk-duration",
        type=positive_float,
        default=300.0,
        metavar="SECONDS",
        help="non-overlapping core chunk duration (default: 300)",
    )
    parser.add_argument(
        "--overlap",
        type=non_negative_float,
        default=5.0,
        metavar="SECONDS",
        help="audio context added to each side of a chunk (default: 5)",
    )
    parser.add_argument(
        "--language",
        help="spoken language code, such as en or fr (default: detect once)",
    )
    parser.add_argument("--beam-size", type=int, default=5, help="beam size (default: 5)")
    parser.add_argument(
        "--no-vad", action="store_true", help="disable the Silero voice activity filter"
    )
    parser.add_argument(
        "--word-timestamps", action="store_true", help="include word timing in JSON"
    )
    parser.add_argument(
        "--output-dir", type=Path, help="output directory (default: beside each input)"
    )
    parser.add_argument(
        "--output-format",
        choices=("all", "txt", "srt", "json"),
        default="all",
        help="files to write (default: all)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    missing = [path for path in args.audio if not path.is_file()]
    if missing:
        for path in missing:
            LOGGER.error("Input file does not exist: %s", path)
        return 2
    if args.beam_size < 1:
        LOGGER.error("--beam-size must be at least 1")
        return 2

    LOGGER.info(
        "Loading model %s on %s using %s",
        args.model,
        args.device,
        args.compute_type,
    )
    try:
        model = WhisperModel(
            args.model,
            device=args.device,
            compute_type=args.compute_type,
        )
    except Exception as error:
        LOGGER.error("Could not load the model: %s", error)
        return 1

    failed = False
    for audio_path in args.audio:
        try:
            result = transcribe_audio(
                model=model,
                audio_path=audio_path,
                model_name=args.model,
                chunk_duration=args.chunk_duration,
                overlap=args.overlap,
                language=args.language,
                beam_size=args.beam_size,
                vad_filter=not args.no_vad,
                word_timestamps=args.word_timestamps,
            )
            paths = write_outputs(
                result, audio_path, args.output_dir, args.output_format
            )
            LOGGER.info(
                "Finished %s (%s, %.1f%% confidence): %s",
                audio_path,
                result.language,
                result.language_probability * 100,
                ", ".join(str(path) for path in paths),
            )
        except Exception as error:
            failed = True
            LOGGER.exception("Could not transcribe %s: %s", audio_path, error)

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
