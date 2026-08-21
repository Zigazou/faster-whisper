"""Audio decoding and chunked Faster Whisper transcription."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio

from .chunking import chunk_windows
from .models import SegmentResult, TranscriptionResult, WordResult

SAMPLE_RATE = 16_000
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TranscriptionConfig:
    """Options that control model inference and audio chunking."""

    model_name: str = "large-v3"
    chunk_duration: float = 300.0
    overlap: float = 5.0
    language: str | None = None
    beam_size: int = 5
    vad_filter: bool = True
    word_timestamps: bool = False

    def validate(self) -> None:
        if self.chunk_duration <= 0:
            raise ValueError("chunk duration must be greater than zero")
        if self.overlap < 0:
            raise ValueError("overlap cannot be negative")
        if self.overlap >= self.chunk_duration:
            raise ValueError("overlap must be shorter than chunk duration")
        if self.beam_size < 1:
            raise ValueError("beam size must be at least 1")


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
    config: TranscriptionConfig,
) -> TranscriptionResult:
    """Decode and transcribe one audio file in overlapping chunks."""
    config.validate()
    LOGGER.info("Decoding %s", audio_path)
    audio = decode_audio(str(audio_path), sampling_rate=SAMPLE_RATE)
    total_samples = len(audio)
    if total_samples == 0:
        raise ValueError("the audio stream is empty")

    duration = total_samples / SAMPLE_RATE
    windows = list(
        chunk_windows(
            total_samples,
            max(1, round(config.chunk_duration * SAMPLE_RATE)),
            round(config.overlap * SAMPLE_RATE),
        )
    )
    resolved_language = config.language
    language_probability = 1.0 if config.language else 0.0
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
            beam_size=config.beam_size,
            vad_filter=config.vad_filter,
            word_timestamps=config.word_timestamps,
            condition_on_previous_text=True,
        )

        # Detect once, then keep the language stable across subsequent chunks.
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
        model=config.model_name,
        segments=results,
    )

