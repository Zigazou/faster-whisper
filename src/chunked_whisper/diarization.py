"""Speaker diarization and transcription alignment."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from faster_whisper.audio import decode_audio

from .models import SegmentResult

DEFAULT_DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"
DIARIZATION_SAMPLE_RATE = 16_000
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpeakerTurn:
    """A time interval attributed to one anonymous speaker."""

    start: float
    end: float
    speaker: str


def load_diarization_pipeline(
    model_name: str = DEFAULT_DIARIZATION_MODEL,
    token: str | None = None,
    device: str = "cpu",
) -> Any:
    """Load a pyannote speaker diarization pipeline on the requested device."""
    try:
        import torch
        from pyannote.audio import Pipeline
    except ImportError as error:
        raise RuntimeError(
            "speaker diarization requires the 'diarization' extra; install it "
            "with: python -m pip install -e '.[diarization]'"
        ) from error

    pipeline = Pipeline.from_pretrained(model_name, token=token)
    if pipeline is None:
        raise RuntimeError(
            f"could not load diarization model {model_name!r}; verify model "
            "access and your Hugging Face token"
        )
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline.to(torch.device(device))
    return pipeline


def diarize_audio(
    pipeline: Any,
    audio_path: Path,
    *,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> list[SpeakerTurn]:
    """Run pyannote and return chronologically ordered anonymous turns."""
    import torch

    options = {
        name: value
        for name, value in (
            ("num_speakers", num_speakers),
            ("min_speakers", min_speakers),
            ("max_speakers", max_speakers),
        )
        if value is not None
    }
    LOGGER.info("Diarizing %s", audio_path)
    # Passing a compressed file path makes pyannote seek and decode each crop
    # independently. Some MP3 files cannot be sought with sample-level accuracy,
    # which can make a crop shorter than pyannote expects. Decode the complete
    # stream once so every crop operates on the same sample-exact waveform.
    audio = decode_audio(str(audio_path), sampling_rate=DIARIZATION_SAMPLE_RATE)
    if len(audio) == 0:
        raise ValueError("the audio stream is empty")
    waveform = torch.from_numpy(audio).unsqueeze(0)
    output = pipeline(
        {
            "waveform": waveform,
            "sample_rate": DIARIZATION_SAMPLE_RATE,
            "uri": audio_path.stem,
        },
        **options,
    )
    annotation = getattr(output, "exclusive_speaker_diarization", output)

    raw_turns = [
        (float(turn.start), float(turn.end), str(speaker))
        for turn, _, speaker in annotation.itertracks(yield_label=True)
    ]
    raw_turns.sort(key=lambda item: (item[0], item[1]))

    # Never expose model-specific identities: labels only distinguish speakers
    # within this file and are assigned in order of first appearance.
    labels: dict[str, str] = {}
    turns: list[SpeakerTurn] = []
    for start, end, original_label in raw_turns:
        label = labels.setdefault(original_label, f"SPEAKER_{len(labels):02d}")
        turns.append(SpeakerTurn(start=start, end=end, speaker=label))
    return turns


def _speaker_for_interval(
    start: float, end: float, turns: list[SpeakerTurn]
) -> str | None:
    """Return the speaker with the greatest overlap with an interval."""
    best_speaker: str | None = None
    best_overlap = 0.0
    for turn in turns:
        if turn.start >= end:
            break
        if turn.end <= start:
            continue
        overlap = min(end, turn.end) - max(start, turn.start)
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = turn.speaker
    return best_speaker


def assign_speakers(
    segments: list[SegmentResult], turns: list[SpeakerTurn]
) -> list[SegmentResult]:
    """Assign speakers to transcript words and split at speaker changes."""
    aligned: list[SegmentResult] = []
    for segment in segments:
        if not segment.words:
            aligned.append(
                SegmentResult(
                    start=segment.start,
                    end=segment.end,
                    text=segment.text,
                    words=[],
                    speaker=_speaker_for_interval(segment.start, segment.end, turns),
                )
            )
            continue

        group_words = []
        group_speaker: str | None = None
        for word in segment.words:
            speaker = _speaker_for_interval(word.start, word.end, turns)
            if group_words and speaker != group_speaker:
                aligned.append(
                    SegmentResult(
                        start=group_words[0].start,
                        end=group_words[-1].end,
                        text="".join(item.word for item in group_words).strip(),
                        words=group_words,
                        speaker=group_speaker,
                    )
                )
                group_words = []
            if not group_words:
                group_speaker = speaker
            group_words.append(word)

        if group_words:
            aligned.append(
                SegmentResult(
                    start=group_words[0].start,
                    end=group_words[-1].end,
                    text="".join(item.word for item in group_words).strip(),
                    words=group_words,
                    speaker=group_speaker,
                )
            )
    return [segment for segment in aligned if segment.text]
