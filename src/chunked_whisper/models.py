"""Data structures that describe recognized speech and transcription outputs."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class WordResult:
    """A single recognized word with timing and confidence metadata.

    Attributes:
        start: Word start time in seconds.
        end: Word end time in seconds.
        word: The recognized token text.
        probability: Confidence score returned by the transcription model.
    """

    start: float
    end: float
    word: str
    probability: float


@dataclass
class SegmentResult:
    """A contiguous transcript segment optionally assigned to a speaker.

    Attributes:
        start: Segment start time in seconds.
        end: Segment end time in seconds.
        text: Text content for the segment.
        words: Individual word results that make up the segment.
        speaker: Anonymous speaker label when diarization is available.
    """

    start: float
    end: float
    text: str
    words: list[WordResult] = field(default_factory=list)
    speaker: str | None = None


@dataclass
class TranscriptionResult:
    """The complete transcription result for one audio source.

    Attributes:
        source: Original source audio path.
        language: Detected or configured language code.
        language_probability: Confidence associated with the language detection.
        duration: Audio duration in seconds.
        model: Model identifier used for transcription.
        segments: Sequence of transcribed segments.
    """

    source: str
    language: str
    language_probability: float
    duration: float
    model: str
    segments: list[SegmentResult]

    @property
    def text(self) -> str:
        """Return the transcript as a single normalized string.

        Returns:
            A single-space concatenation of all segment texts with surrounding
            whitespace removed.
        """
        return " ".join(
            segment.text.strip()
            for segment in self.segments
        ).strip()
