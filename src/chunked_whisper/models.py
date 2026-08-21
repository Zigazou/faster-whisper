"""Application result models."""

from __future__ import annotations

from dataclasses import dataclass, field


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
        """Return all segment text as a single normalized string."""
        return " ".join(segment.text.strip() for segment in self.segments).strip()

