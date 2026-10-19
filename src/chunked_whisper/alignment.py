"""Optional WhisperX forced alignment of existing transcript text."""

from __future__ import annotations

from dataclasses import replace
from logging import getLogger
from math import isfinite
from pathlib import Path
import re

from .models import SegmentResult, WordResult

LOGGER = getLogger(__name__)


class WordAligner:
    """Cache language-specific alignment models across input files.

    Models run on CPU by default to avoid competing with transcription for VRAM.
    Imports and model downloads occur only when alignment is requested.
    """

    def __init__(self, device: str = "cpu", model_name: str | None = None):
        self.device = device
        self.model_name = model_name
        self._models: dict = {}

    def align(
        self, segments: list[SegmentResult], audio_path: Path, language: str
    ) -> list[SegmentResult]:
        """Refine timings while preserving text and falling back on failed segments.

        An incomplete alignment retains the original segment and emits a warning.
        Alignment scores are not used as transcription confidence probabilities.
        """
        if not segments:
            return []
        try:
            from whisperx import align, load_align_model
            from faster_whisper.audio import decode_audio
        except ImportError as error:
            raise RuntimeError(
                "word alignment requires the 'alignment' extra; install it with: "
                "python -m pip install -e '.[alignment]'"
            ) from error

        if language not in self._models:
            LOGGER.info("Loading word alignment model for %s on %s", language, self.device)
            self._models[language] = load_align_model(
                language_code=language, device=self.device, model_name=self.model_name
            )
        model, metadata = self._models[language]
        audio = decode_audio(str(audio_path), sampling_rate=16_000)
        duration = len(audio) / 16_000
        refined = []
        for segment in segments:
            if not segment.text.strip():
                refined.append(segment)
                continue
            output = align(
                [{"start": segment.start, "end": segment.end, "text": segment.text}],
                model, metadata, audio, self.device, return_char_alignments=False,
            )
            aligned_words = output.get("word_segments", [])
            # Keep exact whitespace and punctuation: speaker grouping concatenates words.
            tokens = list(re.finditer(r"\s*\S+", segment.text))
            valid = len(tokens) == len(aligned_words)
            if valid:
                for token, word in zip(tokens, aligned_words):
                    start, end = word.get("start"), word.get("end")
                    if (
                        token.group().strip() != word.get("word", "").strip()
                        or start is None or end is None
                        or not isfinite(start) or not isfinite(end)
                        or not 0 <= start < end <= duration
                    ):
                        valid = False
                        break
            if not valid:
                LOGGER.warning(
                    "Incomplete word alignment at %.2f-%.2fs; keeping original timings",
                    segment.start, segment.end,
                )
                refined.append(segment)
                continue
            words = []
            for index, (token, word) in enumerate(zip(tokens, aligned_words)):
                probability = 0.0
                if (
                    len(segment.words) == len(tokens)
                    and segment.words[index].word.strip() == token.group().strip()
                ):
                    probability = segment.words[index].probability
                text = token.group()
                if index == len(tokens) - 1:
                    text += segment.text[token.end():]
                words.append(WordResult(float(word["start"]), float(word["end"]), text, probability))
            refined.append(replace(
                segment, start=min(w.start for w in words),
                end=max(w.end for w in words), words=words,
            ))
        return refined
