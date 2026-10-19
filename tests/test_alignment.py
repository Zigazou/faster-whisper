import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from chunked_whisper.alignment import WordAligner
from chunked_whisper.diarization import SpeakerTurn, assign_speakers
from chunked_whisper.models import SegmentResult, WordResult


class AlignmentTests(unittest.TestCase):
    def run_alignment(self, words, segments, aligner=None):
        backend = SimpleNamespace(
            load_align_model=Mock(return_value=(object(), {})),
            align=Mock(return_value={"word_segments": words}),
        )
        with (
            patch.dict(sys.modules, {"whisperx": backend}),
            patch("faster_whisper.audio.decode_audio", return_value=np.zeros(24 * 16000)),
        ):
            result = (aligner or WordAligner()).align(segments, Path("audio.wav"), "fr")
        return result, backend

    def test_refined_boundary_assigns_article_to_next_speaker(self):
        segment = SegmentResult(16.98, 22, "Le diable, Satan.", words=[
            WordResult(16.98, 17.54, "Le", 0.9),
            WordResult(17.54, 18, " diable,", 0.8),
            WordResult(18, 22, " Satan.", 0.7),
        ])
        result, _ = self.run_alignment([
            {"word": "Le", "start": 17.5, "end": 17.6},
            {"word": "diable,", "start": 17.6, "end": 18},
            {"word": "Satan.", "start": 18, "end": 22},
        ], [segment])
        speakers = assign_speakers(result, [
            SpeakerTurn(12.74, 17.5, "SPEAKER_00"),
            SpeakerTurn(17.5, 22, "SPEAKER_01"),
        ])
        self.assertEqual(len(speakers), 1)
        self.assertEqual(speakers[0].speaker, "SPEAKER_01")
        self.assertEqual(speakers[0].text, segment.text)
        self.assertEqual(result[0].words[0].probability, 0.9)
        self.assertEqual(segment.words[0].start, 16.98)

    def test_incomplete_alignment_keeps_original_text_and_timings(self):
        segment = SegmentResult(1, 3, "En 2014.")
        with self.assertLogs("chunked_whisper.alignment", level="WARNING"):
            result, _ = self.run_alignment([
                {"word": "En", "start": 1, "end": 1.2}, {"word": "2014."},
            ], [segment])
        self.assertIs(result[0], segment)

    def test_model_cached_per_language(self):
        aligner = WordAligner()
        segment = SegmentResult(1, 2, "Bonjour")
        words = [{"word": "Bonjour", "start": 1.1, "end": 1.9}]
        self.run_alignment(words, [segment], aligner)
        _, backend = self.run_alignment(words, [segment], aligner)
        backend.load_align_model.assert_not_called()

    def test_empty_transcript_needs_no_backend(self):
        self.assertEqual(WordAligner().align([], Path("missing.wav"), "fr"), [])
