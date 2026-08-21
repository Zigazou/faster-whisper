import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from chunked_whisper.diarization import (
    SpeakerTurn,
    assign_speakers,
    diarize_audio,
)
from chunked_whisper.models import SegmentResult, WordResult


class FakeAnnotation:
    def itertracks(self, yield_label=False):
        self.yield_label = yield_label
        yield SimpleNamespace(start=2.0, end=3.0), "track", "guest"
        yield SimpleNamespace(start=0.0, end=2.0), "track", "host"
        yield SimpleNamespace(start=3.0, end=4.0), "track", "host"


class DiarizeAudioTests(unittest.TestCase):
    @patch("chunked_whisper.diarization.decode_audio")
    def test_uses_exclusive_timeline_and_anonymizes_labels(self, decode_audio):
        decode_audio.return_value = np.zeros(32_000, dtype=np.float32)
        annotation = FakeAnnotation()
        pipeline = Mock(
            return_value=SimpleNamespace(exclusive_speaker_diarization=annotation)
        )

        turns = diarize_audio(pipeline, Path("interview.wav"), num_speakers=2)

        decode_audio.assert_called_once_with("interview.wav", sampling_rate=16_000)
        pipeline.assert_called_once()
        pipeline_input = pipeline.call_args.args[0]
        self.assertEqual(pipeline_input["sample_rate"], 16_000)
        self.assertEqual(pipeline_input["uri"], "interview")
        self.assertEqual(tuple(pipeline_input["waveform"].shape), (1, 32_000))
        self.assertEqual(pipeline.call_args.kwargs, {"num_speakers": 2})
        self.assertTrue(annotation.yield_label)
        self.assertEqual(
            turns,
            [
                SpeakerTurn(0.0, 2.0, "SPEAKER_00"),
                SpeakerTurn(2.0, 3.0, "SPEAKER_01"),
                SpeakerTurn(3.0, 4.0, "SPEAKER_00"),
            ],
        )

    @patch("chunked_whisper.diarization.decode_audio")
    def test_rejects_empty_audio(self, decode_audio):
        decode_audio.return_value = np.array([], dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "audio stream is empty"):
            diarize_audio(Mock(), Path("empty.mp3"))

    def test_splits_word_timed_segment_at_speaker_change(self):
        words = [
            WordResult(0.0, 0.8, " Hello", 0.9),
            WordResult(0.8, 1.4, " there.", 0.9),
            WordResult(1.4, 2.0, " Hi!", 0.9),
        ]
        segments = [SegmentResult(0.0, 2.0, "Hello there. Hi!", words)]
        turns = [
            SpeakerTurn(0.0, 1.3, "SPEAKER_00"),
            SpeakerTurn(1.3, 2.0, "SPEAKER_01"),
        ]

        aligned = assign_speakers(segments, turns)

        self.assertEqual(
            [(segment.speaker, segment.text) for segment in aligned],
            [
                ("SPEAKER_00", "Hello there."),
                ("SPEAKER_01", "Hi!"),
            ],
        )

    def test_assigns_unworded_segment_by_greatest_overlap(self):
        segment = SegmentResult(0.5, 2.5, "Hello")
        turns = [
            SpeakerTurn(0.0, 1.0, "SPEAKER_00"),
            SpeakerTurn(1.0, 3.0, "SPEAKER_01"),
        ]

        aligned = assign_speakers([segment], turns)

        self.assertEqual(aligned[0].speaker, "SPEAKER_01")


if __name__ == "__main__":
    unittest.main()
